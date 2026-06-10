# %% [markdown]
# # Evaluate ICON default vs finetuned weights on held-out longitudinal CT
#
# Enumerates the Duke patient cohort by sorting ``ref_images/`` and uses the
# *last 20%* of patients as the held-out test set — the same fixed split
# applied by ``2-finetune_icon.py`` (first 80% train, last 20% test).  For
# each test subject the 70th-percentile gated frame is selected as the
# reference and every other frame is registered to it twice with
# ``RegisterTimeSeriesImages``: once with the default uniGradICON weights and
# once with the finetuned checkpoint from ``2-finetune_icon.py``.  The
# resampler-convention inverse transform (which maps moving-grid points back
# to reference-grid points) is applied to each time-point's precomputed
# landmarks to land them in reference space, and the Euclidean error against
# the reference landmarks is recorded.
#
# Run interactively cell-by-cell; all paths are hard-coded.

# %%
import csv
import re
from pathlib import Path
from typing import Optional

import itk
import numpy as np

from physiomotion4d import RegisterTimeSeriesImages, SegmentHeartSimpleware
from physiomotion4d.labelmap_tools import LabelmapTools
from physiomotion4d.landmark_tools import LandmarkTools
from physiomotion4d.transform_tools import TransformTools

# %% [markdown]
# ## 1. Hard-coded paths and configuration

# %%
ref_data_dir = Path("d:/PhysioMotion4D/duke_data/ref_images")
timepoint_base_dir = Path("d:/PhysioMotion4D/duke_data/gated_nii")
segmentation_base_dir = Path("d:/PhysioMotion4D/duke_data/simple_ascardio")

_HERE = Path(__file__).parent
output_dir = _HERE / "results_icon_eval"
finetuned_weights_path = (
    _HERE
    / "results_finetuning"
    / "icon_finetuning"
    / "icon_finetuning_model-2"
    / "checkpoints"
    / "network_weights_final.trch"
)

train_fraction = 0.8
icon_iterations = None
reference_percentile = 0.70
exclude_tokens = ["nop"]
timepoint_re = re.compile(r"_g(?P<timepoint>[0-9]{3})")

methods: list[tuple[str, Optional[Path]]] = [
    ("icon_default", None),
    ("icon_finetuned", finetuned_weights_path),
]

output_dir.mkdir(parents=True, exist_ok=True)
detail_file = output_dir / "landmark_errors_by_point.csv"
summary_file = output_dir / "registration_summary.csv"
warped_ref_detail_file = output_dir / "warped_ref_landmark_errors_by_point.csv"
if detail_file.exists():
    detail_file.unlink()
if warped_ref_detail_file.exists():
    warped_ref_detail_file.unlink()

# %%
ref_files = sorted(
    p
    for p in ref_data_dir.iterdir()
    if p.name.startswith("pm00") and p.suffixes[-2:] == [".nii", ".gz"]
)
all_patient_ids = [p.name[:6] for p in ref_files]
n_train = max(
    1, min(len(all_patient_ids) - 1, round(train_fraction * len(all_patient_ids)))
)
test_subjects = all_patient_ids[n_train:]
print(
    f"Cohort: {len(all_patient_ids)} patients; "
    f"first {n_train} train, last {len(test_subjects)} test."
)
print(f"Held-out test subjects: {test_subjects}")

# %% [markdown]
# ## 3. Reader instance used in the per-frame inner loop
#
# Landmarks are read with :meth:`LandmarkTools.read_landmarks_3dslicer` —
# they were written as ``<stem>_landmark.mrk.json`` (3D Slicer Markups JSON,
# LPS) by ``0-cardiacGatedCT_segment_and_landmark.py``.  Binary registration
# masks come from :meth:`LabelmapTools.convert_labelmap_to_mask` (``>0``
# threshold plus 5 mm dilation), matching the loss-function masks used
# during fine-tuning in ``1-finetune_icon.py``.

# %%
landmark_tools = LandmarkTools()
labelmap_tools = LabelmapTools()
transform_tools = TransformTools()
segmenter = SegmentHeartSimpleware()
segmenter.set_trim_branches(False)


# %% [markdown]
# ## 4. Register and score every test subject under both ICON methods

# %%
summary_rows: list[dict[str, object]] = []

for subject_id in test_subjects:
    source_dir = timepoint_base_dir / subject_id
    print(f"Source directory: {source_dir}")

    seg_dir = segmentation_base_dir / subject_id
    print(f"Segmentation directory: {seg_dir}")

    image_files = [
        p
        for p in sorted(source_dir.glob("*.nii.gz"))
        if not any(t in p.name for t in exclude_tokens)
    ]
    print(f"Found {len(image_files)} image files")
    stems = [p.name[:-7] for p in image_files]
    labelmap_files = [seg_dir / f"{s}_labelmap.nii.gz" for s in stems]
    mask_files = [seg_dir / f"{s}_labelmap_mask.nii.gz" for s in stems]
    landmark_files = [seg_dir / f"{s}_landmark.mrk.json" for s in stems]
    timepoints = [timepoint_re.search(p.name).group("timepoint") for p in image_files]

    reference_index = int(round(reference_percentile * (len(image_files) - 1)))
    print(
        f"\nSubject {subject_id}: {len(image_files)} time points, "
        f"reference index {reference_index} (g{timepoints[reference_index]})"
    )

    fixed_image = itk.imread(str(image_files[reference_index]), pixel_type=itk.F)
    fixed_labelmap = itk.imread(str(labelmap_files[reference_index]))
    if mask_files[reference_index].exists():
        fixed_mask = itk.imread(str(mask_files[reference_index]))
    else:
        fixed_mask = labelmap_tools.convert_labelmap_to_mask(
            fixed_labelmap, dilation_in_mm=5.0
        )
        itk.imwrite(fixed_mask, str(mask_files[reference_index]), compression=True)
    reference_landmarks = landmark_tools.read_landmarks_3dslicer(
        landmark_files[reference_index]
    )

    moving_images = [itk.imread(str(p), pixel_type=itk.F) for p in image_files]
    moving_labelmaps = [itk.imread(str(p)) for p in labelmap_files]
    moving_landmarks = [
        landmark_tools.read_landmarks_3dslicer(str(p)) for p in landmark_files
    ]
    moving_masks = []
    for index, p in enumerate(mask_files):
        if not p.exists():
            mask = labelmap_tools.convert_labelmap_to_mask(
                moving_labelmaps[index], dilation_in_mm=5.0
            )
            itk.imwrite(mask, str(p), compression=True)
            moving_masks.append(mask)
        else:
            mask = itk.imread(str(p))
            moving_masks.append(mask)

    for method_name, weights_path in methods:
        print(f"  Method: {method_name}")
        registrar = RegisterTimeSeriesImages(registration_method="ICON")
        registrar.set_modality("ct")
        registrar.set_fixed_image(fixed_image)
        registrar.set_fixed_mask(fixed_mask)
        registrar.set_number_of_iterations_ICON(icon_iterations)
        if weights_path is not None:
            registrar.registrar_ICON.set_weights_path(str(weights_path))

        result = registrar.register_time_series(
            moving_images=moving_images,
            moving_masks=moving_masks,
            moving_labelmaps=moving_labelmaps,
            reference_frame=reference_index,
            register_reference=False,
            prior_weight=0.0,
        )

        method_dir = output_dir / method_name / subject_id
        method_dir.mkdir(parents=True, exist_ok=True)

        for index, image_file in enumerate(image_files):
            timepoint = timepoints[index]

            inverse_transform = result["inverse_transforms"][index]

            itk.transformwrite(
                result["forward_transforms"][index],
                str(method_dir / f"{subject_id}_g{timepoint}_forward_tfm.hdf"),
                compression=True,
            )
            itk.transformwrite(
                inverse_transform,
                str(method_dir / f"{subject_id}_g{timepoint}_inverse_tfm.hdf"),
                compression=True,
            )

            timepoint_landmarks = moving_landmarks[index]
            shared = sorted(timepoint_landmarks.keys() & reference_landmarks.keys())
            errors: list[tuple[str, float]] = []
            for name in shared:
                warped = inverse_transform.TransformPoint(timepoint_landmarks[name])
                err = float(
                    np.linalg.norm(
                        np.asarray(warped, dtype=np.float64)
                        - np.asarray(reference_landmarks[name], dtype=np.float64)
                    )
                )
                errors.append((name, err))

            with detail_file.open("a", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                if fh.tell() == 0:
                    writer.writerow(
                        ["subject_id", "method", "timepoint", "name", "error_mm"]
                    )
                for name, err in errors:
                    writer.writerow([subject_id, method_name, timepoint, name, err])

            values = np.asarray([e for _, e in errors], dtype=np.float64)
            summary_rows.append(
                {
                    "subject_id": subject_id,
                    "method": method_name,
                    "reference_timepoint": timepoints[reference_index],
                    "timepoint": timepoint,
                    "loss": float(result["losses"][index]),
                    "n_landmarks": int(values.size),
                    "mean_mm": float(np.mean(values)) if values.size else "",
                    "median_mm": float(np.median(values)) if values.size else "",
                    "max_mm": float(np.max(values)) if values.size else "",
                }
            )

            # ------------------------------------------------------------------
            # Warp the reference image back onto each time-point's grid, re-
            # segment with SegmentHeartSimpleware, and compare the resulting
            # landmarks with the time-point's own precomputed landmarks.
            #
            # Per transform_conventions.rst:
            #   - warp fixed image -> moving grid  => inverse_transform +
            #     TransformTools.transform_image  (pull-back)
            #   - warp moving points -> fixed space => inverse_transform +
            #     .TransformPoint()  (push-forward)
            # Both use inverse_transform, but for opposite purposes.
            # Here we use inverse_transform for image warping (row 3 of the
            # table), placing the reference image in time-point space so
            # Simpleware sees anatomy at the correct cardiac phase.
            #
            # Skip the reference frame — warping it to itself is trivial and
            # its own landmarks are already the "ground truth" reference.
            # ------------------------------------------------------------------
            if index == reference_index:
                continue

            warped_ref = transform_tools.transform_image(
                fixed_image,
                inverse_transform,
                moving_images[index],
                interpolation_method="linear",
            )
            itk.imwrite(
                warped_ref,
                method_dir / f"{subject_id}_g{timepoint}_warped_ref.mha",
                compression=True,
            )

            seg_result = segmenter.segment(warped_ref, contrast_enhanced_study=False)
            warped_ref_labelmap = seg_result["labelmap"]
            warped_ref_landmarks = segmenter.get_landmarks()

            itk.imwrite(
                warped_ref_labelmap,
                str(method_dir / f"{subject_id}_g{timepoint}_warped_ref_labelmap.mha"),
                compression=True,
            )
            landmark_tools.write_landmarks_3dslicer(
                warped_ref_landmarks,
                str(
                    method_dir
                    / f"{subject_id}_g{timepoint}_warped_ref_landmarks.mrk.json"
                ),
            )

            # Both warped_ref_landmarks and timepoint_landmarks are in the
            # time-point (moving) image space — compare directly.
            tp_landmarks = moving_landmarks[index]
            shared_warp = sorted(warped_ref_landmarks.keys() & tp_landmarks.keys())
            warp_errors: list[tuple[str, float]] = []
            for name in shared_warp:
                err = float(
                    np.linalg.norm(
                        np.asarray(warped_ref_landmarks[name], dtype=np.float64)
                        - np.asarray(tp_landmarks[name], dtype=np.float64)
                    )
                )
                warp_errors.append((name, err))
                print(f"    Warped-ref landmark {name}: {err:.3f} mm")

            with warped_ref_detail_file.open("a", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                if fh.tell() == 0:
                    writer.writerow(
                        [
                            "subject_id",
                            "method",
                            "timepoint",
                            "name",
                            "error_mm",
                        ]
                    )
                for name, err in warp_errors:
                    writer.writerow([subject_id, method_name, timepoint, name, err])

            warp_vals = np.asarray([e for _, e in warp_errors], dtype=np.float64)
            if warp_vals.size:
                print(
                    f"  Warped-ref landmark errors ({timepoint}): "
                    f"mean={float(np.mean(warp_vals)):.3f} mm  "
                    f"median={float(np.median(warp_vals)):.3f} mm  "
                    f"max={float(np.max(warp_vals)):.3f} mm"
                )

# %% [markdown]
# ## 5. Write the wide-form per-timepoint summary CSV

# %%
if not summary_rows:
    print("No summary rows to write")
else:
    with summary_file.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote summary: {summary_file}")
print(f"Wrote landmark details: {detail_file}")

# %% [markdown]
# ## 6. Per-method aggregate table over all test subjects

# %%
groups: dict[str, list[float]] = {}
with detail_file.open(newline="", encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
        groups.setdefault(row["method"], []).append(float(row["error_mm"]))

header = (
    f"{'Method':<18}{'N':>8}{'Mean (mm)':>12}"
    f"{'Median (mm)':>14}{'P95 (mm)':>12}{'Max (mm)':>12}"
)
print()
print("=" * len(header))
print(f"Landmark error summary ({len(test_subjects)} test subjects)")
print("=" * len(header))
print(header)
print("-" * len(header))
for method_name, _ in methods:
    arr = np.asarray(groups.get(method_name, []), dtype=np.float64)
    if arr.size == 0:
        print(f"{method_name:<18}{0:>8}{'':>12}{'':>14}{'':>12}{'':>12}")
        continue
    print(
        f"{method_name:<18}"
        f"{arr.size:>8}"
        f"{float(np.mean(arr)):>12.3f}"
        f"{float(np.median(arr)):>14.3f}"
        f"{float(np.percentile(arr, 95)):>12.3f}"
        f"{float(np.max(arr)):>12.3f}"
    )
print("=" * len(header))

# %% [markdown]
# ## 7. Per-method aggregate table: warped-reference landmark errors
#
# Compares landmarks extracted from the reference image warped back to each
# time-point's grid (via ``inverse_transform``) against that time-point's own
# precomputed landmarks.  Both sets are in the moving (time-point) image space,
# so errors are Euclidean distances without any additional transform.

# %%
if warped_ref_detail_file.exists():
    warp_groups: dict[str, list[float]] = {}
    with warped_ref_detail_file.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            warp_groups.setdefault(row["method"], []).append(float(row["error_mm"]))

    warp_header = (
        f"{'Method':<18}{'N':>8}{'Mean (mm)':>12}"
        f"{'Median (mm)':>14}{'P95 (mm)':>12}{'Max (mm)':>12}"
    )
    print()
    print("=" * len(warp_header))
    print(
        f"Warped-reference landmark error summary ({len(test_subjects)} test subjects)"
    )
    print("=" * len(warp_header))
    print(warp_header)
    print("-" * len(warp_header))
    for method_name, _ in methods:
        arr = np.asarray(warp_groups.get(method_name, []), dtype=np.float64)
        if arr.size == 0:
            print(f"{method_name:<18}{0:>8}{'':>12}{'':>14}{'':>12}{'':>12}")
            continue
        print(
            f"{method_name:<18}"
            f"{arr.size:>8}"
            f"{float(np.mean(arr)):>12.3f}"
            f"{float(np.median(arr)):>14.3f}"
            f"{float(np.percentile(arr, 95)):>12.3f}"
            f"{float(np.max(arr)):>12.3f}"
        )
    print("=" * len(warp_header))
else:
    print(
        "No warped-reference landmark errors written (all frames were reference frames)."
    )
