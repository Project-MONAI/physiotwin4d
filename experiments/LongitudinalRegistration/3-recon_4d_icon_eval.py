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

from physiomotion4d import RegisterTimeSeriesImages
from physiomotion4d.labelmap_tools import LabelmapTools
from physiomotion4d.landmark_tools import LandmarkTools

# %% [markdown]
# ## 1. Hard-coded paths and configuration

# %%
ref_data_dir = Path("d:/PhysioMotion4D/duke_data/ref_images")
timepoint_base_dir = Path("d:/PhysioMotion4D/duke_data/gated_nii")
segmentation_base_dir = Path("d:/PhysioMotion4D/duke_data/simple_ascardio")

_HERE = Path(__file__).parent
output_dir = _HERE / "results"
finetuned_weights_path = (
    output_dir
    / "icon_finetuned"
    / "icon_finetuned_model"
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
if detail_file.exists():
    detail_file.unlink()

# %% [markdown]
# ## 2. Derive the held-out test cohort
#
# The fixed split is: sort ``ref_data_dir`` by filename, take the *first*
# 80% of patients as train, the *last* 20% as test.  ``1-finetune_icon.py``
# applies the same rule so the two scripts agree without any cached record.

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
            if index == reference_index:
                continue
            timepoint = timepoints[index]
            timepoint_dir = method_dir / timepoint
            timepoint_dir.mkdir(parents=True, exist_ok=True)

            inverse_transform = result["inverse_transforms"][index]
            itk.transformwrite(
                result["forward_transforms"][index],
                str(timepoint_dir / "time_to_reference.hdf"),
                compression=True,
            )
            itk.transformwrite(
                inverse_transform,
                str(timepoint_dir / "reference_to_time.hdf"),
                compression=True,
            )

            # inverse_transform follows the ITK resampler convention — it maps
            # moving-grid points back to reference-grid points, which is what
            # we need to warp time-point landmarks into reference space.
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
