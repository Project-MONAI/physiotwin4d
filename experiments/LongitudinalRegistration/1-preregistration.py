# %% [markdown]
# # Pre-registration: compare ANTS vs Greedy on the Duke gated CT cohort
#
# Registers every gated CT time-point of every Duke patient under
# ``ref_data_dir`` (100% of the cohort -- no train/test split) to that
# patient's reference image, using two backends in turn:
#
#   * :class:`RegisterImagesANTS` (CPU, SyN deformable)
#   * :class:`RegisterImagesGreedy` (CPU, deformable)
#
# For each frame the script records wall-clock registration time, writes
# the warped/resampled moving image to disk, warps the moving labelmap
# into reference space to compute per-label Dice, and warps the moving
# landmarks into reference space to compute squared-error landmark
# statistics (mm^2) against the reference landmarks.
#
# Inputs (same data as ``1-finetune_icon.py``):
#   * ``ref_data_dir / pm*_ref.nii.gz`` -- per-patient reference CT
#   * ``src_data_dir_base / <patient_id> / *.nii.gz`` -- gated CT frames
#   * ``segmentation_dir_base / <patient_id> / <stem>_labelmap.nii.gz``
#     -- per-frame multi-label segmentations
#   * ``segmentation_dir_base / <patient_id> / <stem>_labelmap_mask.nii.gz``
#     -- pre-computed loss-function masks (re-derived on the fly if absent,
#     matching the 5 mm dilation used by ``1-finetune_icon.py``)
#   * ``segmentation_dir_base / <patient_id> / <stem>_landmark.mrk.json``
#     -- per-frame 3D Slicer Markups landmarks in LPS
#
# Outputs under ``results/``:
#   * ``ants/<patient_id>/<timepoint>/<stem>.mha`` and
#     ``greedy/<patient_id>/<timepoint>/<stem>.mha`` -- warped moving image
#     per time point, alongside the forward/inverse transforms (``.hdf``),
#     a ``<stem>_deformation_grid.mha`` visualization of the registration
#     deformation, the warped ``<stem>_labelmap.mha`` and its warped
#     loss-function mask ``<stem>_labelmap_mask.mha``,
#     and the warped ``<stem>_landmark.mrk.json``
#   * ``preregistration_landmarks.csv`` -- per-landmark squared errors
#   * ``preregistration_dice.csv`` -- per-label Dice
#   * ``preregistration_summary.csv`` -- per-(subject, method, timepoint)
#     time, mean Dice, MSE, RMSE
#
# Run interactively cell-by-cell; all paths are hard-coded.

# %%
import csv
import re
import time
from pathlib import Path
from typing import Optional

import itk
import numpy as np

from physiomotion4d.landmark_tools import LandmarkTools
from physiomotion4d.register_images_ants import RegisterImagesANTS
from physiomotion4d.register_images_greedy import RegisterImagesGreedy
from physiomotion4d.register_images_icon import RegisterImagesICON
from physiomotion4d.transform_tools import TransformTools

# %% [markdown]
# ## 1. Hard-coded paths and configuration

# %%
ref_data_dir = Path("d:/PhysioMotion4D/duke_data/ref_images")
src_data_dir_base = Path("d:/PhysioMotion4D/duke_data/gated_nii")
segmentation_dir_base = Path("d:/PhysioMotion4D/duke_data/simple_ascardio")

_HERE = Path(__file__).parent
output_dir = _HERE / "results"
output_dir.mkdir(parents=True, exist_ok=True)

# Reference frames in gated_nii are named ``<stem>_ref.nii.gz``; every
# other ``.nii.gz`` (excluding ``nop`` non-gated references) is a gated
# time point.  Timepoint tag ``g###`` is extracted from each filename.
exclude_tokens = ["nop"]
ref_suffix = "_ref"
timepoint_re = re.compile(r"_g(?P<timepoint>[0-9]{3})")

# Mask dilation matches 1-finetune_icon.py so any masks we have to
# derive here are identical to the ones written by the fine-tune script.
mask_dilation_mm = 5.0

# Iteration schedules.  Kept modest for a cohort-wide comparison; raise
# either list for higher accuracy at the cost of runtime.
number_of_iterations_ANTS = [20, 10, 5]
number_of_iterations_greedy = [40, 20, 10]

methods: list[str] = ["ANTS", "greedy"]

# Debug knob: when non-empty, only these patient IDs are processed.
# Set to ``[]`` (or ``None``) to run the full cohort.
debug_subjects: list[str] = ["pm0002"]

detail_landmarks_file = output_dir / "preregistration_landmarks.csv"
detail_dice_file = output_dir / "preregistration_dice.csv"
summary_file = output_dir / "preregistration_summary.csv"
for previous in (detail_landmarks_file, detail_dice_file, summary_file):
    if previous.exists():
        previous.unlink()

# %% [markdown]
# ## 2. Enumerate the full patient cohort
#
# Sort ``ref_data_dir`` by filename so the patient order is stable.
# Every patient is processed -- no train/test split.

# %%
ref_files = sorted(
    p
    for p in ref_data_dir.iterdir()
    if p.name.startswith("pm00") and p.suffixes[-2:] == [".nii", ".gz"]
)
all_patient_ids = [p.name[:6] for p in ref_files]
print(f"Found {len(all_patient_ids)} patients under {ref_data_dir}")
if debug_subjects:
    cohort = [pid for pid in all_patient_ids if pid in debug_subjects]
    print(
        f"DEBUG: restricting cohort to {debug_subjects} -> "
        f"{len(cohort)} matching patients"
    )
else:
    cohort = all_patient_ids
print(f"Patient cohort: {cohort}")

# %% [markdown]
# ## 3. Helpers: labelmap warping, per-label Dice, landmark squared error

# %%
landmark_tools = LandmarkTools()
transform_tools = TransformTools()


def per_label_dice(
    fixed_labelmap: itk.Image, warped_labelmap: itk.Image
) -> dict[int, float]:
    """Return ``{label_id: Dice}`` for every positive label present in
    either the fixed or the warped labelmap.

    Arrays come back from :func:`itk.array_from_image` in shape
    ``(Z, Y, X)`` (numpy reverses ITK's index order); we compare element-wise
    so the axis convention does not matter as long as both labelmaps live
    on the same reference grid (guaranteed because ``warped_labelmap`` was
    resampled with ``fixed_labelmap`` as the reference image).
    """
    fixed_array = itk.array_from_image(fixed_labelmap)
    warped_array = itk.array_from_image(warped_labelmap)
    labels = sorted(
        {int(v) for v in np.unique(fixed_array)}
        | {int(v) for v in np.unique(warped_array)}
    )
    labels = [label for label in labels if label > 0]

    dice_by_label: dict[int, float] = {}
    for label in labels:
        a = fixed_array == label
        b = warped_array == label
        denom = int(a.sum()) + int(b.sum())
        if denom == 0:
            continue
        intersection = int(np.logical_and(a, b).sum())
        dice_by_label[label] = 2.0 * intersection / denom
    return dice_by_label


def warp_landmarks(
    inverse_transform: itk.Transform,
    moving_landmarks: dict[str, tuple[float, float, float]],
) -> dict[str, tuple[float, float, float]]:
    """Warp every moving landmark into reference space.

    Point/landmark warping uses ``inverse_transform`` -- the moving-space ->
    fixed-space point map -- which is the opposite of the transform used to
    warp the moving image onto the fixed grid (images pull back; points push
    forward). Returns a ``{label: (x, y, z)}`` dict in LPS. See
    docs/developer/transform_conventions.
    """
    return {
        name: tuple(float(c) for c in inverse_transform.TransformPoint(point))
        for name, point in moving_landmarks.items()
    }


def landmark_squared_errors(
    warped_landmarks: dict[str, tuple[float, float, float]],
    reference_landmarks: dict[str, tuple[float, float, float]],
) -> list[tuple[str, float]]:
    """Return per-landmark squared Euclidean error in mm^2 between the
    reference-space ``warped_landmarks`` and the matching reference
    landmarks, in sorted-name order.
    """
    shared = sorted(warped_landmarks.keys() & reference_landmarks.keys())
    errors: list[tuple[str, float]] = []
    for name in shared:
        diff = np.asarray(warped_landmarks[name], dtype=np.float64) - np.asarray(
            reference_landmarks[name], dtype=np.float64
        )
        errors.append((name, float(np.dot(diff, diff))))
    return errors


def load_or_derive_mask(labelmap: itk.Image, mask_path: Path) -> itk.Image:
    """Return the cached ``<stem>_labelmap_mask.nii.gz`` next to the
    labelmap, or derive it via :meth:`RegisterImagesICON.create_mask`
    (threshold ``>0`` plus 5 mm physical-radius dilation) and write it
    out so subsequent runs and the ICON eval reuse the same mask.
    """
    if mask_path.exists():
        return itk.imread(str(mask_path))
    mask = RegisterImagesICON.create_mask(labelmap, dilation_mm=mask_dilation_mm)
    itk.imwrite(mask, str(mask_path), compression=True)
    return mask


# %% [markdown]
# ## 4. Drive the comparison: every patient x every method
#
# For each patient: load the reference image, labelmap, mask, and
# landmarks; load every gated frame (excluding ``nop`` and ``_ref``) with
# its labelmap, mask, and landmarks; then register each frame to the
# reference under both backends.  Each frame starts from identity so the
# ANTS-vs-Greedy comparison is independent across frames.

# %%
summary_rows: list[dict[str, object]] = []

# (subject_id, method, timepoint) for frames that produced no usable
# landmark errors -- either no landmark file or no labels shared with the
# reference.  Echoed in a highlighted block at the end of the run.
frames_missing_landmarks: list[tuple[str, str, str]] = []

for subject_index, subject_id in enumerate(cohort):
    print(f"\n=== Subject {subject_index + 1}/{len(cohort)}: {subject_id} ===")
    src_dir = src_data_dir_base / subject_id
    seg_dir = segmentation_dir_base / subject_id

    if not src_dir.is_dir():
        print(f"  Skipping {subject_id}: source dir {src_dir} not found")
        continue
    if not seg_dir.is_dir():
        print(f"  Skipping {subject_id}: segmentation dir {seg_dir} not found")
        continue

    # Locate this patient's reference frame in gated_nii (matches the
    # `_ref.nii.gz` filename under ref_data_dir).
    ref_file = next((p for p in ref_files if p.name.startswith(subject_id)), None)
    if ref_file is None:
        print(f"  Skipping {subject_id}: no reference image found")
        continue
    ref_stem = ref_file.name[:-7]
    ref_labelmap_path = seg_dir / f"{ref_stem}_labelmap.nii.gz"
    ref_mask_path = seg_dir / f"{ref_stem}_labelmap_mask.nii.gz"
    ref_landmark_path = seg_dir / f"{ref_stem}_landmark.mrk.json"
    if not ref_labelmap_path.exists() or not ref_landmark_path.exists():
        print(
            f"  Skipping {subject_id}: missing reference labelmap or "
            f"landmarks under {seg_dir}"
        )
        continue

    fixed_image = itk.imread(str(ref_file), pixel_type=itk.F)
    fixed_labelmap = itk.imread(str(ref_labelmap_path))
    fixed_mask = load_or_derive_mask(fixed_labelmap, ref_mask_path)
    reference_landmarks = landmark_tools.read_landmarks_3dslicer(ref_landmark_path)

    # Gated moving frames (exclude `nop` and the `_ref` frame itself).
    gated_files = sorted(
        p
        for p in src_dir.glob("*.nii.gz")
        if not any(token in p.name for token in exclude_tokens)
        and not p.name.endswith(f"{ref_suffix}.nii.gz")
    )
    moving_records: list[dict[str, object]] = []
    for image_path in gated_files:
        stem = image_path.name[:-7]
        labelmap_path = seg_dir / f"{stem}_labelmap.nii.gz"
        mask_path = seg_dir / f"{stem}_labelmap_mask.nii.gz"
        landmark_path = seg_dir / f"{stem}_landmark.mrk.json"
        if not labelmap_path.exists():
            print(f"    Dropping {stem}: no labelmap at {labelmap_path}")
            continue
        match = timepoint_re.search(image_path.name)
        if match is None:
            print(f"    Dropping {stem}: no g### timepoint tag in name")
            continue
        moving_records.append(
            {
                "stem": stem,
                "timepoint": match.group("timepoint"),
                "image_path": image_path,
                "labelmap_path": labelmap_path,
                "mask_path": mask_path,
                "landmark_path": landmark_path if landmark_path.exists() else None,
            }
        )
    if not moving_records:
        print(f"  Skipping {subject_id}: no usable gated frames")
        continue

    print(f"  {len(moving_records)} moving frames; reference {ref_file.name}")

    print(f"  Loading {len(moving_records)} moving images / labelmaps / masks ...")
    moving_images = []
    moving_labelmaps = []
    moving_masks = []
    moving_landmarks_list: list[Optional[dict[str, tuple[float, float, float]]]] = []
    for r_index, r in enumerate(moving_records):
        print(
            f"    [{r_index + 1}/{len(moving_records)}] g{r['timepoint']}  {r['stem']}"
        )
        moving_image = itk.imread(str(r["image_path"]), pixel_type=itk.F)
        labelmap = itk.imread(str(r["labelmap_path"]))
        moving_images.append(moving_image)
        moving_labelmaps.append(labelmap)
        moving_masks.append(load_or_derive_mask(labelmap, r["mask_path"]))
        landmark_path = r["landmark_path"]
        if landmark_path is None:
            moving_landmarks_list.append(None)
        else:
            moving_landmarks_list.append(
                landmark_tools.read_landmarks_3dslicer(landmark_path)
            )

    for method_name in methods:
        print(f"\n  --- Method: {method_name} ---")
        if method_name == "ANTS":
            reg = RegisterImagesANTS()
            reg.set_number_of_iterations(number_of_iterations_ANTS)
        else:
            reg = RegisterImagesGreedy()
            reg.set_number_of_iterations(number_of_iterations_greedy)
        reg.set_transform_type("Deformable")
        # NCC ("CC") outperforms MeanSquares for same-modality CT registration.
        reg.set_metric("CC")
        reg.set_modality("ct")
        reg.set_mask_dilation(mask_dilation_mm)
        reg.set_fixed_image(fixed_image)
        reg.set_fixed_mask(fixed_mask)

        method_dir = output_dir / method_name.lower() / subject_id
        method_dir.mkdir(parents=True, exist_ok=True)

        method_t_start = time.perf_counter()
        for index, record in enumerate(moving_records):
            timepoint = record["timepoint"]
            stem = record["stem"]
            print(
                f"    [{method_name} {index + 1}/{len(moving_records)}] "
                f"g{timepoint}  registering ...",
                flush=True,
            )

            frame_t_start = time.perf_counter()
            reg_result = reg.register(
                moving_image=moving_images[index],
                moving_mask=moving_masks[index],
            )
            frame_elapsed = time.perf_counter() - frame_t_start

            forward_transform = reg_result["forward_transform"]
            inverse_transform = reg_result["inverse_transform"]
            frame_loss = float(reg_result["loss"])
            print(f"      done in {frame_elapsed:.1f} s, loss={frame_loss:.4f}")

            itk.transformwrite(
                forward_transform,
                str(method_dir / f"{stem}_fwd.hdf"),
                compression=True,
            )
            itk.transformwrite(
                inverse_transform,
                str(method_dir / f"{stem}_inv.hdf"),
                compression=True,
            )

            # Visualize the deformation as a warped grid: a regular grid built
            # in reference space, resampled through forward_transform -- the same
            # transform used below to warp the moving image onto the fixed grid,
            # so the grid and the warped image deform consistently.
            deformation_grid = transform_tools.convert_field_to_grid_visualization(
                forward_transform, fixed_image
            )
            itk.imwrite(
                deformation_grid,
                str(method_dir / f"{stem}_deformation_grid.mha"),
                compression=True,
            )

            # Warp the moving image into reference space and save it
            # (forward_transform resamples the moving image onto the fixed grid).
            warped_image = transform_tools.transform_image(
                moving_images[index],
                forward_transform,
                fixed_image,
                interpolation_method="linear",
            )
            itk.imwrite(
                warped_image,
                str(method_dir / f"{stem}.mha"),
                compression=True,
            )

            # Warp the moving labelmap onto the fixed grid (forward_transform;
            # nearest neighbour preserves label IDs) for per-label Dice.
            warped_labelmap = transform_tools.transform_image(
                moving_labelmaps[index],
                forward_transform,
                fixed_labelmap,
                interpolation_method="nearest",
            )
            itk.imwrite(
                warped_labelmap,
                str(method_dir / f"{stem}_labelmap.mha"),
                compression=True,
            )

            # Warp the moving loss-function mask onto the fixed grid
            # (forward_transform; nearest neighbour preserves the binary ROI)
            # so downstream fine-tuning reuses it instead of re-deriving a
            # mask from the warped labelmap.
            warped_mask = transform_tools.transform_image(
                moving_masks[index],
                forward_transform,
                fixed_mask,
                interpolation_method="nearest",
            )
            itk.imwrite(
                warped_mask,
                str(method_dir / f"{stem}_labelmap_mask.mha"),
                compression=True,
            )

            dice_by_label = per_label_dice(fixed_labelmap, warped_labelmap)
            with detail_dice_file.open("a", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                if fh.tell() == 0:
                    writer.writerow(
                        ["subject_id", "method", "timepoint", "label", "dice"]
                    )
                for label, dice in dice_by_label.items():
                    writer.writerow([subject_id, method_name, timepoint, label, dice])
            mean_dice = (
                float(np.mean(list(dice_by_label.values())))
                if dice_by_label
                else float("nan")
            )

            # Warp the moving landmarks into reference space, save them next
            # to the transforms, then score squared error vs the reference.
            moving_landmarks = moving_landmarks_list[index]
            if moving_landmarks is None:
                sq_errors: list[tuple[str, float]] = []
            else:
                warped_landmarks = warp_landmarks(inverse_transform, moving_landmarks)
                landmark_tools.write_landmarks_3dslicer(
                    warped_landmarks,
                    str(method_dir / f"{stem}_landmark.mrk.json"),
                )
                sq_errors = landmark_squared_errors(
                    warped_landmarks, reference_landmarks
                )
            with detail_landmarks_file.open("a", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                if fh.tell() == 0:
                    writer.writerow(
                        [
                            "subject_id",
                            "method",
                            "timepoint",
                            "name",
                            "sq_err_mm2",
                        ]
                    )
                for name, sq_err in sq_errors:
                    writer.writerow([subject_id, method_name, timepoint, name, sq_err])

            sq_values = np.asarray([e for _, e in sq_errors], dtype=np.float64)
            if sq_values.size:
                mse_mm2 = float(np.mean(sq_values))
                rmse_mm = float(np.sqrt(mse_mm2))
            else:
                mse_mm2 = float("nan")
                rmse_mm = float("nan")
                # Highlight frames with no usable landmarks so they are not
                # silently scored as NaN in the CSV / summary table.
                reason = (
                    "no landmark file"
                    if moving_landmarks is None
                    else "no landmarks shared with reference"
                )
                frames_missing_landmarks.append((subject_id, method_name, timepoint))
                print(
                    f"      >>> WARNING: {subject_id} {method_name} "
                    f"g{timepoint} has NO landmarks ({reason})",
                    flush=True,
                )

            summary_rows.append(
                {
                    "subject_id": subject_id,
                    "method": method_name,
                    "timepoint": timepoint,
                    "time_sec": float(frame_elapsed),
                    "loss": frame_loss,
                    "n_labels": int(len(dice_by_label)),
                    "mean_dice": mean_dice,
                    "n_landmarks": int(sq_values.size),
                    "mse_mm2": mse_mm2,
                    "rmse_mm": rmse_mm,
                }
            )

        method_elapsed = time.perf_counter() - method_t_start
        print(
            f"  [{method_name}] subject {subject_id} total "
            f"{method_elapsed:.1f} s "
            f"({method_elapsed / len(moving_records):.1f} s/frame)"
        )

# %% [markdown]
# ## 5. Write the per-(subject, method, timepoint) summary CSV

# %%
if summary_rows:
    with summary_file.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nWrote summary:    {summary_file}")
    print(f"Wrote landmarks:  {detail_landmarks_file}")
    print(f"Wrote dice:       {detail_dice_file}")
else:
    print("\nNo frames processed; nothing to summarize.")

# %% [markdown]
# ## 5b. Highlight frames that produced no landmark errors

# %%
if frames_missing_landmarks:
    banner = "!" * 70
    print(f"\n{banner}")
    print(
        f"WARNING: {len(frames_missing_landmarks)} frame(s) missing ALL "
        f"landmarks (scored as NaN):"
    )
    for subject_id, method_name, timepoint in frames_missing_landmarks:
        print(f"  - {subject_id}  {method_name}  g{timepoint}")
    print(banner)
else:
    print("\nAll processed frames had at least one scored landmark.")

# %% [markdown]
# ## 6. Per-method aggregate table across the whole cohort
#
# Reports mean per-frame registration time, mean / median / p95 of the
# squared landmark errors (mm^2), the matching RMSE in mm, and the mean
# per-label Dice averaged across (subject, timepoint, label) entries.

# %%
if summary_rows:
    sq_by_method: dict[str, list[float]] = {}
    with detail_landmarks_file.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            sq_by_method.setdefault(row["method"], []).append(float(row["sq_err_mm2"]))

    dice_by_method: dict[str, list[float]] = {}
    with detail_dice_file.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            dice_by_method.setdefault(row["method"], []).append(float(row["dice"]))

    time_by_method: dict[str, list[float]] = {}
    for row in summary_rows:
        method_name = str(row["method"])
        time_by_method.setdefault(method_name, []).append(float(row["time_sec"]))

    header = (
        f"{'Method':<10}{'N_lm':>8}{'MSE(mm2)':>12}{'RMSE(mm)':>12}"
        f"{'p95(mm2)':>12}{'meanDice':>12}{'sec/frame':>12}"
    )
    print()
    print("=" * len(header))
    print(f"Pre-registration comparison ({len(all_patient_ids)} patients)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for method_name in methods:
        sq_arr = np.asarray(sq_by_method.get(method_name, []), dtype=np.float64)
        dice_arr = np.asarray(dice_by_method.get(method_name, []), dtype=np.float64)
        time_arr = np.asarray(time_by_method.get(method_name, []), dtype=np.float64)
        if sq_arr.size == 0:
            print(f"{method_name:<10}{0:>8}{'':>12}{'':>12}{'':>12}{'':>12}{'':>12}")
            continue
        mse = float(np.mean(sq_arr))
        rmse = float(np.sqrt(mse))
        p95 = float(np.percentile(sq_arr, 95))
        mean_dice_val = float(np.mean(dice_arr)) if dice_arr.size else float("nan")
        mean_time = float(np.mean(time_arr)) if time_arr.size else float("nan")
        print(
            f"{method_name:<10}"
            f"{sq_arr.size:>8}"
            f"{mse:>12.3f}"
            f"{rmse:>12.3f}"
            f"{p95:>12.3f}"
            f"{mean_dice_val:>12.3f}"
            f"{mean_time:>12.2f}"
        )
    print("=" * len(header))
