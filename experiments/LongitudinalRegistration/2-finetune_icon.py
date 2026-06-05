# %% [markdown]
# # Fine-tune uniGradICON on Duke 4D Gated CT Data
#
# Discovers per-patient gated CT images and their precomputed
# SegmentHeartSimpleware labelmaps and applies the project-wide fixed 80/20
# train/test split (sort patients in ``ref_data_dir`` by filename; the first
# 80% are train, the last 20% are test).  The train cohort is handed to
# :class:`WorkflowFineTuneICONRegistration`, which builds the paired dataset
# JSON, YAML config, and derived loss-function masks, then launches
# ``unigradicon.finetuning.finetune`` as a subprocess.
#
# ``2-recon_4d_icon_eval.py`` re-derives the same split from the same sorted
# patient list — no cached split file is needed.
#
# Each patient directory under ``src_data_dir_base`` is one ``subject_id``;
# all of that patient's gated time-point frames form a paired training group.
# Frames whose labelmap is missing on disk are dropped from the dataset.
#
# In addition to the original ``gated_nii`` frames, each patient's training
# group is augmented with that patient's ANTS- and Greedy-warped frames
# written by ``1-initial_registration.py`` (warped image + labelmap per gated
# frame, under ``output_dir / <method> / <patient_id>``).  Because the warped
# frames are merged into the *same* ``subject_id`` group, uniGradICON pairs the
# original gated frames and both backends' pre-registered frames together.

# %%
import os
from pathlib import Path
from typing import Optional

import itk

from physiomotion4d import WorkflowFineTuneICONRegistration
from physiomotion4d.labelmap_tools import LabelmapTools

# %% [markdown]
# ## 1. Configure data, output locations, and the train/test split

# %%
ref_data_dir = Path("d:/PhysioMotion4D/duke_data/ref_images")
src_data_dir_base = Path("d:/PhysioMotion4D/duke_data/gated_nii")
labelmap_dir_base = Path("d:/PhysioMotion4D/duke_data/simple_ascardio")

# Where the workflow writes the dataset JSON, YAML config, derived masks, and
# the uniGradICON ``checkpoints/`` tree.  experiment_dir resolves to
# ``output_dir / fine_tune_name``.
_HERE = Path(__file__).parent
output_dir = _HERE / "results_finetuning"
fine_tune_name = "icon_finetuning"

# Pre-registration augmentation: ``1-initial_registration.py`` warps every gated
# moving frame into reference space with these backends and writes the warped
# image + labelmap under ``initial_registration_dir / <method>.lower() /
# <patient_id>``.  Those warped frames are merged into each patient's training
# group below (section 4b).
initial_registration_dir = output_dir
initial_registration_methods = ["Greedy"]

# Fixed train/test split: sort patients in ``ref_data_dir`` by filename;
# first 80% are train, last 20% are test.  ``2-recon_4d_icon_eval.py`` applies
# the same rule so the two scripts agree without a cached split record.
train_fraction = 0.8

# Local clone of uniGradICON (feat-add-finetuning branch) — prepended to
# PYTHONPATH so the subprocess picks up the local source instead of the
# installed package.  Set to ``None`` to use the pip-installed unigradicon.
unigradicon_src_path: Optional[Path] = Path(__file__).parent / "uniGradICON" / "src"

# %% [markdown]
# ## 2. Enumerate patients and apply the fixed 80/20 split
#
# Sort ``ref_data_dir`` by filename to produce the canonical patient order.
# The first 80% become the train cohort; the last 20% are the held-out test
# cohort that ``2-recon_4d_icon_eval.py`` will evaluate.

# %%
ref_files = sorted(
    p
    for p in ref_data_dir.iterdir()
    if p.name.startswith("pm00") and p.suffixes[-2:] == [".nii", ".gz"]
)
all_patient_ids = [p.name[:6] for p in ref_files]
print(f"Found {len(all_patient_ids)} patients under {ref_data_dir}")

if len(all_patient_ids) < 2:
    raise FileNotFoundError(
        f"Need at least 2 patients to form a train/test split; "
        f"discovered {len(all_patient_ids)} under {ref_data_dir}"
    )

n_train = max(
    1,
    min(len(all_patient_ids) - 1, round(train_fraction * len(all_patient_ids))),
)
train_subjects = all_patient_ids[:n_train]
test_subjects = all_patient_ids[n_train:]
print(f"  Train (first {n_train}): {train_subjects}")
print(f"  Test  (last {len(test_subjects)}): {test_subjects}")

# %% [markdown]
# ## 3. Gather the train cohort's gated frames and labelmaps and masks
#
# For each train-cohort patient, list gated frames in
# ``src_data_dir_base / <patient_id>`` (excluding ``"nop"`` non-gated
# references) and pair each frame with its
# ``<stem>_labelmap.nii.gz`` and ``<stem>_mask.nii.gz`` under ``labelmap_dir_base / <patient_id>``.
# Patients with no source directory or no valid frames are skipped here only
# — they remain part of the canonical train list above, but contribute no
# training data.  Missing labelmaps are recorded as ``None`` so the workflow
# skips just that frame.

# %%
train_image_files: list[list[Path]] = []
train_labelmap_files: list[list[Optional[Path]]] = []
train_mask_files: list[list[Optional[Path]]] = []
valid_train_subjects: list[str] = []

mask_dilation_mm = 3.0
labelmap_tools = LabelmapTools()


# %%
def load_or_derive_mask(labelmap_path: Path) -> Optional[Path]:
    """Create (or reuse) a loss-function mask next to ``labelmap_path``.

    Thresholds the labelmap at ``>0`` and dilates by ``mask_dilation_mm`` mm
    via :meth:`LabelmapTools.convert_labelmap_to_mask`, writing the result as
    ``<labelmap_stem>_mask.nii.gz`` in the labelmap's own directory.  Handles
    both ``.nii.gz`` (original Simpleware labelmaps) and ``.mha``
    (pre-registration warped labelmaps).  Returns the mask path; existing
    masks on disk are reused unmodified.
    """
    if not labelmap_path.exists():
        return None

    name = labelmap_path.name
    if name.endswith(".nii.gz"):
        stem = name[:-7]
    elif name.endswith(".mha"):
        stem = name[:-4]
    else:
        stem = labelmap_path.stem
    mask_p = labelmap_path.parent / f"{stem}_mask.nii.gz"
    if not mask_p.exists():
        mask = labelmap_tools.convert_labelmap_to_mask(
            itk.imread(str(labelmap_path)), dilation_in_mm=mask_dilation_mm
        )
        itk.imwrite(mask, str(mask_p), compression=True)
    return mask_p


# %%
def gather_warped_frames(
    method_dir: Path,
) -> tuple[list[Path], list[Optional[Path]], list[Optional[Path]]]:
    """Return ``(warped_image_paths, warped_labelmap_paths, warped_mask_paths)`` for one
    ``initial_registration_dir / <method> / <patient_id>`` directory.

    Enumerates the warped moving images (``<stem>.mha``), excluding the
    ``_labelmap.mha`` and ``_mask.mha``
    companions, and pairs each with its ``<stem>_labelmap.mha`` (``None`` when
    that labelmap is absent).  Returns empty lists when ``method_dir`` does
    not exist.
    """
    if not method_dir.is_dir():
        return [], [], []
    companion_suffixes = (
        "_labelmap.mha",
        "_mask.mha",
    )
    image_paths: list[Path] = []
    labelmap_paths: list[Optional[Path]] = []
    mask_paths: list[Optional[Path]] = []
    for image in sorted(method_dir.glob("*.mha")):
        if image.name.endswith(companion_suffixes):
            continue
        stem = image.name[:-4]
        labelmap = method_dir / f"{stem}_labelmap.mha"
        mask = method_dir / f"{stem}_mask.mha"
        image_paths.append(image)
        labelmap_paths.append(labelmap if labelmap.exists() else None)
        mask_paths.append(mask if mask.exists() else None)
    return image_paths, labelmap_paths, mask_paths


# %%
for patient_id in train_subjects:
    src_dir = src_data_dir_base / patient_id
    seg_dir = labelmap_dir_base / patient_id

    if not src_dir.is_dir():
        print(f"  Skipping {patient_id}: source dir {src_dir} not found")
        continue

    frame_names = sorted(
        f for f in os.listdir(src_dir) if "nop" not in f and f.endswith(".nii.gz")
    )
    if not frame_names:
        print(f"  Skipping {patient_id}: no valid frames in {src_dir}")
        continue

    image_paths = [src_dir / f for f in frame_names]
    labelmap_paths: list[Optional[Path]] = []
    mask_paths: list[Optional[Path]] = []
    for f in frame_names:
        labelmap = seg_dir / f.replace(".nii.gz", "_labelmap.nii.gz")
        labelmap_paths.append(labelmap if labelmap.exists() else None)
        mask = load_or_derive_mask(labelmap)
        mask_paths.append(mask)

    train_image_files.append(image_paths)
    train_labelmap_files.append(labelmap_paths)
    train_mask_files.append(mask_paths)
    valid_train_subjects.append(patient_id)

    n_seg = sum(1 for s in labelmap_paths if s is not None)
    print(f"  {patient_id}: {len(image_paths)} frames, {n_seg} with labelmap")


for subject_index, patient_id in enumerate(valid_train_subjects):
    for method_name in initial_registration_methods:
        method_dir = initial_registration_dir / method_name.lower() / patient_id
        warped_images, warped_labelmaps, warped_masks = gather_warped_frames(method_dir)
        if not warped_images:
            print(
                f"  {patient_id}/{method_name}: no initial-registered frames "
                f"in {method_dir}"
            )
            continue
        train_image_files[subject_index].extend(warped_images)
        train_labelmap_files[subject_index].extend(warped_labelmaps)
        train_mask_files[subject_index].extend(warped_masks)
        n_warped = sum(1 for labelmap in warped_labelmaps if labelmap is not None)
        print(
            f"  {patient_id}/{method_name}: +{len(warped_images)} warped frames, "
            f"{n_warped} with labelmap"
        )

# %%
workflow = WorkflowFineTuneICONRegistration(
    subject_image_files=[
        [str(image_path) for image_path in image_paths]
        for image_paths in train_image_files
    ],
    output_dir=output_dir,
    fine_tune_name=fine_tune_name,
    subject_ids=valid_train_subjects,
    subject_labelmap_files=[
        [
            str(labelmap_path) if labelmap_path is not None else None
            for labelmap_path in labelmap_paths
        ]
        for labelmap_paths in train_labelmap_files
    ],
    subject_mask_files=[
        [str(mask_path) if mask_path is not None else None for mask_path in mask_paths]
        for mask_paths in train_mask_files
    ],
    mask_dilation_mm=mask_dilation_mm,
    unigradicon_src_path=unigradicon_src_path,
    epochs=500,
)

weights_path = workflow.run_fine_tuning()
print(f"\nFine-tuning complete. Expected weights at: {weights_path}")
print(f"Held-out test cohort (for 2-recon_4d_icon_eval.py): {test_subjects}")
