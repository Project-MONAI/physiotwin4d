"""
Tutorial 1: Heart-Gated CT to Animated USD

Purpose
-------
Convert a 4D cardiac CT scan (multiple gated time frames) into an animated USD
model suitable for visualization in NVIDIA Omniverse. The workflow segments the
heart and surrounding anatomy from a reference frame, registers all other frames
to that reference using deep learning or classical registration, and assembles
the resulting time-varying surface meshes into a single USD file with anatomical
materials applied.

Inputs
------
- A 4D NRRD sequence file (``*.seq.nrrd``) **or** a list of 3D CT volumes
  (``*.mha`` / ``*.nrrd``) representing successive cardiac phases.
  Expected location: ``data/Slicer-Heart-CT/TruncalValve_4DCT.seq.nrrd``
- Optional: a reference frame image to fix the cardiac phase used as the
  segmentation source.

Outputs
-------
- ``output_dir/cardiac_model.dynamic_painted.usd`` - animated USD with
  anatomy materials
- Screenshots (PNG) for documentation and regression testing:
  - ``reference_frame_axial.png`` - axial slice of the reference CT frame
  - ``segmentation_overlay.png`` - segmentation mask overlaid on reference
  - ``contours_3d.png`` - 3-D isometric view of the current-run contours

Strengths
---------
- Single call (``WorkflowConvertImageToUSD.process()``) runs the full pipeline.
- Supports both GPU-accelerated ICON registration and CPU-capable Greedy registration.
- Automatically detects contrast enhancement and adjusts segmentation thresholds.
- Output is Omniverse-ready with anatomical materials (USDAnatomyTools).

Weaknesses / Limitations
------------------------
- Requires a GPU for ICON registration (``registration_method=RegisterImagesICON()``);
  use ``registration_method=RegisterImagesGreedy()`` for CPU-only environments
  (about 10x slower).
- Segmentation quality depends on TotalSegmentator's training distribution;
  unusual pathologies or pediatric anatomy may degrade results.
- Large 4D datasets (>20 phases, high resolution) can require 32 GB+ RAM.

Classes Used
------------
- WorkflowConvertImageToUSD (workflow_convert_image_to_usd.py):
    Orchestrates the full pipeline: 4D NRRD -> segmentation -> registration ->
    contour extraction -> USD export.
- SegmentChestTotalSegmentator (segment_chest_total_segmentator.py):
    Deep-learning segmentation of 117 anatomical structures (used internally).
- RegisterImagesICON / RegisterImagesANTS (register_images_icon.py / _ants.py):
    Frame-to-frame image registration (used internally).
- ContourTools (contour_tools.py):
    Extracts and transforms surface meshes from segmentation masks (used internally).
- USDAnatomyTools (usd_anatomy_tools.py):
    Applies clinical material colours to USD prims (used internally).

Data Required
-------------
See data/README.md for download instructions and dataset licensing.
Dataset: Slicer-Heart-CT - https://github.com/SlicerHeart/SlicerHeart
This script expects the data to already exist at
``data/Slicer-Heart-CT/TruncalValve_4DCT.seq.nrrd``. Run the repository data
download notebook or download the file manually before running this tutorial.
"""

# %%
# Imports
from __future__ import annotations

import logging
from pathlib import Path

import itk

from physiotwin4d import (
    RegisterImagesICON,
    SegmentChestTotalSegmentatorWithContrast,
    TestTools,
    WorkflowConvertImageToUSD,
)

# %%
# Only run if this script is not imported as a module

# nnUNetv2 (used by TotalSegmentator inside WorkflowConvertImageToUSD)
# spawns a multiprocessing.Pool. On Windows the spawn start method re-imports
# this script in each child; without the __name__ == "__main__" guard around
# the top-level work, that re-import fires workflow.process() again and
# Python's spawn-cascade detector raises RuntimeError.
if __name__ != "__main__":
    exit(0)

# %%
# Data directory specification
repo_root = Path(__file__).resolve().parent.parent
tutorials_dir = Path(__file__).resolve().parent

class_name = "tutorial_01_heart_gated_ct_to_usd"

output_dir = tutorials_dir / "output" / "tutorial_01_heart"

test_mode = TestTools.running_as_test()
if test_mode:
    data_dir = repo_root / "data" / "test" / "slicer_heart_small"
    number_of_registration_iterations = 1
    frame_files = sorted(data_dir.glob("slice_???.mha"))[0:2]
else:
    data_dir = repo_root / "data" / "Slicer-Heart-CT"
    number_of_registration_iterations = 10
    frame_files = sorted(data_dir.glob("slice_???.mha"))

log_level = logging.INFO

registration_method = RegisterImagesICON(log_level=log_level)
registration_method.set_number_of_iterations(number_of_registration_iterations)

segmentation_method = SegmentChestTotalSegmentatorWithContrast(log_level=log_level)
segmentation_method.set_has_academic_license(True)


# %%
# Directory setup and data reading

output_dir.mkdir(parents=True, exist_ok=True)

input_filenames = [str(path) for path in frame_files]
if not input_filenames:
    raise FileNotFoundError(
        "Slicer-Heart-CT data not found. Checked:\n"
        + f"  - {data_dir}"
        + "\n"
        + "See data/README.md for download instructions."
    )

time_series_images = [itk.imread(str(path)) for path in input_filenames]
reference_image = time_series_images[int(0.7 * len(time_series_images))]

print("Number of time-series images:", len(time_series_images))

# %%
# Workflow initialization

workflow = WorkflowConvertImageToUSD(
    time_series_images=time_series_images,
    reference_image=reference_image,
    output_directory=str(output_dir),
    usd_project_name="cardiac_model",
    registration_method=registration_method,
    segmentation_method=segmentation_method,
    log_level=log_level,
    save_assets=True,
)

# %%
# Workflow execution
workflow_results = workflow.process()

# if dynamic_labelmap_ids is not None, there are two USD files
if len(workflow.dynamic_labelmap_ids) > 0:
    usd_file = output_dir / workflow_results["dynamic"]
else:
    usd_file = output_dir / workflow_results["all"]

# %%
# Result saving
tt = TestTools(
    class_name=class_name,
    results_dir=output_dir,
    log_level=log_level,
)

screenshots: list[Path] = []

test_image_num = int(0.7 * len(input_filenames))
test_image_path = output_dir / f"slice_{test_image_num:03d}_registered.mha"
if test_image_path.exists():
    test_image = itk.imread(str(test_image_path))
    screenshots.append(
        tt.save_screenshot_image_slice(
            test_image,
            f"slice_{test_image_num:03d}_registered_test.png",
            axis=0,
            slice_fraction=0.5,
            colormap="gray",
            vmin=-200,
            vmax=600,
        )
    )

    test_labelmap_path = output_dir / f"slice_{test_image_num:03d}_labelmap.mha"
    if test_labelmap_path.exists():
        test_labelmap = itk.imread(str(test_labelmap_path))
        screenshots.append(
            tt.save_screenshot_image_slice(
                test_image,
                f"slice_{test_image_num:03d}_labelmap_test.png",
                axis=0,
                slice_fraction=0.5,
                colormap="gray",
                vmin=-200,
                vmax=600,
                overlay_mask=test_labelmap,
            )
        )

if usd_file.exists():
    screenshots.append(
        tt.save_screenshot_openusd(
            usd_file,
            "cardiac_model_test.png",
        )
    )

tutorial_results = {"usd_file": str(usd_file), "screenshots": screenshots}
