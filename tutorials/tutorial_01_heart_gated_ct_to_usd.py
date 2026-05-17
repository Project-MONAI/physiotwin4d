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
- Supports both GPU-accelerated ICON registration and CPU-capable ANTs registration.
- Automatically detects contrast enhancement and adjusts segmentation thresholds.
- Output is Omniverse-ready with anatomical materials (USDAnatomyTools).

Weaknesses / Limitations
------------------------
- Requires a GPU for ICON registration (``registration_method='ICON'``); use
  ``registration_method='ANTS'`` for CPU-only environments (about 10x slower).
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
- RegisterImagesICON / RegisterImagesANTs (register_images_icon.py / _ants.py):
    Frame-to-frame image registration (used internally).
- ContourTools (contour_tools.py):
    Extracts and transforms surface meshes from segmentation masks (used internally).
- USDAnatomyTools (usd_anatomy_tools.py):
    Applies clinical material colours to USD prims (used internally).

Data Required
-------------
See data/README.md for download instructions and dataset licensing.
Dataset: Slicer-Heart-CT - https://github.com/Slicer-Heart-CT/Slicer-Heart-CT
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

from physiomotion4d.test_tools import TestTools
from physiomotion4d.workflow_convert_image_to_usd import (
    WorkflowConvertImageToUSD,
)

# nnUNetv2 (used by TotalSegmentator inside WorkflowConvertImageToUSD)
# spawns a multiprocessing.Pool. On Windows the spawn start method re-imports
# this script in each child; without the __name__ == "__main__" guard around
# the top-level work, that re-import fires workflow.process() again and
# Python's spawn-cascade detector raises RuntimeError.
if __name__ == "__main__":
    # %%
    # Data directory specification
    REPO_ROOT = Path(__file__).resolve().parent.parent
    TUTORIALS_DIR = Path(__file__).resolve().parent
    DATA_DIR = REPO_ROOT / "data"
    FULL_DATA_DIR = DATA_DIR / "Slicer-Heart-CT"
    TEST_DATA_DIR = DATA_DIR / "test" / "slicer_heart_small"
    OUTPUT_DIR = TUTORIALS_DIR / "output" / "tutorial_01"
    REGISTRATION_METHOD = "ANTS"
    LOG_LEVEL = logging.INFO

    # %%
    # Data reading
    test_mode = TestTools.running_as_test()

    data_dir = TEST_DATA_DIR if test_mode else FULL_DATA_DIR
    output_dir = OUTPUT_DIR
    registration_method = REGISTRATION_METHOD
    log_level = LOG_LEVEL

    output_dir.mkdir(parents=True, exist_ok=True)

    if test_mode:
        number_of_registration_iterations = 1
    else:
        number_of_registration_iterations = 10

    # %%
    frame_files = sorted(data_dir.glob("slice_???.mha"))
    if test_mode:
        frame_files = frame_files[:2]

    input_filenames = [str(path) for path in frame_files]
    if not input_filenames:
        raise FileNotFoundError(
            "Slicer-Heart-CT data not found. Checked:\n"
            + f"  - {data_dir}"
            + "\n"
            + "See data/README.md for download instructions."
        )

    # %%
    # Workflow initialization
    workflow = WorkflowConvertImageToUSD(
        input_filenames=input_filenames,
        contrast_enhanced=True,
        output_directory=str(output_dir),
        project_name="cardiac_model",
        registration_method=registration_method,
        number_of_registration_iterations=number_of_registration_iterations,
        log_level=log_level,
        save_registered_images=True,
        save_registration_transforms=True,
        save_labelmaps=True,
    )

    # %%
    # Workflow execution
    usd_file = output_dir / workflow.process()

    # %%
    # Result saving
    tt = TestTools(
        class_name="tutorial_01_heart_gated_ct_to_usd",
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
