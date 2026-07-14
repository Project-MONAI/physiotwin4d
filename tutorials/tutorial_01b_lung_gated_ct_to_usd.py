"""
Tutorial 1b: Lung-Gated 4D CT to Animated USD

Purpose
-------
Convert a respiratory-gated 4D lung CT scan (multiple breathing phases) into an
animated USD model suitable for visualization in NVIDIA Omniverse. The workflow
segments the lungs and surrounding chest anatomy from a reference phase,
registers all other respiratory phases to that reference using deep-learning
registration, and assembles the resulting time-varying surface meshes into a
single USD file with anatomical materials applied.

Inputs
------
- A set of 3D CT volumes (``*.mha``) representing successive respiratory
  phases of one DirLab-4DCT case.
  Expected location: ``data/DirLab-4DCT/Case1Pack_T??.mha`` (already converted
  to Hounsfield units by ``data/DirLab-4DCT/fix_downloaded_data.py``).
- The mid-inspiration phase (index ~0.7 through the series) is used as the
  reference frame for segmentation and registration.

Outputs
-------
- An animated USD file with anatomy materials, written under ``output_dir``
  and named after the workflow's ``usd_project_name``.
- Screenshots (PNG) for documentation and regression testing:
  - ``slice_<n>_registered_test.png`` - axial slice of the registered
    reference phase
  - ``slice_<n>_labelmap_test.png`` - segmentation mask overlaid on that slice
  - a rendered view of the exported USD model

Strengths
---------
- Single call (``WorkflowConvertImageToUSD.process()``) runs the full pipeline.
- Supports both GPU-accelerated ICON registration and CPU-capable Greedy registration.
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
    Orchestrates the full pipeline: CT phases -> segmentation -> registration ->
    contour extraction -> USD export.
- SegmentChestTotalSegmentator (segment_chest_total_segmentator.py):
    Deep-learning segmentation of 117 anatomical structures (used internally).
- RegisterImagesICON (register_images_icon.py):
    Frame-to-frame image registration (used internally).
- ContourTools (contour_tools.py):
    Extracts and transforms surface meshes from segmentation masks (used internally).
- USDAnatomyTools (usd_anatomy_tools.py):
    Applies clinical material colours to USD prims (used internally).

Data Required
-------------
See data/README.md for download instructions and dataset licensing.
Dataset: DirLab-4DCT - see ``data/DirLab-4DCT/README.md``.
This script expects the HU-corrected ``Case1Pack_T??.mha`` phase volumes to
already exist under ``data/DirLab-4DCT/``. Download the DirLab-4DCT case and run
``data/DirLab-4DCT/fix_downloaded_data.py`` before running this tutorial.
"""

# %%
# Imports
from __future__ import annotations

import logging
from pathlib import Path

import itk

from physiotwin4d.register_images_icon import RegisterImagesICON
from physiotwin4d.test_tools import TestTools
from physiotwin4d.workflow_convert_image_to_usd import (
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
    FULL_DATA_DIR = DATA_DIR / "DirLab-4DCT"
    TEST_DATA_DIR = DATA_DIR / "test" / "DirLab-4DCT"
    OUTPUT_DIR = TUTORIALS_DIR / "output" / "tutorial_01b"
    LOG_LEVEL = logging.INFO

    # %%
    # Data reading
    test_mode = TestTools.running_as_test()

    data_dir = TEST_DATA_DIR if test_mode else FULL_DATA_DIR
    output_dir = OUTPUT_DIR
    log_level = LOG_LEVEL

    output_dir.mkdir(parents=True, exist_ok=True)

    if test_mode:
        number_of_registration_iterations = 1
    else:
        number_of_registration_iterations = 10

    # %%
    # .mha files are DirLab-4DCT data already converted to HU by
    # data/DirLab-4DCT/fix_downloaded_data.py.
    frame_files = sorted(data_dir.glob("Case1Pack_T??.mha"))
    if test_mode:
        frame_files = frame_files[:2]

    input_filenames = [str(path) for path in frame_files]
    if not input_filenames:
        raise FileNotFoundError(
            "DirLab-4DCT data not found. Checked:\n"
            + f"  - {data_dir}"
            + "\n"
            + "See data/README.md for download instructions."
        )

    time_series_images = [itk.imread(str(path)) for path in input_filenames]
    reference_image = time_series_images[int(0.7 * len(time_series_images))]

    print("Number of time-series images:", len(time_series_images))

    # %%
    # Workflow initialization
    registration_method = RegisterImagesICON(log_level=log_level)
    registration_method.set_number_of_iterations(number_of_registration_iterations)

    workflow = WorkflowConvertImageToUSD(
        time_series_images=time_series_images,
        reference_image=reference_image,
        output_directory=str(output_dir),
        usd_project_name="lung_model",
        registration_method=registration_method,
        log_level=log_level,
        save_assets=True,
    )

    # %%
    # Workflow execution
    usd_files = workflow.process()
    # if dynamic_labelmap_ids is not None, there are two USD files
    if len(workflow.dynamic_labelmap_ids) > 0:
        usd_file = output_dir / usd_files["dynamic"]
    else:
        usd_file = output_dir / usd_files["all"]

    # %%
    # Result saving
    tt = TestTools(
        class_name="tutorial_01b_lung_gated_ct_to_usd",
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
                "lung_model_test.png",
            )
        )

    tutorial_results = {"usd_file": str(usd_file), "screenshots": screenshots}
