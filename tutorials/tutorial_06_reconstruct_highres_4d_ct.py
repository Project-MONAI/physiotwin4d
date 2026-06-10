"""
Tutorial 6: Reconstruct High-Resolution 4D CT

Purpose
-------
Register a short CT time series to a fixed reference image and save the
reconstructed frames. DirLab does not provide a separate high-resolution
breath-hold reference image, so this tutorial uses one available respiratory
phase as the fixed reference.

Data Required
-------------
Full data: ``data/DirLab-4DCT/Case1``
Test data: ``data/test/DirLab-4DCT/Case1``
"""

# %%
# Imports
from __future__ import annotations

import logging
from pathlib import Path

import itk

from physiomotion4d.test_tools import TestTools
from physiomotion4d.workflow_reconstruct_highres_4d_ct import (
    WorkflowReconstructHighres4DCT,
)

# nnUNetv2 (used by TotalSegmentator inside several workflows) spawns a
# multiprocessing.Pool. On Windows the spawn start method re-imports this
# script in each child; without the __name__ == "__main__" guard around
# top-level work, that re-import fires the segmenter again and Python's
# spawn-cascade detector raises RuntimeError. Wrapping consistently across
# tutorials also matches the style of tutorial_01.
if __name__ == "__main__":
    # %%
    # Data directory specification
    REPO_ROOT = Path(__file__).resolve().parent.parent
    TUTORIALS_DIR = Path(__file__).resolve().parent
    DATA_DIR = REPO_ROOT / "data"
    FULL_DATA_DIR = DATA_DIR / "DirLab-4DCT" / "Case1"
    TEST_DATA_DIR = DATA_DIR / "test" / "DirLab-4DCT" / "Case1"
    OUTPUT_DIR = TUTORIALS_DIR / "output" / "tutorial_06"
    BASELINES_DIR = REPO_ROOT / "tests" / "baselines"
    MAX_FRAMES = 4
    REGISTRATION_METHOD = "Greedy_ICON"
    LOG_LEVEL = logging.INFO

    # %%
    # Data reading
    test_mode = TestTools.running_as_test()

    data_dir = TEST_DATA_DIR if test_mode else FULL_DATA_DIR
    output_dir = OUTPUT_DIR
    registration_method = REGISTRATION_METHOD
    log_level = LOG_LEVEL

    if test_mode:
        max_frames = min(MAX_FRAMES, 3)
        number_of_iterations_Greedy = [1, 0]
    else:
        max_frames = MAX_FRAMES
        number_of_iterations_Greedy = [30, 15, 7, 3]

    output_dir.mkdir(parents=True, exist_ok=True)

    phase_files = sorted(list(data_dir.glob("*.mhd")) + list(data_dir.glob("*.mha")))
    if not phase_files:
        raise FileNotFoundError(
            f"No DirLab phase images found under {data_dir}.\n"
            "See data/README.md for download instructions."
        )

    phase_files = phase_files[:max_frames]
    time_series = [itk.imread(str(path)) for path in phase_files]
    fixed_image = time_series[0]

    # %%
    # Workflow initialization
    workflow = WorkflowReconstructHighres4DCT(
        time_series_images=time_series,
        fixed_image=fixed_image,
        reference_frame=0,
        registration_method=registration_method,
        log_level=log_level,
    )
    workflow.set_modality("ct")
    workflow.set_number_of_iterations_Greedy(number_of_iterations_Greedy)

    # %%
    # Workflow execution
    result = workflow.run_workflow()

    # %%
    # Result saving
    reconstructed_images: list[itk.Image] = result["reconstructed_images"]
    reconstructed_files: list[Path] = []
    for frame_index, image in enumerate(reconstructed_images):
        out_path = output_dir / f"reconstructed_frame_{frame_index:03d}.mha"
        itk.imwrite(image, str(out_path), compression=True)
        reconstructed_files.append(out_path)

    tt = TestTools(
        class_name="tutorial_06_reconstruct_highres_4d_ct",
        results_dir=output_dir,
        baselines_dir=BASELINES_DIR,
        log_level=log_level,
    )

    screenshots: list[Path] = []
    screenshots.append(
        tt.save_screenshot_image_slice(
            fixed_image,
            "reference_frame.png",
            axis=0,
            slice_fraction=0.5,
            colormap="gray",
        )
    )
    if reconstructed_images:
        screenshots.append(
            tt.save_screenshot_image_slice(
                reconstructed_images[0],
                "reconstructed_frame.png",
                axis=0,
                slice_fraction=0.5,
                colormap="gray",
            )
        )

    tutorial_results = {
        "reconstructed_images": reconstructed_images,
        "reconstructed_files": reconstructed_files,
        "screenshots": screenshots,
    }
