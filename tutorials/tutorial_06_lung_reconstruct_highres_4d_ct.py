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

# Imports
from __future__ import annotations

import logging
from pathlib import Path

import itk

from physiotwin4d import (
    RegisterImagesGreedyICON,
    TestTools,
    WorkflowReconstructHighres4DCT,
)

# Only run if this script is not imported as a module

# nnUNetv2 (used by TotalSegmentator inside several workflows) spawns a
# multiprocessing.Pool. On Windows the spawn start method re-imports this
# script in each child; without the __name__ == "__main__" guard around
# top-level work, that re-import fires the segmenter again and Python's
# spawn-cascade detector raises RuntimeError.
if __name__ == "__main__":
    # Data directory specification
    repo_root = Path(__file__).resolve().parent.parent
    tutorials_dir = Path(__file__).resolve().parent

    class_name = "tutorial_06_lung_reconstruct_highres_4d_ct"

    output_dir = tutorials_dir / "output" / "tutorial_06_lung"
    baselines_dir = repo_root / "tests" / "baselines"

    # .mha files are DirLab-4DCT data already converted to HU by
    # data/DirLab-4DCT/fix_downloaded_data.py.
    case_glob = "Case1Pack_T??.mha"

    test_mode = TestTools.running_as_test()
    if test_mode:
        data_dir = repo_root / "data" / "test" / "DirLab-4DCT"
        number_of_iterations_greedy = [1, 0]
    else:
        data_dir = repo_root / "data" / "DirLab-4DCT"
        number_of_iterations_greedy = [30, 15, 7, 3]

    log_level = logging.INFO

    registration_method = RegisterImagesGreedyICON(log_level=log_level)
    registration_method.greedy.set_number_of_iterations(number_of_iterations_greedy)
    registration_method.icon.set_mass_preservation(True)  # For non-contrast CT

    # Directory setup and data reading

    output_dir.mkdir(parents=True, exist_ok=True)

    phase_files = sorted(data_dir.glob(case_glob))
    if not phase_files:
        raise FileNotFoundError(
            f"No DirLab phase images found under {data_dir}.\n"
            "See data/README.md for download instructions."
        )

    time_series = [itk.imread(str(path)) for path in phase_files]
    reference_image = time_series[0]

    # Workflow initialization

    workflow = WorkflowReconstructHighres4DCT(
        time_series_images=time_series,
        reference_image=reference_image,
        reference_time_frame=6,
        registration_method=registration_method,
        log_level=log_level,
    )
    workflow.set_modality("ct")

    # Workflow execution
    result = workflow.process()

    # Result saving
    forward_transform = result["forward_transforms"]
    inverse_transform = result["inverse_transforms"]
    reconstructed_images: list[itk.Image] = result["reconstructed_images"]
    reconstructed_files: list[Path] = []
    for frame_index, image in enumerate(reconstructed_images):
        out_path = output_dir / f"reconstructed_frame_{frame_index:03d}.mha"
        itk.imwrite(image, str(out_path), compression=True)
        reconstructed_files.append(out_path)

        out_path = output_dir / f"reconstructed_frame_{frame_index:03d}_fwd.hdf"
        itk.transformwrite(forward_transform[frame_index], str(out_path))

        out_path = output_dir / f"reconstructed_frame_{frame_index:03d}_inv.hdf"
        itk.transformwrite(inverse_transform[frame_index], str(out_path))

    # Testing
    tt = TestTools(
        class_name=class_name,
        results_dir=output_dir,
        baselines_dir=baselines_dir,
        log_level=log_level,
    )

    screenshots: list[Path] = []
    screenshots.append(
        tt.save_screenshot_image_slice(
            reference_image,
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
