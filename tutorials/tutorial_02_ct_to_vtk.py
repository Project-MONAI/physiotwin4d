"""
Tutorial 2: CT Segmentation to VTK Surfaces

Purpose
-------
Segment one 3D CT frame into anatomical groups and save combined VTK surface
and voxel mesh files. The output can be inspected directly in PyVista or used
as input for Tutorial 5.

Data Required
-------------
Full data: ``data/Slicer-Heart-CT/slice_???.mha``
Test data: ``data/test/slicer_heart_small/slice_???.mha``
"""

# %%
# Imports
from __future__ import annotations

import logging
from pathlib import Path

import itk
import pyvista as pv

from physiomotion4d.test_tools import TestTools
from physiomotion4d.workflow_convert_image_to_vtk import WorkflowConvertImageToVTK

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
    FULL_DATA_DIR = DATA_DIR / "Slicer-Heart-CT"
    TEST_DATA_DIR = DATA_DIR / "test" / "slicer_heart_small"
    OUTPUT_DIR = TUTORIALS_DIR / "output" / "tutorial_02"
    BASELINES_DIR = REPO_ROOT / "tests" / "baselines"
    LOG_LEVEL = logging.INFO

    # %%
    # Data reading
    test_mode = TestTools.running_as_test()

    data_dir = TEST_DATA_DIR if test_mode else FULL_DATA_DIR
    output_dir = OUTPUT_DIR
    log_level = LOG_LEVEL

    output_dir.mkdir(parents=True, exist_ok=True)

    frame_files = sorted(data_dir.glob("slice_???.mha"))
    if not frame_files:
        raise FileNotFoundError(
            "Slicer-Heart-CT frame data not found. Checked:\n"
            + f"  - {data_dir}\n"
            + "See data/README.md for download instructions."
        )

    ct_file = frame_files[0]
    ct_image = itk.imread(str(ct_file))

    # %%
    # Workflow initialization
    workflow = WorkflowConvertImageToVTK(
        segmentation_method="ChestTotalSegmentator",
        log_level=log_level,
    )

    # %%
    # Workflow execution
    result = workflow.run_workflow(
        input_image=ct_image,
        contrast_enhanced_study=True,
    )

    # %%
    # Result saving
    surface_file = Path(
        WorkflowConvertImageToVTK.save_combined_surface(
            result["surfaces"],
            str(output_dir),
            prefix="patient",
        )
    )
    mesh_file = Path(
        WorkflowConvertImageToVTK.save_combined_mesh(
            result["meshes"],
            str(output_dir),
            prefix="patient",
        )
    )
    labelmap_file = output_dir / "patient_labelmap.mha"
    itk.imwrite(result["labelmap"], str(labelmap_file), compression=True)

    tt = TestTools(
        class_name="tutorial_02_ct_to_vtk",
        results_dir=output_dir,
        baselines_dir=BASELINES_DIR,
        log_level=log_level,
    )

    screenshots: list[Path] = []
    screenshots.append(
        tt.save_screenshot_image_slice(
            ct_image,
            "segmentation_overlay.png",
            axis=0,
            slice_fraction=0.5,
            colormap="gray",
            vmin=-200,
            vmax=600,
            overlay_mask=result["labelmap"],
        )
    )

    surfaces = [
        surface for surface in result["surfaces"].values() if surface is not None
    ]
    if surfaces:
        combined_surface = pv.merge(surfaces) if len(surfaces) > 1 else surfaces[0]
        screenshots.append(
            tt.save_screenshot_mesh(
                combined_surface,
                "vtk_surfaces.png",
                camera_position="iso",
                color="lightblue",
                opacity=0.85,
            )
        )

    tutorial_results = {
        "result": result,
        "surface_file": surface_file,
        "mesh_file": mesh_file,
        "labelmap_file": labelmap_file,
        "screenshots": screenshots,
    }
