"""
Tutorial 5: VTK Surface Series to USD

Purpose
-------
Convert the VTK surface output from Tutorial 2, or another VTK-compatible mesh,
into a USD file with anatomy materials.

Data Required
-------------
Preferred input: ``tutorials/output/tutorial_02/patient_surfaces.vtp``
Fallback input: any ``*.vtp`` under ``data`` or ``data/test``
"""

# %%
# Imports
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pyvista as pv

from physiotwin4d.test_tools import TestTools
from physiotwin4d.workflow_convert_vtk_to_usd import WorkflowConvertVTKToUSD

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
    FULL_DATA_DIR = DATA_DIR
    TEST_DATA_DIR = DATA_DIR / "test"
    OUTPUT_DIR = TUTORIALS_DIR / "output" / "tutorial_05"
    BASELINES_DIR = REPO_ROOT / "tests" / "baselines"
    TUTORIAL_02_SURFACE = (
        TUTORIALS_DIR / "output" / "tutorial_02" / "patient_surfaces.vtp"
    )
    VTK_FILE: Optional[Path] = None
    LOG_LEVEL = logging.INFO

    # %%
    # Data reading
    test_mode = TestTools.running_as_test()

    data_dir = TEST_DATA_DIR if test_mode else FULL_DATA_DIR
    output_dir = OUTPUT_DIR
    vtk_file = VTK_FILE
    log_level = LOG_LEVEL

    output_dir.mkdir(parents=True, exist_ok=True)

    if vtk_file is None and TUTORIAL_02_SURFACE.exists():
        vtk_file = TUTORIAL_02_SURFACE
    if vtk_file is None:
        vtk_candidates = sorted(data_dir.rglob("*.vtp"))
        if not vtk_candidates:
            raise FileNotFoundError(
                "No VTK surface file found. Run Tutorial 2 first or place a "
                f"*.vtp file under {data_dir}."
            )
        vtk_file = vtk_candidates[0]

    # %%
    # Workflow initialization
    mesh = pv.read(str(vtk_file))
    workflow = WorkflowConvertVTKToUSD(
        input_meshes=[mesh],
        usd_project_name="surfaces",
        output_directory=output_dir,
        appearance="anatomy",
        anatomy_type="heart",
        separate_by_connectivity=True,
        log_level=log_level,
    )

    # %%
    # Workflow execution
    usd_file = workflow.process()

    # %%
    # Result saving
    tt = TestTools(
        class_name="tutorial_05_vtk_to_usd",
        results_dir=output_dir,
        baselines_dir=BASELINES_DIR,
        log_level=log_level,
    )

    screenshots: list[Path] = []
    screenshots.append(
        tt.save_screenshot_openusd(
            usd_file,
            "usd_mesh_rendering.png",
        )
    )

    tutorial_results = {"usd_file": usd_file, "screenshots": screenshots}
