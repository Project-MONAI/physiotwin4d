"""
Tutorial 3: VTK Surface Series to USD

Purpose
-------
Convert the VTK surface output from Tutorial 2, or another VTK-compatible mesh,
into a USD file with anatomy materials.

Data Required
-------------
Preferred input: ``tutorials/output/tutorial_02_heart/patient_surfaces.vtp``
Fallback input: any ``*.vtp`` under ``data`` or ``data/test``
"""

# Imports
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pyvista as pv

from physiotwin4d import (
    TestTools,
    WorkflowConvertVTKToUSD,
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

    class_name = "tutorial_03_heart_vtk_to_usd"

    output_dir = tutorials_dir / "output" / "tutorial_03_heart"
    baselines_dir = repo_root / "tests" / "baselines"

    # Preferred input: the combined surface saved by Tutorial 2. Leave vtk_file as
    # None to auto-discover (Tutorial 2 output first, then any *.vtp under data_dir).
    tutorial_02_surface = (
        tutorials_dir / "output" / "tutorial_02_heart" / "patient_surfaces.vtp"
    )
    vtk_file: Optional[Path] = None

    test_mode = TestTools.running_as_test()
    if test_mode:
        data_dir = repo_root / "data" / "test"
    else:
        data_dir = repo_root / "data"

    log_level = logging.INFO

    # Directory setup and data reading

    output_dir.mkdir(parents=True, exist_ok=True)

    if vtk_file is None and tutorial_02_surface.exists():
        vtk_file = tutorial_02_surface
    if vtk_file is None:
        vtk_candidates = sorted(data_dir.rglob("*.vtp"))
        if not vtk_candidates:
            raise FileNotFoundError(
                "No VTK surface file found. Run Tutorial 2 first or place a "
                f"*.vtp file under {data_dir}."
            )
        vtk_file = vtk_candidates[0]

    mesh = pv.read(str(vtk_file))

    # Workflow initialization

    workflow = WorkflowConvertVTKToUSD(
        input_meshes=[mesh],
        usd_project_name="surfaces",
        output_directory=output_dir,
        appearance="anatomy",
        anatomy_type="heart",
        separate_by_connectivity=True,
        log_level=log_level,
    )

    # Workflow execution
    usd_file = workflow.process()

    # Testing
    tt = TestTools(
        class_name=class_name,
        results_dir=output_dir,
        baselines_dir=baselines_dir,
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
