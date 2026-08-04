"""
Tutorial 5: VTK Surface Series to USD

Purpose
-------
Convert the VTK surface output from Tutorial 4, or another VTK-compatible mesh,
into a USD file with anatomy materials.

Data Required
-------------
Preferred input: ``tutorials/output/tutorial_04_heart/patient_surfaces.vtp``
"""

# Imports
from __future__ import annotations

import logging
from pathlib import Path

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

    class_name = "tutorial_05_heart_vtk_to_usd"

    output_dir = tutorials_dir / "output" / "tutorial_05_heart"
    baselines_dir = repo_root / "tests" / "baselines"

    project_name = "tutorial_04_heart"

    # Preferred input: the combined surface saved by Tutorial 4.
    vtk_file = tutorials_dir / "output" / "tutorial_04_heart" / "patient_surfaces.vtp"

    log_level = logging.INFO

    # Directory setup and data reading

    output_dir.mkdir(parents=True, exist_ok=True)

    mesh = pv.read(str(vtk_file))

    # Workflow initialization

    workflow = WorkflowConvertVTKToUSD(
        input_meshes=[mesh],
        usd_project_name=project_name,
        output_directory=output_dir,
        appearance="anatomy",
        anatomy_type="heart",
        separate_by_connectivity=True,
        log_level=log_level,
    )

    # Workflow execution
    results = workflow.process()

    # Testing
    tt = TestTools(
        class_name=class_name,
        results_dir=output_dir,
        baselines_dir=baselines_dir,
        log_level=log_level,
    )

    screenshots = [
        tt.save_screenshot_openusd(
            results["usd_file"],
            f"{project_name}_usd_mesh_rendering.png",
        )
    ]

    tutorial_results = {"usd_file": results["usd_file"], "screenshots": screenshots}
