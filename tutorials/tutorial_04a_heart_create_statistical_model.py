"""
Tutorial 4a: Create a PCA Statistical Shape Model

Purpose
-------
Build a PCA statistical shape model from a reference mesh and a small population
of sample meshes. Tutorial 5a can reuse the saved ``pca_model.json``.

Data Required
-------------
Full data: ``data/KCL-Heart-Model``
Test data: ``data/test/KCL-Heart-Model``
"""

# %%
# Imports
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyvista as pv

from physiotwin4d.test_tools import TestTools
from physiotwin4d.workflow_create_statistical_model import (
    WorkflowCreateStatisticalModel,
)

# nnUNetv2 (used by TotalSegmentator inside several workflows) spawns a
# multiprocessing.Pool. On Windows the spawn start method re-imports this
# script in each child; without the __name__ == "__main__" guard around
# top-level work, that re-import fires the segmenter again and Python's
# spawn-cascade detector raises RuntimeError. Wrapping consistently across
# tutorials also matches the style of tutorial_01a.
if __name__ == "__main__":
    # %%
    # Data directory specification
    REPO_ROOT = Path(__file__).resolve().parent.parent
    TUTORIALS_DIR = Path(__file__).resolve().parent
    DATA_DIR = REPO_ROOT / "data"
    FULL_DATA_DIR = DATA_DIR / "KCL-Heart-Model"
    TEST_DATA_DIR = DATA_DIR / "test" / "KCL-Heart-Model"
    OUTPUT_DIR = TUTORIALS_DIR / "output" / "tutorial_04a"
    BASELINES_DIR = REPO_ROOT / "tests" / "baselines"
    PCA_COMPONENTS = 10
    LOG_LEVEL = logging.INFO

    # %%
    # Data reading
    test_mode = TestTools.running_as_test()

    data_dir = TEST_DATA_DIR if test_mode else FULL_DATA_DIR
    output_dir = OUTPUT_DIR
    log_level = LOG_LEVEL

    if test_mode:
        pca_components = min(PCA_COMPONENTS, 5)
    else:
        pca_components = PCA_COMPONENTS

    output_dir.mkdir(parents=True, exist_ok=True)

    reference_file = data_dir / "average_mesh.vtk"
    if not reference_file.exists():
        raise FileNotFoundError(
            f"KCL-Heart-Model reference mesh not found: {reference_file}\n"
            "See data/README.md for download instructions."
        )

    sample_dir = data_dir / "input_meshes"
    sample_files = sorted(sample_dir.glob("*.vtk"))
    if not sample_files:
        sample_files = sorted(data_dir.glob("*.vtk"))
        sample_files = [
            path for path in sample_files if path.name != reference_file.name
        ]
    if len(sample_files) < 3:
        raise FileNotFoundError(
            f"Need at least 3 sample meshes under {sample_dir} or {data_dir}.\n"
            "See data/README.md for download instructions."
        )

    reference_mesh = cast(pv.DataSet, pv.read(str(reference_file)))
    sample_meshes = [cast(pv.DataSet, pv.read(str(path))) for path in sample_files]

    # %%
    # Workflow initialization
    workflow = WorkflowCreateStatisticalModel(
        sample_meshes=sample_meshes,
        reference_mesh=reference_mesh,
        pca_number_of_components=pca_components,
        log_level=log_level,
    )

    # %%
    # Workflow execution
    result = workflow.run_workflow()

    # %%
    # Result saving
    pca_model: dict[str, Any] = result["pca_model"]
    mean_surface: pv.PolyData = result["pca_mean_surface"]

    model_file = output_dir / "pca_model.json"
    with model_file.open("w", encoding="utf-8") as f:
        json.dump(pca_model, f, indent=2)

    mean_surface_file = output_dir / "pca_mean_surface.vtp"
    mean_surface.save(str(mean_surface_file))

    tt = TestTools(
        class_name="tutorial_04a_heart_create_statistical_model",
        results_dir=output_dir,
        baselines_dir=BASELINES_DIR,
        log_level=log_level,
    )

    screenshots: list[Path] = []
    screenshots.append(
        tt.save_screenshot_mesh(
            mean_surface,
            "pca_mean_model.png",
            camera_position="iso",
            color="steelblue",
            opacity=0.9,
        )
    )

    components = pca_model.get("components", [])
    eigenvalues = pca_model.get("eigenvalues", [])
    mean_points = np.asarray(mean_surface.points)
    mode_count = min(2, pca_components, len(components), len(eigenvalues))

    try:
        pv.start_xvfb()
    except Exception:
        pass

    for mode_idx in range(mode_count):
        sigma = float(np.sqrt(eigenvalues[mode_idx]))
        mode_offsets = np.asarray(components[mode_idx]).reshape(-1, 3)

        minus_mesh = mean_surface.copy()
        minus_mesh.points = mean_points - 2.0 * sigma * mode_offsets
        plus_mesh = mean_surface.copy()
        plus_mesh.points = mean_points + 2.0 * sigma * mode_offsets

        plotter = pv.Plotter(off_screen=True, window_size=[1200, 500], shape=(1, 3))
        plotter.subplot(0, 0)
        plotter.add_mesh(minus_mesh, color="royalblue", opacity=0.9)
        plotter.camera_position = "iso"
        plotter.subplot(0, 1)
        plotter.add_mesh(mean_surface, color="steelblue", opacity=0.9)
        plotter.camera_position = "iso"
        plotter.subplot(0, 2)
        plotter.add_mesh(plus_mesh, color="coral", opacity=0.9)
        plotter.camera_position = "iso"

        png_path = output_dir / f"pca_mode_{mode_idx + 1:02d}.png"
        plotter.screenshot(str(png_path))
        plotter.close()
        screenshots.append(png_path)

    tutorial_results = {
        "pca_model": pca_model,
        "mean_surface": mean_surface,
        "model_file": model_file,
        "mean_surface_file": mean_surface_file,
        "screenshots": screenshots,
    }
