"""
Tutorial 4: Create a PCA Statistical Shape Model

Purpose
-------
Build a PCA statistical shape model from a reference mesh and a small population
of sample meshes. Tutorial 5 can reuse the saved ``pca_model.json``.

Data Required
-------------
Full data: ``data/KCL-Heart-Model``
Test data: ``data/test/KCL-Heart-Model``
"""

# Imports
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import itk
import numpy as np
import pyvista as pv

from physiotwin4d import (
    ContourTools,
    SegmentNVSegmentCTMRI,
    TestTools,
    WorkflowConvertImageToVTK,
    WorkflowCreateStatisticalModel,
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

    class_name = "tutorial_04_lung_create_statistical_model"

    output_dir = tutorials_dir / "output" / "tutorial_04_lung"
    baselines_dir = repo_root / "tests" / "baselines"

    data_dir = repo_root / "data" / "DirLab-4DCT"
    pca_components = 7

    log_level = logging.INFO

    # Directory setup and data reading

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create lung surface files
    segmentation_method = SegmentNVSegmentCTMRI(log_level=log_level)
    workflow_method = WorkflowConvertImageToVTK(
        segmentation_method=segmentation_method, log_level=log_level
    )

    contour_tools = ContourTools(log_level=log_level)

    sample_image_files = sorted(data_dir.glob("Case*T70.mha"))
    sample_surfaces = []
    for sample_image_file in sample_image_files:
        sample_surface_file = output_dir / f"{sample_image_file.stem}.vtp"
        if not sample_surface_file.exists():
            sample_image = itk.imread(str(sample_image_file))
            result = workflow_method.process(
                input_image=sample_image,
                anatomy_groups=["lung"],
                extract_label_surfaces=True,
            )
            surfaces = result["label_surfaces"]
            contour_tools.save_combined_surfaces(surfaces, str(sample_surface_file))

            sample_labelmap = result["labelmap"]
            sample_labelmap_file = (
                output_dir / f"{sample_image_file.stem}_labelmap.nii.gz"
            )
            itk.imwrite(sample_labelmap, str(sample_labelmap_file))
        else:
            sample_surface = pv.read(str(sample_surface_file))
        sample_surfaces.append(sample_surface)

    reference_index = int(len(sample_surfaces) * 0.7)
    reference_surface = sample_surfaces[reference_index]

    # Workflow initialization

    workflow = WorkflowCreateStatisticalModel(
        sample_meshes=sample_surfaces,
        reference_mesh=reference_surface,
        pca_number_of_components=pca_components,
        log_level=log_level,
    )

    # Workflow execution
    result = workflow.process()

    # Result saving
    pca_model: dict[str, Any] = result["pca_model"]
    model_file = output_dir / "pca_model.json"
    with model_file.open("w", encoding="utf-8") as f:
        json.dump(pca_model, f, indent=2)

    mean_surface = result["pca_mean_surface"]
    mean_surface_file = output_dir / "pca_mean_surface.vtp"
    mean_surface.save(str(mean_surface_file))

    # Testing
    tt = TestTools(
        class_name=class_name,
        results_dir=output_dir,
        baselines_dir=baselines_dir,
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

    xvfb_started = False
    try:
        pv.start_xvfb()
        xvfb_started = True
    except Exception:
        pass

    try:
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
    finally:
        # Pair start_xvfb with cleanup, guarded like the startup above so
        # environments without Xvfb (e.g. Windows, pyvista >= 0.48) are unaffected.
        if xvfb_started:
            try:
                pv.stop_xvfb()
            except Exception:
                pass

    tutorial_results = {
        "pca_model": pca_model,
        "mean_surface": mean_surface,
        "model_file": model_file,
        "mean_surface_file": mean_surface_file,
        "screenshots": screenshots,
    }
