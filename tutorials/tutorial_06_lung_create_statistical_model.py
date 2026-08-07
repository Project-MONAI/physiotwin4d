"""
Tutorial 6 (Lung): Create a PCA Statistical Shape Model

Purpose
-------
Build a PCA statistical shape model of the lungs from the DIR-Lab population.
Each case's T70 phase is segmented, an unbiased mean surface is built with
``WorkflowCreateMeanSurface``, and the population is decomposed into shape
modes. Tutorials 7 and 8 reuse the saved ``pca_model.json``.

Data Required
-------------
Full data: ``data/DirLab-4DCT/Case*T70.mha``
DirLab-4DCT is not auto-downloaded — see ``data/DirLab-4DCT/README.md``.

Outputs (under ``tutorials/output/tutorial_06_lung/``)
-----------------------------------------------------
- ``<case>_T70.vtp`` / ``<case>_T70_labelmap.nii.gz`` - per-case segmentations,
  cached and reused by Tutorial 8
- ``reference_mean_surface.vtp`` - the unbiased atlas surface
- ``pca_model.json`` and ``pca_mean_surface.vtp`` - the shape model
- ``pca_mode_<k>_{minus,plus}_2sigma.vtp`` and ``pca_mode_<k>.png``

Runtime
-------
One GPU segmentation per case, then ``mean_surface_iterations`` deformable
registrations per case to build the atlas. This is the slowest of Tutorials
1-7; every intermediate is cached on disk, so a re-run is cheap.
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
    WorkflowCreateMeanSurface,
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

    class_name = "tutorial_06_lung_create_statistical_model"

    output_dir = tutorials_dir / "output" / "tutorial_06_lung"
    baselines_dir = repo_root / "tests" / "baselines"

    data_dir = repo_root / "data" / "DirLab-4DCT"

    number_of_pca_components = 5

    # Atlas iterations used to build the reference surface; 1 is a single
    # template-biased pass.
    mean_surface_iterations = 3

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
            itk.imwrite(sample_labelmap, str(sample_labelmap_file), compression=True)
        sample_surfaces.append(pv.read(str(sample_surface_file)))

    # The reference surface defines the topology every PCA input is expressed
    # in, so picking one case makes the model inherit that case's shape. Use the
    # unbiased mean of the population instead. Cached: it costs one deformable
    # registration per case per atlas iteration.
    reference_surface_file = output_dir / "reference_mean_surface.vtp"
    if not reference_surface_file.exists():
        mean_workflow = WorkflowCreateMeanSurface(
            surfaces=sample_surfaces, log_level=log_level
        )
        mean_workflow.set_number_of_iterations(mean_surface_iterations)
        mean_result = mean_workflow.process()
        mean_result["mean_surface"].save(str(reference_surface_file))
    reference_surface = pv.read(str(reference_surface_file))

    # Workflow initialization

    workflow = WorkflowCreateStatisticalModel(
        sample_meshes=sample_surfaces,
        reference_mesh=reference_surface,
        number_of_pca_components=number_of_pca_components,
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
    mode_count = number_of_pca_components

    mode_surface_files: list[Path] = []
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

            minus_file = output_dir / f"pca_mode_{mode_idx + 1:02d}_minus_2sigma.vtp"
            plus_file = output_dir / f"pca_mode_{mode_idx + 1:02d}_plus_2sigma.vtp"
            minus_mesh.save(str(minus_file))
            plus_mesh.save(str(plus_file))
            mode_surface_files.extend([minus_file, plus_file])

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
        "mode_surface_files": mode_surface_files,
        "screenshots": screenshots,
    }
