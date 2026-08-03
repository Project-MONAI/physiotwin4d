"""
Tutorial 5: Fit Statistical Shape Model to Patient Data

Purpose
-------
Fit a generic anatomical template mesh to one or more patient-like surface
meshes. If Tutorial 4 has already written ``pca_model.json``, the workflow uses
that model to constrain the fitted shape.

Data Required
-------------
PCA model: Tutorial 4 output (``output/tutorial_04_heart``)
Patient image: ``data/DirLab-4DCT`` (test: ``data/test/DirLab-4DCT``)
"""

# Imports
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional, cast

import itk
import pyvista as pv

from physiotwin4d import (
    ContourTools,
    SegmentChestTotalSegmentator,
    # SegmentHeartSimplewareTrimmedBranches,
    # SegmentChestTotalSegmentatorWithContrast,
    TestTools,
    WorkflowFitStatisticalModelToPatient,
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

    project_name = "tutorial_05_heart_to_lung"

    output_dir = tutorials_dir / "output" / project_name
    baselines_dir = repo_root / "tests" / "baselines"

    # PCA model + mean surface produced by Tutorial 4.
    pca_json = tutorials_dir / "output" / "tutorial_04_heart" / "pca_model.json"
    pca_mean_file = (
        tutorials_dir / "output" / "tutorial_04_heart" / "pca_mean_surface.vtp"
    )
    # BYOD example:
    # pca_mean_file = Path("D:/PhysioTwin4D/kcl-heart-pca/pca-vol-kcl/pca_mean.vtu")
    # pca_json = Path("D:/PhysioTwin4D/kcl-heart-pca/pca-vol-kcl/pca_model.json")

    test_mode = TestTools.running_as_test()
    if test_mode:
        data_dir = repo_root / "data" / "test" / "DirLab-4DCT"
    else:
        data_dir = repo_root / "data" / "DirLab-4DCT"
    patient_image_file = data_dir / "Case1Pack_T70.mha"

    log_level = logging.INFO

    segmentation_method = SegmentChestTotalSegmentator()
    segmentation_method.set_has_academic_license(True)
    # segmentation_method = SegmentHeartSimplewareTrimmedBranches() # Use when available
    #     and images are contrast-enhanced.
    # segmentation_method = SegmentChestTotalSegmentatorWithContrast() # Use when
    #     contrast-enhanced images and Simpleware is not available.

    # Directory setup and data reading

    output_dir.mkdir(parents=True, exist_ok=True)

    if not pca_mean_file.exists():
        raise FileNotFoundError(
            f"Tutorial 4 PCA mean surface not found: {pca_mean_file}\n"
            "Run Tutorial 4 first (see data/README.md for download instructions)."
        )
    pca_mean = cast(pv.DataSet, pv.read(str(pca_mean_file)))

    pca_model: Optional[dict[str, Any]] = None
    if pca_json.exists():
        with pca_json.open(encoding="utf-8") as f:
            pca_model = json.load(f)

    if not patient_image_file.exists():
        raise FileNotFoundError(
            f"DirLab-4DCT patient image not found: {patient_image_file}\n"
            "See data/README.md for download instructions."
        )
    patient_image = itk.imread(str(patient_image_file))

    if not (output_dir / f"{project_name}_patient_image.nii.gz").exists():
        itk.imwrite(
            patient_image,
            output_dir / f"{project_name}_patient_image.nii.gz",
            compression=True,
        )

        segmentation_result = segmentation_method.segment(patient_image)
        patient_labelmap = segmentation_result["labelmap"]
        itk.imwrite(
            patient_labelmap,
            output_dir / f"{project_name}_patient_labelmap.nii.gz",
            compression=True,
        )

        heart_labelmap = segmentation_result["heart"]
        itk.imwrite(
            heart_labelmap,
            output_dir / f"{project_name}_heart_labelmap.nii.gz",
            compression=True,
        )

        contour_tools = ContourTools()
        heart_surface = contour_tools.extract_contours(labelmap_image=heart_labelmap)
        heart_surface.save(output_dir / f"{project_name}_heart_surface.vtp")

    else:
        patient_labelmap = itk.imread(
            output_dir / f"{project_name}_patient_labelmap.nii.gz"
        )
        heart_labelmap = itk.imread(
            output_dir / f"{project_name}_heart_labelmap.nii.gz"
        )
        heart_surface = cast(
            pv.PolyData, pv.read(output_dir / f"{project_name}_heart_surface.vtp")
        )

    # Workflow initialization

    workflow = WorkflowFitStatisticalModelToPatient(
        template_model=pca_mean,
        patient_models=[heart_surface],
        patient_image=patient_image,
        patient_labelmap=heart_labelmap,
        log_level=log_level,
        labelmap_interior_object_ids=[141, 142, 143, 144],
        # These are the internal chambers of the heart when using TotalSegmentator.
    )
    if pca_model is not None:
        workflow.set_use_pca_registration(
            use_pca_registration=True,
            pca_model=pca_model,
            use_surface=False,
        )

    # Workflow execution
    workflow_results = workflow.process()

    # Result saving
    registered_coefficients = workflow.pca_coefficients
    if registered_coefficients is not None:
        registered_coefficients_path = (
            output_dir / f"{project_name}_registered_coefficients.json"
        )
        with registered_coefficients_path.open(mode="w", encoding="utf-8") as f:
            json.dump(registered_coefficients.tolist(), f)

    template_mesh = workflow.pca_template_model
    assert template_mesh is not None, "pca_template_model must be set after process()"
    template_mesh.save(str(output_dir / f"{project_name}_template_mesh.vtu"))

    template_surface = workflow.pca_template_model_surface
    assert template_surface is not None, (
        "pca_template_model_surface must be set after process()"
    )
    template_surface.save(str(output_dir / f"{project_name}_template_surface.vtp"))

    registered_mesh = workflow_results["registered_template_model"]
    registered_mesh.save(
        str(output_dir / f"{project_name}_template_mesh_registered.vtu")
    )

    registered_surface = workflow_results["registered_template_model_surface"]
    registered_surface.save(
        str(output_dir / f"{project_name}_template_surface_registered.vtp")
    )

    # Testing
    TestTools(
        class_name=project_name,
        results_dir=output_dir,
        baselines_dir=baselines_dir,
        log_level=log_level,
    )

    try:
        pv.start_xvfb()
    except Exception:
        pass

    screenshots: list[Path] = []

    before_path = output_dir / f"{project_name}_model_before_registration.png"
    plotter = pv.Plotter(off_screen=True, window_size=[800, 600])
    plotter.add_mesh(pca_mean, color="dodgerblue", opacity=0.6)
    plotter.add_mesh(heart_surface, color="tomato", opacity=0.6)
    plotter.camera_position = "iso"
    plotter.screenshot(str(before_path))
    plotter.close()
    screenshots.append(before_path)

    after_path = output_dir / f"{project_name}_model_after_registration.png"
    plotter = pv.Plotter(off_screen=True, window_size=[800, 600])
    plotter.add_mesh(registered_surface, color="limegreen", opacity=0.7)
    plotter.add_mesh(heart_surface, color="tomato", opacity=0.4)
    plotter.camera_position = "iso"
    plotter.screenshot(str(after_path))
    plotter.close()
    screenshots.append(after_path)

    tutorial_results = {
        "registered_mesh": registered_mesh,
        "registered_surface": registered_surface,
        "screenshots": screenshots,
    }
