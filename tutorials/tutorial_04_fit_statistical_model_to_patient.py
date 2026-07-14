"""
Tutorial 4: Fit Statistical Shape Model to Patient Data

Purpose
-------
Fit a generic anatomical template mesh to one or more patient-like surface
meshes. If Tutorial 3 has already written ``pca_model.json``, the workflow uses
that model to constrain the fitted shape.

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
from typing import Any, Optional, cast

import itk
import numpy as np
import pyvista as pv

from physiotwin4d import (
    ContourTools,
    SegmentChestTotalSegmentator,
    #SegmentHeartSimplewareTrimmedBranches,
    #SegmentChestTotalSegmentatorWithContrast,
    TestTools,
    WorkflowFitStatisticalModelToPatient,
)

# %%
# Data directory specification

# nnUNetv2 (used by TotalSegmentator inside several workflows) spawns a
# multiprocessing.Pool. On Windows the spawn start method re-imports this
# script in each child; without the __name__ == "__main__" guard around
# top-level work, that re-import fires the segmenter again and Python's
# spawn-cascade detector raises RuntimeError.
if __name__ == "__main__":
    REPO_ROOT = Path(__file__).resolve().parent.parent
    TUTORIALS_DIR = Path(__file__).resolve().parent
    DATA_DIR = REPO_ROOT / "data" / "DirLab-4DCT"
    OUTPUT_DIR = TUTORIALS_DIR / "output" / "tutorial_04"
    BASELINES_DIR = REPO_ROOT / "tests" / "baselines"
    PCA_JSON = TUTORIALS_DIR / "output" / "tutorial_03" / "pca_model.json"
    PCA_MEAN_FILE = TUTORIALS_DIR / "output" / "tutorial_03" / "pca_mean_surface.vtp"
    PATIENT_IMAGE_FILE = DATA_DIR / "Case1Pack_T70.mhd"
    SEGMENTATION_METHOD = SegmentChestTotalSegmentator()
    SEGMENTATION_METHOD.set_has_highres_heart_license(True)
    # SEGMENTATION_METHOD = SegmentHeartSimplewareTrimmedBranches() # Use when available
    #     and images are contrast-enhanced.
    # SEGMENTATION_METHOD = SegmentChestTotalSegmentatorWithContrast() # Use when contrast-enhanced
    #     images and Simpleware is not available.
    LOG_LEVEL = logging.INFO

    TEST_MODE = TestTools.running_as_test()

    data_dir = DATA_DIR
    output_dir = OUTPUT_DIR
    pca_json = PCA_JSON
    pca_mean_file = PCA_MEAN_FILE
    patient_image_file = PATIENT_IMAGE_FILE
    segmentation_method = SEGMENTATION_METHOD
    log_level = LOG_LEVEL

    output_dir.mkdir(parents=True, exist_ok=True)

    if not pca_mean_file.exists():
        raise FileNotFoundError(
            f"DirLab-4DCT template not found: {pca_mean_file}\n"
            "See data/README.md for download instructions."
        )
    pca_mean = cast(pv.DataSet, pv.read(str(pca_mean_file)))

    pca_model: Optional[dict[str, Any]] = None
    if pca_json.exists():
        with pca_json.open(encoding="utf-8") as f:
            pca_model = json.load(f)

    if not patient_image_file.exists():
        raise FileNotFoundError(
            f"DirLab-4DCT template not found: {patient_image_file}\n"
            "See data/README.md for download instructions."
        )
    patient_image = itk.imread(str(patient_image_file))
    
    # DirLab data is not in Hounsfield units, so we need to convert it to HU.
    patient_image_arr = itk.GetArrayFromImage(patient_image)
    patient_image_arr = patient_image_arr - 1024
    patient_image_arr = np.clip(patient_image_arr, -1024, 2048)
    patient_image_new = itk.GetImageFromArray(patient_image_arr)
    patient_image_new.CopyInformation(patient_image)
    patient_image = patient_image_new
    itk.imwrite(patient_image, output_dir / "patient_image.nii.gz")

    segmentation_result = segmentation_method.segment(
        patient_image_new,
    )
    patient_labelmap = segmentation_result["labelmap"]
    itk.imwrite(patient_labelmap, output_dir / "patient_labelmap.nii.gz")

    heart_labelmap = segmentation_result["heart"]
    itk.imwrite(heart_labelmap, output_dir / "heart_labelmap.nii.gz")

    contour_tools = ContourTools()
    heart_surface = contour_tools.extract_contours(
        labelmap_image=heart_labelmap,
    )
    heart_surface.save(output_dir / "heart_surface.vtp")

    # %%
    # Workflow initialization
    workflow = WorkflowFitStatisticalModelToPatient(
        template_model=pca_mean,
        patient_models=[heart_surface],
        patient_image=patient_image,
        patient_labelmap=heart_labelmap,
        log_level=log_level,
        labelmap_interior_object_ids=[141, 142, 143, 144],
    )
    workflow.set_use_pca_registration(
        use_pca_registration=True,
        pca_model=pca_model,
        use_surface=False,
    )

    # %%
    # Workflow execution
    workflow_results = workflow.process()

    # %%
    # Result saving
    registered_coefficients = workflow.pca_coefficients
    registered_coefficients_path = output_dir / "registered_coefficients.json"
    with registered_coefficients_path.open(mode="w", encoding="utf-8") as f:
        json.dump(registered_coefficients.tolist(), f)

    template_mesh = workflow.pca_template_model
    template_mesh.save(str(output_dir / "template_mesh.vtp"))

    template_surface = workflow.pca_template_model_surface
    template_surface.save(str(output_dir / "template_surface.vtp"))

    registered_mesh = workflow_results["registered_template_model"]
    registered_mesh.save(str(output_dir / "template_mesh_registered.vtp"))

    registered_surface = workflow_results["registered_template_model_surface"]
    registered_surface.save(str(output_dir / "template_surface_registered.vtp"))


    # %%
    try:
        pv.start_xvfb()
    except Exception:
        pass

    screenshots: list[Path] = []

    before_path = output_dir / "model_before_registration.png"
    plotter = pv.Plotter(off_screen=True, window_size=[800, 600])
    plotter.add_mesh(pca_mean, color="dodgerblue", opacity=0.6)
    plotter.add_mesh(heart_surface, color="tomato", opacity=0.6)
    plotter.camera_position = "iso"
    plotter.screenshot(str(before_path))
    plotter.close()
    screenshots.append(before_path)

    after_path = output_dir / "model_after_registration.png"
    plotter = pv.Plotter(off_screen=True, window_size=[800, 600])
    plotter.add_mesh(registered_surface, color="limegreen", opacity=0.7)
    plotter.add_mesh(heart_surface, color="tomato", opacity=0.4)
    plotter.camera_position = "iso"
    plotter.screenshot(str(after_path))
    plotter.close()
    screenshots.append(after_path)

    TestTools(
        class_name="tutorial_04_fit_statistical_model_to_patient",
        results_dir=output_dir,
        baselines_dir=BASELINES_DIR,
        log_level=log_level,
    )

    tutorial_results = {
        "registered_mesh": registered_mesh,
        "registered_surface": registered_surface,
        "screenshots": screenshots,
    }
