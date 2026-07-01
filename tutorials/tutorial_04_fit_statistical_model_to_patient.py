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

import pyvista as pv

from physiomotion4d.test_tools import TestTools
from physiomotion4d.workflow_fit_statistical_model_to_patient import (
    WorkflowFitStatisticalModelToPatient,
)

# nnUNetv2 (used by TotalSegmentator inside several workflows) spawns a
# multiprocessing.Pool. On Windows the spawn start method re-imports this
# script in each child; without the __name__ == "__main__" guard around
# top-level work, that re-import fires the segmenter again and Python's
# spawn-cascade detector raises RuntimeError.
if __name__ == "__main__":
    # %%
    # Data directory specification
    REPO_ROOT = Path(__file__).resolve().parent.parent
    TUTORIALS_DIR = Path(__file__).resolve().parent
    DATA_DIR = REPO_ROOT / "data"
    FULL_DATA_DIR = DATA_DIR / "KCL-Heart-Model"
    TEST_DATA_DIR = DATA_DIR / "test" / "KCL-Heart-Model"
    OUTPUT_DIR = TUTORIALS_DIR / "output" / "tutorial_04"
    BASELINES_DIR = REPO_ROOT / "tests" / "baselines"
    PCA_JSON = TUTORIALS_DIR / "output" / "tutorial_03" / "pca_model.json"
    LOG_LEVEL = logging.INFO

    # %%
    # Data reading
    test_mode = TestTools.running_as_test()

    data_dir = TEST_DATA_DIR if test_mode else FULL_DATA_DIR
    output_dir = OUTPUT_DIR
    pca_json = PCA_JSON
    log_level = LOG_LEVEL

    output_dir.mkdir(parents=True, exist_ok=True)

    template_file = data_dir / "pca_mean.vtu"
    if not template_file.exists():
        raise FileNotFoundError(
            f"KCL-Heart-Model template not found: {template_file}\n"
            "See data/README.md for download instructions."
        )

    template_data = cast(pv.DataSet, pv.read(str(template_file)))
    if isinstance(template_data, pv.PolyData):
        template_model = template_data
    else:
        template_model = template_data.extract_surface(algorithm="dataset_surface")

    sample_files = sorted((data_dir / "sample_meshes").glob("*.vtu"))
    if not sample_files:
        sample_files = sorted(data_dir.glob("*.vtu"))
    sample_files = [path for path in sample_files if path.name != "pca_mean.vtu"]
    sample_files = sample_files[:3]
    if not sample_files:
        raise FileNotFoundError(
            f"No patient-like sample meshes found under {data_dir}.\n"
            "See data/README.md for download instructions."
        )

    patient_models: list[pv.DataSet] = []
    for sample_file in sample_files:
        sample_data = cast(pv.DataSet, pv.read(str(sample_file)))
        if isinstance(sample_data, pv.PolyData):
            patient_models.append(sample_data)
        else:
            patient_models.append(
                sample_data.extract_surface(algorithm="dataset_surface")
            )

    pca_model: Optional[dict[str, Any]] = None
    if pca_json.exists():
        with pca_json.open(encoding="utf-8") as f:
            pca_model = json.load(f)

    # %%
    # Workflow initialization
    workflow = WorkflowFitStatisticalModelToPatient(
        template_model=template_model,
        patient_models=patient_models,
        log_level=log_level,
    )
    if pca_model is not None:
        workflow.set_use_pca_registration(True, pca_model=pca_model)

    # %%
    # Workflow execution
    result = workflow.run_workflow()

    # %%
    # Result saving
    registered_surface: pv.PolyData = result["registered_template_model_surface"]
    registered_file = output_dir / "registered_template.vtp"
    registered_surface.save(str(registered_file))

    patient_combined = (
        pv.merge(patient_models) if len(patient_models) > 1 else patient_models[0]
    )
    patient_surface = cast(pv.PolyData, patient_combined)

    try:
        pv.start_xvfb()
    except Exception:
        pass

    screenshots: list[Path] = []

    before_path = output_dir / "model_before_registration.png"
    plotter = pv.Plotter(off_screen=True, window_size=[800, 600])
    plotter.add_mesh(template_model, color="dodgerblue", opacity=0.6)
    plotter.add_mesh(patient_surface, color="tomato", opacity=0.6)
    plotter.camera_position = "iso"
    plotter.screenshot(str(before_path))
    plotter.close()
    screenshots.append(before_path)

    after_path = output_dir / "model_after_registration.png"
    plotter = pv.Plotter(off_screen=True, window_size=[800, 600])
    plotter.add_mesh(registered_surface, color="limegreen", opacity=0.7)
    plotter.add_mesh(patient_surface, color="tomato", opacity=0.4)
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
        "registered_model": registered_surface,
        "registered_file": registered_file,
        "screenshots": screenshots,
    }
