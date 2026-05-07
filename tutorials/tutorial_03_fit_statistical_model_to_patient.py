"""
Tutorial 3: Fit Statistical Shape Model to Patient Data

Purpose
-------
Register a generic anatomical template (e.g., a statistical heart model) to
patient-specific surface meshes derived from medical imaging. The multi-stage
pipeline performs ICP rough alignment, optional PCA-constrained shape fitting,
mask-based deformable registration, and optional image-based refinement. The
result is a patient-specific instance of the template mesh that matches the
patient anatomy.

Inputs
------
- Template model (``pv.UnstructuredGrid`` / ``.vtu``): generic anatomical mesh,
  e.g., the KCL heart model.
  Expected location: ``data/KCL-Heart-Model/pca_mean.vtu``
- Patient surface models (list of ``pv.PolyData`` / ``.vtp``): anatomy surfaces
  extracted from the patient CT (e.g., from Tutorial 2).
  Expected location: ``data/KCL-Heart-Model/sample_meshes/*.vtu`` (used as
  stand-in patient models for demonstration).
- Optional patient CT image (``itk.Image``): used for image-based refinement.

Outputs
-------
- ``output_dir/registered_template.vtp`` - template mesh fitted to patient
- Screenshots (PNG):
  - ``model_before_registration.png`` - template and patient overlaid (pre-ICP)
  - ``model_after_registration.png`` - registered template on patient

Strengths
---------
- Combines three complementary registration strategies in sequence, each
  correcting different scales of misalignment.
- PCA-constrained fitting (optional) prevents anatomically implausible
  deformations by constraining shape variation to the training population.
- Automatic mask generation means patient meshes are the only required input;
  a CT image is optional.

Weaknesses / Limitations
------------------------
- Requires the KCL-Heart-Model dataset (manual download; see data/README.md).
- Deformable registration (ANTs) is the slowest stage (~5-15 min on CPU).
- PCA mode is only beneficial when the template was trained on a population
  that includes the patient's anatomical variant.
- ICON-based image refinement requires a GPU.

Classes Used
------------
- WorkflowFitStatisticalModelToPatient (workflow_fit_statistical_model_to_patient.py):
    Orchestrates ICP -> (optional PCA) -> mask-to-mask -> (optional image) pipeline.
- RegisterModelsICP (register_models_icp.py):
    Centroid alignment followed by ICP affine registration (used internally).
- RegisterModelsDistanceMaps (register_models_distance_maps.py):
    ANTs deformable registration via signed distance maps (used internally).
- ContourTools (contour_tools.py):
    Creates reference images and masks from meshes (used internally).

CLI Equivalent
--------------
The same main outputs (without screenshots) can be produced via the CLI::

    physiomotion4d-fit-statistical-model-to-patient \\
        --template-model data/KCL-Heart-Model/pca_mean.vtu \\
        --patient-models data/KCL-Heart-Model/sample_meshes/sample_000.vtu \\
        --output-dir ./output/tutorial_03

See ``src/physiomotion4d/cli/fit_statistical_model_to_patient.py`` for full
CLI documentation.

Data Required
-------------
See data/README.md for download instructions and dataset licensing.
Dataset: KCL-Heart-Model - manual download required.
Place files under ``data/KCL-Heart-Model/`` as described in data/README.md.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, cast

import pyvista as pv

from physiomotion4d.workflow_fit_statistical_model_to_patient import (
    WorkflowFitStatisticalModelToPatient,
)


def _extract_surface(mesh: pv.DataSet) -> pv.PolyData:
    """Extract a PolyData surface using the library's standard VTK algorithm."""
    if isinstance(mesh, pv.PolyData):
        return mesh
    return cast(pv.PolyData, mesh.extract_surface(algorithm="dataset_surface"))


def run_tutorial(
    data_dir: Path,
    output_dir: Path,
    *,
    log_level: int = logging.INFO,
) -> dict[str, Any]:
    """Run Tutorial 3: Fit Statistical Shape Model to Patient Data.

    Args:
        data_dir: Root of the ``data/`` directory (see data/README.md).
        output_dir: Directory to write outputs and screenshots.
        log_level: Python logging level.

    Returns:
        dict with keys:

        - ``'registered_model'`` (pv.PolyData): fitted template surface.
        - ``'registered_file'`` (Path): path to saved ``.vtp``.
        - ``'screenshots'`` (list[Path]): paths to saved PNG screenshots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    kcl_dir = data_dir / "KCL-Heart-Model"
    template_file = kcl_dir / "pca_mean.vtu"
    if not template_file.exists():
        raise FileNotFoundError(
            f"KCL-Heart-Model template not found: {template_file}\n"
            "See data/README.md for manual download instructions."
        )

    template_model = pv.read(str(template_file))
    template_model = _extract_surface(template_model)

    # Use a subset of sample meshes as stand-in patient models
    sample_files = sorted((kcl_dir / "sample_meshes").glob("*.vtu"))[:3]
    if not sample_files:
        sample_files = sorted(kcl_dir.glob("*.vtu"))[:3]
    if not sample_files:
        raise FileNotFoundError(
            f"No sample meshes found under {kcl_dir}.\n"
            "See data/README.md for manual download instructions."
        )
    patient_models = []
    for sample_file in sample_files:
        sample_model = pv.read(str(sample_file))
        patient_models.append(_extract_surface(sample_model))

    workflow = WorkflowFitStatisticalModelToPatient(
        template_model=template_model,
        patient_models=patient_models,
        log_level=log_level,
    )
    result = workflow.run_workflow()

    registered_surface: pv.PolyData = result["registered_template_model_surface"]
    registered_file = output_dir / "registered_template.vtp"
    registered_surface.save(str(registered_file))

    screenshots: list[Path] = []

    patient_combined = (
        pv.merge(patient_models) if len(patient_models) > 1 else patient_models[0]
    )

    # Before: template (blue) + patient (red)
    try:
        pv.start_xvfb()
    except Exception:
        pass
    plotter = pv.Plotter(off_screen=True, window_size=[800, 600])
    plotter.add_mesh(
        template_model,
        color="dodgerblue",
        opacity=0.6,
        label="Template",
    )
    plotter.add_mesh(
        _extract_surface(patient_combined),
        color="tomato",
        opacity=0.6,
        label="Patient",
    )
    plotter.camera_position = "iso"
    before_path = output_dir / "model_before_registration.png"
    before_path.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(before_path))
    plotter.close()
    screenshots.append(before_path)

    # After: registered template (green) + patient (red)
    plotter2 = pv.Plotter(off_screen=True, window_size=[800, 600])
    plotter2.add_mesh(
        registered_surface, color="limegreen", opacity=0.7, label="Registered"
    )
    plotter2.add_mesh(
        _extract_surface(patient_combined),
        color="tomato",
        opacity=0.4,
        label="Patient",
    )
    plotter2.camera_position = "iso"
    after_path = output_dir / "model_after_registration.png"
    plotter2.screenshot(str(after_path))
    plotter2.close()
    screenshots.append(after_path)

    return {
        "registered_model": registered_surface,
        "registered_file": registered_file,
        "screenshots": screenshots,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Root data directory (default: ./data)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output") / "tutorial_03",
        help="Output directory (default: ./output/tutorial_03)",
    )
    args = parser.parse_args()

    results = run_tutorial(args.data_dir, args.output_dir)
    print(f"Registered model: {results['registered_file']}")
    print(f"Screenshots:      {[str(p) for p in results['screenshots']]}")
