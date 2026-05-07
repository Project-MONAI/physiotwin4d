"""
Tutorial 4: Create a PCA Statistical Shape Model

Purpose
-------
Build a PCA (Principal Component Analysis) statistical shape model from a
population of anatomical meshes aligned to a reference. The model captures
the mean shape and the principal modes of geometric variation across the
population. The resulting model can be used in Tutorial 3 to constrain
patient-specific fitting to anatomically plausible shapes.

Inputs
------
- A reference mesh (``pv.DataSet`` / ``.vtu``): defines the template topology.
  Expected location: ``data/KCL-Heart-Model/pca_mean.vtu``
- A collection of sample meshes (list of ``pv.DataSet`` / ``.vtu``):
  population shapes to learn from.
  Expected location: ``data/KCL-Heart-Model/sample_meshes/*.vtu``

Outputs
-------
- ``output_dir/pca_model.json`` - PCA model (eigenvectors, eigenvalues, mean)
- ``output_dir/pca_mean_surface.vtp`` - mean shape as a surface
- Screenshots (PNG):
  - ``pca_mean_model.png`` - 3-D view of the PCA mean surface
  - ``pca_mode_01.png`` - mean +/- 2 sigma for the first PCA mode (side-by-side)
  - ``pca_mode_02.png`` - mean +/- 2 sigma for the second PCA mode

Strengths
---------
- Single call to ``WorkflowCreateStatisticalModel.run_workflow()`` covers the
  full pipeline: ICP alignment, deformable correspondence, and PCA.
- Returns a pure-Python dict (eigenvectors as numpy arrays) compatible with
  ``WorkflowFitStatisticalModelToPatient.set_use_pca_registration()``.
- Surface-mode PCA (``solve_for_surface_pca=True``, default) is faster and
  sufficient for most cardiac applications.

Weaknesses / Limitations
------------------------
- Requires the KCL-Heart-Model dataset (manual download; see data/README.md).
- Population size directly affects model quality; small populations (<20 meshes)
  produce unreliable high-order modes.
- ICP alignment (step 2) assumes all sample meshes share a common approximate
  orientation; large pose variation may degrade correspondence.
- Deformable correspondence (step 3) uses ANTs, which is slow on CPU.

Classes Used
------------
- WorkflowCreateStatisticalModel (workflow_create_statistical_model.py):
    Runs the full pipeline: ICP -> deformable correspondence -> PCA.
- RegisterModelsICP (register_models_icp.py):
    Aligns each sample to the reference (used internally).
- RegisterModelsDistanceMaps (register_models_distance_maps.py):
    Dense deformable correspondence via signed distance maps (used internally).

CLI Equivalent
--------------
The same main outputs (without screenshots) can be produced via the CLI::

    physiomotion4d-create-statistical-model \\
        --sample-meshes-dir data/KCL-Heart-Model/sample_meshes \\
        --reference-mesh data/KCL-Heart-Model/pca_mean.vtu \\
        --pca-components 10 \\
        --output-dir ./output/tutorial_04

See ``src/physiomotion4d/cli/create_statistical_model.py`` for full CLI
documentation.

Data Required
-------------
See data/README.md for download instructions and dataset licensing.
Dataset: KCL-Heart-Model - manual download required.
Place files under ``data/KCL-Heart-Model/`` as described in data/README.md.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyvista as pv

from physiomotion4d.test_tools import TestTools
from physiomotion4d.workflow_create_statistical_model import (
    WorkflowCreateStatisticalModel,
)


def run_tutorial(
    data_dir: Path,
    output_dir: Path,
    *,
    pca_components: int = 10,
    max_samples: int = 20,
    log_level: int = logging.INFO,
) -> dict[str, Any]:
    """Run Tutorial 4: Create a PCA Statistical Shape Model.

    Args:
        data_dir: Root of the ``data/`` directory (see data/README.md).
        output_dir: Directory to write outputs and screenshots.
        pca_components: Number of PCA modes to retain.
        max_samples: Maximum number of sample meshes to use (cap for speed).
        log_level: Python logging level.

    Returns:
        dict with keys:

        - ``'pca_model'`` (dict): PCA model dict (eigenvectors, eigenvalues, mean).
        - ``'mean_surface'`` (pv.PolyData): mean shape surface.
        - ``'model_file'`` (Path): path to saved ``pca_model.json``.
        - ``'mean_surface_file'`` (Path): path to saved ``pca_mean_surface.vtp``.
        - ``'screenshots'`` (list[Path]): paths to saved PNG screenshots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    kcl_dir = data_dir / "KCL-Heart-Model"
    reference_file = kcl_dir / "pca_mean.vtu"
    if not reference_file.exists():
        raise FileNotFoundError(
            f"KCL-Heart-Model reference mesh not found: {reference_file}\n"
            "See data/README.md for manual download instructions."
        )

    sample_dir = kcl_dir / "sample_meshes"
    sample_files = sorted(sample_dir.glob("*.vtu"))
    if not sample_files:
        sample_files = sorted(kcl_dir.glob("*.vtu"))
    sample_files = [f for f in sample_files if f.name != "pca_mean.vtu"]
    sample_files = sample_files[:max_samples]
    if len(sample_files) < 3:
        raise FileNotFoundError(
            f"Need at least 3 non-reference sample meshes under {sample_dir} "
            f"or {kcl_dir}.\n"
            "See data/README.md for manual download instructions."
        )

    reference_mesh = cast(pv.DataSet, pv.read(str(reference_file)))
    sample_meshes = [cast(pv.DataSet, pv.read(str(f))) for f in sample_files]

    workflow = WorkflowCreateStatisticalModel(
        sample_meshes=sample_meshes,
        reference_mesh=reference_mesh,
        pca_number_of_components=pca_components,
        log_level=log_level,
    )
    result = workflow.run_workflow()

    mean_surface: pv.PolyData = result["mean_surface"]
    mean_surface_file = output_dir / "pca_mean_surface.vtp"
    mean_surface.save(str(mean_surface_file))

    # Serialise the JSON-safe parts of the PCA model
    pca_model: dict[str, Any] = result["pca_model"]
    model_file = output_dir / "pca_model.json"
    json_safe: dict[str, Any] = {}
    for k, v in pca_model.items():
        if isinstance(v, np.ndarray):
            json_safe[k] = v.tolist()
        elif isinstance(v, (int, float, str, bool, list)):
            json_safe[k] = v
    with open(model_file, "w") as fh:
        json.dump(json_safe, fh, indent=2)

    # Screenshots
    tt = TestTools(
        results_dir=output_dir,
        baselines_dir=output_dir / "baselines",
        class_name="tutorial_04",
        results_output_dir=output_dir,
        log_level=log_level,
    )

    screenshots: list[Path] = []

    # Mean model
    screenshots.append(
        tt.save_screenshot_mesh(
            mean_surface,
            "pca_mean_model.png",
            camera_position="iso",
            color="steelblue",
            opacity=0.9,
        )
    )

    # First two PCA modes: show mean +/- 2 sigma side-by-side
    eigenvectors: Any = pca_model.get("eigenvectors")
    eigenvalues: Any = pca_model.get("eigenvalues")
    mean_points = np.asarray(mean_surface.points)

    try:
        pv.start_xvfb()
    except Exception:
        pass

    for mode_idx in range(min(2, pca_components)):
        if eigenvectors is None or eigenvalues is None:
            break

        sigma = float(np.sqrt(eigenvalues[mode_idx]))
        ev = np.asarray(eigenvectors[:, mode_idx]).reshape(-1, 3)

        minus_mesh = mean_surface.copy()
        minus_mesh.points = mean_points - 2 * sigma * ev
        plus_mesh = mean_surface.copy()
        plus_mesh.points = mean_points + 2 * sigma * ev

        plotter = pv.Plotter(off_screen=True, window_size=[1200, 500], shape=(1, 3))
        plotter.subplot(0, 0)
        plotter.add_mesh(minus_mesh, color="royalblue", opacity=0.9)
        plotter.add_text("mean - 2 sigma", font_size=10)
        plotter.camera_position = "iso"
        plotter.subplot(0, 1)
        plotter.add_mesh(mean_surface, color="steelblue", opacity=0.9)
        plotter.add_text("mean", font_size=10)
        plotter.camera_position = "iso"
        plotter.subplot(0, 2)
        plotter.add_mesh(plus_mesh, color="coral", opacity=0.9)
        plotter.add_text("mean + 2 sigma", font_size=10)
        plotter.camera_position = "iso"

        png_name = f"pca_mode_{mode_idx + 1:02d}.png"
        png_path = output_dir / png_name
        png_path.parent.mkdir(parents=True, exist_ok=True)
        plotter.screenshot(str(png_path))
        plotter.close()
        screenshots.append(png_path)

    return {
        "pca_model": pca_model,
        "mean_surface": mean_surface,
        "model_file": model_file,
        "mean_surface_file": mean_surface_file,
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
        default=Path("output") / "tutorial_04",
        help="Output directory (default: ./output/tutorial_04)",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=10,
        help="Number of PCA modes to retain (default: 10)",
    )
    args = parser.parse_args()

    results = run_tutorial(
        args.data_dir,
        args.output_dir,
        pca_components=args.pca_components,
    )
    print(f"PCA model:    {results['model_file']}")
    print(f"Mean surface: {results['mean_surface_file']}")
    print(f"Screenshots:  {[str(p) for p in results['screenshots']]}")
