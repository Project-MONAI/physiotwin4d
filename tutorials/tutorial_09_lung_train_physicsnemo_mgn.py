"""
Tutorial 9 (Lung, MGN): Train a PhysicsNeMo MeshGraphNet on the Fitted Lung SSM

Purpose
-------
Runs on the public DIR-Lab 4D CT data. A thin driver over the reusable
:class:`physiotwin4d.WorkflowTrainPhysicsNeMo` workflow:

1. Discover the per-phase SSM surfaces produced by Tutorial 8
   (``tutorial_08_lung_fit_model_to_4d_patients.py``), write the training target
   for each phase, and write one JSON manifest per case. The target here is the
   per-vertex displacement from the case's reference surface, stored as a
   ``displacement`` point-data array — the workflow reads targets verbatim and
   never derives them. Respiratory stages are parsed from the ``T{PP}`` phase
   filenames and written explicitly into the manifest (the workflow never parses
   filenames).

2. Split the cases into train / validation / held-out test and train the
   MeshGraphNet (``WorkflowTrainPhysicsNeMo`` driving ``TrainPhysicsNeMoMGN``).

3. Evaluate the held-out test cases against their ground-truth phases with
   :class:`physiotwin4d.WorkflowInferPhysicsNeMo` wrapped in
   :class:`physiotwin4d.WorkflowInferMovement`.

Why a GNN?
----------
The SSM surface has a fixed topology across all cases and lung tissue is a
continuum: adjacent vertices co-vary smoothly. MeshGraphNet encodes that prior
directly by passing messages along mesh edges, giving an explicit
continuum-deformation inductive bias the MLP must infer from coordinates alone.

Node features (per vertex):   [mean_shape_x, mean_shape_y, mean_shape_z, pca_c1 ... pca_cN, stage]
Edge features (per edge):     [rel_x, rel_y, rel_z, distance]   (from the mean shape)
Output (per vertex):          [dx, dy, dz]  (displacement in mm)

Extra Install Required
----------------------
PhysicsNeMo and PyTorch Geometric must be installed::

    pip install "physiotwin4d[physicsnemo]"

Data Required
-------------
SSM surfaces: Tutorial 8 output (``output/tutorial_08_lung/Case*Pack/``)
PCA mean surface: Tutorial 6 output
(``output/tutorial_06_lung/pca_mean_surface.vtp``, alongside ``pca_model.json``)

Outputs
-------
Manifests are written under ``output/tutorial_09_lung_mgn/manifests_mgn/``:

  * ``Case*Pack_manifest.json``  - per-case training manifest
  * ``Case*Pack_T??_ssm_surface_target.vtp`` - per-phase displacement targets

The model and its evaluation land in the directory training actually used —
``output/tutorial_09_lung_mgn/`` normally, or a fresh ``..._2`` sibling when
resuming (see ``resume_from``), which is what ``tutorial_results`` reports as
``model_directory``:

  * ``mgn_stage_model.pt``      - trained MeshGraphNet checkpoint
  * ``eval_mgn/Case*Pack/``     - predicted surfaces per held-out case
  * ``predicted_surface.png`` / ``rmse_surface.png`` - screenshots
"""

# Imports
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import pyvista as pv

from physiotwin4d import (
    TestTools,
    TrainPhysicsNeMoMGN,
    WorkflowInferMovement,
    WorkflowInferPhysicsNeMo,
    WorkflowTrainPhysicsNeMo,
)

# Point-data array the tutorial writes its targets into and the manifests name.
TARGET_ARRAY = "displacement"


def _respiratory_stage_from_filename(surface_file: Path) -> float:
    """Extract the normalized respiratory stage [0, 1] from a ``T{PP}`` filename stem."""
    for part in surface_file.stem.split("_"):
        if part.startswith("T") and part[1:].isdigit():
            return int(part[1:]) / 100.0
    raise ValueError(f"Cannot parse respiratory phase from filename: {surface_file}")


def _write_target_mesh(
    phase_file: Path, ref_points: np.ndarray, targets_dir: Path
) -> Path:
    """Write one phase's training target and return the mesh path.

    The target is the per-vertex displacement from the case's reference surface,
    stored as the ``TARGET_ARRAY`` point-data array on a copy of the phase
    surface. Any other per-vertex quantity could be written here instead — the
    training workflow reads whatever array the manifest names.
    """
    phase_mesh = pv.read(str(phase_file))
    phase_points = np.asarray(phase_mesh.points, dtype=np.float32)
    phase_mesh.point_data[TARGET_ARRAY] = phase_points - ref_points
    target_path = targets_dir / f"{phase_file.stem}_target.vtp"
    phase_mesh.save(str(target_path))
    return target_path


def _write_case_manifest(case_dir: Path, manifests_dir: Path) -> Optional[Path]:
    """Write a per-case manifest JSON; return its path (or None if incomplete).

    A case needs a reference SSM surface, a PCA coefficient file, and at least
    two respiratory-phase surfaces.
    """
    case_id = case_dir.name
    ref_file = case_dir / f"{case_id}_ssm_surface.vtp"
    pca_file = case_dir / f"{case_id}_ssm_pca_coefficients.json"
    phase_files = sorted(case_dir.glob(f"{case_id}_T??_ssm_surface.vtp"))
    if not ref_file.exists() or not pca_file.exists() or len(phase_files) < 2:
        return None

    manifests_dir.mkdir(parents=True, exist_ok=True)
    ref_points = np.asarray(pv.read(str(ref_file)).points, dtype=np.float32)
    manifest = {
        "subject_id": case_id,
        "reference_mesh": str(ref_file),
        "pca_coefficients": str(pca_file),
        "target_array": TARGET_ARRAY,
        "phases": [
            {
                "mesh": str(_write_target_mesh(phase_file, ref_points, manifests_dir)),
                "stage": _respiratory_stage_from_filename(phase_file),
            }
            for phase_file in phase_files
        ],
    }
    manifest_path = manifests_dir / f"{case_id}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


# Only run if this script is not imported as a module

# PhysicsNeMo and torch spawn worker processes for data loading. On Windows the
# spawn start method re-imports this script in each child; without the
# __name__ == "__main__" guard around top-level work, that re-import would
# restart training in every worker.
if __name__ == "__main__":
    # Data directory specification
    tutorials_dir = Path(__file__).resolve().parent
    # Fitted SSM surfaces and PCA coefficients written by Tutorial 8 (lung).
    data_dir = tutorials_dir / "output" / "tutorial_08_lung"
    # PCA mean surface written by Tutorial 6 (lung); pca_model.json must sit
    # beside it, which is how Tutorial 6 writes them.
    ssm_mean_surface_file = (
        tutorials_dir / "output" / "tutorial_06_lung" / "pca_mean_surface.vtp"
    )
    # All outputs (manifests, checkpoints, evaluation surfaces) are written here.
    output_dir = tutorials_dir / "output" / "tutorial_09_lung_mgn"
    manifests_dir = output_dir / "manifests_mgn"

    # Warm-start from a previous run's checkpoint; None trains from scratch. When
    # resuming, training writes to a fresh sibling directory (``..._2``), e.g.
    # tutorials/output/tutorial_09_lung_mgn_2/mgn_stage_model_epoch_00200.pt
    resume_from: Optional[Path] = None

    # Training hyperparameters
    epochs = 1500
    batch_size = 4  # mini-batch measured in (case, phase) graphs
    learning_rate = 1.0e-3
    processor_size = 3  # message-passing hops
    hidden_dim = 128
    num_layers = 2  # MLP layers inside each encoder / processor / decoder block

    # Explicit held-out splits; every other discovered case is used for training.
    # Case1 is also the case held out of the Tutorial 2 ICON finetuning.
    test_cases = ["Case1Pack"]
    val_cases: list[str] = []
    log_level = logging.INFO

    class_name = "tutorial_09_lung_train_physicsnemo_mgn"
    logging.basicConfig(level=log_level)
    logger = logging.getLogger(class_name)

    # In test mode, train for a couple of epochs to keep the run tractable.
    test_mode = TestTools.running_as_test()
    if test_mode:
        epochs = 2

    if not ssm_mean_surface_file.exists():
        raise FileNotFoundError(
            f"Tutorial 6 PCA mean surface not found: {ssm_mean_surface_file}\n"
            "Run tutorials/tutorial_06_lung_create_statistical_model.py first."
        )

    # Step 1: build one manifest per valid case and partition into splits
    manifests: dict[str, Path] = {}
    for case_dir in sorted(data_dir.glob("Case*Pack")):
        manifest_path = _write_case_manifest(case_dir, manifests_dir)
        if manifest_path is not None:
            manifests[case_dir.name] = manifest_path

    if len(manifests) < 3:
        raise RuntimeError(
            f"Found only {len(manifests)} valid case(s) under {data_dir}; need at "
            "least 3 for a train / val / test split. Run "
            "tutorials/tutorial_08_lung_fit_model_to_4d_patients.py first."
        )

    unknown = [
        case_id for case_id in test_cases + val_cases if case_id not in manifests
    ]
    if unknown:
        raise ValueError(f"Split cases not found: {unknown}")

    test_manifests = [manifests[case_id] for case_id in test_cases]
    val_manifests = [manifests[case_id] for case_id in val_cases]
    train_manifests = [
        manifest_path
        for case_id, manifest_path in manifests.items()
        if case_id not in test_cases and case_id not in val_cases
    ]
    logger.info(
        "Case split - train: %d, val: %d, test: %d",
        len(train_manifests),
        len(val_manifests),
        len(test_manifests),
    )

    # Step 2: train the MeshGraphNet. The training method carries the network and
    # its hyper-parameters; the workflow feeds it manifests and saves the results.
    training_method = TrainPhysicsNeMoMGN(log_level=log_level)
    training_method.set_epochs(epochs)
    training_method.set_batch_size(batch_size)
    training_method.set_learning_rate(learning_rate)
    training_method.set_processor_size(processor_size)
    training_method.set_hidden_dim(hidden_dim)
    training_method.set_num_layers(num_layers)

    train_workflow = WorkflowTrainPhysicsNeMo(
        train_manifests=train_manifests,
        val_manifests=val_manifests,
        pca_mean_mesh=ssm_mean_surface_file,
        output_directory=output_dir,
        resume_from=resume_from,
        training_method=training_method,
        log_level=log_level,
    )
    train_result = train_workflow.process()

    # Step 3: evaluate held-out test cases against their ground-truth phases.
    # When resuming, training writes to a fresh sibling directory, so evaluate the
    # model from the directory training actually used, not the original output_dir.
    model_directory = train_result["output_directory"]
    infer_workflow = WorkflowInferPhysicsNeMo(
        model_directory=model_directory, log_level=log_level
    )
    # The targets are displacements from each case's reference surface, so the
    # raw predictions are turned back into surfaces by the displacement decoder.
    displacement_workflow = WorkflowInferMovement(infer_workflow, log_level=log_level)

    tutorial_results: dict[str, Any] = {
        "model_directory": model_directory,
        "cases": {},
    }
    for case_id in test_cases:
        logger.info("Evaluating held-out case %s", case_id)
        tutorial_results["cases"][case_id] = displacement_workflow.process(
            manifests[case_id],
            output_directory=model_directory / "eval_mgn" / case_id,
        )

    # Testing: render the first predicted surface of the last held-out case and
    # the RMSE-colored reference surface beside it.
    tt = TestTools(
        class_name=class_name,
        results_dir=model_directory,
        baselines_dir=tutorials_dir.parent / "tests" / "baselines" / class_name,
        log_level=log_level,
    )
    last_case = tutorial_results["cases"][test_cases[-1]]
    tutorial_results["screenshots"] = [
        tt.save_screenshot_mesh(
            cast(pv.DataSet, pv.read(str(last_case["predicted_surfaces"][0]))),
            "predicted_surface.png",
            camera_position="iso",
            color="limegreen",
        ),
        tt.save_screenshot_mesh(
            cast(pv.DataSet, pv.read(str(last_case["rmse_surface"]))),
            "rmse_surface.png",
            camera_position="iso",
            color="orange",
        ),
    ]
