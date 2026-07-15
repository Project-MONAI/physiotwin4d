"""
Tutorial 9c (MGN): Train a PhysicsNeMo MeshGraphNet for cardiac mesh stages.

Second stage of the cardiac 4D deep-learning pipeline (Tutorials 08cd -> 09c/09d
-> 10c/10d).  This tutorial is a thin driver over the reusable
:class:`physiotwin4d.WorkflowTrainPhysicsNeMoMGN` workflow: it discovers the
per-time-point SSM surfaces produced by Tutorial 8cd
(``tutorial_08cd_byod_fit_model_to_patients.py``), writes one JSON manifest per
subject, splits the subjects into train / validation / held-out test, trains the
MeshGraphNet, and evaluates the held-out test subjects with
:class:`physiotwin4d.WorkflowInferPhysicsNeMoMGN`.  The companion MLP tutorial is
``tutorial_09d_byod_train_physicsnemo_mlp.py``.

Why a GNN?
----------
The SSM mesh has a fixed topology across all subjects and cardiac tissue is a
continuum: adjacent vertices co-vary smoothly.  MeshGraphNet encodes that prior
directly by passing messages along mesh edges, giving an explicit
continuum-deformation inductive bias the MLP must infer from coordinates alone.

Node features (per vertex):   [mean_shape_x, mean_shape_y, mean_shape_z, pca_c1 ... pca_cN, stage]
Edge features (per edge):     [rel_x, rel_y, rel_z, distance]   (from the mean shape)
Output (per vertex):          [dx, dy, dz]  (displacement in mm)

Bring Your Own Data
-------------------
The path constants below point at a local ``D:/PhysioTwin4D/`` layout produced by
Tutorial 8cd, not at the repository ``data/`` directory.  Edit them to match your
own data location.  Run Tutorial 8cd first.

Extra Install Required
----------------------
PhysicsNeMo and PyTorch Geometric must be installed::

    pip install "physiotwin4d[physicsnemo]"
"""

# %%
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from physiotwin4d import WorkflowInferPhysicsNeMoMGN, WorkflowTrainPhysicsNeMoMGN


def _gating_stage_from_filename(mesh_file: Path) -> float:
    """Extract the normalized cardiac stage [0, 1] from a ``g0TT`` filename stem."""
    for part in mesh_file.stem.split("_"):
        if part.startswith("g") and part[1:].isdigit():
            return int(part[1:]) / 100.0
    raise ValueError(f"Cannot parse gating percentage from filename: {mesh_file}")


def _write_subject_manifest(subject_dir: Path, manifests_dir: Path) -> Optional[Path]:
    """Write a per-subject manifest JSON; return its path (or None if incomplete).

    A subject needs a reference SSM surface, a PCA coefficient file, and at least
    two gated-phase surfaces.  Stages are parsed from the ``g0TT`` phase filenames
    and written explicitly into the manifest (the workflow itself never parses
    filenames).
    """
    sid = subject_dir.name
    ref_file = subject_dir / f"{sid}_ssm_surface.vtp"
    pca_file = subject_dir / f"{sid}_ssm_pca_coefficients.json"
    phase_files = sorted(subject_dir.glob(f"{sid}_g0*_ssm_surface.vtp"))
    if not ref_file.exists() or not pca_file.exists() or len(phase_files) < 2:
        return None

    manifest = {
        "subject_id": sid,
        "reference_surface": str(ref_file),
        "pca_coefficients": str(pca_file),
        "phases": [
            {"surface": str(pf), "stage": _gating_stage_from_filename(pf)}
            for pf in phase_files
        ],
    }
    manifests_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifests_dir / f"{sid}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


if __name__ == "__main__":
    # %%
    TUTORIALS_DIR = Path(__file__).resolve().parent
    FITTED_MESHES_DIR = Path("D:/PhysioTwin4D/duke_data/fitted_kcl_meshes")
    PCA_MEAN_VTU = Path("D:/PhysioTwin4D/kcl-heart-pca/pca-vol-kcl/pca_mean.vtu")
    OUTPUT_DIR = TUTORIALS_DIR / "output_mgn"
    MANIFESTS_DIR = TUTORIALS_DIR / "manifests_mgn"

    EPOCHS = 1500
    BATCH_SIZE_GRAPHS = 4  # mini-batch measured in (subject, phase) graphs
    LEARNING_RATE = 1.0e-3
    PROCESSOR_SIZE = 3  # message-passing hops
    HIDDEN_DIM = 128
    NUM_LAYERS = 2  # MLP layers inside each encoder / processor / decoder block

    # Explicit held-out splits; every other discovered subject is used for training.
    TEST_SUBJECTS = ["pm0028"]
    VAL_SUBJECTS = ["pm0027"]
    LOG_LEVEL = logging.INFO

    def run_tutorial() -> dict[str, Any]:
        """Discover subjects, train a MeshGraphNet, and evaluate the test split."""
        logging.basicConfig(level=LOG_LEVEL)

        # Build one manifest per valid subject and partition into splits.
        manifests: dict[str, Path] = {}
        for subject_dir in sorted(FITTED_MESHES_DIR.glob("pm????")):
            manifest_path = _write_subject_manifest(subject_dir, MANIFESTS_DIR)
            if manifest_path is not None:
                manifests[subject_dir.name] = manifest_path

        if len(manifests) < 3:
            raise RuntimeError(
                f"Found only {len(manifests)} valid subject(s); need at least 3 "
                "for a train / val / test split."
            )

        unknown = [s for s in TEST_SUBJECTS + VAL_SUBJECTS if s not in manifests]
        if unknown:
            raise ValueError(f"Split subjects not found: {unknown}")

        test_manifests = [manifests[s] for s in TEST_SUBJECTS]
        val_manifests = [manifests[s] for s in VAL_SUBJECTS]
        train_manifests = [
            p
            for sid, p in manifests.items()
            if sid not in TEST_SUBJECTS and sid not in VAL_SUBJECTS
        ]
        logging.info(
            "Subject split - train: %d, val: %d, test: %d",
            len(train_manifests),
            len(val_manifests),
            len(test_manifests),
        )

        # Train the MeshGraphNet.
        trainer = WorkflowTrainPhysicsNeMoMGN(
            train_manifests=train_manifests,
            val_manifests=val_manifests,
            pca_mean_mesh=PCA_MEAN_VTU,
            output_directory=OUTPUT_DIR,
            log_level=LOG_LEVEL,
        )
        trainer.set_epochs(EPOCHS)
        trainer.set_batch_size(BATCH_SIZE_GRAPHS)
        trainer.set_learning_rate(LEARNING_RATE)
        trainer.set_processor_size(PROCESSOR_SIZE)
        trainer.set_hidden_dim(HIDDEN_DIM)
        trainer.set_num_layers(NUM_LAYERS)
        train_result = trainer.process()

        # Evaluate held-out test subjects against their ground-truth phases.
        infer = WorkflowInferPhysicsNeMoMGN(
            model_directory=OUTPUT_DIR, log_level=LOG_LEVEL
        )
        eval_outputs: dict[str, Any] = {}
        for sid in TEST_SUBJECTS:
            eval_outputs[sid] = infer.predict(
                manifests[sid], output_directory=OUTPUT_DIR / "eval_mgn" / sid
            )

        return {"training": train_result, "evaluation": eval_outputs}

    # %%
    tutorial_results = run_tutorial()
