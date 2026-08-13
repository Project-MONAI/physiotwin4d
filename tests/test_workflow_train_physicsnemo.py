"""Synthetic training run covering what a checkpoint needs beside it.

A long training run is evaluated from its intermittent checkpoints while it is
still going, so everything inference reads from the model directory -- the
template mesh, the shared graph tensors, the PCA assets and the metadata -- has
to be on disk by the time the first of those checkpoints is written, not only
after the last epoch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import pyvista as pv

pytest.importorskip("torch")
pytest.importorskip("physicsnemo")
pytest.importorskip("torch_geometric")

from physiotwin4d import (  # noqa: E402
    TrainPhysicsNeMoMGN,
    WorkflowInferPhysicsNeMo,
    WorkflowTrainPhysicsNeMo,
)

_TARGET_ARRAY = "displacement"
_STAGES = (0.0, 1.0)


def _sphere() -> pv.PolyData:
    """Small sphere shared by the template, reference and phase meshes."""
    return pv.Sphere(radius=10.0, theta_resolution=8, phi_resolution=8)


def _write_subject(subject_id: str, directory: Path, offset: float) -> Path:
    """Write one subject's reference mesh, phase targets and manifest."""
    directory.mkdir(parents=True, exist_ok=True)
    reference = _sphere()
    reference.save(str(directory / "reference.vtp"))
    (directory / "coefficients.json").write_text(
        json.dumps([offset, -offset]), encoding="utf-8"
    )

    phases = []
    for index, stage in enumerate(_STAGES):
        phase_mesh = _sphere()
        phase_mesh.point_data[_TARGET_ARRAY] = np.full(
            (phase_mesh.n_points, 3), offset + stage, dtype=np.float32
        )
        phase_file = directory / f"phase_{index}.vtp"
        phase_mesh.save(str(phase_file))
        phases.append({"mesh": str(phase_file), "stage": stage})

    manifest_path = directory / f"{subject_id}_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "subject_id": subject_id,
                "reference_mesh": str(directory / "reference.vtp"),
                "pca_coefficients": str(directory / "coefficients.json"),
                "target_array": _TARGET_ARRAY,
                "phases": phases,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path


def _write_template(directory: Path) -> Path:
    """Write the PCA template mesh and the model JSON that sits beside it."""
    directory.mkdir(parents=True, exist_ok=True)
    template = _sphere()
    template_file = directory / "pca_mean_surface.vtp"
    template.save(str(template_file))
    (directory / "pca_model.json").write_text(
        json.dumps(
            {
                "mean": np.asarray(template.points, dtype=np.float64).ravel().tolist(),
                "components": np.zeros((2, template.n_points * 3)).tolist(),
            }
        ),
        encoding="utf-8",
    )
    return template_file


class _RecordingMGN(TrainPhysicsNeMoMGN):
    """MeshGraphNet method that lists the model directory at each checkpoint."""

    def __init__(self, model_directory: Path) -> None:
        super().__init__()
        self.model_directory = model_directory
        self.snapshots: list[set[str]] = []

    def build_checkpoint(self, model: Any, stats: dict) -> dict[str, Any]:
        self.snapshots.append({path.name for path in self.model_directory.iterdir()})
        return cast("dict[str, Any]", super().build_checkpoint(model, stats))


def _train(tmp_path: Path) -> tuple[Path, _RecordingMGN]:
    """Run two epochs over two synthetic subjects; return the model directory."""
    template_file = _write_template(tmp_path / "template")
    manifests = [
        _write_subject(
            f"subject_{index:02d}", tmp_path / f"subject_{index:02d}", offset
        )
        for index, offset in enumerate((0.5, -0.5))
    ]
    model_directory = tmp_path / "weights"

    method = _RecordingMGN(model_directory)
    method.set_epochs(2)
    method.set_batch_size(1)
    method.set_processor_size(1)
    method.set_hidden_dim(8)
    method.set_num_layers(1)
    # Every epoch, so the first intermittent checkpoint lands in epoch one.
    method.rmse_log_interval = 1

    workflow = WorkflowTrainPhysicsNeMo(
        train_manifests=manifests,
        val_manifests=[],
        pca_mean_mesh=template_file,
        output_directory=model_directory,
        training_method=method,
    )
    workflow.process()
    return model_directory, method


def test_first_checkpoint_has_its_companions(tmp_path: Path) -> None:
    """Inference's inputs are on disk before the first checkpoint is written."""
    _, method = _train(tmp_path)

    assert len(method.snapshots) > 1, "expected intermittent checkpoints, not just one"
    assert {
        "pca_mean_template.vtp",
        "pca_mean_surface.vtp",
        "pca_model.json",
        "shared_edge_index.pt",
        "shared_edge_features.pt",
        "mgn_stage_model_metadata.json",
    } <= method.snapshots[0]


def test_an_intermittent_checkpoint_can_be_inferred_from(tmp_path: Path) -> None:
    """The model directory loads at an epoch, not only at the final weights."""
    model_directory, _ = _train(tmp_path)

    infer = WorkflowInferPhysicsNeMo(model_directory=model_directory, epoch=1)
    targets = infer.predict(np.array([0.5, -0.5], dtype=np.float32), stage=0.5)

    assert targets.shape == (infer.template_mesh.n_points, 3)
    assert np.all(np.isfinite(targets))
