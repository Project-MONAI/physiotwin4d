"""Per-structure scoring of an inferred motion model.

The metric definitions are checked against volumes whose overlap, volume and
separation are known by construction, then the whole workflow is run once on a
tiny synthetic case so the report and the CSV are exercised end to end --- that
is where a mis-indexed stage or a missing provenance field would hide.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import itk
import numpy as np
import pytest
import pyvista as pv

from physiotwin4d import WorkflowEvaluateMovement

_SPACING_MM = 1.0
_GRID_SIZE = 40
_RADIUS_MM = 10.0


def _sphere() -> pv.PolyData:
    """Small sphere shared by the template, reference and phase meshes."""
    return pv.Sphere(radius=_RADIUS_MM, theta_resolution=8, phi_resolution=8)


def _ball_labelmap(radius_mm: float, label: int = 1) -> itk.Image:
    """A centered ball of ``label`` on a grid whose origin puts it at the middle."""
    axis = (np.arange(_GRID_SIZE) - (_GRID_SIZE - 1) / 2.0) * _SPACING_MM
    z, y, x = np.meshgrid(axis, axis, axis, indexing="ij")
    array = np.where(x**2 + y**2 + z**2 <= radius_mm**2, label, 0).astype(np.uint8)
    image = itk.GetImageFromArray(array)
    image.SetSpacing([_SPACING_MM] * 3)
    image.SetOrigin([-(_GRID_SIZE - 1) / 2.0 * _SPACING_MM] * 3)
    return image


def test_dice_of_a_half_overlap() -> None:
    """Dice is twice the intersection over the summed sizes."""
    truth = np.array([1, 1, 1, 1, 0, 0], dtype=np.uint8)
    predicted = np.array([1, 1, 0, 0, 1, 1], dtype=np.uint8)

    assert WorkflowEvaluateMovement.dice(truth, predicted, 1) == 0.5


def test_dice_is_not_a_number_when_neither_volume_has_the_label() -> None:
    """A label neither volume contains has no overlap to report, not a zero one."""
    empty = np.zeros(8, dtype=np.uint8)

    assert np.isnan(WorkflowEvaluateMovement.dice(empty, empty, 3))


def test_volume_counts_voxels_in_cubic_millimeters() -> None:
    """Volume is the label's voxel count times the voxel volume."""
    labels = np.array([2, 2, 2, 0, 1], dtype=np.uint8)

    assert WorkflowEvaluateMovement.volume_mm3(labels, 2, 0.5) == 1.5


def test_surface_rmse_of_two_concentric_spheres_is_their_radius_gap() -> None:
    """A surface offset by 1 mm everywhere scores 1 mm, from either direction."""
    inner = pv.Sphere(radius=10.0, theta_resolution=60, phi_resolution=60)
    outer = pv.Sphere(radius=11.0, theta_resolution=60, phi_resolution=60)

    assert WorkflowEvaluateMovement.surface_rmse_mm(inner, outer) == pytest.approx(
        1.0, abs=0.05
    )


def _trained_model_directory(tmp_path: Path) -> Path:
    """Train a two-epoch MeshGraphNet on two synthetic subjects."""
    from physiotwin4d import TrainPhysicsNeMoMGN, WorkflowTrainPhysicsNeMo

    template = _sphere()
    template_dir = tmp_path / "template"
    template_dir.mkdir(parents=True, exist_ok=True)
    template_file = template_dir / "pca_mean_surface.vtp"
    template.save(str(template_file))
    (template_dir / "pca_model.json").write_text(
        json.dumps(
            {
                "mean": np.asarray(template.points, dtype=np.float64).ravel().tolist(),
                "components": np.zeros((2, template.n_points * 3)).tolist(),
            }
        ),
        encoding="utf-8",
    )

    manifests = []
    for index, offset in enumerate((0.5, -0.5)):
        subject_dir = tmp_path / f"subject_{index:02d}"
        subject_dir.mkdir(parents=True, exist_ok=True)
        _sphere().save(str(subject_dir / "reference.vtp"))
        (subject_dir / "coefficients.json").write_text(
            json.dumps([offset, -offset]), encoding="utf-8"
        )
        phases = []
        for phase_index, stage in enumerate((0.0, 1.0)):
            phase_mesh = _sphere()
            phase_mesh.point_data["displacement"] = np.full(
                (phase_mesh.n_points, 3), offset * stage, dtype=np.float32
            )
            phase_file = subject_dir / f"phase_{phase_index}.vtp"
            phase_mesh.save(str(phase_file))
            phases.append({"mesh": str(phase_file), "stage": stage})
        manifest_file = subject_dir / "manifest.json"
        manifest_file.write_text(
            json.dumps(
                {
                    "subject_id": f"subject_{index:02d}",
                    "reference_mesh": str(subject_dir / "reference.vtp"),
                    "pca_coefficients": str(subject_dir / "coefficients.json"),
                    "target_array": "displacement",
                    "phases": phases,
                }
            ),
            encoding="utf-8",
        )
        manifests.append(manifest_file)

    method = TrainPhysicsNeMoMGN()
    method.set_epochs(2)
    method.set_batch_size(1)
    method.set_processor_size(1)
    method.set_hidden_dim(8)
    method.set_num_layers(1)

    model_directory = tmp_path / "weights"
    WorkflowTrainPhysicsNeMo(
        train_manifests=manifests,
        val_manifests=[],
        pca_mean_mesh=template_file,
        output_directory=model_directory,
        training_method=method,
    ).process()
    return model_directory


def test_every_stage_and_structure_reaches_the_report(tmp_path: Path) -> None:
    """One row per stage and structure, with the run's provenance on it."""
    pytest.importorskip("torch")
    pytest.importorskip("physicsnemo")
    pytest.importorskip("torch_geometric")

    from physiotwin4d import WorkflowInferMovement, WorkflowInferPhysicsNeMo

    model_directory = _trained_model_directory(tmp_path)

    case_dir = tmp_path / "case"
    case_dir.mkdir(parents=True, exist_ok=True)
    reference_mesh_file = case_dir / "reference.vtp"
    _sphere().save(str(reference_mesh_file))
    shape_parameters = case_dir / "coefficients.json"
    shape_parameters.write_text(json.dumps([0.25, -0.25]), encoding="utf-8")

    reference_labelmap = _ball_labelmap(_RADIUS_MM)
    ground_truth = {0.0: _ball_labelmap(_RADIUS_MM), 1.0: _ball_labelmap(9.0)}

    output_directory = tmp_path / "evaluation"
    workflow = WorkflowEvaluateMovement(
        movement_workflow=WorkflowInferMovement(
            WorkflowInferPhysicsNeMo(model_directory=model_directory)
        ),
        label_names={1: "ball"},
    )
    result = workflow.process(
        case_id="synthetic_case",
        shape_parameters=shape_parameters,
        reference_mesh=reference_mesh_file,
        reference_labelmap=reference_labelmap,
        ground_truth_labelmaps=ground_truth,
        output_directory=output_directory,
        smoothing_sigma_mm=2.0,
        evaluation_spacing_mm=_SPACING_MM,
    )

    # Two stages, one structure.
    assert len(result["rows"]) == 2
    assert len(result["predicted_surfaces"]) == 2
    assert len(result["warped_labelmaps"]) == 2

    with Path(result["csv_file"]).open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert {
        "case_id",
        "stage",
        "label_id",
        "label_name",
        "dice",
        "volume_truth_mm3",
        "volume_predicted_mm3",
        "volume_difference_mm3",
        "volume_difference_percent",
        "surface_rmse_mm",
        "shape_parameters_file",
        "pca_c01",
        "pca_c02",
        "network_weights_file",
        "network_weights_created",
        "network_weights_modified",
        "network_epoch",
    } <= set(rows[0])
    assert all(0.0 <= float(row["dice"]) <= 1.0 for row in rows)

    report = Path(result["report_file"]).read_text(encoding="utf-8")
    assert "synthetic_case" in report
    assert str(model_directory) in report
    assert "ball" in report
    assert Path(result["volume_plot_file"]).name in report
    assert Path(result["volume_plot_file"]).exists()
