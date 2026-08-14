"""Workflow for predicting mesh-stage targets with a trained PhysicsNeMo model.

The workflow owns everything around the network: the checkpoint and its
normalization statistics, the shared PCA template mesh, manifests, and the
writing of predicted meshes and error statistics. The network itself is supplied
as an inference method (:class:`physiotwin4d.InferPhysicsNeMoMGN` or
:class:`physiotwin4d.InferPhysicsNeMoMLP`).

Predictions are the targets the model was trained on, whatever those are — the
manifest's ``target_array`` values at each template point. For the common case
where those targets are displacements from the subject's reference mesh, wrap
this workflow in :class:`physiotwin4d.WorkflowInferMovement` to get
reconstructed surfaces, mm error statistics and deformation fields.

PhysicsNeMo (and, for the MGN, PyTorch Geometric) are optional dependencies,
imported lazily so ``import physiotwin4d`` works without them.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import pyvista as pv

from . import physicsnemo_tools as pnt
from .infer_physicsnemo_base import InferPhysicsNeMoBase
from .infer_physicsnemo_mgn import InferPhysicsNeMoMGN
from .physiotwin4d_base import PhysioTwin4DBase


class WorkflowInferPhysicsNeMo(PhysioTwin4DBase):
    """Predict per-point targets for a subject at requested stages.

    The network is supplied as an inference method — pass a
    :class:`physiotwin4d.InferPhysicsNeMoMGN` or
    :class:`physiotwin4d.InferPhysicsNeMoMLP` instance as ``inference_method``;
    a default MeshGraphNet method is used when none is given.
    """

    def __init__(
        self,
        model_directory: Path,
        inference_method: Optional[InferPhysicsNeMoBase] = None,
        epoch: Optional[int] = None,
        log_level: int | str = logging.INFO,
    ) -> None:
        """Load a trained model and its normalization statistics.

        Args:
            model_directory: Directory written by
                :class:`physiotwin4d.WorkflowTrainPhysicsNeMo` (holds
                ``<tag>_stage_model.pt``, ``pca_mean_template.vtp`` or ``.vtu``
                and, for the MGN, the shared graph tensors).
            inference_method: Inference method carrying the network. Defaults to
                a new :class:`physiotwin4d.InferPhysicsNeMoMGN`.
            epoch: Optional intermittent-checkpoint epoch to load
                (``<tag>_stage_model_epoch_#####.pt``). When ``None`` the final
                weights stored in the main checkpoint are used.
            log_level: Logging level. Default: ``logging.INFO``.

        Raises:
            FileNotFoundError: If the model checkpoint or the template mesh is
                missing.
            TypeError: If ``inference_method`` is neither None nor an
                InferPhysicsNeMoBase instance.
        """
        super().__init__(class_name=self.__class__.__name__, log_level=log_level)
        import torch

        if inference_method is not None and not isinstance(
            inference_method, InferPhysicsNeMoBase
        ):
            raise TypeError(
                "inference_method must be an InferPhysicsNeMoBase instance, got "
                f"{type(inference_method).__name__}"
            )
        self.inference_method = (
            inference_method
            if inference_method is not None
            else InferPhysicsNeMoMGN(log_level=log_level)
        )

        self.model_directory = Path(model_directory)
        tag = self.inference_method.model_tag
        if epoch is not None:
            checkpoint_file = (
                self.model_directory / f"{tag}_stage_model_epoch_{epoch:05d}.pt"
            )
        else:
            checkpoint_file = self.model_directory / f"{tag}_stage_model.pt"
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_file}")
        self.epoch = epoch
        self.checkpoint_file = checkpoint_file

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_info("Loading %s model from %s", tag.upper(), checkpoint_file)
        meta = torch.load(str(checkpoint_file), map_location="cpu", weights_only=True)
        self._meta = meta

        # Normalization statistics and target description.
        self.coordinate_mean = np.array(meta["coordinate_mean"], dtype=np.float32)
        self.coordinate_scale = np.array(meta["coordinate_scale"], dtype=np.float32)
        self.pca_mean = np.array(meta["pca_mean"], dtype=np.float32)
        self.pca_scale = np.array(meta["pca_scale"], dtype=np.float32)
        self.target_scale = float(meta["target_scale"])
        self.n_target = int(meta["n_target"])
        self.target_array = str(meta.get("target_array", "target"))

        # Shared template mesh (node coordinates + output topology).
        self._template_mesh = self._load_template_mesh()
        self._template_coords = np.asarray(self._template_mesh.points, dtype=np.float32)
        self._mean_coords_norm = (
            self._template_coords - self.coordinate_mean
        ) / self.coordinate_scale

        # Build the network, load its weights and attach it to the method.
        model = self.inference_method.build_model(meta).to(self._device)
        self.inference_method.load_artifacts(
            self.model_directory, len(self._template_coords), self._device
        )
        state = self._load_weights(epoch)
        model.load_state_dict(pnt.strip_compile_prefix(state))
        model.eval()
        self.inference_method.set_model(model, self._device)

        # Optional PCA reconstruction assets (manifest-free inference).
        self._pca_model: Optional[dict] = None
        self._pca_mean_dataset: Optional[pv.DataSet] = None

    # ─────────────────────────── Shared assets ─────────────────────────────
    @property
    def template_mesh(self) -> pv.DataSet:
        """The shared PCA template mesh defining node order and topology."""
        return self._template_mesh

    def _load_template_mesh(self) -> pv.DataSet:
        """Read the template mesh the training workflow copied into the model dir."""
        for suffix in (".vtp", ".vtu"):
            candidate = self.model_directory / f"pca_mean_template{suffix}"
            if candidate.exists():
                return cast(pv.DataSet, pv.read(str(candidate)))
        raise FileNotFoundError(
            f"pca_mean_template.vtp/.vtu not found in {self.model_directory}"
        )

    def _load_weights(self, epoch: Optional[int]) -> dict:
        """Return the state dict for the requested epoch (or final weights)."""
        import torch

        if epoch is None:
            return dict(self._meta["model_state_dict"])
        tag = self.inference_method.model_tag
        epoch_file = self.model_directory / f"{tag}_stage_model_epoch_{epoch:05d}.pt"
        if not epoch_file.exists():
            raise FileNotFoundError(f"Epoch checkpoint not found: {epoch_file}")
        ckpt = torch.load(str(epoch_file), map_location="cpu", weights_only=True)
        # Self-describing checkpoints wrap the weights under "model_state_dict";
        # bare/legacy epoch checkpoints are the state dict itself.
        return cast(dict, ckpt.get("model_state_dict", ckpt))

    def load_pca_assets(self) -> tuple[pv.DataSet, dict]:
        """Load (and cache) the PCA template mesh and model for reconstruction."""
        if self._pca_mean_dataset is not None and self._pca_model is not None:
            return self._pca_mean_dataset, self._pca_model

        pca_model_file = self.model_directory / "pca_model.json"
        if not pca_model_file.exists():
            raise FileNotFoundError(
                f"pca_model.json not found in {self.model_directory}; it is "
                "required for manifest-free reconstruction. Re-run training with a "
                "pca_mean_mesh whose directory contains pca_model.json."
            )
        # The PCA template mesh (volume) was copied next to pca_model.json.
        mesh_candidates = [
            p
            for p in self.model_directory.glob("*")
            if p.suffix in (".vtu", ".vtk", ".vtp")
            and not p.name.startswith("pca_mean_template")
        ]
        pca_model = json.loads(pca_model_file.read_text(encoding="utf-8"))
        expected = int(np.asarray(pca_model["components"]).shape[1]) // 3
        mesh: Optional[pv.DataSet] = None
        for candidate in mesh_candidates:
            dataset = pv.read(str(candidate))
            if dataset.n_points == expected:
                mesh = dataset
                break
        if mesh is None:
            raise FileNotFoundError(
                f"No PCA template mesh with {expected} points found in "
                f"{self.model_directory} to match pca_model.json."
            )
        self._pca_mean_dataset = mesh
        self._pca_model = pca_model
        return mesh, pca_model

    def reference_points_from_coefficients(self, pca_coeffs: np.ndarray) -> np.ndarray:
        """Reconstruct a subject's reference points in the template's domain.

        The PCA model may be volumetric while the model was trained on the
        template's surface (``use_template_surface``), so the reconstruction is
        surface-extracted when its point count does not match the template's.
        """
        mean_mesh, pca_model = self.load_pca_assets()
        points = pnt.reconstruct_reference_points(mean_mesh, pca_model, pca_coeffs)
        if points.shape[0] == self._template_mesh.n_points:
            return points
        deformed = mean_mesh.copy(deep=True)
        deformed.points = points
        surface = deformed.extract_surface(algorithm="dataset_surface")
        if surface.n_points != self._template_mesh.n_points:
            raise ValueError(
                f"PCA reconstruction yields {points.shape[0]} points "
                f"({surface.n_points} on its surface), but the template has "
                f"{self._template_mesh.n_points}."
            )
        return np.asarray(surface.points, dtype=np.float32)

    # ─────────────────────────── Core predictor ────────────────────────────
    def predict(self, pca_coeffs: np.ndarray, stage: float) -> np.ndarray:
        """Predict ``(n_points, n_target)`` targets for a subject at a stage."""
        pca_norm = (pca_coeffs - self.pca_mean) / self.pca_scale
        node_feats = pnt.build_node_features(self._mean_coords_norm, pca_norm, stage)
        return self.inference_method.predict(node_feats) * self.target_scale

    def predicted_mesh(self, targets: np.ndarray) -> pv.DataSet:
        """Return a template copy carrying ``targets`` as its target array."""
        mesh = self._template_mesh.copy(deep=True)
        mesh.point_data[self.target_array] = targets
        return mesh

    # ─────────────────────────── Public API ────────────────────────────────
    def process(
        self,
        subject_manifest: Path,
        stages: Optional[list[float]] = None,
        output_directory: Optional[Path] = None,
    ) -> dict[str, Any]:
        """Predict a subject's targets from a manifest.

        When ``stages`` is ``None`` every phase in the manifest is predicted and,
        because the stored target array is available, per-phase error statistics
        are computed and written. When ``stages`` is given those arbitrary stages
        are predicted without comparison.

        Args:
            subject_manifest: Path to the subject manifest JSON.
            stages: Optional list of stages to predict.
            output_directory: Output directory; defaults to
                ``<model_directory>/<subject_id>``.

        Returns:
            Dict with ``subject_id``, ``predicted_meshes`` (paths) and, in the
            phase mode, ``statistics`` and ``statistics_file``.
        """
        manifest = pnt.parse_manifest(subject_manifest)
        pca_coeffs = pnt.load_pca_coefficients(manifest.pca_coefficients)

        out_dir = (
            Path(output_directory)
            if output_directory is not None
            else self.model_directory / manifest.subject_id
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = self.inference_method.model_tag
        self.log_section("INFER %s [%s]", tag.upper(), manifest.subject_id)

        suffix = ".vtp" if isinstance(self._template_mesh, pv.PolyData) else ".vtu"
        sid = manifest.subject_id
        meshes: list[Path] = []
        stats: list[dict] = []

        requested = stages if stages is not None else [p.stage for p in manifest.phases]
        for index, stage in enumerate(requested):
            predicted = self.predict(pca_coeffs, stage)
            path = out_dir / f"{sid}_pred_s{int(stage * 100):03d}{suffix}"
            self.predicted_mesh(predicted).save(str(path))
            meshes.append(path)

            if stages is not None:
                self.log_info("stage %.3f -> %s", stage, path.name)
                continue

            actual = pnt.load_target_array(
                manifest.phases[index].mesh, manifest.target_array
            )
            if actual.shape != predicted.shape:
                raise ValueError(
                    f"Stored '{manifest.target_array}' targets in "
                    f"{manifest.phases[index].mesh} have shape {actual.shape}, "
                    f"but the model predicts {predicted.shape}."
                )
            stats.append(self._error_row(sid, stage, predicted, actual))
            self.log_info(
                "stage %.3f: mean abs error=%.4f  max abs error=%.4f",
                stage,
                stats[-1]["mean_abs_error"],
                stats[-1]["max_abs_error"],
            )

        result: dict[str, Any] = {"subject_id": sid, "predicted_meshes": meshes}
        if stages is None:
            stats_file = out_dir / "statistics_per_phase.csv"
            with stats_file.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(stats[0].keys()))
                writer.writeheader()
                writer.writerows(stats)
            result["statistics"] = stats
            result["statistics_file"] = stats_file
        return result

    @staticmethod
    def _error_row(
        subject_id: str, stage: float, pred: np.ndarray, actual: np.ndarray
    ) -> dict:
        """Per-phase error statistics between predicted and stored targets."""
        errors = np.abs(pred - actual)
        return {
            "subject_id": subject_id,
            "stage": stage,
            "n_points": int(pred.shape[0]),
            "n_target": int(pred.shape[1]),
            "mean_abs_error": float(errors.mean()),
            "median_abs_error": float(np.median(errors)),
            "max_abs_error": float(errors.max()),
            "rms_error": float(np.sqrt(np.mean((pred - actual) ** 2))),
        }
