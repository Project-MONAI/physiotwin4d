"""Workflows for inferring cardiac mesh stages with trained PhysicsNeMo models.

A shared base class :class:`WorkflowInferPhysicsNeMo` loads a model produced by
:mod:`physiotwin4d.workflow_train_physicsnemo` and predicts SSM surface
displacements at requested stages; the concrete
:class:`WorkflowInferPhysicsNeMoMGN` and :class:`WorkflowInferPhysicsNeMoMLP`
subclasses supply the network-specific seams.

Three prediction entry points share one core displacement predictor:

- :meth:`WorkflowInferPhysicsNeMo.predict` — manifest-driven prediction of every
  gated phase (with error statistics when the phase ground-truth surface exists)
  or of arbitrary requested stages.
- :meth:`WorkflowInferPhysicsNeMo.predict_single` — manifest-free single-subject
  prediction from a PCA shape-parameter JSON and a target stage, with an
  optional ground-truth surface for error reporting.
- :meth:`WorkflowInferPhysicsNeMo.create_deformation_field` — rasterize the
  inferred deformation and reference-surface normals onto a caller-supplied
  reference image grid.

PhysicsNeMo (and, for the MGN, PyTorch Geometric) are optional dependencies,
imported lazily so ``import physiotwin4d`` works without them.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, cast

import itk
import numpy as np
import pyvista as pv

from . import physicsnemo_tools as pnt
from .physiotwin4d_base import PhysioTwin4DBase

if TYPE_CHECKING:  # typed for mypy; imported lazily at runtime
    import torch


class WorkflowInferPhysicsNeMo(PhysioTwin4DBase):
    """Base class for inferring cardiac mesh stages from a trained model.

    Not instantiated directly — use :class:`WorkflowInferPhysicsNeMoMGN` or
    :class:`WorkflowInferPhysicsNeMoMLP`. Subclasses implement the seams
    :meth:`_build_model`, :meth:`_load_extra_artifacts` and
    :meth:`_network_predict`, and set the class attribute ``_model_tag``.
    """

    _model_tag: str = "base"

    def __init__(
        self,
        model_directory: Path,
        epoch: Optional[int] = None,
        log_level: int | str = logging.INFO,
    ) -> None:
        """Load a trained model and its normalization statistics.

        Args:
            model_directory: Directory written by the matching training
                workflow (holds ``<tag>_stage_model.pt``, ``pca_mean_surface.vtp``
                and, for the MGN, the shared graph tensors).
            epoch: Optional intermittent-checkpoint epoch to load
                (``<tag>_stage_model_epoch_#####.pt``). When ``None`` the final
                weights stored in the main checkpoint are used.
            log_level: Logging level. Default: ``logging.INFO``.

        Raises:
            FileNotFoundError: If the model checkpoint is missing.
        """
        super().__init__(class_name=self.__class__.__name__, log_level=log_level)
        import torch

        self.model_directory = Path(model_directory)
        checkpoint_file = self.model_directory / f"{self._model_tag}_stage_model.pt"
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_file}")

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_info(
            "Loading %s model from %s", self._model_tag.upper(), checkpoint_file
        )
        meta = torch.load(str(checkpoint_file), map_location="cpu", weights_only=True)
        self._meta = meta

        # Normalization statistics.
        self.coordinate_mean = np.array(meta["coordinate_mean"], dtype=np.float32)
        self.coordinate_scale = np.array(meta["coordinate_scale"], dtype=np.float32)
        self.pca_mean = np.array(meta["pca_mean"], dtype=np.float32)
        self.pca_scale = np.array(meta["pca_scale"], dtype=np.float32)
        self.displacement_scale = float(meta["displacement_scale"])

        # Shared mean-shape surface (node coordinates + output topology).
        mean_surface_file = self.model_directory / "pca_mean_surface.vtp"
        if not mean_surface_file.exists():
            raise FileNotFoundError(
                f"pca_mean_surface.vtp not found in {self.model_directory}"
            )
        self._mean_surface: pv.PolyData = cast(
            pv.PolyData, pv.read(str(mean_surface_file))
        )
        self._mean_shape_coords = np.asarray(
            self._mean_surface.points, dtype=np.float32
        )
        self._mean_coords_norm = (
            self._mean_shape_coords - self.coordinate_mean
        ) / self.coordinate_scale

        # Build the network and load weights.
        self._model = self._build_model(meta).to(self._device)
        self._load_extra_artifacts()
        state = self._load_weights(epoch)
        self._model.load_state_dict(pnt.strip_compile_prefix(state))
        self._model.eval()

        # Optional PCA reconstruction assets (manifest-free inference).
        self._pca_model: Optional[dict] = None
        self._pca_mean_dataset: Optional[pv.DataSet] = None

    # ─────────────────────────── Network seams ─────────────────────────────
    def _build_model(self, meta: dict) -> "torch.nn.Module":
        """Rebuild the network from checkpoint metadata. Subclass-specific."""
        raise NotImplementedError

    def _load_extra_artifacts(self) -> None:
        """Load any architecture-specific artifacts (MGN graph tensors)."""
        raise NotImplementedError

    def _network_predict(self, node_feats: np.ndarray) -> np.ndarray:
        """Run the network over all vertices; return ``(n, 3)`` raw output."""
        raise NotImplementedError

    # ─────────────────────────── Core predictor ────────────────────────────
    def _load_weights(self, epoch: Optional[int]) -> dict:
        """Return the state dict for the requested epoch (or final weights)."""
        import torch

        if epoch is None:
            return dict(self._meta["model_state_dict"])
        epoch_file = (
            self.model_directory / f"{self._model_tag}_stage_model_epoch_{epoch:05d}.pt"
        )
        if not epoch_file.exists():
            raise FileNotFoundError(f"Epoch checkpoint not found: {epoch_file}")
        ckpt = torch.load(str(epoch_file), map_location="cpu", weights_only=True)
        # Self-describing checkpoints wrap the weights under "model_state_dict";
        # bare/legacy epoch checkpoints are the state dict itself.
        return cast(dict, ckpt.get("model_state_dict", ckpt))

    def _predict_displacements(
        self, pca_coeffs: np.ndarray, stage: float
    ) -> np.ndarray:
        """Predict per-vertex displacements (mm) for a subject at a stage."""
        pca_norm = (pca_coeffs - self.pca_mean) / self.pca_scale
        node_feats = pnt.build_node_features(self._mean_coords_norm, pca_norm, stage)
        return self._network_predict(node_feats) * self.displacement_scale

    def _load_pca_assets(self) -> tuple[pv.DataSet, dict]:
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
            if p.suffix in (".vtu", ".vtk", ".vtp") and p.name != "pca_mean_surface.vtp"
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

    # ─────────────────────────── Public API ────────────────────────────────
    def predict(
        self,
        subject_manifest: Path,
        stages: Optional[list[float]] = None,
        output_directory: Optional[Path] = None,
    ) -> dict[str, Any]:
        """Predict a subject's surfaces from a manifest.

        When ``stages`` is ``None`` every gated phase in the manifest is
        predicted and, because the phase ground-truth surface is available,
        per-phase and per-point error statistics are computed and written. When
        ``stages`` is given those arbitrary stages are predicted without error
        comparison.

        Args:
            subject_manifest: Path to the subject manifest JSON.
            stages: Optional list of RR-interval fractions to predict.
            output_directory: Output directory; defaults to
                ``<model_directory>/<subject_id>``.

        Returns:
            Dict with ``subject_id``, ``predicted_surfaces`` (paths), and, in the
            phase mode, ``statistics`` and ``rmse_surface``.
        """
        manifest = pnt.parse_manifest(subject_manifest)
        ref_mesh = cast(pv.PolyData, pv.read(str(manifest.reference_surface)))
        ref_points = np.asarray(ref_mesh.points, dtype=np.float32)
        pca_coeffs = pnt.load_pca_coefficients(manifest.pca_coefficients)

        out_dir = (
            Path(output_directory)
            if output_directory is not None
            else self.model_directory / manifest.subject_id
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        self.log_section("INFER %s [%s]", self._model_tag.upper(), manifest.subject_id)

        if stages is not None:
            return self._predict_arbitrary_stages(
                manifest.subject_id, ref_mesh, ref_points, pca_coeffs, stages, out_dir
            )
        return self._predict_phases(manifest, ref_mesh, ref_points, pca_coeffs, out_dir)

    def _predict_arbitrary_stages(
        self,
        subject_id: str,
        ref_mesh: pv.PolyData,
        ref_points: np.ndarray,
        pca_coeffs: np.ndarray,
        stages: list[float],
        out_dir: Path,
    ) -> dict[str, Any]:
        """Predict requested stages without ground-truth comparison."""
        surfaces: list[Path] = []
        for stage in stages:
            pred_points = ref_points + self._predict_displacements(pca_coeffs, stage)
            pred_mesh = ref_mesh.copy(deep=True)
            pred_mesh.points = pred_points
            path = (
                out_dir / f"{subject_id}_ssm_surface_pred_s{int(stage * 100):03d}.vtp"
            )
            pred_mesh.save(str(path))
            surfaces.append(path)
            self.log_info("stage %.3f -> %s", stage, path.name)
        return {"subject_id": subject_id, "predicted_surfaces": surfaces}

    def _predict_phases(
        self,
        manifest: pnt.SubjectManifest,
        ref_mesh: pv.PolyData,
        ref_points: np.ndarray,
        pca_coeffs: np.ndarray,
        out_dir: Path,
    ) -> dict[str, Any]:
        """Predict all manifest phases and compute error statistics vs ground truth."""
        sid = manifest.subject_id
        n_points = ref_points.shape[0]
        sq_err_sum = np.zeros(n_points, dtype=np.float64)
        stats: list[dict] = []
        surfaces: list[Path] = []

        for phase in manifest.phases:
            pred_points = ref_points + self._predict_displacements(
                pca_coeffs, phase.stage
            )
            pred_mesh = ref_mesh.copy(deep=True)
            pred_mesh.points = pred_points
            tag = f"s{int(phase.stage * 100):03d}"
            path = out_dir / f"{sid}_{tag}_ssm_surface_pred.vtp"
            pred_mesh.save(str(path))
            surfaces.append(path)

            actual = np.asarray(pv.read(str(phase.surface)).points, dtype=np.float32)
            euclidean = np.linalg.norm(pred_points - actual, axis=1)
            sq_err_sum += euclidean.astype(np.float64) ** 2
            stats.append(self._error_row(sid, phase.stage, pred_points, actual))
            self.log_info(
                "stage %.3f: mean=%.3f mm  max=%.3f mm",
                phase.stage,
                stats[-1]["mean_error_mm"],
                stats[-1]["max_error_mm"],
            )

        point_rmse = np.sqrt(sq_err_sum / len(manifest.phases)).astype(np.float32)
        rmse_mesh = ref_mesh.copy(deep=True)
        rmse_mesh.point_data["RMSE_mm"] = point_rmse
        rmse_file = out_dir / f"{sid}_ssm_surface_rmse.vtp"
        rmse_mesh.save(str(rmse_file))

        stats_file = out_dir / "statistics_per_phase.csv"
        with stats_file.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(stats[0].keys()))
            writer.writeheader()
            writer.writerows(stats)

        return {
            "subject_id": sid,
            "predicted_surfaces": surfaces,
            "rmse_surface": rmse_file,
            "statistics": stats,
            "statistics_file": stats_file,
        }

    def predict_single(
        self,
        shape_parameters: Path,
        stage: float,
        ground_truth: Optional[Path] = None,
        output_directory: Optional[Path] = None,
    ) -> dict[str, Any]:
        """Predict one subject at one stage without a manifest.

        The subject reference surface is reconstructed from the PCA shape
        parameters (``P = mean + Σ b_i·std_i·eigenvector_i``) in the SSM/PCA
        frame, so the prediction is self-consistent even with no reference
        surface file.

        Args:
            shape_parameters: JSON file with the subject PCA coefficient vector.
            stage: Target RR-interval fraction to predict.
            ground_truth: Optional surface ``.vtp`` for error reporting.
            output_directory: Output directory; defaults to
                ``<model_directory>/single_prediction``.

        Returns:
            Dict with ``predicted_surface`` (path), ``predicted_points``, and,
            when ``ground_truth`` is supplied, ``statistics``.
        """
        coeffs = pnt.load_pca_coefficients(shape_parameters)
        mean_mesh, pca_model = self._load_pca_assets()
        ref_points = pnt.reconstruct_reference_points(mean_mesh, pca_model, coeffs)
        pred_points = ref_points + self._predict_displacements(coeffs, stage)

        out_dir = (
            Path(output_directory)
            if output_directory is not None
            else self.model_directory / "single_prediction"
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        pred_mesh = self._mean_surface.copy(deep=True)
        pred_mesh.points = pred_points
        stem = Path(shape_parameters).stem
        path = out_dir / f"{stem}_pred_s{int(stage * 100):03d}.vtp"
        pred_mesh.save(str(path))
        self.log_info("single prediction stage %.3f -> %s", stage, path.name)

        result: dict[str, Any] = {
            "predicted_surface": path,
            "predicted_points": pred_points,
        }
        if ground_truth is not None:
            actual = np.asarray(pv.read(str(ground_truth)).points, dtype=np.float32)
            result["statistics"] = self._error_row(stem, stage, pred_points, actual)
            self.log_info(
                "ground-truth error: mean=%.3f mm  max=%.3f mm",
                result["statistics"]["mean_error_mm"],
                result["statistics"]["max_error_mm"],
            )
        return result

    def create_deformation_field(
        self,
        shape_parameters: Path,
        stage: float,
        reference_image: itk.Image,
        output_directory: Optional[Path] = None,
    ) -> dict[str, Any]:
        """Rasterize the inferred deformation onto a reference image grid.

        Each mesh vertex is binned by its **reference (undeformed) position**
        into ``reference_image``'s voxel grid. Each voxel of the deformation
        field holds the mean network displacement ``(dx, dy, dz)`` of the
        vertices that fall in it; each voxel of the normal image holds the mean
        (renormalized) reference-surface normal of those vertices. Empty voxels
        are zero.

        Args:
            shape_parameters: JSON file with the subject PCA coefficient vector.
            stage: Target RR-interval fraction for the deformation.
            reference_image: The frame's image; defines the output grid geometry
                (size, spacing, origin, direction).
            output_directory: If given, the two images are written there as
                compressed ``.mha`` files.

        Returns:
            Dict with ``deformation_field`` and ``normal_image`` (ITK vector
            images) and, when written, their paths.
        """
        coeffs = pnt.load_pca_coefficients(shape_parameters)
        mean_mesh, pca_model = self._load_pca_assets()
        ref_points = pnt.reconstruct_reference_points(mean_mesh, pca_model, coeffs)
        disps = self._predict_displacements(coeffs, stage)

        # Reference (undeformed) surface normals.
        ref_surface = self._mean_surface.copy(deep=True)
        ref_surface.points = ref_points
        ref_surface = ref_surface.compute_normals(
            point_normals=True, cell_normals=False, auto_orient_normals=True
        )
        normals = np.asarray(ref_surface.point_data["Normals"], dtype=np.float64)

        size = itk.size(reference_image)  # x, y, z
        sx, sy, sz = int(size[0]), int(size[1]), int(size[2])
        disp_sum = np.zeros((sz, sy, sx, 3), dtype=np.float64)
        normal_sum = np.zeros((sz, sy, sx, 3), dtype=np.float64)
        count = np.zeros((sz, sy, sx), dtype=np.float64)

        for i in range(ref_points.shape[0]):
            point = [float(c) for c in ref_points[i]]
            idx = reference_image.TransformPhysicalPointToIndex(point)
            ix, iy, iz = int(idx[0]), int(idx[1]), int(idx[2])
            if 0 <= ix < sx and 0 <= iy < sy and 0 <= iz < sz:
                disp_sum[iz, iy, ix] += disps[i]
                normal_sum[iz, iy, ix] += normals[i]
                count[iz, iy, ix] += 1.0

        occupied = count > 0
        disp_field = np.zeros_like(disp_sum, dtype=np.float32)
        normal_field = np.zeros_like(normal_sum, dtype=np.float32)
        disp_field[occupied] = (disp_sum[occupied] / count[occupied, None]).astype(
            np.float32
        )
        mean_normal = normal_sum[occupied] / count[occupied, None]
        norm = np.linalg.norm(mean_normal, axis=1, keepdims=True)
        norm = np.where(norm == 0.0, 1.0, norm)
        normal_field[occupied] = (mean_normal / norm).astype(np.float32)

        deformation_image = self._vector_image_like(disp_field, reference_image)
        normal_image = self._vector_image_like(normal_field, reference_image)
        self.log_info(
            "Deformation field: %d/%d voxels populated by %d vertices",
            int(occupied.sum()),
            sx * sy * sz,
            ref_points.shape[0],
        )

        result: dict[str, Any] = {
            "deformation_field": deformation_image,
            "normal_image": normal_image,
        }
        if output_directory is not None:
            out_dir = Path(output_directory)
            out_dir.mkdir(parents=True, exist_ok=True)
            field_path = out_dir / "deformation_field.mha"
            normal_path = out_dir / "surface_normal_field.mha"
            itk.imwrite(deformation_image, str(field_path), compression=True)
            itk.imwrite(normal_image, str(normal_path), compression=True)
            result["deformation_field_file"] = field_path
            result["normal_image_file"] = normal_path
        return result

    @staticmethod
    def _vector_image_like(array: np.ndarray, reference_image: itk.Image) -> itk.Image:
        """Wrap a ``(z, y, x, 3)`` array as an ITK vector image on ``reference``'s grid."""
        image = itk.image_from_array(np.ascontiguousarray(array), is_vector=True)
        image.SetSpacing(reference_image.GetSpacing())
        image.SetOrigin(reference_image.GetOrigin())
        image.SetDirection(reference_image.GetDirection())
        return image

    @staticmethod
    def _error_row(
        subject_id: str, stage: float, pred: np.ndarray, actual: np.ndarray
    ) -> dict:
        """Per-phase error statistics between predicted and actual points."""
        errors = pred - actual
        euclidean = np.linalg.norm(errors, axis=1)
        return {
            "subject_id": subject_id,
            "stage": stage,
            "n_points": int(len(euclidean)),
            "mean_error_mm": float(euclidean.mean()),
            "median_error_mm": float(np.median(euclidean)),
            "max_error_mm": float(euclidean.max()),
            "rms_error_mm": float(np.sqrt(np.mean(euclidean**2))),
            "std_error_mm": float(euclidean.std()),
            "mean_abs_error_x_mm": float(np.abs(errors[:, 0]).mean()),
            "mean_abs_error_y_mm": float(np.abs(errors[:, 1]).mean()),
            "mean_abs_error_z_mm": float(np.abs(errors[:, 2]).mean()),
        }


class WorkflowInferPhysicsNeMoMGN(WorkflowInferPhysicsNeMo):
    """Infer cardiac mesh stages with a trained PhysicsNeMo MeshGraphNet."""

    _model_tag = "mgn"

    def _build_model(self, meta: dict) -> "torch.nn.Module":

        try:
            import torch_geometric  # noqa: F401 - needed by the graph seams

            from physicsnemo.models.meshgraphnet import MeshGraphNet
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "The MGN inferencer requires PhysicsNeMo and PyTorch Geometric. "
                'Install with: pip install "physiotwin4d[physicsnemo]" && '
                "pip install torch-geometric"
            ) from exc

        num_layers = int(meta.get("num_layers", 2))
        hidden_dim = int(meta["hidden_dim"])
        model = MeshGraphNet(
            input_dim_nodes=int(meta["in_features"]),
            input_dim_edges=int(meta.get("input_dim_edges", 4)),
            output_dim=3,
            processor_size=int(meta["processor_size"]),
            hidden_dim_processor=hidden_dim,
            hidden_dim_node_encoder=hidden_dim,
            num_layers_node_encoder=num_layers,
            hidden_dim_node_decoder=hidden_dim,
            num_layers_node_decoder=num_layers,
            hidden_dim_edge_encoder=hidden_dim,
            num_layers_edge_encoder=num_layers,
            num_layers_edge_processor=num_layers,
            num_layers_node_processor=num_layers,
            aggregation="mean",
            num_processor_checkpoint_segments=int(
                meta.get("num_processor_checkpoint_segments", 0)
            ),
        )
        return cast("torch.nn.Module", model)

    def _load_extra_artifacts(self) -> None:
        import torch
        from torch_geometric.data import Data

        edge_index = torch.load(
            str(self.model_directory / "shared_edge_index.pt"),
            map_location="cpu",
            weights_only=True,
        )
        edge_feats = torch.load(
            str(self.model_directory / "shared_edge_features.pt"),
            map_location="cpu",
            weights_only=True,
        )
        self._shared_edge_index = edge_index
        self._shared_edge_feats = edge_feats.to(self._device)
        self._shared_graph = Data(
            edge_index=edge_index, num_nodes=len(self._mean_shape_coords)
        ).to(self._device)

    def _network_predict(self, node_feats: np.ndarray) -> np.ndarray:
        import torch

        nf = torch.from_numpy(node_feats.astype(np.float32)).to(self._device)
        with torch.no_grad():
            pred = self._model(nf, self._shared_edge_feats, self._shared_graph)
        return np.asarray(pred.cpu().numpy(), dtype=np.float32)


class WorkflowInferPhysicsNeMoMLP(WorkflowInferPhysicsNeMo):
    """Infer cardiac mesh stages with a trained PhysicsNeMo FullyConnected model."""

    _model_tag = "mlp"
    _INFER_CHUNK = 262144

    def _build_model(self, meta: dict) -> "torch.nn.Module":

        try:
            from physicsnemo.models.mlp import FullyConnected
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "The MLP inferencer requires PhysicsNeMo, an optional dependency. "
                'Install with: pip install "physiotwin4d[physicsnemo]"'
            ) from exc

        model = FullyConnected(
            in_features=int(meta["in_features"]),
            layer_size=int(meta["layer_size"]),
            out_features=3,
            num_layers=int(meta["num_layers"]),
            activation_fn="silu",
            skip_connections=True,
        )
        return cast("torch.nn.Module", model)

    def _load_extra_artifacts(self) -> None:
        # MLP has no shared graph artifacts.
        return None

    def _network_predict(self, node_feats: np.ndarray) -> np.ndarray:
        import torch

        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(node_feats), self._INFER_CHUNK):
                block = node_feats[start : start + self._INFER_CHUNK].astype(np.float32)
                tensor = torch.from_numpy(block).to(self._device)
                chunks.append(self._model(tensor).cpu().numpy())
        return np.vstack(chunks).astype(np.float32)
