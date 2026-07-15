"""Workflows for training PhysicsNeMo cardiac mesh-stage models.

A shared base class :class:`WorkflowTrainPhysicsNeMo` holds every step common to
the two supported networks; the concrete
:class:`WorkflowTrainPhysicsNeMoMGN` (MeshGraphNet) and
:class:`WorkflowTrainPhysicsNeMoMLP` (fully connected) subclasses supply only
the network-specific seams.  Both learn the same task as tutorials 9a/9b: given
per-vertex features ``[mean_shape_x, mean_shape_y, mean_shape_z, pca_c1..cN,
stage]`` predict a per-vertex displacement ``(dx, dy, dz)`` from the subject's
SSM reference surface (the Option B convention).

Design highlights:

- **Data is a list of per-subject manifest files** (see
  :func:`physiotwin4d.physicsnemo_tools.parse_manifest`).  The caller chooses the
  train / validation / held-out-test split externally; the trainer receives the
  training manifests and validation manifest(s) and reports validation RMSE
  intermittently as training proceeds.
- **The dataset streams lazily** through
  :class:`physiotwin4d.physicsnemo_tools.PhaseSampleDataset` with a bounded RAM
  cache, so the training set need not fit in memory (unlike the tutorials, which
  pre-stack the whole dataset onto the GPU).
- **Coordinates are always the PCA mean-shape surface** (shared across
  subjects); the subject is described by its PCA parameters and the stage.

MLP note: the tutorial MLP drew mini-batches of individual points shuffled
across all subjects/phases.  Streaming by file makes the natural batch unit a
group of whole ``(subject, phase)`` samples, so the MLP batches several samples
per step and shuffles points *within* the batch to retain gradient mixing
(``batch_size`` therefore counts samples, not points).  The MGN keeps per-sample
vertex order intact because it indexes the shared mesh graph.

PhysicsNeMo (and, for the MGN, PyTorch Geometric) are optional dependencies;
they are imported lazily so ``import physiotwin4d`` works without them.
"""

from __future__ import annotations

import csv
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, cast

import numpy as np
import pyvista as pv

from . import physicsnemo_tools as pnt
from .physicsnemo_tools import PhaseSampleDataset, SubjectManifest, _Sample
from .physiotwin4d_base import PhysioTwin4DBase
from .test_tools import TestTools

if TYPE_CHECKING:  # typed for mypy; imported lazily at runtime
    import torch


class WorkflowTrainPhysicsNeMo(PhysioTwin4DBase):
    """Base class for training a PhysicsNeMo cardiac mesh-stage model.

    Not instantiated directly — use :class:`WorkflowTrainPhysicsNeMoMGN` or
    :class:`WorkflowTrainPhysicsNeMoMLP`.  Subclasses implement the network
    seams (:meth:`_build_model`, :meth:`_setup_model_inputs`, :meth:`_forward`,
    :meth:`_checkpoint_extra`, :meth:`_save_extra_artifacts`) and set the class
    attributes ``_model_tag``, ``_architecture_name`` and
    ``_shuffle_points_within_batch``.
    """

    # Network identity / behavior — overridden by subclasses.
    _model_tag: str = "base"
    _architecture_name: str = "base"
    _shuffle_points_within_batch: bool = False

    def __init__(
        self,
        train_manifests: list[Path],
        val_manifests: list[Path],
        pca_mean_mesh: Path,
        output_directory: Path,
        resume_from: Optional[Path] = None,
        log_level: int | str = logging.INFO,
    ) -> None:
        """Initialize the trainer.

        Args:
            train_manifests: Per-subject manifest files for the training set.
            val_manifests: Per-subject manifest files for the validation set
                (used for intermittent RMSE reporting during training). May be
                empty to skip validation.
            pca_mean_mesh: PCA template mesh whose point count matches
                ``pca_model.json`` (typically ``pca_mean.vtu``). Its extracted
                surface defines the shared node coordinates and — for the MGN —
                the mesh-graph topology. The sibling ``pca_model.json`` (if
                present) is copied into ``output_directory`` for inference.
            output_directory: Directory for checkpoints, metadata and logs.
            resume_from: Optional ``*_stage_model.pt`` to resume from; its
                normalization statistics are inherited so the loaded weights stay
                valid, and a fresh numbered output directory is used.
            log_level: Logging level. Default: ``logging.INFO``.

        Raises:
            ValueError: If ``train_manifests`` is empty.
            FileNotFoundError: If ``pca_mean_mesh`` does not exist.
        """
        super().__init__(class_name=self.__class__.__name__, log_level=log_level)

        if not train_manifests:
            raise ValueError("train_manifests cannot be empty.")
        pca_mean_mesh = Path(pca_mean_mesh)
        if not pca_mean_mesh.exists():
            raise FileNotFoundError(f"pca_mean_mesh not found: {pca_mean_mesh}")

        self.train_manifest_paths = [Path(p) for p in train_manifests]
        self.val_manifest_paths = [Path(p) for p in val_manifests]
        self.pca_mean_mesh = pca_mean_mesh
        self.output_directory = Path(output_directory)
        self.resume_from = Path(resume_from) if resume_from is not None else None

        # PCA assets shared by every subject.
        self._pca_mean_dataset: pv.DataSet = pv.read(str(pca_mean_mesh))
        self._mean_surface: pv.PolyData = self._pca_mean_dataset.extract_surface(
            algorithm="dataset_surface"
        )
        self._mean_shape_coords = np.asarray(
            self._mean_surface.points, dtype=np.float32
        )
        self._pca_model_path: Optional[Path] = None
        candidate = pca_mean_mesh.parent / "pca_model.json"
        if candidate.exists():
            self._pca_model_path = candidate

        # Shared hyper-parameters (subclasses may override the batch default).
        self.epochs: int = 1500
        self.batch_size: int = 4
        self.learning_rate: float = 1.0e-3
        self.cache_max_samples: int = 0
        self.rmse_log_interval: int = 100
        self.loss_log_interval: int = 10
        self.seed: int = 42

        # Results (populated by process()).
        self.checkpoint_file: Optional[Path] = None
        self.metadata_file: Optional[Path] = None
        self.training_loss: Optional[list[float]] = None
        self.val_rmse_log: Optional[list[dict]] = None

    # ─────────────────────────── Tuning setters ────────────────────────────
    def set_epochs(self, epochs: int) -> None:
        """Set the number of training epochs."""
        if epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {epochs}")
        self.epochs = epochs

    def set_batch_size(self, batch_size: int) -> None:
        """Set the mini-batch size, measured in ``(subject, phase)`` samples."""
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.batch_size = batch_size

    def set_learning_rate(self, learning_rate: float) -> None:
        """Set the Adam learning rate."""
        if learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be > 0, got {learning_rate}")
        self.learning_rate = learning_rate

    def set_cache_size(self, cache_max_samples: int) -> None:
        """Set the RAM cache budget (decoded phase arrays); ``0`` = unbounded."""
        if cache_max_samples < 0:
            raise ValueError(f"cache_max_samples must be >= 0, got {cache_max_samples}")
        self.cache_max_samples = cache_max_samples

    # ─────────────────────────── Network seams ─────────────────────────────
    def _build_model(self, in_features: int) -> "torch.nn.Module":
        """Construct the (uncompiled) network. Implemented by subclasses."""
        raise NotImplementedError

    def _setup_model_inputs(self, device: "torch.device") -> None:
        """Prepare any shared per-forward inputs (MGN graph tensors)."""
        raise NotImplementedError

    def _forward(
        self, model: "torch.nn.Module", node_feats: "torch.Tensor", batch_len: int
    ) -> "torch.Tensor":
        """Run the network for a flattened ``(batch_len * n_points, F)`` batch."""
        raise NotImplementedError

    def _checkpoint_extra(self) -> dict:
        """Return architecture-specific fields to store in the checkpoint."""
        raise NotImplementedError

    def _save_extra_artifacts(self, output_dir: Path) -> None:
        """Save any architecture-specific artifacts (MGN graph tensors)."""
        raise NotImplementedError

    # ─────────────────────────── Main workflow ─────────────────────────────
    def process(self) -> dict[str, Any]:
        """Train the model and write checkpoints, metadata and logs.

        Returns:
            Dict with ``output_directory``, ``checkpoint``, ``metadata``,
            ``training_loss`` and ``val_rmse_log``.
        """
        import torch

        self.log_section(
            "STARTING PHYSICSNEMO %s TRAINING WORKFLOW", self._model_tag.upper()
        )

        epochs = self.epochs
        if TestTools.running_as_test():
            epochs = min(epochs, 2)
            self.log_info("Test mode: reducing epochs to %d", epochs)

        output_dir = self._resolve_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        self.log_info("Output directory: %s", output_dir)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_info("Device: %s", device.type)

        subjects = self._load_subjects()
        resume_ckpt = self._load_resume_checkpoint()
        stats = self._compute_normalization(subjects, resume_ckpt)

        train_dataset, val_dataset = self._build_datasets(subjects, stats)
        self.log_info(
            "Training samples: %d, validation samples: %d, in_features=%d, "
            "displacement_scale=%.4f mm",
            len(train_dataset),
            len(val_dataset),
            train_dataset.n_features,
            stats["displacement_scale"],
        )

        model, losses, rmse_log = self._train(
            train_dataset, val_dataset, stats, device, epochs, output_dir
        )
        self._save_model(model, subjects, stats, losses, rmse_log, output_dir, epochs)

        self.log_section("PHYSICSNEMO %s TRAINING COMPLETE", self._model_tag.upper())
        return {
            "output_directory": output_dir,
            "checkpoint": self.checkpoint_file,
            "metadata": self.metadata_file,
            "training_loss": losses,
            "val_rmse_log": rmse_log,
        }

    # ─────────────────────────── Internal steps ────────────────────────────
    def _resolve_output_dir(self) -> Path:
        """Return the output directory, using a fresh sibling when resuming."""
        base = self.output_directory
        if self.resume_from is None or not base.exists():
            return base
        n = 1
        while True:
            candidate = base.parent / f"{base.name}_{n}"
            if not candidate.exists():
                return candidate
            n += 1

    def _load_subjects(self) -> dict[str, dict]:
        """Parse every manifest and load reference points + PCA coefficients."""
        n_points = len(self._mean_shape_coords)
        subjects: dict[str, dict] = {}

        def _load(paths: list[Path], split: str) -> None:
            for manifest_path in paths:
                manifest: SubjectManifest = pnt.parse_manifest(manifest_path)
                if manifest.subject_id in subjects:
                    raise ValueError(
                        f"Duplicate subject_id '{manifest.subject_id}': already "
                        f"loaded in the '{subjects[manifest.subject_id]['split']}' "
                        f"split, seen again in the '{split}' split. Each subject "
                        "must appear in exactly one manifest."
                    )
                ref_mesh = pv.read(str(manifest.reference_surface))
                ref_points = np.asarray(ref_mesh.points, dtype=np.float32)
                if ref_points.shape[0] != n_points:
                    raise ValueError(
                        f"{manifest.reference_surface} has {ref_points.shape[0]} "
                        f"points, expected {n_points} (mean-shape topology)."
                    )
                subjects[manifest.subject_id] = {
                    "split": split,
                    "ref_points": ref_points,
                    "pca_coeffs": pnt.load_pca_coefficients(manifest.pca_coefficients),
                    "phases": manifest.phases,
                }

        _load(self.train_manifest_paths, "train")
        _load(self.val_manifest_paths, "val")
        n_train = sum(1 for s in subjects.values() if s["split"] == "train")
        if n_train == 0:
            raise ValueError("No training subjects were loaded.")
        return subjects

    def _load_resume_checkpoint(self) -> Optional[dict]:
        """Load prior-run normalization statistics when resuming."""
        if self.resume_from is None:
            return None
        import torch

        self.log_info("Resuming from %s", self.resume_from)
        return cast(
            dict,
            torch.load(str(self.resume_from), map_location="cpu", weights_only=True),
        )

    def _compute_normalization(
        self, subjects: dict[str, dict], resume_ckpt: Optional[dict]
    ) -> dict:
        """Compute (or inherit) coordinate, PCA and displacement statistics."""
        # Inherit the exact stats when the checkpoint carries them (final models
        # and, since this change, periodic epoch checkpoints). Bare/legacy epoch
        # checkpoints hold only weights: recompute from the data, which is
        # identical for an unchanged subject set (the normal resume case).
        if resume_ckpt is not None and "coordinate_mean" in resume_ckpt:
            return {
                "coordinate_mean": np.array(resume_ckpt["coordinate_mean"], np.float32),
                "coordinate_scale": np.array(
                    resume_ckpt["coordinate_scale"], np.float32
                ),
                "pca_mean": np.array(resume_ckpt["pca_mean"], np.float32),
                "pca_scale": np.array(resume_ckpt["pca_scale"], np.float32),
                "displacement_scale": float(resume_ckpt["displacement_scale"]),
            }
        if resume_ckpt is not None:
            self.log_warning(
                "Resume checkpoint has no normalization stats (bare weights-only "
                "checkpoint); recomputing them from the current data."
            )

        coord = self._mean_shape_coords
        coordinate_mean = coord.mean(axis=0)
        coordinate_scale = np.where(coord.std(axis=0) == 0.0, 1.0, coord.std(axis=0))

        train_pca = np.vstack(
            [s["pca_coeffs"] for s in subjects.values() if s["split"] == "train"]
        )
        pca_mean = train_pca.mean(axis=0)
        pca_scale = np.where(train_pca.std(axis=0) == 0.0, 1.0, train_pca.std(axis=0))

        displacement_scale = self._compute_displacement_scale(subjects)
        return {
            "coordinate_mean": coordinate_mean.astype(np.float32),
            "coordinate_scale": coordinate_scale.astype(np.float32),
            "pca_mean": pca_mean.astype(np.float32),
            "pca_scale": pca_scale.astype(np.float32),
            "displacement_scale": displacement_scale,
        }

    def _compute_displacement_scale(self, subjects: dict[str, dict]) -> float:
        """One streaming pass over training targets for the max abs displacement."""
        n_points = len(self._mean_shape_coords)
        max_abs = 0.0
        for data in subjects.values():
            if data["split"] != "train":
                continue
            ref_points = data["ref_points"]
            for phase in data["phases"]:
                mesh = pv.read(str(phase.surface))
                if mesh.n_points != n_points:
                    raise ValueError(
                        f"{phase.surface} has {mesh.n_points} points, "
                        f"expected {n_points}."
                    )
                disp = np.asarray(mesh.points, dtype=np.float32) - ref_points
                max_abs = max(max_abs, float(np.max(np.abs(disp))))
        return max_abs if max_abs > 0.0 else 1.0

    def _build_datasets(
        self, subjects: dict[str, dict], stats: dict
    ) -> tuple[PhaseSampleDataset, PhaseSampleDataset]:
        """Build lazy train and validation datasets sharing the mean coords."""
        mean_coords_norm = (self._mean_shape_coords - stats["coordinate_mean"]) / stats[
            "coordinate_scale"
        ]

        def _samples(split: str) -> list[_Sample]:
            out: list[_Sample] = []
            for sid, data in sorted(subjects.items()):
                if data["split"] != split:
                    continue
                pca_norm = (data["pca_coeffs"] - stats["pca_mean"]) / stats["pca_scale"]
                for phase in data["phases"]:
                    out.append(
                        _Sample(
                            subject_id=sid,
                            pca_norm=pca_norm.astype(np.float32),
                            ref_points=data["ref_points"],
                            phase_surface=phase.surface,
                            stage=phase.stage,
                        )
                    )
            return out

        train_dataset = PhaseSampleDataset(
            _samples("train"),
            mean_coords_norm,
            stats["displacement_scale"],
            self.cache_max_samples,
        )
        val_dataset = PhaseSampleDataset(
            _samples("val"),
            mean_coords_norm,
            stats["displacement_scale"],
            self.cache_max_samples,
        )
        return train_dataset, val_dataset

    def _iter_batches(
        self, dataset: PhaseSampleDataset, rng: Any, shuffle: bool
    ) -> Any:
        """Yield ``(node_feats, targets, batch_len)`` flattened mini-batches."""
        n = len(dataset)
        order = rng.permutation(n) if shuffle else np.arange(n)
        for start in range(0, n, self.batch_size):
            idx = order[start : start + self.batch_size]
            pairs = [dataset[int(i)] for i in idx]
            node_feats = np.vstack([p[0] for p in pairs])
            targets = np.vstack([p[1] for p in pairs])
            if shuffle and self._shuffle_points_within_batch:
                perm = rng.permutation(len(node_feats))
                node_feats = node_feats[perm]
                targets = targets[perm]
            yield node_feats, targets, len(idx)

    def _train(
        self,
        train_dataset: PhaseSampleDataset,
        val_dataset: PhaseSampleDataset,
        stats: dict,
        device: "torch.device",
        epochs: int,
        output_dir: Path,
    ) -> tuple["torch.nn.Module", list[float], list[dict]]:
        """Run the training loop, returning the model and loss/RMSE logs."""
        import torch

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        in_features = train_dataset.n_features

        model = self._build_model(in_features).to(device)
        if self.resume_from is not None:
            ckpt = torch.load(
                str(self.resume_from), map_location=device, weights_only=True
            )
            state = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(pnt.strip_compile_prefix(state))
            self.log_info("Loaded model weights from %s", self.resume_from)

        self._setup_model_inputs(device)

        if sys.platform != "win32":
            try:
                model = cast("torch.nn.Module", torch.compile(model))
                self.log_info("torch.compile enabled.")
            except Exception as exc:  # pragma: no cover - platform dependent
                self.log_info("torch.compile skipped (%s).", exc)
        else:
            self.log_info("torch.compile skipped on Windows.")

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        loss_fn = torch.nn.MSELoss()
        displacement_scale = stats["displacement_scale"]

        losses: list[float] = []
        rmse_log: list[dict] = []
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            n_rows = 0
            for node_feats, targets, batch_len in self._iter_batches(
                train_dataset, rng, shuffle=True
            ):
                nf = torch.from_numpy(node_feats).to(device)
                tgt = torch.from_numpy(targets).to(device)
                optimizer.zero_grad(set_to_none=True)
                with self._autocast(device):
                    pred = self._forward(model, nf, batch_len)
                    loss = loss_fn(pred, tgt)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach()) * len(nf)
                n_rows += len(nf)
            losses.append(epoch_loss / max(n_rows, 1))

            if (epoch + 1) % self.loss_log_interval == 0 or epoch + 1 == epochs:
                self.log_info(
                    "  epoch %05d/%d  loss=%.6f", epoch + 1, epochs, losses[-1]
                )

            if (epoch + 1) % self.rmse_log_interval == 0 or epoch + 1 == epochs:
                train_rmse = self._evaluate_rmse(
                    model, train_dataset, displacement_scale, device
                )
                val_rmse = (
                    self._evaluate_rmse(model, val_dataset, displacement_scale, device)
                    if len(val_dataset) > 0
                    else float("nan")
                )
                rmse_log.append(
                    {
                        "epoch": epoch + 1,
                        "train_rmse_mm": train_rmse,
                        "val_rmse_mm": val_rmse,
                    }
                )
                ckpt_path = (
                    output_dir
                    / f"{self._model_tag}_stage_model_epoch_{epoch + 1:05d}.pt"
                )
                torch.save(self._build_checkpoint(model, stats), ckpt_path)
                self.log_info(
                    "  intermittent test epoch %05d/%d  train RMSE=%.4f mm  "
                    "val RMSE=%.4f mm  checkpoint=%s",
                    epoch + 1,
                    epochs,
                    train_rmse,
                    val_rmse,
                    ckpt_path.name,
                )
        model.eval()
        return model, losses, rmse_log

    def _autocast(self, device: "torch.device") -> Any:
        """BF16 autocast on CUDA; a no-op context elsewhere."""
        import contextlib

        import torch

        if device.type == "cuda":
            return torch.amp.autocast(device.type, dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _evaluate_rmse(
        self,
        model: "torch.nn.Module",
        dataset: PhaseSampleDataset,
        displacement_scale: float,
        device: "torch.device",
    ) -> float:
        """Euclidean per-point RMSE in mm over a dataset."""
        import torch

        rng = np.random.default_rng(0)
        model.eval()
        total_sq = 0.0
        n_points = 0
        with torch.no_grad():
            for node_feats, targets, batch_len in self._iter_batches(
                dataset, rng, shuffle=False
            ):
                nf = torch.from_numpy(node_feats).to(device)
                pred = self._forward(model, nf, batch_len).cpu().numpy()
                err = (pred - targets) * displacement_scale
                total_sq += float(np.sum(err**2))
                n_points += err.shape[0]
        model.train()
        return float(np.sqrt(total_sq / max(n_points, 1)))

    def _build_checkpoint(
        self, model: "torch.nn.Module", stats: dict
    ) -> dict[str, Any]:
        """Assemble a self-describing checkpoint (weights + normalization stats).

        Both the periodic epoch checkpoints and the final model share this
        payload so training can resume from — and inference can load — any saved
        checkpoint, not just the final one.
        """
        checkpoint: dict[str, Any] = {
            "model_state_dict": pnt.uncompiled_state_dict(model),
            "architecture": self._architecture_name,
            "in_features": 3 + int(stats["pca_mean"].shape[0]) + 1,
            "n_pca": int(stats["pca_mean"].shape[0]),
            "coordinate_mean": stats["coordinate_mean"].tolist(),
            "coordinate_scale": stats["coordinate_scale"].tolist(),
            "pca_mean": stats["pca_mean"].tolist(),
            "pca_scale": stats["pca_scale"].tolist(),
            "displacement_scale": stats["displacement_scale"],
        }
        checkpoint.update(self._checkpoint_extra())
        return checkpoint

    def _save_model(
        self,
        model: "torch.nn.Module",
        subjects: dict[str, dict],
        stats: dict,
        losses: list[float],
        rmse_log: list[dict],
        output_dir: Path,
        epochs: int,
    ) -> None:
        """Persist the checkpoint, metadata, logs and shared PCA assets."""
        import torch

        in_features = 3 + int(stats["pca_mean"].shape[0]) + 1
        tag = self._model_tag
        checkpoint_file = output_dir / f"{tag}_stage_model.pt"
        metadata_file = output_dir / f"{tag}_stage_model_metadata.json"

        train_ids = sorted(s for s, d in subjects.items() if d["split"] == "train")
        val_ids = sorted(s for s, d in subjects.items() if d["split"] == "val")

        checkpoint = self._build_checkpoint(model, stats)
        checkpoint["train_subject_ids"] = train_ids
        checkpoint["val_subject_ids"] = val_ids
        checkpoint["resumed_from"] = str(self.resume_from) if self.resume_from else None
        torch.save(checkpoint, checkpoint_file)

        n_pca = int(stats["pca_mean"].shape[0])
        input_feature_names = (
            ["mean_shape_x", "mean_shape_y", "mean_shape_z"]
            + [f"pca_c{i + 1}" for i in range(n_pca)]
            + ["stage"]
        )
        metadata = {
            "architecture": self._architecture_name,
            "input_features": input_feature_names,
            "output_features": ["dx", "dy", "dz"],
            "in_features": in_features,
            "n_mesh_points": int(self._mean_shape_coords.shape[0]),
            "epochs": epochs,
            "learning_rate": self.learning_rate,
            "batch_size_samples": self.batch_size,
            "coordinate_mean": stats["coordinate_mean"].tolist(),
            "coordinate_scale": stats["coordinate_scale"].tolist(),
            "pca_mean": stats["pca_mean"].tolist(),
            "pca_scale": stats["pca_scale"].tolist(),
            "displacement_scale": stats["displacement_scale"],
            "displacement_convention": "Option B: relative to subject reference surface",
            "resumed_from": str(self.resume_from) if self.resume_from else None,
        }
        metadata.update(self._checkpoint_extra())
        metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        (output_dir / "training_losses.json").write_text(
            json.dumps(losses, indent=2), encoding="utf-8"
        )
        (output_dir / "training_validation_rmse.json").write_text(
            json.dumps(rmse_log, indent=2), encoding="utf-8"
        )
        with (output_dir / "training_validation_rmse.csv").open(
            "w", newline="", encoding="utf-8"
        ) as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["epoch", "train_rmse_mm", "val_rmse_mm"]
            )
            writer.writeheader()
            writer.writerows(rmse_log)

        # Copy PCA assets so the model directory is self-contained for inference.
        shutil.copy2(self.pca_mean_mesh, output_dir / self.pca_mean_mesh.name)
        self._mean_surface.save(str(output_dir / "pca_mean_surface.vtp"))
        if self._pca_model_path is not None:
            shutil.copy2(self._pca_model_path, output_dir / "pca_model.json")

        self._save_extra_artifacts(output_dir)

        self.checkpoint_file = checkpoint_file
        self.metadata_file = metadata_file
        self.training_loss = losses
        self.val_rmse_log = rmse_log
        self.log_info("Model saved to %s", checkpoint_file)


class WorkflowTrainPhysicsNeMoMGN(WorkflowTrainPhysicsNeMo):
    """Train a PhysicsNeMo :class:`MeshGraphNet` on cardiac mesh stages.

    The mesh-graph topology and edge features are extracted once from the shared
    mean-shape surface and reused for every ``(subject, phase)`` sample; PyTorch
    Geometric batches join disconnected sub-graphs.
    """

    _model_tag = "mgn"
    _architecture_name = "physicsnemo.models.meshgraphnet.MeshGraphNet"
    _shuffle_points_within_batch = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.batch_size = 4  # graphs per step
        self.processor_size: int = 3
        self.hidden_dim: int = 128
        self.num_layers: int = 2
        self.num_processor_checkpoint_segments: int = 0
        # Runtime MGN state (set in _setup_model_inputs).
        self._device: Optional["torch.device"] = None
        self._shared_graph: Any = None
        self._shared_edge_index: Any = None
        self._shared_edge_feats: Any = None
        self._batched_graph_cache: dict[int, tuple[Any, Any]] = {}

    def set_processor_size(self, processor_size: int) -> None:
        """Set the number of message-passing hops."""
        if processor_size < 1:
            raise ValueError(f"processor_size must be >= 1, got {processor_size}")
        self.processor_size = processor_size

    def set_hidden_dim(self, hidden_dim: int) -> None:
        """Set the processor/encoder/decoder hidden dimension."""
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")
        self.hidden_dim = hidden_dim

    def set_num_layers(self, num_layers: int) -> None:
        """Set the MLP layer count inside each encoder/processor/decoder block."""
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.num_layers = num_layers

    def _build_model(self, in_features: int) -> "torch.nn.Module":

        try:
            import torch_geometric  # noqa: F401 - needed by the graph seams

            from physicsnemo.models.meshgraphnet import MeshGraphNet
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "The MGN trainer requires PhysicsNeMo and PyTorch Geometric. "
                'Install with: pip install "physiotwin4d[physicsnemo]" && '
                "pip install torch-geometric"
            ) from exc

        model = MeshGraphNet(
            input_dim_nodes=in_features,
            input_dim_edges=4,  # rel_x, rel_y, rel_z, distance
            output_dim=3,
            processor_size=self.processor_size,
            hidden_dim_processor=self.hidden_dim,
            hidden_dim_node_encoder=self.hidden_dim,
            num_layers_node_encoder=self.num_layers,
            hidden_dim_node_decoder=self.hidden_dim,
            num_layers_node_decoder=self.num_layers,
            hidden_dim_edge_encoder=self.hidden_dim,
            num_layers_edge_encoder=self.num_layers,
            num_layers_edge_processor=self.num_layers,
            num_layers_node_processor=self.num_layers,
            aggregation="mean",
            num_processor_checkpoint_segments=self.num_processor_checkpoint_segments,
        )
        return cast("torch.nn.Module", model)

    def _setup_model_inputs(self, device: "torch.device") -> None:
        from torch_geometric.data import Data

        self._device = device
        self._shared_edge_index = pnt.mesh_to_edge_index(self._mean_surface)
        self._shared_edge_feats = pnt.compute_edge_features(
            self._mean_shape_coords, self._shared_edge_index
        )
        self._shared_graph = Data(
            edge_index=self._shared_edge_index,
            num_nodes=len(self._mean_shape_coords),
        )
        self._batched_graph_cache = {}

    def _batched_graph(self, batch_len: int) -> tuple[Any, Any]:
        """Return (and cache) a batched graph + tiled edge features for ``batch_len``."""
        cached = self._batched_graph_cache.get(batch_len)
        if cached is not None:
            return cached
        from torch_geometric.data import Batch

        assert self._device is not None
        graph = Batch.from_data_list([self._shared_graph] * batch_len).to(self._device)
        edge_feats = self._shared_edge_feats.repeat(batch_len, 1).to(self._device)
        self._batched_graph_cache[batch_len] = (graph, edge_feats)
        return graph, edge_feats

    def _forward(
        self, model: "torch.nn.Module", node_feats: "torch.Tensor", batch_len: int
    ) -> "torch.Tensor":
        graph, edge_feats = self._batched_graph(batch_len)
        return cast("torch.Tensor", model(node_feats, edge_feats, graph))

    def _checkpoint_extra(self) -> dict:
        return {
            "processor_size": self.processor_size,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_processor_checkpoint_segments": self.num_processor_checkpoint_segments,
            "input_dim_edges": 4,
        }

    def _save_extra_artifacts(self, output_dir: Path) -> None:
        import torch

        torch.save(self._shared_edge_index, output_dir / "shared_edge_index.pt")
        torch.save(self._shared_edge_feats, output_dir / "shared_edge_features.pt")


class WorkflowTrainPhysicsNeMoMLP(WorkflowTrainPhysicsNeMo):
    """Train a PhysicsNeMo :class:`FullyConnected` (MLP) on cardiac mesh stages.

    Each surface point is an independent training row; batches group several
    ``(subject, phase)`` samples and shuffle points within the batch to retain
    gradient mixing while still streaming from disk.
    """

    _model_tag = "mlp"
    _architecture_name = "physicsnemo.models.mlp.FullyConnected"
    _shuffle_points_within_batch = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.batch_size = 32  # samples per step (points shuffled within the batch)
        self.layer_size: int = 512
        self.num_layers: int = 6

    def set_layer_size(self, layer_size: int) -> None:
        """Set the hidden layer width."""
        if layer_size < 1:
            raise ValueError(f"layer_size must be >= 1, got {layer_size}")
        self.layer_size = layer_size

    def set_num_layers(self, num_layers: int) -> None:
        """Set the number of fully connected layers."""
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.num_layers = num_layers

    def _build_model(self, in_features: int) -> "torch.nn.Module":

        try:
            from physicsnemo.models.mlp import FullyConnected
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "The MLP trainer requires PhysicsNeMo, an optional dependency. "
                'Install with: pip install "physiotwin4d[physicsnemo]"'
            ) from exc

        model = FullyConnected(
            in_features=in_features,
            layer_size=self.layer_size,
            out_features=3,
            num_layers=self.num_layers,
            activation_fn="silu",
            skip_connections=True,
        )
        return cast("torch.nn.Module", model)

    def _setup_model_inputs(self, device: "torch.device") -> None:
        # MLP needs no shared graph inputs.
        return None

    def _forward(
        self, model: "torch.nn.Module", node_feats: "torch.Tensor", batch_len: int
    ) -> "torch.Tensor":
        return cast("torch.Tensor", model(node_feats))

    def _checkpoint_extra(self) -> dict:
        return {"layer_size": self.layer_size, "num_layers": self.num_layers}

    def _save_extra_artifacts(self, output_dir: Path) -> None:
        # No MGN-only graph artifacts for the MLP.
        return None
