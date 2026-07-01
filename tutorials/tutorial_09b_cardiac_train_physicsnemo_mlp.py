"""
Tutorial 9b (MLP): Train a PhysicsNeMo MLP for cardiac mesh stage prediction.

Second stage of the cardiac 4D deep-learning pipeline (Tutorials 8 -> 9 -> 10).
This tutorial consumes the per-time-point SSM-warped surfaces created by
Tutorial 8 (``tutorial_08_cardiac_fit_model.py``).  It trains a single shared
PhysicsNeMo fully connected (MLP) model across all training subjects that maps a
surface point ``(x, y, z, pca_c1 ... pca_cN, stage)`` to the point's
displacement from that subject's SSM reference surface, where ``stage`` is the
normalized cardiac stage (RR-interval fraction).  Once trained, the model
predicts a cardiac mesh at any requested stage without re-running image
registration.  Evaluate the trained model with Tutorial 10b
(``tutorial_10b_cardiac_eval_physicsnemo_mlp.py``).

The companion Tutorial 9a (``tutorial_09a_cardiac_train_physicsnemo_mgn.py``)
solves the same task with a MeshGraphNet so the two architectures can be compared
directly; both use the same Option B displacement convention (targets relative to
each subject's own SSM reference surface).  Subjects are split into
train / val / test via the explicit ``TEST_SUBJECTS`` / ``VAL_SUBJECTS`` lists.

Bring Your Own Data
-------------------
This is a bring-your-own-data tutorial: the path constants below point at a local
``D:/PhysioMotion4D/`` layout produced by Tutorial 8, not at the repository
``data/`` directory.  Edit them to match your own data location.

Data Required
-------------
Run Tutorial 8 first so
``D:/PhysioMotion4D/duke_data/fitted_kcl_meshes/pm00??/`` contains:

  * ``pm00XX_ssm_surface.vtp``        - reference (template) SSM surface
  * ``pm00XX_ssm_pca_coefficients.json`` - fitted PCA coefficient vector
  * ``pm00XX_g0TT_ssm_surface.vtp``   - SSM surface at each gated phase TT%

Outputs (under ``OUTPUT_DIR``)
------------------------------
  * ``physicsnemo_stage_model.pt``        - weights + normalization metadata
  * ``physicsnemo_stage_model_epoch_*.pt``- intermittent checkpoints
  * ``physicsnemo_stage_model_metadata.json``, ``training_losses.json``,
    ``training_validation_rmse.{json,csv}`` - logs
  * ``OUTPUT_DIR/test_predictions/statistics_*.csv`` and per-subject predicted
    surfaces under ``OUTPUT_DIR/pm00XX/``

Extra Install Required
----------------------
PhysicsNeMo is an optional dependency of PhysioMotion4D. Install it with::

    pip install "physiomotion4d[physicsnemo]"

PhysicsNeMo itself requires Python >= 3.11.
"""

# %%
from __future__ import annotations

import csv
import json
import logging
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import pyvista as pv
import torch


from physiomotion4d.test_tools import TestTools

try:
    from physicsnemo.models.mlp import FullyConnected
except ImportError as exc:  # pragma: no cover - import-time guard
    raise ImportError(
        "Tutorial 9b requires PhysicsNeMo, which is an optional dependency. "
        'Install with: pip install "physiomotion4d[physicsnemo]" '
        "(requires Python >= 3.11).",
    ) from exc


# nnUNetv2 (used by TotalSegmentator inside several workflows) spawns a
# multiprocessing.Pool. On Windows the spawn start method re-imports this
# script in each child; without the __name__ == "__main__" guard around
# top-level work, that re-import fires the segmenter again and Python's
# spawn-cascade detector raises RuntimeError. Wrapping consistently across
# tutorials also matches the style of tutorial_01.
if __name__ == "__main__":
    # %%
    TUTORIALS_DIR = Path(__file__).resolve().parent
    FITTED_MESHES_DIR = Path("D:/PhysioMotion4D/duke_data/fitted_kcl_meshes")
    PCA_MEAN_VTU = Path("D:/PhysioMotion4D/kcl-heart-pca/pca-vol-kcl/pca_mean.vtu")
    EPOCHS = 10000
    OUTPUT_DIR = TUTORIALS_DIR / "output"
    RMSE_LOG_INTERVAL = (
        500  # epochs between train/val RMSE reports and checkpoint saves
    )
    LOSS_LOG_INTERVAL = 50  # epochs between loss-only console updates
    BATCH_SIZE = 262144  # mini-batch size; full dataset lives on GPU (95 GB VRAM)
    LEARNING_RATE = 1.0e-3
    LAYER_SIZE = (
        512  # wider than single-subject model to share capacity across subjects
    )
    NUM_LAYERS = 6
    # Explicit subject lists for held-out evaluation splits.  All remaining
    # subjects are used for training.  An error is raised if any listed subject
    # is not found in FITTED_MESHES_DIR.  Set to None to skip that split entirely.
    TEST_SUBJECTS: Optional[list[str]] = ["pm0028"]
    VAL_SUBJECTS: Optional[list[str]] = ["pm0027"]
    # When True, the x/y/z input coordinates are taken from the PCA mean-shape surface
    # (same for every subject).  Each subject's PCA coefficients + stage then fully
    # describe a query; the displacement target is still relative to that subject's own
    # SSM surface (Option B).  When False, each subject's own ssm_surface.vtp coordinates
    # are used as inputs instead.
    USE_MEAN_SHAPE_COORDS = True
    LOG_LEVEL = logging.INFO
    # Leave as None to train from scratch.  Set to the path of a prior run's
    # "physicsnemo_stage_model.pt" to resume training from those weights.  When
    # resuming, a new, numbered output directory is always created so the prior
    # run's files are never modified.  Normalization stats (coordinate / PCA /
    # displacement scales) are inherited from the checkpoint so the weights remain
    # valid; only the train/val/test subject split may differ.
    RESUME_FROM_WEIGHTS: Optional[Path] = None

    def _next_output_dir(base: Path) -> Path:
        """Return the next unused sibling of *base* by appending _1, _2, ..."""
        if not base.exists():
            return base
        n = 1
        while True:
            candidate = base.parent / f"{base.name}_{n}"
            if not candidate.exists():
                return candidate
            n += 1

    def _gating_stage_from_filename(mesh_file: Path) -> float:
        """Extract the normalised cardiac stage [0, 1] from a ``g0TT`` filename stem.

        For example ``pm0002_g050_ssm_surface.vtp`` -> ``0.50``.
        """
        stem = mesh_file.stem  # e.g. "pm0002_g050_ssm_surface"
        for part in stem.split("_"):
            if part.startswith("g") and part[1:].isdigit():
                return int(part[1:]) / 100.0
        raise ValueError(f"Cannot parse gating percentage from filename: {mesh_file}")

    def _uncompiled_state_dict(model: torch.nn.Module) -> dict:
        """Return the base model's state dict, unwrapping torch.compile if applied."""
        return cast(dict, getattr(model, "_orig_mod", model).state_dict())

    def _infer_all_points(
        model: "FullyConnected",
        norm_coords: np.ndarray,
        norm_pca: np.ndarray,
        stage: float,
        displacement_scale: float,
        device: "torch.device",
    ) -> np.ndarray:
        """Run batched inference over all surface points; return raw displacements (mm)."""
        n = len(norm_coords)
        pca_tile = np.tile(norm_pca, (n, 1))
        stage_col = np.full((n, 1), stage, dtype=np.float32)
        pred_inputs = np.hstack([norm_coords, pca_tile, stage_col])
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, n, BATCH_SIZE):
                stop = min(start + BATCH_SIZE, n)
                t = torch.from_numpy(pred_inputs[start:stop].astype(np.float32)).to(
                    device
                )
                chunks.append(model(t).cpu().numpy())
        return np.vstack(chunks) * displacement_scale

    def run_tutorial() -> dict[str, Any]:
        """Train a single shared PhysicsNeMo model across all subjects and evaluate.

        Each training sample is a surface point with inputs
        ``(x, y, z, pca_c1 ... pca_cN, stage)`` and target the cardiac-motion
        displacement from that subject's SSM reference surface to the gated phase
        (Option B).  Coordinates are taken from the PCA mean shape when
        ``USE_MEAN_SHAPE_COORDS`` is True, or from each subject's own SSM surface
        otherwise.

        Returns
        -------
        dict[str, Any]
            Per-subject predicted mesh and evaluation paths, plus shared model paths.
        """
        fitted_meshes_dir = FITTED_MESHES_DIR
        pca_mean_vtu = PCA_MEAN_VTU
        epochs = EPOCHS
        rmse_log_interval = RMSE_LOG_INTERVAL
        loss_log_interval = LOSS_LOG_INTERVAL
        batch_size = BATCH_SIZE
        learning_rate = LEARNING_RATE
        layer_size = LAYER_SIZE
        num_layers = NUM_LAYERS
        test_subjects = TEST_SUBJECTS
        val_subjects = VAL_SUBJECTS
        use_mean_shape_coords = USE_MEAN_SHAPE_COORDS
        log_level = LOG_LEVEL
        resume_from_weights = RESUME_FROM_WEIGHTS

        logging.basicConfig(level=log_level)

        # In test mode, train for only a couple of epochs so the tutorial test
        # exercises the full pipeline without a long GPU run.
        if TestTools.running_as_test():
            epochs = min(epochs, 2)
            logging.info("Test mode: reducing epochs to %d", epochs)

        # When resuming, always write into a fresh numbered directory so prior
        # outputs (weights, logs, CSVs) are never overwritten.
        if resume_from_weights is not None:
            output_dir = _next_output_dir(OUTPUT_DIR)
            logging.info(
                f"Resuming from {resume_from_weights}; "
                f"new output directory: {output_dir}"
            )
        else:
            output_dir = OUTPUT_DIR
        test_output_dir = output_dir / "test_predictions"

        output_dir.mkdir(parents=True, exist_ok=True)
        test_output_dir.mkdir(parents=True, exist_ok=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ------------------------------------------------------------------ #
        # 1. Optionally load PCA mean-shape surface (shared coordinate space) #
        # ------------------------------------------------------------------ #
        mean_shape_coords: Optional[np.ndarray] = None
        if use_mean_shape_coords:
            logging.info(f"Loading PCA mean shape from {pca_mean_vtu}")
            mean_vol = pv.read(str(pca_mean_vtu))
            mean_surf = mean_vol.extract_surface(algorithm="dataset_surface")
            mean_shape_coords = np.asarray(mean_surf.points, dtype=np.float32)
            logging.info(f"Mean shape surface: {len(mean_shape_coords)} points")

        # ------------------------------------------------------------------ #
        # 2. Discover and load all subjects                                   #
        # ------------------------------------------------------------------ #
        subjects: dict[str, dict] = {}

        for subject_dir in sorted(fitted_meshes_dir.glob("pm????")):
            sid = subject_dir.name
            ref_file = subject_dir / f"{sid}_ssm_surface.vtp"
            pca_file = subject_dir / f"{sid}_ssm_pca_coefficients.json"
            mesh_files = sorted(subject_dir.glob(f"{sid}_g0*_ssm_surface.vtp"))

            missing = [p for p in (ref_file, pca_file) if not p.exists()]
            if missing or len(mesh_files) < 2:
                msg = (
                    f"Skipping {sid}: missing files {[str(p) for p in missing]}"
                    if missing
                    else f"Skipping {sid}: only {len(mesh_files)} gated phase(s) found"
                )
                logging.info(msg)
                continue

            ref_mesh = pv.read(str(ref_file))
            subjects[sid] = {
                "subject_dir": subject_dir,
                "ref_mesh": ref_mesh,
                "ref_points": np.asarray(ref_mesh.points, dtype=np.float32),
                "pca_coeffs": np.array(
                    json.loads(pca_file.read_text(encoding="utf-8")), dtype=np.float32
                ),
                "mesh_files": mesh_files,
            }

        if len(subjects) < 3:
            raise RuntimeError(
                f"Found only {len(subjects)} valid subject(s); need at least 3 for a "
                "train / val / test subject split."
            )

        n_pca = next(iter(subjects.values()))["pca_coeffs"].shape[0]
        n_mesh_points = next(iter(subjects.values()))["ref_points"].shape[0]

        if (
            use_mean_shape_coords
            and mean_shape_coords is not None
            and len(mean_shape_coords) != n_mesh_points
        ):
            raise RuntimeError(
                "Mean-shape surface topology mismatch: expected "
                f"{n_mesh_points} points but got {len(mean_shape_coords)}."
            )

        # ------------------------------------------------------------------ #
        # 3. Validate explicit subject splits and derive training set         #
        # ------------------------------------------------------------------ #
        all_sids = set(subjects.keys())

        # Normalise None -> empty list for uniform handling below.
        test_list = test_subjects if test_subjects is not None else []
        val_list = val_subjects if val_subjects is not None else []

        unknown_test = [s for s in test_list if s not in all_sids]
        unknown_val = [s for s in val_list if s not in all_sids]
        if unknown_test or unknown_val:
            parts = []
            if unknown_test:
                parts.append(f"TEST_SUBJECTS not found: {unknown_test}")
            if unknown_val:
                parts.append(f"VAL_SUBJECTS not found: {unknown_val}")
            raise ValueError(
                "Subject(s) listed in split configuration do not exist in "
                f"{fitted_meshes_dir}:\n  " + "\n  ".join(parts)
            )

        overlap = set(test_list) & set(val_list)
        if overlap:
            raise ValueError(
                f"Subject(s) appear in both TEST_SUBJECTS and VAL_SUBJECTS: {sorted(overlap)}"
            )

        test_sids: set[str] = set(test_list)
        val_sids: set[str] = set(val_list)
        train_sids: set[str] = all_sids - test_sids - val_sids

        if not train_sids:
            raise ValueError(
                "No subjects remain for training after applying TEST_SUBJECTS and VAL_SUBJECTS."
            )

        logging.info(
            f"Subject split - train: {len(train_sids)}, "
            f"val: {len(val_sids)}, test: {len(test_sids)}"
        )
        logging.info(f"  train subjects: {sorted(train_sids)}")
        logging.info(f"  val   subjects: {sorted(val_sids)}")
        logging.info(f"  test  subjects: {sorted(test_sids)}")

        # ------------------------------------------------------------------ #
        # 4. Normalisation statistics                                         #
        # When resuming, inherit stats from the prior checkpoint so that the  #
        # loaded weights remain valid.  Otherwise compute from training data. #
        # ------------------------------------------------------------------ #
        resume_ckpt: Optional[dict] = None
        if resume_from_weights is not None:
            logging.info(f"Loading prior weights from {resume_from_weights}")
            resume_ckpt = torch.load(
                str(resume_from_weights), map_location="cpu", weights_only=True
            )
            coordinate_mean = np.array(resume_ckpt["coordinate_mean"], dtype=np.float32)
            coordinate_scale = np.array(
                resume_ckpt["coordinate_scale"], dtype=np.float32
            )
            pca_mean_vec = np.array(resume_ckpt["pca_mean"], dtype=np.float32)
            pca_scale_vec = np.array(resume_ckpt["pca_scale"], dtype=np.float32)
            displacement_scale = float(resume_ckpt["displacement_scale"])
            logging.info(
                "Reusing normalization statistics from prior checkpoint "
                f"(displacement_scale={displacement_scale:.4f} mm)."
            )
        else:
            # Coordinates: use mean shape if available, else pool training ref surfaces.
            if use_mean_shape_coords and mean_shape_coords is not None:
                coord_ref = mean_shape_coords
            else:
                coord_ref = np.vstack([subjects[s]["ref_points"] for s in train_sids])
            coordinate_mean = coord_ref.mean(axis=0)
            coordinate_scale = coord_ref.std(axis=0)
            coordinate_scale = np.where(coordinate_scale == 0.0, 1.0, coordinate_scale)

            # PCA coefficients: per-dimension stats from training subjects only.
            train_pca = np.vstack([subjects[s]["pca_coeffs"] for s in train_sids])
            pca_mean_vec = train_pca.mean(axis=0)
            pca_scale_vec = train_pca.std(axis=0)
            pca_scale_vec = np.where(pca_scale_vec == 0.0, 1.0, pca_scale_vec)

        # ------------------------------------------------------------------ #
        # 5. Build combined training and validation datasets                  #
        # ------------------------------------------------------------------ #
        logging.info("Building training and validation datasets ...")
        training_inputs: list[np.ndarray] = []
        training_targets: list[np.ndarray] = []
        val_inputs: list[np.ndarray] = []
        val_targets: list[np.ndarray] = []

        def _build_rows(
            files: list[Path],
            norm_coords: np.ndarray,
            pca_tile: np.ndarray,
            ref_pts: np.ndarray,
        ) -> tuple[list[np.ndarray], list[np.ndarray]]:
            inp_rows, tgt_rows = [], []
            for mesh_file in files:
                mesh = pv.read(str(mesh_file))
                if mesh.n_points != n_mesh_points:
                    raise ValueError(
                        f"{mesh_file} has {mesh.n_points} points, expected {n_mesh_points}."
                    )
                stage = _gating_stage_from_filename(mesh_file)
                stage_col = np.full((len(norm_coords), 1), stage, dtype=np.float32)
                inp_rows.append(np.hstack([norm_coords, pca_tile, stage_col]))
                tgt_rows.append(np.asarray(mesh.points, dtype=np.float32) - ref_pts)
            return inp_rows, tgt_rows

        for sid in sorted(train_sids):
            data = subjects[sid]
            coords = (
                mean_shape_coords
                if use_mean_shape_coords and mean_shape_coords is not None
                else data["ref_points"]
            )
            norm_coords = (coords - coordinate_mean) / coordinate_scale
            norm_pca = (data["pca_coeffs"] - pca_mean_vec) / pca_scale_vec
            pca_tile = np.tile(norm_pca, (n_mesh_points, 1))
            rows_in, rows_tgt = _build_rows(
                data["mesh_files"], norm_coords, pca_tile, data["ref_points"]
            )
            training_inputs.extend(rows_in)
            training_targets.extend(rows_tgt)

        for sid in sorted(val_sids):
            data = subjects[sid]
            coords = (
                mean_shape_coords
                if use_mean_shape_coords and mean_shape_coords is not None
                else data["ref_points"]
            )
            norm_coords = (coords - coordinate_mean) / coordinate_scale
            norm_pca = (data["pca_coeffs"] - pca_mean_vec) / pca_scale_vec
            pca_tile = np.tile(norm_pca, (n_mesh_points, 1))
            rows_in, rows_tgt = _build_rows(
                data["mesh_files"], norm_coords, pca_tile, data["ref_points"]
            )
            val_inputs.extend(rows_in)
            val_targets.extend(rows_tgt)

        inputs_array = np.vstack(training_inputs).astype(np.float32)
        targets_array = np.vstack(training_targets).astype(np.float32)
        if resume_ckpt is None:
            # Fresh run: derive displacement_scale from the training targets.
            displacement_scale = float(np.max(np.abs(targets_array)))
            if displacement_scale == 0.0:
                displacement_scale = 1.0
        targets_array /= displacement_scale

        has_val = len(val_inputs) > 0
        if has_val:
            val_inputs_array = np.vstack(val_inputs).astype(np.float32)
            val_targets_array = (
                np.vstack(val_targets).astype(np.float32) / displacement_scale
            )
        else:
            logging.info("No validation subjects configured; skipping val RMSE.")

        logging.info(
            f"Training set: {len(inputs_array):,} rows, "
            f"val set: {len(val_inputs) if has_val else 0:,} rows, "
            f"in_features={inputs_array.shape[1]}, "
            f"displacement_scale={displacement_scale:.4f} mm"
        )

        # ------------------------------------------------------------------ #
        # 5. Train single shared model                                        #
        # ------------------------------------------------------------------ #
        in_features = 3 + n_pca + 1  # xyz + pca_coefficients + stage
        model = FullyConnected(
            in_features=in_features,
            layer_size=layer_size,
            out_features=3,
            num_layers=num_layers,
            activation_fn="silu",
            skip_connections=True,
        ).to(device)
        if resume_ckpt is not None:
            state = resume_ckpt.get("model_state_dict", resume_ckpt)
            model.load_state_dict(state)
            logging.info("Loaded model weights from prior checkpoint.")

        # torch.compile requires Triton, which is Linux-only; skip silently on Windows.
        if sys.platform != "win32":
            try:
                model = torch.compile(model)
                logging.info(
                    "torch.compile enabled - first epoch slower while JIT warms up."
                )
            except Exception as _compile_err:
                logging.info(
                    f"torch.compile skipped ({_compile_err}); running in eager mode."
                )
        else:
            logging.info("torch.compile skipped on Windows (Triton unavailable).")

        # Dataset (~3.5 GB) fits comfortably in the 95 GB GPU; keep it resident on GPU
        # to eliminate per-batch CPU->GPU transfers (the primary utilization bottleneck).
        logging.info("Moving training and validation tensors to GPU ...")
        inputs_tensor = torch.from_numpy(inputs_array).to(device)
        targets_tensor = torch.from_numpy(targets_array).to(device)
        if has_val:
            val_inputs_tensor = torch.from_numpy(val_inputs_array).to(device)
            val_targets_tensor = torch.from_numpy(val_targets_array).to(device)
        n_train = len(inputs_tensor)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = torch.nn.MSELoss()

        def _batched_rmse_mm(inputs: torch.Tensor, targets: torch.Tensor) -> float:
            """Euclidean RMSE in mm, computed in batches (tensors already on GPU)."""
            total_sq = 0.0
            n_total = 0
            with torch.no_grad():
                for start in range(0, len(inputs), batch_size):
                    stop = min(start + batch_size, len(inputs))
                    err_mm = (
                        model(inputs[start:stop]) - targets[start:stop]
                    ) * displacement_scale
                    total_sq += float(torch.sum(err_mm**2))
                    n_total += stop - start
            return float(np.sqrt(total_sq / n_total))

        losses: list[float] = []
        rmse_log: list[dict] = []

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            # Shuffle indices on GPU - avoids any CPU involvement mid-epoch.
            perm = torch.randperm(n_train, device=device)
            for start in range(0, n_train, batch_size):
                idx = perm[start : start + batch_size]
                batch_in = inputs_tensor[idx]
                batch_tgt = targets_tensor[idx]
                optimizer.zero_grad(set_to_none=True)
                # BF16 autocast: Blackwell tensor cores deliver ~2x FP32 throughput on BF16.
                # No GradScaler needed - BF16 exponent range is wide enough to skip scaling.
                with torch.amp.autocast(device.type, dtype=torch.bfloat16):
                    loss = loss_fn(model(batch_in), batch_tgt)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach()) * len(batch_in)
            losses.append(epoch_loss / n_train)

            if (epoch + 1) % loss_log_interval == 0:
                logging.info(
                    "  epoch %05d/%d  loss=%.6f", epoch + 1, epochs, losses[-1]
                )

            if (epoch + 1) % rmse_log_interval == 0 or epoch + 1 == epochs:
                model.eval()
                train_rmse = _batched_rmse_mm(inputs_tensor, targets_tensor)
                val_rmse = (
                    _batched_rmse_mm(val_inputs_tensor, val_targets_tensor)
                    if has_val
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
                    output_dir / f"physicsnemo_stage_model_epoch_{epoch + 1:05d}.pt"
                )
                torch.save(_uncompiled_state_dict(model), ckpt_path)
                logging.info(
                    "INTERMITTENT TEST  epoch %05d/%d  "
                    "train RMSE=%.4f mm  val RMSE=%.4f mm  checkpoint=%s",
                    epoch + 1,
                    epochs,
                    train_rmse,
                    val_rmse,
                    ckpt_path.name,
                )
        model.eval()

        # ------------------------------------------------------------------ #
        # 6. Save shared model weights + metadata                             #
        # ------------------------------------------------------------------ #
        checkpoint_file = output_dir / "physicsnemo_stage_model.pt"
        metadata_file = output_dir / "physicsnemo_stage_model_metadata.json"
        losses_file = output_dir / "training_losses.json"

        torch.save(
            {
                "model_state_dict": _uncompiled_state_dict(model),
                "in_features": in_features,
                "layer_size": layer_size,
                "num_layers": num_layers,
                "coordinate_mean": coordinate_mean.tolist(),
                "coordinate_scale": coordinate_scale.tolist(),
                "pca_mean": pca_mean_vec.tolist(),
                "pca_scale": pca_scale_vec.tolist(),
                "displacement_scale": displacement_scale,
                "use_mean_shape_coords": use_mean_shape_coords,
                "pca_mean_vtu": str(pca_mean_vtu) if use_mean_shape_coords else None,
                "train_subject_ids": sorted(train_sids),
                "val_subject_ids": sorted(val_sids),
                "test_subject_ids": sorted(test_sids),
                "resumed_from": str(resume_from_weights)
                if resume_from_weights
                else None,
            },
            checkpoint_file,
        )
        input_feature_names = (
            (
                ["mean_shape_x", "mean_shape_y", "mean_shape_z"]
                if use_mean_shape_coords
                else ["ssm_x", "ssm_y", "ssm_z"]
            )
            + [f"pca_c{i + 1}" for i in range(n_pca)]
            + ["stage"]
        )
        metadata_file.write_text(
            json.dumps(
                {
                    "architecture": "physicsnemo.models.mlp.FullyConnected",
                    "input_features": input_feature_names,
                    "output_features": ["dx", "dy", "dz"],
                    "n_subjects": len(subjects),
                    "in_features": in_features,
                    "layer_size": layer_size,
                    "num_layers": num_layers,
                    "epochs": epochs,
                    "n_mesh_points": n_mesh_points,
                    "learning_rate": learning_rate,
                    "coordinate_mean": coordinate_mean.tolist(),
                    "coordinate_scale": coordinate_scale.tolist(),
                    "pca_mean": pca_mean_vec.tolist(),
                    "pca_scale": pca_scale_vec.tolist(),
                    "displacement_scale": displacement_scale,
                    "use_mean_shape_coords": use_mean_shape_coords,
                    "displacement_convention": "Option B: relative to subject ssm_surface",
                    "resumed_from": str(resume_from_weights)
                    if resume_from_weights
                    else None,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        losses_file.write_text(json.dumps(losses, indent=2), encoding="utf-8")
        rmse_log_file = output_dir / "training_validation_rmse.json"
        rmse_log_file.write_text(json.dumps(rmse_log, indent=2), encoding="utf-8")
        rmse_csv_file = output_dir / "training_validation_rmse.csv"
        with rmse_csv_file.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["epoch", "train_rmse_mm", "val_rmse_mm"]
            )
            writer.writeheader()
            writer.writerows(rmse_log)

        # Copy PCA assets so the output directory is self-contained for replay.
        pca_src_dir = pca_mean_vtu.parent
        shutil.copy2(pca_mean_vtu, output_dir / pca_mean_vtu.name)
        pca_model_src = pca_src_dir / "pca_model.json"
        if pca_model_src.exists():
            shutil.copy2(pca_model_src, output_dir / pca_model_src.name)
        # Save the extracted mean-shape surface alongside the volume mesh.
        if use_mean_shape_coords and mean_shape_coords is not None:
            mean_surf.save(str(output_dir / "pca_mean_surface.vtp"))

        logging.info(f"Shared model saved to {checkpoint_file}")

        # ------------------------------------------------------------------ #
        # ------------------------------------------------------------------ #
        # 7. Evaluate test and val subjects: all phases -> output/pm00XX/    #
        # ------------------------------------------------------------------ #
        def _evaluate_subject(sid: str, split_label: str) -> tuple[list[dict], Path]:
            data = subjects[sid]
            coords = (
                mean_shape_coords
                if use_mean_shape_coords and mean_shape_coords is not None
                else data["ref_points"]
            )
            norm_coords_full = (coords - coordinate_mean) / coordinate_scale
            norm_pca = (data["pca_coeffs"] - pca_mean_vec) / pca_scale_vec

            subj_out_dir = output_dir / sid
            subj_out_dir.mkdir(parents=True, exist_ok=True)

            sq_err_sum = np.zeros(n_mesh_points, dtype=np.float64)
            stats = []

            for phase_file in data["mesh_files"]:
                stage = _gating_stage_from_filename(phase_file)
                pred_disps = _infer_all_points(
                    model, norm_coords_full, norm_pca, stage, displacement_scale, device
                )
                pred_points = data["ref_points"] + pred_disps

                pred_mesh = data["ref_mesh"].copy(deep=True)
                pred_mesh.points = pred_points
                gating_tag = phase_file.stem.split("_ssm_surface")[0].split("_")[-1]
                pred_mesh.save(
                    subj_out_dir / f"{sid}_{gating_tag}_ssm_surface_pred.vtp"
                )

                actual_points = np.asarray(
                    pv.read(str(phase_file)).points, dtype=np.float32
                )
                errors = pred_points - actual_points
                euclidean = np.linalg.norm(errors, axis=1)
                sq_err_sum += euclidean.astype(np.float64) ** 2
                stats.append(
                    {
                        "subject_id": sid,
                        "split": split_label,
                        "gating_tag": gating_tag,
                        "stage": stage,
                        "n_points": len(euclidean),
                        "mean_error_mm": float(euclidean.mean()),
                        "median_error_mm": float(np.median(euclidean)),
                        "max_error_mm": float(euclidean.max()),
                        "rms_error_mm": float(np.sqrt(np.mean(euclidean**2))),
                        "std_error_mm": float(euclidean.std()),
                        "mean_abs_error_x_mm": float(np.abs(errors[:, 0]).mean()),
                        "mean_abs_error_y_mm": float(np.abs(errors[:, 1]).mean()),
                        "mean_abs_error_z_mm": float(np.abs(errors[:, 2]).mean()),
                    }
                )
                logging.info(
                    f"{sid} [{split_label}] {gating_tag}: "
                    f"mean={stats[-1]['mean_error_mm']:.3f} mm  "
                    f"max={stats[-1]['max_error_mm']:.3f} mm"
                )

            point_rmse = np.sqrt(sq_err_sum / len(data["mesh_files"])).astype(
                np.float32
            )
            rmse_mesh = data["ref_mesh"].copy(deep=True)
            rmse_mesh.point_data["RMSE_mm"] = point_rmse
            rmse_file = subj_out_dir / f"{sid}_ssm_surface_rmse.vtp"
            rmse_mesh.save(rmse_file)
            logging.info(
                f"{sid} [{split_label}] per-point RMSE: "
                f"mean={point_rmse.mean():.3f} mm  max={point_rmse.max():.3f} mm"
            )
            return stats, rmse_file

        all_stats = []
        tutorial_outputs = {}

        for sid in sorted(test_sids):
            stats, rmse_file = _evaluate_subject(sid, "test")
            all_stats.extend(stats)
            tutorial_outputs[sid] = {
                "split": "test",
                "rmse_file": rmse_file,
                "final_loss": losses[-1],
                "n_phases": len(subjects[sid]["mesh_files"]),
            }

        for sid in sorted(val_sids):
            stats, rmse_file = _evaluate_subject(sid, "val")
            all_stats.extend(stats)
            tutorial_outputs[sid] = {
                "split": "val",
                "rmse_file": rmse_file,
                "final_loss": losses[-1],
                "n_phases": len(subjects[sid]["mesh_files"]),
            }

        # ------------------------------------------------------------------ #
        # 8. Write CSV statistics (split column distinguishes test vs val)    #
        # ------------------------------------------------------------------ #
        if all_stats:
            per_phase_csv = test_output_dir / "statistics_per_phase.csv"
            with per_phase_csv.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(all_stats[0].keys()))
                writer.writeheader()
                writer.writerows(all_stats)
            logging.info(f"Per-phase statistics -> {per_phase_csv}")

            subject_rows = defaultdict(list)
            for row in all_stats:
                subject_rows[row["subject_id"]].append(row)

            summary_rows = [
                {
                    "subject_id": sid,
                    "split": rows[0]["split"],
                    "n_phases": len(rows),
                    "mean_error_mm": float(np.mean([r["mean_error_mm"] for r in rows])),
                    "mean_max_error_mm": float(
                        np.mean([r["max_error_mm"] for r in rows])
                    ),
                    "overall_max_error_mm": float(
                        np.max([r["max_error_mm"] for r in rows])
                    ),
                    "mean_rms_error_mm": float(
                        np.mean([r["rms_error_mm"] for r in rows])
                    ),
                }
                for sid, rows in sorted(subject_rows.items())
            ]
            summary_csv = test_output_dir / "statistics_summary.csv"
            with summary_csv.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
                writer.writeheader()
                writer.writerows(summary_rows)
            logging.info(f"Summary statistics -> {summary_csv}")

        return tutorial_outputs

    # %%
    # Run this cell in VS Code or Cursor:
    tutorial_results = run_tutorial()
