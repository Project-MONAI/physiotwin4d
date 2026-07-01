"""
Tutorial 9a (MGN): MeshGraphNet model for cardiac mesh stage prediction.

Second stage of the cardiac 4D deep-learning pipeline (Tutorials 8 -> 9 -> 10).
Drop-in companion to the MLP trainer Tutorial 9b
(``tutorial_09b_cardiac_train_physicsnemo_mlp.py``).  It uses the same
per-time-point SSM surfaces created by Tutorial 8
(``tutorial_08_cardiac_fit_model.py``) and the same Option B displacement
convention, but replaces the FullyConnected MLP with a PhysicsNeMo MeshGraphNet
that explicitly exploits the surface mesh topology via message passing between
neighbouring vertices.  Outputs land in ``OUTPUT_DIR`` (``output_mgn/``) so the
two models can be evaluated and compared side by side.  Evaluate the trained
model with Tutorial 10a (``tutorial_10a_cardiac_eval_physicsnemo_mgn.py``).

Why a GNN?
----------
The SSM mesh has a fixed, consistent topology across all subjects.  Cardiac tissue is a
physical continuum - adjacent vertices co-vary smoothly.  The MLP trainer must infer
this from xyz coordinates alone.  MeshGraphNet encodes the prior directly by passing
messages along mesh edges, giving the model an explicit continuum-deformation inductive
bias.

Node features (per vertex):   [norm_x, norm_y, norm_z, pca_c1 ... pca_cN, stage]
Edge features (per edge):     [rel_x, rel_y, rel_z, distance]   (derived from mean shape)
Output (per vertex):          [dx, dy, dz]  (displacement in mm, after rescaling)

The edge topology is extracted once from the mean-shape surface and shared across every
(subject, phase) sample.  PyTorch Geometric's ``Batch.from_data_list`` handles
mini-batching by joining disconnected sub-graphs.  The shared graph topology
(``shared_edge_index.pt`` / ``shared_edge_features.pt``) is saved alongside the weights
so Tutorial 10a (``tutorial_10a_cardiac_eval_physicsnemo_mgn.py``) can replay it
at inference time.

Bring Your Own Data
-------------------
This is a bring-your-own-data tutorial: the path constants below point at a local
``D:/PhysioMotion4D/`` layout produced by Tutorial 8, not at the repository
``data/`` directory.  Edit them to match your own data location.

Data Required
-------------
Run Tutorial 8 (``tutorial_08_cardiac_fit_model.py``) first (same requirement as
the MLP trainer).

Extra Install Required
----------------------
PhysicsNeMo and PyTorch Geometric must be installed::

    pip install "physiomotion4d[physicsnemo]"
    pip install torch-geometric

``torch_scatter`` (a PyTorch Geometric backend) must be built from source when using a
custom NVIDIA PyTorch build because no matching pre-built wheel exists on data.pyg.org::

    pip install torch-scatter --no-build-isolation
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
    import torch_geometric.utils as pyg_utils
    from torch_geometric.data import Batch, Data

    from physicsnemo.models.meshgraphnet import MeshGraphNet
except ImportError as exc:
    raise ImportError(
        "Tutorial 9a requires PhysicsNeMo and PyTorch Geometric. Install with:\n"
        '  pip install "physiomotion4d[physicsnemo]"\n'
        "  pip install torch-geometric"
    ) from exc


if __name__ == "__main__":
    # %%
    TUTORIALS_DIR = Path(__file__).resolve().parent
    FITTED_MESHES_DIR = Path("D:/PhysioMotion4D/duke_data/fitted_kcl_meshes")
    PCA_MEAN_VTU = Path("D:/PhysioMotion4D/kcl-heart-pca/pca-vol-kcl/pca_mean.vtu")
    EPOCHS = 1500
    OUTPUT_DIR = TUTORIALS_DIR / "output_mgn"
    RMSE_LOG_INTERVAL = 100
    LOSS_LOG_INTERVAL = (
        1  # print every epoch so we can measure per-epoch time immediately
    )
    # Mini-batch size in (subject, phase) *graphs*.
    # concat_efeat stores (BxE, 3H) FP32 per processor step x PROCESSOR_SIZE steps:
    #   PROCESSOR_SIZE=3, B=4, H=128: 3 x 4M x 384 x 4 = 18.4 GB  -> safe, good GPU util
    #   PROCESSOR_SIZE=10, B=2, H=128: 10 x 2M x 384 x 4 = 30.7 GB -> safe but slow (83 h)
    # 3 hops is sufficient for local mesh-continuity; 10 hops adds marginal benefit here.
    BATCH_SIZE_GRAPHS = 4
    LEARNING_RATE = 1.0e-3
    # MeshGraphNet hyper-parameters
    PROCESSOR_SIZE = 3  # 3 message-passing hops: enough for surface continuity,
    # ~3x faster than 10; 10K epochs ~ 8 h vs 83 h
    HIDDEN_DIM = 128  # sufficient capacity for 27 training subjects
    # Gradient checkpointing (0 = disabled).
    # checkpointing=5 caused the training loop to stall: Batch.from_data_list was called
    # 380K times (38 batches x 10K epochs), each round-tripping 8M edges CPU<->GPU.
    # With B=2 the full activation storage fits without checkpointing.
    NUM_PROCESSOR_CHECKPOINT_SEGMENTS = 0
    NUM_LAYERS_PROCESSOR = 2  # MLP layers inside each processor step
    NUM_LAYERS_ENCODER = 2
    NUM_LAYERS_DECODER = 2

    TEST_SUBJECTS: Optional[list[str]] = ["pm0028"]
    VAL_SUBJECTS: Optional[list[str]] = ["pm0027"]
    USE_MEAN_SHAPE_COORDS = True
    LOG_LEVEL = logging.INFO
    RESUME_FROM_WEIGHTS: Optional[Path] = None

    # ---------------------------------------------------------------------- #

    def _next_output_dir(base: Path) -> Path:
        if not base.exists():
            return base
        n = 1
        while True:
            candidate = base.parent / f"{base.name}_{n}"
            if not candidate.exists():
                return candidate
            n += 1

    def _gating_stage_from_filename(mesh_file: Path) -> float:
        stem = mesh_file.stem
        for part in stem.split("_"):
            if part.startswith("g") and part[1:].isdigit():
                return int(part[1:]) / 100.0
        raise ValueError(f"Cannot parse gating percentage from filename: {mesh_file}")

    def _uncompiled_state_dict(model: torch.nn.Module) -> dict:
        """Return the base model's state dict, unwrapping torch.compile if applied."""
        return cast(dict, getattr(model, "_orig_mod", model).state_dict())

    def _mesh_to_edge_index(poly: pv.PolyData) -> torch.Tensor:
        """Extract undirected edge_index from triangulated PyVista PolyData faces."""
        faces = poly.faces.reshape(-1, 4)[:, 1:]  # (F, 3) - strip leading count
        src = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
        dst = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
        edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
        return cast(torch.Tensor, pyg_utils.to_undirected(edge_index))

    def _compute_edge_features(
        coords: np.ndarray, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Build (N_edges, 4) edge feature tensor: [rel_x, rel_y, rel_z, distance]."""
        ei = edge_index.numpy()
        disp = coords[ei[1]] - coords[ei[0]]  # (N_edges, 3)
        dist = np.linalg.norm(disp, axis=1, keepdims=True)  # (N_edges, 1)
        return torch.tensor(np.hstack([disp, dist]), dtype=torch.float32)

    def _batched_rmse_mm(
        model: MeshGraphNet,
        node_feats_gpu: torch.Tensor,  # (N_samples, n_mesh_points, in_features)
        targets_gpu: torch.Tensor,  # (N_samples, n_mesh_points, 3)
        full_batch_graph: "Data",
        full_edge_feats: torch.Tensor,
        partial_batch_graph: "Data",
        partial_edge_feats: torch.Tensor,
        displacement_scale: float,
        batch_size: int,
        n_mesh_points: int,
        in_features: int,
    ) -> float:
        """Euclidean RMSE in mm over pre-stacked GPU tensors (no CPU transfers)."""
        n_samples = node_feats_gpu.shape[0]
        total_sq = 0.0
        n_total = 0
        with torch.no_grad():
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                b = end - start
                nf = node_feats_gpu[start:end].reshape(b * n_mesh_points, in_features)
                tgt = targets_gpu[start:end].reshape(b * n_mesh_points, 3)
                bg = full_batch_graph if b == batch_size else partial_batch_graph
                ef = full_edge_feats if b == batch_size else partial_edge_feats
                pred = model(nf, ef, bg)
                err_mm = (pred - tgt) * displacement_scale
                total_sq += float(torch.sum(err_mm**2))
                n_total += b * n_mesh_points
        return float(np.sqrt(total_sq / n_total))

    def _infer_all_points(
        model: MeshGraphNet,
        norm_coords: np.ndarray,
        norm_pca: np.ndarray,
        stage: float,
        shared_graph: Data,
        shared_edge_feats: torch.Tensor,
        displacement_scale: float,
        device: torch.device,
    ) -> np.ndarray:
        """Run inference for a single (subject, phase) sample; return displacements (mm)."""
        n = len(norm_coords)
        pca_tile = np.tile(norm_pca, (n, 1))
        stage_col = np.full((n, 1), stage, dtype=np.float32)
        node_feats = torch.tensor(
            np.hstack([norm_coords, pca_tile, stage_col]), dtype=torch.float32
        ).to(device)
        graph = shared_graph.clone().to(device)
        edge_feats = shared_edge_feats.to(device)
        with torch.no_grad():
            pred = model(node_feats, edge_feats, graph)
        return np.asarray(pred.cpu().numpy()) * displacement_scale

    def run_tutorial() -> dict[str, Any]:
        """Train a MeshGraphNet model across all subjects and evaluate.

        Same input/output convention as the MLP trainer (Option B displacements
        relative to each subject's SSM reference surface) so results are directly
        comparable.
        """
        fitted_meshes_dir = FITTED_MESHES_DIR
        pca_mean_vtu = PCA_MEAN_VTU
        epochs = EPOCHS
        rmse_log_interval = RMSE_LOG_INTERVAL
        loss_log_interval = LOSS_LOG_INTERVAL
        batch_size_graphs = BATCH_SIZE_GRAPHS
        learning_rate = LEARNING_RATE
        test_subjects = TEST_SUBJECTS
        val_subjects = VAL_SUBJECTS
        use_mean_shape_coords = USE_MEAN_SHAPE_COORDS
        resume_from_weights = RESUME_FROM_WEIGHTS

        logging.basicConfig(level=LOG_LEVEL)

        # In test mode, train for only a couple of epochs so the tutorial test
        # exercises the full pipeline without a long GPU run.
        if TestTools.running_as_test():
            epochs = min(epochs, 2)
            logging.info("Test mode: reducing epochs to %d", epochs)

        if resume_from_weights is not None:
            output_dir = _next_output_dir(OUTPUT_DIR)
            logging.info(f"Resuming from {resume_from_weights}; output: {output_dir}")
        else:
            output_dir = OUTPUT_DIR
        test_output_dir = output_dir / "test_predictions"
        output_dir.mkdir(parents=True, exist_ok=True)
        test_output_dir.mkdir(parents=True, exist_ok=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ------------------------------------------------------------------ #
        # 1. Load PCA mean-shape surface (shared coordinate and topology)     #
        # ------------------------------------------------------------------ #
        logging.info(f"Loading PCA mean shape from {pca_mean_vtu}")
        mean_vol = pv.read(str(pca_mean_vtu))
        mean_surf: pv.PolyData = mean_vol.extract_surface(algorithm="dataset_surface")
        mean_shape_coords = np.asarray(mean_surf.points, dtype=np.float32)
        logging.info(
            f"Mean shape surface: {len(mean_shape_coords)} points, "
            f"{mean_surf.n_faces} faces"
        )

        # ------------------------------------------------------------------ #
        # 2. Build shared graph topology and edge features (from mean shape)  #
        # ------------------------------------------------------------------ #
        logging.info("Building shared mesh graph from mean-shape faces ...")
        shared_edge_index = _mesh_to_edge_index(mean_surf)
        shared_edge_feats = _compute_edge_features(mean_shape_coords, shared_edge_index)
        n_mesh_edges = shared_edge_index.shape[1]
        logging.info(
            f"Graph: {len(mean_shape_coords)} nodes, {n_mesh_edges} edges "
            f"(~{n_mesh_edges / mean_surf.n_faces:.1f}x n_faces - expected ~6 for triangles)"
        )
        # shared_graph holds connectivity only; node/edge features passed separately
        shared_graph = Data(
            edge_index=shared_edge_index,
            num_nodes=len(mean_shape_coords),
        )

        # ------------------------------------------------------------------ #
        # 3. Discover and load all subjects                                   #
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
                f"Found only {len(subjects)} valid subject(s); need at least 3."
            )

        n_pca = next(iter(subjects.values()))["pca_coeffs"].shape[0]
        n_mesh_points = next(iter(subjects.values()))["ref_points"].shape[0]

        if n_mesh_points != len(mean_shape_coords):
            raise RuntimeError(
                f"SSM surfaces have {n_mesh_points} points but mean shape has "
                f"{len(mean_shape_coords)} - topology mismatch."
            )

        # ------------------------------------------------------------------ #
        # 4. Validate splits                                                  #
        # ------------------------------------------------------------------ #
        all_sids = set(subjects.keys())
        test_list = test_subjects if test_subjects is not None else []
        val_list = val_subjects if val_subjects is not None else []

        unknown = [s for s in test_list + val_list if s not in all_sids]
        if unknown:
            raise ValueError(
                f"Split subjects not found in {fitted_meshes_dir}: {unknown}"
            )
        overlap = set(test_list) & set(val_list)
        if overlap:
            raise ValueError(f"Subjects in both TEST and VAL splits: {sorted(overlap)}")

        test_sids: set[str] = set(test_list)
        val_sids: set[str] = set(val_list)
        train_sids: set[str] = all_sids - test_sids - val_sids
        if not train_sids:
            raise ValueError("No subjects remain for training.")

        logging.info(
            f"Subject split - train: {len(train_sids)}, "
            f"val: {len(val_sids)}, test: {len(test_sids)}"
        )

        # ------------------------------------------------------------------ #
        # 5. Normalisation statistics                                         #
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
        else:
            coord_ref = (
                mean_shape_coords
                if use_mean_shape_coords
                else np.vstack([subjects[s]["ref_points"] for s in train_sids])
            )
            coordinate_mean = coord_ref.mean(axis=0)
            coordinate_scale = coord_ref.std(axis=0)
            coordinate_scale = np.where(coordinate_scale == 0.0, 1.0, coordinate_scale)

            train_pca = np.vstack([subjects[s]["pca_coeffs"] for s in train_sids])
            pca_mean_vec = train_pca.mean(axis=0)
            pca_scale_vec = train_pca.std(axis=0)
            pca_scale_vec = np.where(pca_scale_vec == 0.0, 1.0, pca_scale_vec)

        # ------------------------------------------------------------------ #
        # 6. Build sample lists: one entry per (subject, phase) graph         #
        # ------------------------------------------------------------------ #
        logging.info("Building per-sample graph feature lists ...")

        def _build_samples(
            sids: set[str],
        ) -> list[tuple[torch.Tensor, torch.Tensor]]:
            """Return list of (node_feats, targets) tensors for each (subject, phase)."""
            samples: list[tuple[torch.Tensor, torch.Tensor]] = []
            for sid in sorted(sids):
                data = subjects[sid]
                coords = (
                    mean_shape_coords if use_mean_shape_coords else data["ref_points"]
                )
                norm_coords = (coords - coordinate_mean) / coordinate_scale
                norm_pca = (data["pca_coeffs"] - pca_mean_vec) / pca_scale_vec
                pca_tile = np.tile(norm_pca, (n_mesh_points, 1))
                for mesh_file in data["mesh_files"]:
                    mesh = pv.read(str(mesh_file))
                    if mesh.n_points != n_mesh_points:
                        raise ValueError(
                            f"{mesh_file} has {mesh.n_points} points, "
                            f"expected {n_mesh_points}."
                        )
                    stage = _gating_stage_from_filename(mesh_file)
                    stage_col = np.full((n_mesh_points, 1), stage, dtype=np.float32)
                    node_feats = torch.tensor(
                        np.hstack([norm_coords, pca_tile, stage_col]),
                        dtype=torch.float32,
                    )
                    targets_raw = (
                        np.asarray(mesh.points, dtype=np.float32) - data["ref_points"]
                    )
                    samples.append(
                        (node_feats, torch.tensor(targets_raw, dtype=torch.float32))
                    )
            return samples

        train_samples = _build_samples(train_sids)
        val_samples = _build_samples(val_sids)

        # Derive displacement_scale from training targets (or inherit from checkpoint).
        if resume_ckpt is None:
            all_targets = torch.cat([s[1] for s in train_samples])
            displacement_scale = float(torch.max(torch.abs(all_targets)))
            if displacement_scale == 0.0:
                displacement_scale = 1.0

        # Normalise targets in-place.
        train_samples = [(n, t / displacement_scale) for n, t in train_samples]
        val_samples = [(n, t / displacement_scale) for n, t in val_samples]

        in_features = 3 + n_pca + 1
        logging.info(
            f"Training samples: {len(train_samples)}, val: {len(val_samples)}, "
            f"in_features={in_features}, displacement_scale={displacement_scale:.4f} mm"
        )

        # ------------------------------------------------------------------ #
        # 7. Model                                                            #
        # ------------------------------------------------------------------ #
        model = MeshGraphNet(
            input_dim_nodes=in_features,
            input_dim_edges=4,  # rel_x, rel_y, rel_z, distance
            output_dim=3,
            processor_size=PROCESSOR_SIZE,
            hidden_dim_processor=HIDDEN_DIM,
            hidden_dim_node_encoder=HIDDEN_DIM,
            num_layers_node_encoder=NUM_LAYERS_ENCODER,
            hidden_dim_node_decoder=HIDDEN_DIM,
            num_layers_node_decoder=NUM_LAYERS_DECODER,
            hidden_dim_edge_encoder=HIDDEN_DIM,
            num_layers_edge_processor=NUM_LAYERS_PROCESSOR,
            num_layers_node_processor=NUM_LAYERS_PROCESSOR,
            aggregation="mean",
            num_processor_checkpoint_segments=NUM_PROCESSOR_CHECKPOINT_SEGMENTS,
        ).to(device)

        if resume_ckpt is not None:
            state = resume_ckpt.get("model_state_dict", resume_ckpt)
            model.load_state_dict(state)
            logging.info("Loaded model weights from prior checkpoint.")

        # torch.compile: Linux-only (Triton unavailable on Windows).
        if sys.platform != "win32":
            try:
                model = torch.compile(model)
                logging.info("torch.compile enabled.")
            except Exception as _e:
                logging.info(f"torch.compile skipped ({_e}).")
        else:
            logging.info("torch.compile skipped on Windows.")

        # Move shared graph tensors to GPU once (used for single-sample inference).
        shared_edge_feats_gpu = shared_edge_feats.to(device)
        shared_graph_gpu = shared_graph.clone().to(device)

        # Pre-stack all training/val tensors onto GPU (~5 GB total - fits easily).
        # Eliminates per-batch CPU->GPU transfers that were a primary stall source.
        logging.info("Pre-stacking training and validation tensors onto GPU ...")
        train_node_feats_gpu = torch.stack([s[0] for s in train_samples]).to(device)
        train_targets_gpu = torch.stack([s[1] for s in train_samples]).to(device)
        n_val = len(val_samples)
        has_val = n_val > 0
        if has_val:
            val_node_feats_gpu = torch.stack([s[0] for s in val_samples]).to(device)
            val_targets_gpu = torch.stack([s[1] for s in val_samples]).to(device)
        else:
            logging.info("No validation subjects configured; skipping val RMSE.")
        n_train = len(train_samples)

        # Pre-build batched graph and edge features for the full-batch size (and for any
        # partial last batch).  All samples share the same mesh topology, so these can be
        # built once and reused every step of every epoch - the previous code rebuilt them
        # ~380 K times (38 batches x 10 K epochs), each involving a costly CPU/GPU round
        # trip across 8 M edge indices.
        logging.info(
            "Pre-building batched graph and edge features (built once, reused every step) ..."
        )
        full_batch_graph = Batch.from_data_list([shared_graph] * batch_size_graphs).to(
            device
        )
        full_edge_feats = shared_edge_feats.repeat(batch_size_graphs, 1).to(device)
        n_partial = n_train % batch_size_graphs
        if n_partial > 0:
            partial_batch_graph = Batch.from_data_list([shared_graph] * n_partial).to(
                device
            )
            partial_edge_feats = shared_edge_feats.repeat(n_partial, 1).to(device)
        else:
            partial_batch_graph = full_batch_graph
            partial_edge_feats = full_edge_feats
        if has_val:
            n_val_partial = n_val % batch_size_graphs
            if n_val_partial > 0:
                val_partial_batch_graph = Batch.from_data_list(
                    [shared_graph] * n_val_partial
                ).to(device)
                val_partial_edge_feats = shared_edge_feats.repeat(n_val_partial, 1).to(
                    device
                )
            else:
                val_partial_batch_graph = full_batch_graph
                val_partial_edge_feats = full_edge_feats

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = torch.nn.MSELoss()

        losses: list[float] = []
        rmse_log: list[dict] = []

        import time as _time

        # ------------------------------------------------------------------ #
        # 8. Training loop                                                    #
        # ------------------------------------------------------------------ #
        for epoch in range(epochs):
            _t0 = _time.perf_counter()
            model.train()
            epoch_loss = torch.zeros(
                (), device=device
            )  # accumulate on GPU, one sync/epoch
            # Shuffle indices on GPU - no CPU involvement.
            perm = torch.randperm(n_train, device=device)

            for start in range(0, n_train, batch_size_graphs):
                idx = perm[start : start + batch_size_graphs]
                b = int(idx.shape[0])

                # Index pre-stacked GPU tensors; reshape is a view (no copy).
                node_feats = train_node_feats_gpu[idx].reshape(
                    b * n_mesh_points, in_features
                )
                targets = train_targets_gpu[idx].reshape(b * n_mesh_points, 3)
                batch_graph = (
                    full_batch_graph if b == batch_size_graphs else partial_batch_graph
                )
                edge_feats = (
                    full_edge_feats if b == batch_size_graphs else partial_edge_feats
                )

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device.type, dtype=torch.bfloat16):
                    pred = model(node_feats, edge_feats, batch_graph)
                    loss = loss_fn(pred, targets)
                loss.backward()
                optimizer.step()
                epoch_loss = epoch_loss + loss.detach() * (b * n_mesh_points)

            losses.append(
                float(epoch_loss / (n_train * n_mesh_points))
            )  # one GPU sync/epoch

            _epoch_s = _time.perf_counter() - _t0
            if (epoch + 1) % loss_log_interval == 0:
                logging.info(
                    "  epoch %05d/%d  loss=%.6f  %.1fs/epoch  ETA %.1fh",
                    epoch + 1,
                    epochs,
                    losses[-1],
                    _epoch_s,
                    _epoch_s * (epochs - epoch - 1) / 3600,
                )

            if (epoch + 1) % rmse_log_interval == 0 or epoch + 1 == epochs:
                model.eval()
                train_rmse = _batched_rmse_mm(
                    model,
                    train_node_feats_gpu,
                    train_targets_gpu,
                    full_batch_graph,
                    full_edge_feats,
                    partial_batch_graph,
                    partial_edge_feats,
                    displacement_scale,
                    batch_size_graphs,
                    n_mesh_points,
                    in_features,
                )
                val_rmse = (
                    _batched_rmse_mm(
                        model,
                        val_node_feats_gpu,
                        val_targets_gpu,
                        full_batch_graph,
                        full_edge_feats,
                        val_partial_batch_graph,
                        val_partial_edge_feats,
                        displacement_scale,
                        batch_size_graphs,
                        n_mesh_points,
                        in_features,
                    )
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
                ckpt_path = output_dir / f"mgn_stage_model_epoch_{epoch + 1:05d}.pt"
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
        # 9. Save model weights + metadata                                    #
        # ------------------------------------------------------------------ #
        checkpoint_file = output_dir / "mgn_stage_model.pt"
        metadata_file = output_dir / "mgn_stage_model_metadata.json"

        torch.save(
            {
                "model_state_dict": _uncompiled_state_dict(model),
                "in_features": in_features,
                "processor_size": PROCESSOR_SIZE,
                "hidden_dim": HIDDEN_DIM,
                "coordinate_mean": coordinate_mean.tolist(),
                "coordinate_scale": coordinate_scale.tolist(),
                "pca_mean": pca_mean_vec.tolist(),
                "pca_scale": pca_scale_vec.tolist(),
                "displacement_scale": displacement_scale,
                "use_mean_shape_coords": use_mean_shape_coords,
                "train_subject_ids": sorted(train_sids),
                "val_subject_ids": sorted(val_sids),
                "test_subject_ids": sorted(test_sids),
                "resumed_from": str(resume_from_weights)
                if resume_from_weights
                else None,
            },
            checkpoint_file,
        )
        # Save shared graph topology for inference replay.
        torch.save(shared_edge_index, output_dir / "shared_edge_index.pt")
        torch.save(shared_edge_feats, output_dir / "shared_edge_features.pt")

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
                    "architecture": "physicsnemo.models.meshgraphnet.MeshGraphNet",
                    "input_node_features": input_feature_names,
                    "input_edge_features": ["rel_x", "rel_y", "rel_z", "distance"],
                    "output_features": ["dx", "dy", "dz"],
                    "n_subjects": len(subjects),
                    "n_mesh_points": n_mesh_points,
                    "n_mesh_edges": n_mesh_edges,
                    "in_features": in_features,
                    "processor_size": PROCESSOR_SIZE,
                    "hidden_dim": HIDDEN_DIM,
                    "epochs": epochs,
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
        (output_dir / "training_losses.json").write_text(
            json.dumps(losses, indent=2), encoding="utf-8"
        )
        rmse_log_file = output_dir / "training_validation_rmse.json"
        rmse_log_file.write_text(json.dumps(rmse_log, indent=2), encoding="utf-8")
        rmse_csv_file = output_dir / "training_validation_rmse.csv"
        with rmse_csv_file.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["epoch", "train_rmse_mm", "val_rmse_mm"]
            )
            writer.writeheader()
            writer.writerows(rmse_log)

        # Copy PCA assets so the output directory is self-contained.
        pca_src_dir = pca_mean_vtu.parent
        shutil.copy2(pca_mean_vtu, output_dir / pca_mean_vtu.name)
        pca_model_src = pca_src_dir / "pca_model.json"
        if pca_model_src.exists():
            shutil.copy2(pca_model_src, output_dir / pca_model_src.name)
        mean_surf.save(str(output_dir / "pca_mean_surface.vtp"))

        logging.info(f"Model saved to {checkpoint_file}")

        # ------------------------------------------------------------------ #
        # 10. Evaluate test and val subjects                                  #
        # ------------------------------------------------------------------ #
        def _evaluate_subject(sid: str, split_label: str) -> tuple[list[dict], Path]:
            data = subjects[sid]
            coords = mean_shape_coords if use_mean_shape_coords else data["ref_points"]
            norm_coords = (coords - coordinate_mean) / coordinate_scale
            norm_pca = (data["pca_coeffs"] - pca_mean_vec) / pca_scale_vec

            subj_out_dir = output_dir / sid
            subj_out_dir.mkdir(parents=True, exist_ok=True)

            sq_err_sum = np.zeros(n_mesh_points, dtype=np.float64)
            stats = []

            for phase_file in data["mesh_files"]:
                stage = _gating_stage_from_filename(phase_file)
                pred_disps = _infer_all_points(
                    model,
                    norm_coords,
                    norm_pca,
                    stage,
                    shared_graph_gpu,
                    shared_edge_feats_gpu,
                    displacement_scale,
                    device,
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

        if all_stats:
            per_phase_csv = test_output_dir / "statistics_per_phase.csv"
            with per_phase_csv.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(all_stats[0].keys()))
                writer.writeheader()
                writer.writerows(all_stats)

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

        return tutorial_outputs

    # %%
    tutorial_results = run_tutorial()
