"""
Tutorial 10a (MGN): Predict cardiac stage meshes for a subject using a trained
PhysicsNeMo MeshGraphNet.

Final stage of the cardiac 4D deep-learning pipeline (Tutorials 8 -> 9 -> 10).
Loads the MeshGraphNet checkpoint trained by Tutorial 9a
(``tutorial_09a_cardiac_train_physicsnemo_mgn.py``) and predicts cardiac
surfaces for one subject.  Can be run from the command line, or cell-by-cell /
as a tutorial test via the ``run_tutorial`` entry point (which uses the
``DEFAULT_SUBJECT`` / ``DEFAULT_EPOCH`` constants below).

This is a bring-your-own-data tutorial: the path constants below point at a local
``D:/PhysioTwin4D/`` layout and the Tutorial 9a run directory, not at the
repository ``data/`` directory.

Usage (command line)
--------------------
    py tutorial_10a_cardiac_eval_physicsnemo_mgn.py pm0028 --epoch 1500 --out results/pm0028_mgn
    py tutorial_10a_cardiac_eval_physicsnemo_mgn.py pm0028 --epoch 1500 --out results/pm0028_mgn --stages 0.0 0.25 0.5 0.75

Arguments
    subject    Subject ID, e.g. pm0028
    --epoch    Training epoch whose checkpoint to load, e.g. 1500
    --out      Output directory (created if missing)
    --stages   RR-interval fractions to predict (0-1). If omitted, predicts at
               every gated phase found in the subject's fitted-meshes directory
               and computes error statistics against the ground-truth surfaces.
               When supplied, only the requested stages are predicted (no error
               stats, since ground-truth surfaces may not exist).

Reads weights from OUTPUT_DIR/mgn_stage_model_epoch_EEEEE.pt and normalization
metadata from OUTPUT_DIR/mgn_stage_model.pt, both produced by Tutorial 9a. The
shared mesh graph is replayed from OUTPUT_DIR/shared_edge_index.pt and
OUTPUT_DIR/shared_edge_features.pt.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import pyvista as pv
import torch

try:
    from torch_geometric.data import Data

    from physicsnemo.models.meshgraphnet import MeshGraphNet
except ImportError as exc:
    raise ImportError(
        "Requires PhysicsNeMo and PyTorch Geometric. Install with:\n"
        '  pip install "physiotwin4d[physicsnemo]"\n'
        "  pip install torch-geometric"
    ) from exc

logger = logging.getLogger("tutorial_10a_cardiac_eval_physicsnemo_mgn")

TUTORIALS_DIR = Path(__file__).resolve().parent
FITTED_MESHES_DIR = Path("D:/PhysioTwin4D/duke_data/fitted_kcl_meshes")
# Tutorial 9a run directory to evaluate (matches that trainer's OUTPUT_DIR).
OUTPUT_DIR = TUTORIALS_DIR / "output_mgn"

# Defaults used by run_tutorial() when this file is run with no CLI arguments
# (e.g. cell-by-cell execution or as a tutorial test).
DEFAULT_SUBJECT = "pm0028"
DEFAULT_EPOCH = 1500
DEFAULT_OUT_DIR = TUTORIALS_DIR / "output_mgn" / "eval_mgn" / DEFAULT_SUBJECT

# These match tutorial_09a_cardiac_train_physicsnemo_mgn.py. Older checkpoints
# store only processor_size and hidden_dim in metadata, so keep the layer
# defaults explicit.
DEFAULT_NUM_LAYERS_PROCESSOR = 2
DEFAULT_NUM_LAYERS_ENCODER = 2
DEFAULT_NUM_LAYERS_DECODER = 2
DEFAULT_NUM_PROCESSOR_CHECKPOINT_SEGMENTS = 0


def _latest_epoch_checkpoint(output_dir: Path) -> Optional[int]:
    """Return the highest epoch number among saved checkpoints, or None if none exist."""
    epochs = []
    for ckpt in output_dir.glob("mgn_stage_model_epoch_*.pt"):
        try:
            epochs.append(int(ckpt.stem.rsplit("_", 1)[-1]))
        except ValueError:
            continue
    return max(epochs) if epochs else None


def _gating_stage_from_filename(mesh_file: Path) -> float:
    for part in mesh_file.stem.split("_"):
        if part.startswith("g") and part[1:].isdigit():
            return int(part[1:]) / 100.0
    raise ValueError(f"Cannot parse gating percentage from filename: {mesh_file}")


def _state_dict_from_checkpoint(checkpoint: Any) -> dict[str, torch.Tensor]:
    state = (
        checkpoint.get("model_state_dict", checkpoint)
        if isinstance(checkpoint, dict)
        else checkpoint
    )
    if not isinstance(state, dict):
        raise TypeError(
            f"Checkpoint did not contain a state dict: {type(checkpoint)!r}"
        )
    return state


def _strip_compile_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if all(k.startswith("_orig_mod.") for k in state):
        return {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    return state


def _build_model(meta: dict[str, Any], device: torch.device) -> MeshGraphNet:
    return MeshGraphNet(
        input_dim_nodes=int(meta["in_features"]),
        input_dim_edges=4,
        output_dim=3,
        processor_size=int(meta["processor_size"]),
        hidden_dim_processor=int(meta["hidden_dim"]),
        hidden_dim_node_encoder=int(meta["hidden_dim"]),
        num_layers_node_encoder=int(
            meta.get("num_layers_node_encoder", DEFAULT_NUM_LAYERS_ENCODER)
        ),
        hidden_dim_node_decoder=int(meta["hidden_dim"]),
        num_layers_node_decoder=int(
            meta.get("num_layers_node_decoder", DEFAULT_NUM_LAYERS_DECODER)
        ),
        hidden_dim_edge_encoder=int(meta["hidden_dim"]),
        num_layers_edge_processor=int(
            meta.get("num_layers_edge_processor", DEFAULT_NUM_LAYERS_PROCESSOR)
        ),
        num_layers_node_processor=int(
            meta.get("num_layers_node_processor", DEFAULT_NUM_LAYERS_PROCESSOR)
        ),
        aggregation="mean",
        num_processor_checkpoint_segments=int(
            meta.get(
                "num_processor_checkpoint_segments",
                DEFAULT_NUM_PROCESSOR_CHECKPOINT_SEGMENTS,
            )
        ),
    ).to(device)


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
    n = len(norm_coords)
    pca_tile = np.tile(norm_pca, (n, 1))
    stage_col = np.full((n, 1), stage, dtype=np.float32)
    node_feats = torch.tensor(
        np.hstack([norm_coords, pca_tile, stage_col]),
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        pred = model(node_feats, shared_edge_feats, shared_graph)
    return np.asarray(pred.cpu().numpy()) * displacement_scale


def predict(
    subject: str,
    epoch: int,
    out_dir: Path,
    stages: Optional[list[float]] = None,
) -> dict[str, Any]:
    subject_dir = FITTED_MESHES_DIR / subject
    ref_file = subject_dir / f"{subject}_ssm_surface.vtp"
    pca_file = subject_dir / f"{subject}_ssm_pca_coefficients.json"

    for p in (ref_file, pca_file):
        if not p.exists():
            sys.exit(f"Missing: {p}")

    if stages is None:
        phase_files = sorted(subject_dir.glob(f"{subject}_g0*_ssm_surface.vtp"))
        if not phase_files:
            sys.exit(f"No gated phase files found in {subject_dir}")
    else:
        bad = [s for s in stages if not 0.0 <= s <= 1.0]
        if bad:
            sys.exit(f"--stages values must be in [0, 1]; got: {bad}")
        phase_files = []

    meta_ckpt = OUTPUT_DIR / "mgn_stage_model.pt"
    epoch_ckpt = OUTPUT_DIR / f"mgn_stage_model_epoch_{epoch:05d}.pt"
    edge_index_file = OUTPUT_DIR / "shared_edge_index.pt"
    edge_features_file = OUTPUT_DIR / "shared_edge_features.pt"
    for p in (meta_ckpt, epoch_ckpt, edge_index_file, edge_features_file):
        if not p.exists():
            sys.exit(f"Missing trained GNN artifact: {p}")

    meta = torch.load(meta_ckpt, map_location="cpu", weights_only=True)
    coordinate_mean = np.array(meta["coordinate_mean"], dtype=np.float32)
    coordinate_scale = np.array(meta["coordinate_scale"], dtype=np.float32)
    pca_mean_vec = np.array(meta["pca_mean"], dtype=np.float32)
    pca_scale_vec = np.array(meta["pca_scale"], dtype=np.float32)
    displacement_scale = float(meta["displacement_scale"])
    use_mean_shape_coords = bool(meta["use_mean_shape_coords"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(meta, device)
    epoch_state = torch.load(epoch_ckpt, map_location=device, weights_only=True)
    model.load_state_dict(
        _strip_compile_prefix(_state_dict_from_checkpoint(epoch_state))
    )
    model.eval()

    shared_edge_index = torch.load(
        edge_index_file, map_location=device, weights_only=True
    )
    shared_edge_feats = torch.load(
        edge_features_file, map_location=device, weights_only=True
    )
    n_graph_nodes = int(shared_edge_index.max().item()) + 1
    shared_graph = Data(edge_index=shared_edge_index, num_nodes=n_graph_nodes).to(
        device
    )
    shared_edge_feats = shared_edge_feats.to(device)

    ref_mesh = cast(pv.PolyData, pv.read(str(ref_file)))
    ref_points = np.asarray(ref_mesh.points, dtype=np.float32)
    pca_coeffs = np.array(
        json.loads(pca_file.read_text(encoding="utf-8")), dtype=np.float32
    )

    if use_mean_shape_coords:
        mean_surf_vtp = OUTPUT_DIR / "pca_mean_surface.vtp"
        if not mean_surf_vtp.exists():
            sys.exit(f"Mean-shape surface not found: {mean_surf_vtp}")
        coords = np.asarray(pv.read(str(mean_surf_vtp)).points, dtype=np.float32)
    else:
        coords = ref_points

    if len(coords) != len(ref_points):
        sys.exit(
            f"Topology mismatch: coordinate surface has {len(coords)} points, "
            f"but {ref_file} has {len(ref_points)} points."
        )
    if len(coords) != n_graph_nodes:
        sys.exit(
            f"Graph topology mismatch: graph has {n_graph_nodes} nodes, "
            f"but coordinate surface has {len(coords)} points."
        )
    if pca_coeffs.shape != pca_mean_vec.shape:
        sys.exit(
            f"PCA coefficient mismatch: subject has shape {pca_coeffs.shape}, "
            f"checkpoint expects {pca_mean_vec.shape}."
        )

    norm_coords = (coords - coordinate_mean) / coordinate_scale
    norm_pca = (pca_coeffs - pca_mean_vec) / pca_scale_vec

    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Subject: %s  epoch: %s  device: %s", subject, epoch, device)

    if stages is None:
        return _predict_with_errors(
            subject,
            epoch,
            ref_mesh,
            ref_points,
            norm_coords,
            norm_pca,
            phase_files,
            model,
            shared_graph,
            shared_edge_feats,
            displacement_scale,
            device,
            out_dir,
        )
    return _predict_arbitrary_stages(
        subject,
        ref_mesh,
        ref_points,
        norm_coords,
        norm_pca,
        stages,
        model,
        shared_graph,
        shared_edge_feats,
        displacement_scale,
        device,
        out_dir,
    )


def _predict_with_errors(
    subject: str,
    epoch: int,
    ref_mesh: pv.PolyData,
    ref_points: np.ndarray,
    norm_coords: np.ndarray,
    norm_pca: np.ndarray,
    phase_files: list[Path],
    model: MeshGraphNet,
    shared_graph: Data,
    shared_edge_feats: torch.Tensor,
    displacement_scale: float,
    device: torch.device,
    out_dir: Path,
) -> dict[str, Any]:
    n_points = len(ref_points)
    sq_err_sum = np.zeros(n_points, dtype=np.float64)
    results: list[dict[str, Any]] = []
    predicted_files: list[Path] = []

    for phase_file in phase_files:
        stage = _gating_stage_from_filename(phase_file)
        pred_disps = _infer_all_points(
            model,
            norm_coords,
            norm_pca,
            stage,
            shared_graph,
            shared_edge_feats,
            displacement_scale,
            device,
        )
        pred_points = (ref_points + pred_disps).astype(np.float32)
        actual_points = np.asarray(pv.read(str(phase_file)).points, dtype=np.float32)
        error_vec = pred_points - actual_points
        error_mag = np.linalg.norm(error_vec, axis=1).astype(np.float32)
        sq_err_sum += error_mag.astype(np.float64) ** 2
        gating_tag = phase_file.stem.split("_ssm_surface")[0].split("_")[-1]
        results.append(
            {
                "gating_tag": gating_tag,
                "stage": stage,
                "pred_points": pred_points,
                "error_vec": error_vec,
                "error_mag": error_mag,
            }
        )

    point_rmse = np.sqrt(sq_err_sum / len(results)).astype(np.float32)

    stats_rows: list[dict[str, Any]] = []
    for r in results:
        pred_mesh = ref_mesh.copy(deep=True)
        pred_mesh.points = r["pred_points"]
        pred_mesh.point_data["error_x"] = r["error_vec"][:, 0]
        pred_mesh.point_data["error_y"] = r["error_vec"][:, 1]
        pred_mesh.point_data["error_z"] = r["error_vec"][:, 2]
        pred_mesh.point_data["error_mm"] = r["error_mag"]
        pred_mesh.point_data["rmse_mm"] = point_rmse

        tag = r["gating_tag"]
        out_file = out_dir / f"{subject}_{tag}_ssm_surface_pred.vtp"
        pred_mesh.save(str(out_file))
        predicted_files.append(out_file)
        logger.info("  %s", out_file.name)

        em = r["error_mag"]
        stats_rows.append(
            {
                "subject": subject,
                "epoch": epoch,
                "gating_tag": tag,
                "stage": r["stage"],
                "n_points": n_points,
                "mean_error_mm": float(em.mean()),
                "median_error_mm": float(np.median(em)),
                "rms_error_mm": float(np.sqrt(np.mean(em**2))),
                "std_error_mm": float(em.std()),
                "max_error_mm": float(em.max()),
                "mean_abs_error_x_mm": float(np.abs(r["error_vec"][:, 0]).mean()),
                "mean_abs_error_y_mm": float(np.abs(r["error_vec"][:, 1]).mean()),
                "mean_abs_error_z_mm": float(np.abs(r["error_vec"][:, 2]).mean()),
            }
        )

    stats_rows.append(
        {
            "subject": subject,
            "epoch": epoch,
            "gating_tag": "ALL",
            "stage": "",
            "n_points": n_points,
            "mean_error_mm": float(np.mean([r["mean_error_mm"] for r in stats_rows])),
            "median_error_mm": float(
                np.mean([r["median_error_mm"] for r in stats_rows])
            ),
            "rms_error_mm": float(
                np.sqrt(sq_err_sum.sum() / (n_points * len(results)))
            ),
            "std_error_mm": float(np.mean([r["std_error_mm"] for r in stats_rows])),
            "max_error_mm": float(np.max([r["max_error_mm"] for r in stats_rows])),
            "mean_abs_error_x_mm": float(
                np.mean([r["mean_abs_error_x_mm"] for r in stats_rows])
            ),
            "mean_abs_error_y_mm": float(
                np.mean([r["mean_abs_error_y_mm"] for r in stats_rows])
            ),
            "mean_abs_error_z_mm": float(
                np.mean([r["mean_abs_error_z_mm"] for r in stats_rows])
            ),
        }
    )

    csv_path = out_dir / "statistics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(stats_rows[0].keys()))
        writer.writeheader()
        writer.writerows(stats_rows)

    summary = next(r for r in stats_rows if r["gating_tag"] == "ALL")
    logger.info(
        "%d predictions -> %s  (overall RMS=%.4f mm, mean=%.4f mm, max=%.4f mm, "
        "statistics -> %s)",
        len(results),
        out_dir,
        summary["rms_error_mm"],
        summary["mean_error_mm"],
        summary["max_error_mm"],
        csv_path.name,
    )
    return {
        "subject": subject,
        "epoch": epoch,
        "predicted_files": predicted_files,
        "statistics_csv": csv_path,
        "summary": summary,
    }


def _predict_arbitrary_stages(
    subject: str,
    ref_mesh: pv.PolyData,
    ref_points: np.ndarray,
    norm_coords: np.ndarray,
    norm_pca: np.ndarray,
    stages: list[float],
    model: MeshGraphNet,
    shared_graph: Data,
    shared_edge_feats: torch.Tensor,
    displacement_scale: float,
    device: torch.device,
    out_dir: Path,
) -> dict[str, Any]:
    predicted_files: list[Path] = []
    for stage in stages:
        pred_disps = _infer_all_points(
            model,
            norm_coords,
            norm_pca,
            stage,
            shared_graph,
            shared_edge_feats,
            displacement_scale,
            device,
        )
        pred_mesh = ref_mesh.copy(deep=True)
        pred_mesh.points = (ref_points + pred_disps).astype(np.float32)
        tag = f"s{round(stage * 100):03d}"
        out_file = out_dir / f"{subject}_{tag}_ssm_surface_pred.vtp"
        pred_mesh.save(str(out_file))
        predicted_files.append(out_file)
        logger.info("  %s", out_file.name)

    logger.info("%d predictions -> %s", len(stages), out_dir)
    return {
        "subject": subject,
        "predicted_files": predicted_files,
        "statistics_csv": None,
        "summary": None,
    }


def run_tutorial() -> dict[str, Any]:
    """Tutorial / test entry point: evaluate DEFAULT_SUBJECT at the latest checkpoint.

    Used when the script is run with no command-line arguments (cell-by-cell
    execution or as a tutorial test).  Picks the highest-numbered checkpoint
    under OUTPUT_DIR so this works whether Tutorial 9a ran a full training
    pass or the reduced test-mode epoch count; falls back to DEFAULT_EPOCH if
    no checkpoints are found. Returns the prediction outputs dict.
    """
    epoch = _latest_epoch_checkpoint(OUTPUT_DIR) or DEFAULT_EPOCH
    return predict(DEFAULT_SUBJECT, epoch, DEFAULT_OUT_DIR)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Predict cardiac stage meshes for one subject with MeshGraphNet.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example: py tutorial_10a_cardiac_eval_physicsnemo_mgn.py "
            "pm0028 --epoch 1500 --out results/pm0028_mgn"
        ),
    )
    ap.add_argument("subject", help="Subject ID, e.g. pm0028")
    ap.add_argument(
        "--epoch", type=int, required=True, help="Training epoch, e.g. 1500"
    )
    ap.add_argument("--out", type=Path, required=True, help="Output directory")
    ap.add_argument(
        "--stages",
        type=float,
        nargs="+",
        metavar="FRAC",
        help=(
            "RR-interval fractions to predict, e.g. --stages 0.0 0.25 0.5 0.75 "
            "(omit to predict at every existing gated phase with error statistics)"
        ),
    )
    args = ap.parse_args()
    predict(args.subject, args.epoch, args.out, stages=args.stages)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) > 1:
        main()
    else:
        # %%
        # Tutorial / test entry point (no CLI arguments)
        tutorial_results = run_tutorial()
