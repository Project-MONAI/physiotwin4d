"""
Tutorial 10b (MLP): Predict cardiac stage meshes for a subject using a trained
PhysicsNeMo MLP checkpoint.

Final stage of the cardiac 4D deep-learning pipeline (Tutorials 8 -> 9 -> 10).
Loads the MLP checkpoint trained by Tutorial 9b
(``tutorial_09b_cardiac_train_physicsnemo_mlp.py``) and predicts cardiac
surfaces for one subject.  Can be run from the command line, or cell-by-cell /
as a tutorial test via the ``run_tutorial`` entry point (which uses the
``DEFAULT_SUBJECT`` / ``DEFAULT_EPOCH`` constants below).

This is a bring-your-own-data tutorial: the path constants below point at a local
``D:/PhysioMotion4D/`` layout and the Tutorial 9b run directory, not at the
repository ``data/`` directory.

Usage (command line)
--------------------
    py tutorial_10b_cardiac_eval_physicsnemo_mlp.py pm0002 --epoch 5000 --out results/pm0002
    py tutorial_10b_cardiac_eval_physicsnemo_mlp.py pm0002 --epoch 5000 --out results/pm0002 --stages 0.0 0.25 0.5 0.75

Arguments
    subject    Subject ID, e.g. pm0002
    --epoch    Training epoch whose checkpoint to load, e.g. 5000
    --out      Output directory (created if missing)
    --stages   RR-interval fractions to predict (0-1).  If omitted, predicts at
               every gated phase found in the subject's fitted-meshes directory
               and computes error statistics against the ground-truth surfaces.
               When supplied, only the requested stages are predicted (no error
               stats, since ground-truth surfaces may not exist).

Reads weights from  OUTPUT_DIR/physicsnemo_stage_model_epoch_EEEEE.pt
and normalisation metadata from  OUTPUT_DIR/physicsnemo_stage_model.pt,
both produced by Tutorial 9b.  (The ``OUTPUT_DIR`` constant below selects which
run directory to evaluate.)

Outputs (--stages omitted)
--------------------------
One predicted .vtp surface per gated phase, each carrying per-point arrays:

    error_x/y/z     signed offset (pred - target) in mm
    error_mm        Euclidean distance from target in mm
    rmse_mm         per-point RMSE over all stages (same value in every file)

statistics.csv      per-stage and summary error statistics

Outputs (--stages supplied)
---------------------------
One predicted .vtp surface per requested stage; no error arrays or CSV.
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
    from physicsnemo.models.mlp import FullyConnected
except ImportError as exc:
    raise ImportError(
        'Requires PhysicsNeMo. Install with: pip install "physiomotion4d[physicsnemo]"'
    ) from exc

logger = logging.getLogger("tutorial_10b_cardiac_eval_physicsnemo_mlp")

TUTORIALS_DIR = Path(__file__).resolve().parent
FITTED_MESHES_DIR = Path("D:/PhysioMotion4D/duke_data/fitted_kcl_meshes")
# Tutorial 9b run directory to evaluate (matches that trainer's OUTPUT_DIR).
OUTPUT_DIR = TUTORIALS_DIR / "output"
BATCH_SIZE = 262144

# Defaults used by run_tutorial() when this file is run with no CLI arguments
# (e.g. cell-by-cell execution or as a tutorial test).
DEFAULT_SUBJECT = "pm0028"
DEFAULT_EPOCH = 5000
DEFAULT_OUT_DIR = TUTORIALS_DIR / "output" / "eval_mlp" / DEFAULT_SUBJECT


def _latest_epoch_checkpoint(output_dir: Path) -> Optional[int]:
    """Return the highest epoch number among saved checkpoints, or None if none exist."""
    epochs = []
    for ckpt in output_dir.glob("physicsnemo_stage_model_epoch_*.pt"):
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


def _infer(
    model: "FullyConnected",
    norm_coords: np.ndarray,
    pca_tile: np.ndarray,
    stage: float,
    displacement_scale: float,
    device: "torch.device",
) -> np.ndarray:
    n = len(norm_coords)
    stage_col = np.full((n, 1), stage, dtype=np.float32)
    pred_inputs = np.hstack([norm_coords, pca_tile, stage_col])
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, n, BATCH_SIZE):
            stop = min(start + BATCH_SIZE, n)
            t = torch.from_numpy(pred_inputs[start:stop]).to(device)
            chunks.append(model(t).cpu().numpy())
    return np.vstack(chunks) * displacement_scale


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

    # When --stages is not given, discover gated phase files and validate them.
    if stages is None:
        phase_files = sorted(subject_dir.glob(f"{subject}_g0*_ssm_surface.vtp"))
        if not phase_files:
            sys.exit(f"No gated phase files found in {subject_dir}")
    else:
        bad = [s for s in stages if not 0.0 <= s <= 1.0]
        if bad:
            sys.exit(f"--stages values must be in [0, 1]; got: {bad}")
        phase_files = []

    meta_ckpt = OUTPUT_DIR / "physicsnemo_stage_model.pt"
    if not meta_ckpt.exists():
        sys.exit(
            f"Metadata checkpoint not found: {meta_ckpt}\n"
            "Run Tutorial 9b (tutorial_09b_cardiac_train_physicsnemo_mlp.py) first."
        )
    meta = torch.load(meta_ckpt, map_location="cpu", weights_only=True)

    epoch_ckpt = OUTPUT_DIR / f"physicsnemo_stage_model_epoch_{epoch:05d}.pt"
    if not epoch_ckpt.exists():
        sys.exit(f"Epoch checkpoint not found: {epoch_ckpt}")

    coordinate_mean = np.array(meta["coordinate_mean"], dtype=np.float32)
    coordinate_scale = np.array(meta["coordinate_scale"], dtype=np.float32)
    pca_mean_vec = np.array(meta["pca_mean"], dtype=np.float32)
    pca_scale_vec = np.array(meta["pca_scale"], dtype=np.float32)
    displacement_scale = float(meta["displacement_scale"])
    use_mean_shape_coords: bool = meta["use_mean_shape_coords"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FullyConnected(
        in_features=int(meta["in_features"]),
        layer_size=int(meta["layer_size"]),
        out_features=3,
        num_layers=int(meta["num_layers"]),
        activation_fn="silu",
        skip_connections=True,
    ).to(device)
    model.load_state_dict(
        torch.load(epoch_ckpt, map_location=device, weights_only=True)
    )
    model.eval()

    ref_mesh = cast(pv.PolyData, pv.read(str(ref_file)))
    ref_points = np.asarray(ref_mesh.points, dtype=np.float32)
    pca_coeffs = np.array(
        json.loads(pca_file.read_text(encoding="utf-8")), dtype=np.float32
    )

    if use_mean_shape_coords:
        mean_surf_vtp = OUTPUT_DIR / "pca_mean_surface.vtp"
        pca_mean_vtu_path = meta.get("pca_mean_vtu")
        if mean_surf_vtp.exists():
            coords = np.asarray(pv.read(str(mean_surf_vtp)).points, dtype=np.float32)
        elif pca_mean_vtu_path and Path(pca_mean_vtu_path).exists():
            vol = pv.read(str(pca_mean_vtu_path))
            coords = np.asarray(
                vol.extract_surface(algorithm="dataset_surface").points,
                dtype=np.float32,
            )
        else:
            sys.exit(
                f"Mean-shape surface not found. Expected {mean_surf_vtp} "
                f"or {pca_mean_vtu_path}"
            )
    else:
        coords = ref_points

    if len(coords) != len(ref_points):
        sys.exit(
            f"Topology mismatch: coordinate surface has {len(coords)} points, "
            f"but {ref_file} has {len(ref_points)} points."
        )
    if pca_coeffs.shape != pca_mean_vec.shape:
        sys.exit(
            f"PCA coefficient mismatch: subject has shape {pca_coeffs.shape}, "
            f"checkpoint expects {pca_mean_vec.shape}."
        )

    norm_coords = (coords - coordinate_mean) / coordinate_scale
    norm_pca = (pca_coeffs - pca_mean_vec) / pca_scale_vec
    pca_tile = np.tile(norm_pca, (len(norm_coords), 1))

    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Subject: %s  epoch: %s  device: %s", subject, epoch, device)

    n_points = len(ref_points)

    if stages is None:
        return _predict_with_errors(
            subject,
            epoch,
            ref_mesh,
            ref_points,
            norm_coords,
            pca_tile,
            displacement_scale,
            phase_files,
            model,
            device,
            out_dir,
            n_points,
        )
    return _predict_arbitrary_stages(
        subject,
        ref_mesh,
        ref_points,
        norm_coords,
        pca_tile,
        displacement_scale,
        stages,
        model,
        device,
        out_dir,
    )


def _predict_with_errors(
    subject: str,
    epoch: int,
    ref_mesh: pv.PolyData,
    ref_points: np.ndarray,
    norm_coords: np.ndarray,
    pca_tile: np.ndarray,
    displacement_scale: float,
    phase_files: list[Path],
    model: "FullyConnected",
    device: "torch.device",
    out_dir: Path,
    n_points: int,
) -> dict[str, Any]:
    """Predict at each existing gated phase; compute and embed error statistics."""
    sq_err_sum = np.zeros(n_points, dtype=np.float64)
    results: list[dict] = []
    predicted_files: list[Path] = []

    for phase_file in phase_files:
        stage = _gating_stage_from_filename(phase_file)
        pred_disps = _infer(
            model, norm_coords, pca_tile, stage, displacement_scale, device
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

    stats_rows: list[dict] = []
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
    pca_tile: np.ndarray,
    displacement_scale: float,
    stages: list[float],
    model: "FullyConnected",
    device: "torch.device",
    out_dir: Path,
) -> dict[str, Any]:
    """Predict at caller-specified RR-interval fractions; no ground-truth comparison."""
    predicted_files: list[Path] = []
    for stage in stages:
        pred_disps = _infer(
            model, norm_coords, pca_tile, stage, displacement_scale, device
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
    under OUTPUT_DIR so this works whether Tutorial 9b ran a full training
    pass or the reduced test-mode epoch count; falls back to DEFAULT_EPOCH if
    no checkpoints are found. Returns the prediction outputs dict.
    """
    epoch = _latest_epoch_checkpoint(OUTPUT_DIR) or DEFAULT_EPOCH
    return predict(DEFAULT_SUBJECT, epoch, DEFAULT_OUT_DIR)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Predict cardiac stage meshes for one subject.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example: py tutorial_10b_cardiac_eval_physicsnemo_mlp.py "
            "pm0002 --epoch 5000 --out results/pm0002"
        ),
    )
    ap.add_argument("subject", help="Subject ID, e.g. pm0002")
    ap.add_argument(
        "--epoch", type=int, required=True, help="Training epoch, e.g. 5000"
    )
    ap.add_argument("--out", type=Path, required=True, help="Output directory")
    ap.add_argument(
        "--stages",
        type=float,
        nargs="+",
        metavar="FRAC",
        help="RR-interval fractions to predict, e.g. --stages 0.0 0.25 0.5 0.75  "
        "(omit to predict at every existing gated phase with error statistics)",
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
