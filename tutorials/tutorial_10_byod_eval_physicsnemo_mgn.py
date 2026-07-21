"""
Tutorial 10 (MGN): Predict cardiac stage meshes for a subject with a trained
PhysicsNeMo MeshGraphNet.

Final stage of the cardiac 4D deep-learning pipeline (Tutorials 8 -> 9 -> 10).
This tutorial is a thin driver over
:class:`physiotwin4d.WorkflowInferPhysicsNeMoMGN`.  It builds a per-subject
manifest from the fitted-meshes directory and calls the workflow to predict
cardiac surfaces for one subject, loading the model trained by Tutorial 9
(``tutorial_09_byod_train_physicsnemo_mgn.py``).

This is a bring-your-own-data tutorial: the path constants below point at a local
``D:/PhysioTwin4D/`` layout and the Tutorial 9 run directory, not at the
repository ``data/`` directory.

Usage (command line)
--------------------
    py tutorial_10_byod_eval_physicsnemo_mgn.py pm0028 --out results/pm0028_mgn
    py tutorial_10_byod_eval_physicsnemo_mgn.py pm0028 --out results/pm0028_mgn --stages 0.0 0.25 0.5 0.75

Arguments
    subject    Subject ID, e.g. pm0028
    --epoch    Optional intermittent-checkpoint epoch; omit to use the final
               weights stored in the main checkpoint.
    --out      Output directory (created if missing)
    --stages   RR-interval fractions to predict (0-1). If omitted, predicts at
               every gated phase in the subject's manifest and computes error
               statistics against the ground-truth surfaces. When supplied, only
               the requested stages are predicted (no error statistics).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import pyvista as pv

from physiotwin4d import WorkflowConvertVTKToUSD, WorkflowInferPhysicsNeMoMGN

TUTORIALS_DIR = Path(__file__).resolve().parent
FITTED_MESHES_DIR = Path("D:/PhysioTwin4D/duke_data/fitted_kcl_meshes")
# Tutorial 9 run directory to evaluate (matches that trainer's OUTPUT_DIR).
MODEL_DIR = TUTORIALS_DIR / "output" / "tutorial_09_byod_mgn_3"

EPOCH = 1500

DEFAULT_SUBJECT = "pm0027"
DEFAULT_OUT_DIR = MODEL_DIR / "tutorial_10_byod_mgn" / DEFAULT_SUBJECT


def _gating_stage_from_filename(mesh_file: Path) -> float:
    for part in mesh_file.stem.split("_"):
        if part.startswith("g") and part[1:].isdigit():
            return int(part[1:]) / 100.0
    raise ValueError(f"Cannot parse gating percentage from filename: {mesh_file}")


def _write_subject_manifest(subject: str, out_dir: Path) -> Path:
    """Build a manifest for *subject* from the fitted-meshes directory."""
    subject_dir = FITTED_MESHES_DIR / subject
    ref_file = subject_dir / f"{subject}_ssm_surface.vtp"
    pca_file = subject_dir / f"{subject}_ssm_pca_coefficients.json"
    phase_files = sorted(subject_dir.glob(f"{subject}_g0*_ssm_surface.vtp"))
    for required in (ref_file, pca_file):
        if not required.exists():
            sys.exit(f"Missing: {required}")
    if not phase_files:
        sys.exit(f"No gated phase files found in {subject_dir}")

    manifest = {
        "subject_id": subject,
        "reference_surface": str(ref_file),
        "pca_coefficients": str(pca_file),
        "phases": [
            {"surface": str(pf), "stage": _gating_stage_from_filename(pf)}
            for pf in phase_files
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"{subject}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def predict(
    subject: str,
    out_dir: Path,
    epoch: Optional[int] = None,
    stages: Optional[list[float]] = None,
) -> dict[str, Any]:
    """Predict cardiac surfaces for *subject* using the trained MeshGraphNet."""
    manifest_path = _write_subject_manifest(subject, out_dir)
    infer = WorkflowInferPhysicsNeMoMGN(model_directory=MODEL_DIR, epoch=epoch)
    result = infer.predict(manifest_path, stages=stages, output_directory=out_dir)

    # Assemble the ordered predicted surfaces into a 4D USD, colored with the heart
    # anatomy material via USDAnatomyTools (appearance="anatomy").
    surfaces = [pv.read(str(path)) for path in result["predicted_surfaces"]]
    usd_workflow = WorkflowConvertVTKToUSD(
        input_meshes=surfaces,
        usd_project_name=f"{subject}_mgn_4d",
        output_directory=out_dir,
        appearance="anatomy",
        anatomy_type="heart",
        separate_by_connectivity=True,
        frames_per_second=float(len(surfaces)),
    )
    result["usd_file"] = usd_workflow.process()["usd_file"]
    return result


def run_tutorial() -> dict[str, Any]:
    """Tutorial / test entry point: evaluate DEFAULT_SUBJECT with the final weights."""
    return predict(DEFAULT_SUBJECT, DEFAULT_OUT_DIR, epoch=EPOCH)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Predict cardiac stage meshes for one subject with MeshGraphNet.",
    )
    ap.add_argument("subject", help="Subject ID, e.g. pm0028")
    ap.add_argument(
        "--epoch", type=int, default=None, help="Checkpoint epoch (default: final)"
    )
    ap.add_argument("--out", type=Path, required=True, help="Output directory")
    ap.add_argument(
        "--stages",
        type=float,
        nargs="+",
        metavar="FRAC",
        help="RR-interval fractions to predict (omit for per-phase error stats).",
    )
    args = ap.parse_args()
    predict(args.subject, args.out, epoch=args.epoch, stages=args.stages)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) > 1:
        main()
    else:
        # Tutorial / test entry point (no CLI arguments)
        tutorial_results = run_tutorial()
