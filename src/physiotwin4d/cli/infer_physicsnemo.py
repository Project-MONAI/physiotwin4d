"""Command-line interface for PhysicsNeMo cardiac mesh-stage inference.

Loads a trained model directory and predicts either from a per-subject manifest
(``--manifest``) or from a PCA shape-parameter file (``--shape-parameters``). The
network is auto-detected from the checkpoint files unless ``--network`` is given.
With ``--reference-image`` a deformation field and surface-normal image are
rasterized onto that image's grid.
"""

import argparse
import sys
from pathlib import Path


def _detect_network(model_dir: Path) -> str:
    """Return 'mgn' or 'mlp' based on the checkpoint present in ``model_dir``."""
    for tag in ("mgn", "mlp"):
        if (model_dir / f"{tag}_stage_model.pt").exists():
            return tag
    raise FileNotFoundError(
        f"No <tag>_stage_model.pt found in {model_dir}; pass --network explicitly."
    )


def main() -> int:
    """CLI entry point for PhysicsNeMo inference."""
    parser = argparse.ArgumentParser(
        description="Infer cardiac mesh stages with a trained PhysicsNeMo model.",
    )
    parser.add_argument("--model-dir", required=True, help="Trained model directory.")
    parser.add_argument(
        "--network",
        choices=("mgn", "mlp", "auto"),
        default="auto",
        help="Network architecture (auto-detected from the checkpoint by default).",
    )
    parser.add_argument("--epoch", type=int, default=None, help="Checkpoint epoch.")
    parser.add_argument("--output", default=None, help="Output directory.")

    # Manifest-driven mode.
    parser.add_argument("--manifest", default=None, help="Per-subject manifest JSON.")
    parser.add_argument(
        "--stages",
        nargs="*",
        type=float,
        default=None,
        help="Arbitrary stages to predict (manifest mode; omit for phase eval).",
    )

    # Manifest-free single-subject mode.
    parser.add_argument(
        "--shape-parameters", default=None, help="PCA shape-parameter JSON file."
    )
    parser.add_argument(
        "--stage", type=float, default=None, help="Target stage (single-subject mode)."
    )
    parser.add_argument(
        "--ground-truth", default=None, help="Optional ground-truth surface .vtp."
    )
    parser.add_argument(
        "--reference-image",
        default=None,
        help="Reference image; when given, write a deformation field + normal image.",
    )

    args = parser.parse_args()
    model_dir = Path(args.model_dir)
    network = args.network if args.network != "auto" else _detect_network(model_dir)
    output = Path(args.output) if args.output else None

    from ..workflow_infer_physicsnemo import (
        WorkflowInferPhysicsNeMo,
        WorkflowInferPhysicsNeMoMGN,
        WorkflowInferPhysicsNeMoMLP,
    )

    workflow: WorkflowInferPhysicsNeMo
    if network == "mgn":
        workflow = WorkflowInferPhysicsNeMoMGN(
            model_directory=model_dir, epoch=args.epoch
        )
    else:
        workflow = WorkflowInferPhysicsNeMoMLP(
            model_directory=model_dir, epoch=args.epoch
        )

    if args.manifest is not None:
        result = workflow.predict(
            Path(args.manifest), stages=args.stages, output_directory=output
        )
        print(f"Predicted {len(result['predicted_surfaces'])} surface(s).")
        return 0

    if args.shape_parameters is not None:
        if args.stage is None:
            parser.error("--stage is required with --shape-parameters.")
        if args.reference_image is not None:
            import itk

            reference_image = itk.imread(args.reference_image)
            result = workflow.create_deformation_field(
                Path(args.shape_parameters),
                args.stage,
                reference_image,
                output_directory=output,
            )
            print(
                f"Deformation field written to {result.get('deformation_field_file')}"
            )
            return 0

        ground_truth = Path(args.ground_truth) if args.ground_truth else None
        result = workflow.predict_single(
            Path(args.shape_parameters),
            args.stage,
            ground_truth=ground_truth,
            output_directory=output,
        )
        print(f"Predicted surface written to {result['predicted_surface']}")
        return 0

    parser.error("Provide either --manifest or --shape-parameters.")


if __name__ == "__main__":
    sys.exit(main())
