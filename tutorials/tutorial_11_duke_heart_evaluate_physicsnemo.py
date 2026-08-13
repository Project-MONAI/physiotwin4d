"""
Tutorial 11 (Duke Heart, MGN): Score Predicted Heart Motion Per Chamber

Purpose
-------
Duke counterpart of ``tutorial_11_lung_evaluate_physicsnemo.py``.  Measures how
close the size and shape of the heart inferred by Tutorial 10 are to the heart
actually imaged, one gated frame at a time and one chamber at a time.  The case
is ``ParametersDukeHeartLabelmaps.hold_out_case``, held out of every fit in this
chain, so this scores generalization rather than recall.

1. Read the ground truth.  Unlike the lung chain, no segmentation is needed:
   this cohort ships one labelmap per gated frame, each already carrying the
   four chambers, the myocardium and the whole heart.

2. Score the prediction: :class:`physiotwin4d.WorkflowEvaluateMovement` carries
   the reference frame's labelmap into every other frame with the network's own
   deformation, and compares the result to that frame's labelmap --- volume
   difference, Dice and surface RMSE per chamber.

   The shape model this network moves is one structure, the whole heart minus
   its chamber cavities, so the chambers exist only in the acquired labelmaps.
   Going through the labelmaps rather than through the model's surface is what
   makes per-chamber scores possible at all.

3. Write ``evaluation_report.md`` and ``evaluation_metrics.csv``, both carrying
   the hold-out case name, the case's shape parameters, and the network weights
   path with its dates.

Extra Install Required
----------------------
PhysicsNeMo and PyTorch Geometric must be installed::

    pip install "physiotwin4d[physicsnemo]"

Data Required
-------------
  * ``data/Duke-Heart-4DLabelmaps/<case>/*_labelmap.nii.gz`` - gated frames
  * ``output/tutorial_08_duke_heart/<case>/`` - Tutorial 8 SSM surface + coefficients
  * ``network_weights/physicsnemo_mgn_duke_heart_motion/`` - Tutorial 9 checkpoint

Outputs (under ``output/tutorial_11_duke_heart/<case>/``)
---------------------------------------------------------
  * ``evaluation_report.md``    - per-chamber accuracy of the prediction
  * ``evaluation_metrics.csv``  - one row per stage and structure
  * ``volume_vs_stage.png``     - each structure's volume across the stages
  * ``<case>_ssm_pca_coefficients_s{TTT}_pred.vtp`` - predicted surface per stage
"""

# Imports
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, cast

import itk
import pyvista as pv
from parameters_duke_heart_labelmaps import DUKE_HEART

from physiotwin4d import (
    SegmentHeartSimplewareTrimmedBranches,
    TestTools,
    WorkflowEvaluateMovement,
    WorkflowInferMovement,
    WorkflowInferPhysicsNeMo,
)

LABELMAP_SUFFIX = "_labelmap.nii.gz"

# The four chambers, plus the myocardium and the whole heart for context: 5 and
# 6 are what the shape model itself represents, 1-4 are the cavities it does
# not.  The great vessels and coronaries (7-10) are left out; they come and go
# between frames and are not part of the model.
HEART_LABEL_IDS = [1, 2, 3, 4, 5, 6]


def _cardiac_stage_from_filename(labelmap_file: Path) -> float:
    """Extract the normalized cardiac stage [0, 1] from a ``g{PPP}`` filename stem."""
    for part in labelmap_file.name.split("_"):
        if part.startswith("g") and part[1:].isdigit():
            return int(part[1:]) / 100.0
    raise ValueError(f"Cannot parse cardiac gate from filename: {labelmap_file}")


# Only run if this script is not imported as a module

# PhysicsNeMo and torch spawn worker processes. On Windows the spawn start
# method re-imports this script in each child; without the
# __name__ == "__main__" guard around top-level work, that re-import would
# restart the whole evaluation in every worker.
if __name__ == "__main__":
    # Data directory specification
    repo_root = Path(__file__).resolve().parent.parent
    tutorials_dir = Path(__file__).resolve().parent

    class_name = "tutorial_11_duke_heart_evaluate_physicsnemo"

    # Case to score: the case held out of every fit in this chain.
    case_id = DUKE_HEART.hold_out_case

    # Fitted SSM surface and PCA coefficients written by Tutorial 8 (Duke Heart).
    case_dir = tutorials_dir / "output" / "tutorial_08_duke_heart" / case_id
    # Weights Tutorial 9 trained, and the checkpoint epoch Tutorial 10 infers
    # with; None uses the final weights.
    model_dir = DUKE_HEART.mgn_weights_dir
    epoch: Optional[int] = None

    # Gaussian sigma, in mm, that spreads the predicted surface displacements
    # into the continuous field the labelmap is resampled through.
    smoothing_sigma_mm = 10.0
    # Isotropic pitch every metric is measured on.  Coarser than these
    # labelmaps, whose in-plane pitch is finer than the accuracy being reported,
    # and still below the thinnest wall of the heart.
    evaluation_spacing_mm = 1.0

    output_dir = tutorials_dir / "output" / "tutorial_11_duke_heart" / case_id
    log_level = logging.INFO

    logging.basicConfig(level=log_level)
    logger = logging.getLogger(class_name)

    test_mode = TestTools.running_as_test()
    labelmap_dir = DUKE_HEART.hold_out_directory(test_mode) / case_id

    # Directory setup and data reading

    output_dir.mkdir(parents=True, exist_ok=True)

    reference_mesh_file = case_dir / f"{case_id}_ssm_surface.vtp"
    pca_file = case_dir / f"{case_id}_ssm_pca_coefficients.json"
    for required_file in (reference_mesh_file, pca_file):
        if not required_file.exists():
            raise FileNotFoundError(
                f"Tutorial 8 output not found: {required_file}\n"
                "Run tutorials/tutorial_08_duke_heart_fit_model_to_4d_patients.py "
                "first."
            )

    frame_files = sorted(labelmap_dir.glob(f"*{LABELMAP_SUFFIX}"))
    if not frame_files:
        raise FileNotFoundError(
            f"No gated labelmaps found in {labelmap_dir}.\n"
            "See data/Duke-Heart-4DLabelmaps/README.md."
        )
    reference_files = [
        path for path in frame_files if path.name.endswith(f"_ref{LABELMAP_SUFFIX}")
    ]
    if not reference_files:
        raise FileNotFoundError(
            f"No *_ref{LABELMAP_SUFFIX} frame in {labelmap_dir}; Tutorial 8 fitted "
            "the SSM to that frame, so it is the one the deformations start from."
        )

    # Step 1: ground truth, one labelmap per gated frame, as acquired.
    ground_truth_labelmaps: dict[float, itk.Image] = {
        _cardiac_stage_from_filename(frame_file): itk.imread(str(frame_file))
        for frame_file in frame_files
    }
    logger.info("Case %s: %d gated frames", case_id, len(ground_truth_labelmaps))

    # Step 2: score every frame, per chamber, against its own labelmap.  The
    # segmenter is instantiated for its taxonomy alone; no model is loaded.
    all_labels = SegmentHeartSimplewareTrimmedBranches(
        log_level=logging.WARNING
    ).taxonomy.all_labels()
    # The taxonomy's "heart" is the whole heart minus its chamber cavities --
    # the muscle the shape model represents -- so the report names it that way.
    heart_names = {
        label: "heart muscle" if all_labels[label] == "heart" else all_labels[label]
        for label in HEART_LABEL_IDS
    }

    infer_workflow = WorkflowInferPhysicsNeMo(
        model_directory=model_dir, epoch=epoch, log_level=log_level
    )
    evaluate_workflow = WorkflowEvaluateMovement(
        movement_workflow=WorkflowInferMovement(infer_workflow, log_level=log_level),
        label_names=heart_names,
        log_level=log_level,
    )
    result = evaluate_workflow.process(
        case_id=case_id,
        shape_parameters=pca_file,
        reference_mesh=reference_mesh_file,
        reference_labelmap=itk.imread(str(reference_files[0])),
        ground_truth_labelmaps=ground_truth_labelmaps,
        output_directory=output_dir,
        smoothing_sigma_mm=smoothing_sigma_mm,
        evaluation_spacing_mm=evaluation_spacing_mm,
    )

    # Step 3: the report and the CSV are written by the workflow.
    logger.info("Report: %s", result["report_file"])
    logger.info("Metrics: %s", result["csv_file"])

    tutorial_results: dict[str, Any] = dict(result)

    # Testing
    tt = TestTools(
        class_name=class_name,
        results_dir=output_dir,
        baselines_dir=repo_root / "tests" / "baselines" / class_name,
        log_level=log_level,
    )
    tutorial_results["screenshots"] = [
        tt.save_screenshot_mesh(
            cast(pv.DataSet, pv.read(str(result["predicted_surfaces"][0]))),
            "predicted_surface.png",
            camera_position="iso",
            color="limegreen",
        ),
        tt.save_screenshot_image_slice(
            itk.imread(str(result["warped_labelmaps"][0])),
            "warped_labelmap.png",
            axis=0,
            slice_fraction=0.5,
            colormap="viridis",
        ),
    ]
