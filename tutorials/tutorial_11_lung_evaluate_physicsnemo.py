"""
Tutorial 11 (Lung, MGN): Score Predicted Lung Motion Per Lobe

Purpose
-------
Measures how close the size and shape of the lung inferred by Tutorial 10 are to
the lung actually imaged, one respiratory phase at a time and one lobe at a
time.  The case is ``ParametersLungCTDirLab.mgn_hold_out_case``, held out of the
Tutorial 9 training, so this scores generalization rather than recall.

1. Build the ground truth: segment every gated CT frame of the case
   independently, giving one labelmap per respiratory phase whose lobes were
   never seen by the shape model or by the network.

2. Score the prediction: :class:`physiotwin4d.WorkflowEvaluateMovement` carries
   the reference phase's labelmap into every other phase with the network's own
   deformation, and compares the result to that phase's segmentation --- volume
   difference and surface RMSE per lobe.  Dice is left out: a lobe barely
   changes shape over a breath compared to how big it is, so the overlap
   fraction stays above 0.96 however well or badly the motion is predicted, and
   describes the lobe rather than the motion.

3. Write ``evaluation_report.md`` and ``evaluation_metrics.csv``, both carrying
   the hold-out case name, the case's shape parameters, and the network weights
   path with its dates.

Step 1 costs one segmentation pass per phase on first run and is cached
afterwards; Step 2 is the workflow, so this script only chooses the case, the
lobes and the ground truth.

Extra Install Required
----------------------
PhysicsNeMo and PyTorch Geometric must be installed::

    pip install "physiotwin4d[physicsnemo]"

Data Required
-------------
  * ``data/DirLab-4DCT/<case>_T??.mha``  - the gated CT sequence
  * ``output/tutorial_08_lung/<case>/``  - Tutorial 8 SSM surface + coefficients
  * ``network_weights/physicsnemo_mgn_lung_motion/`` - Tutorial 9 checkpoint

Outputs (under ``output/tutorial_11_lung/<case>/``)
---------------------------------------------------
  * ``evaluation_report.md``    - per-lobe accuracy of the prediction
  * ``evaluation_metrics.csv``  - one row per stage and lobe
  * ``volume_vs_stage.png``     - each lobe's volume across the stages
  * ``ground_truth/<case>_T{PP}_labelmap.nii.gz`` - cached per-phase segmentation
  * ``<case>_ssm_pca_coefficients_s{TTT}_pred.vtp`` - predicted surface per stage
"""

# Imports
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, cast

import itk
import pyvista as pv
from parameters_lung_ct_dirlab import LUNG_CT_DIRLAB

from physiotwin4d import (
    SegmentNVSegmentCTMRI,
    TestTools,
    WorkflowEvaluateMovement,
    WorkflowInferMovement,
    WorkflowInferPhysicsNeMo,
)

# The five lobes of ``SegmentNVSegmentCTMRI``.  Its "lung" group also carries
# whole-lung, tumor and airway labels, which are not lobes.
LOBE_LABEL_IDS = [28, 29, 30, 31, 32]


def _respiratory_stage_from_filename(image_file: Path) -> float:
    """Extract the normalized respiratory stage [0, 1] from a ``T{PP}`` filename stem."""
    for part in image_file.stem.split("_"):
        if part.startswith("T") and part[1:].isdigit():
            return int(part[1:]) / 100.0
    raise ValueError(f"Cannot parse respiratory phase from filename: {image_file}")


# Only run if this script is not imported as a module

# nnUNetv2 and torch spawn worker processes. On Windows the spawn start method
# re-imports this script in each child; without the __name__ == "__main__" guard
# around top-level work, that re-import would restart the whole evaluation in
# every worker.
if __name__ == "__main__":
    # Data directory specification
    repo_root = Path(__file__).resolve().parent.parent
    tutorials_dir = Path(__file__).resolve().parent

    class_name = "tutorial_11_lung_evaluate_physicsnemo"

    # Case to score: the case Tutorial 9 held out of training.
    case_id = LUNG_CT_DIRLAB.mgn_hold_out_case
    # Phase Tutorial 8 fitted the SSM to, and therefore the phase whose anatomy
    # the predicted deformations carry into every other phase.
    reference_phase = "T70"

    # Fitted SSM surface and PCA coefficients written by Tutorial 8 (lung).
    case_dir = tutorials_dir / "output" / "tutorial_08_lung" / case_id
    # Weights Tutorial 9 trained, and the checkpoint epoch Tutorial 10 infers
    # with; None uses the final weights.
    model_dir = LUNG_CT_DIRLAB.mgn_weights_dir
    epoch: Optional[int] = None

    # Gaussian sigma, in mm, that spreads the predicted surface displacements
    # into the continuous field the labelmap is resampled through.
    smoothing_sigma_mm = 10.0
    # Isotropic pitch every metric is measured on.  Coarser than the CT, whose
    # in-plane pitch is finer than the accuracy being reported, and fine enough
    # that a lobe boundary is not quantized away.
    evaluation_spacing_mm = 2.0

    output_dir = tutorials_dir / "output" / "tutorial_11_lung" / case_id
    ground_truth_dir = output_dir / "ground_truth"
    log_level = logging.INFO

    logging.basicConfig(level=log_level)
    logger = logging.getLogger(class_name)

    test_mode = TestTools.running_as_test()
    data_dir = LUNG_CT_DIRLAB.input_directory(test_mode)

    # Directory setup and data reading

    ground_truth_dir.mkdir(parents=True, exist_ok=True)

    reference_mesh_file = case_dir / f"{case_id}_ssm_surface.vtp"
    pca_file = case_dir / f"{case_id}_ssm_pca_coefficients.json"
    for required_file in (reference_mesh_file, pca_file):
        if not required_file.exists():
            raise FileNotFoundError(
                f"Tutorial 8 output not found: {required_file}\n"
                "Run tutorials/tutorial_08_lung_fit_model_to_4d_patients.py first."
            )

    frame_files = sorted(data_dir.glob(f"{case_id}_T??.mha"))
    if not frame_files:
        raise FileNotFoundError(
            f"No {case_id}_T??.mha frames found under {data_dir}.\n"
            "See data/DirLab-4DCT/README.md for download instructions."
        )

    # Step 1: ground truth.  Every gated frame is segmented on its own, so the
    # lobes each phase is scored against came from that phase's image rather
    # than from a registration or a shape-model fit.  Segmentation dominates the
    # runtime, so each labelmap is cached and reused on a re-run.
    segmenter = SegmentNVSegmentCTMRI(log_level=log_level)
    ground_truth_labelmaps: dict[float, itk.Image] = {}
    for frame_file in frame_files:
        labelmap_file = ground_truth_dir / f"{frame_file.stem}_labelmap.nii.gz"
        if not labelmap_file.exists():
            logger.info("Segmenting ground-truth frame %s", frame_file.name)
            segmentation_result = segmenter.segment(itk.imread(str(frame_file)))
            itk.imwrite(
                segmentation_result["labelmap"], str(labelmap_file), compression=True
            )
        ground_truth_labelmaps[_respiratory_stage_from_filename(frame_file)] = (
            itk.imread(str(labelmap_file))
        )

    reference_labelmap_file = (
        ground_truth_dir / f"{case_id}_{reference_phase}_labelmap.nii.gz"
    )
    if not reference_labelmap_file.exists():
        raise FileNotFoundError(
            f"Reference phase {reference_phase} is not among {data_dir}'s frames, "
            f"so {reference_labelmap_file} was never segmented."
        )

    # Step 2: score every phase, per lobe, against its own segmentation.
    all_labels = segmenter.taxonomy.all_labels()
    lobe_names = {label: all_labels[label] for label in LOBE_LABEL_IDS}

    infer_workflow = WorkflowInferPhysicsNeMo(
        model_directory=model_dir, epoch=epoch, log_level=log_level
    )
    evaluate_workflow = WorkflowEvaluateMovement(
        movement_workflow=WorkflowInferMovement(infer_workflow, log_level=log_level),
        label_names=lobe_names,
        log_level=log_level,
    )
    result = evaluate_workflow.process(
        case_id=case_id,
        shape_parameters=pca_file,
        reference_mesh=reference_mesh_file,
        reference_labelmap=itk.imread(str(reference_labelmap_file)),
        ground_truth_labelmaps=ground_truth_labelmaps,
        output_directory=output_dir,
        smoothing_sigma_mm=smoothing_sigma_mm,
        evaluation_spacing_mm=evaluation_spacing_mm,
        # A lobe barely changes shape over a breath compared to how big it is,
        # so Dice says more about the lobe than about the motion. Volume
        # difference and surface RMSE are what resolve it here.
        include_dice=False,
    )

    # Step 3: the report and the CSV are written by the workflow.
    logger.info("Report: %s", result["report_file"])
    logger.info("Metrics: %s", result["csv_file"])

    tutorial_results: dict[str, Any] = dict(result)
    tutorial_results["ground_truth_labelmap_dir"] = ground_truth_dir

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
