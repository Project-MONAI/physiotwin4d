"""
Tutorial 10 (Duke Heart, MGN): Predict a Heart Surface at One Cardiac Stage

Purpose
-------
Final stage of the Duke heart 4D deep-learning pipeline (Tutorials 8 -> 9 -> 10),
the counterpart of ``tutorial_10_lung_infer_physicsnemo_mgn.py``.  A thin driver
over :class:`physiotwin4d.WorkflowInferPhysicsNeMo` and its displacement decoder
:class:`physiotwin4d.WorkflowInferMovement`:

1. Discover the per-frame SSM surfaces that Tutorial 8 (Duke Heart)
   (``tutorial_08_duke_heart_fit_model_to_4d_patients.py``) wrote for the test
   case, and pick the cardiac stage to predict.  Stages are parsed from the
   ``g{PPP}`` gate tag of the frame filenames.

2. Predict that case's surface at the chosen stage with the MeshGraphNet trained
   by Tutorial 9 (``tutorial_09_duke_heart_train_physicsnemo_mgn.py``).  The
   network predicts per-vertex displacements, so the decoder adds them to the
   case's reference SSM surface and scores the result in millimetres against the
   ground-truth frame surface.

3. Write the predicted surface as a USD (``WorkflowConvertVTKToUSD``, colored
   with the heart anatomy material).

For command-line use with path arguments, use the installed
``physiotwin4d-infer-physicsnemo`` CLI instead of editing this script.

Extra Install Required
----------------------
PhysicsNeMo and PyTorch Geometric must be installed::

    pip install "physiotwin4d[physicsnemo]"

Data Required
-------------
  * ``output/tutorial_08_duke_heart/<case>/`` - Tutorial 8 SSM surfaces
  * ``network_weights/physicsnemo_mgn_duke_heart_motion/mgn_stage_model.pt``
    - Tutorial 9 checkpoint
    (``ParametersDukeHeartLabelmaps.mgn_weights_dir``)

Outputs (under ``output/tutorial_10_duke_heart_mgn/<case>/``)
------------------------------------------------------------
  * ``<case>_ssm_pca_coefficients_pred_s{TTT}.vtp`` - predicted surface
  * ``<case>_mgn_s{TTT}.usd``                       - USD of that surface
"""

# Imports
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, cast

import pyvista as pv
from parameters_duke_heart_labelmaps import DUKE_HEART

from physiotwin4d import (
    TestTools,
    WorkflowConvertVTKToUSD,
    WorkflowInferMovement,
    WorkflowInferPhysicsNeMo,
)

# Gated frames carry a ``g{PPP}`` tag naming their percentage of the R-R
# interval; this is what a per-frame SSM surface is matched and staged by.
PHASE_SURFACE_PATTERN = "*_g[0-9][0-9][0-9]_*_ssm_surface.vtp"


def _cardiac_stage_from_filename(surface_file: Path) -> float:
    """Extract the normalized cardiac stage [0, 1] from a ``g{PPP}`` filename stem."""
    for part in surface_file.stem.split("_"):
        if part.startswith("g") and part[1:].isdigit():
            return int(part[1:]) / 100.0
    raise ValueError(f"Cannot parse cardiac gate from filename: {surface_file}")


# Only run if this script is not imported as a module

# PhysicsNeMo and torch spawn worker processes. On Windows the spawn start
# method re-imports this script in each child; without the
# __name__ == "__main__" guard around top-level work, that re-import would
# restart the prediction in every worker.
if __name__ == "__main__":
    # Data directory specification
    tutorials_dir = Path(__file__).resolve().parent
    # Fitted SSM surfaces and PCA coefficients written by Tutorial 8 (Duke Heart).
    data_dir = tutorials_dir / "output" / "tutorial_08_duke_heart"
    # The network Tutorial 9 (Duke Heart) trained.
    model_dir = DUKE_HEART.mgn_weights_dir
    # Intermittent-checkpoint epoch to load; None uses the final weights.
    epoch: Optional[int] = None

    # Case to predict; the held-out test case of Tutorial 9 (Duke Heart).
    case_id = DUKE_HEART.hold_out_case
    # Fraction through the case's ordered gated frames to predict.
    stage_fraction = 0.7

    output_dir = tutorials_dir / "output" / "tutorial_10_duke_heart_mgn" / case_id
    log_level = logging.INFO

    class_name = "tutorial_10_duke_heart_infer_physicsnemo"
    logging.basicConfig(level=log_level)
    logger = logging.getLogger(class_name)

    # Directory setup and data reading

    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_file = model_dir / "mgn_stage_model.pt"
    if not checkpoint_file.exists():
        raise FileNotFoundError(
            f"Tutorial 9 checkpoint not found: {checkpoint_file}\n"
            "Run tutorials/tutorial_09_duke_heart_train_physicsnemo_mgn.py first."
        )

    case_dir = data_dir / case_id
    reference_file = case_dir / f"{case_id}_ssm_surface.vtp"
    pca_file = case_dir / f"{case_id}_ssm_pca_coefficients.json"
    phase_files = sorted(case_dir.glob(PHASE_SURFACE_PATTERN))
    for required_file in (reference_file, pca_file):
        if not required_file.exists():
            raise FileNotFoundError(
                f"Tutorial 8 output not found: {required_file}\n"
                "Run tutorials/tutorial_08_duke_heart_fit_model_to_4d_patients.py "
                "first."
            )
    if not phase_files:
        raise FileNotFoundError(f"No gated frame surfaces found in {case_dir}")

    # Step 1: pick the test frame - the one 70% of the way through the case's
    # ordered gated frames - and read its stage and ground-truth surface.
    stages = [_cardiac_stage_from_filename(f) for f in phase_files]
    # Clamped, so a stage_fraction of 1.0 picks the last frame rather than one
    # past it.
    test_index = min(int(stage_fraction * len(stages)), len(stages) - 1)
    test_stage = stages[test_index]
    ground_truth_file = phase_files[test_index]
    logger.info(
        "Case %s: predicting stage %.2f (%s) of %d frames",
        case_id,
        test_stage,
        ground_truth_file.name,
        len(stages),
    )

    # Step 2: predict the case's surface at that stage with the trained
    # MeshGraphNet. The model predicts displacements, so the displacement
    # decoder adds them to the case's reference SSM surface and scores the
    # result against the ground-truth frame surface in millimetres.
    infer_workflow = WorkflowInferPhysicsNeMo(
        model_directory=model_dir, epoch=epoch, log_level=log_level
    )
    infer_result = WorkflowInferMovement(
        infer_workflow, log_level=log_level
    ).predict_single(
        shape_parameters=pca_file,
        stage=test_stage,
        reference_mesh=reference_file,
        ground_truth=ground_truth_file,
        output_directory=output_dir,
    )

    # Step 3: write the predicted surface as a USD, colored with the heart
    # anatomy material via USDAnatomyTools (appearance="anatomy").  The SSM is
    # one structure, the whole heart minus its chamber cavities, so the surface
    # is kept whole rather than split by connectivity.
    usd_workflow = WorkflowConvertVTKToUSD(
        input_meshes=[pv.read(str(infer_result["predicted_surface"]))],
        usd_project_name=f"{case_id}_mgn_s{int(test_stage * 100):03d}",
        output_directory=output_dir,
        appearance="anatomy",
        anatomy_type="heart",
        separate_by_connectivity=False,
        log_level=log_level,
    )
    usd_file = usd_workflow.process()["usd_file"]

    tutorial_results: dict[str, Any] = dict(infer_result)
    tutorial_results["stage"] = test_stage
    tutorial_results["ground_truth_file"] = ground_truth_file
    tutorial_results["usd_file"] = usd_file

    # Testing: render the predicted surface beside the ground-truth frame it is
    # scored against.
    tt = TestTools(
        class_name=class_name,
        results_dir=output_dir,
        baselines_dir=tutorials_dir.parent / "tests" / "baselines" / class_name,
        log_level=log_level,
    )
    tutorial_results["screenshots"] = [
        tt.save_screenshot_mesh(
            cast(pv.DataSet, pv.read(str(infer_result["predicted_surface"]))),
            "predicted_surface.png",
            camera_position="iso",
            color="limegreen",
        ),
        tt.save_screenshot_mesh(
            cast(pv.DataSet, pv.read(str(ground_truth_file))),
            "ground_truth_surface.png",
            camera_position="iso",
            color="steelblue",
        ),
    ]
