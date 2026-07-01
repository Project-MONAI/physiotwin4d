"""
Tutorial 8: Fit the Cardiac SSM and Propagate It Through Gated Phases

Purpose
-------
First stage of the cardiac 4D deep-learning pipeline (Tutorials 8 -> 9 -> 10).
For each patient it turns gated CT scans into the statistical-shape-model (SSM)
surfaces and volume meshes that the Tutorial 9 trainers
(``tutorial_09a_cardiac_train_physicsnemo_mgn.py`` /
``tutorial_09b_cardiac_train_physicsnemo_mlp.py``) consume:

1. Fit the KCL PCA heart model to the reference phase. A surface is extracted
   from the reference labelmap and the KCL PCA volume model is fitted with
   PCA-based registration (``WorkflowFitStatisticalModelToPatient`` with
   ``use_pca_registration=True``, surface fitting disabled). This yields the
   patient's PCA coefficients plus the fitted SSM volume mesh and surface, all
   sharing the model's fixed topology.

2. Propagate the fitted mesh to every gated phase. Each gated time point is
   registered to the reference with the deep-learning ICON registrar
   (``WorkflowReconstructHighres4DCT``). The forward transform for each phase
   warps the fitted SSM mesh and surface (``TransformTools.transform_pvcontour``,
   with deformation magnitude attached), producing one
   ``*_g{TT}_ssm_mesh.vtu`` / ``*_ssm_surface.vtp`` per phase.

Bring Your Own Data
-------------------
This is a bring-your-own-data tutorial. Unlike Tutorials 1-6, it does not use the
repository ``data/`` directory or a downloadable sample; the path constants below
point at a local ``D:/PhysioMotion4D/`` layout. Edit them to match your own data.

Data Required
-------------
  * ``D:/PhysioMotion4D/duke_data/gated_nii/pm00??/``       - gated NIfTI CT per patient
  * ``D:/PhysioMotion4D/duke_data/simple_ascardio/pm00??/`` - matching labelmaps
  * ``D:/PhysioMotion4D/kcl-heart-pca/pca-vol-kcl/``        - PCA model (pca_mean.vtu, pca_model.json)
  * ``D:/PhysioMotion4D/duke_data/icon_registration/``      - ICON registration weights

Outputs (per patient, under ``OUTPUT_DIR/pm00??/``)
---------------------------------------------------
  * ``*_ssm_pca_coefficients.json``           - fitted PCA coefficient vector
  * ``*_ssm_pca_mesh.vtu`` / ``*_ssm_pca_surface.vtp`` - PCA template before final warp
  * ``*_ssm_mesh.vtu`` / ``*_ssm_surface.vtp``         - fitted reference SSM mesh/surface
  * ``*_g{TT}_ssm_mesh.vtu`` / ``*_g{TT}_ssm_surface.vtp`` - SSM warped to gated phase TT%
  * ``*_g{TT}_warped_ref.mha``, ``*_g{TT}_*_tfm.hdf``  - registration artifacts
  * ``*_g{TT}_ref_labelmap.nii.gz``           - reference labelmap warped to each phase
"""

# %%
# Imports
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import itk
import pyvista as pv

from physiomotion4d import (
    ContourTools,
    TransformTools,
    WorkflowFitStatisticalModelToPatient,
    WorkflowReconstructHighres4DCT,
)
from physiomotion4d.test_tools import TestTools

# nnUNetv2 (used by TotalSegmentator inside several workflows) spawns a
# multiprocessing.Pool. On Windows the spawn start method re-imports this
# script in each child; without the __name__ == "__main__" guard around
# top-level work, that re-import fires the segmenter again and Python's
# spawn-cascade detector raises RuntimeError.
if __name__ == "__main__":
    # %%
    # Path configuration (bring-your-own-data: edit for your local layout)
    DATA_DIR = Path("D:/PhysioMotion4D/duke_data/gated_nii")
    LABELMAP_DIR = Path("D:/PhysioMotion4D/duke_data/simple_ascardio")
    SSM_MEAN_MESH_FILE = Path(
        "D:/PhysioMotion4D/kcl-heart-pca/pca-vol-kcl/pca_mean.vtu"
    )
    SSM_MODEL_FILE = Path("D:/PhysioMotion4D/kcl-heart-pca/pca-vol-kcl/pca_model.json")
    ICON_WEIGHTS_PATH = Path(
        "D:/PhysioMotion4D/duke_data/icon_registration/"
        "icon_ct_cardiac_gated_weights.trch"
    )
    # All outputs (fitted meshes, transforms, warped labelmaps) are written here;
    # this is also the directory the Tutorial 9 trainers read from.
    OUTPUT_DIR = Path("D:/PhysioMotion4D/duke_data/fitted_kcl_meshes")
    # Simpleware's heart interior chamber labels, excluded from the distance map.
    LABELMAP_INTERIOR_OBJECT_IDS = [1, 2, 3, 4]
    # Recompute the expensive fit/registration steps (True) or reload cached
    # results from OUTPUT_DIR (False).
    RECOMPUTE = True
    LOG_LEVEL = logging.INFO

    logging.basicConfig(level=LOG_LEVEL)
    logger = logging.getLogger("tutorial_08_cardiac_fit_model")

    # In test mode, limit the run to a single patient to keep it tractable.
    test_mode = TestTools.running_as_test()

    # %%
    # Load the statistical atlas model
    ssm_mean_mesh = pv.read(str(SSM_MEAN_MESH_FILE))
    with SSM_MODEL_FILE.open(encoding="utf-8") as f:
        ssm_model = json.load(f)

    # %%
    # Discover patients
    patient_dirs = sorted(DATA_DIR.glob("pm00??"))
    if test_mode:
        patient_dirs = patient_dirs[:1]

    tutorial_results: dict[str, Any] = {"patients": {}}

    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        logger.info("%s", "=" * 48)
        logger.info("Processing patient %s", patient_id)
        logger.info("%s", "=" * 48)

        patient_output_dir = OUTPUT_DIR / patient_id
        patient_output_dir.mkdir(parents=True, exist_ok=True)

        ref_image_files = list(patient_dir.glob("*ref.nii.gz"))
        if len(ref_image_files) != 1:
            raise ValueError(f"Expected 1 ref image file, found {len(ref_image_files)}")
        ref_image_file = ref_image_files[0]
        ref_image = itk.imread(str(ref_image_file))

        ref_labelmap_file = ref_image_file.name.replace(".nii.gz", "_labelmap.nii.gz")
        ref_labelmap = itk.imread(str(LABELMAP_DIR / patient_id / ref_labelmap_file))

        # %%
        # Step 1: fit the statistical model to the reference phase
        contour_tools = ContourTools()
        ref_surface = contour_tools.extract_contours(ref_labelmap)

        ssm_pca_coefficients_path = (
            patient_output_dir / f"{patient_id}_ssm_pca_coefficients.json"
        )
        ssm_mesh_path = patient_output_dir / f"{patient_id}_ssm_mesh.vtu"
        ssm_surface_path = patient_output_dir / f"{patient_id}_ssm_surface.vtp"

        if RECOMPUTE:
            ssm_fit_workflow = WorkflowFitStatisticalModelToPatient(
                template_model=ssm_mean_mesh,
                patient_image=ref_image,
                patient_models=[ref_surface],
                patient_labelmap=ref_labelmap,
                labelmap_interior_object_ids=LABELMAP_INTERIOR_OBJECT_IDS,
                log_level=LOG_LEVEL,
            )
            ssm_fit_workflow.set_use_pca_registration(
                use_pca_registration=True,
                pca_model=ssm_model,
                pca_uses_surface=False,
            )

            ssm_fit_workflow_result = ssm_fit_workflow.run_workflow()

            ssm_pca_coefficients = ssm_fit_workflow.pca_coefficients
            assert ssm_pca_coefficients is not None, (
                "pca_coefficients must be set after run_workflow() with "
                "use_pca_registration=True"
            )
            with ssm_pca_coefficients_path.open(mode="w", encoding="utf-8") as f:
                json.dump(ssm_pca_coefficients.tolist(), f)

            ssm_pca_template_model = ssm_fit_workflow.pca_template_model
            assert ssm_pca_template_model is not None
            ssm_pca_template_model.save(
                str(patient_output_dir / f"{patient_id}_ssm_pca_mesh.vtu")
            )

            ssm_pca_template_model_surface = ssm_fit_workflow.pca_template_model_surface
            assert ssm_pca_template_model_surface is not None
            ssm_pca_template_model_surface.save(
                str(patient_output_dir / f"{patient_id}_ssm_pca_surface.vtp")
            )

            ssm_mesh_fitted = ssm_fit_workflow_result["registered_template_model"]
            ssm_surface_fitted = ssm_fit_workflow_result[
                "registered_template_model_surface"
            ]

            ssm_mesh_fitted.save(str(ssm_mesh_path))
            ssm_surface_fitted.save(str(ssm_surface_path))
        else:
            ssm_mesh_fitted = pv.read(str(ssm_mesh_path))
            ssm_surface_fitted = pv.read(str(ssm_surface_path))

        # %%
        # Step 2: register every gated phase to the reference
        gated_files = sorted(
            file
            for file in patient_dir.glob("*.nii.gz")
            if file != ref_image_file and "nop" not in file.name and "_g" in file.stem
        )

        time_series = []
        time_series_ids = []
        for gated_file in gated_files:
            time_series.append(itk.imread(str(gated_file)))
            time_id = gated_file.name.split("_g")[1][:3]
            time_series_ids.append(time_id)

        if RECOMPUTE:
            reg_workflow = WorkflowReconstructHighres4DCT(
                time_series_images=time_series,
                fixed_image=ref_image,
                registration_method="ICON",
            )
            reg_workflow.registrar.registrar_ICON.set_weights_path(
                str(ICON_WEIGHTS_PATH)
            )
            reg_workflow.set_modality("ct")
            reg_workflow.set_number_of_iterations_ICON(None)
            reg_result = reg_workflow.run_workflow()

            reconstructed_images = reg_result["reconstructed_images"]
        else:
            reconstructed_images = []
            for time_id in time_series_ids:
                image_path = (
                    patient_output_dir / f"{patient_id}_g{time_id}_warped_ref.mha"
                )
                reconstructed_images.append(itk.imread(str(image_path)))

        # %%
        # Step 3: warp the fitted SSM mesh/surface to every gated phase
        phase_outputs = []
        for image_index, image in enumerate(reconstructed_images):
            time_id = time_series_ids[image_index]
            logger.info("Patient %s: warping to time point %s", patient_id, time_id)

            if RECOMPUTE:
                image_path = (
                    patient_output_dir / f"{patient_id}_g{time_id}_warped_ref.mha"
                )
                itk.imwrite(image, str(image_path), compression=True)

                fwd_tfm = reg_result["forward_transforms"][image_index]
                itk.transformwrite(
                    fwd_tfm,
                    str(
                        patient_output_dir / f"{patient_id}_g{time_id}_forward_tfm.hdf"
                    ),
                )

                inv_tfm = reg_result["inverse_transforms"][image_index]
                itk.transformwrite(
                    inv_tfm,
                    str(
                        patient_output_dir / f"{patient_id}_g{time_id}_inverse_tfm.hdf"
                    ),
                )

                # Warp the reference labelmap to this phase. Written under
                # OUTPUT_DIR (never back into the input labelmap directory).
                labelmap = TransformTools().transform_image(
                    ref_labelmap, inv_tfm, image, "nearest"
                )
                itk.imwrite(
                    labelmap,
                    str(
                        patient_output_dir
                        / f"{patient_id}_g{time_id}_ref_labelmap.nii.gz"
                    ),
                    compression=True,
                )
            else:
                fwd_tfm = itk.transformread(
                    str(patient_output_dir / f"{patient_id}_g{time_id}_forward_tfm.hdf")
                )

            mesh = TransformTools().transform_pvcontour(
                ssm_mesh_fitted, fwd_tfm, with_deformation_magnitude=True
            )
            mesh_path = patient_output_dir / f"{patient_id}_g{time_id}_ssm_mesh.vtu"
            mesh.save(str(mesh_path))

            surface = TransformTools().transform_pvcontour(
                ssm_surface_fitted, fwd_tfm, with_deformation_magnitude=True
            )
            surface_path = (
                patient_output_dir / f"{patient_id}_g{time_id}_ssm_surface.vtp"
            )
            surface.save(str(surface_path))
            phase_outputs.append({"time_id": time_id, "surface_file": surface_path})

        tutorial_results["patients"][patient_id] = {
            "pca_coefficients_file": ssm_pca_coefficients_path,
            "ssm_surface_file": ssm_surface_path,
            "phase_outputs": phase_outputs,
        }
