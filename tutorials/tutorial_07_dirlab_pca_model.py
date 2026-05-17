"""Tutorial 7: Build and fit a PCA lung-lobe model from DirLab 4D CT cases.

This tutorial uses one respiratory phase from each available DirLab case. It
segments the five lung lobes, builds a surface PCA model from all but two
cases, then fits that model to every available case.
"""

# %%
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, cast

import itk
import numpy as np
import pyvista as pv

from physiomotion4d.contour_tools import ContourTools
from physiomotion4d.workflow_convert_image_to_vtk import WorkflowConvertImageToVTK
from physiomotion4d.workflow_create_statistical_model import (
    WorkflowCreateStatisticalModel,
)
from physiomotion4d.test_tools import TestTools
from physiomotion4d.workflow_fit_statistical_model_to_patient import (
    WorkflowFitStatisticalModelToPatient,
)


# nnUNetv2 (used by TotalSegmentator inside several workflows) spawns a
# multiprocessing.Pool. On Windows the spawn start method re-imports this
# script in each child; without the __name__ == "__main__" guard around
# top-level work, that re-import fires the segmenter again and Python's
# spawn-cascade detector raises RuntimeError. Wrapping consistently across
# tutorials also matches the style of tutorial_01.
if __name__ == "__main__":
    # %%
    REPO_ROOT = Path(__file__).resolve().parent.parent
    TUTORIALS_DIR = Path(__file__).resolve().parent
    DATA_DIR = REPO_ROOT / "data"
    FULL_DATA_DIR = DATA_DIR
    TEST_DATA_DIR = DATA_DIR / "test"
    OUTPUT_DIR = TUTORIALS_DIR / "output" / "tutorial_07_dirlab_pca_model"
    PCA_COMPONENTS = 5
    LOG_LEVEL = logging.INFO

    # %%
    def create_meshes(
        data_dir: Path,
        output_dir: Path,
        log_level: int = logging.INFO,
    ) -> dict[str, Path]:
        """Segment DirLab CT cases and save one five-lobe surface mesh per case.

        Parameters
        ----------
        data_dir
            Repository data directory containing ``DirLab-4DCT``.
        output_dir
            Directory where tutorial outputs are written.
        log_level
            Logging level for the conversion workflow.

        Returns
        -------
        dict[str, Path]
            Output directory and saved lung-lobe surface mesh filenames.
        """

        dirlab_dir = data_dir / "DirLab-4DCT"
        meshes_dir = output_dir / "meshes"
        meshes_dir.mkdir(parents=True, exist_ok=True)

        case_prefixes = [
            "Case1Pack",
            "Case2Pack",
            "Case3Pack",
            "Case4Pack",
            "Case5Pack",
            "Case6Pack",
            "Case7Pack",
            "Case8Deploy",
            "Case9Pack",
            "Case10Pack",
        ]
        lung_lobe_ids = {
            10: "lung_upper_lobe_left",
            11: "lung_lower_lobe_left",
            12: "lung_upper_lobe_right",
            13: "lung_middle_lobe_right",
            14: "lung_lower_lobe_right",
        }

        mesh_files: list[Path] = []
        contour_tools = ContourTools(log_level=log_level)
        for case_number, case_prefix in enumerate(case_prefixes, start=1):
            mesh_file = meshes_dir / f"{case_prefix}_lung_lobes.vtp"
            if mesh_file.exists():
                mesh_files.append(mesh_file)
                continue

            case_dir = dirlab_dir / f"Case{case_number}"
            phase_files = sorted(case_dir.glob("*.mha")) + sorted(
                case_dir.glob("*.mhd")
            )
            if not phase_files:
                phase_files = sorted(dirlab_dir.glob(f"{case_prefix}_T*.mha"))
                phase_files += sorted(dirlab_dir.glob(f"{case_prefix}_T*.mhd"))
            if not phase_files and case_number == 8:
                phase_files = sorted(dirlab_dir.glob("Case8Pack_T*.mha"))
                phase_files += sorted(dirlab_dir.glob("Case8Pack_T*.mhd"))
            if not phase_files:
                print(f"Skipping {case_prefix}: no DirLab phase image found")
                continue

            print(f"Segmenting {case_prefix} from {phase_files[0].name}")
            image = itk.imread(str(phase_files[0]))
            workflow = WorkflowConvertImageToVTK(
                segmentation_method="ChestTotalSegmentator",
                log_level=log_level,
            )
            result = workflow.run_workflow(
                input_image=image,
                contrast_enhanced_study=False,
                anatomy_groups=["lung"],
            )

            labelmap = result["labelmap"]
            labelmap_arr = itk.GetArrayFromImage(labelmap)
            lobe_surfaces: list[pv.PolyData] = []
            for label_id, lobe_name in lung_lobe_ids.items():
                lobe_arr = (labelmap_arr == label_id).astype(np.uint8)
                if int(lobe_arr.sum()) == 0:
                    print(f"Skipping {case_prefix}: missing {lobe_name}")
                    lobe_surfaces = []
                    break

                lobe_mask = itk.GetImageFromArray(lobe_arr)
                lobe_mask.CopyInformation(labelmap)
                lobe_surface = contour_tools.extract_contours(lobe_mask)
                lobe_surface.field_data["LungLobeName"] = np.array([lobe_name])
                lobe_surface.field_data["LungLobeLabel"] = np.array([label_id])
                lobe_surfaces.append(lobe_surface)

            if len(lobe_surfaces) != len(lung_lobe_ids):
                continue

            lung_surface = cast(
                pv.PolyData,
                pv.merge(lobe_surfaces, merge_points=False),
            )
            lung_surface.save(mesh_file)
            mesh_files.append(mesh_file)

        return {"meshes_dir": meshes_dir, **{mesh.stem: mesh for mesh in mesh_files}}

    def create_model(
        meshes_dir: Path,
        output_dir: Path,
        pca_components: int = 5,
        log_level: int = logging.INFO,
    ) -> dict[str, Any]:
        """Create a surface PCA model from all but the final two DirLab lobe meshes.

        Parameters
        ----------
        meshes_dir
            Directory containing lung-lobe surface files from :func:`create_meshes`.
        output_dir
            Directory where model outputs are written.
        pca_components
            Requested PCA component count.
        log_level
            Logging level for the model creation workflow.

        Returns
        -------
        dict[str, Any]
            PCA model, saved model filenames, training files, and all mesh filenames.
        """

        model_dir = output_dir / "pca_model"
        model_dir.mkdir(parents=True, exist_ok=True)

        mesh_files = sorted(meshes_dir.glob("*_lung_lobes.vtp"))
        if len(mesh_files) < 4:
            raise ValueError("At least four DirLab lung-lobe meshes are needed.")

        training_files = mesh_files[:-2]
        held_out_files = mesh_files[-2:]
        sample_meshes = [pv.read(str(mesh_file)) for mesh_file in training_files]
        reference_mesh = pv.read(str(training_files[0]))
        component_count = min(pca_components, max(1, len(sample_meshes) - 1))

        workflow = WorkflowCreateStatisticalModel(
            sample_meshes=sample_meshes,
            reference_mesh=reference_mesh,
            pca_number_of_components=component_count,
            solve_for_surface_pca=True,
            log_level=log_level,
        )
        result = workflow.run_workflow()

        mean_surface_file = model_dir / "pca_mean_surface.vtp"
        pca_model_file = model_dir / "pca_model.json"
        result["pca_mean_surface"].save(mean_surface_file)
        pca_model_file.write_text(
            json.dumps(result["pca_model"], indent=2),
            encoding="utf-8",
        )

        return {
            "pca_model": result["pca_model"],
            "mean_surface_file": mean_surface_file,
            "pca_model_file": pca_model_file,
            "training_files": training_files,
            "held_out_files": held_out_files,
            "mesh_files": mesh_files,
        }

    def fit_model(
        model_result: dict[str, Any],
        output_dir: Path,
        log_level: int = logging.INFO,
    ) -> dict[str, Path]:
        """Fit the surface PCA lung-lobe model to every available DirLab mesh.

        Parameters
        ----------
        model_result
            Dictionary returned by :func:`create_model`.
        output_dir
            Directory where fit outputs are written.
        log_level
            Logging level for the fitting workflow.

        Returns
        -------
        dict[str, Path]
            Saved fitted surface filenames.
        """

        fits_dir = output_dir / "fits"
        fits_dir.mkdir(parents=True, exist_ok=True)

        template_model = pv.read(str(model_result["mean_surface_file"]))
        fitted_files: dict[str, Path] = {}

        for patient_file in model_result["mesh_files"]:
            patient_model = pv.read(str(patient_file))
            workflow = WorkflowFitStatisticalModelToPatient(
                template_model=template_model,
                patient_models=[patient_model],
                log_level=log_level,
            )
            workflow.set_use_pca_registration(
                True,
                pca_model=model_result["pca_model"],
                pca_number_of_modes=0,
                pca_uses_surface=True,
            )
            workflow.set_use_mask_to_mask_registration(False)

            result = workflow.run_workflow()
            fitted_surface = result["registered_template_model_surface"]
            fitted_file = fits_dir / f"{patient_file.stem}_pca_fit.vtp"
            fitted_surface.save(fitted_file)
            fitted_files[patient_file.stem] = fitted_file

        return fitted_files

    def run_tutorial() -> dict[str, Any]:
        """Run mesh creation, PCA model creation, and PCA fitting in sequence.

        Returns
        -------
        dict[str, Any]
            Mesh, PCA model, and fitted model output information.
        """

        data_dir = TEST_DATA_DIR if TestTools.running_as_test() else FULL_DATA_DIR
        output_dir = OUTPUT_DIR
        pca_components = PCA_COMPONENTS
        log_level = LOG_LEVEL

        output_dir.mkdir(parents=True, exist_ok=True)
        mesh_result = create_meshes(data_dir, output_dir, log_level=log_level)
        model_result = create_model(
            mesh_result["meshes_dir"],
            output_dir,
            pca_components=pca_components,
            log_level=log_level,
        )
        fit_result = fit_model(model_result, output_dir, log_level=log_level)

        return {
            "mesh_result": mesh_result,
            "model_result": model_result,
            "fit_result": fit_result,
        }

    # %%
    # Run this cell in VS Code or Cursor:
    tutorial_results = run_tutorial()
