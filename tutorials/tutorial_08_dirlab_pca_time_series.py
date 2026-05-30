"""
Tutorial 8: Propagate DirLab PCA lung-lobe fits through each 4D CT time series.

This tutorial uses the per-case PCA-fitted reference meshes created by Tutorial
7. For each DirLab case, it registers every respiratory phase to the reference
phase used by Tutorial 7, saves the image transforms, and applies the
reference-to-phase transform to the fitted mesh so each time point has a
PCA-derived lung-lobe surface.

Data Required
-------------
See data/README.md for DirLab-4DCT download instructions. Run Tutorial 7 first
so ``output/tutorial_07_dirlab_pca_model/fits`` contains the per-case fitted
reference-stage meshes.
"""

# %%
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import itk
import pyvista as pv

from physiomotion4d.register_time_series_images import RegisterTimeSeriesImages
from physiomotion4d.test_tools import TestTools
from physiomotion4d.transform_tools import TransformTools


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
    OUTPUT_DIR = TUTORIALS_DIR / "output" / "tutorial_08_dirlab_pca_time_series"
    TUTORIAL_07_OUTPUT_DIR = TUTORIALS_DIR / "output" / "tutorial_07_dirlab_pca_model"
    CASE: Optional[int] = None
    LOG_LEVEL = logging.INFO

    DIRLAB_CASE_PREFIXES = [
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

    def run_tutorial() -> dict[str, Any]:
        """Run Tutorial 8: propagate Tutorial 7 PCA fits through DirLab time series.

        Returns
        -------
        dict[str, Any]
            Per-case transform filenames, propagated mesh filenames, and losses.
        """

        data_dir = TEST_DATA_DIR if TestTools.running_as_test() else FULL_DATA_DIR
        output_dir = OUTPUT_DIR
        tutorial_07_output_dir = TUTORIAL_07_OUTPUT_DIR
        case = CASE
        log_level = LOG_LEVEL

        output_dir.mkdir(parents=True, exist_ok=True)
        dirlab_dir = data_dir / "DirLab-4DCT"
        fits_dir = tutorial_07_output_dir / "fits"
        transform_tools = TransformTools(log_level=log_level)

        if case is None:
            case_numbers = list(range(1, 11))
        else:
            case_numbers = [case]

        tutorial_outputs: dict[str, Any] = {}
        for case_number in case_numbers:
            case_prefix = DIRLAB_CASE_PREFIXES[case_number - 1]
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
                print(f"Skipping {case_prefix}: no DirLab phase images found")
                continue

            fitted_mesh_file = fits_dir / f"{case_prefix}_lung_lobes_pca_fit.vtp"
            if not fitted_mesh_file.exists():
                raise FileNotFoundError(
                    f"Missing Tutorial 7 fitted mesh: {fitted_mesh_file}. "
                    "Run Tutorial 7 before Tutorial 8."
                )

            print(f"Registering {case_prefix}: {len(phase_files)} phases")
            case_output_dir = output_dir / case_prefix
            transforms_dir = case_output_dir / "transforms"
            meshes_dir = case_output_dir / "meshes"
            transforms_dir.mkdir(parents=True, exist_ok=True)
            meshes_dir.mkdir(parents=True, exist_ok=True)

            time_series = [itk.imread(str(phase_file)) for phase_file in phase_files]
            fixed_image = time_series[0]

            registrar = RegisterTimeSeriesImages(
                registration_method="ANTS_ICON",
                log_level=log_level,
            )
            registrar.set_modality("ct")
            registrar.set_fixed_image(fixed_image)
            registration_result = registrar.register_time_series(
                moving_images=time_series,
                reference_frame=0,
                register_reference=False,
            )

            fitted_reference_mesh = pv.read(str(fitted_mesh_file))
            case_transform_files: list[Path] = []
            case_mesh_files: list[Path] = []

            forward_transforms = registration_result["forward_transforms"]
            inverse_transforms = registration_result["inverse_transforms"]
            if not (
                len(phase_files) == len(forward_transforms) == len(inverse_transforms)
            ):
                raise ValueError(
                    f"{case_prefix}: length mismatch between phase_files "
                    f"({len(phase_files)}), forward_transforms "
                    f"({len(forward_transforms)}), and inverse_transforms "
                    f"({len(inverse_transforms)})."
                )

            for phase_file, phase_to_reference, reference_to_phase in zip(
                phase_files,
                forward_transforms,
                inverse_transforms,
            ):
                phase_name = phase_file.stem
                phase_to_reference_file = transforms_dir / (
                    f"{phase_name}_phase_to_reference.hdf"
                )
                reference_to_phase_file = transforms_dir / (
                    f"{phase_name}_reference_to_phase.hdf"
                )
                itk.transformwrite(
                    phase_to_reference,
                    str(phase_to_reference_file),
                    compression=True,
                )
                itk.transformwrite(
                    reference_to_phase,
                    str(reference_to_phase_file),
                    compression=True,
                )

                # Warp the reference-space fitted mesh into this phase's space.
                # Warping reference -> phase POINTS uses the forward transform
                # (the fixed -> moving point map), which is the opposite of the
                # transform used to warp an image into phase space (images pull
                # back, points push forward). The forward transform is named
                # "phase_to_reference" after its image-warp role. See
                # docs/developer/transform_conventions.
                phase_mesh = transform_tools.transform_pvcontour(
                    fitted_reference_mesh,
                    phase_to_reference,
                    with_deformation_magnitude=True,
                )
                phase_mesh_file = meshes_dir / f"{phase_name}_pca_fit.vtp"
                phase_mesh.save(phase_mesh_file)

                case_transform_files.extend(
                    [phase_to_reference_file, reference_to_phase_file]
                )
                case_mesh_files.append(phase_mesh_file)

            tutorial_outputs[case_prefix] = {
                "reference_phase": phase_files[0],
                "phase_files": phase_files,
                "fitted_reference_mesh": fitted_mesh_file,
                "transform_files": case_transform_files,
                "mesh_files": case_mesh_files,
                "losses": registration_result["losses"],
            }

        return tutorial_outputs

    # %%
    # Run this cell in VS Code or Cursor:
    tutorial_results = run_tutorial()
