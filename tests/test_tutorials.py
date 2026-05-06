"""Experiment tests that run each tutorial end-to-end and compare screenshots.

Each test class maps to one tutorial script.  Tests are gated behind
``--run-experiments`` (handled by conftest.py) and require the relevant dataset
to be present (see data/README.md).

Screenshot comparison uses the existing ITK-based baseline infrastructure:

1. The tutorial's ``run_tutorial()`` saves PNGs to the results directory.
2. Each PNG is read back with ``itk.imread`` (ITK handles PNG natively).
3. ``TestTools.write_result_image`` + ``compare_result_to_baseline_image`` compare
   the PNG against a stored baseline with loose per-pixel tolerances.

Run all tutorial tests::

    pytest tests/test_tutorials.py --run-experiments -v

Create baselines on first run::

    pytest tests/test_tutorials.py --run-experiments --create-baselines -v
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import itk
import pytest

from physiomotion4d.test_tools import TestTools

# Tolerances for screenshot comparison. Loose to survive minor rendering
# differences across OS / GPU / driver versions.
_PX_TOL = 10.0  # per-pixel absolute error (0-255 range)
_MAX_PX = 2000  # maximum number of pixels allowed above _PX_TOL
_TOT_TOL = float("inf")  # use the pixel-count criterion only


def _compare_screenshots(
    screenshots: list[Path],
    tt: TestTools,
) -> None:
    """Read each PNG as itk.Image and compare against baseline."""
    for png_path in screenshots:
        if not png_path.exists():
            pytest.fail(f"Screenshot not created: {png_path}")
        img = itk.imread(str(png_path))
        tt.write_result_image(img, png_path.name)
        assert tt.compare_result_to_baseline_image(
            png_path.name,
            per_pixel_absolute_error_tol=_PX_TOL,
            max_number_of_pixels_above_tol=_MAX_PX,
            total_absolute_error_tol=_TOT_TOL,
        ), f"Screenshot baseline mismatch: {png_path.name}"


# -----------------------------------------------------------------------------
# Tutorial 1 - Heart-Gated CT to Animated USD
# -----------------------------------------------------------------------------


@pytest.mark.experiment
@pytest.mark.requires_data
@pytest.mark.slow
class TestTutorial01HeartGatedCTToUSD:
    """End-to-end test for tutorial_01_heart_gated_ct_to_usd.py."""

    _class_name = "tutorial_01_heart_gated_ct_to_usd"

    def test_run(self, test_directories: dict[str, Path]) -> None:
        from tutorials.tutorial_01_heart_gated_ct_to_usd import run_tutorial

        out_dir = test_directories["output"] / self._class_name
        results: dict[str, Any] = run_tutorial(
            data_dir=test_directories["data"],
            output_dir=out_dir,
            registration_method="ants",
        )
        assert results["usd_file"], "USD file path should not be empty"
        assert Path(results["usd_file"]).exists(), "USD file should exist"

        tt = TestTools(
            class_name=self._class_name,
            results_dir=test_directories["output"],
            baselines_dir=test_directories["baselines"],
        )
        _compare_screenshots(results["screenshots"], tt)


# -----------------------------------------------------------------------------
# Tutorial 2 - CT Segmentation to VTK
# -----------------------------------------------------------------------------


@pytest.mark.experiment
@pytest.mark.requires_data
@pytest.mark.slow
class TestTutorial02CTToVTK:
    """End-to-end test for tutorial_02_ct_to_vtk.py."""

    _class_name = "tutorial_02_ct_to_vtk"

    def test_run(self, test_directories: dict[str, Path]) -> None:
        from tutorials.tutorial_02_ct_to_vtk import run_tutorial

        out_dir = test_directories["output"] / self._class_name
        results: dict[str, Any] = run_tutorial(
            data_dir=test_directories["data"],
            output_dir=out_dir,
        )
        assert results["surface_file"].exists(), "Combined VTP surface should exist"
        assert results["mesh_file"].exists(), "Combined VTU mesh should exist"

        tt = TestTools(
            class_name=self._class_name,
            results_dir=test_directories["output"],
            baselines_dir=test_directories["baselines"],
        )
        _compare_screenshots(results["screenshots"], tt)


# -----------------------------------------------------------------------------
# Tutorial 3 - Fit Statistical Model to Patient
# -----------------------------------------------------------------------------


@pytest.mark.experiment
@pytest.mark.requires_data
@pytest.mark.slow
class TestTutorial03FitStatisticalModelToPatient:
    """End-to-end test for tutorial_03_fit_statistical_model_to_patient.py."""

    _class_name = "tutorial_03_fit_statistical_model_to_patient"

    def test_run(self, test_directories: dict[str, Path]) -> None:
        kcl_dir = test_directories["data"] / "KCL-Heart-Model"
        if not (kcl_dir / "pca_mean.vtu").exists():
            pytest.skip(
                "KCL-Heart-Model not downloaded. See data/README.md for instructions."
            )

        from tutorials.tutorial_03_fit_statistical_model_to_patient import run_tutorial

        out_dir = test_directories["output"] / self._class_name
        results: dict[str, Any] = run_tutorial(
            data_dir=test_directories["data"],
            output_dir=out_dir,
        )
        assert results["registered_file"].exists(), "Registered VTP should exist"

        tt = TestTools(
            class_name=self._class_name,
            results_dir=test_directories["output"],
            baselines_dir=test_directories["baselines"],
        )
        _compare_screenshots(results["screenshots"], tt)


# -----------------------------------------------------------------------------
# Tutorial 4 - Create Statistical Shape Model
# -----------------------------------------------------------------------------


@pytest.mark.experiment
@pytest.mark.requires_data
@pytest.mark.slow
class TestTutorial04CreateStatisticalModel:
    """End-to-end test for tutorial_04_create_statistical_model.py."""

    _class_name = "tutorial_04_create_statistical_model"

    def test_run(self, test_directories: dict[str, Path]) -> None:
        kcl_dir = test_directories["data"] / "KCL-Heart-Model"
        if not (kcl_dir / "pca_mean.vtu").exists():
            pytest.skip(
                "KCL-Heart-Model not downloaded. See data/README.md for instructions."
            )

        from tutorials.tutorial_04_create_statistical_model import run_tutorial

        out_dir = test_directories["output"] / self._class_name
        results: dict[str, Any] = run_tutorial(
            data_dir=test_directories["data"],
            output_dir=out_dir,
            pca_components=5,
            max_samples=10,
        )
        assert results["model_file"].exists(), "pca_model.json should exist"
        assert results["mean_surface_file"].exists(), "Mean surface VTP should exist"

        tt = TestTools(
            class_name=self._class_name,
            results_dir=test_directories["output"],
            baselines_dir=test_directories["baselines"],
        )
        _compare_screenshots(results["screenshots"], tt)


# -----------------------------------------------------------------------------
# Tutorial 5 - VTK to USD
# -----------------------------------------------------------------------------


@pytest.mark.experiment
@pytest.mark.requires_data
@pytest.mark.slow
class TestTutorial05VTKToUSD:
    """End-to-end test for tutorial_05_vtk_to_usd.py."""

    _class_name = "tutorial_05_vtk_to_usd"

    def test_run(self, test_directories: dict[str, Path]) -> None:
        # Prefer Tutorial 2 output; fall back to any .vtp in data
        tutorial2_vtp = (
            test_directories["output"]
            / "tutorial_02_ct_to_vtk"
            / "patient_surfaces.vtp"
        )
        vtk_file = tutorial2_vtp if tutorial2_vtp.exists() else None
        if vtk_file is None:
            found = list(test_directories["data"].rglob("*.vtp"))
            if not found:
                pytest.skip(
                    "No VTK file available. Run Tutorial 2 first or place a .vtp "
                    "file under data/."
                )
            vtk_file = found[0]

        from tutorials.tutorial_05_vtk_to_usd import run_tutorial

        out_dir = test_directories["output"] / self._class_name
        results: dict[str, Any] = run_tutorial(
            data_dir=test_directories["data"],
            output_dir=out_dir,
            vtk_file=vtk_file,
        )
        assert results["usd_file"], "USD file path should not be empty"
        assert Path(results["usd_file"]).exists(), "USD file should exist"

        tt = TestTools(
            class_name=self._class_name,
            results_dir=test_directories["output"],
            baselines_dir=test_directories["baselines"],
        )
        _compare_screenshots(results["screenshots"], tt)


# -----------------------------------------------------------------------------
# Tutorial 6 - Reconstruct High-Resolution 4D CT
# -----------------------------------------------------------------------------


@pytest.mark.experiment
@pytest.mark.requires_data
@pytest.mark.slow
class TestTutorial06ReconstructHighres4DCT:
    """End-to-end test for tutorial_06_reconstruct_highres_4d_ct.py."""

    _class_name = "tutorial_06_reconstruct_highres_4d_ct"

    def test_run(self, test_directories: dict[str, Path]) -> None:
        dirlab_dir = test_directories["data"] / "DirLab-4DCT" / "Case1"
        if not dirlab_dir.exists():
            pytest.skip(
                "DirLab-4DCT Case1 not downloaded. See data/README.md for instructions."
            )

        from tutorials.tutorial_06_reconstruct_highres_4d_ct import run_tutorial

        out_dir = test_directories["output"] / self._class_name
        results: dict[str, Any] = run_tutorial(
            data_dir=test_directories["data"],
            output_dir=out_dir,
            case=1,
            max_frames=3,
            registration_method="ants",
        )
        assert results["reconstructed_files"], (
            "At least one reconstructed frame expected"
        )
        for f in results["reconstructed_files"]:
            assert f.exists(), f"Reconstructed frame missing: {f}"

        tt = TestTools(
            class_name=self._class_name,
            results_dir=test_directories["output"],
            baselines_dir=test_directories["baselines"],
        )
        _compare_screenshots(results["screenshots"], tt)
