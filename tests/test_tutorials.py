"""Tutorial tests that run each tutorial end-to-end and compare screenshots.

Each test class maps to one tutorial script.  Tests are gated behind
``--run-tutorials`` (handled by conftest.py) and require the relevant dataset
to be present (see data/README.md).

Screenshot comparison uses the existing ITK-based baseline infrastructure:

1. Each tutorial script saves PNGs directly to its ``OUTPUT_DIR``.
2. ``TestTools.compare_result_to_baseline_image`` reads each PNG from that
   directory and compares it against a stored baseline with loose tolerances.

Run all tutorial tests::

    pytest tests/test_tutorials.py --run-tutorials -v

Create baselines on first run::

    pytest tests/test_tutorials.py --run-tutorials --create-baselines -v
"""

from __future__ import annotations

import importlib.util
import runpy
from pathlib import Path
from typing import Any

import pytest

from physiotwin4d.test_tools import TestTools

# Tolerances for screenshot comparison. Loose to survive minor rendering
# differences across OS / GPU / driver versions.
_PX_TOL = 10.0  # per-pixel absolute error (0-255 range)
_MAX_PX = 2000  # maximum number of pixels allowed above _PX_TOL
_TOT_TOL = float("inf")  # use the pixel-count criterion only
_REPO_ROOT = Path(__file__).parent.parent


@pytest.fixture(autouse=True)
def _enable_tutorial_test_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run tutorials against repo data/test through TestTools mode switching."""
    monkeypatch.setenv("PHYSIOTWIN_RUNNING_AS_TEST", "1")


def _compare_screenshots(
    screenshots: list[Path],
    tt: TestTools,
) -> None:
    """Read each PNG as itk.Image and compare against baseline."""
    if not screenshots:
        pytest.fail("No screenshots produced by tutorial script")

    for png_path in screenshots:
        if not png_path.exists():
            pytest.fail(f"Screenshot not created: {png_path}")
        assert tt.compare_result_to_baseline_image(
            png_path.name,
            per_pixel_absolute_error_tol=_PX_TOL,
            max_number_of_pixels_above_tol=_MAX_PX,
            total_absolute_error_tol=_TOT_TOL,
        ), f"Screenshot baseline mismatch: {png_path.name}"


def _run_tutorial_script(script_name: str) -> dict[str, Any]:
    """Run a tutorial script with no command-line arguments."""
    namespace = runpy.run_path(
        str(_REPO_ROOT / "tutorials" / script_name),
        run_name="__main__",
    )
    results = namespace.get("tutorial_results")
    assert isinstance(results, dict), f"{script_name} did not set tutorial_results"
    return results


@pytest.mark.tutorial
@pytest.mark.slow
class TestTutorial01HeartGatedCTToUSD:
    """End-to-end test for tutorial_01_heart_gated_ct_to_usd.py."""

    _class_name = "tutorial_01_heart_gated_ct_to_usd"

    def test_run(self, test_directories: dict[str, Path]) -> None:
        out_dir = _REPO_ROOT / "tutorials" / "output" / "tutorial_01_heart"
        results = _run_tutorial_script("tutorial_01_heart_gated_ct_to_usd.py")
        assert results["usd_file"], "USD file path should not be empty"
        assert Path(results["usd_file"]).exists(), "USD file should exist"
        assert results["screenshots"], "Tutorial 1 should produce screenshots"

        tt = TestTools(
            class_name=self._class_name,
            results_dir=out_dir,
            baselines_dir=test_directories["baselines"] / self._class_name,
        )
        _compare_screenshots(results["screenshots"], tt)


# -----------------------------------------------------------------------------
# Tutorial 3 - Reconstruct High-Resolution 4D CT
# -----------------------------------------------------------------------------


@pytest.mark.tutorial
@pytest.mark.slow
class TestTutorial03HeartReconstructHighres4DCT:
    """End-to-end test for tutorial_03_heart_reconstruct_highres_4d_ct.py."""

    _class_name = "tutorial_03_heart_reconstruct_highres_4d_ct"

    def test_run(
        self, test_directories: dict[str, Path], test_images: list[Any]
    ) -> None:
        out_dir = _REPO_ROOT / "tutorials" / "output" / "tutorial_03_heart"
        results = _run_tutorial_script("tutorial_03_heart_reconstruct_highres_4d_ct.py")
        assert results["reconstructed_files"], (
            "At least one reconstructed frame expected"
        )
        for f in results["reconstructed_files"]:
            assert f.exists(), f"Reconstructed frame missing: {f}"

        tt = TestTools(
            class_name=self._class_name,
            results_dir=out_dir,
            baselines_dir=test_directories["baselines"] / self._class_name,
        )
        _compare_screenshots(results["screenshots"], tt)


@pytest.mark.tutorial
@pytest.mark.slow
class TestTutorial03LungReconstructHighres4DCT:
    """End-to-end test for tutorial_03_lung_reconstruct_highres_4d_ct.py."""

    _class_name = "tutorial_03_lung_reconstruct_highres_4d_ct"

    def test_run(self, test_directories: dict[str, Path]) -> None:
        # Match the phase files the script itself globs, not a directory layout
        # it never uses.
        dirlab_dir = test_directories["data"] / "DirLab-4DCT"
        if not list(dirlab_dir.glob("Case1Pack_T??.mha")):
            pytest.skip(
                "DirLab-4DCT Case1Pack phases not downloaded. See data/README.md "
                "for instructions."
            )

        out_dir = _REPO_ROOT / "tutorials" / "output" / "tutorial_03_lung"
        results = _run_tutorial_script("tutorial_03_lung_reconstruct_highres_4d_ct.py")
        assert results["reconstructed_files"], (
            "At least one reconstructed frame expected"
        )
        for f in results["reconstructed_files"]:
            assert f.exists(), f"Reconstructed frame missing: {f}"

        tt = TestTools(
            class_name=self._class_name,
            results_dir=out_dir,
            baselines_dir=test_directories["baselines"] / self._class_name,
        )
        _compare_screenshots(results["screenshots"], tt)


# -----------------------------------------------------------------------------
# Tutorial 4 - CT Segmentation to VTK
# -----------------------------------------------------------------------------


@pytest.mark.tutorial
@pytest.mark.slow
class TestTutorial04HeartCTToVTK:
    """End-to-end test for tutorial_04_heart_ct_to_vtk.py."""

    _class_name = "tutorial_04_heart_ct_to_vtk"

    def test_run(
        self, test_directories: dict[str, Path], test_images: list[Any]
    ) -> None:
        out_dir = _REPO_ROOT / "tutorials" / "output" / "tutorial_04_heart"
        results = _run_tutorial_script("tutorial_04_heart_ct_to_vtk.py")
        assert results["surface_file"].exists(), "Combined VTP surface should exist"

        tt = TestTools(
            class_name=self._class_name,
            results_dir=out_dir,
            baselines_dir=test_directories["baselines"] / self._class_name,
        )
        _compare_screenshots(results["screenshots"], tt)


# -----------------------------------------------------------------------------
# Tutorial 5 - VTK to USD
# -----------------------------------------------------------------------------


@pytest.mark.tutorial
@pytest.mark.slow
class TestTutorial05HeartVTKToUSD:
    """End-to-end test for tutorial_05_heart_vtk_to_usd.py."""

    _class_name = "tutorial_05_heart_vtk_to_usd"

    def test_run(
        self, test_directories: dict[str, Path], test_images: list[Any]
    ) -> None:
        # The script reads this exact directory and offers no input override,
        # so bootstrap Tutorial 4 rather than pointing it at other surfaces.
        input_dir = _REPO_ROOT / "tutorials" / "output" / "tutorial_04_heart"
        if not list(input_dir.glob("patient_*.vtp")):
            _run_tutorial_script("tutorial_04_heart_ct_to_vtk.py")
            assert list(input_dir.glob("patient_*.vtp")), (
                f"Tutorial 4 bootstrap did not create surfaces in: {input_dir}"
            )

        out_dir = _REPO_ROOT / "tutorials" / "output" / "tutorial_05_heart"
        results = _run_tutorial_script("tutorial_05_heart_vtk_to_usd.py")
        assert results["usd_file"], "USD file path should not be empty"
        assert Path(results["usd_file"]).exists(), "USD file should exist"
        assert len(results["structures"]) > 1, (
            "Per-structure surfaces expected, so that each becomes its own prim"
        )

        tt = TestTools(
            class_name=self._class_name,
            results_dir=out_dir,
            baselines_dir=test_directories["baselines"] / self._class_name,
        )
        _compare_screenshots(results["screenshots"], tt)


# -----------------------------------------------------------------------------
# Tutorial 6 - Create Statistical Shape Model
# -----------------------------------------------------------------------------


@pytest.mark.tutorial
@pytest.mark.slow
class TestTutorial06CreateStatisticalModel:
    """End-to-end test for tutorial_06_heart_create_statistical_model.py."""

    _class_name = "tutorial_06_heart_create_statistical_model"

    def test_run(
        self, test_directories: dict[str, Path], download_kcl_heart_model: Path
    ) -> None:
        out_dir = _REPO_ROOT / "tutorials" / "output" / "tutorial_06_heart"
        results = _run_tutorial_script("tutorial_06_heart_create_statistical_model.py")
        assert results["model_file"].exists(), "pca_model.json should exist"
        assert results["mean_surface_file"].exists(), "Mean surface VTP should exist"

        tt = TestTools(
            class_name=self._class_name,
            results_dir=out_dir,
            baselines_dir=test_directories["baselines"] / self._class_name,
        )
        _compare_screenshots(results["screenshots"], tt)


@pytest.mark.tutorial
@pytest.mark.slow
class TestTutorial07FitStatisticalModelToPatient:
    """End-to-end test for tutorial_07_heart_fit_statistical_model_to_patient.py."""

    _class_name = "tutorial_07_heart_fit_statistical_model_to_patient"

    def test_run(
        self, test_directories: dict[str, Path], download_kcl_heart_model: Path
    ) -> None:
        # The patient scan comes from DIR-Lab, which must be acquired manually.
        if not (
            test_directories["data"] / "DirLab-4DCT" / "Case1Pack_T70.mha"
        ).exists():
            pytest.skip(
                "DirLab-4DCT Case1Pack_T70 not downloaded. See data/README.md "
                "for instructions."
            )

        pca_json = (
            _REPO_ROOT / "tutorials" / "output" / "tutorial_06_heart" / "pca_model.json"
        )
        if not pca_json.exists():
            _run_tutorial_script("tutorial_06_heart_create_statistical_model.py")
            assert pca_json.exists(), (
                "Tutorial 6 bootstrap did not create the expected PCA model file: "
                f"{pca_json}"
            )

        out_dir = _REPO_ROOT / "tutorials" / "output" / "tutorial_07_heart"
        results = _run_tutorial_script(
            "tutorial_07_heart_fit_statistical_model_to_patient.py"
        )
        # ``out_dir.name`` is the tutorial's ``project_name`` file prefix.
        registered_surface_file = (
            out_dir / f"{out_dir.name}_template_surface_registered.vtp"
        )
        assert registered_surface_file.exists(), "Registered surface VTP should exist"

        tt = TestTools(
            class_name=self._class_name,
            results_dir=out_dir,
            baselines_dir=test_directories["baselines"] / self._class_name,
        )
        _compare_screenshots(results["screenshots"], tt)


# -----------------------------------------------------------------------------
# Tutorials 9 and 10 - PhysicsNeMo train and infer
#
# Both need the optional [physicsnemo] extra and the Tutorial 8 fitted meshes,
# so they skip rather than fail when either is absent.
# -----------------------------------------------------------------------------


def _require_physicsnemo_and_tutorial_08() -> Path:
    """Skip unless the MGN dependencies and three Tutorial 8 cases are present."""
    if importlib.util.find_spec("physicsnemo") is None:
        pytest.skip("PhysicsNeMo not installed (optional [physicsnemo] extra).")
    if importlib.util.find_spec("torch_geometric") is None:
        pytest.skip(
            "PyTorch Geometric not installed; the MGN trainer needs it in addition "
            'to PhysicsNeMo. Install with: pip install "physiotwin4d[physicsnemo]" '
            "&& pip install torch-geometric"
        )
    data_dir = _REPO_ROOT / "tutorials" / "output" / "tutorial_08_lung"
    if len(list(data_dir.glob("Case*Pack"))) < 3:
        pytest.skip(
            "Fewer than three Tutorial 8 cases under tutorials/output/tutorial_08_lung. "
            "Run tutorial_08_lung_fit_model_to_4d_patients.py first."
        )
    return data_dir


@pytest.mark.tutorial
@pytest.mark.slow
@pytest.mark.requires_physicsnemo
class TestTutorial09LungTrainPhysicsNeMoMGN:
    """End-to-end test for tutorial_09_lung_train_physicsnemo_mgn.py."""

    _class_name = "tutorial_09_lung_train_physicsnemo_mgn"

    def test_run(self, test_directories: dict[str, Path]) -> None:
        _require_physicsnemo_and_tutorial_08()

        results = _run_tutorial_script("tutorial_09_lung_train_physicsnemo_mgn.py")
        model_dir = Path(results["model_directory"])
        assert (model_dir / "mgn_stage_model.pt").exists(), "Checkpoint should exist"
        assert results["cases"], "At least one held-out case should be evaluated"

        # The model goes to the shared weights directory; the manifests, the
        # evaluation and the screenshots stay under the tutorial's output.
        tt = TestTools(
            class_name=self._class_name,
            results_dir=_REPO_ROOT / "tutorials" / "output" / "tutorial_09_lung_mgn",
            baselines_dir=test_directories["baselines"] / self._class_name,
        )
        _compare_screenshots(results["screenshots"], tt)


@pytest.mark.tutorial
@pytest.mark.slow
@pytest.mark.requires_physicsnemo
class TestTutorial10LungInferPhysicsNeMoMGN:
    """End-to-end test for tutorial_10_lung_infer_physicsnemo_mgn.py."""

    _class_name = "tutorial_10_lung_infer_physicsnemo_mgn"

    def test_run(self, test_directories: dict[str, Path]) -> None:
        _require_physicsnemo_and_tutorial_08()

        # ParametersLungCTDirLab.mgn_weights_dir, where Tutorial 9 trains to.
        model_dir = (
            _REPO_ROOT / "tutorials" / "network_weights" / "physicsnemo_mgn_lung_motion"
        )
        if not (model_dir / "mgn_stage_model.pt").exists():
            _run_tutorial_script("tutorial_09_lung_train_physicsnemo_mgn.py")
            assert (model_dir / "mgn_stage_model.pt").exists(), (
                f"Tutorial 9 bootstrap did not create a checkpoint under {model_dir}"
            )

        results = _run_tutorial_script("tutorial_10_lung_infer_physicsnemo_mgn.py")
        assert Path(results["predicted_surface"]).exists(), (
            "Predicted surface should exist"
        )
        assert Path(results["usd_file"]).exists(), "USD file should exist"

        out_dir = (
            _REPO_ROOT / "tutorials" / "output" / "tutorial_10_lung_mgn" / "Case1Pack"
        )
        tt = TestTools(
            class_name=self._class_name,
            results_dir=out_dir,
            baselines_dir=test_directories["baselines"] / self._class_name,
        )
        _compare_screenshots(results["screenshots"], tt)
