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
import sys
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

    _class_name = "tutorial_01"

    def test_run(self, test_directories: dict[str, Path]) -> None:
        out_dir = _REPO_ROOT / "tutorials" / "output" / "tutorial_01"
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
# Tutorial 2 - CT Segmentation to VTK
# -----------------------------------------------------------------------------


@pytest.mark.tutorial
@pytest.mark.slow
class TestTutorial02CTToVTK:
    """End-to-end test for tutorial_02_ct_to_vtk.py."""

    _class_name = "tutorial_02_ct_to_vtk"

    def test_run(self, test_directories: dict[str, Path]) -> None:
        out_dir = _REPO_ROOT / "tutorials" / "output" / "tutorial_02"
        results = _run_tutorial_script("tutorial_02_ct_to_vtk.py")
        assert results["surface_file"].exists(), "Combined VTP surface should exist"
        assert results["mesh_file"].exists(), "Combined VTU mesh should exist"

        tt = TestTools(
            class_name=self._class_name,
            results_dir=out_dir,
            baselines_dir=test_directories["baselines"] / self._class_name,
        )
        _compare_screenshots(results["screenshots"], tt)


# -----------------------------------------------------------------------------
# Tutorial 3 - Create Statistical Shape Model
# -----------------------------------------------------------------------------


@pytest.mark.tutorial
@pytest.mark.slow
class TestTutorial03CreateStatisticalModel:
    """End-to-end test for tutorial_03_create_statistical_model.py."""

    _class_name = "tutorial_03_create_statistical_model"

    def test_run(self, test_directories: dict[str, Path]) -> None:
        kcl_dir = test_directories["data"] / "KCL-Heart-Model"
        if not (kcl_dir / "pca_mean.vtu").exists():
            pytest.skip(
                "KCL-Heart-Model not downloaded. See data/README.md for instructions."
            )

        out_dir = _REPO_ROOT / "tutorials" / "output" / "tutorial_03"
        results = _run_tutorial_script("tutorial_03_create_statistical_model.py")
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
class TestTutorial04FitStatisticalModelToPatient:
    """End-to-end test for tutorial_04_fit_statistical_model_to_patient.py."""

    _class_name = "tutorial_04_fit_statistical_model_to_patient"

    def test_run(self, test_directories: dict[str, Path]) -> None:
        kcl_dir = test_directories["data"] / "KCL-Heart-Model"
        if not (kcl_dir / "pca_mean.vtu").exists():
            pytest.skip(
                "KCL-Heart-Model not downloaded. See data/README.md for instructions."
            )

        pca_json = (
            _REPO_ROOT / "tutorials" / "output" / "tutorial_03" / "pca_model.json"
        )
        if not pca_json.exists():
            _run_tutorial_script("tutorial_03_create_statistical_model.py")
            assert pca_json.exists(), (
                "Tutorial 3 bootstrap did not create the expected PCA model file: "
                f"{pca_json}"
            )

        out_dir = _REPO_ROOT / "tutorials" / "output" / "tutorial_04"
        results = _run_tutorial_script(
            "tutorial_04_fit_statistical_model_to_patient.py"
        )
        assert results["registered_file"].exists(), "Registered VTP should exist"

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
class TestTutorial05VTKToUSD:
    """End-to-end test for tutorial_05_vtk_to_usd.py."""

    _class_name = "tutorial_05_vtk_to_usd"

    def test_run(self, test_directories: dict[str, Path]) -> None:
        # Prefer Tutorial 2 output; fall back to any .vtp in data
        tutorial2_vtp = (
            _REPO_ROOT / "tutorials" / "output" / "tutorial_02" / "patient_surfaces.vtp"
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

        out_dir = _REPO_ROOT / "tutorials" / "output" / "tutorial_05"
        results = _run_tutorial_script("tutorial_05_vtk_to_usd.py")
        assert results["usd_file"], "USD file path should not be empty"
        assert Path(results["usd_file"]).exists(), "USD file should exist"

        tt = TestTools(
            class_name=self._class_name,
            results_dir=out_dir,
            baselines_dir=test_directories["baselines"] / self._class_name,
        )
        _compare_screenshots(results["screenshots"], tt)


# -----------------------------------------------------------------------------
# Tutorials 8-10 - Cardiac mesh stage-prediction pipeline (bring-your-own-data)
#
# These tutorials use a local ``D:/PhysioTwin4D/`` cardiac dataset and (for
# Tutorials 9 and 10) the optional PhysicsNeMo dependency, so they are skipped
# automatically unless that data / those checkpoints are present. They produce
# no screenshots; the tests assert the tutorial ran and populated
# ``tutorial_results``.
# -----------------------------------------------------------------------------

_CARDIAC_DATA_ROOT = Path("D:/PhysioTwin4D")
_CARDIAC_FITTED_MESHES_DIR = _CARDIAC_DATA_ROOT / "duke_data" / "fitted_kcl_meshes"
_TUTORIALS_DIR = _REPO_ROOT / "tutorials"


def _physicsnemo_available() -> bool:
    """True if the optional PhysicsNeMo dependency is importable."""
    return importlib.util.find_spec("physicsnemo") is not None


def _torch_geometric_available() -> bool:
    """True if the optional PyTorch Geometric dependency is importable."""
    return importlib.util.find_spec("torch_geometric") is not None


def _run_eval_tutorial(script_name: str) -> dict[str, Any]:
    """Run a Tutorial 10 eval script through its no-argument ``run_tutorial`` path.

    ``runpy`` does not reset ``sys.argv``, so without this the eval scripts would
    see pytest's arguments and try to parse them as CLI options. Force a single
    argv entry so the ``len(sys.argv) > 1`` dispatch selects ``run_tutorial``.
    """
    saved_argv = sys.argv
    sys.argv = [script_name]
    try:
        return _run_tutorial_script(script_name)
    finally:
        sys.argv = saved_argv


@pytest.mark.tutorial
@pytest.mark.slow
class TestTutorial08CardiacFitModel:
    """End-to-end test for tutorial_08_cardiac_fit_model.py (bring-your-own-data)."""

    def test_run(self) -> None:
        if not (_CARDIAC_DATA_ROOT / "duke_data" / "gated_nii").exists():
            pytest.skip(
                "Cardiac dataset not present at D:/PhysioTwin4D/. Tutorial 8 is "
                "bring-your-own-data; see tutorials/README.md."
            )
        results = _run_tutorial_script("tutorial_08_cardiac_fit_model.py")
        assert "patients" in results, "Tutorial 8 should report processed patients"


@pytest.mark.tutorial
@pytest.mark.slow
@pytest.mark.requires_gpu
class TestTutorial09aCardiacTrainMGN:
    """End-to-end test for tutorial_09a_cardiac_train_physicsnemo_mgn.py."""

    def test_run(self) -> None:
        if not _physicsnemo_available():
            pytest.skip("PhysicsNeMo not installed (optional [physicsnemo] extra).")
        if not _torch_geometric_available():
            pytest.skip("torch-geometric not installed (required for MeshGraphNet).")
        if not _CARDIAC_FITTED_MESHES_DIR.exists():
            pytest.skip("Tutorial 8 cardiac output not present; run Tutorial 8 first.")
        results = _run_tutorial_script("tutorial_09a_cardiac_train_physicsnemo_mgn.py")
        assert isinstance(results, dict)


@pytest.mark.tutorial
@pytest.mark.slow
@pytest.mark.requires_gpu
class TestTutorial09bCardiacTrainMLP:
    """End-to-end test for tutorial_09b_cardiac_train_physicsnemo_mlp.py."""

    def test_run(self) -> None:
        if not _physicsnemo_available():
            pytest.skip("PhysicsNeMo not installed (optional [physicsnemo] extra).")
        if not _CARDIAC_FITTED_MESHES_DIR.exists():
            pytest.skip("Tutorial 8 cardiac output not present; run Tutorial 8 first.")
        results = _run_tutorial_script("tutorial_09b_cardiac_train_physicsnemo_mlp.py")
        assert isinstance(results, dict)


@pytest.mark.tutorial
@pytest.mark.slow
class TestTutorial10aCardiacEvalMGN:
    """End-to-end test for tutorial_10a_cardiac_eval_physicsnemo_mgn.py."""

    def test_run(self) -> None:
        if not _physicsnemo_available():
            pytest.skip("PhysicsNeMo not installed (optional [physicsnemo] extra).")
        if not _torch_geometric_available():
            pytest.skip("torch-geometric not installed (required for MeshGraphNet).")
        checkpoint = _TUTORIALS_DIR / "output_mgn" / "mgn_stage_model.pt"
        if not checkpoint.exists() or not _CARDIAC_FITTED_MESHES_DIR.exists():
            pytest.skip(
                "Tutorial 9a checkpoint or cardiac data not present; "
                "run Tutorials 8 and 9a first."
            )
        results = _run_eval_tutorial("tutorial_10a_cardiac_eval_physicsnemo_mgn.py")
        assert "predicted_files" in results


@pytest.mark.tutorial
@pytest.mark.slow
class TestTutorial10bCardiacEvalMLP:
    """End-to-end test for tutorial_10b_cardiac_eval_physicsnemo_mlp.py."""

    def test_run(self) -> None:
        if not _physicsnemo_available():
            pytest.skip("PhysicsNeMo not installed (optional [physicsnemo] extra).")
        checkpoint = _TUTORIALS_DIR / "output" / "physicsnemo_stage_model.pt"
        if not checkpoint.exists() or not _CARDIAC_FITTED_MESHES_DIR.exists():
            pytest.skip(
                "Tutorial 9b checkpoint or cardiac data not present; "
                "run Tutorials 8 and 9b first."
            )
        results = _run_eval_tutorial("tutorial_10b_cardiac_eval_physicsnemo_mlp.py")
        assert "predicted_files" in results


# -----------------------------------------------------------------------------
# Tutorial 6 - Reconstruct High-Resolution 4D CT
# -----------------------------------------------------------------------------


@pytest.mark.tutorial
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

        out_dir = _REPO_ROOT / "tutorials" / "output" / "tutorial_06"
        results = _run_tutorial_script("tutorial_06_reconstruct_highres_4d_ct.py")
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
