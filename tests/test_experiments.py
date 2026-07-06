#!/usr/bin/env python
"""
Test suite for running experiment scripts.

These tests execute Python scripts in the experiments/ directory. Each subdirectory
in experiments/ gets its own test that runs all scripts in that subdirectory in
alphanumeric order.

Scripts are Jupytext percent-format files (# %% cell separators), converted from the
original Jupyter notebooks while preserving git history via git mv.

WARNING: These are EXTREMELY long-running tests that may take hours to complete.
They are NOT part of CI/CD and should only be run manually.

Usage:
    # Run all experiment tests
    pytest tests/test_experiments.py -v -m experiment

    # Run a specific experiment subdirectory
    pytest tests/test_experiments.py::test_experiment_colormap_vtk_to_usd -v

    # Run with detailed output
    pytest tests/test_experiments.py -v -s -m experiment

Note: These tests require all dependencies installed and GPU/CUDA support for
many of the experiments.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

# Base directories
REPO_ROOT = Path(__file__).parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"

# Experiment subdirectories to test (in order of complexity/dependencies)
EXPERIMENT_SUBDIRS = [
    "Colormap-VTK_To_USD",
    "Convert_VTK_To_USD",
    # 'DisplacementField_To_USD',  # Disabled - scripts not ready
    "Reconstruct4DCT",
    "Heart-VTKSeries_To_USD",
    "Heart-GatedCT_To_USD",
    "Heart-Create_Statistical_Model",
    "Heart-Statistical_Model_To_Patient",
    "Lung-GatedCT_To_USD",
    # 'Lung-VesselsAirways',  # Disabled - scripts not ready
]


def get_scripts_in_subdir(subdir_name: str) -> list[Path]:
    """
    Get all Python scripts in a subdirectory, sorted alphanumerically.

    Args:
        subdir_name: Name of the subdirectory in experiments/

    Returns:
        List of Path objects for .py script files, sorted alphanumerically
    """
    subdir = EXPERIMENTS_DIR / subdir_name
    if not subdir.exists():
        return []

    scripts = sorted(subdir.glob("*.py"))
    return scripts


def execute_script(script_path: Path, timeout: int = 3600) -> dict[str, Any]:
    """
    Execute a Python experiment script.

    Args:
        script_path: Path to the .py script file
        timeout: Maximum execution time in seconds (default: 1 hour)

    Returns:
        Dictionary with execution results:
            - success: bool
            - stdout: str
            - stderr: str
            - returncode: int

    Raises:
        subprocess.TimeoutExpired: If script execution exceeds timeout
    """
    print(f"\n{'=' * 80}")
    print(f"Executing script: {script_path.name}")
    print(f"Path: {script_path}")
    print(f"Timeout: {timeout} seconds ({timeout // 60} minutes)")
    print(f"{'=' * 80}\n")

    cmd = [sys.executable, str(script_path)]

    # So scripts can use reduced parameters when run as tests
    env = os.environ.copy()
    env["PHYSIOMOTION_RUNNING_AS_TEST"] = "1"

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout * 1.5,  # Give extra time for startup overhead
            cwd=script_path.parent,  # Run in script's directory
            env=env,
            check=False,
        )

        success = result.returncode == 0

        if success:
            print(f"OK: {script_path.name}")
        else:
            print(f"FAILED: {script_path.name}")
            print(f"Return code: {result.returncode}")
            if result.stderr:
                print(f"Error output:\n{result.stderr}")

        return {
            "success": success,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }

    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {script_path.name}")
        print(f"Exceeded: {timeout} seconds")
        raise


def _heart_statistical_model_pca_prerequisites_met() -> tuple[bool, str]:
    """
    Check whether PCA model outputs from Heart-Create_Statistical_Model exist.

    Heart-Statistical_Model_To_Patient scripts expect these artifacts from
    the Heart-Create_Statistical_Model experiment (notably 5-compute_pca_model.py).

    Returns:
        (True, "") if all required paths exist, else (False, reason_message).
    """
    pca_output_dir = (
        EXPERIMENTS_DIR / "Heart-Create_Statistical_Model" / "kcl-heart-model"
    )
    pca_json = pca_output_dir / "pca_model.json"
    pca_mean_vtp = pca_output_dir / "pca_mean.vtp"

    if not pca_output_dir.is_dir():
        return (
            False,
            f"PCA model output directory not found: {pca_output_dir}. "
            "Run the Heart-Create_Statistical_Model experiment first "
            "(e.g. pytest tests/test_experiments.py::test_experiment_create_statistical_model -v -s --run-experiments).",
        )
    if not pca_json.is_file():
        return (
            False,
            f"PCA model JSON not found: {pca_json}. "
            "Complete Heart-Create_Statistical_Model (including 5-compute_pca_model.py) first.",
        )
    if not pca_mean_vtp.is_file():
        return (
            False,
            f"PCA mean surface not found: {pca_mean_vtp}. "
            "Complete Heart-Create_Statistical_Model (including 5-compute_pca_model.py) first.",
        )
    return (True, "")


def run_experiment_scripts(subdir_name: str, timeout_per_script: int = 3600) -> None:
    """
    Run all Python scripts in an experiment subdirectory in alphanumeric order.

    IMPORTANT: Scripts are executed SEQUENTIALLY in alphanumeric order within
    this function. This ensures proper dependency handling even when running
    tests with multiple pytest workers (e.g., pytest -n 2).

    The sequential execution is enforced by:
    1. Using a standard Python for loop (not parallelized)
    2. Each script must complete before the next begins
    3. Failures in earlier scripts prevent later ones from running

    Args:
        subdir_name: Name of the subdirectory in experiments/
        timeout_per_script: Timeout in seconds for each script (default: 1 hour)

    Raises:
        AssertionError: If any script fails to execute successfully
    """
    scripts = get_scripts_in_subdir(subdir_name)

    if not scripts:
        pytest.skip(f"No scripts found in experiments/{subdir_name}")

    print(f"\n{'#' * 80}")
    print(f"# Experiment: {subdir_name}")
    print(f"# Found {len(scripts)} script(s)")
    print("# Sequential execution enforced (scripts run in order)")
    print(f"{'#' * 80}\n")

    failed_scripts: list[dict[str, Any]] = []
    successful_scripts: list[str] = []

    for i, script in enumerate(scripts, 1):
        print(f"\n--- Script {i}/{len(scripts)} ---")
        print(f"Sequential execution: script {i} must complete before {i + 1} starts")

        try:
            result = execute_script(script, timeout=timeout_per_script)

            if result["success"]:
                successful_scripts.append(script.name)
            else:
                failed_scripts.append(
                    {
                        "name": script.name,
                        "returncode": result["returncode"],
                        "stderr": result["stderr"],
                    }
                )
                # Stop execution on first failure to maintain dependencies
                print(f"\nStopping execution: {script.name} failed")
                print("Remaining scripts in this experiment will not run.")
                break

        except subprocess.TimeoutExpired:
            failed_scripts.append(
                {
                    "name": script.name,
                    "returncode": -1,
                    "stderr": f"Timeout after {timeout_per_script} seconds",
                }
            )
            # Stop execution on timeout
            print(f"\nStopping execution: {script.name} timed out")
            print("Remaining scripts in this experiment will not run.")
            break

        except Exception as e:
            failed_scripts.append(
                {"name": script.name, "returncode": -2, "stderr": str(e)}
            )
            # Stop execution on exception
            print(f"\nStopping execution: {script.name} raised exception")
            print("Remaining scripts in this experiment will not run.")
            break

    # Print summary
    print(f"\n{'=' * 80}")
    print(f"Experiment Summary: {subdir_name}")
    print(f"{'=' * 80}")
    print(f"Total scripts: {len(scripts)}")
    print(f"Successful: {len(successful_scripts)}")
    print(f"Failed: {len(failed_scripts)}")

    if successful_scripts:
        print("\nSuccessful scripts:")
        for name in successful_scripts:
            print(f"  - {name}")

    if failed_scripts:
        print("\nFailed scripts:")
        for failure in failed_scripts:
            print(f"  - {failure['name']}")
            print(f"    Return code: {failure['returncode']}")
            if failure["stderr"]:
                # Print first few lines of error
                error_lines = failure["stderr"].split("\n")[:10]
                for line in error_lines:
                    print(f"    {line}")

    print(f"{'=' * 80}\n")

    # Assert all scripts succeeded
    assert not failed_scripts, (
        f"{len(failed_scripts)} script(s) failed in {subdir_name}: "
        f"{[f['name'] for f in failed_scripts]}"
    )


# ============================================================================
# Test Functions - One per Experiment Subdirectory
# ============================================================================


@pytest.mark.experiment
@pytest.mark.slow
@pytest.mark.xdist_group(
    name="experiment_colormap"
)  # Prevent parallel execution within group
def test_experiment_colormap_vtk_to_usd() -> None:
    """
    Test Colormap-VTK_To_USD experiment scripts.

    This experiment demonstrates converting VTK files with colormaps to USD format.

    EXECUTION MODEL:
    - Scripts run SEQUENTIALLY in alphanumeric order within this test
    - This test function is atomic - pytest-xdist treats it as a single unit
    - Multiple experiment tests CAN run in parallel (different subdirectories)
    - Scripts within THIS experiment CANNOT run in parallel or out of order
    """
    run_experiment_scripts("Colormap-VTK_To_USD", timeout_per_script=3600)


# DISABLED - Scripts not ready
# @pytest.mark.experiment
# @pytest.mark.slow
# def test_experiment_displacement_field_to_usd():
#     """
#     Test DisplacementField_To_USD experiment scripts.
#
#     This experiment demonstrates converting registration displacement fields to USD
#     format for visualization in PhysicsNeMo and Omniverse.
#     """
#     run_experiment_scripts('DisplacementField_To_USD', timeout_per_script=3600)


@pytest.mark.experiment
@pytest.mark.slow
@pytest.mark.requires_gpu
@pytest.mark.xdist_group(name="experiment_reconstruct4dct")
def test_experiment_reconstruct_4dct() -> None:
    """
    Test Reconstruct4DCT experiment scripts.

    This experiment demonstrates 4D CT reconstruction techniques.

    EXECUTION MODEL:
    - Scripts run SEQUENTIALLY in alphanumeric order within this test
    - Each script must complete before the next begins
    - Failure in one script stops execution of remaining scripts
    """
    run_experiment_scripts("Reconstruct4DCT", timeout_per_script=1200)


@pytest.mark.experiment
@pytest.mark.slow
@pytest.mark.xdist_group(name="experiment_heart_vtk")
def test_experiment_heart_vtk_series_to_usd() -> None:
    """
    Test Heart-VTKSeries_To_USD experiment scripts.

    This experiment converts heart VTK time series data to USD format.

    EXECUTION ORDER (ENFORCED):
    1. 0-download_and_convert_4d_to_3d.py (downloads data)
    2. 1-heart_vtkseries_to_usd.py (uses downloaded data)

    Each script must complete successfully before the next begins.
    """
    run_experiment_scripts("Heart-VTKSeries_To_USD", timeout_per_script=5400)


@pytest.mark.experiment
@pytest.mark.slow
@pytest.mark.requires_gpu
@pytest.mark.xdist_group(name="experiment_heart_gated_ct")
def test_experiment_heart_gated_ct_to_usd() -> None:
    """
    Test Heart-GatedCT_To_USD experiment scripts.

    This is the main cardiac imaging pipeline experiment with strict dependencies.

    EXECUTION ORDER (STRICTLY ENFORCED):
    1. 0-download_and_convert_4d_to_3d.py (downloads and converts data)
    2. 1-register_images.py (registers converted images)
    3. 2-generate_segmentation.py (segments registered images)
    4. 3-transform_dynamic_and_static_contours.py (transforms segmentations)
    5. 4-merge_dynamic_and_static_usd.py (merges into final USD)

    Each script depends on outputs from previous scripts.
    Execution stops on first failure to prevent cascading errors.
    """
    run_experiment_scripts("Heart-GatedCT_To_USD", timeout_per_script=5400)


@pytest.mark.experiment
@pytest.mark.slow
@pytest.mark.xdist_group(name="experiment_convert_vtk_to_usd")
def test_experiment_convert_vtk_to_usd() -> None:
    """
    Test Convert_VTK_To_USD experiment scripts.

    This experiment demonstrates VTK to USD conversion using the library classes.

    EXECUTION ORDER (ENFORCED):
    1. convert_chop_valve_to_usd.py (converts CHOP valve data)
    2. convert_vtk_to_usd_using_class.py (demonstrates library usage)

    Sequential execution ensures examples build on each other.
    """
    run_experiment_scripts("Convert_VTK_To_USD", timeout_per_script=3600)


@pytest.mark.experiment
@pytest.mark.slow
@pytest.mark.xdist_group(name="experiment_create_statistical_model")
def test_experiment_create_statistical_model() -> None:
    """
    Test Heart-Create_Statistical_Model experiment scripts.

    This experiment demonstrates creating a PCA statistical shape model from the
    KCL Heart Model dataset.

    EXECUTION ORDER (ENFORCED):
    1. 1-input_meshes_to_input_surfaces.py (convert meshes to surfaces)
    2. 2-input_surfaces_to_surfaces_aligned.py (align surfaces)
    3. 3-registration_based_correspondence.py (establish point correspondence)
    4. 4-surfaces_aligned_correspond_to_pca_inputs.py (prepare PCA inputs)
    5. 5-compute_pca_model.py (compute PCA model using sklearn)

    Sequential execution ensures data dependencies are met.
    """
    run_experiment_scripts("Heart-Create_Statistical_Model", timeout_per_script=5400)


@pytest.mark.experiment
@pytest.mark.slow
@pytest.mark.requires_gpu
@pytest.mark.xdist_group(name="experiment_heart_statistical_model")
def test_experiment_heart_statistical_model_to_patient() -> None:
    """
    Test Heart-Statistical_Model_To_Patient experiment scripts.

    This experiment demonstrates heart model to patient registration using
    statistical shape models (PCA).

    PREREQUISITE: Complete Heart-Create_Statistical_Model experiment first to generate
    the PCA model data required for this experiment.

    If PCA outputs (kcl-heart-model/pca_model.json, pca_mean.vtp) are missing, this test
    is skipped with a clear message so it can be run in isolation after generating them.

    EXECUTION ORDER (ENFORCED):
    1. heart_model_to_model_icp_itk.py (ICP registration)
    2. heart_model_to_model_registration_pca.py (PCA-based registration)
    3. heart_model_to_patient.py (applies registration to patient)

    Sequential execution ensures registration results are available for subsequent steps.
    """
    prereq_met, skip_reason = _heart_statistical_model_pca_prerequisites_met()
    if not prereq_met:
        pytest.skip(skip_reason)

    run_experiment_scripts(
        "Heart-Statistical_Model_To_Patient", timeout_per_script=7200
    )


@pytest.mark.experiment
@pytest.mark.slow
@pytest.mark.requires_gpu
@pytest.mark.xdist_group(name="experiment_lung_gated_ct")
def test_experiment_lung_gated_ct_to_usd() -> None:
    """
    Test Lung-GatedCT_To_USD experiment scripts.

    This is the lung imaging pipeline experiment using DirLab 4DCT data.

    EXECUTION ORDER (STRICTLY ENFORCED):
    1. 0-register_dirlab_4dct.py (registers lung 4DCT data)
    2. 1-make_dirlab_models.py (creates 3D models from registered data)
    3. 2-paint_dirlab_models.py (applies textures/materials to models)
    4. Experiment_ArrangeOnStage.py (arranges models in USD scene)
    5. Experiment_CombineModels.py (combines models into single USD)
    6. Experiment_SegReg.py (segmentation and registration experiments)
    7. Experiment_SubSurfaceScatter.py (applies advanced materials)

    Each script depends on outputs from previous scripts.
    Execution is sequential and stops on first failure.
    """
    run_experiment_scripts("Lung-GatedCT_To_USD", timeout_per_script=5400)


# DISABLED - Scripts not ready
# @pytest.mark.experiment
# @pytest.mark.slow
# @pytest.mark.requires_gpu
# @pytest.mark.requires_data
# def test_experiment_lung_vessels_airways():
#     """
#     Test Lung-VesselsAirways experiment scripts.
#
#     This experiment demonstrates specialized vessel and airway segmentation
#     using deep learning models.
#     Expected scripts (in order):
#     - 0-GenData.py
#     """
#     run_experiment_scripts('Lung-VesselsAirways', timeout_per_script=3600)


# ============================================================================
# Discovery Test - Validate Experiment Structure
# ============================================================================


@pytest.mark.experiment
def test_experiment_structure() -> None:
    """
    Validate the structure of the experiments directory.

    This test checks that:
    1. The experiments directory exists
    2. Each expected subdirectory exists
    3. Each subdirectory contains at least one .py script
    """
    assert EXPERIMENTS_DIR.exists(), (
        f"Experiments directory not found: {EXPERIMENTS_DIR}"
    )

    missing_subdirs = []
    empty_subdirs = []

    for subdir_name in EXPERIMENT_SUBDIRS:
        subdir = EXPERIMENTS_DIR / subdir_name

        if not subdir.exists():
            missing_subdirs.append(subdir_name)
            continue

        scripts = list(subdir.glob("*.py"))
        if not scripts:
            empty_subdirs.append(subdir_name)

    # Report findings
    if missing_subdirs:
        print(f"\nWARNING: Missing subdirectories: {missing_subdirs}")

    if empty_subdirs:
        print(f"\nWARNING: Empty subdirectories (no scripts): {empty_subdirs}")

    # Print discovered scripts
    print("\nDiscovered Scripts:")
    for subdir_name in EXPERIMENT_SUBDIRS:
        scripts = get_scripts_in_subdir(subdir_name)
        if scripts:
            print(f"\n{subdir_name}/ ({len(scripts)} script(s)):")
            for s in scripts:
                print(f"  - {s.name}")

    assert not missing_subdirs, f"Missing subdirectories: {missing_subdirs}"
    assert not empty_subdirs, f"Empty subdirectories: {empty_subdirs}"


# ============================================================================
# Helper Test - Script Discovery
# ============================================================================


@pytest.mark.experiment
@pytest.mark.parametrize("subdir_name", EXPERIMENT_SUBDIRS)
def test_list_scripts_in_subdir(subdir_name: str) -> None:
    """
    List all scripts in each experiment subdirectory.

    This helper test can be used to preview what scripts will be run
    without actually executing them.

    Usage:
        pytest tests/test_experiments.py::test_list_scripts_in_subdir -v -s
    """
    scripts = get_scripts_in_subdir(subdir_name)

    print(f"\n{subdir_name}/ - {len(scripts)} script(s):")
    for i, s in enumerate(scripts, 1):
        print(f"  {i}. {s.name}")

    assert scripts, f"No scripts found in {subdir_name}"
