#!/usr/bin/env python
"""
Shared pytest fixtures for PhysioMotion4D tests.

This file defines fixtures that are available to all test modules
in the tests directory via pytest's automatic fixture discovery.
"""

import os
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import itk
import pytest

from physiomotion4d.contour_tools import ContourTools
from physiomotion4d.convert_nrrd_4d_to_3d import ConvertNRRD4DTo3D
from physiomotion4d.register_images_ants import RegisterImagesANTs
from physiomotion4d.register_images_greedy import RegisterImagesGreedy
from physiomotion4d.register_images_icon import RegisterImagesICON
from physiomotion4d.segment_chest_total_segmentator import SegmentChestTotalSegmentator
from physiomotion4d.segment_heart_simpleware import SegmentHeartSimpleware
from physiomotion4d.transform_tools import TransformTools

# ============================================================================
# Pytest Configuration - Command Line Options
# ============================================================================

# Module-level variable to store config for access in hooks
_pytest_config: Optional[pytest.Config] = None


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add custom command-line options for pytest."""
    parser.addoption(
        "--run-experiments",
        action="store_true",
        default=False,
        help="Run experiment tests (extremely long-running notebook tests)",
    )
    parser.addoption(
        "--run-tutorials",
        action="store_true",
        default=False,
        help="Run tutorial tests (data/GPU gated tutorial scripts)",
    )
    parser.addoption(
        "--create-baselines",
        action="store_true",
        default=False,
        help="Create baseline files from current test outputs when missing (otherwise missing baseline fails)",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with custom markers and settings."""
    global _pytest_config
    _pytest_config = config

    from physiomotion4d import test_tools as _test_tools

    _test_tools.set_create_baseline_if_missing(
        config.getoption("--create-baselines", default=False)
    )

    config.addinivalue_line(
        "markers",
        "experiment: marks tests that run experiment notebooks (extremely slow, manual only)",
    )
    config.addinivalue_line(
        "markers",
        "tutorial: marks tests that run tutorial scripts (data/GPU gated, manual only)",
    )
    # Initialize test timing storage
    config._test_timings = {  # type: ignore[attr-defined]
        "tests": [],
        "total_time": 0.0,
        "start_time": datetime.now(),
    }


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """
    Automatically skip experiment and tutorial tests unless their opt-in flags
    are passed.

    This ensures that experiment tests are opt-in only and won't run
    accidentally when running the normal test suite.
    """
    for item in items:
        if "experiment" in item.keywords and not config.getoption("--run-experiments"):
            item.add_marker(
                pytest.mark.skip(
                    reason="Experiment tests require --run-experiments flag to run"
                )
            )
        if "tutorial" in item.keywords and not config.getoption("--run-tutorials"):
            item.add_marker(
                pytest.mark.skip(
                    reason="Tutorial tests require --run-tutorials flag to run"
                )
            )


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """
    Collect test timing information after each test completes.

    This hook is called for each phase of test execution (setup, call, teardown).
    We only collect timing from the 'call' phase which is the actual test execution.
    """
    if report.when == "call":
        # Use the module-level config reference
        if _pytest_config is None:
            return

        # Store test timing information
        test_info = {
            "nodeid": report.nodeid,
            "duration": report.duration,
            "outcome": report.outcome,
            "is_experiment": "experiment" in report.keywords,
            "is_tutorial": "tutorial" in report.keywords,
        }

        _pytest_config._test_timings["tests"].append(test_info)  # type: ignore[attr-defined]
        _pytest_config._test_timings["total_time"] += report.duration  # type: ignore[attr-defined]


def pytest_terminal_summary(
    terminalreporter: Any,
    exitstatus: int,
    config: pytest.Config,
) -> None:
    """
    Print comprehensive test timing report after all tests complete.

    This hook is called at the end of the test session to display
    timing statistics for all tests, including experiment tests.
    """
    timings = config._test_timings  # type: ignore[attr-defined]
    tests = timings["tests"]

    if not tests:
        return

    # Calculate session duration
    session_duration = datetime.now() - timings["start_time"]

    # Separate regular, tutorial, and experiment tests
    regular_tests = [
        t for t in tests if not t["is_experiment"] and not t["is_tutorial"]
    ]
    tutorial_tests = [t for t in tests if t["is_tutorial"]]
    experiment_tests = [t for t in tests if t["is_experiment"]]

    # Write the timing report
    terminalreporter.write_sep("=", "TEST TIMING REPORT", bold=True)
    terminalreporter.write_line("")

    # Session summary
    terminalreporter.write_line(f"Session Duration: {session_duration}")
    terminalreporter.write_line(
        f"Total Test Time: {timedelta(seconds=int(timings['total_time']))}"
    )
    terminalreporter.write_line(f"Total Tests: {len(tests)}")
    terminalreporter.write_line("")

    sorted_regular = sorted(regular_tests, key=lambda x: x["duration"], reverse=True)

    # Regular tests section
    if regular_tests:
        terminalreporter.write_sep("-", "Regular Tests", bold=True)
        terminalreporter.write_line(f"Count: {len(regular_tests)}")

        # Calculate total time
        regular_total = sum(t["duration"] for t in regular_tests)
        terminalreporter.write_line(
            f"Total Time: {timedelta(seconds=int(regular_total))}"
        )
        terminalreporter.write_line("")
        terminalreporter.write_line("Individual Test Times:")
        for test in sorted_regular:
            outcome_symbol = "+" if test["outcome"] == "passed" else "x"
            duration_str = _format_duration(test["duration"])
            terminalreporter.write_line(
                f"  {outcome_symbol} {duration_str:>10s}  {test['nodeid']}"
            )
        terminalreporter.write_line("")

    # Tutorial tests section
    if tutorial_tests:
        terminalreporter.write_sep("-", "Tutorial Tests", bold=True)
        terminalreporter.write_line(f"Count: {len(tutorial_tests)}")
        sorted_tutorials = sorted(
            tutorial_tests, key=lambda x: x["duration"], reverse=True
        )
        tutorial_total = sum(t["duration"] for t in tutorial_tests)
        terminalreporter.write_line(
            f"Total Time: {timedelta(seconds=int(tutorial_total))}"
        )
        terminalreporter.write_line("")
        terminalreporter.write_line("Individual Test Times:")
        for test in sorted_tutorials:
            outcome_symbol = "+" if test["outcome"] == "passed" else "x"
            duration_str = _format_duration(test["duration"])
            terminalreporter.write_line(
                f"  {outcome_symbol} {duration_str:>10s}  {test['nodeid']}"
            )
        terminalreporter.write_line("")

    # Experiment tests section
    if experiment_tests:
        terminalreporter.write_sep("-", "Experiment Tests", bold=True)
        terminalreporter.write_line(f"Count: {len(experiment_tests)}")

        # Sort by duration (longest first)
        sorted_experiments = sorted(
            experiment_tests, key=lambda x: x["duration"], reverse=True
        )

        # Calculate total time
        experiment_total = sum(t["duration"] for t in experiment_tests)
        terminalreporter.write_line(
            f"Total Time: {timedelta(seconds=int(experiment_total))}"
        )
        terminalreporter.write_line("")

        # Show all experiment tests with timing
        terminalreporter.write_line("Individual Test Times:")
        for test in sorted_experiments:
            outcome_symbol = "+" if test["outcome"] == "passed" else "x"
            duration_str = _format_duration(test["duration"])
            terminalreporter.write_line(
                f"  {outcome_symbol} {duration_str:>10s}  {test['nodeid']}"
            )
        terminalreporter.write_line("")

    # Top 10 slowest tests overall
    if len(tests) > 10:
        terminalreporter.write_sep("-", "Top 10 Slowest Tests", bold=True)
        sorted_all = sorted(tests, key=lambda x: x["duration"], reverse=True)[:10]

        for i, test in enumerate(sorted_all, 1):
            outcome_symbol = "+" if test["outcome"] == "passed" else "x"
            duration_str = _format_duration(test["duration"])
            if test["is_experiment"]:
                test_type = "[EXP]"
            elif test["is_tutorial"]:
                test_type = "[TUT]"
            else:
                test_type = "[REG]"
            terminalreporter.write_line(
                f"  {i:2d}. {outcome_symbol} {duration_str:>10s} {test_type} {test['nodeid']}"
            )
        terminalreporter.write_line("")

    # Statistics by outcome
    passed = sum(1 for t in tests if t["outcome"] == "passed")
    failed = sum(1 for t in tests if t["outcome"] == "failed")
    skipped = sum(1 for t in tests if t["outcome"] == "skipped")

    terminalreporter.write_sep("-", "Test Outcomes", bold=True)
    terminalreporter.write_line(f"Passed:  {passed}")
    terminalreporter.write_line(f"Failed:  {failed}")
    terminalreporter.write_line(f"Skipped: {skipped}")
    terminalreporter.write_line("")


def _format_duration(seconds: float) -> str:
    """Format duration in a human-readable way."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.0f}s"


# Directory and Data Download Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def test_directories() -> dict[str, Path]:
    """Set up test directories for data and results."""
    data_dir = Path(__file__).parent.parent / "data" / "test"
    output_dir = Path(__file__).parent / "results"
    baselines_dir = Path(__file__).parent / "baselines"

    # Create directories if they don't exist
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    baselines_dir.mkdir(parents=True, exist_ok=True)

    return {"data": data_dir, "output": output_dir, "baselines": baselines_dir}


@pytest.fixture(scope="session")
def download_test_data(test_directories: dict[str, Path]) -> Path:
    """Download TruncalValve 4D CT data."""
    data_dir = test_directories["data"]
    input_image_filename = data_dir / "TruncalValve_4DCT.seq.nrrd"

    # Check if file already exists
    if input_image_filename.exists():
        print(f"\nData file already exists: {input_image_filename}")
        return input_image_filename

    # Try to download if not found locally
    input_image_url = "https://github.com/SlicerHeart/SlicerHeart/releases/download/TestingData/TruncalValve_4DCT.seq.nrrd"
    print(f"\nDownloading TruncalValve 4D CT data from {input_image_url}...")

    try:
        urllib.request.urlretrieve(input_image_url, str(input_image_filename))
        print(f"Downloaded to {input_image_filename}")
    except urllib.error.URLError as e:
        msg = (
            f"Could not download test data: {e}. "
            f"Please manually place TruncalValve_4DCT.seq.nrrd in {data_dir}"
        )
        if os.environ.get("CI"):
            pytest.fail(msg)
        else:
            pytest.skip(msg)

    return input_image_filename


# ============================================================================
# Image Conversion Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def test_images(
    download_test_data: Path,
    test_directories: dict[str, Path],
) -> list[Any]:
    """Convert and resample 4D NRRD data; return pre-resampled time points."""
    data_dir = test_directories["data"]

    # Convert 4D NRRD to 3D time series if not already done
    slice_000 = data_dir / "slice_000.mha"
    slice_007 = data_dir / "slice_007.mha"
    if not slice_000.exists() or not slice_007.exists():
        print("\nConverting 4D NRRD to 3D time series...")
        conv = ConvertNRRD4DTo3D()
        conv.load_nrrd_4d(str(download_test_data))
        conv.save_3d_images(str(data_dir / "slice"))
    else:
        print("\n3D slice files already exist")

    # Resample each slice_???.mha to 1.5x1.5x1.5 mm and save as slice_???_sml.mha
    target_spacing = [1.5, 1.5, 1.5]
    for slice_file in sorted(data_dir.glob("slice_???.mha")):
        sml_file = slice_file.with_name(slice_file.stem + "_sml.mha")
        if not sml_file.exists():
            print(f"\nResampling {slice_file.name} -> {sml_file.name} ...")
            img = itk.imread(str(slice_file))
            input_spacing = list(img.GetSpacing())
            input_size = list(itk.size(img))
            output_size = [
                int(round(input_size[i] * input_spacing[i] / target_spacing[i]))
                for i in range(3)
            ]
            interpolator = itk.LinearInterpolateImageFunction.New(img)
            resampler = itk.ResampleImageFilter.New(Input=img)
            resampler.SetInterpolator(interpolator)
            resampler.SetOutputSpacing(target_spacing)
            resampler.SetSize(output_size)
            resampler.SetOutputOrigin(img.GetOrigin())
            resampler.SetOutputDirection(img.GetDirection())
            resampler.Update()
            itk.imwrite(resampler.GetOutput(), str(sml_file), compression=True)
    print("\nResampled slice files up to date")

    slice_files = sorted(data_dir.glob("slice_???_sml.mha"))
    if len(slice_files) < 3:
        pytest.skip("Resampled slice files not found.")

    images = [itk.imread(str(f)) for f in slice_files]
    print(f"\nLoaded {len(images)} time points for testing")
    return images


@pytest.fixture(scope="session")
def test_labelmaps(
    segmenter_total_segmentator: SegmentChestTotalSegmentator,
    test_images: list[Any],
    test_directories: dict[str, Path],
) -> list[dict[str, Any]]:
    """
    Segment each time point with TotalSegmentator and return result dicts.
    Labelmaps are cached at data_dir / slice_???_sml_labelmap.mha.
    """
    data_dir = test_directories["data"]
    slice_files = sorted(data_dir.glob("slice_???_sml.mha"))

    results: list[dict[str, Any]] = []
    for img, slice_file in zip(test_images, slice_files):
        labelmap_file = data_dir / f"{slice_file.stem}_labelmap.mha"
        if not labelmap_file.exists():
            print(f"\nSegmenting {slice_file.name} ...")
            result = segmenter_total_segmentator.segment(
                img, contrast_enhanced_study=False
            )
            itk.imwrite(result["labelmap"], str(labelmap_file), compression=True)

        labelmap = itk.imread(str(labelmap_file))
        masks = segmenter_total_segmentator.create_anatomy_group_masks(labelmap)
        results.append(
            {
                "labelmap": labelmap,
                "lung": masks["lung"],
                "heart": masks["heart"],
                "major_vessels": masks["major_vessels"],
                "bone": masks["bone"],
                "soft_tissue": masks["soft_tissue"],
                "other": masks["other"],
                "contrast": masks["contrast"],
            }
        )

    return results


@pytest.fixture(scope="session")
def test_transforms(
    registrar_ants: RegisterImagesANTs,
    test_images: list[Any],
    test_directories: dict[str, Path],
) -> dict[str, Any]:
    """
    Perform ANTs registration and return results.
    Generates them if not already present, otherwise loads from disk.
    Transforms are cached in data_dir alongside the slice files.
    """
    data_dir = test_directories["data"]

    frame_tag = "001_to_007"
    inverse_transform_path = data_dir / f"ants_inverse_transform_{frame_tag}.hdf"
    forward_transform_path = data_dir / f"ants_forward_transform_{frame_tag}.hdf"

    if inverse_transform_path.exists() and forward_transform_path.exists():
        print("\nLoading existing ANTs registration results...")
        try:
            inverse_transform = itk.transformread(str(inverse_transform_path))
            forward_transform = itk.transformread(str(forward_transform_path))
            return {
                "inverse_transform": inverse_transform,
                "forward_transform": forward_transform,
            }
        except (RuntimeError, Exception) as e:
            print(f"Error loading transforms: {e}")
            print("Regenerating registration results...")
            inverse_transform_path.unlink(missing_ok=True)
            forward_transform_path.unlink(missing_ok=True)

    # Perform registration if files don't exist or loading failed
    print("\nPerforming ANTs registration...")
    fixed_image = test_images[7]
    moving_image = test_images[1]

    registrar_ants.set_fixed_image(fixed_image)
    result = registrar_ants.register(moving_image=moving_image)

    inverse_transform = result["inverse_transform"]
    forward_transform = result["forward_transform"]

    itk.transformwrite(inverse_transform, str(inverse_transform_path), compression=True)
    itk.transformwrite(forward_transform, str(forward_transform_path), compression=True)
    return {
        "inverse_transform": inverse_transform,
        "forward_transform": forward_transform,
    }


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def segmenter_total_segmentator() -> SegmentChestTotalSegmentator:
    """Create a SegmentChestTotalSegmentator instance."""
    return SegmentChestTotalSegmentator()


@pytest.fixture(scope="session")
def segmenter_simpleware() -> SegmentHeartSimpleware:
    """Create a SegmentHeartSimpleware instance."""
    return SegmentHeartSimpleware()


@pytest.fixture(scope="session")
def contour_tools() -> ContourTools:
    """Create a ContourTools instance."""
    return ContourTools()


@pytest.fixture(scope="session")
def registrar_ants() -> RegisterImagesANTs:
    """Create a RegisterImagesANTs instance."""
    return RegisterImagesANTs()


@pytest.fixture(scope="session")
def registrar_greedy() -> RegisterImagesGreedy:
    """Create a RegisterImagesGreedy instance."""
    return RegisterImagesGreedy()


@pytest.fixture(scope="session")
def registrar_icon() -> RegisterImagesICON:
    """Create a RegisterImagesICON instance."""
    return RegisterImagesICON()


@pytest.fixture(scope="session")
def transform_tools() -> TransformTools:
    """Create a TransformTools instance."""
    return TransformTools()
