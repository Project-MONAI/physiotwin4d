# GitHub Workflows for Tests

This document describes the GitHub Actions workflows configured for running tests on pull requests.

## Overview

The project has two main workflow files:

1. **`.github/workflows/ci.yml`** (Main CI Workflow)
   - Runs on every PR and push to main/master/develop
   - Unit tests (cross-platform: Ubuntu + Windows)
   - Integration tests with external data
   - GPU tests (self-hosted runners)
   - Code quality checks

2. **`.github/workflows/test-slow.yml`** (Scheduled Slow Tests)
   - Runs nightly at 2 AM UTC
   - Slow/GPU-intensive tests (registration, segmentation)
   - Self-hosted GPU runners only

3. **`.github/workflows/docs.yml`** (Documentation)
   - Builds Sphinx documentation
   - Deploys to GitHub Pages

**Git LFS**: Workflows that run tests (`ci.yml`, `test-slow.yml`) use `actions/checkout` with `lfs: true` so baseline files in `tests/baselines/` (`.hdf`, `.mha`) are fetched and tests can compare against them.

## Main CI Workflow

File: `.github/workflows/ci.yml`

This comprehensive CI workflow combines unit tests, integration tests, GPU tests, and code quality checks.

## Workflow Jobs

### 1. Unit Tests (`unit-tests` job)
**Trigger**: All pull requests and pushes to main, master, and develop branches
**Platforms**: Ubuntu, Windows
**Python versions**: 3.10, 3.11, 3.12

**What runs**:
- Tests that don't require external data
- Tests that aren't marked as slow
- Coverage reporting to Codecov
- Cross-platform validation

**Command**:
```bash
pytest tests/ -v -m "not slow and not requires_data" --cov=physiomotion4d --cov-report=xml
```

**Tests included**:
- Basic unit tests (e.g., ImageTools conversions)
- Fast integration tests with mocked data

**Tests excluded**:
- Data-dependent tests (marked with `@pytest.mark.requires_data`)
- Slow/GPU tests (marked with `@pytest.mark.slow`)

---

### 2. Integration Tests with Data (`integration-tests` job)
**Trigger**: Pull requests only
**Platform**: Ubuntu only
**Python version**: 3.10
**Dependencies**: Requires `unit-tests` to pass first

**What runs**:
Tests that require downloading external data, executed in sequence with caching.

#### Test Steps:

1. **Data Download Tests**
   ```bash
   pytest tests/test_download_heart_data.py -v --cov=src/physiomotion4d --cov-report=xml
   ```
   - Downloads TruncalValve 4D CT data
   - Sets up test directories
   - Cached for subsequent runs

2. **Data Conversion Tests**
   ```bash
   pytest tests/test_convert_nrrd_4d_to_3d.py -v --cov=src/physiomotion4d --cov-append --cov-report=xml
   ```
   - Converts 4D NRRD to 3D time series
   - Creates slice files
   - Depends on downloaded data

3. **Contour Tools Tests**
   ```bash
   pytest tests/test_contour_tools.py -v -m "not slow" --cov=src/physiomotion4d --cov-append --cov-report=xml
   ```
   - Tests contour extraction
   - Tests mesh manipulation
   - Uses segmentation results (creates if missing)
   - Excludes slow GPU-dependent segmentation

4. **USD Conversion Tests**
   ```bash
   pytest tests/test_convert_vtk_to_usd_polymesh.py -v -m "not slow" --cov=src/physiomotion4d --cov-append --cov-report=xml
   ```
   - Tests VTK to USD conversion
   - Uses contour data
   - Multiple conversion scenarios

5. **Existing USD Tests**
   ```bash
   pytest tests/test_usd_merge.py tests/test_usd_time_preservation.py -v --cov=src/physiomotion4d --cov-append --cov-report=xml
   ```
   - USD file merging
   - Time preservation validation

#### Caching Strategy:
- **Cached directories**: `tests/data/`, `tests/results/`
- **Cache key**: `test-data-${{ hashFiles('tests/test_*.py') }}-v2`
- **Restore keys**: `test-data-`
- **Cache invalidation**: Changes to test files or manual version bump

#### Error Handling:
All data-dependent test steps use `continue-on-error: true` to prevent CI failures due to:
- Network issues during data download
- Transient failures
- Missing dependencies

Coverage is still uploaded even if tests fail.

### 3. GPU Tests (`gpu-tests` job)
**Trigger**: All pull requests and pushes to main, master, and develop branches
**Platform**: Self-hosted Linux runners with GPU
**Python versions**: 3.10, 3.11
**Dependencies**: Requires `unit-tests` to pass first

**What runs**:
- All non-slow tests with GPU support
- PyTorch with CUDA 12.6
- GPU-accelerated deep learning tests
- Requires self-hosted runners with NVIDIA GPUs

**Command**:
```bash
pytest tests/ -v -m "not slow" --cov=physiomotion4d --cov-report=xml
```

**Environment**:
- `CUDA_VISIBLE_DEVICES: 0`
- Self-hosted runners with NVIDIA GPU
- PyTorch with CUDA support

**Tests included**:
- GPU-accelerated model inference (when available)
- Tests that benefit from GPU but don't require hours of compute
- Fast integration tests on GPU hardware

**Note**: This job continues even on error (`continue-on-error: true`) since self-hosted GPU runners may not always be available.

### 4. Code Quality Checks (`code-quality` job)
**Trigger**: All pull requests and pushes to main, master, and develop branches
**Platform**: Ubuntu only
**Python version**: 3.10

**What runs**:
- Code formatting checks (Black)
- Import sorting checks (isort)
- Linting (Ruff, Flake8)
- Style enforcement

**Tools**:
1. **Black**: Code formatting
   ```bash
   black --check src/ tests/
   ```

2. **isort**: Import sorting
   ```bash
   isort --check-only src/ tests/
   ```

3. **Ruff**: Fast Python linter
   ```bash
   ruff check src/ tests/
   ```

4. **Flake8**: Additional style checks
   ```bash
   flake8 src/ tests/
   ```

**Note**: All checks use `continue-on-error: true` to avoid blocking PRs on style issues while still providing feedback.

---

## Tests Excluded from CI

### Slow Tests (Marked with `@pytest.mark.slow`)

These tests are **NOT** run in CI (even on GPU runners) because they require extended compute time:

1. **ANTs Registration Tests** (`test_register_images_ants.py`)
   - Requires: ANTsPy library
   - Markers: `@pytest.mark.requires_data`, `@pytest.mark.slow`
   - Run locally: `pytest tests/test_register_images_ants.py -v -s`
   - Why excluded: Computationally intensive, slow execution time

2. **ICON Registration Tests** (`test_register_images_icon.py`)
   - Requires: CUDA GPU, ICON library
   - Markers: `@pytest.mark.requires_data`, `@pytest.mark.slow`
   - Run locally: `pytest tests/test_register_images_icon.py -v -s`
   - Why excluded: Requires GPU, deep learning inference

3. **Transform Tools Tests** (`test_transform_tools.py`)
   - Requires: Registration results from ANTs tests
   - Markers: `@pytest.mark.requires_data`, `@pytest.mark.slow`
   - Run locally: `pytest tests/test_transform_tools.py -v -s`
   - Why excluded: Depends on slow registration tests

4. **TotalSegmentator Tests** (`test_segment_chest_total_segmentator.py`)
   - Requires: CUDA GPU, TotalSegmentator library
   - Markers: `@pytest.mark.requires_data`, `@pytest.mark.slow`
   - Run locally: `pytest tests/test_segment_chest_total_segmentator.py -v -s`
   - Why excluded: Requires GPU, model inference

**Why excluded**:
- These tests take 5-15 minutes each, even with GPU acceleration
- Registration algorithms are computationally intensive
- Deep learning model inference requires significant GPU memory and time
- CI runtime would exceed reasonable limits (even on self-hosted GPU runners)
- Better suited for nightly/scheduled testing or local development

**Local testing**:
```bash
# Run all slow tests (registration + segmentation)
pytest tests/ -v -m "slow"

# Run only registration tests
pytest tests/test_register_images_ants.py tests/test_register_images_icon.py -v -s

# Run only segmentation tests
pytest tests/test_segment_chest_total_segmentator.py -v -s
```

**Scheduled slow tests**:
These tests run automatically on a schedule via `.github/workflows/test-slow.yml`:
- **Schedule**: Nightly at 2 AM UTC
- **Trigger**: Also available via manual workflow dispatch
- **Platform**: Self-hosted Linux GPU runners
- **Command**: `pytest tests/ -v -m "slow"`
- **Purpose**: Regular validation of computationally intensive tests without blocking PRs

---

## Coverage Reporting

### Unit Tests Coverage
- **Flag**: `unittests`
- **When**: Ubuntu + Python 3.11 only
- **Upload**: After unit tests complete

### Integration Tests Coverage
- **Flag**: `integration-tests`
- **When**: After all data-dependent tests complete
- **Upload**: Even if tests fail (`fail_ci_if_error: false`)

### GPU Tests Coverage
- **Flag**: `gpu-tests`
- **When**: Python 3.11 on self-hosted GPU runners
- **Upload**: After GPU tests complete

### Slow Tests Coverage
- **Flag**: `slow-tests-gpu`
- **When**: Nightly scheduled runs on self-hosted GPU runners
- **Upload**: After slow tests complete

### Viewing Coverage
Coverage reports are uploaded to Codecov and can be viewed at:
- Repository Codecov dashboard
- Pull request comments (automatic)
- Coverage badge in README

---

## Test Execution Flow

```
Pull Request Created
    ↓
┌───────────────────────────────────────────────┐
│ Unit Tests (test job)                         │
│ - Runs on Ubuntu, Windows, macOS              │
│ - Python 3.11, 3.12                          │
│ - Fast, no external data                      │
└───────────────────────────────────────────────┘
    ↓
┌───────────────────────────────────────────────┐
│ Integration Tests (test-with-data job)        │
│ - Runs on Ubuntu only                         │
│ - Python 3.11                                 │
│                                               │
│ 1. Cache Check (tests/data/, tests/results/) │
│    ↓                                          │
│ 2. Download Data (if cache miss)             │
│    ↓                                          │
│ 3. Convert 4D to 3D                          │
│    ↓                                          │
│ 4. Test Contour Tools (no GPU)              │
│    ↓                                          │
│ 5. Test USD Conversion                       │
│    ↓                                          │
│ 6. Test USD Utilities                        │
│    ↓                                          │
│ 7. Upload Coverage                           │
└───────────────────────────────────────────────┘
    ↓
Tests Complete (Pass/Fail reported on PR)
```

---

## Manual Workflow Dispatch

Currently not configured, but can be added for:
- Running GPU tests on self-hosted runners
- Running full test suite on-demand
- Testing with different configurations

---

## Monitoring and Debugging

### Check Workflow Status
1. Go to repository → Actions tab
2. Select workflow run
3. View job logs

### Common Issues

**Data Download Fails**:
- Check network connectivity
- Verify data URL is accessible
- Review cache restoration logs

**Cache Miss**:
- Normal on first run
- Check cache key matches pattern
- Verify test files haven't changed significantly

**Test Timeouts**:
- Increase timeout in workflow
- Optimize slow tests
- Consider parallel execution

### Workflow Files
- Main workflow: `.github/workflows/ci.yml`
- Test configuration: `tests/conftest.py`
- Test documentation: `tests/TEST_ORGANIZATION.md`

---

## Future Enhancements

Potential improvements for the workflow:

1. **Self-hosted GPU Runner**
   - Add GPU-enabled runner for segmentation tests
   - Run full test suite including GPU tests

2. **Parallel Test Execution**
   - Run independent tests in parallel
   - Reduce total CI time

3. **Matrix Testing for Data Tests**
   - Test with different data sources
   - Test with different model versions

4. **Scheduled Runs**
   - Daily/weekly full test runs
   - Regression testing with baseline data

5. **Manual Workflow Dispatch**
   - On-demand test execution
   - Custom parameter inputs

