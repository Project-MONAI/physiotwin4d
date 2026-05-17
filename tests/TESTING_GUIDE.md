# PhysioMotion4D Testing Guide

This guide explains how to set up and use the PhysioMotion4D test suite.

## Test Suite Overview

The test suite validates the complete PhysioMotion4D pipeline:
- **Data download and conversion** - 4D NRRD to 3D time series
- **Segmentation** - TotalSegmentator chest CT segmentation
- **Registration** - ANTs and ICON deformable registration
- **Contour extraction** - PyVista mesh generation from segmentation masks
- **USD conversion** - VTK to USD format for Omniverse
- **Transform tools** - ITK transform manipulation and visualization
- **USD utilities** - File merging and time-varying data preservation

## Quick Start

### 1. Install Git LFS (required)

Baseline files in `tests/baselines/` (`.hdf` and `.mha`) are tracked with **Git LFS**. Install Git LFS and pull LFS files so baseline comparison tests can run:

```bash
# Install: brew install git-lfs (macOS), sudo apt-get install git-lfs (Linux), or see https://git-lfs.github.com
git lfs install
git lfs pull   # if you already cloned
```

### 2. Install test dependencies

```bash
# Using uv (recommended)
uv pip install -e ".[test]"
```

### 3. Run all tests (excluding slow ones)

```bash
pytest tests/ -m "not slow" -v
```

### 4. Run specific test modules

```bash
# Data preparation tests (fast)
pytest tests/test_download_heart_data.py -v
pytest tests/test_convert_image_4d_to_3d.py -v

# Registration tests (slow, ~5-10 min each)
pytest tests/test_register_images_ants.py -v
pytest tests/test_register_images_icon.py -v

# USD tests (fast)
pytest tests/test_usd_merge.py -v
pytest tests/test_usd_time_preservation.py -v
```

### 5. Run with coverage

```bash
pytest tests/ --cov=src/physiomotion4d --cov-report=html
```

## Test Timeouts

All tests have a **global timeout of 900 seconds (15 minutes)** to prevent hanging. Individual tests may complete much faster. If a test times out, it will be automatically terminated.


## Test Organization

### Test Dependencies

Tests are organized hierarchically with fixtures providing reusable data:

```
test_download_heart_data
    ↓
test_convert_image_4d_to_3d
    ↓                    ↓
    ↓                    ├─→ test_register_images_ants ──→ test_transform_tools
    ↓                    ├─→ test_register_images_icon
    ↓                    ↓
test_segment_chest_total_segmentator ────→ test_contour_tools
                                                ↓
                                           test_convert_vtk_to_usd_polymesh
```

### Test Markers

- `@pytest.mark.slow` - Long-running tests (>30 seconds)
- `@pytest.mark.requires_data` - Tests requiring external data download
- `@pytest.mark.integration` - Integration tests vs unit tests
- `@pytest.mark.timeout(seconds)` - Per-test timeout override

### Skipped Tests

Some tests are skipped or marked slow due to:

**GPU-Dependent Tests** (skipped in CI):
- `test_segment_chest_total_segmentator.py` - Requires GPU for inference

**Computationally Intensive Tests**:
- Registration tests (ANTs, ICON) - Marked slow, run locally only
- Transform tools tests - Depend on registration results

## Test Data

### Automatic Data Management

Tests automatically manage data with intelligent fallback:

1. **Check test directory**: `tests/data/Slicer-Heart-CT/`
2. **Check main data directory**: `data/Slicer-Heart-CT/` (copies if found)
3. **Download if needed**: From Slicer-Heart-CT GitHub
4. **Skip test if unavailable**: Graceful failure with helpful message

This approach:
- Avoids re-downloading 1.2GB file multiple times
- Reuses existing project data
- Works offline if data already present
- Provides clear error messages

### Slicer-Heart-CT Dataset

- **Source**: https://github.com/Slicer-Heart-CT/Slicer-Heart-CT
- **File**: `TruncalValve_4DCT.seq.nrrd`
- **Size**: ~1.2 GB (4D NRRD)
- **Anatomy**: Pediatric cardiac CT with truncal valve
- **Phases**: 21 temporal phases (tests use first 2)
- **Location**: Downloaded to `tests/data/Slicer-Heart-CT/`

### Test Output

All test outputs are saved to organized directories:

```
tests/
├── data/                                # Downloaded/input data
│   └── Slicer-Heart-CT/
│       ├── TruncalValve_4DCT.seq.nrrd
│       ├── slice_000.mha
│       └── slice_001.mha
└── results/                             # Test outputs
    ├── segmentation_total_segmentator/
    ├── contour_tools/
    ├── usd_polymesh/
    ├── registration_ants/
    ├── registration_icon/
    └── transform_tools/
```

## Troubleshooting

### Data Download Issues

**Problem**: HTTP 404 error when downloading data

**Solution**: **Fixed!** Tests now automatically:
1. Check for existing data in `data/Slicer-Heart-CT/`
2. Copy from main data directory if available
3. Only attempt download if not found locally
4. Skip test with helpful message if all methods fail

**Manual fix**: Place `TruncalValve_4DCT.seq.nrrd` in `data/Slicer-Heart-CT/`

### Memory Errors (Segmentation Tests)

**Problem**: `RuntimeError: not enough memory`

**Root Cause**: Segmentation models require significant RAM and GPU memory

**Solutions**:
- Skip segmentation tests: `pytest tests/ --ignore=tests/test_segment_chest_total_segmentator.py`
- Run on system with sufficient RAM and GPU memory
- Tests are automatically skipped in CI

### Test Timeout

**Problem**: Test exceeds 15-minute timeout

**Current Settings**:
- Global timeout: 900 seconds (15 minutes)
- Registration tests: ~5-10 minutes each with GPU
- Segmentation tests: ~10-15 minutes with GPU

**Solutions**:
- Ensure GPU is available (much faster than CPU)
- Run slow tests individually: `pytest tests/test_register_images_ants.py -v`
- Override timeout: Use `@pytest.mark.timeout(1800)` decorator in test or `-o timeout=1800` on command line

### ITK Size Indexing Errors

**Problem**: `TypeError: in method 'itkSize3___getitem__', argument 2 of type 'unsigned long'`

**Solution**: **Fixed!** ITK Size objects now properly converted to tuples before numpy indexing:
```python
size_itk = itk.size(image)
size = (int(size_itk[0]), int(size_itk[1]), int(size_itk[2]))
```

### Fixture Name Errors

**Problem**: `NameError: name 'segmenter' is not defined`

**Solution**: **Fixed!** Tests now use correct fixture names:
- `segmenter_total_segmentator` for TotalSegmentator
- `registrar_ants` for ANTs
- `registrar_icon` for ICON

### Transform Type Mismatches

**Problem**: Tests expecting `DisplacementFieldTransform` but getting `CompositeTransform`

**Solution**: **Fixed!** Tests now accept both types since registration may return either depending on whether initial transforms are provided.

## CI/CD Integration

### GitHub Actions example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -e ".[dev]"

      - name: Run fast tests
        run: |
          pytest tests/ -m "not slow" -v

      - name: Run integration tests
        run: |
          pytest tests/ -v
        timeout-minutes: 15
```

### GitLab CI example

```yaml
test:
  stage: test
  script:
    - pip install -e ".[dev]"
    - pytest tests/ -v
  timeout: 15m
  artifacts:
    when: on_failure
    paths:
      - tests/baseline/
      - pytest-*.xml
```

## Best Practices

### Writing New Tests

1. **Use descriptive names**: `test_feature_does_something_specific`
2. **Add markers**: `@pytest.mark.slow`, `@pytest.mark.requires_data`
3. **Use shared fixtures**: Defined in `conftest.py`, avoid test file imports
4. **Document behavior**: Clear docstrings explaining validation
5. **Handle failures gracefully**: Informative error messages
6. **Respect test organization**: Follow dependency hierarchy
7. **Save outputs properly**: Use organized subdirectories in `tests/results/`

### Using Fixtures

**Good** - Use fixtures from conftest.py:
```python
def test_something(segmenter_total_segmentator, test_images):
    result = segmenter_total_segmentator.segment(test_images[0])
```

**Bad** - Don't import from other test files:
```python
from test_segment_chest import segmenter  # Will fail
```

### Memory-Intensive Tests

For tests requiring >16GB RAM or GPU:

```python
@pytest.mark.slow
@pytest.mark.requires_data
def test_large_model(test_images):
    """Test that requires significant resources."""
    # Test implementation
```

Document requirements in docstring and README.

### Test Data Management

1. **Reuse fixtures**: Don't re-generate data that fixtures provide
2. **Check for existing results**: Load from disk before regenerating
3. **Clean up appropriately**: Tests should clean up only their own temporary files
4. **Use appropriate directories**:
   - `tests/data/` for input data
   - `tests/results/` for outputs

## FAQ

**Q: How do I run specific tests?**
```bash
# Specific test file
pytest tests/test_usd_merge.py -v

# Specific test class
pytest tests/test_usd_merge.py::TestUSDMerge -v

# Specific test method
pytest tests/test_usd_merge.py::TestUSDMerge::test_specific_test -v
```

**Q: What if I don't have a GPU?**
- Most tests run on CPU (slower but functional)
- Segmentation tests (TotalSegmentator) require GPU
- Registration tests (ANTs, ICON) benefit from GPU but work on CPU
- Skip GPU tests: `pytest tests/ --ignore=tests/test_segment_chest_total_segmentator.py`

**Q: How do I run tests without downloading data?**
Place `TruncalValve_4DCT.seq.nrrd` in `data/Slicer-Heart-CT/` or `tests/data/Slicer-Heart-CT/` before running tests. The test will automatically detect and use it.

**Q: Can I use different test data?**
Yes! Modify the `download_test_data` fixture in `tests/conftest.py` to point to your data file. You may need to adjust expected results.

**Q: Why do some tests take so long?**
- Registration (ANTs/ICON): 5-10 minutes each (deformable registration is computationally intensive)
- Segmentation (TotalSegmentator): 10-15 minutes (deep learning inference on full CT volumes)
- Data download: First time only (~1.2GB file)
- Everything else: <1 minute

**Q: How do I clean up test outputs?**
```bash
# Remove all test outputs (keeps input data)
rm -rf tests/results/

# Remove downloaded data too
rm -rf tests/data/Slicer-Heart-CT/
```

**Q: What if a test fails?**
1. Check the error message - recent fixes address most common issues
2. Review `tests/TEST_FIXES_SUMMARY.md` for known issues
3. Ensure all dependencies are installed: `pip install -e ".[test]"`
4. Try running the specific failing test with `-v -s` for verbose output

## GitHub Actions Integration

Tests automatically run on pull requests via `.github/workflows/ci.yml`. The workflow:

- Runs fast tests (USD, conversion, basic validation)
- Skips slow tests (registration, segmentation)
- Caches test data to speed up subsequent runs
- Generates coverage reports

See `tests/GITHUB_WORKFLOWS.md` for detailed CI/CD documentation.

## Additional Resources

- **Test organization**: `tests/TEST_ORGANIZATION.md`
- **GitHub workflows**: `tests/GITHUB_WORKFLOWS.md`
- **Main README**: `../README.md`
- **Example workflows**: `../experiments/Heart-GatedCT_To_USD/`
- **Pytest docs**: https://docs.pytest.org/
- **ITK docs**: https://docs.itk.org/