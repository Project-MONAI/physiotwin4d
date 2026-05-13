# Experiment Tests Guide

This guide explains how to use the automated test suite for PhysioMotion4D experiment scripts.

## Overview

The `test_experiments.py` module provides automated testing for all percent-cell
Python scripts in the `experiments/` directory. Each subdirectory gets its own
test that executes every `*.py` script in alphanumeric order.

**WARNING:** These tests are **extremely long-running** and may take multiple
hours to complete.

**PROTECTION:** Experiment tests are **opt-in only**. They will NOT run with
standard pytest commands like `pytest tests/` or `pytest -v tests/`. You must
explicitly pass the `--run-experiments` flag to run them.

## Quick Start

### List Available Experiments

```bash
# See all scripts that would be run (without executing)
pytest tests/test_experiments.py::test_list_scripts_in_subdir -v -s --run-experiments

# Validate experiment directory structure
pytest tests/test_experiments.py::test_experiment_structure -v --run-experiments
```

### Run All Experiment Tests

```bash
# Run ALL experiments (may take many hours!)
pytest tests/test_experiments.py -v --run-experiments

# Or run with marker filter (also works)
pytest tests/test_experiments.py -v -m experiment --run-experiments
```

**IMPORTANT:** Experiment tests require the `--run-experiments` flag to run.
Without this flag, they are automatically skipped.

### Run Individual Experiments

```bash
# Colormap VTK to USD (~2 hours)
pytest tests/test_experiments.py::test_experiment_colormap_vtk_to_usd -v -s --run-experiments

# Reconstruct 4D CT (~4 hours, requires GPU)
pytest tests/test_experiments.py::test_experiment_reconstruct_4dct -v -s --run-experiments

# Heart VTK Series to USD (~3 hours, requires data)
pytest tests/test_experiments.py::test_experiment_heart_vtk_series_to_usd -v -s --run-experiments

# Heart Gated CT to USD (~6 hours, requires GPU & data)
pytest tests/test_experiments.py::test_experiment_heart_gated_ct_to_usd -v -s --run-experiments

# Convert VTK to USD (~2 hours, requires data)
pytest tests/test_experiments.py::test_experiment_convert_vtk_to_usd -v -s --run-experiments

# Create Statistical Model (~3 hours, requires data, includes manual steps)
pytest tests/test_experiments.py::test_experiment_create_statistical_model -v -s --run-experiments

# Heart Statistical Model to Patient (~4 hours, requires GPU & data)
pytest tests/test_experiments.py::test_experiment_heart_statistical_model_to_patient -v -s --run-experiments

# Lung Gated CT to USD (~6 hours, requires GPU & data)
pytest tests/test_experiments.py::test_experiment_lung_gated_ct_to_usd -v -s --run-experiments

# DISABLED (scripts not ready):
# - test_experiment_displacement_field_to_usd
# - test_experiment_lung_vessels_airways
```

**NOTE:** All experiment tests require the `--run-experiments` flag. Without
it, they are automatically skipped.

## Test Details

### Available Experiment Tests

| Test Name                                            | Subdirectory                          | Expected Duration | Requirements   | Status     |
| ---------------------------------------------------- | ------------------------------------- | ----------------- | -------------- | ---------- |
| `test_experiment_colormap_vtk_to_usd`                | `Colormap-VTK_To_USD/`                | ~2 hours          | Basic          | Active     |
| `test_experiment_convert_vtk_to_usd`                 | `Convert_VTK_To_USD/`                 | ~2 hours          | Data           | Active     |
| ~~`test_experiment_displacement_field_to_usd`~~      | ~~`DisplacementField_To_USD/`~~       | ~~~2 hours~~      | ~~Basic~~      | Disabled   |
| `test_experiment_reconstruct_4dct`                   | `Reconstruct4DCT/`                    | ~4 hours          | GPU            | Active     |
| `test_experiment_heart_vtk_series_to_usd`            | `Heart-VTKSeries_To_USD/`             | ~3 hours          | Data           | Active     |
| `test_experiment_heart_gated_ct_to_usd`              | `Heart-GatedCT_To_USD/`               | ~6 hours          | GPU + Data     | Active     |
| `test_experiment_create_statistical_model`           | `Heart-Create_Statistical_Model/`     | ~3 hours          | Data           | Active     |
| `test_experiment_heart_statistical_model_to_patient` | `Heart-Statistical_Model_To_Patient/` | ~4 hours          | GPU + Data     | Active     |
| `test_experiment_lung_gated_ct_to_usd`               | `Lung-GatedCT_To_USD/`                | ~6 hours          | GPU + Data     | Active     |
| ~~`test_experiment_lung_vessels_airways`~~           | ~~`Lung-VesselsAirways/`~~            | ~~~2 hours~~      | ~~GPU + Data~~ | Disabled   |

**Note:** Disabled tests are commented out in the code and will not run. They
can be re-enabled when the scripts are ready.

### Execution Order

Within each subdirectory, scripts are executed in **alphanumeric order**
(the runner uses `sorted(subdir.glob("*.py"))`):

- `0-download_and_convert_4d_to_3d.py` (runs first)
- `1-register_images.py` (runs second)
- `2-generate_segmentation.py` (runs third)
- etc.

This ensures that scripts with dependencies run in the correct sequence.

## Requirements

### System Requirements

- **CPU:** Multi-core processor (8+ cores recommended)
- **RAM:** 32GB minimum (64GB recommended for large experiments)
- **GPU:** NVIDIA GPU with CUDA support (RTX 3090 or better recommended)
- **Disk:** 100GB+ free space for data and outputs
- **OS:** Linux (Ubuntu 20.04+) or Windows 10/11

### Software Requirements

```bash
# Install all dependencies
pip install -e ".[test]"

# Or with uv (recommended)
uv pip install -e ".[test]"
```

### Data Requirements

Some experiments require external data downloads:
- Heart experiments: Slicer-Heart-CT dataset (~1.2GB)
- Lung experiments: DirLab 4DCT dataset (varies by case)

Data is typically downloaded automatically by the first script in each sequence.

## Test Markers

All experiment tests are marked with:

- `@pytest.mark.experiment` - Identifies as experiment test (excludes from CI/CD)
- `@pytest.mark.slow` - Indicates long-running test
- `@pytest.mark.requires_gpu` - Requires CUDA-capable GPU (when applicable)
- `@pytest.mark.requires_data` - Requires external data download (when applicable)
- `@pytest.mark.timeout(seconds)` - Sets maximum execution time

## Running as Test: PHYSIOMOTION_RUNNING_AS_TEST

When you run experiment tests with `pytest ... --run-experiments`, the test
runner sets the environment variable **`PHYSIOMOTION_RUNNING_AS_TEST=1`** before
executing each script. Scripts can read this to use **reduced parameters**
(fewer iterations, fewer files, smaller resolution) so test runs complete in
reasonable time.

### How it works

- **Test runner** ([test_experiments.py](test_experiments.py)): `execute_script()`
  passes `env` with `PHYSIOMOTION_RUNNING_AS_TEST=1` to the subprocess that runs
  the script as a standalone Python file, so the script process sees the variable.
- **Scripts**: Early in the script (after imports or with other config),
  compute a boolean and use it to choose quick vs full parameters.

### Recommended check in scripts

Use either of these:

**Option 1 - inline (stdlib only):**

```python
import os

running_as_test = os.environ.get("PHYSIOMOTION_RUNNING_AS_TEST", "").lower() in ("1", "true", "yes")
```

**Option 2 - shared helper (recommended):**

```python
from physiomotion4d.test_tools import TestTools

# Then use TestTools.running_as_test() where you need it, e.g.:
quick_run = TestTools.running_as_test()
max_iterations = 100 if TestTools.running_as_test() else 2000
```

### Semantics

- **Truthy values** (case-insensitive): `1`, `true`, `yes` -> script should use
  fast/small parameters.
- **Unset or falsy**: use full parameters (normal interactive or production run).

Scripts that support this will run quickly when executed as tests and at full
fidelity when run manually.

## Usage Tips

### Run with Detailed Output

```bash
# Show all output from scripts (recommended)
pytest tests/test_experiments.py::test_experiment_heart_gated_ct_to_usd -v -s
```

The `-s` flag shows all stdout/stderr, including:
- Script execution progress
- Cell outputs (`# %%` percent-cell scripts are run as ordinary Python)
- Error messages
- Execution summaries

### Monitor Progress

Each script execution prints:
```
================================================================================
Executing script: 1-register_images.py
Path: experiments/Heart-GatedCT_To_USD/1-register_images.py
Timeout: 5400 seconds (90 minutes)
================================================================================

... (script output) ...

Successfully executed: 1-register_images.py
```

### Handle Failures

If a script fails:
1. Check the error output in the pytest summary
2. Open the script to inspect the failing cell
3. Fix the issue (code, data, environment)
4. Re-run the specific test

The scripts are executed as ordinary Python (`python <script>.py`); cell
boundaries from `# %%` markers are only meaningful inside cell-aware editors
and do not affect test execution.

### Skip Tests by Marker

```bash
# Skip GPU-dependent experiments
pytest tests/test_experiments.py -m "experiment and not requires_gpu" -v --run-experiments

# Skip data-dependent experiments
pytest tests/test_experiments.py -m "experiment and not requires_data" -v --run-experiments

# Run only fast experiments (relatively speaking!)
pytest tests/test_experiments.py -m "experiment and not requires_gpu and not requires_data" -v --run-experiments
```

**Note:** The `--run-experiments` flag is always required, regardless of marker
filters.

## CI/CD Exclusion

**These tests are NEVER run in CI/CD workflows.**

Experiment tests are protected by requiring the `--run-experiments` flag, which
is never used in CI/CD. Even if someone accidentally runs:

```bash
# These commands will NOT run experiment tests (they'll be skipped)
pytest tests/ -v
pytest tests/ -v -m experiment  # Still skipped without --run-experiments!
pytest tests/test_experiments.py -v  # Still skipped!
```

The ONLY way to run experiment tests is:

```bash
# This is the ONLY way experiment tests will run
pytest tests/test_experiments.py -v --run-experiments
```

This protection ensures experiment tests are:
- Never accidentally triggered in CI/CD
- Never run by developers doing normal testing
- Explicitly opt-in only for validation purposes

## Troubleshooting

### Test Timeout

If a test times out:
1. Check the timeout setting in the test function
2. Consider increasing it if needed:
   ```python
   @pytest.mark.timeout(14400)  # 4 hours
   ```
3. Or override via command line:
   ```bash
   pytest tests/test_experiments.py::test_experiment_... -v --timeout=14400
   ```

### Out of Memory

If you get OOM errors:
1. Close other applications
2. Reduce batch sizes in scripts (if applicable)
3. Use smaller test datasets
4. Consider running on a machine with more RAM

### GPU Not Available

If GPU tests fail due to CUDA issues:
1. Verify CUDA installation: `nvidia-smi`
2. Check PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
3. Reinstall PyTorch with CUDA support
4. Skip GPU tests: `-m "experiment and not requires_gpu"`

### Data Download Failures

If data downloads fail:
1. Check internet connection
2. Verify URLs in scripts are still valid
3. Manually download data and place in expected location
4. Check disk space

### Script Execution Errors

If a script fails:
1. Open the script in your editor (VS Code / Cursor with the Python extension)
2. Run the cells in order to identify the issue
3. Check for:
   - Missing dependencies
   - Incorrect paths
   - Outdated APIs
   - Hardware limitations

## Advanced Usage

### Customizing Timeouts

Edit `test_experiments.py` to adjust per-script or per-test timeouts:

```python
# Per-script timeout (default: 3600 seconds = 1 hour)
run_experiment_scripts('Heart-GatedCT_To_USD', timeout_per_script=7200)

# Per-test timeout decorator
@pytest.mark.timeout(21600)  # 6 hours
def test_experiment_...():
    ...
```

### Adding New Experiments

To add a new experiment directory:

1. Create the subdirectory in `experiments/`
2. Add `# %%` percent-cell Python scripts (use numeric prefixes for ordering)
3. Add to `EXPERIMENT_SUBDIRS` list in `test_experiments.py`:
   ```python
   EXPERIMENT_SUBDIRS = [
       'Colormap-VTK_To_USD',
       'YourNewExperiment',  # Add here
       ...
   ]
   ```
4. Create a test function:
   ```python
   @pytest.mark.experiment
   @pytest.mark.slow
   @pytest.mark.timeout(7200)
   def test_experiment_your_new_experiment():
       """Test YourNewExperiment scripts."""
       run_experiment_scripts('YourNewExperiment', timeout_per_script=3600)
   ```

### Parallel Execution

**Use with caution** - Experiments use significant GPU/CPU resources.

You can run multiple experiment subdirectories in parallel using pytest-xdist:

```bash
# Run with multiple workers (different subdirectories run in parallel)
pytest tests/test_experiments.py -v -n 2 --run-experiments

# Run with auto-detection of CPU count
pytest tests/test_experiments.py -v -n auto --run-experiments
```

**IMPORTANT GUARANTEES:**
- Scripts **within** each subdirectory run **SEQUENTIALLY** in alphanumeric order
- Different subdirectories **CAN** run in parallel (if resources allow)
- Each test function is atomic - pytest-xdist won't split script execution
- `@pytest.mark.xdist_group` markers prevent conflicts if needed
- Execution stops on first failure within each subdirectory

**Example:**
```bash
# With -n 2, these could run simultaneously:
# Worker 1: test_experiment_colormap_vtk_to_usd
# Worker 2: test_experiment_heart_vtk_series_to_usd

# But within Worker 1's test, scripts run in strict order:
# 1. colormap_vtk_to_usd.py (must complete first)
# ... (any subsequent scripts in that directory)
```

## FAQ

**Q: How long will all experiments take?**
A: Approximately 32 hours total on a high-end workstation (GPU, 64GB RAM). Note:
DisplacementField_To_USD and Lung-VesselsAirways are currently disabled.

**Q: Can I run these on CPU only?**
A: Some experiments will work, but most require GPU and will be extremely slow
or fail on CPU.

**Q: Will these tests modify the scripts?**
A: No, scripts are executed as ordinary Python processes. The scripts may
write outputs to disk (per their own logic) but the script files themselves are
not modified.

**Q: What if I only want to test specific scripts?**
A: The test suite runs all scripts in a subdirectory. To test individual
scripts, run them manually with `python path/to/script.py` or use VS
Code/Cursor's "Run Cell" support for percent-cell scripts.

**Q: Can I use these tests for regression testing?**
A: Yes, but you'll need to add output validation. Currently, tests only check
that scripts execute without errors.

**Q: Are these tests deterministic?**
A: Not always - some experiments use random seeds, AI models, or external data
that may change.

## Related Documentation

- **[tests/README.md](README.md)** - Main testing documentation
- **[tests/TESTING_GUIDE.md](TESTING_GUIDE.md)** - Comprehensive testing guide
- **[tests/PARALLEL_EXECUTION_GUIDE.md](PARALLEL_EXECUTION_GUIDE.md)** - Parallel execution details
- **[tests/EXPERIMENT_FLAG_USAGE.md](EXPERIMENT_FLAG_USAGE.md)** - --run-experiments flag explanation
- **[experiments/README.md](../experiments/README.md)** - Experiment documentation
- **[.github/workflows/ci.yml](../.github/workflows/ci.yml)** - CI/CD configuration

## Contributing

If you add new experiments or improve the test suite, please:

1. Update `EXPERIMENT_SUBDIRS` list
2. Create appropriate test functions
3. Document requirements and expected duration
4. Update this guide
5. Test locally before submitting PR

## Support

If you encounter issues:

1. Check this guide and related documentation
2. Review the test output carefully
3. Try running the script manually with `python path/to/script.py`
4. Check GitHub issues for similar problems
5. Open a new issue with:
   - Test command used
   - Complete error output
   - System information (OS, Python version, GPU)
   - Environment details (conda/venv, package versions)

---

**Remember:** These tests are for validation and regression testing. They are
NOT part of CI/CD and must be run manually.
