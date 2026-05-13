# Parallel Execution Guide for Experiment Tests

This guide explains how parallel execution works with experiment tests and how
script dependencies are enforced.

## Overview

Experiment tests support parallel execution at the **subdirectory level** while
maintaining **strict sequential execution** of scripts within each
subdirectory.

## Execution Model

### Sequential Within Subdirectories (Enforced)

Within each experiment subdirectory, scripts run in **strict alphanumeric order**:

```python
# Example: Heart-GatedCT_To_USD scripts
1. 0-download_and_convert_4d_to_3d.py            # Must complete first
2. 1-register_images.py                          # Uses data from step 1
3. 2-generate_segmentation.py                    # Uses data from step 2
4. 3-transform_dynamic_and_static_contours.py    # Uses data from step 3
5. 4-merge_dynamic_and_static_usd.py             # Uses data from step 4
# ... and so on
```

**Key guarantees:**
- Scripts execute in a Python `for` loop (inherently sequential)
- Each script must complete before the next begins
- **Execution stops on first failure** to prevent cascading errors
- This behavior is enforced even when using `pytest -n` (multiple workers)

### Parallel Across Subdirectories (Allowed)

Different experiment subdirectories CAN run in parallel:

```bash
# With pytest -n 2, this could happen:
Worker 1: test_experiment_colormap_vtk_to_usd
Worker 2: test_experiment_heart_vtk_series_to_usd

# Both run simultaneously because they're independent experiments
```

## Implementation Details

### 1. Test Function Atomicity

Each test function is treated as an atomic unit by pytest-xdist:

```python
@pytest.mark.xdist_group(name='experiment_heart_gated_ct')
def test_experiment_heart_gated_ct_to_usd():
    """Test function is atomic - xdist won't split it"""
    run_experiment_scripts('Heart-GatedCT_To_USD', ...)
```

pytest-xdist **cannot** and **will not** parallelize the code inside a test
function. The entire function runs on a single worker.

### 2. Sequential Loop Enforcement

Inside `run_experiment_scripts()`, scripts run in a standard Python loop:

```python
for i, script in enumerate(scripts, 1):
    # Execute script i
    result = execute_script(script, timeout=timeout_per_script)

    if not result['success']:
        # STOP on failure - don't execute remaining scripts
        break
```

This loop is **inherently sequential** and cannot be parallelized.

### 3. Xdist Group Markers

Each test has a unique `@pytest.mark.xdist_group` marker:

```python
@pytest.mark.xdist_group(name='experiment_colormap')
def test_experiment_colormap_vtk_to_usd():
    ...

@pytest.mark.xdist_group(name='experiment_heart_gated_ct')
def test_experiment_heart_gated_ct_to_usd():
    ...
```

This ensures tests in the same group don't run in parallel (though currently
each test has a unique group, allowing all to run in parallel).

### 4. Fail-Fast Behavior

When a script fails, execution stops immediately:

```python
if not result['success']:
    print('Stopping execution: script failed')
    print('Remaining scripts in this experiment will not run.')
    break  # Stop the loop - don't run remaining scripts
```

This prevents:
- Wasting time on dependent scripts that will fail
- Cascading errors from missing dependencies
- Confusing error messages from downstream failures

## Usage Examples

### Serial Execution (One Worker)

```bash
# Run all experiments sequentially (one at a time)
pytest tests/test_experiments.py -v --run-experiments

# Behavior:
# - Colormap experiment runs (all scripts in order)
# - Then Reconstruct4DCT runs (all scripts in order)
# - Then Heart-VTKSeries runs (all scripts in order)
# - ... and so on
```

### Parallel Execution (Multiple Workers)

```bash
# Run with 2 workers
pytest tests/test_experiments.py -v -n 2 --run-experiments

# Possible behavior:
# Worker 1: Colormap experiment (scripts sequential within)
# Worker 2: Heart-VTKSeries experiment (scripts sequential within)
# When Worker 1 finishes, it picks up Reconstruct4DCT
# When Worker 2 finishes, it picks up Heart-Statistical_Model_To_Patient
# ... and so on
```

### Auto Worker Detection

```bash
# Let pytest-xdist determine optimal worker count
pytest tests/test_experiments.py -v -n auto --run-experiments

# Behavior:
# - Spawns one worker per CPU core
# - Distributes test functions (subdirectories) across workers
# - Within each worker, scripts run sequentially
```

## Dependency Management

### Within-Subdirectory Dependencies

Scripts in the same subdirectory often have dependencies:

```text
0-download.py -> 1-process.py -> 2-analyze.py -> 3-visualize.py
```

**Protected by:**
- Sequential `for` loop execution
- Alphanumeric ordering
- Fail-fast on error

### Cross-Subdirectory Independence

Different subdirectories should be independent:

```
Colormap-VTK_To_USD/    # Independent
Heart-GatedCT_To_USD/   # Independent
Lung-GatedCT_To_USD/    # Independent
```

**Best practices:**
- Each subdirectory has its own data/outputs
- No shared state between subdirectories
- Tests can run in any order
- Tests can run in parallel

## Resource Considerations

### GPU Constraints

If multiple experiments use GPU:

```bash
# Limit workers to avoid GPU contention
pytest tests/test_experiments.py -v -n 1 --run-experiments  # Serial only

# Or set CUDA_VISIBLE_DEVICES per worker (advanced)
```

### Memory Constraints

Experiments can be memory-intensive:

```bash
# Run with fewer workers if memory is limited
pytest tests/test_experiments.py -v -n 2 --run-experiments  # Only 2 at once
```

### Disk I/O

Experiments generate large files:

```bash
# Ensure experiments write to separate directories
# (Already handled by experiment structure)
```

## Troubleshooting

### Scripts Running Out of Order

**Symptom:** Later scripts fail due to missing dependencies

**Diagnosis:**
1. Check that scripts have numeric prefixes (0-, 1-, 2-)
2. Verify `sorted()` in `get_scripts_in_subdir()` is working
3. Check for parallel execution attempts (shouldn't happen)

**Fix:** Scripts are sorted alphanumerically. Use proper prefixes.

### Parallel Execution Causing Errors

**Symptom:** Tests interfere with each other

**Diagnosis:**
1. Check if subdirectories share data/output directories
2. Verify each test uses unique resources
3. Check for global state or shared files

**Fix:**
- Ensure each subdirectory is independent
- Use unique output paths per subdirectory
- Add xdist_group markers to prevent specific tests from running in parallel

### Worker Failures

**Symptom:** pytest-xdist workers crash or hang

**Diagnosis:**
1. Check resource limits (memory, GPU)
2. Look for deadlocks or race conditions
3. Verify scripts don't have infinite loops

**Fix:**
- Reduce worker count (`-n 1` for serial)
- Increase timeouts if needed
- Fix problematic scripts

## Verification

### Test Sequential Execution

```bash
# Add debug output to verify ordering
pytest tests/test_experiments.py::test_experiment_heart_gated_ct_to_usd -v -s --run-experiments

# Look for output like:
# --- Script 1/7 ---
# Sequential execution: script 1 must complete before 2 starts
# ... (script 1 output) ...
# Successfully executed: 0-download_and_convert_4d_to_3d.py
# --- Script 2/7 ---
# Sequential execution: script 2 must complete before 3 starts
# ... (script 2 output) ...
```

### Test Parallel Execution

```bash
# Run with verbose output
pytest tests/test_experiments.py -v -n 2 --run-experiments

# Look for output indicating different workers:
# [gw0] PASSED tests/test_experiments.py::test_experiment_colormap_vtk_to_usd
# [gw1] PASSED tests/test_experiments.py::test_experiment_heart_vtk_series_to_usd
```

### Test Fail-Fast Behavior

```bash
# Temporarily break a middle script
# Run the test and verify remaining scripts don't execute
pytest tests/test_experiments.py::test_experiment_heart_gated_ct_to_usd -v -s --run-experiments

# Should see:
# Failed to execute: 2-generate_segmentation.py
# Stopping execution: 2-generate_segmentation.py failed
# Remaining scripts in this experiment will not run.
```

## Summary

| Aspect | Behavior | Enforcement |
|--------|----------|-------------|
| Scripts within subdirectory | **Sequential, strict order** | Python `for` loop, fail-fast |
| Subdirectories (test functions) | **Can run in parallel** | pytest-xdist distribution |
| Script execution | **One at a time per test** | Standard function execution |
| Failure handling | **Stop on first error** | `break` statement in loop |
| Worker assignment | **One test per worker** | pytest-xdist scheduling |
| Order enforcement | **Alphanumeric sorting** | `sorted(glob('*.py'))` |

**Bottom line:** Your scripts within each subdirectory will ALWAYS run in
order, even with `pytest -n`. The parallelism only occurs at the
subdirectory/test level, not within a test.

## Related Documentation

- **[EXPERIMENT_TESTS_GUIDE.md](EXPERIMENT_TESTS_GUIDE.md)** - Main experiment tests guide
- **[test_experiments.py](test_experiments.py)** - Test implementation
- **[conftest.py](conftest.py)** - Pytest configuration

---

**Key Takeaway:** Parallel execution is safe and won't break dependencies.
Scripts in each subdirectory run sequentially no matter what.
