# Using the --run-experiments Flag

This document explains how the `--run-experiments` flag works and why it exists.

## Purpose

The `--run-experiments` flag provides protection against accidentally running extremely long-running experiment notebook tests. These tests can take 20+ hours to complete and are resource-intensive.

## How It Works

### Automatic Protection

Experiment tests are **automatically skipped** in all these scenarios:

```bash
# All of these will SKIP experiment tests:
pytest tests/
pytest tests/ -v
pytest tests/test_experiments.py
pytest tests/test_experiments.py -v
pytest tests/ -m experiment  # Even with the marker!
pytest -v tests/
```

### Explicit Opt-In Required

To run experiment tests, you MUST use the `--run-experiments` flag:

```bash
# This is the ONLY way experiment tests will run:
pytest tests/test_experiments.py -v --run-experiments

# Or with specific tests:
pytest tests/test_experiments.py::test_experiment_colormap_vtk_to_usd -v --run-experiments

# Or with marker filters:
pytest tests/test_experiments.py -m "experiment and not requires_gpu" -v --run-experiments
```

## Implementation Details

The protection is implemented in `tests/conftest.py` using pytest hooks:

1. **`pytest_addoption`**: Registers the `--run-experiments` command-line option
2. **`pytest_configure`**: Registers the `experiment` marker
3. **`pytest_collection_modifyitems`**: Automatically skips tests marked with `@pytest.mark.experiment` unless `--run-experiments` is provided

### Code Snippet

```python
def pytest_collection_modifyitems(config, items):
    """Automatically skip experiment tests unless --run-experiments is passed."""
    if config.getoption('--run-experiments'):
        return  # User explicitly requested experiment tests
    
    # Skip all tests marked with @pytest.mark.experiment
    skip_experiments = pytest.mark.skip(
        reason='Experiment tests require --run-experiments flag to run'
    )
    for item in items:
        if 'experiment' in item.keywords:
            item.add_marker(skip_experiments)
```

## Benefits

### 1. Prevents Accidental Execution

Developers and CI/CD systems won't accidentally run 20+ hour test suites:

```bash
# Safe - won't run experiment tests
pytest tests/ -v
```

### 2. Clear Intent Required

When someone uses `--run-experiments`, it's clear they understand what they're doing:

```bash
# Intentional - user knows this will take hours
pytest tests/test_experiments.py -v --run-experiments
```

### 3. Self-Documenting

The flag name makes the purpose obvious:
- `--run-experiments` clearly indicates these are experiment-related tests
- Error message tells users what flag to use if they need the tests

### 4. CI/CD Protection

CI/CD workflows can never accidentally trigger these tests because they'll never include the `--run-experiments` flag.

## Viewing Skipped Tests

To see that experiment tests are being skipped:

```bash
# Show skipped tests
pytest tests/test_experiments.py -v

# Sample output:
# tests/test_experiments.py::test_experiment_colormap_vtk_to_usd SKIPPED
# tests/test_experiments.py::test_experiment_reconstruct_4dct SKIPPED
# ...
```

To see why they're skipped:

```bash
# Show skip reasons
pytest tests/test_experiments.py -v -rs

# Sample output:
# SKIPPED [6] conftest.py:XX: Experiment tests require --run-experiments flag to run
```

## Adding More Protected Tests

To protect additional long-running tests, simply:

1. Mark them with `@pytest.mark.experiment`
2. They'll automatically be protected by the same mechanism

```python
@pytest.mark.experiment
@pytest.mark.slow
def test_my_long_running_experiment():
    """This test requires --run-experiments flag."""
    # ... test code ...
```

## Removing Protection (Not Recommended)

If you ever need to disable this protection (not recommended):

1. Edit `tests/conftest.py`
2. Remove or comment out the `pytest_collection_modifyitems` function
3. Tests will run with standard pytest commands

**Warning:** Removing this protection means developers might accidentally run 20+ hour test suites.

## Test-Mode Flag: PHYSIOMOTION_RUNNING_AS_TEST

When experiment tests run (with `--run-experiments`), the test runner also sets **`PHYSIOMOTION_RUNNING_AS_TEST=1`** so that scripts can use reduced parameters (e.g. fewer iterations, fewer files) and finish faster. Scripts should read this variable and choose quick vs full parameters accordingly. See [EXPERIMENT_TESTS_GUIDE.md](EXPERIMENT_TESTS_GUIDE.md#running-as-test-physiomotion_running_as_test) for the recommended check and the `physiomotion4d.test_tools.TestTools.running_as_test()` helper.

## Related Documentation

- **[EXPERIMENT_TESTS_GUIDE.md](EXPERIMENT_TESTS_GUIDE.md)** - Complete guide to experiment tests (including PHYSIOMOTION_RUNNING_AS_TEST)
- **[README.md](README.md)** - Main testing documentation
- **[conftest.py](conftest.py)** - Pytest configuration implementation

## Summary

- ✅ Experiment tests are **opt-in only**
- ✅ Requires explicit `--run-experiments` flag
- ✅ Automatically skipped without the flag
- ✅ Protected from accidental CI/CD execution
- ✅ Clear, self-documenting behavior

**Remember:** If you see experiment tests being skipped and want to run them, add `--run-experiments` to your pytest command!
