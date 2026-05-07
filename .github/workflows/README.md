# GitHub Actions Workflows

This directory contains GitHub Actions workflows for automated testing and CI/CD.

## Workflows

### `ci.yml` - Main CI Pipeline

Runs on every push and pull request to main branches. Includes:

- **unit-tests**: Cross-platform unit tests
  - Runs on Ubuntu and Windows
  - Python 3.10, 3.11, and 3.12
  - Uses PyTorch CPU version to avoid GPU dependencies
  - Excludes slow tests and tests requiring external data
  - Generates coverage reports

- **integration-tests**: Integration tests with external data
  - Runs on Ubuntu only, for pull requests
  - Downloads and caches test data
  - Tests data processing pipelines

- **gpu-tests**: Tests on self-hosted GPU runners
  - **DISABLED BY DEFAULT** - Only runs when:
    - Manually triggered via workflow_dispatch, OR
    - PR has the `run-gpu-tests` label
  - Requires self-hosted runner with `[self-hosted, linux, gpu]` labels
  - Uses PyTorch with CUDA 13.0 support
  - Timeout: 30 minutes

- **code-quality**: Static code analysis
  - Ruff formatting and linting checks
  - mypy type checking (continue-on-error: true)
  - Ruff checks will fail the build if code style issues are found

### `test-slow.yml` - Long-Running Tests

Runs nightly at 2 AM UTC or on manual trigger. Includes:

- **test-slow-gpu**: Slow tests requiring GPU
  - Tests marked with `slow` marker
  - Extended timeout: 60 minutes
  - Uses self-hosted GPU runners
  - Will wait indefinitely if no runner is available

### `docs.yml` - Documentation Build and Deploy

Two-job workflow for building and deploying Sphinx documentation:

**Job 1: build-docs** (runs on all events - PRs and pushes)
- Installs documentation dependencies
- Builds HTML documentation with Sphinx
- Checks for warnings
- Uploads documentation artifacts (retained for 7 days)

**Job 2: deploy** (runs only on push to main)
- Downloads built documentation
- Deploys to GitHub Pages using GitHub's official deployment action
- Uses `github-pages` environment with protection rules
- No gh-pages branch needed (modern deployment workflow)

This separation ensures PRs can build and validate docs without triggering environment protection rules.

## Caching Strategy

The workflows use multiple caching layers to speed up builds:

1. **Python package cache** via `setup-python` action
   - Caches pip packages based on `pyproject.toml` hash

2. **Additional pip cache** via `actions/cache`
   - Caches `~/.cache/pip` directory
   - Separate keys for CPU, GPU, and integration tests
   - Hierarchical restore keys for fallback

## GPU Support

### Important: GPU Tests Are Disabled by Default

⚠️ **GPU tests do NOT run automatically** to prevent jobs from waiting indefinitely in the queue when no runner is available.

To run GPU tests, you must either:
1. **Manually trigger the workflow**: Go to Actions > CI > Run workflow
2. **Add the `run-gpu-tests` label** to your pull request

### Self-Hosted Runners

GPU tests require self-hosted runners with:
- Linux OS
- NVIDIA GPU with CUDA 13.0 support
- Runner labels: `[self-hosted, linux, gpu]`

**Why are GPU tests disabled by default?**
- GitHub Actions jobs wait indefinitely for a self-hosted runner if none are available
- The `timeout-minutes` setting only applies AFTER a runner picks up the job
- This can block CI pipelines and create confusion when runners are offline

### Setting Up Self-Hosted GPU Runners

1. **Install GitHub Actions Runner**:
   ```bash
   # Download and configure runner from GitHub repository Settings > Actions > Runners
   ```

2. **Install NVIDIA Drivers and CUDA**:
   ```bash
   # Install NVIDIA drivers
   sudo apt-get install nvidia-driver-535
   
   # Install CUDA toolkit 13.0
   # CUDA 13.0 requires Ubuntu 22.04 LTS or later. These cuda-keyring_1.1-1_all.deb
   # and cuda-toolkit-13 commands fail on Ubuntu 20.04 runners.
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb
   sudo apt-get update
   sudo apt-get install cuda-toolkit-13
   ```

   Self-hosted GPU runners should be upgraded to Ubuntu 22.04 LTS or later
   before installing CUDA 13.0.

3. **Configure Runner Labels**:
   - Add labels: `self-hosted`, `linux`, `gpu`
   - Verify GPU is accessible: `nvidia-smi`

4. **Start the Runner**:
   ```bash
   ./run.sh
   ```

### How to Run GPU Tests

**Option 1: Manual Workflow Trigger**
1. Go to your repository on GitHub
2. Click "Actions" tab
3. Select "CI" workflow from the left sidebar
4. Click "Run workflow" button
5. Select branch and click "Run workflow"

**Option 2: Add Label to PR**
1. Open your pull request
2. Add the `run-gpu-tests` label
3. The CI workflow will automatically include GPU tests

**Option 3: Run Locally**
```bash
# Install with CUDA support
uv pip install -e ".[test,cuda13]"

# Run GPU tests
pytest tests/ -v -m "not slow"
pytest tests/ -v -m "slow"  # For long-running tests
```

### GitHub-Hosted Runners

GitHub-hosted runners do **not** have GPU support. All GPU tests require self-hosted runners with NVIDIA GPUs.

## Test Dependencies

Test dependencies are installed from `pyproject.toml`:

```bash
pip install -e ".[test]"
```

This installs:
- pytest >= 7.0.0
- pytest-cov >= 4.0.0
- pytest-xdist >= 3.0.0
- pytest-timeout >= 2.0.0
- coverage[toml] >= 7.0.0

## Test Markers

Tests should be marked appropriately:

```python
import pytest

@pytest.mark.unit
def test_simple_function():
    """Fast unit test"""
    pass

@pytest.mark.integration
def test_full_pipeline():
    """Integration test"""
    pass

@pytest.mark.slow
def test_long_running():
    """Long-running test"""
    pass

@pytest.mark.requires_gpu
def test_gpu_function():
    """Test requiring GPU"""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")
    pass
```

## Running Tests Locally

### CPU Tests
```bash
# Install dependencies
pip install -e ".[test]"

# Run unit tests
pytest tests/ -m "unit and not requires_gpu"

# Run with coverage
pytest tests/ -m "unit and not requires_gpu" --cov=physiomotion4d
```

### GPU Tests
```bash
# Install with CUDA support
uv pip install -e ".[test,cuda13]"

# Run all tests (including GPU)
pytest tests/ -m "not slow"

# Run slow tests
pytest tests/ -m "slow"
```

## Coverage Reports

Coverage reports are:
1. Uploaded to Codecov (if configured)
2. Stored as artifacts for 7 days
3. Available as HTML reports in the `htmlcov/` directory

## Troubleshooting

### GPU Tests Not Running

GPU tests are disabled by default. If you want to run them:
1. **Check if GPU tests should run**: They only run on manual trigger or with `run-gpu-tests` label
2. **Verify self-hosted runner is online**: Settings > Actions > Runners
3. **Check runner labels**: Runner must have `self-hosted`, `linux`, and `gpu` labels
4. **Verify GPU accessibility**: Run `nvidia-smi` on the runner machine
5. **Check workflow logs**: Look for "Waiting for a runner" or "runner assignment" messages

If GPU tests are stuck "Waiting for a runner":
- The runner is offline or not properly configured
- Cancel the workflow run (GPU tests won't hold up other jobs due to `continue-on-error: true`)

### Cache Not Working

If builds are slow:
1. Check cache hit/miss in workflow logs
2. Verify `pyproject.toml` hasn't changed unexpectedly
3. Try clearing caches: Settings > Actions > Caches

### Test Failures

For test failures:
1. Check individual test logs in the workflow run
2. Run tests locally to reproduce
3. Use `pytest -v --tb=long` for detailed error traces
4. Check if tests are marked correctly (unit/integration/slow/requires_gpu)

