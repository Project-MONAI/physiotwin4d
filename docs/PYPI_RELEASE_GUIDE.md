# PyPI Release Guide for PhysioTwin4D

This guide provides step-by-step instructions for building and uploading PhysioTwin4D to PyPI.

## Prerequisites

### 1. Install Build Tools

```bash
pip install --upgrade build twine
```

### 2. PyPI Account Setup

- Create accounts on:
  - PyPI (production): https://pypi.org/account/register/
  - TestPyPI (testing): https://test.pypi.org/account/register/

### 3. Configure API Tokens

1. Generate API tokens:
   - PyPI: https://pypi.org/manage/account/token/
   - TestPyPI: https://test.pypi.org/manage/account/token/

2. Create `~/.pypirc` (or `%USERPROFILE%\.pypirc` on Windows):

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-AgEIcHlwaS5vcmcC...  # Your PyPI token

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-AgENdGVzdC5weXBp...  # Your TestPyPI token
```

## Pre-Release Checklist

### 1. Update Version Number

Edit `pyproject.toml`:
```toml
[tool.bumpver]
current_version = "2026.07.0"  # Update this
```

Or use bumpver:
```bash
# Install bumpver if not already installed
pip install bumpver

# Bump patch version
bumpver update --patch

# Bump month version (for new month)
bumpver update --minor

# Bump year version (for new year)
bumpver update --major
```

### 2. Update Release Notes

Add release notes for the new version in `docs/changelog.rst` and in the
GitHub release description.

### 3. Run Code Quality Checks

```bash
# Format code
black src/

# Sort imports
isort src/

# Run linters
flake8 src/
pylint src/

# Run tests (if available)
pytest tests/
```

### 4. Test Installation Locally

```bash
# Install in editable mode
pip install -e .

# Verify imports
python -c "import physiotwin4d; print(physiotwin4d.__version__)"

# Test CLI commands
physiotwin4d --help
physiotwin4d-convert-image-to-usd --help
```

## Building the Package

### 1. Clean Previous Builds

```bash
# Remove old build artifacts
rm -rf build/ dist/ *.egg-info
# On Windows: rmdir /s /q build dist *.egg-info
```

### 2. Build Distribution Files

```bash
python -m build
```

This creates:
- `dist/physiotwin4d-2026.07.0.tar.gz` (source distribution)
- `dist/physiotwin4d-2026.07.0-py3-none-any.whl` (wheel distribution)

### 3. Verify Build Contents

```bash
# List contents of the wheel
unzip -l dist/physiotwin4d-*.whl

# Check if all necessary files are included
tar -tzf dist/physiotwin4d-*.tar.gz
```

## Testing on TestPyPI (Recommended)

### 1. Upload to TestPyPI

```bash
twine upload --repository testpypi dist/*
```

### 2. Install from TestPyPI

```bash
# Create a fresh virtual environment
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple \
    physiotwin4d
```

Note: `--extra-index-url` is needed because dependencies are on PyPI, not TestPyPI.

### 3. Test the Installation

```python
# Test imports
import physiotwin4d
from physiotwin4d import WorkflowConvertImageToUSD

print(f"Version: {physiotwin4d.__version__}")
print(WorkflowConvertImageToUSD.__name__)
```

```bash
# Test CLI
physiotwin4d-convert-image-to-usd --help
```

## Publishing to PyPI

### 1. Final Checks

- [ ] All tests pass
- [ ] TestPyPI installation works correctly
- [ ] Version number is updated
- [ ] Release notes are updated
- [ ] README.md has correct installation instructions
- [ ] All files committed to git

### 2. Upload to PyPI

```bash
twine upload dist/*
```

### 3. Verify Upload

Visit: https://pypi.org/project/physiotwin4d/

### 4. Test Installation from PyPI

```bash
# Fresh environment
python -m venv pypi_test
source pypi_test/bin/activate

# Install from PyPI
pip install physiotwin4d

# Verify
python -c "import physiotwin4d; print(physiotwin4d.__version__)"
```

## Post-Release Steps

### 1. Tag the Release in Git

```bash
git tag -a v2026.07.0 -m "Release version 2026.07.0"
git push origin v2026.07.0
```

### 2. Create GitHub/GitLab Release

Create a release on your repository with:
- Tag: `v2026.07.0`
- Release notes from `docs/changelog.rst`
- Attach distribution files (optional)

### 3. Update Documentation

- Update any version-specific documentation
- Update installation instructions if needed
- Announce the release (mailing lists, social media, etc.)

## Troubleshooting

### Build Errors

**Problem**: Missing files in distribution
```bash
# Solution: Check MANIFEST.in includes all necessary files
# Add missing patterns to MANIFEST.in
```

**Problem**: Import errors after installation
```bash
# Solution: Verify package structure
# Check that src/physiotwin4d/__init__.py exports necessary classes
```

### Upload Errors

**Problem**: `403 Forbidden` error
```bash
# Solution: Check API token permissions
# Regenerate token if necessary
```

**Problem**: Package name already exists
```bash
# Solution: You cannot re-upload the same version
# Bump version number and rebuild
```

**Problem**: File already exists
```bash
# Solution: You cannot replace existing files on PyPI
# Delete dist/ and rebuild with new version number
```

### Installation Issues

**Problem**: Dependencies not installing
```bash
# Solution: Check dependency specifications in pyproject.toml
# Ensure all dependencies are available on PyPI
```

**Problem**: CUDA/PyTorch issues
```bash
# Solution: Users may need to install PyTorch separately with CUDA support
# Add note to README about PyTorch installation
```

## Useful Commands Reference

```bash
# Check package metadata
twine check dist/*

# Upload to specific repository
twine upload --repository testpypi dist/*
twine upload --repository pypi dist/*

# Build only wheel
python -m build --wheel

# Build only source distribution
python -m build --sdist

# Install in editable mode with extras
pip install -e ".[dev,test]"

# View package information
python -m pip show physiotwin4d
```

## Best Practices

1. **Always test on TestPyPI first** before uploading to production PyPI
2. **Use semantic/calendar versioning** consistently
3. **Keep release notes updated** with every release
4. **Tag releases in git** for traceability
5. **Test installation in clean environment** before releasing
6. **Document breaking changes** clearly in release notes
7. **Keep dependencies up to date** but avoid breaking changes

## Additional Resources

- [Python Packaging User Guide](https://packaging.python.org/)
- [PyPI Help](https://pypi.org/help/)
- [Setuptools Documentation](https://setuptools.pypa.io/)
- [Twine Documentation](https://twine.readthedocs.io/)
- [PEP 517 - Build Backend](https://peps.python.org/pep-0517/)
- [PEP 621 - Project Metadata](https://peps.python.org/pep-0621/)
