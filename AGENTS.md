# AGENTS.md

Role-based guidance for AI agents working in this repository.

PhysioMotion4D converts 4D CT scans into animated USD models for NVIDIA
Omniverse. It is an **early-alpha** scientific Python library. Clarity beats
premature optimization. Breaking changes are acceptable. Backward compatibility
is not a goal.

## Role

We are developing open-source code for scientific AI libraries. Leverage
GPU-accelerated methods when appropriate.

## Priorities

1. Accuracy.
2. Clarity, maintainability, and simplicity.
3. Consistency with the rest of the platform and open-source standards.
4. Documentation.
5. Testing.

## Behavior

1. Do not assume. Do not hide confusion. Surface tradeoffs.
2. Minimum code that solves the problem. Nothing speculative.
3. Touch only what you must. Clean up only your own mess.
4. Define success criteria. Loop until verified.

## Developer Tool Prerequisites

Non-Python tools used by contributor workflows:

- **Codex CLI** (`codex`) - can run the `.agents/` slash skills and is the
  default PR-review agent for `ai_agent_github_reviews.py`.
- **Claude Code CLI** (`claude`) - can run the `.agents/` slash skills and
  `ai_agent_github_reviews.py --agent claude`.
  Install: `winget install Anthropic.ClaudeCode`.
- **gh CLI** (`gh`) - required by `ai_agent_github_reviews.py` to fetch PR
  review data. Install: `winget install GitHub.cli` then `gh auth login`.
  Not installable via pip/uv; it is a compiled Go binary.

## Common Commands

Prefer the repository-local virtual environment at `.\venv`. Activate it before
issuing Python commands so `python`, console scripts, and `uv pip` all use that
environment. If activation is not possible, invoke
`.\venv\Scripts\python.exe -m ...` directly. Use `uv run ...` only when the
local `venv` is unavailable and you need uv to create or sync an environment.

```powershell
# Create the repo-local environment if it does not already exist
uv venv venv
.\venv\Scripts\Activate.ps1

# Install in editable mode
uv pip install -e .

# Lint and format
python -m ruff check . --fix && python -m ruff format .

# Type checking
python -m mypy src/ tests/

# All pre-commit hooks
python -m pre_commit run --all-files

# Fast tests
python -m pytest tests/ -v

# Single test file or test by name
python -m pytest tests/test_contour_tools.py -v
python -m pytest tests/test_contour_tools.py::test_extract_surface -v

# Opt-in test buckets
python -m pytest tests/ -v --run-slow
python -m pytest tests/ -v --run-gpu
python -m pytest tests/ -v --run-simpleware
python -m pytest tests/ -v --run-physicsnemo
python -m pytest tests/ -v --run-experiments
python -m pytest tests/ -v --run-tutorials

# Enable every bucket at once (equivalent to passing all --run-* flags)
python -m pytest tests/ -v --run-all

# Typical local GPU profile
python -m pytest tests/ -v --run-gpu --run-slow

# Coverage
python -m pytest tests/ --cov=src/physiomotion4d --cov-report=html

# Create missing baselines
python -m pytest tests/ --create-baselines
```

Version bumping: `bumpver update --patch`, `--minor`, or `--major`.

## Codex Sandbox

- If a Python command fails with
  `No Python at '"C:\Users\saylward\AppData\Local\Programs\Python\Python311\python.exe'`,
  do not assume Python 3.11 is missing. The Codex sandbox can break the
  launcher or venv path.
- Use the temporary validation environment instead:
  `C:\Users\saylward\AppData\Local\Temp\pm4d-uv-env\Scripts\python.exe`.
  It has been verified as Python 3.11.9 with `ruff` available.
- Run that temporary venv outside the sandbox when needed. Treat this as an
  environment/sandbox workaround, not a dependency or installation problem.

## Universal Rules

- Read the relevant source files before proposing changes.
- Runtime classes for workflows, segmentation, registration, and USD tools
  inherit from `PhysioMotion4DBase`; new runtime classes must too. Standalone
  utility scripts and data/container/helper classes do not.
- In classes that inherit from `PhysioMotion4DBase`, use `self.log_info()` and
  `self.log_debug()`, never `print()`. Standalone scripts may use `print()`.
- No emojis in `.py` files; avoid them in docs too. Windows cp1252 encoding
  has broken this project before.
- The public VTK→USD entry point is `ConvertVTKToUSD`. Experiments, CLIs,
  tests, and tutorials must use it. Do not import from the `vtk_to_usd/`
  subpackage directly outside of `convert_vtk_to_usd.py` and the subpackage
  itself.
- Scripts that instantiate `SegmentChestTotalSegmentator` must guard the
  top-level invocation with `if __name__ == "__main__":` on Windows
  (`torch.multiprocessing` requires it).
- Double quotes for strings and docstrings. Keep lines at or
  below 88 characters.
- Full type hints are required under strict mypy. Use `Optional[X]`, not
  `X | None`.
- Run `python -m pytest tests/ -v` from the active `.\venv` to verify changes.
  Slow, GPU, Simpleware,
  experiment, and tutorial tests are auto-skipped unless their opt-in flag is
  passed.
- The `requires_data` marker no longer exists. Tests that need external data
  download it automatically via the session fixtures.
- Consult `docs/API_MAP.md` to locate classes, methods, and signatures before
  searching manually.
- Do not commit changes or make pull requests unless specifically told to do so.

## Data Conventions

- Images are `itk.Image` objects with axes X, Y, Z [, T] in LPS world space.
  `itk.imread` normalizes DICOM, NIfTI, MHA, and NRRD inputs to LPS. Persist
  images with `itk.imwrite(..., compression=True)`.
- 4D time series use shape `(X, Y, Z, T)`. Never silently squeeze or permute
  axes.
- Surfaces are `pv.PolyData` in LPS, inherited from the source `itk.Image` via
  `itk.vtk_image_from_image`.
- Convert surfaces to USD right-handed Y-up only at USD export by
  `vtk_to_usd.lps_points_to_usd`:
  USD `+X=Left`, `+Y=Superior`, `+Z=Anterior`.
- Masks are ITK images with integer labels. Keep anatomy group IDs consistent
  across segmenters.
- Transforms are ITK composite transforms stored in compressed `.hdf` files.
- State axis order and shape explicitly in every docstring and comment that
  touches arrays.

## Implementation Role

- Summarize current behavior in 2-4 sentences before editing.
- Identify success criteria or metrics.
- Refer to `*_tools.py` files for commonly used routines.
- Refer to `workspace/reference_code`, when available, for third-party library
  usage.
- Propose a numbered plan; confirm before implementing non-trivial structural
  changes.
- Keep diffs small and reviewable.
- Prefer editing existing modules over creating new ones.
- Call out breaking changes explicitly.
- No backward-compatibility shims: just change the code.

## Testing Role

- Strongly prefer real downloaded test data over synthetic data. Request the
  session fixtures `test_directories`, `download_test_data`, and `test_images`
  so standard datasets are fetched automatically on first use.
- Real data exercises preprocessing, resampling, dtype handling, and
  world-frame metadata paths that synthetic toy volumes silently bypass.
- Only fall back to synthetic `itk.Image` or `pv.PolyData` inputs when the
  behavior under test is a pure unit such as axis arithmetic or dict routing,
  or when real data would push the test into a slow, GPU, or Simpleware bucket
  that does not fit the test's purpose. Keep synthetic volumes at or below 64
  voxels per side and say so in the docstring.
- State image shape and axis order in every test docstring, for example:
  `shape (X, Y, Z, T) = (64, 64, 32, 1), LPS world frame`.
- When a test produces an image or surface, compare against a baseline using
  `src/physiomotion4d/test_tools.py` utilities such as `TestTools`.
- Store baselines under `tests/baselines/`, which is tracked by Git LFS. Run
  `git lfs pull` after cloning.
- Run with `--create-baselines` to materialize missing baselines on first use.
- `tests/conftest.py` owns session-scoped fixtures that chain download,
  convert, segment, and register.
- Mark tests that need a GPU, a slow runtime, or a licensed Simpleware install
  with `@pytest.mark.requires_gpu`, `@pytest.mark.slow`, or
  `@pytest.mark.requires_simpleware`.
- Mark experiment and tutorial tests with `@pytest.mark.experiment` or
  `@pytest.mark.tutorial`.
- Tests that just need downloadable data need no marker; the fixture chain
  handles it.
- Prefer images from `ROOT/data/test/slicer_heart_small` for tests.
- Prefer storing test results in subdirectories under `./results/<test_name>`.

## Documentation Role

- Update docstrings for every changed public method. Keep claims factual.
- Document with docstrings and inline comments.
- Do not create new `.md` files unless explicitly requested.
- Regenerate `docs/API_MAP.md` after any public API change from the active
  `.\venv`: `python utils/generate_api_map.py`.

## Architecture Role

- Propose a numbered design plan with tradeoffs before structural changes.
- Identify every file that will change and how the class hierarchy is affected.
- Flag changes at the ITK/PyVista boundary or the RAS to Y-up coordinate
  transform as high-risk.

## File Operations

- Use `git mv` and `git rm`, not `mv` or `rm`, to preserve history.
