# CLAUDE.md

Project guidance for Claude Code in this repository.

Codex and other AI agents should use `AGENTS.md` as the primary shared
instructions file. Claude-specific behavior and slash-command usage remain here.

## Role:
We are developing open-source code for scientific AI libraries. Leverage GPU-accelerated methods when appropriate.

## Priorities (Ordered)
1) accuracy
2) clarity/maintainability/simplicity
3) consistency with the rest of the platform and open source standards
4) documentation
5) testing

## Behavior Guidelines
1) Don't assume.  Don't hide confuction. Surface tradeoffs.
2) Minimum code that solves the problem.  Nothing speculative.
3) Touch only what you must.  Clean up only your own mess.
4) Define success criteria.  Loop until verified.

## Commands

**Python launcher:** Use `py` on this Windows system (not `python`).

```bash
# Install in editable mode (preferred)
uv pip install -e .

# Lint and format
ruff check . --fix && ruff format .

# Type checking
mypy src/ tests/

# All pre-commit hooks
pre-commit run --all-files

# Fast tests (recommended for development)
py -m pytest tests/ -m "not slow and not requires_data" -v

# Single test file or test by name
py -m pytest tests/test_contour_tools.py -v
py -m pytest tests/test_contour_tools.py::test_extract_surface -v

# Skip GPU-dependent tests
py -m pytest tests/ --ignore=tests/test_segment_chest_total_segmentator.py \
              --ignore=tests/test_register_images_icon.py

# With coverage
py -m pytest tests/ --cov=src/physiomotion4d --cov-report=html

# Experiment script tests (very slow, opt-in)
py -m pytest tests/ --run-experiments

# Create missing baselines
py -m pytest tests/ --create-baselines
```

**Version bumping:** `bumpver update --patch` (or `--minor`, `--major`)

## Architecture

All classes inherit from `PhysioMotion4DBase` (`physiomotion4d_base.py`), which provides
a shared logger. Use `self.log_info()`, `self.log_debug()` — never `print()`.

Consult `docs/API_MAP.md` for the full index of classes, methods, and signatures.
Regenerate it after any public API change: `py utils/generate_api_map.py`

**Key data conventions:**
- Images: `itk.Image`, axes X, Y, Z [, T] in LPS world space (ITK's native
  frame; `itk.imread` normalizes DICOM, NIfTI, MHA, and NRRD inputs to LPS)
  stored using itk.imwrite with compression=True
- 4D time series: shape `(X, Y, Z, T)` — never silently squeeze or permute axes
- Surfaces: `pv.PolyData` in LPS (inherited from the source `itk.Image` via
  `itk.vtk_image_from_image`); converted to USD right-handed Y-up only at USD
  export by `vtk_to_usd.lps_points_to_usd` (USD +X=Left, +Y=Superior, +Z=Anterior)
- Masks: ITK images with integer labels; consistent anatomy group IDs across all segmenters
- Transforms: ITK composite transforms stored in `.hdf` files with compression
- State axis order and shape explicitly in every docstring and comment that touches arrays

## Testing

- Baselines in `tests/baselines/` via Git LFS — run `git lfs pull` after cloning
- `tests/conftest.py`: session-scoped fixtures chaining download → convert → segment → register
- `src/physiomotion4d/test_tools.py`: baseline comparison utilities (`TestTools`, etc.)
- Markers: `slow`, `requires_gpu`, `requires_data`, `experiment`, `tutorial`
- Prefer images from `ROOT/data/test/slicer_heart_small` for tests
- Prefer storing results in subdirs `./results/<test_name>`

## Working Process

Before editing any code:
1. Read the relevant source file(s) in full.
2. Summarize current behavior in 2–4 sentences.
3. Identify success criteria / metrics
4. Refer to *_tools.py files for commonly used routines
5. Refer to workspace/reference_code (when available) for third-party libraries
6. Propose a numbered plan; confirm before implementing non-trivial changes.
7. Follow the behavior guidelines given above.
8. Implement in small, reviewable diffs.
9. Update docstrings and tests for every changed public method.
10. Call out breaking changes explicitly.

Breaking changes are acceptable. Backward-compatibility shims are not.

## Agents and Skills

Role-specific subagents live in `.agents/agents/`; slash-command skills in
`.agents/skills/`. See `AGENTS.md` for role-based guidance that applies across
Claude, Codex, and other AI tooling.

- `/plan` — inspect files, summarize design, produce a numbered plan (no code changes)
- `/impl` — read → summarize → plan → implement in small diffs
- `/test-feature` — propose test plan, write synthetic-data pytest tests
- `/doc-feature` — update docstrings and regenerate API map

## File Operations

Use `git mv` / `git rm` — not `mv` / `rm` — to preserve history.

## Documentation Policy

Do **not** create new `.md` files unless explicitly requested.
Document via docstrings and inline comments.

## Code Style

- Single quotes for strings; double quotes for docstrings
- Full type hints (`mypy` strict; `disallow_untyped_defs = true`)
- `Optional[X]` not `X | None` (ruff `UP007` suppressed)
- Breaking changes are acceptable — backward compatibility is not a priority
- Max line length: 88 characters
- Follow behavior guidelines.
