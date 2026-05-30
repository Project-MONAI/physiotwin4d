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
1) Don't assume.  Don't hide confusion. Surface tradeoffs.
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

# Fast tests (recommended for development — slow/GPU/Simpleware/experiment
# /tutorial tests are auto-skipped unless their opt-in flag is passed)
py -m pytest tests/ -v

# Single test file or test by name
py -m pytest tests/test_contour_tools.py -v
py -m pytest tests/test_contour_tools.py::test_extract_surface -v

# Opt-in buckets (each flag enables one marker family)
py -m pytest tests/ -v --run-slow         # tests marked 'slow'
py -m pytest tests/ -v --run-gpu          # tests marked 'requires_gpu'
py -m pytest tests/ -v --run-simpleware   # tests marked 'requires_simpleware'
py -m pytest tests/ -v --run-physicsnemo  # tests marked 'requires_physicsnemo'
py -m pytest tests/ -v --run-experiments  # tests marked 'experiment'
py -m pytest tests/ -v --run-tutorials    # tests marked 'tutorial'

# Enable every bucket at once (equivalent to passing all --run-* flags).
# Self-hosted CI GPU runner uses this after installing [test,cuda13,physicsnemo].
py -m pytest tests/ -v --run-all

# Typical local GPU profile.
py -m pytest tests/ -v --run-gpu --run-slow

# With coverage
py -m pytest tests/ --cov=src/physiomotion4d --cov-report=html

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
- Markers (all opt-in via `--run-<bucket>`): `slow`, `requires_gpu`,
  `requires_simpleware`, `experiment`, `tutorial`. Data-dependent tests no
  longer use a marker — they pull data through fixtures and run by default.
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
11. Do not commit changes or make pull requests unless specifically told to do so.

Breaking changes are acceptable. Backward-compatibility shims are not.

## Agents and Skills

Role-specific subagents live in `.agents/agents/`; slash-command skills in
`.agents/skills/`. See `AGENTS.md` for role-based guidance that applies across
Claude, Codex, and other AI tooling.

- `/plan` — inspect files, summarize design, produce a numbered plan (no code changes)
- `/impl` — read → summarize → plan → implement in small diffs
- `/test-feature` — propose test plan, write real-data-driven pytest tests with baselines
- `/doc-feature` — update docstrings (and remind you to run `/regen-api-map`)
- `/regen-api-map` — regenerate `docs/API_MAP.md` and report public-API changes
- `/check-conventions` — audit changed files against project hard rules
  (base-class, logging, coordinate frame, USD entry point, Windows mp guard,
  quoting, type hints, line length, emoji ban)
- `/simplify-staged` — readability / quality pass over `git diff HEAD`
- `/commit` — stage tracked changes, draft `<TAG>: …` message, loop until hooks pass
- `/review-pr <NUMBER>` — drive `utils/ai_agent_github_reviews.py` to triage
  a PR's review comments and apply accepted edits as pending changes

## File Operations

Use `git mv` / `git rm` — not `mv` / `rm` — to preserve history.

## Documentation Policy

Do **not** create new `.md` files unless explicitly requested.
Document via docstrings and inline comments.

## Code Style

- Double quotes for strings and docstrings
- Full type hints (`mypy` strict; `disallow_untyped_defs = true`)
- `Optional[X]` not `X | None` (ruff `UP007` suppressed)
- Breaking changes are acceptable — backward compatibility is not a priority
- Max line length: 88 characters
- Follow behavior guidelines.
