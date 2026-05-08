# AGENTS.md

Role-based guidance for AI agents working in this repository.

PhysioMotion4D converts 4D CT scans into animated USD models for NVIDIA Omniverse.
It is an **early-alpha** scientific Python library. Clarity beats premature optimization.
Breaking changes are acceptable. Backward compatibility is not a goal.

## Developer tool prerequisites

Non-Python tools used by contributor workflows:

- **Codex CLI** (`codex`) — can run the `.agents/` slash skills and
  is the default PR-review agent for `ai_agent_github_reviews.py`.
- **Claude Code CLI** (`claude`) — can run the `.agents/` slash skills and
  `ai_agent_github_reviews.py --agent claude`.
  Install: `winget install Anthropic.ClaudeCode`
- **gh CLI** (`gh`) — required by `ai_agent_github_reviews.py` to fetch PR review data.
  Install: `winget install GitHub.cli` then `gh auth login`
  Not installable via pip/uv — it is a compiled Go binary.

## Universal rules

- Read the relevant source files before proposing changes.
- Runtime classes (workflow, segmentation, registration, USD tools) inherit from
  `PhysioMotion4DBase`; new runtime classes must too. Standalone utility scripts
  and data/container/helper classes do not.
- In classes that inherit from `PhysioMotion4DBase`, use `self.log_info()` /
  `self.log_debug()` — never `print()`. Standalone scripts may use `print()`.
- Single quotes for strings; double quotes for docstrings. 88-char line limit.
- Full type hints (`mypy` strict). Use `Optional[X]` not `X | None`.
- Run `py -m pytest tests/ -m "not slow and not requires_data" -v` to verify changes.
- Consult `docs/API_MAP.md` to locate classes and methods before searching manually.

## Implementation role

- Summarize current behavior → propose numbered plan → implement.
- Keep diffs small and reviewable. Call out breaking changes explicitly.
- Prefer editing existing modules over creating new ones.
- No backward-compat shims: just change the code.

## Testing role

- Prefer synthetic `itk.Image` and small `pv.PolyData` surfaces — not real patient data.
- State image shape and axis order in every test docstring: e.g. `shape (X, Y, Z, T)`.
- Keep synthetic volumes ≤64 voxels per side for speed.
- Mark tests that genuinely need real data with `@pytest.mark.requires_data`.
- Use `test_tools.py` baseline utilities for surface and image regression checks.

## Documentation role

- Update docstrings for every changed public method. Keep claims factual.
- Do not create new `.md` files unless explicitly requested.
- Regenerate `docs/API_MAP.md` after any public API change:
  `py utils/generate_api_map.py`

## Architecture role

- Propose a numbered design plan with trade-offs before structural changes.
- Identify every file that will change and how the class hierarchy is affected.
- Flag changes at the ITK↔PyVista boundary or the RAS→Y-up coordinate transform as high-risk.
