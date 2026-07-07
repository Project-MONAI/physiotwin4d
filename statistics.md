# PhysioTwin4D - Software Development Statistics

**Report Generated:** July 7, 2026
**Project Version:** 2026.07.0
**Status:** Beta (Development Status: 4 - Beta)

---

## Executive Summary

PhysioTwin4D is a collection of methods, workflows, tutorials, and CLI tools
for creating personalized physiological digital twins from 3D medical images.
This report summarizes development effort, code quality, and project maturity.

### Key Metrics at a Glance

| Metric                         | Value                                          |
| ------------------------------ | ---------------------------------------------- |
| **Total Lines of Code**        | ~57,100                                        |
| **Development Period**         | December 5, 2025 - July 6, 2026 (~7 months)    |
| **Total Commits**              | 83                                             |
| **Primary Developer**          | 1 (Stephen Aylward), plus 1 outside contributor |

---

## Detailed Code Statistics

### Lines of Code Breakdown

| Category                                | Files          | Lines of Code | Percentage |
| ---------------------------------------- | -------------- | -------------- | ---------- |
| **Core Python Source (`src/`)**          | 59 files       | 22,401         | 39.2%      |
| **Test Suite (`tests/`)**                | 33 files       | 9,502          | 16.6%      |
| **Experiment Scripts (`experiments/`)**  | 41 files       | 8,494          | 14.9%      |
| **Tutorial Scripts (`tutorials/`)**      | 11 files       | 3,918          | 6.9%       |
| **Utility Scripts (`utils/`)**           | 3 files        | 1,956          | 3.4%       |
| **Documentation (`docs/*.rst`)**         | 73 files       | 7,064          | 12.4%      |
| **Markdown (repo-wide READMEs, guides)** | 30 files       | 3,780          | 6.6%       |
| **TOTAL**                                | **250 files**  | **~57,100**    | **100%**   |

All experiment and tutorial sources are plain `.py` files. Each uses `# %%`
percent-cell markers so the same file can be executed end-to-end with
`python <script>.py` or stepped through cell-by-cell in VS Code / Cursor.

### Core Module Highlights (Python Source)

| Module                                          | Lines | Purpose                                        |
| ------------------------------------------------ | ----- | ---------------------------------------------- |
| `usd_tools.py`                                   | 1,519 | USD file manipulation and inspection           |
| `transform_tools.py`                             | 1,237 | ITK transform utilities                        |
| `convert_vtk_to_usd.py`                          | 1,001 | High-level VTK -> USD converter                |
| `workflow_fit_statistical_model_to_patient.py`   | 948   | Model-to-patient registration workflow         |
| `workflow_fine_tune_icon_registration.py`        | 912   | Fine-tuning workflow for Icon registration      |
| `register_models_pca.py`                         | 854   | PCA-based shape model registration             |
| `register_images_ants.py`                        | 759   | ANTs-based image registration                  |
| `vtk_to_usd/` subpackage                         | 2,661 | Low-level VTK -> USD building blocks (9 files) |
| `cli/` subpackage                                | 2,153 | CLI entry-point scripts (11 commands)          |
| `contour_tools.py`                               | 670   | Mesh extraction and contour manipulation       |
| `register_images_greedy.py`                      | 624   | Greedy classical deformable registration       |
| `register_time_series_images.py`                 | 600   | Time series registration for 4D CT             |

---

## Project Maturity Indicators

| Indicator                  | Status                                              |
| --------------------------- | ---------------------------------------------------- |
| **Documentation Coverage**  | Sphinx site + per-package READMEs                    |
| **Test Suite Present**      | Yes (`tests/` with baselines via Git LFS)             |
| **CI/CD Pipeline**          | GitHub Actions (Ubuntu + Windows; Python 3.11/3.12), plus a self-hosted Windows GPU runner |
| **Dependency Management**   | `pyproject.toml`, `uv`-friendly                       |
| **Code Quality Tools**      | Ruff (lint + format), mypy                            |
| **Example Scripts**         | 41 experiment scripts + 11 tutorial scripts           |
| **Version Management**      | Calendar versioning via bumpver                       |
| **API Reference**           | Google-style docstrings + generated `docs/API_MAP.md` (via `py utils/generate_api_map.py`) |
| **Package Distribution**    | PyPI-ready                                            |

---

## Technical Complexity Assessment

### Domain Complexity

PhysioTwin4D operates across several technically demanding domains:

| Domain                   | Complexity Level | Key Technologies                       |
| ------------------------- | ----------------- | ---------------------------------------- |
| **Medical Imaging**      | Very High         | ITK, MONAI, nibabel, pydicom, pynrrd     |
| **Deep Learning**        | High               | PyTorch, CUDA 13, transformers            |
| **3D Graphics / USD**    | High               | VTK, PyVista, OpenUSD, trimesh            |
| **Image Registration**   | Very High          | ANTs, Greedy, Icon, UniGradICON           |
| **AI Segmentation**      | High               | TotalSegmentator, Simpleware bridge       |
| **Geometric Processing** | High               | ICP, PCA, distance maps, statistical shape models |

### Architectural Sophistication

- Class hierarchy depth: 3-4 levels (well-structured inheritance from
  `PhysioTwin4DBase`)
- Module coupling: medium (clear separation between segmentation,
  registration, USD conversion, and workflow layers)
- Public API surface documented via generated `docs/API_MAP.md`
- ~25 major external dependencies (medical imaging, AI/ML, USD, registration)

---

## Dependencies & Infrastructure

### Core Dependencies (selected)

| Category              | Key Packages                                        |
| ---------------------- | ----------------------------------------------------- |
| **Medical Imaging**    | ITK, MONAI, nibabel, pydicom, pynrrd                 |
| **Deep Learning**      | PyTorch, CuPy (CUDA 13), transformers                |
| **Registration**       | ANTs (antspyx), picsl-greedy, icon-registration, UniGradICON |
| **3D Graphics / USD**  | VTK, PyVista, USD-core, trimesh, netgen-mesher        |
| **AI Segmentation**    | TotalSegmentator                                      |
| **Development Tools**  | pytest, pytest-cov, pytest-xdist, ruff, mypy, sphinx, uv |

### Infrastructure Files

| File             | Purpose                                             |
| ---------------- | --------------------------------------------------- |
| `pyproject.toml` | Modern Python packaging, dependencies, tool configs |
| `README.md`      | Repository highlights and quick start               |
| `LICENSE`        | Apache 2.0 license                                  |
| `CLAUDE.md`      | Per-repo guidance for Claude Code                   |
| `AGENTS.md`      | Per-repo guidance for AI coding agents              |

---

## Quality Metrics

### Code Quality Configuration

- **Ruff** - Formatting and linting (line length: 88)
- **mypy** - Strict type checking (`disallow_untyped_defs = true`)
- **pre-commit** - Hooks for ruff + mypy + fast tests on push

### Testing Framework

- **pytest** - Testing framework
- **pytest-cov** - Coverage reporting
- **pytest-xdist** - Parallel test execution
- **pytest-timeout** - Per-test timeout (15 min default)

**Test Categories** (opt-in buckets via marker flags):
- Unit and integration tests (fast, run by default)
- `slow` - slower tests (opt-in via `--run-slow`)
- `requires_gpu` - GPU/CUDA-dependent tests (opt-in via `--run-gpu`)
- `requires_simpleware` - tests needing a local Synopsys Simpleware Medical install (opt-in via `--run-simpleware`)
- `requires_physicsnemo` - tests needing the optional `[physicsnemo]` extra (opt-in via `--run-physicsnemo`)
- `experiment` - runs experiment scripts end-to-end (opt-in via `--run-experiments`; multi-hour)
- `tutorial` - runs tutorial scripts end-to-end (opt-in via `--run-tutorials`)

---

## Documentation Statistics

| Type                  | Count                   | Lines |
| ---------------------- | ------------------------ | ----- |
| **Markdown files**    | 30 (repo-wide READMEs, guides) | 3,780 |
| **reStructuredText**  | 73 files under `docs/`   | 7,064 |
| **Python docstrings** | All public modules       | embedded |
| **API map**           | Generated on demand via `py utils/generate_api_map.py` | n/a (not checked in) |

### Documentation Highlights

- Sphinx site (published to GitHub Pages) covering getting started,
  tutorials, CLI & scripts, API reference, developer guides, contributing,
  testing, FAQ, and troubleshooting
- Per-subpackage READMEs and `CLAUDE.md` files (e.g.
  `src/physiotwin4d/vtk_to_usd/CLAUDE.md`)
- Shared `.agents/` configuration: 4 role-specific subagents
  (`.agents/agents/`) and 9 slash-command skills (`.agents/skills/`) for
  Claude Code and other AI coding agents

---

## Summary

PhysioTwin4D is a beta-quality scientific toolkit for creating personalized
physiological digital twins: it extracts anatomic models from 3D medical
images and uses AI surrogates - together with statistical shape models for
subject-specific characterization and cross-subject correspondence - to
estimate a subject's physiological processes, currently cardiac and
respiratory motion. It is built on top of established medical imaging, AI/ML,
and 3D graphics libraries with a small, focused public API and a
percent-cell-script example/tutorial layout that runs both interactively and
unattended.

---

**Last Updated:** July 7, 2026
