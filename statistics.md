# PhysioTwin4D - Software Development Statistics

**Report Generated:** August 5, 2026
**Project Version:** 2026.07.3
**Status:** Beta (Development Status: 4 - Beta)

---

## Executive Summary

PhysioTwin4D is a collection of methods, workflows, tutorials, and CLI tools
for creating personalized physiological digital twins from 3D medical images.
This report summarizes development effort, code quality, and project maturity.

### Key Metrics at a Glance

| Metric                         | Value                                          |
| ------------------------------ | ---------------------------------------------- |
| **Total Lines of Code**        | ~53,400                                        |
| **Development Period**         | December 5, 2025 - August 5, 2026 (~8 months)  |
| **Total Commits**              | 114                                            |
| **Primary Developer**          | 1 (Stephen Aylward), plus 1 outside contributor |

---

## Detailed Code Statistics

### Lines of Code Breakdown

| Category                                | Files          | Lines of Code | Percentage |
| ---------------------------------------- | -------------- | -------------- | ---------- |
| **Core Python Source (`src/`)**          | 73 files       | 22,663         | 42.4%      |
| **Test Suite (`tests/`)**                | 36 files       | 8,225          | 15.4%      |
| **Experiment Scripts (`experiments/`)**  | 46 files       | 7,897          | 14.8%      |
| **Tutorial Scripts (`tutorials/`)**      | 20 files       | 4,153          | 7.8%       |
| **Utility Scripts (`utils/`)**           | 3 files        | 1,460          | 2.7%       |
| **Documentation (`docs/*.rst`)**         | 85 files       | 6,069          | 11.4%      |
| **Markdown (repo-wide READMEs, guides)** | 35 files       | 2,958          | 5.5%       |
| **TOTAL**                                | **298 files**  | **~53,400**    | **100%**   |

All experiment and tutorial sources are plain `.py` files run with
`python <script>.py`. Experiment scripts additionally carry `# %%` percent-cell
markers, so they can be stepped through cell-by-cell in VS Code / Cursor;
tutorials are straightforward top-to-bottom scripts.

### Core Module Highlights (Python Source)

| Module                                          | Lines | Purpose                                         |
| ------------------------------------------------ | ----- | ----------------------------------------------- |
| `usd_tools.py`                                   | 1,290 | USD file manipulation and inspection            |
| `usd_anatomy_tools.py`                           | 986   | OmniSurface materials for labeled anatomy       |
| `convert_vtk_to_usd.py`                          | 893   | High-level VTK -> USD converter                 |
| `workflow_fit_statistical_model_to_patient.py`   | 826   | Model-to-patient registration workflow          |
| `transform_tools.py`                             | 724   | ITK transform utilities                         |
| `register_models_pca.py`                         | 721   | PCA-based shape model registration              |
| `register_images_ants.py`                        | 645   | ANTs-based image registration                   |
| `segment_nv_segment_ct_mri.py`                   | 634   | NVIDIA CT/MRI segmentation bundle bridge        |
| `vtk_to_usd/` subpackage                         | 2,193 | Low-level VTK -> USD building blocks (9 files)  |
| `cli/` subpackage                                | 2,129 | CLI entry-point scripts (11 commands, 13 files) |
| `register_images_greedy.py`                      | 556   | Greedy classical deformable registration        |
| `register_time_series_images.py`                 | 511   | Time series registration for 4D CT              |
| `contour_tools.py`                               | 502   | Mesh extraction and contour manipulation        |

---

## Project Maturity Indicators

| Indicator                  | Status                                              |
| --------------------------- | ---------------------------------------------------- |
| **Documentation Coverage**  | Sphinx site + per-package READMEs                    |
| **Test Suite Present**      | Yes (`tests/` with baselines via Git LFS)             |
| **CI/CD Pipeline**          | GitHub Actions (Ubuntu + Windows; Python 3.11/3.12), plus a self-hosted Windows GPU runner |
| **Dependency Management**   | `pyproject.toml`, `uv`-friendly                       |
| **Code Quality Tools**      | Ruff (lint + format), mypy                            |
| **Example Scripts**         | 46 experiment scripts + 15 tutorial scripts           |
| **Version Management**      | Calendar versioning via bumpver                       |
| **API Reference**           | Google-style docstrings + Sphinx API docs under `docs/api/` |
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
- Public API surface documented via Sphinx API docs under `docs/api/`
- 23 required external dependencies (medical imaging, AI/ML, USD, registration),
  plus optional extras for CUDA, PhysicsNeMo, docs and development

---

## Dependencies & Infrastructure

### Core Dependencies (selected)

| Category              | Key Packages                                        |
| ---------------------- | ----------------------------------------------------- |
| **Medical Imaging**    | ITK, MONAI, nibabel, pydicom, pynrrd                 |
| **Deep Learning**      | PyTorch, CuPy (CUDA 13), transformers                |
| **Registration**       | ANTs (antspyx), picsl-greedy, icon-registration, UniGradICON |
| **3D Graphics / USD**  | VTK, PyVista, USD-core, trimesh                       |
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
- `tutorial` - runs tutorial scripts end-to-end
  (opt-in via `--run-tutorials`; multi-hour)

---

## Documentation Statistics

| Type                  | Count                   | Lines |
| ---------------------- | ------------------------ | ----- |
| **Markdown files**    | 35 (repo-wide READMEs, guides) | 2,958 |
| **reStructuredText**  | 85 files under `docs/`   | 6,069 |
| **Python docstrings** | All public modules       | embedded |
| **Knowledge graph**   | `graphify-out/`, refreshed via `graphify update .` | n/a (not checked in) |

### Documentation Highlights

- Sphinx site (published to GitHub Pages) covering getting started,
  tutorials, CLI & scripts, API reference, developer guides, contributing,
  testing, FAQ, and troubleshooting
- Per-subpackage READMEs and `CLAUDE.md` files (e.g.
  `src/physiotwin4d/vtk_to_usd/CLAUDE.md`)
- Shared `.agents/` configuration: 4 role-specific subagents
  (`.agents/agents/`) and 8 slash-command skills (`.agents/skills/`) for
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
plain-Python-script example/tutorial layout that runs both interactively and
unattended.

---

**Last Updated:** August 5, 2026
