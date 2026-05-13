# PhysioMotion4D - Software Development Statistics

**Report Generated:** May 13, 2026
**Project Version:** 2026.05.07
**Status:** Beta (Development Status: 4 - Beta)

---

## Executive Summary

PhysioMotion4D is a medical imaging package for generating anatomic models in
NVIDIA Omniverse with physiological motion derived from 4D medical images.
This report summarizes development effort, code quality, and project maturity.

### Key Metrics at a Glance

| Metric                         | Value                                          |
| ------------------------------ | ---------------------------------------------- |
| **Total Lines of Code**        | ~42,000                                        |
| **Development Period**         | December 5, 2025 - May 13, 2026 (~5 months)    |
| **Total Commits**              | 63                                             |
| **Primary Developer**          | 1 (Stephen Aylward)                            |

---

## Detailed Code Statistics

### Lines of Code Breakdown

| Category                             | Files     | Lines of Code | Percentage |
| ------------------------------------ | --------- | ------------- | ---------- |
| **Core Python Source (`src/`)**      | 49 files  | 19,670        | 47.3%      |
| **Test Suite (`tests/`)**            | 24 files  | 7,669         | 18.4%      |
| **Experiment Scripts (`experiments/`)** | 35 files | 6,687        | 16.1%      |
| **Tutorial Scripts (`tutorials/`)**  | 10 files  | 1,690         | 4.1%       |
| **Documentation (`docs/` + repo READMEs)** | ~90 files | ~6,300 rst + ~6,700 md (~13,000 total) | 31%   |
| **TOTAL**                            | **~210 files** | **~42,000** | **100%**  |

All experiment and tutorial sources are plain `.py` files. Each uses `# %%`
percent-cell markers so the same file can be executed end-to-end with
`python <script>.py` or stepped through cell-by-cell in VS Code / Cursor.

### Core Module Highlights (Python Source)

| Module                                          | Approx Lines | Purpose                                        |
| ----------------------------------------------- | ------------ | ---------------------------------------------- |
| `usd_tools.py`                                  | ~1,500       | USD file manipulation and inspection           |
| `transform_tools.py`                            | ~1,100       | ITK transform utilities                        |
| `register_models_pca.py`                        | ~820         | PCA-based shape model registration             |
| `workflow_fit_statistical_model_to_patient.py`  | ~740         | Model-to-patient registration workflow         |
| `register_images_ants.py`                       | ~720         | ANTs-based image registration                  |
| `segment_anatomy_base.py`                       | ~670         | Base class for anatomy segmentation            |
| `convert_vtk_to_usd.py`                         | ~800         | High-level VTK -> USD converter                |
| `vtk_to_usd/` subpackage                        | 2,657        | Low-level VTK -> USD building blocks (9 files) |
| `cli/` subpackage                               | 1,788        | CLI entry-point scripts (8 commands)           |
| `workflow_convert_heart_gated_ct_to_usd.py`     | ~540         | Heart CT to USD workflow                       |

---

## Project Maturity Indicators

| Indicator                  | Status                                 |
| -------------------------- | -------------------------------------- |
| **Documentation Coverage** | Sphinx site + per-package READMEs      |
| **Test Suite Present**     | Yes (`tests/` with baselines via Git LFS) |
| **CI/CD Pipeline**         | GitHub Actions (Ubuntu + Windows; Python 3.11/3.12) |
| **Dependency Management**  | `pyproject.toml`, `uv`-friendly        |
| **Code Quality Tools**     | Ruff (lint + format), mypy             |
| **Example Scripts**        | 35 experiment scripts + 10 tutorial scripts |
| **Version Management**     | Calendar versioning via bumpver        |
| **API Reference**          | Google-style docstrings + `docs/API_MAP.md` |
| **Package Distribution**   | PyPI-ready                             |

---

## Technical Complexity Assessment

### Domain Complexity

PhysioMotion4D operates across several technically demanding domains:

| Domain                   | Complexity Level | Key Technologies                       |
| ------------------------ | ---------------- | -------------------------------------- |
| **Medical Imaging**      | Very High        | ITK, MONAI, nibabel, pynrrd            |
| **Deep Learning**        | High             | PyTorch, CUDA 13, transformers         |
| **3D Graphics / USD**    | High             | VTK, PyVista, OpenUSD                  |
| **Image Registration**   | Very High        | ANTs, Icon, UniGradICON                |
| **AI Segmentation**      | High             | TotalSegmentator, Simpleware bridge    |
| **Geometric Processing** | High             | ICP, PCA, distance maps                |

### Architectural Sophistication

- Class hierarchy depth: 3-4 levels (well-structured inheritance from
  `PhysioMotion4DBase`)
- Module coupling: medium (clear separation between segmentation,
  registration, USD conversion, and workflow layers)
- Public API surface documented in `docs/API_MAP.md`
- ~25 major external dependencies (medical imaging, AI/ML, USD, registration)

---

## Dependencies & Infrastructure

### Core Dependencies (selected)

| Category              | Key Packages                                    |
| --------------------- | ----------------------------------------------- |
| **Medical Imaging**   | ITK, TubeTK, MONAI, nibabel, pynrrd             |
| **Deep Learning**     | PyTorch, CuPy (CUDA 13), transformers           |
| **Registration**      | ANTs, icon-registration, UniGradICON            |
| **3D Graphics / USD** | VTK, PyVista, USD-core                          |
| **AI Segmentation**   | TotalSegmentator                                |
| **Development Tools** | pytest, pytest-xdist, ruff, mypy, sphinx, uv    |

### Infrastructure Files

| File             | Purpose                                             |
| ---------------- | --------------------------------------------------- |
| `pyproject.toml` | Modern Python packaging, dependencies, tool configs |
| `README.md`      | Repository overview and quick start                 |
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

**Test Categories:**
- Unit tests (fast, isolated)
- Integration tests (slower, multi-component)
- GPU-dependent tests (segmentation, registration)
- Data-dependent tests (requires external downloads)
- Tutorial tests (run each tutorial script end-to-end and compare screenshots)
- Experiment tests (opt-in via `--run-experiments`; multi-hour run times)

---

## Documentation Statistics

| Type                  | Approx Count       | Approx Lines |
| --------------------- | ------------------ | ------------ |
| **Markdown files**    | ~50 (incl. READMEs across `experiments/`, `tests/`, etc.) | ~7,800 |
| **reStructuredText**  | 71 files under `docs/` | ~6,300 |
| **Python docstrings** | All public modules | embedded     |
| **API map**           | 1 generated file (`docs/API_MAP.md`) | ~1,200 |

### Documentation Highlights

- Quickstart, tutorials, examples, and architecture under `docs/`
- Per-subpackage READMEs (e.g. `src/physiomotion4d/vtk_to_usd/CLAUDE.md`)
- Contribution and testing guides
- FAQ and troubleshooting sections in test docs

---

## Summary

PhysioMotion4D is a beta-quality medical imaging toolkit that bridges 4D CT
data, AI segmentation and registration, VTK geometry, and OpenUSD output for
NVIDIA Omniverse. It is built on top of established medical imaging and 3D
graphics libraries with a small, focused public API and a percent-cell-script
example/tutorial layout that runs both interactively and unattended.

---

**Last Updated:** May 13, 2026
