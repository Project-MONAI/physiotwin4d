# PhysioMotion4D

**Generate anatomic models in Omniverse with physiological motion derived from 4D medical images.**

PhysioMotion4D is a comprehensive medical imaging package that converts 4D CT scans (particularly heart and lung gated CT data) into dynamic 3D models for visualization in NVIDIA Omniverse. The package provides state-of-the-art deep learning-based image processing, segmentation, registration, and USD file generation capabilities.

## Documentation

Start with the hosted documentation:

**https://project-monai.github.io/physiomotion4d/**

The documentation is the primary entry point for users and contributors. It
includes:

- **Tutorials**: runnable end-to-end workflows and their required datasets.
- **Getting Started**: installation, quickstart, examples, and architecture.
- **CLI & Scripts Guide**: command-line tools for common conversion,
  segmentation, registration, and USD workflows.
- **API Reference**: public workflow classes, registration classes,
  segmentation classes, USD tools, and utilities.
- **Developer Guides**: architecture, extension points, workflow design, and
  implementation conventions.
- **Contributing, Testing, and Troubleshooting**: project practices, validation
  commands, and common setup issues.

> **Not validated for clinical use.** PhysioMotion4D is a 2026.05.07 beta
> research and visualization toolkit. It is not a medical device and must not
> be used for diagnosis, treatment planning, or clinical decision-making.

## Key Features

- **Complete 4D Medical Imaging Pipeline**: End-to-end processing from 4D CT data to animated USD models
- **Multiple AI Segmentation Methods**: TotalSegmentator and Simpleware cardiac segmentation
- **Deep Learning Registration**: GPU-accelerated image registration using Icon algorithm
- **NVIDIA Omniverse Integration**: Direct USD file export for medical visualization
- **Physiological Motion Analysis**: Capture and visualize cardiac and respiratory motion
- **Flexible Workflow Control**: Step-based processing with checkpoint management

## Supported Applications

- **Cardiac Imaging**: Heart-gated CT processing with cardiac motion analysis
- **Pulmonary Imaging**: Lung 4D-CT processing with respiratory motion tracking
- **Medical Education**: Interactive 3D anatomical models with physiological motion
- **Research Visualization**: Advanced medical imaging research in Omniverse
- **Clinical Planning**: Dynamic anatomical models for treatment planning

## Installation

### Prerequisites

- Python 3.11+ (Python 3.11 or 3.12 recommended)
- NVIDIA GPU with CUDA 13 — recommended for production use; CPU-only installation is supported but slow
- 16GB+ RAM (32GB+ recommended for large datasets)
- NVIDIA Omniverse (for USD visualization)
- **Git LFS** (required for running tests: baseline files in `tests/baselines/` are stored with Git LFS; install from [git-lfs.github.com](https://git-lfs.github.com), then run `git lfs install` and `git lfs pull` after cloning)

### Installation from PyPI

```bash
# CPU-only PyPI install — works out of the box; a runtime warning points to the GPU extra
pip install physiomotion4d

# CUDA 13 install (recommended for production)
uv pip install "physiomotion4d[cuda13]"
```

The `[cuda13]` extra installs CuPy. In uv-managed source environments, PyTorch,
torchvision, and torchaudio resolve from the CUDA 13.0 PyTorch wheel index.
There is no need to install PyTorch separately.

### Installation from Source

1. **Clone the repository** (Git LFS is required for tests; install it first from [git-lfs.github.com](https://git-lfs.github.com)):
   ```bash
   git clone https://github.com/Project-MONAI/physiomotion4d.git
   cd physiomotion4d
   git lfs install   # if not already done
   git lfs pull     # fetch .hdf and .mha baselines in tests/baselines/
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install uv package manager** (recommended):
   ```bash
   pip install uv
   ```

4. **Install PhysioMotion4D**:
   ```bash
   # CUDA 13 PyTorch is the default for uv-managed source environments
   uv pip install -e "."

   # Add CuPy for CUDA 13 GPU acceleration
   uv pip install -e ".[cuda13]"
   ```

### Verify Installation

```python
import physiomotion4d
from physiomotion4d import WorkflowConvertImageToUSD

print(f"PhysioMotion4D version: {physiomotion4d.__version__}")
print(WorkflowConvertImageToUSD.__name__)
```

## Package Architecture

### Core Components

- **Workflow Classes**: Complete end-to-end pipeline processors
  - `WorkflowConvertImageToUSD`: 3D/4D image to USD processing workflow
  - `WorkflowCreateStatisticalModel`: Create PCA statistical shape model from sample meshes
  - `WorkflowFitStatisticalModelToPatient`: Model-to-patient registration workflow
- **Segmentation Classes**: Multiple AI-based chest segmentation implementations
  - `SegmentChestTotalSegmentator`: TotalSegmentator-based segmentation
  - `SegmentAnatomyBase`: Base class for custom segmentation methods
- **Registration Classes**: Multiple registration methods for different use cases
  - Image-to-Image Registration:
    - `RegisterImagesICON`: Deep learning-based registration using Icon algorithm
    - `RegisterImagesANTs`: Classical deformable registration using ANTs
    - `RegisterTimeSeriesImages`: Specialized time series registration for 4D CT
  - Model-to-Image/Model Registration:
    - `RegisterModelsPCA`: PCA-based statistical shape model registration
    - `RegisterModelsICP`: ICP-based surface registration
    - `RegisterModelsDistanceMaps`: Mask-based deformable model registration
  - `RegisterImagesBase`: Base class for custom registration methods
- **Base Classes**: Foundation classes providing common functionality
  - `PhysioMotion4DBase`: Base class providing standardized logging and debug settings
- **Utility Classes**: Tools for data manipulation and conversion
  - `TransformTools`: Comprehensive transform manipulation utilities
  - `USDTools`: USD file manipulation for Omniverse integration
  - `USDAnatomyTools`: Apply surgical materials to anatomy meshes
  - `ImageTools`: Medical image processing utilities
  - `ContourTools`: Mesh extraction and contour manipulation
- **USD Conversion**: VTK to USD conversion for Omniverse visualization
  - `ConvertVTKToUSD`: High-level converter for PyVista/VTK objects with colormap support
  - `vtk_to_usd` module: Advanced low-level file conversion library
    - `convert_vtk_file()`: Single-file VTK/VTP/VTU to USD facade
    - `read_vtk_file()`: Read VTK/VTP/VTU files into MeshData
    - `ConversionSettings`: Configurable conversion parameters
    - `MaterialData`: USD material definitions

### Key Dependencies

- **Medical Imaging**: ITK, TubeTK, MONAI, nibabel, PyVista
- **AI/ML**: PyTorch, CuPy (CUDA 13), transformers, MONAI
- **Registration**: icon-registration, unigradicon
- **Visualization**: USD-core, PyVista
- **Segmentation**: TotalSegmentator

## Getting Started: Tutorials

The `tutorials/` directory contains nine end-to-end Python scripts, one for each
major workflow. They are the recommended starting point for new users.

| # | Script | Workflow | Dataset |
|---|--------|----------|---------|
| 1 | `tutorials/tutorial_01_heart_gated_ct_to_usd.py` | Heart-gated CT to animated USD | Slicer-Heart-CT (prepare first) |
| 2 | `tutorials/tutorial_02_ct_to_vtk.py` | CT to VTK surfaces | Slicer-Heart-CT (prepare first) |
| 3 | `tutorials/tutorial_03_create_statistical_model.py` | Build PCA shape model | KCL-Heart-Model (manual) |
| 4 | `tutorials/tutorial_04_fit_statistical_model_to_patient.py` | Fit statistical model to patient | KCL-Heart-Model plus Tutorial 3 output |
| 5 | `tutorials/tutorial_05_vtk_to_usd.py` | VTK surfaces to animated USD | output of tutorial 2 |
| 6 | `tutorials/tutorial_06_reconstruct_highres_4d_ct.py` | Reconstruct high-res 4D CT | DirLab-4DCT (manual) |
| 7 | `tutorials/tutorial_07_dirlab_pca_model.py` | Build a surface PCA lung-lobe model and fit all cases | DirLab-4DCT (manual) |
| 8 | `tutorials/tutorial_08_dirlab_pca_time_series.py` | Propagate PCA-fitted lung-lobe meshes through DirLab time series | DirLab-4DCT plus Tutorial 7 output |
| 9 | `tutorials/tutorial_09_physicsnemo_mesh_stage_model.py` | Train a PhysicsNeMo mesh stage model | Tutorial 8 output |

Each tutorial is a `# %%` percent-cell Python script. Paths are defined near
the top of the script; edit those constants for custom data/output locations,
or use the installed `physiomotion4d-*` CLI commands when you want path
arguments.

```bash
# Tutorial 1 (CPU-safe ANTs registration; requires Slicer-Heart-CT data)
python tutorials/tutorial_01_heart_gated_ct_to_usd.py

# Tutorial 2 (CT to VTK)
python tutorials/tutorial_02_ct_to_vtk.py
```

See `tutorials/README.md` for the full tutorial index, dataset preparation
instructions, recommended run order, and tutorial-test instructions.

### Minimal Slicer-Heart Quickstart

This quickstart uses the public Slicer-Heart 4D CT sample. Data downloading and
a CUDA-capable GPU are required for practical runtime.

```bash
python -c "from physiomotion4d import DataDownloadTools; DataDownloadTools.DownloadSlicerHeartCTData('data/test')"

physiomotion4d-convert-image-to-usd data/test/TruncalValve_4DCT.seq.nrrd \
    --registration-method ANTS \
    --output-dir output/quickstart \
    --project-name slicer_heart_quickstart
```

## Quick Start

### Command-Line Interface

After installation, PhysioMotion4D provides command-line tools that are automatically added to your PATH:

#### Heart-Gated CT to USD

Process 4D cardiac CT images into dynamic USD models:

```bash
# Process a single 4D cardiac CT file
physiomotion4d-convert-image-to-usd cardiac_4d.nrrd --contrast --output-dir ./results

# Process multiple time frames
physiomotion4d-convert-image-to-usd frame_*.nrrd --contrast --project-name patient_001

# With custom settings
physiomotion4d-convert-image-to-usd cardiac.nrrd \
    --contrast \
    --reference-image ref.mha \
    --registration-iterations 50 \
    --output-dir ./output
```

For Python API usage and advanced customization, see the examples below or refer to the CLI implementation in `src/physiomotion4d/cli/`.

#### Create Statistical Model

Build a PCA statistical shape model from sample meshes aligned to a reference:

```bash
# From a directory of sample meshes
physiomotion4d-create-statistical-model \
    --sample-meshes-dir ./input_meshes \
    --reference-mesh average_mesh.vtk \
    --output-dir ./pca_output

# With custom PCA components
physiomotion4d-create-statistical-model \
    --sample-meshes-dir ./meshes \
    --reference-mesh average_mesh.vtk \
    --output-dir ./pca_output \
    --pca-components 20
```

Outputs: `pca_mean_surface.vtp`, `pca_mean.vtu` (if reference is volumetric), and `pca_model.json`.

#### Heart Model to Patient Registration

Register a generic heart model to patient-specific data:

```bash
# Basic registration
physiomotion4d-fit-statistical-model-to-patient \
    --template-model heart_model.vtu \
    --template-labelmap heart_labelmap.nii.gz \
    --patient-models lv.vtp rv.vtp myo.vtp \
    --patient-image patient_ct.nii.gz \
    --output-dir ./results

# With PCA shape fitting
physiomotion4d-fit-statistical-model-to-patient \
    --template-model heart_model.vtu \
    --template-labelmap heart_labelmap.nii.gz \
    --patient-models lv.vtp rv.vtp myo.vtp \
    --patient-image patient_ct.nii.gz \
    --pca-json pca_model.json \
    --pca-number-of-modes 10 \
    --output-dir ./results
```

For implementation details and advanced usage, see the CLI modules in `src/physiomotion4d/cli/`.

### Python API - Basic Heart-Gated CT Processing

```python
from physiomotion4d import WorkflowConvertImageToUSD

# Initialize processor
processor = WorkflowConvertImageToUSD(
    input_filenames=["path/to/cardiac_4d_ct.nrrd"],
    contrast_enhanced=True,
    output_directory="./results",
    project_name="cardiac_model",
    registration_method='ICON'  # or 'ANTS'
)

# Run complete workflow
final_usd = processor.process()
```

### Python API - Model to Patient Registration

```python
from physiomotion4d import WorkflowFitStatisticalModelToPatient
import pyvista as pv
import itk

# Load generic model and patient data
model_mesh = pv.read("generic_heart_model.vtu")
patient_surfaces = [pv.read("lv.stl"), pv.read("rv.stl")]
reference_image = itk.imread("patient_ct.nii.gz")

# Initialize and run workflow
workflow = WorkflowFitStatisticalModelToPatient(
    template_model=model_mesh,
    patient_models=patient_surfaces,
    patient_image=reference_image,
)

# Run the registration pipeline
result = workflow.run_workflow()
registered_mesh = result["registered_template_model_surface"]
```

### Custom Segmentation

```python
from physiomotion4d import SegmentChestTotalSegmentator
import itk

# Initialize TotalSegmentator segmentation
segmenter = SegmentChestTotalSegmentator()

# Load and segment image
image = itk.imread("chest_ct.nrrd")
masks = segmenter.segment(image, contrast_enhanced_study=True)

# Result always contains "labelmap" plus one entry per anatomy group the
# segmenter registered (heart, lung, bone, major_vessels, soft_tissue,
# contrast, other for SegmentChestTotalSegmentator). The exact key set is
# segmenter-specific; check membership when targeting multiple segmenters.
labelmap = masks["labelmap"]
heart_mask = masks["heart"]
if "lung" in masks:
    lungs_mask = masks["lung"]
```

### Image Registration

```python
from physiomotion4d import RegisterImagesICON, RegisterImagesANTs, RegisterTimeSeriesImages
import itk

# Option 1: Icon deep learning registration (GPU-accelerated)
registerer = RegisterImagesICON()
registerer.set_modality('ct')
registerer.set_fixed_image(itk.imread("reference_frame.mha"))
results = registerer.register(itk.imread("target_frame.mha"))

# Option 2: ANTs classical registration
registerer = RegisterImagesANTs()
registerer.set_fixed_image(itk.imread("reference_frame.mha"))
results = registerer.register(itk.imread("target_frame.mha"))

# Option 3: Time series registration for 4D CT
time_series_reg = RegisterTimeSeriesImages(
    reference_index=0,
    registration_method='icon'  # or 'ants'
)
transforms = time_series_reg.register_time_series(
    image_filenames=["time00.mha", "time01.mha", "time02.mha"]
)

# Get forward and inverse displacement fields
inverse_transform = results["inverse_transform"]  # Fixed to moving
forward_transform = results["forward_transform"]  # Moving to fixed
```

### VTK to USD Conversion

PhysioMotion4D provides two APIs for converting VTK data to USD for NVIDIA Omniverse visualization. Repository workflows, experiments, and CLIs use `ConvertVTKToUSD`; `vtk_to_usd` is a public advanced layer for users who need low-level file conversion primitives.

#### Option 1: High-Level ConvertVTKToUSD (for PyVista/VTK objects)

```python
from physiomotion4d import ConvertVTKToUSD, SegmentChestTotalSegmentator
import pyvista as pv

# Load VTK data
meshes = [pv.read(f"cardiac_frame_{i:03d}.vtp") for i in range(20)]

# Convert to animated USD with anatomical labels. Pass `segmenter` so the
# converter groups labeled prims by anatomy type:
#   /World/CardiacModel/heart/<organ>, /World/CardiacModel/lung/<organ>, ...
# Without `segmenter`, all labeled prims land under /World/CardiacModel/Anatomy.
seg = SegmentChestTotalSegmentator()
converter = ConvertVTKToUSD(
    data_basename='CardiacModel',
    input_polydata=meshes,
    mask_ids=seg.taxonomy.all_labels(),
    segmenter=seg,
    compute_normals=True
)

# Optional: Apply colormap visualization
converter.set_colormap(
    color_by_array='transmembrane_potential',
    colormap='rainbow',
    intensity_range=(-80.0, 20.0)
)

stage = converter.convert('cardiac_motion.usd')
```

#### Option 2: Advanced File-Based vtk_to_usd Facade

```python
from physiomotion4d.vtk_to_usd import (
    ConversionSettings,
    MaterialData,
    convert_vtk_file,
)

# Simple single-file conversion
stage = convert_vtk_file('mesh.vtp', 'output.usd')

# Advanced: custom settings and material
settings = ConversionSettings(
    triangulate_meshes=True,
    compute_normals=True,
    meters_per_unit=1.0,  # USD stage units after built-in mm-to-m scaling
    times_per_second=60.0,
)

material = MaterialData(
    name="cardiac_tissue",
    diffuse_color=(0.9, 0.3, 0.3),
    roughness=0.4,
)

stage = convert_vtk_file(
    'heart.vtp',
    'heart.usd',
    data_basename='Heart',
    settings=settings,
    material=material,
)
```

Features:
- Automatic coordinate system conversion (LPS to USD right-handed Y-up)
- Material system with UsdPreviewSurface
- Preserves all VTK data arrays as USD primvars
- Supports VTP, VTK, and VTU file formats

### Logging and Debug Control

PhysioMotion4D provides standardized logging through the `PhysioMotion4DBase` class, which is inherited by workflow and registration classes.

```python
import logging
from physiomotion4d import WorkflowFitStatisticalModelToPatient, PhysioMotion4DBase

# Control logging level globally for all classes
PhysioMotion4DBase.set_log_level(logging.DEBUG)

# Or filter to show logs from specific classes only
PhysioMotion4DBase.set_log_classes(["WorkflowFitStatisticalModelToPatient", "RegisterModelsPCA"])

# Show all classes again
PhysioMotion4DBase.set_log_all_classes()

# Query which classes are currently filtered
filtered = PhysioMotion4DBase.get_log_classes()
```

Classes that inherit from `PhysioMotion4DBase` provide:
- Standard log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
- Progress reporting for long-running operations
- Class-based log filtering
- Unified logging interface across the package

## Experiments and Examples

The `experiments/` directory contains research scripts that shaped the
toolkit. They are `# %%` percent-cell Python scripts that can be run
top-to-bottom (`python <script>.py`) or stepped through cell-by-cell in VS
Code, Cursor, or any other editor with `# %%` cell-aware support. For the
curated, supported user-facing entry points see `tutorials/` and
`docs/tutorials.rst`.

### Heart-Gated CT (`experiments/Heart-GatedCT_To_USD/`)

Complete cardiac imaging workflow with step-by-step scripts:

- **`0-download_and_convert_4d_to_3d.py`**: Data preparation and 4D to 3D conversion
- **`1-register_images.py`**: Image registration between cardiac phases
- **`2-generate_segmentation.py`**: AI-based cardiac segmentation
- **`3-transform_dynamic_and_static_contours.py`**: Dynamic contour transformation
- **`4-merge_dynamic_and_static_usd.py`**: Final USD model creation and merging

**Sample Data**: The scripts include instructions for downloading cardiac CT datasets from Slicer-Heart-CT.

### Lung-Gated CT (`experiments/Lung-GatedCT_To_USD/`)

Respiratory motion analysis using DirLab 4D-CT benchmark data:

- **`0-register_dirlab_4dct.py`**: Registration of respiratory phases
- **`1-make_dirlab_models.py`**: 3D model generation from lung segmentation
- **`2-paint_dirlab_models.py`**: USD material and visualization enhancement

**Sample Data**: Uses the standard DirLab 4D-CT benchmark datasets. DirLab data
must be downloaded manually and placed under `data/DirLab-4DCT/`; see
`data/README.md` for the expected layout.

### Colormap Visualization (`experiments/Colormap-VTK_To_USD/`)

Time-varying colormap rendering for scalar data visualization in Omniverse:

- **`colormap_vtk_to_usd.py`**: Convert VTK meshes with scalar data to USD with colormaps
- Demonstrates plasma, viridis, rainbow, heat, coolwarm, grayscale, and custom colormaps

### Heart VTK Series (`experiments/Heart-VTKSeries_To_USD/`)

Direct VTK time series to USD conversion for cardiac data:

- **`0-download_and_convert_4d_to_3d.py`**: Data preparation
- **`1-heart_vtkseries_to_usd.py`**: VTK series to USD conversion

### Heart Create Statistical Model (`experiments/Heart-Create_Statistical_Model/`)

Create PCA statistical shape models from population meshes using the KCL Heart Model dataset:

- **`1-input_meshes_to_input_surfaces.py`**: Convert meshes to surfaces
- **`2-input_surfaces_to_surfaces_aligned.py`**: Align population meshes
- **`3-registration_based_correspondence.py`**: Compute point correspondences
- **`4-surfaces_aligned_correspond_to_pca_inputs.py`**: Prepare PCA inputs
- **`5-compute_pca_model.py`**: Compute PCA model using sklearn

**Complete this experiment FIRST** before attempting `Heart-Statistical_Model_To_Patient`.

### Heart Statistical Model to Patient (`experiments/Heart-Statistical_Model_To_Patient/`)

Advanced registration between generic anatomical models and patient-specific data using PCA:

- **`heart_model_to_model_icp_itk.py`**: ICP registration for initial alignment
- **`heart_model_to_model_registration_pca.py`**: PCA-based statistical shape model registration
- **`heart_model_to_patient.py`**: Complete model-to-patient registration workflow

Uses the `WorkflowFitStatisticalModelToPatient` class for three-stage registration:
1. ICP-based rough alignment
2. Mask-to-mask deformable registration
3. Optional PCA-constrained shape fitting

### 4D CT Reconstruction (`experiments/Reconstruct4DCT/`)

Reconstruct 4D CT from sparse time samples using deformable registration:

- **`reconstruct_4d_ct.py`**: Temporal interpolation and 4D reconstruction
- **`reconstruct_4d_ct_class.py`**: Class-based reconstruction approach

### Vessel and Airway Segmentation (`experiments/Lung-VesselsAirways/`)

Specialized deep learning for pulmonary vessel and airway segmentation:

- **`0-GenData.py`**: Training data generation for vessel segmentation models
- Includes trained ResNet18 models for vessel segmentation
- Supporting branch structure test data

### Displacement Field Visualization (`experiments/DisplacementField_To_USD/`)

Convert image registration displacement fields to USD for advanced visualization:

- **`displacement_field_to_usd.py`**: Convert displacement fields to time-varying USD
- **`displacement_field_converter.py`**: DisplacementFieldToUSD class implementation
- Integration with PhysicsNeMo for flow visualization in Omniverse
- Supports streamlines, vector glyphs, and particle advection

## Sample Data Sources

### Cardiac Data
- **Slicer-Heart-CT**: Cardiac gating examples from 3D Slicer
- **Duke CardiacCT**: Research cardiac datasets (requires institutional access)

### Lung Data
- **DirLab 4D-CT**: Public benchmark for respiratory motion
  - Manual download required; see `data/README.md`
  - 10 cases with respiratory motion and landmark validation

### Download Example

The Slicer-Heart sample can be downloaded directly from its public GitHub
release:

```python
from physiomotion4d import DataDownloadTools

data_file = DataDownloadTools.DownloadSlicerHeartCTData("data/Slicer-Heart-CT")
assert DataDownloadTools.VerifySlicerHeartCTData("data/Slicer-Heart-CT")
```

## Development

### Code Quality Tools

PhysioMotion4D uses modern, fast tooling for code quality:

- **Ruff**: Fast linting and formatting (replaces black, isort, flake8, pylint)
- **mypy**: Type checking
- **Pre-commit hooks**: Automatic code quality checks on commit

### Running Quality Checks

```bash
# Check and fix linting issues
ruff check . --fix

# Format code
ruff format .

# Run type checking
mypy src/

# Run all pre-commit hooks
pre-commit run --all-files
```

### IDE Setup

For VS Code or Cursor, install the Ruff extension:
- `charliermarsh.ruff` - Automatic formatting and linting on save

See `docs/contributing.rst` for complete IDE setup instructions.

### Testing

PhysioMotion4D includes comprehensive tests covering the complete pipeline from data download to USD generation.

```bash
# Run all tests
pytest tests/

# Run fast tests only (recommended for development)
pytest tests/ -m "not slow and not requires_data" -v

# Run specific test categories
pytest tests/test_usd_merge.py -v                           # USD merge functionality
pytest tests/test_usd_time_preservation.py -v               # Time-varying data preservation
pytest tests/test_register_images_ants.py -v                # ANTs registration
pytest tests/test_register_images_greedy.py -v             # Greedy registration
pytest tests/test_register_images_icon.py -v                # Icon registration
pytest tests/test_register_time_series_images.py -v         # Time series registration
pytest tests/test_segment_chest_total_segmentator.py -v     # TotalSegmentator
pytest tests/test_contour_tools.py -v                       # Mesh and contour tools
pytest tests/test_image_tools.py -v                         # Image processing utilities
pytest tests/test_transform_tools.py -v                     # Transform operations

# Skip GPU-dependent tests (segmentation and registration)
pytest tests/ --ignore=tests/test_segment_chest_total_segmentator.py \
              --ignore=tests/test_register_images_icon.py

# Run with coverage report
pytest tests/ --cov=src/physiomotion4d --cov-report=html
```

**Test Categories:**
- **Data Pipeline**: Download, conversion, and preprocessing
- **Segmentation**: TotalSegmentator (GPU required)
- **Registration**: ANTs, Icon, and time series methods (slow, ~5-10 min)
- **Geometry & Visualization**: Contour tools, transform tools, VTK to USD
- **USD Utilities**: Merging, time preservation, material handling

Tests automatically run on pull requests via GitHub Actions. See `tests/README.md` for detailed testing guide.

### Developer Tool Prerequisites

| Tool | Required for | Install |
|------|-------------|---------|
| Codex CLI | Default agent for `.agents/` skills and `ai_agent_github_reviews.py` | Install from the current Codex CLI distribution |
| [Claude Code](https://claude.ai/code) | Optional agent for `.agents/` skills and `ai_agent_github_reviews.py --agent claude` | `winget install Anthropic.ClaudeCode` |
| [gh CLI](https://cli.github.com) | `ai_agent_github_reviews.py` | `winget install GitHub.cli`, then `gh auth login` |

### AI-Assisted Development

The repository includes a shared `.agents/` configuration for Codex, Claude Code,
and other AI coding agents. It provides always-on project guidance, four
specialized subagents, and slash-command skills tailored to this codebase.

#### Configuration files

| Path | Purpose |
|------|---------|
| `AGENTS.md` | Shared rules for implementation, testing, docs, and architecture work |
| `CLAUDE.md` | Claude-specific always-on guidance and slash-command usage |
| `.agents/agents/implementation.md` | Subagent: reads source, plans, implements in small diffs |
| `.agents/agents/testing.md` | Subagent: writes synthetic-data pytest tests |
| `.agents/agents/docs.md` | Subagent: updates docstrings and regenerates `docs/API_MAP.md` |
| `.agents/agents/architecture.md` | Subagent: design plans and trade-off analysis (no code written) |
| `.agents/skills/plan/SKILL.md` | `/plan` — inspect and plan before coding |
| `.agents/skills/impl/SKILL.md` | `/impl` — implement a feature or fix |
| `.agents/skills/test-feature/SKILL.md` | `/test-feature` — write tests for a module |
| `.agents/skills/doc-feature/SKILL.md` | `/doc-feature` — update docstrings and API map |

#### Common contributor workflows

**Planning a new feature before writing code**

Use `/plan` to get an inspection of the affected classes, a numbered implementation
plan, and a list of open questions — without touching any files.

```text
/plan add a new segmentation method to SegmentChestTotalSegmentator
```

The agent will read the relevant source, summarize current behavior, list files that
will change, and flag any coordinate-system or shape implications.

---

**Implementing a feature or bug fix**

Use `/impl` for end-to-end implementation: read → summarize → plan → diff → lint.

```text
/impl add set_regularization_weight() to RegisterImagesANTs
```

```text
/impl fix the LPS-to-Y-up transform being applied twice in vtk_to_usd/usd_utils.py
```

The agent will read the affected module, propose a numbered plan, implement in the
smallest reviewable diff, update docstrings, run `ruff`, and call out breaking changes.

---

**Writing tests for a new or changed module**

Use `/test-feature` to get a test plan and a complete pytest file using synthetic
`itk.Image` or `pv.PolyData` objects — no real patient data required.

```text
/test-feature ContourTools.extract_surface — test with a synthetic 32x32x32 sphere mask
```

```text
/test-feature RegisterImagesANTs with a pair of small synthetic ITK images
```

The agent will state image shapes and axis orders in every test docstring, mark
any real-data dependency with `@pytest.mark.requires_data`, and show the exact
run command.

---

**Updating documentation after a change**

Use `/doc-feature` after modifying a public API to refresh docstrings and regenerate
the API map.

```text
/doc-feature update docstrings for RegisterImagesANTs after adding set_regularization_weight
```

The agent will update affected docstrings in NumPy style, add shape/axis annotations
where arrays are involved, and run `py utils/generate_api_map.py`.

---

**Applying PR review suggestions (CodeRabbit / Copilot)**

Use `ai_agent_github_reviews.py` to fetch all review comments for a PR, have
Codex by default screen each one against `AGENTS.md`, apply accepted edits as
pending changes, and write a Markdown summary to the repo root:

```bash
py utils/ai_agent_github_reviews.py --pr 42
py utils/ai_agent_github_reviews.py --pr 42 --agent claude
py utils/ai_agent_github_reviews.py --pr 42 --dry-run   # preview prompt only
py utils/ai_agent_github_reviews.py --pr 42 --since-last-push --dry-run
```

When executing these from the repo root (including in automation), use the project
interpreter: `venv/Scripts/python` on Windows instead of `py`.

The selected agent decides APPLY / REVISE / REJECT for each suggestion, with reasoning.
No changes are committed — review with `git diff`, then `git add -p`.

---

**Setting up an isolated feature branch**

Use the `setup_feature_worktree.py` utility to create a git worktree with its own
venv in one command (Windows):

```bash
py utils/setup_feature_worktree.py my-feature
py utils/setup_feature_worktree.py my-feature --base-branch main
```

This creates a `feature/my-feature` branch, a sibling worktree directory, installs
`uv`, and installs project dependencies — ready to open in a separate editor window.

---

**Architectural planning before a structural change**

For larger changes, describe the goal to the architecture subagent and ask for a
design plan. The agent will produce the six-section format (current state → proposed
change → affected files → trade-offs → open questions → recommended next action)
without writing any code.

```text
/plan redesign the segmentation return type to use a dataclass instead of a tuple
```

## Additional Documentation

The canonical documentation is published at
https://project-monai.github.io/physiomotion4d/.

Use it for tutorials, getting started, CLI usage, API reference, developer
guides, contributing, testing, and troubleshooting. The `experiments/`
directory records prior and ongoing experiments used to shape the toolkit; it
is not the user-facing examples collection.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run code quality checks (`ruff check . --fix && ruff format . && mypy src/`)
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

See `docs/contributing.rst` for detailed contribution guidelines and IDE setup.

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## Acknowledgments

- **NVIDIA Omniverse** team for USD format and visualization platform
- **MONAI** community for medical imaging AI tools
- **DirLab** for providing the 4D-CT benchmark datasets
- **TotalSegmentator** team for segmentation models
- **Icon Registration** team for deep learning registration methods

## Support

- **Issues**: Report bugs and feature requests via GitHub Issues
- **Discussions**: Join community discussions in GitHub Discussions
- **Documentation**: Refer to docstrings and tutorial scripts under `tutorials/`
- **Examples**: Explore comprehensive examples in `experiments/` directory

---

**Get started with the tutorial scripts under `tutorials/` to see PhysioMotion4D in action.**
