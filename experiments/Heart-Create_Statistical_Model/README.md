# Heart-Create_Statistical_Model Experiment

This experiment demonstrates how to create a Principal Component Analysis (PCA) statistical shape model of a heart from a population of meshes using the KCL Heart Model dataset.

**⚠️ IMPORTANT:** Users should complete this experiment BEFORE attempting the `Heart-Statistical_Model_To_Patient` experiment, as it generates the PCA model data required for patient-specific registration.

## Overview

Statistical shape models (SSMs) capture the natural variation in anatomy across a population. By using PCA on a set of aligned heart meshes, we can:
- Identify the principal modes of shape variation
- Generate new plausible heart shapes by varying PCA coefficients
- Register these models to patient-specific data for personalized digital twins

This experiment uses the **King's College London (KCL) four-chamber heart model dataset**, which provides 20 four-chamber heart models derived from CT images.

## Dataset

The experiment requires the **KCL-Heart-Model** dataset located in `data/KCL-Heart-Model/`.

### ⚠️ Manual Download Required

The KCL dataset is NOT automatically downloaded. You must manually download it from:

**🔗 [Virtual cohort of adult healthy four-chamber heart meshes from CT images](https://zenodo.org/records/4590294)**

See `data/KCL-Heart-Model/README.md` for complete download and setup instructions.

### Required Files

- 20 tetrahedral heart meshes (`.vtk` format)
- Mesh labels (chamber, valve, vessel identifiers)
- Optional: existing PCA results from the dataset authors

## Pipeline Overview

This experiment follows a fully automated multi-step process. Each step is a
`# %%` percent-cell Python script that can be run top-to-bottom with
`python <script>.py` or stepped through cell-by-cell in VS Code / Cursor.

### Automated Steps

1. **`1-input_meshes_to_input_surfaces.py`**
   - Converts volumetric tetrahedral meshes to surface representations
   - Extracts relevant anatomical surfaces for shape analysis
   - Prepares data for alignment

2. **`2-input_surfaces_to_surfaces_aligned.py`**
   - Performs initial rigid alignment of surfaces using ICP
   - Centers and orients meshes for consistent positioning
   - Computes and saves the average surface from aligned meshes
   - Prepares aligned data for correspondence computation

3. **`3-registration_based_correspondence.py`**
   - Establishes point correspondences across the population using ANTs SyN deformable registration
   - Uses mask-based distance map registration via `RegisterModelsDistanceMaps`
   - Performs diffeomorphic (smooth, invertible) deformation to the average surface
   - Critical step for meaningful PCA analysis

4. **`4-surfaces_aligned_correspond_to_pca_inputs.py`**
   - Converts corresponded surfaces to PCA input format
   - Prepares data matrices for statistical analysis
   - Validates correspondence quality

5. **`5-compute_pca_model.py`**
   - Computes PCA on the corresponded point sets using sklearn
   - Generates eigenvectors, eigenvalues, and explained variance
   - Exports complete PCA model to `pca_model.json`
   - Produces statistical shape model outputs

## Technical Approach

This experiment uses a fully automated approach combining:

### Registration-Based Correspondence

Instead of traditional mesh parameterization methods (e.g., SPHARM-PDM), this pipeline uses **deformable image registration** to establish correspondences:

- **ANTs SyN (Symmetric Normalization)** performs diffeomorphic registration
- Distance maps from surface meshes create continuous fields for registration
- Progressive registration stages: rigid → affine → SyN deformable
- Mask-based approach focuses registration on anatomical structures

**Advantages:**
- Fully automated (no manual parameter tuning)
- Handles complex topologies naturally
- Diffeomorphic guarantees smooth, invertible deformations
- Integrates seamlessly with medical imaging pipelines

### PCA Computation

Statistical shape modeling uses **sklearn's PCA implementation**:
- Computes eigenvectors and eigenvalues from point correspondences
- Calculates explained variance for each mode
- Exports model in JSON format compatible with `RegisterModelsPCA` class

## Workflow

### 1. Download Data

See `data/KCL-Heart-Model/README.md` for download instructions. Place downloaded mesh files in `data/KCL-Heart-Model/input_meshes/`.

### 2. Run All Scripts in Sequence

```bash
cd experiments/Heart-Create_Statistical_Model/

# Execute scripts in order (each can also be stepped through cell-by-cell
# in VS Code or Cursor via the `# %%` cell markers):
python 1-input_meshes_to_input_surfaces.py     # Extract surfaces from volumetric meshes
python 2-input_surfaces_to_surfaces_aligned.py # Rigid ICP alignment + compute average
python 3-registration_based_correspondence.py  # ANTs SyN deformable correspondence
python 4-surfaces_aligned_correspond_to_pca_inputs.py  # Prepare PCA input matrices
python 5-compute_pca_model.py                  # Compute PCA and export JSON model
```

**Total Runtime:** Approximately 2-4 hours depending on hardware (20 heart meshes, ANTs registration is computationally intensive).

## Outputs

After completing this experiment, you will have generated files in `kcl-heart-model/`:

### Primary Output: PCA Model JSON
- **`pca_model.json`** - Complete PCA statistical model containing:
  - Mean shape (template surface)
  - PCA eigenvectors (shape modes)
  - PCA eigenvalues (variance per mode)
  - Explained variance ratios
  - Number of components
  - Ready for use with `RegisterModelsPCA` class

### Intermediate Outputs
- **`surfaces/`** - Extracted surface meshes from tetrahedral volumes
- **`surfaces_aligned/`** - ICP-aligned surfaces + `average_surface.vtp`
- **`surfaces_aligned_corresponded/`** - Deformably registered surfaces with correspondence
- **`pca_inputs/`** - Point coordinate matrices ready for PCA computation

### Visualizations
- Various visualization plots generated within notebooks showing:
  - Alignment quality
  - Deformation magnitude maps
  - PCA mode variations
  - Explained variance plots

## Usage in Patient Registration

The outputs from this experiment are used in the `Heart-Statistical_Model_To_Patient` experiment:

```python
from physiomotion4d import WorkflowFitStatisticalModelToPatient

# Use PCA model from this experiment
workflow = WorkflowFitStatisticalModelToPatient(
    moving_mesh=mean_shape,
    fixed_meshes=patient_surfaces,
    fixed_image=patient_ct,
    pca_json="pca_model.json",  # From this experiment
    pca_number_of_modes=10
)

registered_mesh = workflow.run_workflow()
```

## Requirements

### Software
- Python 3.11+ with PhysioMotion4D installed
- Jupyter Lab or Jupyter Notebook
- ITK, VTK, PyVista (included with PhysioMotion4D)
- ANTs (Advanced Normalization Tools) - installed automatically with PhysioMotion4D
- scikit-learn for PCA computation

### Data
- KCL Heart Model dataset (20 heart meshes)
- ~5GB disk space for intermediate files
- ~2GB for final outputs

### Compute
- CPU: Multi-core processor (8+ cores recommended for ANTs registration)
- RAM: 16GB minimum (32GB recommended)
- GPU: Not required for this experiment
- Time: ~2-4 hours total (ANTs deformable registration is computationally intensive)

## Citation

If you use this experiment or the KCL dataset, please cite:

> Rodero et al. (2021), "Linking statistical shape models and simulated function in the healthy adult human heart". *PLOS Computational Biology*. DOI: [10.1371/journal.pcbi.1008851](https://doi.org/10.1371/journal.pcbi.1008851)

For ANTs registration:
> Avants BB, et al. (2011). "A reproducible evaluation of ANTs similarity metric performance in brain image registration". *NeuroImage*. DOI: [10.1016/j.neuroimage.2010.09.025](https://doi.org/10.1016/j.neuroimage.2010.09.025)

## Related Experiments

- **`Heart-Statistical_Model_To_Patient`** - Uses PCA model from this experiment to register to patient data
- **`Heart-VTKSeries_To_USD`** - Converts VTK meshes to USD for Omniverse visualization
- **`Convert_VTK_To_USD`** - General VTK to USD conversion utilities

## Support and Resources

- **KCL Dataset**: [https://zenodo.org/records/4590294](https://zenodo.org/records/4590294)
- **ANTs Documentation**: [https://github.com/ANTsX/ANTs](https://github.com/ANTsX/ANTs)
- **PhysioMotion4D Documentation**: See main repository README and API documentation
- **Issues**: Report bugs or request features on the PhysioMotion4D GitHub repository

## Troubleshooting

### Missing Data Files
- Ensure KCL dataset is downloaded to `data/KCL-Heart-Model/input_meshes/`
- Check `data/KCL-Heart-Model/README.md` for download instructions
- Verify all 20 heart mesh files (`.vtk` format) are present

### ANTs Registration Taking Too Long
- ANTs SyN registration is computationally intensive (5-15 minutes per subject)
- Total time for 20 subjects: 2-4 hours is normal
- Consider using a machine with more CPU cores
- Progress is saved incrementally - can resume if interrupted

### Memory Issues
- Close other applications to free RAM
- ANTs registration can use 4-8GB per process
- Process fewer meshes initially to test pipeline
- Use a machine with more RAM (32GB+ recommended)

### Correspondence Quality Issues
- Check alignment quality from step 2 (ICP should produce good initial alignment)
- Verify average surface looks reasonable before step 3
- ANTs parameters are pre-tuned for cardiac anatomy
- If registration fails, check input mesh quality and topology

### Import Errors
- Ensure all PhysioMotion4D dependencies are installed
- Check that ANTs is available: `python -c "import ants; print(ants.__version__)"`
- Reinstall environment if needed: `pip install -e .` in repository root

---

**Next Steps:** After completing this experiment, proceed to `experiments/Heart-Statistical_Model_To_Patient/` to learn how to register the statistical model to patient-specific imaging data.
