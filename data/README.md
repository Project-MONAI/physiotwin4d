# PhysioTwin4D Data Directory

This directory contains sample datasets used for experiments, testing, and development of the PhysioTwin4D library. Each subdirectory contains a specific medical imaging dataset.

## Directory Structure

```text
data/
├── Slicer-Heart-CT/          # 4D cardiac CT with gated cardiac phases (AUTO-DOWNLOAD)
├── DirLab-4DCT/              # 4D lung CT benchmark dataset (MANUAL)
├── KCL-Heart-Model/          # Statistical shape model of the heart (AUTO-DOWNLOAD)
├── CHOP-Valve4D/             # 4D valve models (AUTO-DOWNLOAD)
├── test/                     # pytest-managed cache, not a manual/auto dataset
```

## Data Download Methods

### Automatic Download (Slicer-Heart-CT, KCL-Heart-Model, CHOP-Valve4D)

These datasets can be downloaded automatically via `DataDownloadTools` or the
`physiotwin4d-download-data` CLI:

```bash
physiotwin4d-download-data Slicer-Heart-CT --directory data/Slicer-Heart-CT
physiotwin4d-download-data KCL-Heart-Model --directory data/KCL-Heart-Model
physiotwin4d-download-data CHOP-Valve4D --directory data/CHOP-Valve4D
```

### Manual Download (DirLab-4DCT)

- **DirLab-4DCT**: Respiratory motion benchmark data must be **manually
  downloaded and preprocessed** by the user.

See individual dataset sections below for download instructions and preprocessing requirements.

---

## Slicer-Heart-CT AUTO-DOWNLOAD

### Description

4D cardiac CT dataset with temporal gating showing complete cardiac cycle motion. Pediatric cardiac CT with truncal valve visualization.

### Specifications

- **Format**: `.seq.nrrd` (4D NRRD sequence file)
- **Phases**: 21 temporal cardiac phases
- **Size**: ~1.2 GB
- **Content**: Contrast-enhanced cardiac CT
- **Anatomy**: Heart, great vessels, thoracic structures

### Acknowledgement

Data provided by Jolley Lab at CHOP (Children's Hospital of Philadelphia):

- [https://www.linkedin.com/company/jolleylab](https://www.linkedin.com/company/jolleylab)
- [https://github.com/SlicerHeart/SlicerHeart](https://github.com/SlicerHeart/SlicerHeart)

### Downloading the Data

**Automatic download** (recommended):

```python
from physiotwin4d import DataDownloadTools

DataDownloadTools.DownloadSlicerHeartCTData("data/Slicer-Heart-CT")
assert DataDownloadTools.VerifySlicerHeartCTData("data/Slicer-Heart-CT")
```

**Manual download** (alternative):

```bash
# Direct download link:
wget https://github.com/SlicerHeart/SlicerHeart/releases/download/TestingData/TruncalValve_4DCT.seq.nrrd -P data/Slicer-Heart-CT/
```

### Usage

- Primary dataset for tutorials
- Primary dataset for `Heart-GatedCT_To_USD` experiments
- Used in test suite (`tests/test_download_heart_data.py`)
- Example data for cardiac motion visualization in Omniverse

### Verification Helpers

PhysioTwin4D exposes a small public utility for checking optional dataset
layouts:

```python
from physiotwin4d import DataDownloadTools

DataDownloadTools.VerifySlicerHeartCTData("data/Slicer-Heart-CT")
DataDownloadTools.VerifyDirLab4DCTData("data/DirLab-4DCT")
DataDownloadTools.VerifyKCLHeartModelData("data/KCL-Heart-Model")
DataDownloadTools.VerifyCHOPValve4DData("data/CHOP-Valve4D")
```

---

## DirLab-4DCT MANUAL DOWNLOAD

### Description

Benchmark dataset for 4D CT respiratory motion analysis. Contains 10 cases of lung CT scans at different respiratory phases with annotated landmark points for registration validation.

### Specifications

- **Format**: `.mhd` headers + `.img` raw volumes (MetaImage format)
- **Cases**: 10 patient cases (Case 1-10)
- **Phases**: 10 respiratory phases per case (T00-T90)
- **Content**: Non-contrast lung CT
- **Anatomy**: Lungs, airways, thoracic structures
- **Landmarks**: 300+ annotated points per case for validation

### Acknowledgement

Data provided by the DIR-Lab at MD Anderson Cancer Center:

- **Project**: COPDGene 4D-CT dataset
- **Publication**: Castillo et al., "A reference dataset for deformable image registration spatial accuracy evaluation using the COPDGene study archive"
- **Website**: [https://med.emory.edu/departments/radiation-oncology/research-laboratories/deformable-image-registration/index.html](https://med.emory.edu/departments/radiation-oncology/research-laboratories/deformable-image-registration/index.html)

### Downloading the Data

**MANUAL DOWNLOAD REQUIRED**

Users must manually download and preprocess this dataset. Follow these steps:

**Step 1: Manual Download**

```python
# Using provided utilities in experiment scripts.
# See: experiments/Lung-GatedCT_To_USD/0-register_dirlab_4dct.py
# The script includes download utilities but requires manual execution.
```

**Step 2: User Preprocessing**
Users are responsible for:

- Downloading data from DIR-Lab website
- Extracting and organizing files in the proper directory structure
- Running preprocessing scripts if needed

### Directory Structure

```text
DirLab-4DCT/
├── Case1Pack/
│   ├── Images/              # T00-T50 phase images
│   ├── ExtremePhases/       # T00 and T50 (max inhale/exhale)
│   └── Sampled4D/           # Sampled time points
├── Case1Pack_T00.mhd        # Extracted phase files
├── Case1Pack_T10.mhd
...
├── Case10Pack/
```

### Usage

- Primary dataset for `Lung-GatedCT_To_USD` experiments
- Registration algorithm validation
- Respiratory motion analysis
- Benchmark for deformable registration accuracy

---

## KCL-Heart-Model AUTO-DOWNLOAD

### Description

Statistical shape model (SSM) of the human heart derived from cardiac imaging data. Includes principal component analysis (PCA) modes of shape variation.

### Specifications

- **Format**: `.vtk`, `.vtp` (VTK PolyData formats)
- **Content**:
  - Average heart surface and mesh
  - Individual heart models
- **Components**: Full heart mesh with chambers and vessels

### Files

- `average_mesh.vtk` - Mean heart volume mesh (UnstructuredGrid)
- input_meshes/01.vtk

### Acknowledgement

Data from King's College London (KCL):

- **Repository**: Cardiac imaging research group
- **License**: Check `citation.txt` for proper attribution

### Downloading the Data

**Automatic download** (recommended):

```bash
physiotwin4d-download-data KCL-Heart-Model --directory data/KCL-Heart-Model
```

```python
from physiotwin4d import DataDownloadTools

DataDownloadTools.DownloadKCLHeartModelData("data/KCL-Heart-Model")
assert DataDownloadTools.VerifyKCLHeartModelData("data/KCL-Heart-Model")
```

See `data/KCL-Heart-Model/README.md` for details on what is downloaded.

### Usage

- **Statistical shape model creation** (`experiments/Heart-Create_Statistical_Model/`) ⭐ **Primary use case**
- **Model-to-patient registration** (`experiments/Heart-Statistical_Model_To_Patient/`)
- VTK to USD conversion experiments (`experiments/Convert_VTK_To_USD/`)
- Shape-based cardiac analysis
- Atlas-based segmentation initialization
- Population-based statistical analysis

---

## CHOP-Valve4D AUTO-DOWNLOAD

### Description

Time-varying 4D valve reconstruction models showing valve motion over the cardiac cycle. These datasets represent dynamic valve geometries reconstructed from medical imaging data.

### Specifications

- **Format**: `.vtk` (VTK PolyData files)
- **Content**: Time series of valve surface meshes
- **Valves**: Alterra, TPV25, and other valve types
- **Phases**: Multiple time points per cardiac cycle (200+ frames)
- **Resolution**: High-resolution surface meshes with anatomical features

### Directory Structure

```text
CHOP-Valve4D/
├── Alterra/
│   ├── frame_0000.vtk
│   ├── frame_0001.vtk
│   └── ... (232 frames)
├── TPV25/
│   ├── frame_0000.vtk
│   ├── frame_0001.vtk
│   └── ... (265 frames)
```

### Acknowledgement

Data provided by Jolley Lab at CHOP (Children's Hospital of Philadelphia):

- [https://www.linkedin.com/company/jolleylab](https://www.linkedin.com/company/jolleylab)

### Downloading the Data

**Automatic download** (recommended):

```bash
physiotwin4d-download-data CHOP-Valve4D --directory data/CHOP-Valve4D
```

```python
from physiotwin4d import DataDownloadTools

DataDownloadTools.DownloadCHOPValve4DData("data/CHOP-Valve4D")
assert DataDownloadTools.VerifyCHOPValve4DData("data/CHOP-Valve4D")
```

This fetches the PhysioTwin4D convenience release (VTK/ITK conversion of the
original FEBio model). The original `.feb` source is available separately
from the FEBio website under CC-BY. See `data/CHOP-Valve4D/README.md` for
details and citation.

### Usage

- Time-series VTK to USD conversion (`experiments/Convert_VTK_To_USD/`)
- 4D valve motion visualization in NVIDIA Omniverse
- Temporal cardiac mechanics analysis
- Valve dynamics studies and surgical planning

### Related Resources

- **FEBio**: Finite Element Biomechanics software suite ([https://febio.org/](https://febio.org/))
- **Jolley Lab**: Cardiac imaging and computational modeling research

---

## Data Usage Guidelines

### For Testing

- Tests automatically use cached data when available
- Download occurs only if data is missing
- Tests use subsets (e.g., first 2 time points) for speed

### For Experiments

- Full datasets used for complete analysis
- Results saved to respective `experiments/*/results/` directories
- Original data remains unmodified

### For Development

- Use small subsets for rapid iteration
- Full datasets for validation and benchmarking
- Cache intermediate results to avoid reprocessing

---

## Data Access and Licensing

- **Slicer-Heart-CT** : Public release from GitHub (auto-download available)
- **DirLab-4DCT** : Public benchmark dataset (manual download required, may require registration)
- **KCL-Heart-Model** : Public release from Zenodo (auto-download available)
- **CHOP-Valve4D** : PhysioTwin4D convenience release under CC-BY license (auto-download available)

**Important**: Always cite the original data sources in publications and respect any usage restrictions.

### Summary of Download Methods


| Dataset         | Auto-Download | Manual Required | License         | Source              | Used in Tests            |
| --------------- | ------------- | --------------- | --------------- | ------------------- | ------------------------ |
| Slicer-Heart-CT | Yes           | No              | Public          | GitHub              | Yes                      |
| DirLab-4DCT     | No            | Yes             | Public/Academic | DIR-Lab             | No                       |
| KCL-Heart-Model | Yes           | No              | Check citation  | Zenodo/KCL          | Yes (skipped if missing) |
| CHOP-Valve4D    | Yes           | No              | CC-BY           | GitHub release      | No                       |


---

## References

### Slicer-Heart-CT

- Jolley Lab: [https://www.linkedin.com/company/jolleylab](https://www.linkedin.com/company/jolleylab)
- GitHub: [https://github.com/SlicerHeart/SlicerHeart](https://github.com/SlicerHeart/SlicerHeart)

### DirLab-4DCT

- Castillo et al., "A reference dataset for deformable image registration spatial accuracy evaluation using the COPDGene study archive"
- DIR-Lab: [https://med.emory.edu/departments/radiation-oncology/research-laboratories/deformable-image-registration/](https://med.emory.edu/departments/radiation-oncology/research-laboratories/deformable-image-registration/)

### KCL-Heart-Model

- Rodero et al. (2021), "Linking statistical shape models and simulated function in the healthy adult human heart", *PLOS Computational Biology*
- DOI: [10.1371/journal.pcbi.1008851](https://doi.org/10.1371/journal.pcbi.1008851)
- Zenodo: [https://zenodo.org/records/4590294](https://zenodo.org/records/4590294)

### CHOP-Valve4D

- Jolley Lab (CHOP): [https://www.linkedin.com/company/jolleylab](https://www.linkedin.com/company/jolleylab)
- Original FEBio source model: [repo.febio.org/permalink/project/136](https://repo.febio.org/permalink/project/136)
- License: Creative Commons Attribution (CC-BY)
- Citation: Please acknowledge Jolley Lab at CHOP and the FEBio Project

---

## Tips

1. **Storage**: Ensure adequate disk space (~10-20GB for all datasets)
2. **Download Time**: Initial downloads can be slow; be patient
3. **Organization**: Keep data organized; don't modify original files
4. **Backups**: Consider backing up processed results separately
5. **Documentation**: Update this README when adding new datasets

