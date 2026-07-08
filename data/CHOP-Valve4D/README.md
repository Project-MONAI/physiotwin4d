# CHOP-Valve4D

## Download

Download this dataset automatically with:

```bash
physiotwin4d-download-data CHOP-Valve4D --directory data/CHOP-Valve4D
```

or from Python:

```python
from physiotwin4d import DataDownloadTools

DataDownloadTools.DownloadCHOPValve4DData("data/CHOP-Valve4D")
assert DataDownloadTools.VerifyCHOPValve4DData("data/CHOP-Valve4D")
```

This downloads and extracts three zip archives attached to the
[PhysioTwin4D 2026.07.1 GitHub release](https://github.com/Project-MONAI/physiotwin4d/releases/tag/2026.07.1)
into matching subdirectories:

| Release asset                | Extracted into | Contents                                   | Size    |
| ----------------------------- | --------------- | ------------------------------------------- | ------- |
| `CHOP-Valve4D-Alterra.zip`    | `Alterra/`       | Alterra valve mesh time series (`.vtk`)      | >1 GB   |
| `CHOP-Valve4D-TPV25.zip`      | `TPV25/`         | TPV25 valve mesh time series (`.vtk`)        | >1 GB   |
| `CHOP-Valve4D-CT.zip`         | `CT/`            | Source CT volume + Simpleware segmentation   | —       |

A subdirectory that already contains files is left alone, so re-running the
command resumes an interrupted download.

**Directory structure after download:**
```
data/CHOP-Valve4D/
├── Alterra/
│   ├── frame_0000.vtk
│   ├── frame_0001.vtk
│   └── ...
├── TPV25/
│   ├── frame_0000.vtk
│   ├── frame_0001.vtk
│   └── ...
├── CT/
│   ├── RVOT28-Dias.nii.gz (or .mha)
│   └── Simpleware/
│       └── parts/
└── README.md (this file)
```

## Overview

Time-varying 4D valve reconstruction models showing transcatheter
pulmonary valve motion over the cardiac cycle, derived from an FEBio
finite-element simulation of an Alterra valve deployed in a native right
ventricular outflow tract (RVOT).

### Original Data

The source data is an FEBio finite-element model (`.feb`) published by the
Jolley Lab at CHOP (Children's Hospital of Philadelphia). This repository
does not redistribute the source `.feb` file — it is available from the
FEBio website:

- Project page: https://repo.febio.org/permalink/project/136
- Direct download: https://repo.febio.org/modelRepo/api/v1.05/files/0/136

### Converted Data (PhysioTwin4D Convenience Release)

As a convenience, PhysioTwin4D converts the original FEBio geometry to VTK
(surface and volumetric meshes) and ITK (image) formats, and also provides
a Simpleware segmentation derived from the model. These converted files are
what the `physiotwin4d-download-data CHOP-Valve4D` command above fetches;
they are not tracked in this git repository. The original citation and
license terms below still apply to this converted data.

### Citation

Zelonis, C. N., Maheshwari, J., Wu, W., Maas, S. A., Aslan, S., Sunderland, K.,
... & Jolley, M. A. (2025). Integrated Open-Source Framework for Simulation of
Transcatheter Pulmonary Valves in Native Right Ventricular Outflow Tracts. arXiv
preprint arXiv:2507.06337.

### Acknowledgement

Data provided by the Jolley Lab at CHOP (Children's Hospital of
Philadelphia):
- https://www.linkedin.com/company/jolleylab

## Using This Dataset

- Time-series VTK to USD conversion (`experiments/Convert_VTK_To_USD/`)
- 4D valve motion visualization in NVIDIA Omniverse
- Temporal cardiac mechanics analysis
- Valve dynamics studies and surgical planning
