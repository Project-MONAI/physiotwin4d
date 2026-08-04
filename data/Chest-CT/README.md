# Chest-CT

## Download

Download this dataset automatically with:

```bash
physiotwin4d-download-data Chest-CT --directory data/Chest-CT
```

or from Python:

```python
from physiotwin4d import DataDownloadTools

data_file = DataDownloadTools.DownloadChestCTData("data/Chest-CT")
assert DataDownloadTools.VerifyChestCTData("data/Chest-CT")
```

This fetches a single ~200 MB file from the PhysioTwin4D GitHub release
[2026.07.1](https://github.com/Project-MONAI/physiotwin4d/releases/download/2026.07.1/Chest-CT.mha).
An existing non-empty `Chest-CT.mha` is reused, so re-running the command
resumes an interrupted download.

**Directory structure after download:**
```text
data/Chest-CT/
├── Chest-CT.mha
└── README.md (this file)
```

## Overview

A routine, clinical, 3D chest CT scan. Unlike the gated 4D datasets in this
directory, it is a single static volume — one acquisition, no temporal
phases — so it stands in for the everyday clinical scan a patient-specific
model is fitted to.

### Dataset Details

- **Format**: `.mha` (compressed MetaImage)
- **Dimensionality**: 3D, single time point
- **Size**: ~200 MB
- **Content**: Routine clinical chest CT
- **Anatomy**: Lungs, heart, mediastinum, thoracic skeleton

## Using This Dataset

- Patient image for
  `tutorials/tutorial_07_lung_fit_statistical_model_to_patient.py`, which
  segments the lungs from this scan and fits the lung PCA shape model built
  by `tutorials/tutorial_06_lung_create_statistical_model.py` to them

### Files in This Directory

- `Chest-CT.mha` — the downloaded chest CT volume
