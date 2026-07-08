# DirLab-4DCT

## Download

**Manual download required** — this dataset is not fetched by
`physiotwin4d-download-data`; there is no automatic downloader because
DIR-Lab distributes each case individually and may require registration.

1. Visit the DIR-Lab 4D-CT page and request/download the case archives:
   https://med.emory.edu/departments/radiation-oncology/research-laboratories/deformable-image-registration/downloads-and-reference-data/4dct.html
2. Extract each case's raw images into `data/DirLab-4DCT/` (see layout
   below). The `.mhd` headers already committed in this directory point at
   those raw files, so no conversion step is required — see
   "About the Committed `.mhd` Files" below.

Once populated, check the layout with:

```python
from physiotwin4d import DataDownloadTools

assert DataDownloadTools.VerifyDirLab4DCTData("data/DirLab-4DCT")
```

**Directory structure after download:**
```
data/DirLab-4DCT/
├── Case1Pack/
│   ├── Images/              # T00-T90 phase images
│   ├── ExtremePhases/       # T00 and T50 (max inhale/exhale)
│   └── Sampled4D/           # Sampled time points
├── Case1Pack_T00.mhd        # Already-committed headers, per phase
├── Case1Pack_T10.mhd
...
├── Case10Pack/
├── Convert4DCTToMHD.py      # Documents how the .mhd headers were generated
└── README.md (this file)
```

### About the Committed `.mhd` Files

The `Case*Pack_T*.mhd` files in this directory are already committed to the
repository, but they are only MetaImage **headers** (a few hundred bytes
each) — for example:

```
ObjectType = Image
NDims = 3
DimSize = 256 256 94
...
ElementDataFile = Case1Pack/Images/case1_T00_s.img
```

Each header's `ElementDataFile` points at the raw image data inside the
corresponding `Case*Pack/`/`Case*Deploy/` subdirectory (`.gitignore`d
because those raw volumes are large). These `.mhd` files will not load
until you complete the manual download above and the referenced
`Case*Pack/Images/*.img` files exist alongside them.

`Convert4DCTToMHD.py` is what originally generated these headers from the
raw DIR-Lab archives. It is included for provenance/documentation only —
you do not need to run it; the `.mhd` files are already committed.

## Overview

Benchmark dataset for 4D CT respiratory motion analysis. Contains 10 cases
of lung CT scans at different respiratory phases with annotated landmark
points for registration validation.

### Dataset Details

- **Format**: `.mhd`/`.raw` (MetaImage format)
- **Cases**: 10 patient cases (Case 1-10)
- **Phases**: 10 respiratory phases per case (T00-T90)
- **Content**: Non-contrast lung CT
- **Anatomy**: Lungs, airways, thoracic structures
- **Landmarks**: 300+ annotated points per case for registration validation

### Acknowledgement

Data provided by the DIR-Lab at MD Anderson Cancer Center / Emory
University:
https://med.emory.edu/departments/radiation-oncology/research-laboratories/deformable-image-registration/

### Citation

Dataset: https://med.emory.edu/departments/radiation-oncology/research-laboratories/deformable-image-registration/downloads-and-reference-data/4dct.html

If you use this dataset, please cite:

- Case numbers 4DCT1-4DCT5: Castillo R, Castillo E, Guerra R, Johnson VE,
  McPhail T, Garg AK, Guerrero T. 2009. "A framework for evaluation of
  deformable image registration spatial accuracy using large landmark
  point sets." *Phys Med Biol* 54:1849-1870.
- Case numbers 4DCT6-4DCT10: Castillo E, Castillo R, Martinez J, Shenoy M,
  Guerrero T. 2009. "Four-dimensional deformable image registration using
  trajectory modeling." *Phys Med Biol* 55:305-327.

## Using This Dataset

- Primary dataset for `experiments/Lung-GatedCT_To_USD/`
- Registration algorithm validation
- Respiratory motion analysis
- Benchmark for deformable registration accuracy

### Files in This Directory

- `Convert4DCTToMHD.py` — documents how the committed `.mhd` headers were
  generated from the raw DIR-Lab archives; not needed to run
- `Case*Pack_T*.mhd` — MetaImage headers for each case/phase (see above)

### Additional Resources

- Reference implementation for reading DIR-Lab's raw `.img` case files:
  https://github.com/hsokooti/RegNet/blob/master/functions/preprocessing/dirlab.py
