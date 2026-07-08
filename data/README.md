# PhysioTwin4D Data Directory

This directory holds the example datasets used by PhysioTwin4D's tutorials,
experiments, and tests. Each subdirectory is one dataset and has its own
`README.md` with the download URL(s), directory layout, data sizes, and the
exact `physiotwin4d-download-data` command for that dataset. This file is
just the map — see the subdirectory README before downloading anything.

## Datasets

| Directory                          | Data Type                                              | Provided By                            | Auto-Download |
| ----------------------------------- | ------------------------------------------------------- | --------------------------------------- | ------------- |
| [`Slicer-Heart-CT/`](Slicer-Heart-CT/README.md) | 4D gated cardiac CT (`.seq.nrrd`)         | Jolley Lab (CHOP) / SlicerHeart on GitHub | Yes           |
| [`DirLab-4DCT/`](DirLab-4DCT/README.md)   | 4D lung CT respiratory benchmark + landmarks (`.mhd/.raw`) | DIR-Lab (Emory / MD Anderson)          | No — manual, may require registration |
| [`KCL-Heart-Model/`](KCL-Heart-Model/README.md) | Statistical heart shape model meshes (`.vtk`)   | King's College London, via Zenodo       | Yes           |
| [`CHOP-Valve4D/`](CHOP-Valve4D/README.md) | FEBio valve FE model, converted to VTK/ITK + Simpleware segmentation | Jolley Lab (CHOP), converted by PhysioTwin4D | Yes           |

[`data/test/`](test/README.md) is not a dataset — it is a cache
automatically managed by the pytest infrastructure (mainly from
`Slicer-Heart-CT`) and is not used by the workflows, tutorials, or CLIs.

## Downloading Data

Datasets marked **Yes** above can be fetched with the
`physiotwin4d-download-data` CLI (or the corresponding
`DataDownloadTools.Download*` method), for example:

```bash
physiotwin4d-download-data Slicer-Heart-CT --directory data/Slicer-Heart-CT
physiotwin4d-download-data KCL-Heart-Model --directory data/KCL-Heart-Model
physiotwin4d-download-data CHOP-Valve4D --directory data/CHOP-Valve4D
```

Datasets marked **No** must be downloaded by hand following the
instructions in that dataset's own README (`DirLab-4DCT/README.md`).

### Verification Helpers

Each dataset also has a `DataDownloadTools.Verify*Data()` helper that checks
whether the expected files are already present, so scripts can skip
downloading when the data is cached:

```python
from physiotwin4d import DataDownloadTools

DataDownloadTools.VerifySlicerHeartCTData("data/Slicer-Heart-CT")
DataDownloadTools.VerifyDirLab4DCTData("data/DirLab-4DCT")
DataDownloadTools.VerifyKCLHeartModelData("data/KCL-Heart-Model")
DataDownloadTools.VerifyCHOPValve4DData("data/CHOP-Valve4D")
```

## Guidelines

- Keep original downloaded data unmodified; write experiment/tutorial
  results under `experiments/*/results/` or `tutorials/*/results/`, not
  back into `data/`.
- Always respect the license and citation requirements of each dataset —
  see the "Citation" / "Acknowledgement" section in the relevant
  subdirectory README.
- If you add a new dataset, add both a row above and a subdirectory
  `README.md` following the existing datasets as a template.
