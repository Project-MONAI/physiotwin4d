# PhysioTwin4D Data Directory

This directory holds the sample datasets used for experiments, testing, and
development of the PhysioTwin4D library. Each subdirectory contains one
dataset and its own `README.md` with download instructions, specifications,
and citation — this file is just an index; treat the per-dataset READMEs as
the source of truth.

## Datasets

| Directory | Description | Provided By | Download | README |
| --- | --- | --- | --- | --- |
| `Slicer-Heart-CT/` | 4D cardiac CT with gated cardiac phases | Jolley Lab, Children's Hospital of Philadelphia (CHOP) | Automatic | [Slicer-Heart-CT/README.md](Slicer-Heart-CT/README.md) |
| `DirLab-4DCT/` | 4D lung CT respiratory motion benchmark | DIR-Lab, MD Anderson Cancer Center / Emory University | Manual | [DirLab-4DCT/README.md](DirLab-4DCT/README.md) |
| `KCL-Heart-Model/` | Statistical shape model of the heart | King's College London (KCL) | Automatic | [KCL-Heart-Model/README.md](KCL-Heart-Model/README.md) |
| `CHOP-Valve4D/` | 4D valve reconstruction models | Jolley Lab, CHOP (original FEBio model) | Automatic | [CHOP-Valve4D/README.md](CHOP-Valve4D/README.md) |
| `Chest-CT/` | Routine clinical 3D chest CT scan | PhysioTwin4D GitHub release | Automatic | [Chest-CT/README.md](Chest-CT/README.md) |
| `test/` | pytest-managed cache; not a downloadable dataset | — | N/A | [test/README.md](test/README.md) |

## Automatic Download

`Slicer-Heart-CT`, `KCL-Heart-Model`, `CHOP-Valve4D`, and `Chest-CT` can be
fetched with the `physiotwin4d-download-data` CLI or `DataDownloadTools`; see each
dataset's README for the exact command. `DirLab-4DCT` has no automatic
downloader — DIR-Lab distributes each case individually and may require
registration, so it must be obtained manually; see
[DirLab-4DCT/README.md](DirLab-4DCT/README.md).

## Notes

- Always cite the original data source in publications — see each dataset's
  README for the required citation.
- The full set of datasets is ~10-20 GB; ensure adequate disk space before
  downloading everything.
