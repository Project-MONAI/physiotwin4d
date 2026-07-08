# data/test

This directory is **automatically managed by the pytest infrastructure**
(`tests/conftest.py`) — it is a cache, not a dataset you download or
maintain by hand. It holds data used to run the unit test suite.

This data is **not** used by the workflows, tutorials, or CLIs of the
PhysioTwin4D library; those consume the datasets documented in
[`data/README.md`](../README.md) instead.

## What Lives Here

- `slicer_heart/` — a cached copy of the `Slicer-Heart-CT` 4D CT sequence
  (see the `download_test_data` fixture), split into per-phase `.mha`
  slices by `test_download_heart_data.py`.
- `slicer_heart_small/` — the same phases downsampled to 1.5x1.5x1.5 mm,
  used by tests that need a smaller/faster image (labelmaps and
  transforms computed from this data are cached here too).

Both subdirectories are created on demand by `tests/conftest.py` fixtures
the first time a test needs them, and are `.gitignore`d — do not commit
their contents.

## Regenerating

If this directory is deleted or corrupted, simply re-run the test suite;
the fixtures in `tests/conftest.py` will re-download and rebuild everything
here automatically:

```bash
py -m pytest tests/ -v
```
