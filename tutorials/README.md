# PhysioMotion4D Tutorials

End-to-end Python scripts covering each major workflow in the library.
These are the recommended starting point for new users.

## Before You Begin

Each tutorial requires one or more public datasets.
**See [../data/README.md](../data/README.md)** for download instructions,
dataset licensing, and expected directory layout.

## Tutorial Index

| # | Script | Workflow Class | CLI Command | Dataset |
|---|--------|---------------|-------------|---------|
| 1 | [tutorial_01_heart_gated_ct_to_usd.py](tutorial_01_heart_gated_ct_to_usd.py) | `WorkflowConvertHeartGatedCTToUSD` | `physiomotion4d-heart-gated-ct` | Slicer-Heart-CT (prepare first) |
| 2 | [tutorial_02_ct_to_vtk.py](tutorial_02_ct_to_vtk.py) | `WorkflowConvertCTToVTK` | `physiomotion4d-convert-ct-to-vtk` | Slicer-Heart-CT (prepare first) |
| 3 | [tutorial_03_fit_statistical_model_to_patient.py](tutorial_03_fit_statistical_model_to_patient.py) | `WorkflowFitStatisticalModelToPatient` | `physiomotion4d-fit-statistical-model-to-patient` | KCL-Heart-Model (manual) |
| 4 | [tutorial_04_create_statistical_model.py](tutorial_04_create_statistical_model.py) | `WorkflowCreateStatisticalModel` | `physiomotion4d-create-statistical-model` | KCL-Heart-Model (manual) |
| 5 | [tutorial_05_vtk_to_usd.py](tutorial_05_vtk_to_usd.py) | `WorkflowConvertVTKToUSD` | `physiomotion4d-convert-vtk-to-usd` | Output of tutorial 2 |
| 6 | [tutorial_06_reconstruct_highres_4d_ct.py](tutorial_06_reconstruct_highres_4d_ct.py) | `WorkflowReconstructHighres4DCT` | `physiomotion4d-reconstruct-highres-4d-ct` | DirLab-4DCT (manual) |

## Running a Tutorial

Each tutorial is a standalone Python script with a `run_tutorial(data_dir, output_dir)`
function. Run from the repository root:

```bash
python tutorials/tutorial_01_heart_gated_ct_to_usd.py \
    --data-dir ./data --output-dir ./output

python tutorials/tutorial_02_ct_to_vtk.py \
    --data-dir ./data --output-dir ./output
```

## Running as Pytest Experiment Tests

All tutorials are wired into the test suite under the `experiment` marker.
They run end-to-end and compare generated screenshots against baselines:

```bash
# Run all tutorial tests (requires data download first)
pytest tests/test_tutorials.py --run-experiments -v

# Create baselines on first run
pytest tests/test_tutorials.py --run-experiments --create-baselines -v

# Run a single tutorial test
pytest tests/test_tutorials.py::TestTutorial01HeartGatedCTToUSD --run-experiments -v
```

## Recommended Order

1. **Tutorial 1** and **Tutorial 2** use Slicer-Heart-CT - prepare it per `data/README.md`, then start here.
2. **Tutorial 5** uses the VTK surfaces produced by Tutorial 2 - run Tutorial 2 first.
3. **Tutorials 3 and 4** require the KCL-Heart-Model - download it per `data/README.md`.
4. **Tutorial 6** requires DirLab-4DCT - download it per `data/README.md`.

## For Contributors

Class-level API reference: [../docs/API_MAP.md](../docs/API_MAP.md)
