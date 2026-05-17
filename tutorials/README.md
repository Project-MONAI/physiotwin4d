# PhysioMotion4D Tutorials

End-to-end Python scripts covering each major workflow in the library.
These are the recommended starting point for new users.

## Before You Begin

Each tutorial requires one or more public datasets.
**See [../data/README.md](../data/README.md)** for download instructions,
dataset licensing, and expected directory layout.

## Tutorial Index

| # | Script | Primary API | Dataset |
|---|--------|-------------|---------|
| 1 | [tutorial_01_heart_gated_ct_to_usd.py](tutorial_01_heart_gated_ct_to_usd.py) | `WorkflowConvertImageToUSD` | Slicer-Heart-CT (prepare first) |
| 2 | [tutorial_02_ct_to_vtk.py](tutorial_02_ct_to_vtk.py) | `WorkflowConvertImageToVTK` | Slicer-Heart-CT (prepare first) |
| 3 | [tutorial_03_create_statistical_model.py](tutorial_03_create_statistical_model.py) | `WorkflowCreateStatisticalModel` | KCL-Heart-Model (manual) |
| 4 | [tutorial_04_fit_statistical_model_to_patient.py](tutorial_04_fit_statistical_model_to_patient.py) | `WorkflowFitStatisticalModelToPatient` | KCL-Heart-Model plus Tutorial 3 output |
| 5 | [tutorial_05_vtk_to_usd.py](tutorial_05_vtk_to_usd.py) | `WorkflowConvertVTKToUSD` | Output of tutorial 2 |
| 6 | [tutorial_06_reconstruct_highres_4d_ct.py](tutorial_06_reconstruct_highres_4d_ct.py) | `WorkflowReconstructHighres4DCT` | DirLab-4DCT (manual) |
| 7 | [tutorial_07_dirlab_pca_model.py](tutorial_07_dirlab_pca_model.py) | `WorkflowCreateStatisticalModel`, `WorkflowFitStatisticalModelToPatient` | DirLab-4DCT (manual) |
| 8 | [tutorial_08_dirlab_pca_time_series.py](tutorial_08_dirlab_pca_time_series.py) | `RegisterTimeSeriesImages` | DirLab-4DCT plus Tutorial 7 output |
| 9 | [tutorial_09_physicsnemo_mesh_stage_model.py](tutorial_09_physicsnemo_mesh_stage_model.py) | `physicsnemo.models.mlp.FullyConnected` | Tutorial 8 output |

## Running a Tutorial

Each tutorial is a standalone percent-cell Python script (`# %%`) that can be
run cell-by-cell in VS Code or Cursor, or executed end-to-end as a regular
Python script. Paths are defined near the top of each script. By default, data
is read from the repository `data/` directory and outputs are written under
`tutorials/output/<tutorial_name>/`.

```bash
# Run the whole tutorial from the command line
python tutorials/tutorial_01_heart_gated_ct_to_usd.py
```

In VS Code or Cursor, open the tutorial and use **Run Python File** (or run
the cells in order with **Run Cell**). The script's `if __name__ ==
"__main__":` block executes the workflow and assigns the resulting
`tutorial_results` dict in the script's namespace; the same variable is what
`tests/test_tutorials.py` consumes via `runpy.run_path(..., run_name=
"__main__")`.

To use different paths, edit the constants near the top of the tutorial
script. For repeatable command-line execution with path arguments, use the
installed `physiomotion4d-*` CLI commands instead.

## Running as Pytest Tutorial Tests

All tutorials are wired into the test suite under the `tutorial` marker.
They run end-to-end and compare generated screenshots against baselines:

```bash
# Run all tutorial tests (requires data download first)
pytest tests/test_tutorials.py --run-tutorials -v

# Create baselines on first run
pytest tests/test_tutorials.py --run-tutorials --create-baselines -v

# Run a single tutorial test
pytest tests/test_tutorials.py::TestTutorial01HeartGatedCTToUSD --run-tutorials -v
```

## Recommended Order

1. **Tutorial 1** and **Tutorial 2** use Slicer-Heart-CT - prepare it per `data/README.md`, then start here.
2. **Tutorial 5** uses the VTK surfaces produced by Tutorial 2 - run Tutorial 2 first.
3. **Tutorial 3** creates the PCA statistical model from KCL-Heart-Model.
4. **Tutorial 4** applies the statistical model, consuming Tutorial 3 output.
5. **Tutorial 6** requires DirLab-4DCT - download it per `data/README.md`.
6. **Tutorial 7** creates a surface PCA model of the five lung lobes from DirLab-4DCT, then fits it to every available case.
7. **Tutorial 8** registers DirLab respiratory phases with ANTs+ICON and propagates the Tutorial 7 fitted meshes through each time series.
8. **Tutorial 9** trains a PhysicsNeMo model to predict a PCA-fitted mesh at a user-specified respiratory stage.

## For Contributors

Class-level API reference: [../docs/API_MAP.md](../docs/API_MAP.md)
