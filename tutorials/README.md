# PhysioTwin4D Tutorials

End-to-end Python scripts covering each major workflow in the library.
These are the recommended starting point for new users.

## Before You Begin

Each tutorial requires one or more public datasets.
**See [../data/README.md](../data/README.md)** for download instructions,
dataset licensing, and expected directory layout.

## Tutorial Index

| # | Script | Primary API | Dataset |
|---|--------|-------------|---------|
| 1a | [tutorial_01a_heart_gated_ct_to_usd.py](tutorial_01a_heart_gated_ct_to_usd.py) | `WorkflowConvertImageToUSD` | Slicer-Heart-CT (prepare first) |
| 2 | [tutorial_02_ct_to_vtk.py](tutorial_02_ct_to_vtk.py) | `WorkflowConvertImageToVTK` | Slicer-Heart-CT (prepare first) |
| 3 | [tutorial_03_vtk_to_usd.py](tutorial_03_vtk_to_usd.py) | `WorkflowConvertVTKToUSD` | Output of tutorial 2 |
| 4a | [tutorial_04a_heart_create_statistical_model.py](tutorial_04a_heart_create_statistical_model.py) | `WorkflowCreateStatisticalModel` | KCL-Heart-Model |
| 5a | [tutorial_05a_heart_fit_statistical_model_to_patient.py](tutorial_05a_heart_fit_statistical_model_to_patient.py) | `WorkflowFitStatisticalModelToPatient` | KCL-Heart-Model plus Tutorial 4a output |
| 6 | [tutorial_06_reconstruct_highres_4d_ct.py](tutorial_06_reconstruct_highres_4d_ct.py) | `WorkflowReconstructHighres4DCT` | DirLab-4DCT (manual) |
| 8cd | [tutorial_08cd_byod_fit_model_to_patients.py](tutorial_08cd_byod_fit_model_to_patients.py) | `WorkflowFitStatisticalModelToPatient`, `WorkflowReconstructHighres4DCT` | Bring your own (cardiac gated CT, `D:/PhysioTwin4D/`) |
| 9c | [tutorial_09c_byod_train_physicsnemo_mgn.py](tutorial_09c_byod_train_physicsnemo_mgn.py) | `WorkflowTrainPhysicsNeMoMGN` (requires `[physicsnemo]` extra + `torch-geometric`) | Tutorial 8cd output |
| 9d | [tutorial_09d_byod_train_physicsnemo_mlp.py](tutorial_09d_byod_train_physicsnemo_mlp.py) | `WorkflowTrainPhysicsNeMoMLP` (requires `[physicsnemo]` extra) | Tutorial 8cd output |
| 10c | [tutorial_10c_byod_eval_physicsnemo_mgn.py](tutorial_10c_byod_eval_physicsnemo_mgn.py) | `WorkflowInferPhysicsNeMoMGN` (requires `[physicsnemo]` extra + `torch-geometric`) | Tutorial 9c checkpoint |
| 10d | [tutorial_10d_byod_eval_physicsnemo_mlp.py](tutorial_10d_byod_eval_physicsnemo_mlp.py) | `WorkflowInferPhysicsNeMoMLP` (requires `[physicsnemo]` extra) | Tutorial 9d checkpoint |

> **Tutorials 8cd-10cd are bring-your-own-data.** Unlike the earlier data-driven
> tutorials, they do not use the repository `data/` directory or a downloadable
> sample. Their path constants point at a local `D:/PhysioTwin4D/` cardiac layout
> (gated CT, labelmaps, the KCL volume PCA model, and ICON weights); edit those
> constants to match your own data.

## Running a Tutorial

Each tutorial is a standalone percent-cell Python script (`# %%`) that can be
run cell-by-cell in VS Code or Cursor, or executed end-to-end as a regular
Python script. Paths are defined near the top of each script. By default, data
is read from the repository `data/` directory and outputs are written under
`tutorials/output/<tutorial_name>/`.

```bash
# Run the whole tutorial from the command line
python tutorials/tutorial_01a_heart_gated_ct_to_usd.py
```

In VS Code or Cursor, open the tutorial and use **Run Python File** (or run
the cells in order with **Run Cell**). The script's `if __name__ ==
"__main__":` block executes the workflow and assigns the resulting
`tutorial_results` dict in the script's namespace; the same variable is what
`tests/test_tutorials.py` consumes via `runpy.run_path(..., run_name=
"__main__")`.

To use different paths, edit the constants near the top of the tutorial
script. For repeatable command-line execution with path arguments, use the
installed `physiotwin4d-*` CLI commands instead.

## Running as Pytest Tutorial Tests

All tutorials are wired into the test suite under the `tutorial` marker.
They run end-to-end and compare generated screenshots against baselines:

```bash
# Run all tutorial tests (requires data download first)
pytest tests/test_tutorials.py --run-tutorials -v

# Create baselines on first run
pytest tests/test_tutorials.py --run-tutorials --create-baselines -v

# Run a single tutorial test
pytest tests/test_tutorials.py::TestTutorial01aHeartGatedCTToUSD --run-tutorials -v
```

## Recommended Order

1. **Tutorial 1a** and **Tutorial 2** use Slicer-Heart-CT - prepare it per `data/README.md`, then start here.
2. **Tutorial 3** uses the VTK surfaces produced by Tutorial 2 - run Tutorial 2 first.
3. **Tutorial 4a** creates the PCA statistical model from KCL-Heart-Model.
4. **Tutorial 5a** applies the statistical model, consuming Tutorial 4a output.
5. **Tutorial 6** requires DirLab-4DCT - download it per `data/README.md`.

The cardiac mesh stage-prediction pipeline (Tutorials 8cd -> 9c/9d -> 10c/10d) is
bring-your-own-data and runs in order:

6. **Tutorial 8cd** fits the KCL cardiac PCA model to each patient's reference CT and propagates the fitted SSM mesh through every gated phase (output feeds Tutorial 9c/9d).
7. **Tutorial 9c / 9d** train a PhysicsNeMo MeshGraphNet (9c) and MLP (9d) to predict a cardiac surface at any cardiac stage. PhysicsNeMo is an optional extra: install with `pip install "physiotwin4d[physicsnemo]"` (requires Python >= 3.11); the MeshGraphNet also needs `torch-geometric`.
8. **Tutorial 10c / 10d** load a trained MeshGraphNet (10c) or MLP (10d) checkpoint and predict / score cardiac surfaces for one subject. Each can be run from the command line or, with no arguments, via its `run_tutorial` entry point.

## For Contributors

Class-level API reference: [../docs/API_MAP.md](../docs/API_MAP.md)
