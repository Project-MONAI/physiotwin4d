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
| 1 | [tutorial_01_heart_gated_ct_to_usd.py](tutorial_01_heart_gated_ct_to_usd.py) | `WorkflowConvertImageToUSD` | Slicer-Heart-CT (prepare first) |
| 2 | [tutorial_02_ct_to_vtk.py](tutorial_02_ct_to_vtk.py) | `WorkflowConvertImageToVTK` | Slicer-Heart-CT (prepare first) |
| 3 | [tutorial_03_create_statistical_model.py](tutorial_03_create_statistical_model.py) | `WorkflowCreateStatisticalModel` | KCL-Heart-Model (manual) |
| 4 | [tutorial_04_fit_statistical_model_to_patient.py](tutorial_04_fit_statistical_model_to_patient.py) | `WorkflowFitStatisticalModelToPatient` | KCL-Heart-Model plus Tutorial 3 output |
| 5 | [tutorial_05_vtk_to_usd.py](tutorial_05_vtk_to_usd.py) | `WorkflowConvertVTKToUSD` | Output of tutorial 2 |
| 6 | [tutorial_06_reconstruct_highres_4d_ct.py](tutorial_06_reconstruct_highres_4d_ct.py) | `WorkflowReconstructHighres4DCT` | DirLab-4DCT (manual) |
| 8 | [tutorial_08_cardiac_fit_model.py](tutorial_08_cardiac_fit_model.py) | `WorkflowFitStatisticalModelToPatient`, `WorkflowReconstructHighres4DCT` | Bring your own (cardiac gated CT, `D:/PhysioTwin4D/`) |
| 9a | [tutorial_09a_cardiac_train_physicsnemo_mgn.py](tutorial_09a_cardiac_train_physicsnemo_mgn.py) | `physicsnemo.models.meshgraphnet.MeshGraphNet` (requires `[physicsnemo]` extra + `torch-geometric`) | Tutorial 8 output |
| 9b | [tutorial_09b_cardiac_train_physicsnemo_mlp.py](tutorial_09b_cardiac_train_physicsnemo_mlp.py) | `physicsnemo.models.mlp.FullyConnected` (requires `[physicsnemo]` extra) | Tutorial 8 output |
| 10a | [tutorial_10a_cardiac_eval_physicsnemo_mgn.py](tutorial_10a_cardiac_eval_physicsnemo_mgn.py) | `physicsnemo.models.meshgraphnet.MeshGraphNet` (requires `[physicsnemo]` extra + `torch-geometric`) | Tutorial 9a checkpoint |
| 10b | [tutorial_10b_cardiac_eval_physicsnemo_mlp.py](tutorial_10b_cardiac_eval_physicsnemo_mlp.py) | `physicsnemo.models.mlp.FullyConnected` (requires `[physicsnemo]` extra) | Tutorial 9b checkpoint |

> **Tutorials 8-10 are bring-your-own-data.** Unlike Tutorials 1-6, they do not
> use the repository `data/` directory or a downloadable sample. Their path
> constants point at a local `D:/PhysioTwin4D/` cardiac layout (gated CT,
> labelmaps, the KCL volume PCA model, and ICON weights); edit those constants to
> match your own data. (The former DirLab lung-lobe PCA tutorial, number 7, has
> been removed; the numbering continues at 8.)

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
pytest tests/test_tutorials.py::TestTutorial01HeartGatedCTToUSD --run-tutorials -v
```

## Recommended Order

1. **Tutorial 1** and **Tutorial 2** use Slicer-Heart-CT - prepare it per `data/README.md`, then start here.
2. **Tutorial 5** uses the VTK surfaces produced by Tutorial 2 - run Tutorial 2 first.
3. **Tutorial 3** creates the PCA statistical model from KCL-Heart-Model.
4. **Tutorial 4** applies the statistical model, consuming Tutorial 3 output.
5. **Tutorial 6** requires DirLab-4DCT - download it per `data/README.md`.

The cardiac mesh stage-prediction pipeline (Tutorials 8 -> 9 -> 10) is
bring-your-own-data and runs in order:

6. **Tutorial 8** fits the KCL cardiac PCA model to each patient's reference CT and propagates the fitted SSM mesh through every gated phase (output feeds Tutorial 9).
7. **Tutorial 9a / 9b** train a PhysicsNeMo MeshGraphNet (9a) and MLP (9b) to predict a cardiac surface at any cardiac stage. PhysicsNeMo is an optional extra: install with `pip install "physiotwin4d[physicsnemo]"` (requires Python >= 3.11); the MeshGraphNet also needs `torch-geometric`.
8. **Tutorial 10a / 10b** load a trained MeshGraphNet (10a) or MLP (10b) checkpoint and predict / score cardiac surfaces for one subject. Each can be run from the command line or, with no arguments, via its `run_tutorial` entry point.

## For Contributors

Class-level API reference: [../docs/API_MAP.md](../docs/API_MAP.md)
