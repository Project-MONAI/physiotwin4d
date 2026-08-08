# PhysioTwin4D Tutorials

End-to-end Python scripts covering each major workflow in the library.
These are the recommended starting point for new users.

## Before You Begin

These scripts live only in the source repository — `pip install physiotwin4d`
does not install them. Clone the repository first:

```bash
git clone https://github.com/Project-MONAI/physiotwin4d.git
cd physiotwin4d
```

Each tutorial requires one or more public datasets.
**See [../data/README.md](../data/README.md)** for download instructions,
dataset licensing, and expected directory layout. Run every download from the
top level of the clone: the tutorials resolve their inputs against the
repository root (`<repo>/data/<dataset>`), while
`physiotwin4d-download-data` writes to `data/<dataset>` relative to the
current working directory.

## Tutorial Index

| # | Script | Primary API | Dataset |
|---|--------|-------------|---------|
| 1 | [tutorial_01_heart_gated_ct_to_usd.py](tutorial_01_heart_gated_ct_to_usd.py) | `WorkflowConvertImageToUSD` | Slicer-Heart-CT (prepare first) |
| 1 | [tutorial_01_lung_gated_ct_to_usd.py](tutorial_01_lung_gated_ct_to_usd.py) | `WorkflowConvertImageToUSD` | Lung gated 4D CT (prepare first) |
| 2 | [tutorial_02_lung_finetune_icon.py](tutorial_02_lung_finetune_icon.py) | `WorkflowFinetuneICONRegistration` | DirLab-4DCT (manual) |
| 2 | [lung distancemap variant](tutorial_02_lung_distancemap_finetune_icon.py) | `WorkflowFinetuneICONRegistration` on distance maps | DirLab-4DCT (manual) |
| 2 | [heart distancemap variant](tutorial_02_duke_heart_distancemap_finetune_icon.py) | `WorkflowFinetuneICONRegistration` on distance maps | Duke-Heart-4DLabelmaps (not yet available) |
| 3 | [tutorial_03_heart_reconstruct_highres_4d_ct.py](tutorial_03_heart_reconstruct_highres_4d_ct.py) | `WorkflowReconstructHighres4DCT` | Slicer-Heart-CT (prepare first) |
| 3 | [tutorial_03_lung_reconstruct_highres_4d_ct.py](tutorial_03_lung_reconstruct_highres_4d_ct.py) | `WorkflowReconstructHighres4DCT` | DirLab-4DCT (manual) |
| 4 | [tutorial_04_heart_ct_to_vtk.py](tutorial_04_heart_ct_to_vtk.py) | `WorkflowConvertImageToVTK` | Slicer-Heart-CT (prepare first) |
| 4 | [tutorial_04_lung_ct_to_vtk.py](tutorial_04_lung_ct_to_vtk.py) | `WorkflowConvertImageToVTK` | Lung gated 4D CT (prepare first) |
| 5 | [tutorial_05_heart_vtk_to_usd.py](tutorial_05_heart_vtk_to_usd.py) | `WorkflowConvertVTKToUSD` | Output of tutorial 4 |
| 6 | [tutorial_06_heart_create_statistical_model.py](tutorial_06_heart_create_statistical_model.py) | `WorkflowCreateStatisticalModel` | KCL-Heart-Model |
| 6 | [tutorial_06_lung_create_statistical_model.py](tutorial_06_lung_create_statistical_model.py) | `WorkflowCreateStatisticalModel` | Lung surfaces from Tutorial 4 (lung) |
| 7 | [tutorial_07_heart_fit_statistical_model_to_patient.py](tutorial_07_heart_fit_statistical_model_to_patient.py) | `WorkflowFitStatisticalModelToPatient` | KCL-Heart-Model plus Tutorial 6 output |
| 7 | [tutorial_07_lung_fit_statistical_model_to_patient.py](tutorial_07_lung_fit_statistical_model_to_patient.py) | `WorkflowFitStatisticalModelToPatient` | Chest-CT plus Tutorial 6 (lung) output |
| 8 | [tutorial_08_lung_fit_model_to_4d_patients.py](tutorial_08_lung_fit_model_to_4d_patients.py) | `WorkflowFitStatisticalModelToPatient`, `WorkflowReconstructHighres4DCT` | DirLab-4DCT plus Tutorial 6 (lung) and Tutorial 2 output |
| 9 | [tutorial_09_lung_train_physicsnemo_mgn.py](tutorial_09_lung_train_physicsnemo_mgn.py) | `WorkflowTrainPhysicsNeMo`, `WorkflowInferPhysicsNeMo`, `WorkflowInferMovement` (requires `[physicsnemo]` extra + `torch-geometric`) | Tutorial 8 (lung) output |
| 10 | [tutorial_10_lung_infer_physicsnemo_mgn.py](tutorial_10_lung_infer_physicsnemo_mgn.py) | `WorkflowInferPhysicsNeMo`, `WorkflowInferMovement`, `WorkflowConvertVTKToUSD` (requires `[physicsnemo]` extra + `torch-geometric`) | Tutorial 8 and 9 (lung) output |

The [tutorials page](https://project-monai.github.io/physiotwin4d/tutorials.html)
covers the same set with previews of what each one produces and per-tutorial
notes on running them against your own data.

## Running a Tutorial

Each tutorial is a standalone, straightforward Python script, executed
end-to-end. Paths are defined near the top of each script. By default, data
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

Each numbered step has a heart variant, a lung variant, or both. Follow the
variants for the anatomy you care about: every tutorial consumes the output of
its own anatomy's earlier tutorials, never the other's.

1. **Tutorial 1** converts one gated 4D CT into an animated USD - the heart variant uses Slicer-Heart-CT, the lung variant DirLab-4DCT. Prepare the dataset for your anatomy per `data/README.md`, then start here.
2. **Tutorial 2** requires DirLab-4DCT (download it per `data/README.md`) and finetunes the ICON weights Tutorials 3 (lung) and 8 use when they are present — both fall back to the stock uniGradICON weights otherwise.
3. **Tutorial 3** registers with those weights; the heart variant uses Slicer-Heart-CT, the lung variant DirLab-4DCT.
4. **Tutorial 4** segments a CT into VTK surfaces; the heart variant uses Slicer-Heart-CT, the lung variant DirLab-4DCT.
5. **Tutorial 5** (heart only) uses the VTK surfaces produced by Tutorial 4 (heart) - run Tutorial 4 first.
6. **Tutorial 6** creates the PCA statistical model; the heart variant from KCL-Heart-Model, the lung variant from the DirLab-4DCT `Case*T70.mha` phases, which it segments itself. Both write `pca_model.json` and `pca_mean_surface.vtp` under their own output directory.
7. **Tutorial 7** applies the statistical model, consuming its own anatomy's Tutorial 6 output; the heart variant fits the Tutorial 6 (heart) model, the lung variant fits the Tutorial 6 (lung) model to the routine clinical `Chest-CT` scan (`physiotwin4d-download-data Chest-CT`).

The AI-surrogate pipeline (Tutorials 8 -> 9 -> 10) runs on DIR-Lab and the
Tutorial 6 lung model, in order:

8. **Tutorial 8** fits the lung PCA model to each case's reference phase and propagates the fitted SSM surface through every respiratory phase (output feeds Tutorial 9). It uses the Tutorial 2 ICON weights when they exist.
9. **Tutorial 9** trains a PhysicsNeMo MeshGraphNet to predict the per-vertex motion at any stage. PhysicsNeMo is an optional extra: install with `pip install "physiotwin4d[physicsnemo]"` (requires Python >= 3.11); the MeshGraphNet also needs `torch-geometric`. A `TrainPhysicsNeMoMLP` method exists as a drop-in alternative, without its own tutorial.
10. **Tutorial 10** loads that checkpoint and predicts one case's surface at a requested stage, scoring it against the acquired phase and exporting USD. The case, checkpoint epoch, and stage are constants near the top of the script; for command-line runs with path arguments, use the installed `physiotwin4d-infer-physicsnemo` CLI.

## For Contributors

Class-level API reference: [../docs/api/index.rst](../docs/api/index.rst)

To explore the code with an AI assistant, query the graphify knowledge graph
(`graphify query "<question>"`) instead of grepping — see
[../docs/developer/ai_assistants.rst](../docs/developer/ai_assistants.rst)
