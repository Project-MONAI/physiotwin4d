=========
Tutorials
=========

.. raw:: html

   <section class="pm4d-hero">
     <div class="pm4d-hero__brand">
       <img src="_static/nvidia-logo.svg" alt="NVIDIA logo">
     </div>
     <p class="pm4d-kicker">PhysioMotion4D tutorials</p>
     <h1>Build animated medical USD workflows for NVIDIA Omniverse</h1>
     <p>
       Nine focused tutorials walk through CT segmentation, registration,
       statistical model fitting, high-resolution 4D reconstruction, and USD
       export. Each card links to implementation details, datasets, and the
       percent-cell Python script used to run the workflow.
     </p>
   </section>

   <section class="pm4d-card-grid" aria-label="Tutorial cards">
     <a class="pm4d-card" href="#tutorial-1-heart-gated-ct-to-animated-usd">
       <span class="pm4d-card__number">01</span>
       <h2>Heart-Gated CT to Animated USD</h2>
       <p>Convert cardiac 4D CT frames into registered contours and an animated OpenUSD model.</p>
       <span class="pm4d-card__meta">Slicer-Heart-CT</span>
     </a>
     <a class="pm4d-card" href="#tutorial-2-ct-segmentation-to-vtk-surfaces">
       <span class="pm4d-card__number">02</span>
       <h2>CT Segmentation to VTK Surfaces</h2>
       <p>Segment one CT phase and export patient anatomy as VTK PolyData surfaces.</p>
       <span class="pm4d-card__meta">Slicer-Heart-CT</span>
     </a>
     <a class="pm4d-card" href="#tutorial-3-create-a-pca-shape-model">
       <span class="pm4d-card__number">03</span>
       <h2>Create a PCA Shape Model</h2>
       <p>Build a statistical shape model from aligned cardiac meshes.</p>
       <span class="pm4d-card__meta">KCL-Heart-Model</span>
     </a>
     <a class="pm4d-card" href="#tutorial-4-fit-statistical-model-to-patient">
       <span class="pm4d-card__number">04</span>
       <h2>Fit Statistical Model to Patient</h2>
       <p>Fit a PCA heart model to patient-specific anatomy for model-based reconstruction.</p>
       <span class="pm4d-card__meta">Tutorial 3 output</span>
     </a>
     <a class="pm4d-card" href="#tutorial-5-vtk-surface-series-to-animated-usd">
       <span class="pm4d-card__number">05</span>
       <h2>VTK Surface Series to Animated USD</h2>
       <p>Convert VTK meshes into a time-sampled USD scene for Omniverse playback.</p>
       <span class="pm4d-card__meta">Tutorial 2 output</span>
     </a>
     <a class="pm4d-card" href="#tutorial-6-reconstruct-high-resolution-4d-ct">
       <span class="pm4d-card__number">06</span>
       <h2>Reconstruct High-Resolution 4D CT</h2>
       <p>Register respiratory CT phases and reconstruct a higher-resolution 4D volume series.</p>
       <span class="pm4d-card__meta">DirLab-4DCT</span>
     </a>
     <a class="pm4d-card" href="#tutorial-8-fit-the-cardiac-ssm-and-propagate-through-gated-phases">
       <span class="pm4d-card__number">08</span>
       <h2>Fit the Cardiac SSM and Propagate Through Gated Phases</h2>
       <p>Fit a PCA heart model to the reference phase and propagate it to every gated phase with ICON registration.</p>
       <span class="pm4d-card__meta">Bring your own cardiac data</span>
     </a>
     <a class="pm4d-card" href="#tutorial-9a-9b-train-a-physicsnemo-cardiac-stage-model">
       <span class="pm4d-card__number">09</span>
       <h2>Train a PhysicsNeMo Cardiac Stage Model</h2>
       <p>Train a PhysicsNeMo MeshGraphNet (9a) or MLP (9b) to predict cardiac meshes at requested stages.</p>
       <span class="pm4d-card__meta">Tutorial 8 output</span>
     </a>
     <a class="pm4d-card" href="#tutorial-10a-10b-predict-and-evaluate-cardiac-surfaces">
       <span class="pm4d-card__number">10</span>
       <h2>Predict and Evaluate Cardiac Surfaces</h2>
       <p>Load a Tutorial 9 checkpoint and predict cardiac surfaces at gated phases or caller-specified stages.</p>
       <span class="pm4d-card__meta">Tutorial 9a / 9b output</span>
     </a>
   </section>

Recommended Run Order
=====================

Tutorials are ``# %%`` percent-cell Python scripts. Each script defines its
data and output paths near the top, using repository ``data/`` and ``output/``
directories by default. Edit those constants for tutorial exploration, or use
the installed ``physiomotion4d-*`` CLI commands when you need command-line path
arguments.

1. Run Tutorials 1 and 2 after preparing Slicer-Heart-CT data.
2. Run Tutorial 5 after Tutorial 2 because it consumes Tutorial 2 output.
3. Run Tutorial 3 after downloading KCL-Heart-Model.
4. Run Tutorial 4 after Tutorial 3 because it can consume the PCA model output.
5. Run Tutorial 6 after downloading DirLab-4DCT.
6. Run Tutorial 8 after preparing your own cardiac gated CT, labelmaps, KCL
   volume PCA model, and ICON weights (bring-your-own-data; see the note below).
7. Run Tutorial 9a and/or 9b after Tutorial 8 because they train from its
   fitted meshes.
8. Run Tutorial 10a and/or 10b after Tutorial 9a / 9b because they evaluate
   the trained checkpoints.

Tutorial 1: Heart-Gated CT to Animated USD
==========================================

Script
   ``tutorials/tutorial_01_heart_gated_ct_to_usd.py``

Workflow
   ``WorkflowConvertImageToUSD``

Dataset
   Slicer-Heart-CT, prepared before running the tutorial.

Run
   .. code-block:: bash

      python tutorials/tutorial_01_heart_gated_ct_to_usd.py

Outputs
   Registered phase images, transformed contours, preview screenshots, and an
   animated USD model.

Tutorial 2: CT Segmentation to VTK Surfaces
===========================================

Script
   ``tutorials/tutorial_02_ct_to_vtk.py``

Workflow
   ``WorkflowConvertImageToVTK``

Dataset
   Slicer-Heart-CT, prepared before running the tutorial.

Run
   .. code-block:: bash

      python tutorials/tutorial_02_ct_to_vtk.py

Outputs
   Segmentation artifacts, VTK PolyData surfaces, and preview screenshots.

Tutorial 3: Create a PCA Shape Model
====================================

Script
   ``tutorials/tutorial_03_create_statistical_model.py``

Workflow
   ``WorkflowCreateStatisticalModel``

Dataset
   KCL-Heart-Model, downloaded manually.

Run
   .. code-block:: bash

      python tutorials/tutorial_03_create_statistical_model.py

Outputs
   PCA model files, mean shape, and component diagnostics.

Tutorial 4: Fit Statistical Model to Patient
============================================

Script
   ``tutorials/tutorial_04_fit_statistical_model_to_patient.py``

Workflow
   ``WorkflowFitStatisticalModelToPatient``

Dataset
   KCL-Heart-Model, downloaded manually.

Run
   .. code-block:: bash

      python tutorials/tutorial_04_fit_statistical_model_to_patient.py

Outputs
   Patient-fitted statistical model surfaces and registration diagnostics.

Tutorial 5: VTK Surface Series to Animated USD
==============================================

Script
   ``tutorials/tutorial_05_vtk_to_usd.py``

Workflow
   ``WorkflowConvertVTKToUSD``

Dataset
   Output from Tutorial 2.

Run
   .. code-block:: bash

      python tutorials/tutorial_05_vtk_to_usd.py

Outputs
   Time-sampled USD scene and conversion logs for Omniverse inspection.

Tutorial 6: Reconstruct High-Resolution 4D CT
=============================================

Script
   ``tutorials/tutorial_06_reconstruct_highres_4d_ct.py``

Workflow
   ``WorkflowReconstructHighres4DCT``

Dataset
   DirLab-4DCT, downloaded manually.

Run
   .. code-block:: bash

      python tutorials/tutorial_06_reconstruct_highres_4d_ct.py

Outputs
   Registered respiratory phases, reconstructed high-resolution CT volumes,
   and preview screenshots.

.. note::

   Tutorials 8-10 form the cardiac mesh stage-prediction pipeline and are
   **bring-your-own-data**: unlike Tutorials 1-6 they do not use the repository
   ``data/`` directory or a downloadable sample. Their path constants point at a
   local ``D:/PhysioMotion4D/`` cardiac layout (gated CT, labelmaps, the KCL
   volume PCA model, and ICON weights); edit those constants to match your own
   data. The former DirLab lung-lobe PCA tutorial (number 7) has been removed;
   numbering continues at 8.

Tutorial 8: Fit the Cardiac SSM and Propagate Through Gated Phases
==================================================================

Script
   ``tutorials/tutorial_08_cardiac_fit_model.py``

Workflow
   ``WorkflowFitStatisticalModelToPatient`` (PCA registration) and
   ``WorkflowReconstructHighres4DCT`` (ICON time-series registration)

Dataset
   Bring your own cardiac gated CT, labelmaps, KCL volume PCA model, and ICON
   weights under ``D:/PhysioMotion4D/``.

Run
   .. code-block:: bash

      python tutorials/tutorial_08_cardiac_fit_model.py

Outputs
   Per-patient fitted SSM mesh/surface, PCA coefficients, and the SSM warped to
   every gated phase, all written under ``OUTPUT_DIR``.

Tutorial 9a / 9b: Train a PhysicsNeMo Cardiac Stage Model
=========================================================

Script
   ``tutorials/tutorial_09a_cardiac_train_physicsnemo_mgn.py`` (MeshGraphNet) and
   ``tutorials/tutorial_09b_cardiac_train_physicsnemo_mlp.py`` (MLP)

Workflow
   ``physicsnemo.models.meshgraphnet.MeshGraphNet`` (9a) and
   ``physicsnemo.models.mlp.FullyConnected`` (9b), trained on Tutorial 8 meshes.

Dataset
   Tutorial 8 fitted-mesh outputs.

Extra install
   PhysicsNeMo is an optional dependency. Install with
   ``pip install "physiomotion4d[physicsnemo]"`` (requires Python >= 3.11). The
   MeshGraphNet variant also requires ``torch-geometric``.

Run
   .. code-block:: bash

      python tutorials/tutorial_09a_cardiac_train_physicsnemo_mgn.py
      python tutorials/tutorial_09b_cardiac_train_physicsnemo_mlp.py

Outputs
   Shared PhysicsNeMo checkpoints, training metadata, loss / RMSE histories, and
   held-out predictions written under each trainer's ``OUTPUT_DIR``.

Tutorial 10a / 10b: Predict and Evaluate Cardiac Surfaces
=========================================================

Script
   ``tutorials/tutorial_10a_cardiac_eval_physicsnemo_mgn.py`` (MeshGraphNet) and
   ``tutorials/tutorial_10b_cardiac_eval_physicsnemo_mlp.py`` (MLP)

Workflow
   Load a Tutorial 9 checkpoint and predict cardiac surfaces for one subject at
   each gated phase (with error statistics) or at caller-specified stages.

Dataset
   Tutorial 9a / 9b trained checkpoints plus the Tutorial 8 fitted meshes.

Run
   .. code-block:: bash

      python tutorials/tutorial_10b_cardiac_eval_physicsnemo_mlp.py pm0002 --epoch 5000 --out results/pm0002

   Run with no arguments to use the ``run_tutorial`` entry point and its
   ``DEFAULT_SUBJECT`` / ``DEFAULT_EPOCH`` constants.

Outputs
   Predicted ``.vtp`` surfaces per phase (with per-point error arrays when
   ground truth exists) and a ``statistics.csv`` error summary.

Dataset Notes
=============

The repository-level ``tutorials/README.md`` has the most detailed dataset
preparation notes. The tutorials are also exercised by ``tests/test_tutorials.py``
behind the ``--run-tutorials`` opt-in flag.
