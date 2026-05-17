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
     <a class="pm4d-card" href="#tutorial-7-dirlab-lung-lobe-pca-model">
       <span class="pm4d-card__number">07</span>
       <h2>DirLab Lung-Lobe PCA Model</h2>
       <p>Build a surface PCA model from five lung lobes and fit it to all cases.</p>
       <span class="pm4d-card__meta">DirLab-4DCT</span>
     </a>
     <a class="pm4d-card" href="#tutorial-8-dirlab-pca-time-series-propagation">
       <span class="pm4d-card__number">08</span>
       <h2>DirLab PCA Time-Series Propagation</h2>
       <p>Register respiratory phases with ANTs+ICON and propagate fitted meshes.</p>
       <span class="pm4d-card__meta">Tutorial 7 output</span>
     </a>
     <a class="pm4d-card" href="#tutorial-9-physicsnemo-mesh-stage-model">
       <span class="pm4d-card__number">09</span>
       <h2>PhysicsNeMo Mesh Stage Model</h2>
       <p>Train a PhysicsNeMo MLP to predict lung-lobe meshes at requested stages.</p>
       <span class="pm4d-card__meta">Tutorial 8 output</span>
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
6. Run Tutorial 7 after downloading DirLab-4DCT.
7. Run Tutorial 8 after Tutorial 7 because it consumes fitted PCA meshes.
8. Run Tutorial 9 after Tutorial 8 because it trains from propagated meshes.

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

Tutorial 7: DirLab Lung-Lobe PCA Model
======================================

Script
   ``tutorials/tutorial_07_dirlab_pca_model.py``

Workflow
   ``WorkflowConvertImageToVTK``, ``WorkflowCreateStatisticalModel``, and
   ``WorkflowFitStatisticalModelToPatient``

Dataset
   DirLab-4DCT, downloaded manually.

Run
   .. code-block:: bash

      python tutorials/tutorial_07_dirlab_pca_model.py

Outputs
   Five-lobe lung surface meshes, a surface PCA model, and PCA-fitted surfaces
   for every available case.

Tutorial 8: DirLab PCA Time-Series Propagation
==============================================

Script
   ``tutorials/tutorial_08_dirlab_pca_time_series.py``

Workflow
   ``RegisterTimeSeriesImages`` with ``registration_method='ants_icon'`` and
   ``TransformTools``

Dataset
   DirLab-4DCT plus Tutorial 7 fitted mesh outputs.

Run
   .. code-block:: bash

      python tutorials/tutorial_08_dirlab_pca_time_series.py

Outputs
   Per-case ANTs+ICON transforms and one PCA-fitted lung-lobe surface for each
   DirLab respiratory phase.

Tutorial 9: PhysicsNeMo Mesh Stage Model
========================================

Script
   ``tutorials/tutorial_09_physicsnemo_mesh_stage_model.py``

Workflow
   ``physicsnemo.models.mlp.FullyConnected`` trained on Tutorial 8 meshes.

Dataset
   Tutorial 8 propagated PCA mesh outputs.

Run
   .. code-block:: bash

      python tutorials/tutorial_09_physicsnemo_mesh_stage_model.py

Outputs
   Per-case PhysicsNeMo checkpoints, training metadata, loss histories, and a
   predicted PCA-fitted mesh at the requested normalized respiratory stage.

Dataset Notes
=============

The repository-level ``tutorials/README.md`` has the most detailed dataset
preparation notes. The tutorials are also exercised by ``tests/test_tutorials.py``
behind the ``--run-tutorials`` opt-in flag.
