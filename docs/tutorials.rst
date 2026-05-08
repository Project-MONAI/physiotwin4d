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
       Six focused tutorials walk through CT segmentation, registration,
       statistical model fitting, high-resolution 4D reconstruction, and USD
       export. Each card links to implementation details, datasets, and the
       command used to run the workflow.
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
     <a class="pm4d-card" href="#tutorial-3-fit-statistical-model-to-patient">
       <span class="pm4d-card__number">03</span>
       <h2>Fit Statistical Model to Patient</h2>
       <p>Fit a PCA heart model to patient-specific anatomy for model-based reconstruction.</p>
       <span class="pm4d-card__meta">KCL-Heart-Model</span>
     </a>
     <a class="pm4d-card" href="#tutorial-4-create-a-pca-shape-model">
       <span class="pm4d-card__number">04</span>
       <h2>Create a PCA Shape Model</h2>
       <p>Build a statistical shape model from aligned cardiac meshes.</p>
       <span class="pm4d-card__meta">KCL-Heart-Model</span>
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
   </section>

Recommended Run Order
=====================

1. Run Tutorials 1 and 2 after preparing Slicer-Heart-CT data.
2. Run Tutorial 5 after Tutorial 2 because it consumes Tutorial 2 output.
3. Run Tutorials 3 and 4 after downloading KCL-Heart-Model.
4. Run Tutorial 6 after downloading DirLab-4DCT.

Tutorial 1: Heart-Gated CT to Animated USD
==========================================

Script
   ``tutorials/tutorial_01_heart_gated_ct_to_usd.py``

Workflow
   ``WorkflowConvertHeartGatedCTToUSD``

Dataset
   Slicer-Heart-CT, prepared before running the tutorial.

Command
   .. code-block:: bash

      python tutorials/tutorial_01_heart_gated_ct_to_usd.py \
          --data-dir ./data --output-dir ./output/tutorial_01 \
          --registration-method ants

Outputs
   Registered phase images, transformed contours, preview screenshots, and an
   animated USD model.

Tutorial 2: CT Segmentation to VTK Surfaces
===========================================

Script
   ``tutorials/tutorial_02_ct_to_vtk.py``

Workflow
   ``WorkflowConvertCTToVTK``

Dataset
   Slicer-Heart-CT, prepared before running the tutorial.

Command
   .. code-block:: bash

      python tutorials/tutorial_02_ct_to_vtk.py \
          --data-dir ./data --output-dir ./output/tutorial_02

Outputs
   Segmentation artifacts, VTK PolyData surfaces, and preview screenshots.

Tutorial 3: Fit Statistical Model to Patient
============================================

Script
   ``tutorials/tutorial_03_fit_statistical_model_to_patient.py``

Workflow
   ``WorkflowFitStatisticalModelToPatient``

Dataset
   KCL-Heart-Model, downloaded manually.

Command
   .. code-block:: bash

      python tutorials/tutorial_03_fit_statistical_model_to_patient.py \
          --data-dir ./data --output-dir ./output/tutorial_03

Outputs
   Patient-fitted statistical model surfaces and registration diagnostics.

Tutorial 4: Create a PCA Shape Model
====================================

Script
   ``tutorials/tutorial_04_create_statistical_model.py``

Workflow
   ``WorkflowCreateStatisticalModel``

Dataset
   KCL-Heart-Model, downloaded manually.

Command
   .. code-block:: bash

      python tutorials/tutorial_04_create_statistical_model.py \
          --data-dir ./data --output-dir ./output/tutorial_04

Outputs
   PCA model files, mean shape, and component diagnostics.

Tutorial 5: VTK Surface Series to Animated USD
==============================================

Script
   ``tutorials/tutorial_05_vtk_to_usd.py``

Workflow
   ``WorkflowConvertVTKToUSD``

Dataset
   Output from Tutorial 2.

Command
   .. code-block:: bash

      python tutorials/tutorial_05_vtk_to_usd.py \
          --data-dir ./data --output-dir ./output/tutorial_05 \
          --vtk-file output/tutorial_02/patient_surfaces.vtp

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

Command
   .. code-block:: bash

      python tutorials/tutorial_06_reconstruct_highres_4d_ct.py \
          --data-dir ./data --output-dir ./output/tutorial_06

Outputs
   Registered respiratory phases, reconstructed high-resolution CT volumes,
   and preview screenshots.

Dataset Notes
=============

The repository-level ``tutorials/README.md`` has the most detailed dataset
preparation notes. The tutorials are also exercised by ``tests/test_tutorials.py``
behind the experiment marker.
