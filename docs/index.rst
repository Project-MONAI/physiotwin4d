.. PhysioTwin4D documentation master file

.. title:: PhysioTwin4D Documentation

.. raw:: html

   <section class="pt4d-hero">
     <div class="pt4d-hero__brand">
       <img src="_static/nvidia-logo.svg" alt="NVIDIA logo">
     </div>
     <p class="pt4d-kicker">PhysioTwin4D tutorials</p>
     <h1>Build animated medical USD workflows for NVIDIA Omniverse</h1>
     <p>
       PhysioTwin4D is a collection of methods, workflows, tutorials, and CLI
       tools for creating personalized physiological digital twins from 3D
       medical images. Start with the tutorial cards, then use the
       documentation sections below for installation, CLI workflows, API
       references, developer notes, and contribution guidance.
     </p>

     <p class="pt4d-hero__version">Version {{ pt4d_project_version }}</p>
   </section>

   <section class="pt4d-card-grid" aria-label="Tutorial cards">
     <a class="pt4d-card" href="tutorials.html#tutorial-1a-heart-gated-ct-to-animated-usd">
       <span class="pt4d-card__number">1a</span>
       <h2>Heart-Gated CT to Animated USD</h2>
       <p>Convert cardiac 4D CT frames into registered contours and an animated OpenUSD model.</p>
       <span class="pt4d-card__meta">Slicer-Heart-CT</span>
     </a>
     <a class="pt4d-card" href="tutorials.html#tutorial-2-ct-segmentation-to-vtk-surfaces">
       <span class="pt4d-card__number">02</span>
       <h2>CT Segmentation to VTK Surfaces</h2>
       <p>Segment one CT phase and export patient anatomy as VTK PolyData surfaces.</p>
       <span class="pt4d-card__meta">Slicer-Heart-CT</span>
     </a>
     <a class="pt4d-card" href="tutorials.html#tutorial-3-vtk-surface-series-to-animated-usd">
       <span class="pt4d-card__number">03</span>
       <h2>VTK Surface Series to Animated USD</h2>
       <p>Convert VTK meshes into a time-sampled USD scene for Omniverse playback.</p>
       <span class="pt4d-card__meta">Tutorial 2 output</span>
     </a>
     <a class="pt4d-card" href="tutorials.html#tutorial-4a-create-a-pca-shape-model">
       <span class="pt4d-card__number">4a</span>
       <h2>Create a PCA Shape Model</h2>
       <p>Build a statistical shape model from aligned cardiac meshes.</p>
       <span class="pt4d-card__meta">KCL-Heart-Model</span>
     </a>
     <a class="pt4d-card" href="tutorials.html#tutorial-5a-fit-statistical-model-to-patient">
       <span class="pt4d-card__number">5a</span>
       <h2>Fit Statistical Model to Patient</h2>
       <p>Fit a PCA heart model to patient-specific anatomy for model-based reconstruction.</p>
       <span class="pt4d-card__meta">Tutorial 4a output</span>
     </a>
     <a class="pt4d-card" href="tutorials.html#tutorial-6-reconstruct-high-resolution-4d-ct">
       <span class="pt4d-card__number">06</span>
       <h2>Reconstruct High-Resolution 4D CT</h2>
       <p>Register respiratory CT phases and reconstruct a higher-resolution 4D volume series.</p>
       <span class="pt4d-card__meta">DirLab-4DCT</span>
     </a>
     <a class="pt4d-card" href="tutorials.html#tutorial-8cd-fit-the-cardiac-ssm-and-propagate-through-gated-phases">
       <span class="pt4d-card__number">8cd</span>
       <h2>Fit the Cardiac SSM and Propagate Through Gated Phases</h2>
       <p>Fit a PCA heart model to the reference phase and propagate it to every gated phase with ICON registration.</p>
       <span class="pt4d-card__meta">Bring your own cardiac data</span>
     </a>
     <a class="pt4d-card" href="tutorials.html#tutorial-9c-9d-train-a-physicsnemo-cardiac-stage-model">
       <span class="pt4d-card__number">9cd</span>
       <h2>Train a PhysicsNeMo Cardiac Stage Model</h2>
       <p>Train a PhysicsNeMo MeshGraphNet (9c) or MLP (9d) to predict cardiac meshes at requested stages.</p>
       <span class="pt4d-card__meta">Tutorial 8cd output</span>
     </a>
     <a class="pt4d-card" href="tutorials.html#tutorial-10c-10d-predict-and-evaluate-cardiac-surfaces">
       <span class="pt4d-card__number">10cd</span>
       <h2>Predict and Evaluate Cardiac Surfaces</h2>
       <p>Load a Tutorial 9c/9d checkpoint and predict cardiac surfaces at gated phases or caller-specified stages.</p>
       <span class="pt4d-card__meta">Tutorial 9c / 9d output</span>
     </a>
   </section>

   <section class="pt4d-topic-section" aria-label="Documentation topics">
     <div class="pt4d-section-heading">
       <p class="pt4d-kicker">Documentation</p>
       <h2>Explore the rest of the docs</h2>
     </div>
     <div class="pt4d-topic-grid">
       <a class="pt4d-topic-card" href="installation.html">
         <h3>Installation</h3>
         <p>Set up PhysioTwin4D with CUDA extras, CPU-only options, and required system tools.</p>
       </a>
       <a class="pt4d-topic-card" href="quickstart.html">
         <h3>Getting Started</h3>
         <p>Run your first workflow and understand the basic CT-to-USD processing path.</p>
       </a>
       <a class="pt4d-topic-card" href="tutorials.html">
         <h3>Tutorials &amp; Examples</h3>
         <p>Runnable scripts covering cardiac, lung, segmentation, and USD tasks, with the inner workflow-class calls each one makes.</p>
       </a>
       <a class="pt4d-topic-card" href="cli_scripts/overview.html">
         <h3>CLI Workflows</h3>
         <p>Use production command-line workflows for conversion, reconstruction, modeling, and USD export.</p>
       </a>
       <a class="pt4d-topic-card" href="isaac_for_healthcare.html">
         <h3>Isaac for Healthcare</h3>
         <p>Find PhysioTwin4D workflows and assets for Isaac for Healthcare use cases.</p>
       </a>
       <a class="pt4d-topic-card" href="api/index.html">
         <h3>API Reference</h3>
         <p>Browse classes and modules for workflows, segmentation, registration, USD, and utilities.</p>
       </a>
     <a class="pt4d-topic-card" href="developer/architecture.html">
       <h3>Developer Docs</h3>
       <p>Understand architecture, extension points, coordinate transforms, and implementation boundaries.</p>
     </a>
      <a class="pt4d-topic-card" href="architecture.html">
        <h3>Architecture</h3>
        <p>Trace the actual workflow classes and data flow from CT inputs to USD outputs.</p>
      </a>
       <a class="pt4d-topic-card" href="contributing.html">
         <h3>Contributing</h3>
         <p>Follow repository conventions for code style, testing, documentation, and pull requests.</p>
       </a>
       <a class="pt4d-topic-card" href="testing.html">
         <h3>Testing</h3>
         <p>Run the fast test suite, data-gated tutorial tests, and regression checks.</p>
       </a>
       <a class="pt4d-topic-card" href="troubleshooting.html">
         <h3>Troubleshooting</h3>
         <p>Diagnose environment, data, segmentation, registration, and USD playback issues.</p>
       </a>
     </div>
   </section>

Tutorial Details
================

See :doc:`tutorials` for the recommended run order, commands, datasets, and
per-tutorial implementation details.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   installation
   quickstart
   tutorials
   architecture

.. toctree::
   :maxdepth: 2
   :caption: CLI & Scripts Guide
   :hidden:

   cli_scripts/overview
   cli_scripts/download_data
   cli_scripts/heart_gated_ct
   cli_scripts/create_statistical_model
   cli_scripts/fit_statistical_model_to_patient
   cli_scripts/4dct_reconstruction
   cli_scripts/vtk_to_usd
   cli_scripts/best_practices

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   api/index

.. toctree::
   :maxdepth: 2
   :caption: Developer Guides
   :hidden:

   developer/architecture
   developer/extending
   developer/workflows
   developer/core
   developer/segmentation
   developer/registration_images
   developer/registration_models
   developer/transform_conventions
   developer/usd_generation
   developer/utilities

.. toctree::
   :maxdepth: 1
   :caption: Contributing
   :hidden:

   contributing
   testing

.. toctree::
   :maxdepth: 2
   :caption: Isaac for Healthcare
   :hidden:

   isaac_for_healthcare
   cli_scripts/byod_tutorials

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources
   :hidden:

   faq
   troubleshooting
   references

Clinical Use
============

Not validated for clinical use. PhysioTwin4D {{ pt4d_project_version }} beta
is a research and visualization toolkit, not a medical device. Do not use it
for diagnosis, treatment planning, or clinical decision-making.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
