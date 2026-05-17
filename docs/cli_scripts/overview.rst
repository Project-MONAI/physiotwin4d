====================================
CLI & Scripts Overview
====================================

This section provides comprehensive guides for using PhysioMotion4D's command-line tools to process medical imaging data. These tools are designed for medical imaging experts and physiological simulation researchers who need efficient, reproducible pipelines for converting 4D medical images into dynamic anatomical models for NVIDIA Omniverse.

How to Use These Resources
==========================

PhysioMotion4D exposes the same toolkit through three user-facing layers:

* **Workflows** are Python classes that orchestrate complete processing
  pipelines. Use them when integrating PhysioMotion4D into Python applications
  or when you need programmatic control over inputs, outputs, and parameters.
* **CLIs** are installed command-line wrappers around workflow classes. Use them
  for repeatable processing runs, batch jobs, and environment validation without
  writing Python glue code.
* **Tutorials** are repository scripts that demonstrate each major workflow with
  concrete data preparation, commands, and expected outputs. Use them when first
  learning the toolkit or validating a local installation.

The ``experiments/`` directory tracks prior and ongoing research experiments
that helped define this toolkit. Those experiments are useful historical and
design context, but they are not intended to be examples for users or
developers. For supported usage patterns, start with the tutorials, CLIs, and
workflow API documentation.

Target Audience
===============

These CLI tools are intended for users with:

* Strong medical image analysis expertise
* Understanding of physiological simulation requirements
* Modest Python experience for running scripts
* Familiarity with command-line interfaces

If you are a Python developer looking to extend or integrate PhysioMotion4D into your applications, please refer to the :doc:`../developer/architecture` section.

Available Scripts
=================

Current Scripts
---------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Script
     - Description
   * - :doc:`heart_gated_ct`
     - Process cardiac gated CT to animated heart models with physiological motion
   * - ``physiomotion4d-convert-image-to-vtk``
     - Segment one 3D image and export anatomy-group VTK surfaces and meshes
   * - ``physiomotion4d-convert-image-4d-to-3d``
     - Split a 4D medical image into a 3D time series using ITK readers
   * - :doc:`create_statistical_model`
     - Build a PCA statistical shape model from sample meshes aligned to a reference
   * - :doc:`fit_statistical_model_to_patient`
     - Register generic heart models to patient-specific imaging data and surface models
   * - :doc:`4dct_reconstruction`
     - Reconstruct high-resolution 4D CT from time-series images and a reference
   * - :doc:`vtk_to_usd`
     - Convert VTK anatomical models to USD format with material painting
   * - ``physiomotion4d-visualize-pca-modes``
     - Render PCA model mode visualizations

Installation
============

All scripts are installed with the PhysioMotion4D package:

.. code-block:: bash

   pip install physiomotion4d

After installation, scripts are available as command-line tools with the prefix ``physiomotion4d-``:

.. code-block:: bash

   physiomotion4d-convert-image-to-usd --help

General Workflow
================

All PhysioMotion4D scripts follow a similar pattern:

1. **Input Data**: Provide medical image files (NRRD, NII, MHA formats)
2. **Configuration**: Set processing parameters via command-line flags
3. **Processing Pipeline**: Automated execution of segmentation, registration, and conversion
4. **Output Generation**: USD files ready for Omniverse visualization

Typical Command Structure
--------------------------

.. code-block:: bash

   physiomotion4d-<command> --help

Use each command's ``--help`` output as the source of truth for required
arguments and script-specific options.

Output Organization
-------------------

Each script organizes outputs in a consistent structure:

.. code-block:: text

   output_directory/
   ├── intermediate/          # Segmentations, registrations
   ├── meshes/               # VTK mesh files
   └── usd/                  # Final USD files for Omniverse

Getting Help
============

Each script provides detailed help:

.. code-block:: bash

   physiomotion4d-<script-name> --help

For troubleshooting and common issues, see :doc:`../troubleshooting`.

Next Steps
==========

* Start with :doc:`heart_gated_ct` for a complete example
* See :doc:`best_practices` for optimization tips
* Refer to script-specific pages for detailed usage
