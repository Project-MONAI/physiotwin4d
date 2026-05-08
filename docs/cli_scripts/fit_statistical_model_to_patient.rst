====================================
Heart Model to Patient Registration
====================================

Overview
========

The ``physiomotion4d-fit-statistical-model-to-patient`` command-line tool registers generic anatomical heart models to patient-specific imaging data and surface models. This workflow enables:

* Patient-specific anatomical modeling from generic templates
* Multi-stage registration combining ICP, PCA, and deformable methods
* Transfer of model attributes to patient-specific geometry
* Model-based segmentation refinement using shape priors

The registration pipeline consists of four stages:

1. **ICP Alignment**: Rigid/affine alignment using surface matching
2. **PCA Registration** (optional): Statistical shape model fitting
3. **Mask-to-Mask Registration**: Deformable registration using distance maps
4. **Mask-to-Image Refinement** (optional): Final intensity-based refinement

Installation
============

The script is included with PhysioMotion4D installation:

.. code-block:: bash

   pip install physiomotion4d

Quick Start
===========

Basic Usage
-----------

Register a generic heart model to patient data:

.. code-block:: bash

   physiomotion4d-fit-statistical-model-to-patient \
       --template-model heart_model.vtu \
       --template-labelmap heart_labelmap.nii.gz \
       --patient-models lv.vtp rv.vtp myo.vtp \
       --patient-image patient_ct.nii.gz \
       --output-dir ./results

With PCA Shape Fitting
----------------------

Include statistical shape model fitting:

.. code-block:: bash

   physiomotion4d-fit-statistical-model-to-patient \
       --template-model heart_model.vtu \
       --template-labelmap heart_labelmap.nii.gz \
       --patient-models lv.vtp rv.vtp myo.vtp \
       --patient-image patient_ct.nii.gz \
       --pca-json pca_model.json \
       --pca-number-of-modes 10 \
       --output-dir ./results

Command-Line Arguments
======================

Required Arguments
------------------

``--template-model PATH``
   Path to template/generic heart model file (.vtu, .vtk, .stl)

``--patient-models PATH [PATH ...]``
   Paths to patient-specific surface models (e.g., lv.vtp rv.vtp myo.vtp)

``--output-dir DIR``
   Output directory for results

Optional inputs:

``--patient-image PATH``
   Path to patient CT/MRI image (.nii.gz, .nrrd, .mha). If omitted, the
   workflow creates a reference image from the patient surface models.

``--template-labelmap PATH``
   Path to template labelmap image (.nii.gz, .nrrd, .mha). Required only when
   ``--mask-to-image`` is set.

See :class:`physiomotion4d.WorkflowFitStatisticalModelToPatient` for API documentation.

Template Labelmap Configuration
--------------------------------

``--template-labelmap-muscle-ids ID [ID ...]``
   Label IDs for heart muscle in template labelmap (default: 1)

``--template-labelmap-chamber-ids ID [ID ...]``
   Label IDs for heart chambers in template labelmap (default: 2)

``--template-labelmap-background-ids ID [ID ...]``
   Label IDs for background in template labelmap (default: 0)

PCA Registration Options
-------------------------

``--pca-json PATH``
   Path to PCA JSON file for shape-based registration (optional)

``--pca-number-of-modes NUM``
   Number of PCA modes to use (default: 0, uses all if PCA enabled)

Registration Configuration
---------------------------

``--no-mask-to-mask``
   Disable mask-to-mask deformable registration (default: enabled)

``--mask-to-image``
   Enable mask-to-image refinement registration. Requires
   ``--template-labelmap`` and template label IDs. Disabled by default.

``--use-icon-refinement``
   Enable ICON deep learning registration refinement (default: disabled)

Output Options
--------------

``--output-prefix PREFIX``
   Prefix for output files (default: registered)

Related Scripts
===============

* :doc:`heart_gated_ct` - Process cardiac gated CT data
* :doc:`vtk_to_usd` - Convert model meshes to USD format

Examples
========

Example 1: Basic Registration
------------------------------

.. code-block:: bash

   physiomotion4d-fit-statistical-model-to-patient \
       --template-model heart_model.vtu \
       --template-labelmap heart_labelmap.nii.gz \
       --patient-models lv.vtp rv.vtp myo.vtp \
       --patient-image patient_ct.nii.gz \
       --output-dir results/basic

Example 2: PCA-Based Registration
----------------------------------

.. code-block:: bash

   physiomotion4d-fit-statistical-model-to-patient \
       --template-model heart_model.vtu \
       --template-labelmap heart_labelmap.nii.gz \
       --patient-models lv.vtp rv.vtp \
       --patient-image patient_ct.nii.gz \
       --pca-json pca_model.json \
       --pca-number-of-modes 10 \
       --output-dir results/pca

Output Files
============

Final Results
-------------

* ``{prefix}_model.vtu`` - Final registered volumetric model
* ``{prefix}_model_surface.vtp`` - Final registered surface mesh
* ``{prefix}_labelmap.nii.gz`` - Final registered labelmap

Intermediate Results
--------------------

* ``{prefix}_icp_surface.vtp`` - Result after ICP alignment
* ``{prefix}_pca_surface.vtp`` - Result after PCA fitting (if used)
* ``{prefix}_m2m_surface.vtp`` - Result after mask-to-mask registration

See Also
========

* :doc:`../api/workflows` - Workflow class API reference
* :doc:`heart_gated_ct` - Process cardiac gated CT data
* :doc:`overview` - CLI scripts overview
