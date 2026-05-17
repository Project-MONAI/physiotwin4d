================
Workflow Classes
================

.. module:: physiomotion4d.workflow_convert_image_to_usd
.. module:: physiomotion4d.workflow_convert_image_to_vtk
.. module:: physiomotion4d.workflow_convert_vtk_to_usd
.. module:: physiomotion4d.workflow_create_statistical_model
.. module:: physiomotion4d.workflow_fit_statistical_model_to_patient
.. module:: physiomotion4d.workflow_reconstruct_highres_4d_ct
.. currentmodule:: physiomotion4d

Workflow classes are the highest-level Python API in PhysioMotion4D. They
combine segmentation, registration, contour generation, and USD conversion into
repeatable pipelines. The installed CLI commands are thin wrappers around these
classes.

Available Workflows
===================

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Workflow
     - Typical use
   * - :class:`WorkflowConvertImageToUSD`
     - Convert a 4D cardiac CT sequence into animated USD anatomy.
   * - :class:`WorkflowConvertImageToVTK`
     - Segment one CT image and export anatomy-group VTK surfaces and meshes.
   * - :class:`WorkflowConvertVTKToUSD`
     - Convert VTK/VTP/VTU meshes or time series into USD.
   * - :class:`WorkflowCreateStatisticalModel`
     - Build a PCA shape model from sample meshes aligned to a reference.
   * - :class:`WorkflowFitStatisticalModelToPatient`
     - Fit a template/statistical heart model to patient-specific surfaces.
   * - :class:`WorkflowReconstructHighres4DCT`
     - Reconstruct a high-resolution 4D CT series from phase images and a
       high-resolution reference.

Convert Image to USD
====================

.. autoclass:: WorkflowConvertImageToUSD
   :members:
   :undoc-members:
   :show-inheritance:

.. code-block:: python

   from physiomotion4d import WorkflowConvertImageToUSD

   workflow = WorkflowConvertImageToUSD(
       input_filenames=["cardiac_4d.nrrd"],
       contrast_enhanced=True,
       output_directory="./results",
       project_name="patient_001",
       segmentation_method="ChestTotalSegmentator",
       registration_method="ANTS",
   )

   final_usd = workflow.process()

Image to VTK
============

.. autoclass:: WorkflowConvertImageToVTK
   :members:
   :undoc-members:
   :show-inheritance:

.. code-block:: python

   import itk

   from physiomotion4d import WorkflowConvertImageToVTK

   image = itk.imread("chest_ct.nii.gz")
   workflow = WorkflowConvertImageToVTK(segmentation_method="ChestTotalSegmentator")
   result = workflow.run_workflow(
       input_image=image,
       contrast_enhanced_study=True,
       anatomy_groups=["heart", "major_vessels"],
   )

   WorkflowConvertImageToVTK.save_combined_surface(
       result["surfaces"],
       "./output",
       prefix="patient01",
   )

VTK to USD
==========

.. autoclass:: WorkflowConvertVTKToUSD
   :members:
   :undoc-members:
   :show-inheritance:

.. code-block:: python

   from physiomotion4d import WorkflowConvertVTKToUSD

   workflow = WorkflowConvertVTKToUSD(
       vtk_files=["heart_000.vtp", "heart_001.vtp"],
       output_usd="heart.usd",
       appearance="anatomy",
       anatomy_type="heart",
   )

   output_path = workflow.run()

Statistical Shape Modeling
==========================

.. autoclass:: WorkflowCreateStatisticalModel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: WorkflowFitStatisticalModelToPatient
   :members:
   :undoc-members:
   :show-inheritance:

.. code-block:: python

   import itk
   import pyvista as pv

   from physiomotion4d import WorkflowFitStatisticalModelToPatient

   workflow = WorkflowFitStatisticalModelToPatient(
       template_model=pv.read("template_heart.vtu"),
       patient_models=[pv.read("lv.vtp"), pv.read("rv.vtp")],
       patient_image=itk.imread("patient_ct.nii.gz"),
   )

   result = workflow.run_workflow()

High-Resolution 4D CT Reconstruction
====================================

.. autoclass:: WorkflowReconstructHighres4DCT
   :members:
   :undoc-members:
   :show-inheritance:

.. code-block:: python

   import itk

   from physiomotion4d import WorkflowReconstructHighres4DCT

   time_series_images = [itk.imread(f"phase_{idx:02d}.mha") for idx in range(10)]
   workflow = WorkflowReconstructHighres4DCT(
       time_series_images=time_series_images,
       fixed_image=time_series_images[0],
       reference_frame=0,
       registration_method="ants",
   )

   result = workflow.run_workflow(upsample_to_fixed_resolution=True)

See Also
========

* :doc:`../tutorials`
* :doc:`../cli_scripts/overview`
* :doc:`../architecture`
