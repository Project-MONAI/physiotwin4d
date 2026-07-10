================
Workflow Classes
================

.. module:: physiotwin4d.workflow_convert_image_to_usd
.. module:: physiotwin4d.workflow_convert_image_to_vtk
.. module:: physiotwin4d.workflow_convert_vtk_to_usd
.. module:: physiotwin4d.workflow_create_statistical_model
.. module:: physiotwin4d.workflow_fit_statistical_model_to_patient
.. module:: physiotwin4d.workflow_reconstruct_highres_4d_ct
.. currentmodule:: physiotwin4d

Workflow classes are the highest-level Python API in PhysioTwin4D. They
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

   from physiotwin4d import (
       RegisterImagesICON,
       SegmentChestTotalSegmentatorWithContrast,
       WorkflowConvertImageToUSD,
   )

   workflow = WorkflowConvertImageToUSD(
       input_filenames=["cardiac_4d.nrrd"],
       output_directory="./results",
       usd_project_name="patient_001",
       segmentation_method=SegmentChestTotalSegmentatorWithContrast(),
       registration_method=RegisterImagesICON(),
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

   from physiotwin4d import (
       ContourTools,
       SegmentChestTotalSegmentatorWithContrast,
       WorkflowConvertImageToVTK,
   )

   image = itk.imread("chest_ct.nii.gz")
   workflow = WorkflowConvertImageToVTK(
       segmentation_method=SegmentChestTotalSegmentatorWithContrast()
   )
   result = workflow.process(
       input_image=image,
       anatomy_groups=["heart", "major_vessels"],
   )

   ContourTools.save_combined_surface(
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

   import pyvista as pv
   from physiotwin4d import WorkflowConvertVTKToUSD

   input_meshes = [pv.read("heart_000.vtp"), pv.read("heart_001.vtp")]
   workflow = WorkflowConvertVTKToUSD(
       input_meshes=input_meshes,
       usd_project_name="heart",
       output_directory="./output",
       appearance="anatomy",
       anatomy_type="heart",
   )

   output_path = workflow.process()

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

   from physiotwin4d import WorkflowFitStatisticalModelToPatient

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

   from physiotwin4d import RegisterImagesGreedyICON, WorkflowReconstructHighres4DCT

   time_series_images = [itk.imread(f"phase_{idx:02d}.mha") for idx in range(10)]
   workflow = WorkflowReconstructHighres4DCT(
       time_series_images=time_series_images,
       fixed_image=time_series_images[0],
       reference_frame=0,
       registration_method=RegisterImagesGreedyICON(),
   )

   workflow.set_upsample_to_fixed_resolution(True)
   result = workflow.run_workflow()

See Also
========

* :doc:`../tutorials`
* :doc:`../cli_scripts/overview`
* :doc:`../architecture`
