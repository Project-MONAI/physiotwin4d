==========================
Workflow Development Guide
==========================

Workflow classes coordinate multiple processing steps behind a stable Python API
and, where useful, an installed CLI command.

Current Workflow Mapping
========================

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - CLI command
     - Workflow class
   * - ``physiotwin4d-convert-image-to-usd``
     - :class:`physiotwin4d.WorkflowConvertImageToUSD`
   * - ``physiotwin4d-convert-image-to-vtk``
     - :class:`physiotwin4d.WorkflowConvertImageToVTK`
   * - ``physiotwin4d-convert-vtk-to-usd``
     - :class:`physiotwin4d.WorkflowConvertVTKToUSD`
   * - ``physiotwin4d-create-statistical-model``
     - :class:`physiotwin4d.WorkflowCreateStatisticalModel`
   * - ``physiotwin4d-fit-statistical-model-to-patient``
     - :class:`physiotwin4d.WorkflowFitStatisticalModelToPatient`
   * - ``physiotwin4d-reconstruct-highres-4d-ct``
     - :class:`physiotwin4d.WorkflowReconstructHighres4DCT`
   * - ``physiotwin4d-train-physicsnemo``
     - :class:`physiotwin4d.WorkflowTrainPhysicsNeMo`
   * - ``physiotwin4d-infer-physicsnemo``
     - :class:`physiotwin4d.WorkflowInferPhysicsNeMo`
   * - ``physiotwin4d-convert-image-4d-to-3d``
     - :class:`physiotwin4d.ConvertImage4DTo3D` (a converter, not a workflow)
   * - ``physiotwin4d-download-data``
     - :class:`physiotwin4d.DataDownloadTools` (a utility, not a workflow)
   * - ``physiotwin4d-visualize-pca-modes``
     - Reads a ``pca_model.json`` directly; no workflow class

That is all eleven installed commands. Two workflow classes have no CLI
wrapper: :class:`physiotwin4d.WorkflowFinetuneICONRegistration` and
:class:`physiotwin4d.WorkflowEvaluateMovement`.

Workflow Example
================

.. code-block:: python

   from pathlib import Path

   import itk

   from physiotwin4d import RegisterImagesICON, WorkflowConvertImageToUSD

   frame_files = sorted(Path("data/Slicer-Heart-CT").glob("slice_???.mha"))
   time_series_images = [itk.imread(str(path)) for path in frame_files]

   workflow = WorkflowConvertImageToUSD(
       time_series_images=time_series_images,
       reference_image=time_series_images[0],
       output_directory="./results",
       usd_project_name="patient_001",
       registration_method=RegisterImagesICON(),
   )

   results = workflow.process()

Adding a Workflow
=================

1. Inherit from :class:`physiotwin4d.PhysioTwin4DBase`.
2. Keep the constructor explicit and typed.
3. Use ``self.log_info()`` and ``self.log_debug()`` for runtime status.
4. Keep file I/O behavior predictable and documented.
5. Add a CLI wrapper only when the workflow is useful from the command line.
6. Add focused tests using synthetic data where possible.
7. Run ``graphify update .`` after public API changes — methods added,
   modified, or removed.

Risk Areas
==========

Changes at the ITK-to-PyVista boundary, time-series transform direction, or
LPS-to-USD-Y-up coordinate conversion are high risk and should include focused
tests plus visual or metadata validation.

See Also
========

* :doc:`../api/workflows`
* :doc:`../cli_scripts/overview`
* :doc:`../architecture`
