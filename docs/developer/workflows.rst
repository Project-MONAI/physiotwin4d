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
   * - ``physiomotion4d-convert-image-to-usd``
     - :class:`physiomotion4d.WorkflowConvertImageToUSD`
   * - ``physiomotion4d-convert-image-to-vtk``
     - :class:`physiomotion4d.WorkflowConvertImageToVTK`
   * - ``physiomotion4d-convert-vtk-to-usd``
     - :class:`physiomotion4d.WorkflowConvertVTKToUSD`
   * - ``physiomotion4d-create-statistical-model``
     - :class:`physiomotion4d.WorkflowCreateStatisticalModel`
   * - ``physiomotion4d-fit-statistical-model-to-patient``
     - :class:`physiomotion4d.WorkflowFitStatisticalModelToPatient`
   * - ``physiomotion4d-reconstruct-highres-4d-ct``
     - :class:`physiomotion4d.WorkflowReconstructHighres4DCT`

Workflow Example
================

.. code-block:: python

   from physiomotion4d import WorkflowConvertImageToUSD

   workflow = WorkflowConvertImageToUSD(
       input_filenames=["cardiac_4d.nrrd"],
       contrast_enhanced=True,
       output_directory="./results",
       project_name="patient_001",
       registration_method="ANTS",
   )

   final_usd = workflow.process()

Adding a Workflow
=================

1. Inherit from :class:`physiomotion4d.PhysioMotion4DBase`.
2. Keep the constructor explicit and typed.
3. Use ``self.log_info()`` and ``self.log_debug()`` for runtime status.
4. Keep file I/O behavior predictable and documented.
5. Add a CLI wrapper only when the workflow is useful from the command line.
6. Add focused tests using synthetic data where possible.
7. Regenerate ``docs/API_MAP.md`` after exposing public methods.

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
