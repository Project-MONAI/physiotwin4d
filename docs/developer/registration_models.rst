==================================
Model Registration Developer Guide
==================================

Model registration aligns template meshes to patient surfaces and masks. The
supported high-level entry point is
:class:`physiomotion4d.WorkflowFitStatisticalModelToPatient`.

Recommended Entry Point
=======================

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

Lower-Level Classes
===================

The workflow composes these lower-level registration classes:

* :class:`physiomotion4d.RegisterModelsICP`
* :class:`physiomotion4d.RegisterModelsICPITK`
* :class:`physiomotion4d.RegisterModelsDistanceMaps`
* :class:`physiomotion4d.RegisterModelsPCA`

Use these directly only when developing or testing a specific registration
stage. Their constructors and return dictionaries are documented in
:doc:`../api/model_registration/index`.

Development Notes
=================

* Prefer PyVista mesh objects at the public Python boundary.
* Convert volumetric meshes to surfaces before surface registration when needed.
* Treat ITK/PyVista coordinate transforms as high-risk and add focused tests.
* Keep synthetic test meshes small and deterministic.
* ``RegisterModelsPCA`` returns ``forward_point_transform`` /
  ``inverse_point_transform``. These are **point** transforms whose orientation
  is opposite to the image-registration transforms; see
  :doc:`transform_conventions` before applying them to images or meshes.

See Also
========

* :doc:`transform_conventions`
* :doc:`../api/model_registration/index`
* :doc:`workflows`
