============
Architecture
============

PhysioMotion4D is organized around explicit workflow classes and smaller
registration, segmentation, geometry, and USD utilities. Runtime workflow
classes inherit from :class:`PhysioMotion4DBase` for logging and common runtime
configuration.

.. warning::

   PhysioMotion4D 2026.05.07 beta is not validated for clinical use. It is a
   research and visualization toolkit, not a medical device.

Data Flow
=========

.. code-block:: text

   4D CT / time-series CT
          |
          v
   ConvertImage4DTo3D / ImageTools
          |
          v
   RegisterTimeSeriesImages
      |        |
      |        +--> RegisterImagesANTs / RegisterImagesICON
      v
   SegmentChestTotalSegmentator / SegmentHeartSimpleware
          |
          v
   ContourTools + TransformTools
          |
          v
   WorkflowConvertImageToVTK / ConvertVTKToUSD / WorkflowConvertVTKToUSD
          |
          v
   OpenUSD assets for NVIDIA Omniverse

Primary Workflows
=================

``WorkflowConvertImageToUSD``
   Converts a 4D cardiac CT file or 3D CT time series into registered anatomy
   contours and painted animated USD files.

``WorkflowConvertImageToVTK``
   Segments a 3D CT image and exports anatomy groups as VTK surfaces and voxel
   meshes.

``WorkflowCreateStatisticalModel``
   Aligns a population of meshes to a reference and builds a PCA statistical
   shape model.

``WorkflowFitStatisticalModelToPatient``
   Fits a template/statistical model to patient-specific surfaces with ICP,
   optional PCA fitting, mask-to-mask registration, and optional image
   refinement.

``WorkflowReconstructHighres4DCT``
   Reconstructs higher-resolution 4D CT frames from a time series and a fixed
   high-resolution reference image.

``WorkflowConvertVTKToUSD``
   Converts VTK files to animated USD scenes through the supported workflow
   wrapper. The lower-level :mod:`physiomotion4d.vtk_to_usd` package exposes
   advanced file conversion primitives.

Component Boundaries
====================

Segmentation classes produce anatomy masks or labelmaps from ITK images.
Registration classes produce ITK transforms or transformed meshes. Geometry
utilities bridge ITK masks and PyVista meshes. USD tools are responsible for
OpenUSD stage creation, material assignment, coordinate conversion, and time
samples.

The high-risk boundary is the ITK-to-PyVista-to-USD path. Image data remains in
ITK's native LPS world space until contours are extracted. Meshes are
represented as PyVista objects (still in LPS) before USD export. The VTK-to-USD
layer applies the repository's LPS-to-USD-Y-up coordinate transform during USD
conversion.

CLI Boundary
============

The installed CLI commands in ``pyproject.toml`` are thin wrappers around the
workflow classes. They are the preferred examples for executable API usage:

* ``physiomotion4d-convert-image-to-usd``
* ``physiomotion4d-convert-image-to-vtk``
* ``physiomotion4d-create-statistical-model``
* ``physiomotion4d-fit-statistical-model-to-patient``
* ``physiomotion4d-convert-vtk-to-usd``
* ``physiomotion4d-reconstruct-highres-4d-ct``
* ``physiomotion4d-visualize-pca-modes``
