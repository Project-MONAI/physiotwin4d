====================
API Reference
====================

Complete API documentation for PhysioTwin4D modules.

This section provides detailed documentation for all PhysioTwin4D classes, functions, and modules organized by functionality.

.. toctree::
   :maxdepth: 2
   :caption: Core Modules

   base
   workflows
   segmentation/index
   registration/index
   model_registration/index
   usd/index
   utilities/index
   cli/index

Quick Navigation
================

By Category
-----------

**Core Classes**
   * :class:`~physiotwin4d.PhysioTwin4DBase` - Base class for all components

**Workflows**
   * :class:`~physiotwin4d.WorkflowConvertImageToUSD` - Heart CT to USD
   * :class:`~physiotwin4d.WorkflowCreateStatisticalModel` - Create PCA statistical shape model
   * :class:`~physiotwin4d.WorkflowFitStatisticalModelToPatient` - Heart model registration

**Segmentation**
   * :class:`~physiotwin4d.SegmentAnatomyBase` - Base segmentation class
   * :class:`~physiotwin4d.SegmentChestTotalSegmentator` - TotalSegmentator
   * :class:`~physiotwin4d.SegmentHeartSimpleware` - Simpleware cardiac segmentation

**Image Registration**
   * :class:`~physiotwin4d.RegisterImagesBase` - Base registration class
   * :class:`~physiotwin4d.RegisterImagesANTS` - ANTs registration
   * :class:`~physiotwin4d.RegisterImagesICON` - Icon deep learning registration
   * :class:`~physiotwin4d.RegisterTimeSeriesImages` - 4D time series registration

**Model Registration**
   * :class:`~physiotwin4d.RegisterModelsICP` - Iterative Closest Point
   * :class:`~physiotwin4d.RegisterModelsICPITK` - ICP with ITK
   * :class:`~physiotwin4d.RegisterModelsDistanceMaps` - Distance map-based
   * :class:`~physiotwin4d.RegisterModelsPCA` - PCA-based registration

**USD Tools**
   * :mod:`~physiotwin4d.usd_tools` - USD file utilities
   * :mod:`~physiotwin4d.usd_anatomy_tools` - Anatomical structure tools
   * :class:`~physiotwin4d.ConvertVTKToUSD` - VTK to USD conversion

Module Index
============

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
