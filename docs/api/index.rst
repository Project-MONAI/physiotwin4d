====================
API Reference
====================

Complete API documentation for PhysioMotion4D modules.

This section provides detailed documentation for all PhysioMotion4D classes, functions, and modules organized by functionality.

.. toctree::
   :maxdepth: 2
   :caption: Core Modules

   base
   workflows

.. toctree::
   :maxdepth: 2
   :caption: Segmentation

   segmentation/index
   segmentation/base
   segmentation/totalsegmentator

.. toctree::
   :maxdepth: 2
   :caption: Image Registration

   registration/index
   registration/base
   registration/ants
   registration/icon
   registration/time_series

.. toctree::
   :maxdepth: 2
   :caption: Model Registration

   model_registration/index
   model_registration/icp
   model_registration/icp_itk
   model_registration/distance_maps
   model_registration/pca

.. toctree::
   :maxdepth: 2
   :caption: USD Generation

   usd/index
   usd/tools
   usd/anatomy_tools
   usd/vtk_conversion

.. toctree::
   :maxdepth: 2
   :caption: Utilities

   utilities/index
   utilities/image_tools
   utilities/transform_tools
   utilities/contour_tools
   utilities/nrrd_conversion

Quick Navigation
================

By Category
-----------

**Core Classes**
   * :class:`~physiomotion4d.PhysioMotion4DBase` - Base class for all components

**Workflows**
   * :class:`~physiomotion4d.WorkflowConvertHeartGatedCTToUSD` - Heart CT to USD
   * :class:`~physiomotion4d.WorkflowCreateStatisticalModel` - Create PCA statistical shape model
   * :class:`~physiomotion4d.WorkflowFitStatisticalModelToPatient` - Heart model registration

**Segmentation**
   * :class:`~physiomotion4d.SegmentAnatomyBase` - Base segmentation class
   * :class:`~physiomotion4d.SegmentChestTotalSegmentator` - TotalSegmentator
   * :class:`~physiomotion4d.SegmentHeartSimpleware` - Simpleware cardiac segmentation

**Image Registration**
   * :class:`~physiomotion4d.RegisterImagesBase` - Base registration class
   * :class:`~physiomotion4d.RegisterImagesANTs` - ANTs registration
   * :class:`~physiomotion4d.RegisterImagesICON` - Icon deep learning registration
   * :class:`~physiomotion4d.RegisterTimeSeriesImages` - 4D time series registration

**Model Registration**
   * :class:`~physiomotion4d.RegisterModelsICP` - Iterative Closest Point
   * :class:`~physiomotion4d.RegisterModelsICPITK` - ICP with ITK
   * :class:`~physiomotion4d.RegisterModelsDistanceMaps` - Distance map-based
   * :class:`~physiomotion4d.RegisterModelsPCA` - PCA-based registration

**USD Tools**
   * :mod:`~physiomotion4d.usd_tools` - USD file utilities
   * :mod:`~physiomotion4d.usd_anatomy_tools` - Anatomical structure tools
   * :class:`~physiomotion4d.ConvertVTKToUSD` - VTK to USD conversion

Module Index
============

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
