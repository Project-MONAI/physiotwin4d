====================================
Architecture Overview
====================================

This section provides an overview of PhysioMotion4D's architecture for Python developers looking to understand, use, or extend the package.

Target Audience
===============

This documentation is for:

* Python developers integrating PhysioMotion4D into applications
* Researchers extending or customizing processing methods
* Contributors adding new features or workflows
* Users needing programmatic control beyond CLI scripts

If you are a medical imaging expert looking to process data using command-line tools, see :doc:`../cli_scripts/overview`.

Design Philosophy
=================

PhysioMotion4D follows these core principles:

**Modularity**
   Each processing step (segmentation, registration, conversion) is a separate, reusable component

**Flexibility**
   Workflows are customizable; users can replace or extend individual steps

**Reproducibility**
   All processing steps are logged and parameterized for reproducible results

**Performance**
   GPU acceleration used where available; optimized for large medical imaging datasets

Package Organization
====================

The package is organized into functional modules:

.. code-block:: text

   physiomotion4d/
   ├── Core Classes
   │   └── physiomotion4d_base.py          Base class with common functionality
   │
   ├── Workflow Classes
   │   ├── workflow_convert_heart_gated_ct_to_usd.py      Cardiac CT → USD
   │   └── workflow_fit_statistical_model_to_patient.py    Model → Patient
   │
   ├── Segmentation
   │   ├── segment_anatomy_base.py                Base segmentation
   │   └── segment_chest_total_segmentator.py     TotalSegmentator
   │
   ├── Registration
   │   ├── Image Registration
   │   │   ├── register_images_base.py            Base registration
   │   │   ├── register_images_ants.py            ANTs registration
   │   │   ├── register_images_greedy.py          Greedy registration
   │   │   ├── register_images_icon.py            ICON registration
   │   │   └── register_time_series_images.py     Time series
   │   │
   │   └── Model Registration
   │       ├── register_models_icp.py             ICP registration
   │       ├── register_models_icp_itk.py         ITK ICP
   │       ├── register_models_distance_maps.py   Distance-based
   │       └── register_models_pca.py             PCA-based
   │
   ├── USD Generation
   │   ├── convert_vtk_to_usd_base.py          Base converter
   │   ├── convert_vtk_to_usd_polymesh.py      Polygon meshes
   │   ├── convert_vtk_to_usd_tetmesh.py       Tetrahedral meshes
   │   └── convert_vtk_to_usd.py               Main converter
   │
   └── Utilities
       ├── image_tools.py                          Image manipulation
       ├── transform_tools.py                      Spatial transforms
       ├── contour_tools.py                        Mesh/contour tools
       ├── usd_tools.py                            USD utilities
       ├── usd_anatomy_tools.py                    Anatomical materials
       └── convert_nrrd_4d_to_3d.py               Format conversion

Class Hierarchy
===============

Base Class Pattern
------------------

Most PhysioMotion4D classes inherit from :class:`PhysioMotion4DBase`:

.. code-block:: text

   PhysioMotion4DBase
   ├── Workflow Classes
   │   ├── WorkflowConvertHeartGatedCTToUSD
   │   └── WorkflowFitStatisticalModelToPatient
   ├── Segmentation Classes
   │   ├── SegmentAnatomyBase
   │   │   └── SegmentChestTotalSegmentator
   ├── Registration Classes
   │   ├── RegisterImagesBase
   │   │   ├── RegisterImagesANTs
   │   │   └── RegisterImagesICON
   │   └── (Model registration classes)
   └── Conversion Classes
       └── ConvertVTKToUSD

The base class provides:

* Logging infrastructure
* Parameter management
* File I/O utilities
* Common validation methods

See :doc:`core` for details on the base class.

Workflow Architecture
=====================

Workflows orchestrate multiple processing steps:

**Step-Based Execution**
   Workflows are divided into discrete, checkpointed steps

**Dependency Management**
   Each step declares its inputs and outputs

**State Persistence**
   Intermediate results are saved for resumption/debugging

**Flexible Configuration**
   Parameters control behavior at each step

Example workflow structure:

.. code-block:: python

   class MyWorkflow(PhysioMotion4DBase):

       def process(self):
           """Execute complete workflow."""
           # Step 1: Load data
           images = self.load_data()

           # Step 2: Segment
           segmentation = self.segment(images)

           # Step 3: Register
           transforms = self.register(images)

           # Step 4: Convert
           usd_file = self.convert_to_usd(segmentation, transforms)

           return usd_file

See :doc:`workflows` for detailed workflow documentation.

Data Flow
=========

Typical data flow through PhysioMotion4D:

.. code-block:: text

   4D Medical Image (NRRD)
           ↓
   [Load & Organize]
           ↓
   3D Image Time Series (MHA)
           ↓
   [Segmentation] → Labelmap (MHA)
           ↓
   [Registration] → Transform Fields (HDF5)
           ↓
   [Contour Extraction] → VTK Meshes
           ↓
   [Transform Application] → 4D VTK Meshes
           ↓
   [USD Conversion] → USD Files
           ↓
   Omniverse Visualization

File Formats
------------

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * - Format
     - Usage
     - Tools
   * - NRRD
     - Input 4D medical images
     - ITK, PyNRRD
   * - MHA/MHD
     - 3D images, segmentations
     - ITK, SimpleITK
   * - NII
     - Medical images (alternative)
     - NiBabel, ITK
   * - HDF5
     - Transform fields
     - ANTs, Icon
   * - VTK
     - Surface/volume meshes
     - VTK, PyVista
   * - USD
     - Omniverse scene files
     - OpenUSD, Pixar USD

Extension Points
================

PhysioMotion4D is designed for extension:

**Add New Segmentation Methods**
   Inherit from :class:`SegmentAnatomyBase`

**Add New Registration Methods**
   Inherit from :class:`RegisterImagesBase`

**Create Custom Workflows**
   Inherit from :class:`PhysioMotion4DBase`

**Customize USD Materials**
   Modify or extend :mod:`usd_anatomy_tools`

See :doc:`extending` for implementation guidance.

Relationship to CLI Scripts
============================

CLI commands are implemented in ``src/physiomotion4d/cli/`` as wrappers around workflow classes:

.. code-block:: text

   CLI Command: physiomotion4d-heart-gated-ct
           ↓ (implemented in)
   src/physiomotion4d/cli/convert_heart_gated_ct_to_usd.py
           ↓ (uses)
   src/physiomotion4d/workflow_convert_heart_gated_ct_to_usd.py
           ↓ (orchestrates)
   Segmentation, Registration, Conversion classes

This separation provides:

* **Simple CLI**: Easy-to-use command-line interface
* **Powerful API**: Full programmatic control
* **Reusability**: Workflow classes usable in custom applications

See :doc:`workflows` for mapping between scripts and workflow classes.

Dependencies
============

Core Dependencies
-----------------

* **PyTorch**: Deep learning models (segmentation, registration)
* **ITK**: Image I/O and basic processing
* **VTK**: Mesh processing and visualization
* **OpenUSD**: USD file creation and manipulation
* **NumPy**: Numerical computations

Optional Dependencies
---------------------

* **ANTsPy**: ANTs registration (alternative to Icon)
* **TotalSegmentator**: AI segmentation backend
* **MONAI**: Medical imaging AI framework
* **CuPy**: GPU-accelerated array operations

See ``pyproject.toml`` for complete dependency list.

Performance Considerations
==========================

GPU Acceleration
----------------

* Registration (Icon): 5-10x speedup with GPU
* Segmentation: Requires GPU for best performance
* Automatic GPU detection and fallback to CPU

Memory Management
-----------------

* Large medical images (512³+) require significant RAM
* Intermediate files cached to disk to reduce memory
* Streaming processing for very large datasets

Parallel Processing
-------------------

* Multiple patients can be processed in parallel
* Single patient processing is mostly sequential
* Some operations use multi-threading (ITK, VTK)

Next Steps
==========

* :doc:`core` - Understand the base class
* :doc:`workflows` - Learn about workflow classes
* :doc:`segmentation` - Explore segmentation methods
* :doc:`registration_images` - Study registration approaches
* :doc:`extending` - Create custom functionality
