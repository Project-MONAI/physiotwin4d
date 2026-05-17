===========
Quick Start
===========

This guide will help you get started with PhysioMotion4D quickly.

.. warning::

   **Not validated for clinical use.** PhysioMotion4D 2026.05.07 beta is a
   research and visualization toolkit, not a medical device. Do not use it for
   diagnosis, treatment planning, or clinical decision-making.

.. _tutorials:

Tutorials
=========

The ``tutorials/`` directory contains nine end-to-end scripts, one per major
workflow. Each script is a ``# %%`` percent-cell Python script that exercises
the workflow classes directly. Run as a regular file
(``python tutorials/tutorial_01_...py``) or cell-by-cell in VS Code or Cursor.

See :doc:`tutorials` for the NVIDIA-styled tutorial card index, dataset
requirements, script paths, and workflow details.

After preparing the Slicer-Heart-CT data, run the first two tutorials:

.. code-block:: bash

   python tutorials/tutorial_01_heart_gated_ct_to_usd.py

   python tutorials/tutorial_02_ct_to_vtk.py

Tutorial paths are defined near the top of each script. To use different paths,
edit the script constants or use the installed ``physiomotion4d-*`` CLI commands.
See ``tutorials/README.md`` for dataset download instructions and the
recommended run order.

Recommended run order:

1. Tutorials 1 and 2 first, after preparing Slicer-Heart-CT data.
2. Tutorial 5 after Tutorial 2 (consumes Tutorial 2 output).
3. Tutorial 3 after downloading KCL-Heart-Model.
4. Tutorial 4 after Tutorial 3 because it can consume Tutorial 3 output.
5. Tutorial 6 after downloading DirLab-4DCT.
6. Tutorial 7 after downloading DirLab-4DCT.
7. Tutorial 8 after Tutorial 7 because it consumes the fitted PCA meshes.
8. Tutorial 9 after Tutorial 8 because it trains from propagated meshes.

Prerequisites
=============

Before starting, ensure you have:

* PhysioMotion4D installed (see :doc:`installation`)
* NVIDIA GPU with CUDA 13 - recommended for production performance; see :doc:`installation` for the ``[cuda13]`` extra. A CPU-only PyPI install works for evaluation but is slow.
* 4D cardiac CT data or access to sample datasets

Basic Workflow
==============

Minimal Slicer-Heart Quickstart
-------------------------------

The public Slicer-Heart 4D CT sample can be downloaded automatically and used
as the smallest end-to-end cardiac workflow. Data downloading and a
CUDA-capable GPU are required for practical runtime.

.. code-block:: bash

   python -c "from physiomotion4d import DataDownloadTools; DataDownloadTools.DownloadSlicerHeartCTData('data/test')"

   physiomotion4d-convert-image-to-usd data/test/TruncalValve_4DCT.seq.nrrd \
       --registration-method ANTS \
       --output-dir output/quickstart \
       --project-name slicer_heart_quickstart

Command-Line Interface
----------------------

The fastest way to process cardiac CT data is using the command-line interface:

.. code-block:: bash

   # Process a single 4D cardiac CT file
   physiomotion4d-convert-image-to-usd cardiac_4d.nrrd --contrast --output-dir ./results

   # Process multiple time frames
   physiomotion4d-convert-image-to-usd frame_*.nrrd --contrast --project-name patient_001

   # With custom settings
   physiomotion4d-convert-image-to-usd cardiac.nrrd \
       --contrast \
       --reference-image ref.mha \
       --registration-iterations 50 \
       --output-dir ./output

Python API
----------

For more control, use the Python API:

**Step 1: Import the processor**

.. code-block:: python

   from physiomotion4d import WorkflowConvertImageToUSD

**Step 2: Initialize with your data**

.. code-block:: python

   processor = WorkflowConvertImageToUSD(
       input_filenames=["path/to/cardiac_4d_ct.nrrd"],
       contrast_enhanced=True,
       output_directory="./results",
       project_name="cardiac_model",
       registration_method="ANTS",
   )

**Step 3: Run the workflow**

.. code-block:: python

   # Run complete workflow
   final_usd = processor.process()

   print(f"USD model saved to: {final_usd}")

That's it! The processor will:

1. Convert 4D NRRD to 3D time frames
2. Perform image registration between phases
3. Generate AI-based segmentation
4. Transform contours across time
5. Create animated USD model

Step-by-Step Workflow
======================

For more control over individual steps:

.. code-block:: python

   from physiomotion4d import WorkflowConvertImageToUSD

   # Initialize workflow
   workflow = WorkflowConvertImageToUSD(
       input_filenames=["cardiac_4d.nrrd"],
       contrast_enhanced=True,
       output_directory="./results",
       project_name="cardiac_model",
       registration_method="ANTS",
   )

   final_usd = workflow.process()

Working with Individual Components
===================================

Segmentation Only
-----------------

If you only need segmentation:

.. code-block:: python

   from physiomotion4d import SegmentChestTotalSegmentator
   import itk

   # Initialize segmenter
   segmenter = SegmentChestTotalSegmentator()

   # Load and segment image
   image = itk.imread("chest_ct.nrrd")
   masks = segmenter.segment(image, contrast_enhanced_study=True)

   # Extract individual anatomy masks by key
   heart_mask = masks["heart"]
   vessels_mask = masks["major_vessels"]
   lungs_mask = masks["lung"]
   labelmap = masks["labelmap"]

   # Save results
   itk.imwrite(heart_mask, "heart_mask.nrrd")
   itk.imwrite(labelmap, "labelmap.nrrd")

Image Registration Only
-----------------------

For standalone registration:

.. code-block:: python

   from physiomotion4d.register_images_icon import RegisterImagesICON
   import itk

   # Initialize registration
   registerer = RegisterImagesICON()

   # Load images
   fixed_image = itk.imread("reference_frame.mha")
   moving_image = itk.imread("target_frame.mha")

   # Configure registration
   registerer.set_modality('ct')
   registerer.set_fixed_image(fixed_image)

   # Perform registration
   results = registerer.register(moving_image)

   # Get transformation fields
   inverse_transform = results["inverse_transform"]  # Fixed to moving space
   forward_transform = results["forward_transform"]  # Moving to fixed space

VTK to USD Conversion
---------------------

Convert VTK time series to USD:

.. code-block:: python

   from physiomotion4d import ConvertVTKToUSD

   vtk_files = [f"heart_frame_{i:03d}.vtp" for i in range(10)]
   time_codes = [float(i) for i in range(len(vtk_files))]

   stage = ConvertVTKToUSD.from_files(
       data_basename="Heart",
       vtk_files=vtk_files,
       time_codes=time_codes,
   ).convert("heart_animation.usd")

Sample Data
===========

Download Sample Cardiac CT Data
--------------------------------

.. code-block:: python

   from physiomotion4d import DataDownloadTools

   data_file = DataDownloadTools.DownloadSlicerHeartCTData("sample_data")
   assert DataDownloadTools.VerifySlicerHeartCTData("sample_data")

DirLab-4DCT data is manual-only; see ``data/README.md`` before running the
high-resolution 4D CT reconstruction, lung-lobe PCA model, or PCA time-series
propagation tutorials. Tutorial 9 also requires the PhysicsNeMo dependency
installed with PhysioMotion4D.

Visualizing Results
===================

In NVIDIA Omniverse
-------------------

1. Open NVIDIA Omniverse
2. Launch USD Composer or USD Presenter
3. File -> Open -> Select your generated `.usd` file
4. Press Play to view the animation

Using USD Viewer
----------------

.. code-block:: bash

   # View USD file with usdview (comes with usd-core)
   usdview results/final_model.usd

In PyVista
----------

For quick visualization of VTK meshes:

.. code-block:: python

   import pyvista as pv

   # Load and display
   mesh = pv.read("heart_frame_000.vtp")
   mesh.plot()

Next Steps
==========

Now that you've completed your first workflow:

* Explore :doc:`examples` for more use cases
* Read detailed :doc:`cli_scripts/overview`
* Learn about :doc:`api/segmentation/index` options
* Understand :doc:`api/registration/index` methods
* Check the :doc:`api/base` for advanced usage

.. important::

   **About CLI Commands and Experiments:**

   * **CLI Commands** ⭐ **PRIMARY RESOURCE** - Production-ready workflows with proper class usage
     (``physiomotion4d-convert-image-to-usd``, ``physiomotion4d-create-statistical-model``,
     ``physiomotion4d-fit-statistical-model-to-patient``).
     See ``src/physiomotion4d/cli/`` for implementation details.

   * **experiments/** - Research prototypes and design explorations. These demonstrate conceptual
     approaches for adapting workflows to new anatomical regions and digital twin applications,
     but may contain outdated APIs and should not be copied directly into production code.

Common Issues
=============

**Out of memory errors**

* Resample or crop the input image before running the workflow
* Process fewer frames at once
* Use ANTs registration with ``--registration-method ANTS`` when CUDA is unavailable

**Segmentation quality issues**

* Adjust contrast parameters
* Preprocess images (denoising, normalization)

**USD not animating**

* Check that the input time series has more than one frame
* Validate the generated USD with ``usdchecker final_model.usd``

See :doc:`troubleshooting` for more solutions.
