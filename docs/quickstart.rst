===========
Quick Start
===========

This guide will help you get started with PhysioTwin4D quickly.

.. warning::

   **Not validated for clinical use.** PhysioTwin4D
   {{ pt4d_project_version }} beta is a research and visualization toolkit, not
   a medical device. Do not use it for diagnosis, treatment planning, or
   clinical decision-making.

.. _tutorials:

Tutorials
=========

The ``tutorials/`` directory contains eleven end-to-end scripts covering nine
major workflows (Tutorials 9 and 10 each have MeshGraphNet and MLP variants).
Each script is a ``# %%`` percent-cell Python script that exercises
the workflow classes directly. Run as a regular file
(``python tutorials/tutorial_01_...py``) or cell-by-cell in VS Code or Cursor.

See :doc:`tutorials` for the NVIDIA-styled tutorial card index, dataset
requirements, script paths, and workflow details.

After preparing the Slicer-Heart-CT data, run the first two tutorials:

.. code-block:: bash

   python tutorials/tutorial_01_heart_gated_ct_to_usd.py

   python tutorials/tutorial_02_ct_to_vtk.py

Tutorial paths are defined near the top of each script. To use different paths,
edit the script constants or use the installed ``physiotwin4d-*`` CLI commands.
See ``tutorials/README.md`` for dataset download instructions and the
recommended run order.

Recommended run order:

1. Tutorials 1 and 2 first, after preparing Slicer-Heart-CT data.
2. Tutorial 5 after Tutorial 2 (consumes Tutorial 2 output).
3. Tutorial 3 after downloading KCL-Heart-Model.
4. Tutorial 4 after Tutorial 3 because it can consume Tutorial 3 output.
5. Tutorial 6 after downloading DirLab-4DCT.
6. Tutorial 8 after preparing your own cardiac gated CT, labelmaps, KCL volume
   PCA model, and ICON weights (bring-your-own-data).
7. Tutorial 9a and/or 9b after Tutorial 8 because they train from its fitted
   meshes.
8. Tutorial 10a and/or 10b after Tutorial 9a / 9b because they evaluate the
   trained checkpoints.

Prerequisites
=============

Before starting, ensure you have:

* PhysioTwin4D installed (see :doc:`installation`)
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

   python -c "from physiotwin4d import DataDownloadTools; DataDownloadTools.DownloadSlicerHeartCTData('data/test')"

   physiotwin4d-convert-image-to-usd data/test/TruncalValve_4DCT.seq.nrrd \
       --registration-method ICON \
       --output-dir output/quickstart \
       --project-name slicer_heart_quickstart

Command-Line Interface
----------------------

The fastest way to process cardiac CT data is using the command-line interface:

.. code-block:: bash

   # Process a single 4D cardiac CT file
   physiotwin4d-convert-image-to-usd cardiac_4d.nrrd --contrast --output-dir ./results

   # Process multiple time frames
   physiotwin4d-convert-image-to-usd frame_*.nrrd --contrast --project-name patient_001

   # With custom settings
   physiotwin4d-convert-image-to-usd cardiac.nrrd \
       --contrast \
       --reference-image ref.mha \
       --registration-iterations 50 \
       --output-dir ./output

Python API
----------

For more control, use the Python API:

**Step 1: Import the processor**

.. code-block:: python

   from physiotwin4d import RegisterImagesICON, WorkflowConvertImageToUSD

**Step 2: Initialize with your data**

.. code-block:: python

   processor = WorkflowConvertImageToUSD(
       input_filenames=["path/to/cardiac_4d_ct.nrrd"],
       output_directory="./results",
       project_name="cardiac_model",
       registration_method=RegisterImagesICON(),
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

   from physiotwin4d import RegisterImagesICON, WorkflowConvertImageToUSD

   # Initialize workflow
   workflow = WorkflowConvertImageToUSD(
       input_filenames=["cardiac_4d.nrrd"],
       output_directory="./results",
       project_name="cardiac_model",
       registration_method=RegisterImagesICON(),
   )

   final_usd = workflow.process()

Working with Individual Components
===================================

Segmentation Only
-----------------

If you only need segmentation:

.. code-block:: python

   from physiotwin4d import SegmentChestTotalSegmentatorWithContrast
   import itk

   # Initialize segmenter (use SegmentChestTotalSegmentator for non-contrast studies)
   segmenter = SegmentChestTotalSegmentatorWithContrast()

   # Load and segment image
   image = itk.imread("chest_ct.nrrd")
   masks = segmenter.segment(image)

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

   from physiotwin4d.register_images_icon import RegisterImagesICON
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

   from physiotwin4d import ConvertVTKToUSD

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

   from physiotwin4d import DataDownloadTools

   data_file = DataDownloadTools.DownloadSlicerHeartCTData("sample_data")
   assert DataDownloadTools.VerifySlicerHeartCTData("sample_data")

DirLab-4DCT data is manual-only; see ``data/README.md`` before running the
high-resolution 4D CT reconstruction tutorial. Tutorials 8-10 are
bring-your-own-data cardiac tutorials; see :doc:`tutorials` for their dataset
layout. Tutorials 9a/9b/10a/10b additionally require the optional
``physicsnemo`` extra (``pip install "physiotwin4d[physicsnemo]"``);
PhysicsNeMo itself requires Python >= 3.11.

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

* Explore :doc:`tutorials` for more runnable examples
* Read detailed :doc:`cli_scripts/overview`
* Learn about :doc:`api/segmentation/index` options
* Understand :doc:`api/registration/index` methods
* Check the :doc:`api/base` for advanced usage

.. important::

   **About CLI Commands and Experiments:**

   * **CLI Commands** ⭐ **PRIMARY RESOURCE** - Production-ready workflows with proper class usage
     (``physiotwin4d-convert-image-to-usd``, ``physiotwin4d-create-statistical-model``,
     ``physiotwin4d-fit-statistical-model-to-patient``).
     See ``src/physiotwin4d/cli/`` for implementation details.

   * **experiments/** - Research prototypes and design explorations. These demonstrate conceptual
     approaches for adapting workflows to new anatomical regions and digital twin applications,
     but may contain outdated APIs and should not be copied directly into production code.

Common Issues
=============

**Out of memory errors**

* Resample or crop the input image before running the workflow
* Process fewer frames at once
* Use Greedy registration with ``--registration-method Greedy`` when CUDA is unavailable

**Segmentation quality issues**

* Adjust contrast parameters
* Preprocess images (denoising, normalization)

**USD not animating**

* Check that the input time series has more than one frame
* Validate the generated USD with ``usdchecker final_model.usd``

See :doc:`troubleshooting` for more solutions.
