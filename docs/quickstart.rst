===========
Quick Start
===========

This guide will help you get started with PhysioMotion4D quickly. We'll walk through a basic workflow for processing heart-gated CT data.

Prerequisites
=============

Before starting, ensure you have:

* PhysioMotion4D installed (see :doc:`installation`)
* NVIDIA GPU with CUDA 13 (default) or CUDA 12 — recommended for production performance; see :doc:`installation` for the ``[cuda13]`` and ``[cuda12]`` extras. A CPU-only install works for evaluation but is slow.
* 4D cardiac CT data or access to sample datasets

Basic Workflow
==============

Command-Line Interface
----------------------

The fastest way to process cardiac CT data is using the command-line interface:

.. code-block:: bash

   # Process a single 4D cardiac CT file
   physiomotion4d-heart-gated-ct cardiac_4d.nrrd --contrast --output-dir ./results

   # Process multiple time frames
   physiomotion4d-heart-gated-ct frame_*.nrrd --contrast --project-name patient_001

   # With custom settings
   physiomotion4d-heart-gated-ct cardiac.nrrd \
       --contrast \
       --reference-image ref.mha \
       --registration-iterations 50 \
       --output-dir ./output

Python API
----------

For more control, use the Python API:

**Step 1: Import the processor**

.. code-block:: python

   from physiomotion4d import ProcessHeartGatedCT

**Step 2: Initialize with your data**

.. code-block:: python

   processor = ProcessHeartGatedCT(
       input_filenames=["path/to/cardiac_4d_ct.nrrd"],
       contrast_enhanced=True,
       output_directory="./results",
       project_name="cardiac_model"
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

   from physiomotion4d import ProcessHeartGatedCT

   # Initialize processor
   processor = ProcessHeartGatedCT(
       input_filenames=["cardiac_4d.nrrd"],
       contrast_enhanced=True,
       output_directory="./results"
   )

   # Step 1: Convert 4D to 3D frames
   processor.convert_4d_to_3d()

   # Step 2: Register images
   processor.register_images()

   # Step 3: Generate segmentation
   processor.segment_reference_image()

   # Step 4: Transform contours
   processor.transform_contours()

   # Step 5: Create USD models
   final_usd = processor.create_usd_models()

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

   # Extract individual anatomy masks
   heart_mask, vessels_mask, lungs_mask, bones_mask, \
   soft_tissue_mask, contrast_mask, all_mask, dynamic_mask = masks

   # Save results
   itk.imwrite(heart_mask, "heart_mask.nrrd")

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

   import urllib.request
   import os

   # Create data directory
   os.makedirs("sample_data", exist_ok=True)

   # Download sample from Slicer-Heart-CT
   # (Replace with actual download link)
   url = "https://example.com/sample_cardiac_ct.nrrd"
   urllib.request.urlretrieve(url, "sample_data/cardiac_ct.nrrd")

Or use the DirLab lung dataset:

.. code-block:: python

   from physiomotion4d.data import DirLab4DCT

   # Download DirLab case
   downloader = DirLab4DCT()
   downloader.download_case(1)  # Downloads Case 1

Visualizing Results
===================

In NVIDIA Omniverse
-------------------

1. Open NVIDIA Omniverse
2. Launch USD Composer or USD Presenter
3. File → Open → Select your generated `.usd` file
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
     (``physiomotion4d-heart-gated-ct``, ``physiomotion4d-create-statistical-model``,
     ``physiomotion4d-fit-statistical-model-to-patient``).
     See ``src/physiomotion4d/cli/`` for implementation details.

   * **experiments/** - Research prototypes and design explorations. These demonstrate conceptual
     approaches for adapting workflows to new anatomical regions and digital twin applications,
     but may contain outdated APIs and should not be copied directly into production code.

Common Issues
=============

**Out of memory errors**

* Reduce image size: ``processor.set_downsample_factor(2)``
* Process fewer frames at once
* Use CPU registration: ``processor.set_registration_device('cpu')``

**Segmentation quality issues**

* Adjust contrast parameters
* Preprocess images (denoising, normalization)

**USD not animating**

* Check time samples: ``processor.verify_time_samples()``
* Ensure frame rate is set: ``processor.set_frame_rate(30)``
* Validate USD: ``usdchecker final_model.usd``

See :doc:`troubleshooting` for more solutions.
