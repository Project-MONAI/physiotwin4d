========
Examples
========

This page provides quick examples for common PhysioMotion4D use cases. For detailed workflow guides,
see the :doc:`cli_scripts/overview` section.

.. note::

   **For Production Workflows:** The CLI commands (``physiomotion4d-heart-gated-ct``,
   ``physiomotion4d-create-statistical-model``, ``physiomotion4d-fit-statistical-model-to-patient``)
   and their implementations in ``src/physiomotion4d/cli/``
   are the definitive source for proper library usage, class instantiation, and best practices.

   The ``experiments/`` directory contains research prototypes that informed development but should
   **not** be used as usage examples. They may contain outdated APIs, hardcoded paths, and minimal
   error handling. Consult these experiments only as **conceptual references** when adapting
   workflows to new anatomical regions or digital twin applications.

Complete Workflow Examples
===========================

Heart-Gated CT Processing
--------------------------

Complete end-to-end cardiac CT processing:

.. code-block:: python

   from physiomotion4d import ProcessHeartGatedCT

   # Initialize processor
   processor = ProcessHeartGatedCT(
       input_filenames=["cardiac_4d.nrrd"],
       contrast_enhanced=True,
       output_directory="./results",
       project_name="patient_001"
   )

   # Run complete workflow
   final_usd = processor.process()
   print(f"Generated USD model: {final_usd}")

Lung 4D-CT Processing
----------------------

Process respiratory motion from 4D-CT:

.. code-block:: python

   from physiomotion4d import ProcessHeartGatedCT
   from physiomotion4d.data import DirLab4DCT

   # Download DirLab case
   downloader = DirLab4DCT()
   downloader.download_case(1, output_dir="./data")

   # Process with respiratory parameters
   processor = ProcessHeartGatedCT(
       input_filenames=["./data/DirLab/case1/case1_T00.mha",
                       "./data/DirLab/case1/case1_T50.mha"],
       contrast_enhanced=False,
       output_directory="./results/lung",
       project_name="lung_case1"
   )

   final_usd = processor.process()

Segmentation Examples
=====================

TotalSegmentator
----------------

Quick segmentation with TotalSegmentator:

.. code-block:: python

   from physiomotion4d import SegmentChestTotalSegmentator
   import itk

   # Load image
   image = itk.imread("chest_ct.nrrd")

   # Segment
   segmenter = SegmentChestTotalSegmentator()
   masks = segmenter.segment(image, contrast_enhanced_study=True)

   # Extract masks
   heart, vessels, lungs, bones, soft_tissue, contrast, all_organs, dynamic = masks

   # Save results
   itk.imwrite(heart, "heart_mask.nrrd")
   itk.imwrite(lungs, "lungs_mask.nrrd")

Registration Examples
=====================

ICON Deep Learning Registration
--------------------------------

Fast GPU-accelerated registration:

.. code-block:: python

   from physiomotion4d.register_images_icon import RegisterImagesICON
   import itk

   # Initialize
   registerer = RegisterImagesICON()
   registerer.set_modality('ct')
   registerer.set_device('cuda')
   registerer.set_iterations(100)

   # Load images
   fixed = itk.imread("frame_000.mha")
   moving = itk.imread("frame_005.mha")

   # Register
   registerer.set_fixed_image(fixed)
   results = registerer.register(moving)

   # Get results
   inverse_transform = results["inverse_transform"]
   forward_transform = results["forward_transform"]
   registered = results["registered_image"]

   # Save
   itk.imwrite(registered, "registered.mha")
   itk.transformwrite(forward_transform, "transform_forward.hdf")
   itk.transformwrite(inverse_transform, "transform_inverse.hdf")

Multi-Phase Cardiac Registration
---------------------------------

Register all cardiac phases to reference:

.. code-block:: python

   from physiomotion4d.register_images_icon import RegisterImagesICON
   import itk
   import glob

   registerer = RegisterImagesICON()
   registerer.set_modality('ct')

   # Load reference (e.g., end-diastole)
   reference = itk.imread("frame_000.mha")
   registerer.set_fixed_image(reference)

   # Register all phases
   frame_files = sorted(glob.glob("frame_*.mha"))
   transforms = []

   for frame_file in frame_files[1:]:  # Skip reference
       moving = itk.imread(frame_file)
       results = registerer.register(moving)
       transforms.append(results["inverse_transform"])

       print(f"Registered {frame_file}: similarity = {results['similarity_score']:.3f}")

ANTs Multi-Stage Registration
------------------------------

Advanced registration with multiple stages:

.. code-block:: python

   from physiomotion4d import RegisterImagesANTs
   import itk

   registerer = RegisterImagesANTs()

   fixed = itk.imread("reference.mha")
   moving = itk.imread("target.mha")

   registerer.set_fixed_image(fixed)
   registerer.set_modality('ct')
   registerer.set_transform_type('SyN')

   # Perform registration with SyN (Symmetric Normalization)
   result = registerer.register(moving)

USD Conversion Examples
=======================

VTK Time Series to USD
-----------------------

Convert VTK mesh sequence to animated USD:

.. code-block:: python

   from physiomotion4d import ConvertVTKToUSD
   import glob

   # Get VTK files
   vtk_files = sorted(glob.glob("heart_frame_*.vtp"))

   time_codes = [float(i) for i in range(len(vtk_files))]
   stage = ConvertVTKToUSD.from_files(
       data_basename="Heart",
       vtk_files=vtk_files,
       time_codes=time_codes,
       times_per_second=30,
   ).convert("heart_animation.usd")

   print("USD animation created: heart_animation.usd")

Merge Multiple USD Files
-------------------------

Combine separate anatomical structures:

.. code-block:: python

   from physiomotion4d import USDTools

   tools = USDTools()

   # Merge USD files
   files = [
       "heart_dynamic.usd",
       "lungs_dynamic.usd",
       "vessels_static.usd",
       "bones_static.usd"
   ]

   tools.merge_usd_files(
       input_files=files,
       output_file="complete_thorax.usd",
       flatten=True
   )

Apply Materials to USD
----------------------

Add anatomical materials and colors:

.. code-block:: python

   from physiomotion4d import USDAnatomyTools

   painter = USDAnatomyTools()

   painter.paint_usd_file(
       input_usd="thorax_model.usd",
       output_usd="thorax_painted.usd",
       anatomy_mapping={
           "/World/Heart": "cardiac_muscle",
           "/World/Aorta": "arterial_vessel",
           "/World/VenaCava": "venous_vessel",
           "/World/LeftLung": "lung_tissue",
           "/World/RightLung": "lung_tissue",
           "/World/Ribs": "cortical_bone"
       }
   )

Transform and Contour Examples
===============================

Apply Transform to Images
--------------------------

Warp images using deformation fields:

.. code-block:: python

   from physiomotion4d import TransformTools
   import itk

   tools = TransformTools()

   # Load deformation field and image
   phi = itk.imread("deformation_field.mha")
   moving_image = itk.imread("moving_image.mha")

   # Apply transform
   warped = tools.apply_transform_to_image(moving_image, phi)

   # Save result
   itk.imwrite(warped, "warped_image.mha")

Transform Contours/Meshes
--------------------------

Propagate segmentation contours across time:

.. code-block:: python

   from physiomotion4d import TransformTools
   import itk
   import pyvista as pv

   tools = TransformTools()

   # Load reference contour and transforms
   reference_mesh = pv.read("heart_t0.vtp")

   # Transform to each time point
   for t in range(1, 10):
       phi = itk.imread(f"transform_t0_to_t{t}.mha")
       warped_mesh = tools.apply_transform_to_contour(reference_mesh, phi)
       warped_mesh.save(f"heart_t{t}.vtp")

Extract Surface from Segmentation
----------------------------------

Convert segmentation masks to meshes:

.. code-block:: python

   from physiomotion4d import ContourTools
   import itk

   tools = ContourTools()

   # Load segmentation
   mask = itk.imread("heart_segmentation.nrrd")

   # Extract smooth surface
   mesh = tools.extract_surface(
       mask,
       threshold=0.5,
       smoothing_iterations=20,
       decimate_target=10000  # Target triangle count
   )

   # Save mesh
   mesh.save("heart_surface.vtp")

Visualization Examples
======================

Quick Preview in PyVista
-------------------------

Visualize meshes and images:

.. code-block:: python

   import pyvista as pv
   import itk
   import numpy as np

   # Visualize mesh
   mesh = pv.read("heart.vtp")
   mesh.plot(color='red', opacity=0.8)

   # Visualize image slice
   image = itk.imread("ct_scan.nrrd")
   array = itk.array_from_image(image)

   pl = pv.Plotter()
   pl.add_volume(array, cmap='bone')
   pl.show()

Animated Mesh Sequence
-----------------------

Create animation from mesh sequence:

.. code-block:: python

   import pyvista as pv
   import glob

   # Load mesh sequence
   mesh_files = sorted(glob.glob("heart_t*.vtp"))

   # Create plotter
   pl = pv.Plotter()

   # Animate
   meshes = [pv.read(f) for f in mesh_files]

   for i in range(len(meshes)):
       pl.clear()
       pl.add_mesh(meshes[i], color='red')
       pl.write_frame()

   pl.close()

Batch Processing Examples
==========================

Process Multiple Patients
--------------------------

Batch process multiple datasets:

.. code-block:: python

   from physiomotion4d import ProcessHeartGatedCT
   import glob
   import os

   # Find all patient data
   patient_dirs = glob.glob("data/patient_*")

   for patient_dir in patient_dirs:
       patient_id = os.path.basename(patient_dir)
       input_file = os.path.join(patient_dir, "cardiac_4d.nrrd")

       if not os.path.exists(input_file):
           continue

       print(f"Processing {patient_id}...")

       processor = ProcessHeartGatedCT(
           input_filenames=[input_file],
           contrast_enhanced=True,
           output_directory=f"results/{patient_id}",
           project_name=patient_id
       )

       try:
           final_usd = processor.process()
           print(f"  ✓ Complete: {final_usd}")
       except Exception as e:
           print(f"  ✗ Failed: {e}")

Parallel Segmentation
---------------------

Segment multiple images in parallel:

.. code-block:: python

   from physiomotion4d import SegmentChestTotalSegmentator
   import itk
   import glob
   from concurrent.futures import ProcessPoolExecutor

   def segment_image(filename):
       segmenter = SegmentChestTotalSegmentator()
       image = itk.imread(filename)
       result = segmenter.segment(image, contrast_enhanced_study=True)

       # Save heart mask
       output_name = filename.replace('.nrrd', '_heart.nrrd')
       itk.imwrite(result['heart'], output_name)
       return output_name

   # Process in parallel
   image_files = glob.glob("data/*.nrrd")

   with ProcessPoolExecutor(max_workers=4) as executor:
       results = list(executor.map(segment_image, image_files))

   print(f"Segmented {len(results)} images")

Data Download Examples
======================

Download DirLab Dataset
-----------------------

.. code-block:: python

   from physiomotion4d.data import DirLab4DCT

   downloader = DirLab4DCT()

   # Download all 10 cases
   for case_num in range(1, 11):
       print(f"Downloading case {case_num}...")
       downloader.download_case(case_num, output_dir="./data/DirLab")

   # Load case for processing
   case_data = downloader.load_case(1)
   inhale_image = case_data["inhale"]
   exhale_image = case_data["exhale"]

Custom Workflow Examples
=========================

Step-by-Step Processing
------------------------

Manual control over each step:

.. code-block:: python

   from physiomotion4d import ProcessHeartGatedCT

   processor = ProcessHeartGatedCT(
       input_filenames=["cardiac_4d.nrrd"],
       contrast_enhanced=True,
       output_directory="./results"
   )

   # Step 1: Convert 4D to 3D frames
   print("Converting 4D to 3D...")
   processor.convert_4d_to_3d()

   # Step 2: Register images
   print("Registering images...")
   processor.register_images()

   # Step 3: Segment reference
   print("Segmenting...")
   processor.segment_reference_image()

   # Step 4: Transform contours
   print("Transforming contours...")
   processor.transform_contours()

   # Step 5: Create USD
   print("Creating USD models...")
   final_usd = processor.create_usd_models()

   print(f"Complete: {final_usd}")

Custom Pipeline with Specific Methods
--------------------------------------

Mix and match different components:

.. code-block:: python

   from physiomotion4d import (
       SegmentChestTotalSegmentator,
       RegisterImagesICON,
       TransformTools,
       ConvertVTKToUSD,
       ContourTools
   )
   import itk

   # Load images
   reference = itk.imread("frame_000.mha")
   frames = [itk.imread(f"frame_{i:03d}.mha") for i in range(10)]

   # Segment reference
   segmenter = SegmentChestTotalSegmentator()
   result = segmenter.segment(reference, contrast_enhanced_study=True)
   heart_mask = result['heart']

   # Extract reference contour
   contour_tools = ContourTools()
   reference_mesh = contour_tools.extract_surface(heart_mask)

   # Register and transform
   registerer = RegisterImagesICON()
   registerer.set_fixed_image(reference)
   transform_tools = TransformTools()

   meshes = [reference_mesh]
   for frame in frames[1:]:
       results = registerer.register(frame)
       warped_mesh = transform_tools.apply_transform_to_contour(
           reference_mesh,
           results["inverse_transform"]
       )
       meshes.append(warped_mesh)

   # Save meshes
   for i, mesh in enumerate(meshes):
       mesh.save(f"heart_{i:03d}.vtp")

   vtk_files = [f"heart_{i:03d}.vtp" for i in range(10)]
   time_codes = [float(i) for i in range(len(vtk_files))]
   stage = ConvertVTKToUSD.from_files(
       data_basename="Heart",
       vtk_files=vtk_files,
       time_codes=time_codes,
   ).convert("heart.usd")

See Also
========

* :doc:`cli_scripts/overview` - Detailed command-line workflows
* :doc:`api/index` - Complete API reference
* :doc:`cli_scripts/heart_gated_ct` - Heart-gated CT guide
* :doc:`troubleshooting` - Common issues and solutions
