========
Examples
========

This page provides quick examples for common PhysioMotion4D use cases. For detailed workflow guides,
see the :doc:`cli_scripts/overview` section.

.. note::

   **For Production Workflows:** The CLI commands (``physiomotion4d-convert-image-to-usd``,
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

   from physiomotion4d import WorkflowConvertImageToUSD

   # Initialize workflow
   workflow = WorkflowConvertImageToUSD(
       input_filenames=["cardiac_4d.nrrd"],
       contrast_enhanced=True,
       output_directory="./results",
       project_name="patient_001",
       registration_method="ANTS",
   )

   # Run complete workflow
   final_usd = workflow.process()
   print(f"Generated USD model: {final_usd}")

Lung 4D-CT Reconstruction
--------------------------

Reconstruct a high-resolution 4D CT sequence from manually prepared respiratory
phase images:

.. code-block:: python

   from pathlib import Path

   import itk

   from physiomotion4d import WorkflowReconstructHighres4DCT

   # DirLab-4DCT data is manual-only. Place phase images under ./data/DirLab.
   phase_dir = Path("./data/DirLab/case1")
   phase_files = sorted(list(phase_dir.glob("*.mhd")) + list(phase_dir.glob("*.mha")))
   time_series_images = [itk.imread(str(path)) for path in phase_files]

   fixed_image = time_series_images[0]
   workflow = WorkflowReconstructHighres4DCT(
       time_series_images=time_series_images,
       fixed_image=fixed_image,
       reference_frame=0,
       registration_method="ants",
   )

   result = workflow.run_workflow(upsample_to_fixed_resolution=True)
   reconstructed_images = result["reconstructed_images"]

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

   # Extract masks by anatomy group
   heart = masks["heart"]
   lungs = masks["lung"]
   vessels = masks["major_vessels"]
   labelmap = masks["labelmap"]

   # Save results
   itk.imwrite(heart, "heart_mask.nrrd")
   itk.imwrite(lungs, "lungs_mask.nrrd")
   itk.imwrite(vessels, "major_vessels_mask.nrrd")
   itk.imwrite(labelmap, "labelmap.nrrd")

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
   registerer.set_number_of_iterations(100)

   # Load images
   fixed = itk.imread("frame_000.mha")
   moving = itk.imread("frame_005.mha")

   # Register
   registerer.set_fixed_image(fixed)
   results = registerer.register(moving)

   # Get results
   inverse_transform = results["inverse_transform"]
   forward_transform = results["forward_transform"]
   registered = registerer.get_registered_image()

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

       print(f"Registered {frame_file}: loss = {results['loss']:.3f}")

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

   tools.merge_usd_files("complete_thorax.usd", files)

Apply Materials to USD
----------------------

Add anatomical materials and colors:

.. code-block:: python

   from physiomotion4d import USDAnatomyTools
   from pxr import Usd

   stage = Usd.Stage.Open("thorax_model.usd")
   painter = USDAnatomyTools(stage)

   painter.apply_anatomy_material_to_mesh("/World/Heart", "heart")
   painter.apply_anatomy_material_to_mesh("/World/Aorta", "major_vessels")
   painter.apply_anatomy_material_to_mesh("/World/LeftLung", "lung")
   painter.apply_anatomy_material_to_mesh("/World/RightLung", "lung")
   painter.apply_anatomy_material_to_mesh("/World/Ribs", "bone")

   stage.Export("thorax_painted.usd")

Transform and Contour Examples
===============================

Apply Transform to Images
--------------------------

Warp images using deformation fields:

.. code-block:: python

   from physiomotion4d import TransformTools
   import itk

   tools = TransformTools()

   # Load transform, moving image, and reference image
   phi = itk.transformread("transform_forward.hdf")
   moving_image = itk.imread("moving_image.mha")
   reference_image = itk.imread("reference_image.mha")

   # Apply transform
   warped = tools.transform_image(moving_image, phi, reference_image)

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
       phi = itk.transformread(f"transform_t0_to_t{t}.hdf")
       warped_mesh = tools.transform_pvcontour(reference_mesh, phi)
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

   # Extract smoothed contour surface
   mesh = tools.extract_contours(mask)

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

   pl = pv.Plotter()
   pl.open_gif("heart_animation.gif")

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

   from physiomotion4d import WorkflowConvertImageToUSD
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

       workflow = WorkflowConvertImageToUSD(
           input_filenames=[input_file],
           contrast_enhanced=True,
           output_directory=f"results/{patient_id}",
           project_name=patient_id,
           registration_method="ANTS",
       )

       try:
           final_usd = workflow.process()
           print(f"  Complete: {final_usd}")
       except Exception as e:
           print(f"  Failed: {e}")

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
   max_workers = 1  # Increase only when each worker has enough CPU/GPU memory.

   with ProcessPoolExecutor(max_workers=max_workers) as executor:
       results = list(executor.map(segment_image, image_files))

   print(f"Segmented {len(results)} images")

Data Download Examples
======================

Download Slicer-Heart Dataset
-----------------------------

.. code-block:: python

   from physiomotion4d import DataDownloadTools

   data_file = DataDownloadTools.DownloadSlicerHeartCTData("data/Slicer-Heart-CT")
   assert DataDownloadTools.VerifySlicerHeartCTData("data/Slicer-Heart-CT")

Custom Workflow Examples
=========================

Complete Workflow Processing
----------------------------

Run the supported end-to-end workflow API:

.. code-block:: python

   from physiomotion4d import WorkflowConvertImageToUSD

   workflow = WorkflowConvertImageToUSD(
       input_filenames=["cardiac_4d.nrrd"],
       contrast_enhanced=True,
       output_directory="./results",
       project_name="cardiac_model",
       registration_method="ANTS",
   )

   final_usd = workflow.process()

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
   reference_mesh = contour_tools.extract_contours(heart_mask)

   # Register and transform
   registerer = RegisterImagesICON()
   registerer.set_modality('ct')
   registerer.set_fixed_image(reference)
   transform_tools = TransformTools()

   meshes = [reference_mesh]
   for frame in frames[1:]:
       results = registerer.register(frame)
       warped_mesh = transform_tools.transform_pvcontour(
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
