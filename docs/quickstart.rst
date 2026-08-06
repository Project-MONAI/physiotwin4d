===========
Quick Start
===========

This guide will help you get started with PhysioTwin4D quickly.

Prerequisites
=============

Before starting, ensure you have:

* PhysioTwin4D installed (see :doc:`installation`)
* NVIDIA GPU with CUDA 13 - recommended for production performance; see
  :doc:`installation` for the ``[cuda13]`` extra. A CPU-only PyPI install works
  for evaluation but is slow.
* Disk space for the sample datasets (~10-20 GB for the full set; each dataset
  README lists its own size)

.. _tutorial_scripts:

Get the Tutorial Scripts
========================

The tutorials ship with the source repository, not with the pip package.
``pip install physiotwin4d`` installs the library and the ``physiotwin4d-*``
commands, but it does not create a ``tutorials/`` directory. Clone the
repository to get the scripts:

.. code-block:: bash

   git clone https://github.com/Project-MONAI/physiotwin4d.git
   cd physiotwin4d

To match the scripts to an already-installed release, check out its tag. Tags
are bare version strings, so use the installed version directly:

.. code-block:: bash

   python -c "import physiotwin4d; print(physiotwin4d.__version__)"

   git clone --depth 1 --branch {{ pt4d_project_version }} \
       https://github.com/Project-MONAI/physiotwin4d.git
   cd physiotwin4d

A release tarball works just as well if you would rather not use git:
``https://github.com/Project-MONAI/physiotwin4d/archive/refs/tags/{{ pt4d_project_version }}.tar.gz``.

No further installation step is needed after cloning — the scripts import
``physiotwin4d`` from your environment.

Run every dataset download from the top level of the clone. The tutorials
resolve their inputs against the repository root (``<repo>/data/<dataset>``),
while ``physiotwin4d-download-data`` writes to ``data/<dataset>`` relative to
the current working directory, so downloading from anywhere else puts the data
where the tutorials will not find it.

Then fetch the sample datasets, again from the top level of the clone:

.. code-block:: bash

   # Tutorials 1, 3 and 4 (heart)
   physiotwin4d-download-data Slicer-Heart-CT --directory data/Slicer-Heart-CT

   # Tutorial 6 (heart)
   physiotwin4d-download-data KCL-Heart-Model --directory data/KCL-Heart-Model

   # Tutorial 7 (lung)
   physiotwin4d-download-data Chest-CT --directory data/Chest-CT

``DirLab-4DCT`` is manual-only; see ``data/DirLab-4DCT/README.md``, and
:doc:`cli_scripts/download_data` for every dataset's size and source.

Tutorial 1 needs only ``Slicer-Heart-CT``; with that dataset in place it runs
as a plain script:

.. code-block:: bash

   python tutorials/tutorial_01_heart_gated_ct_to_usd.py

:doc:`tutorials` is the full guide — ten tutorials with previews of what each
one produces, the run order, and per-tutorial notes on pointing them at your
own data. The rest of this page is the same functionality as a CLI call and as
a Python API call, for when you would rather not start from a script.

Basic Workflow
==============

Minimal Slicer-Heart Quickstart
-------------------------------

The public Slicer-Heart 4D CT sample can be downloaded automatically and used
as the smallest end-to-end cardiac workflow. Data downloading and a
CUDA-capable GPU are required for practical runtime.

.. code-block:: bash

   physiotwin4d-download-data Slicer-Heart-CT --directory data/Slicer-Heart-CT

   physiotwin4d-convert-image-to-usd data/Slicer-Heart-CT/TruncalValve_4DCT.seq.nrrd \
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

For more control, use the Python API. The workflow takes **images, not
filenames**: read your series with ITK first, and pick one frame as the
segmentation and registration reference.

.. code-block:: python

   import itk
   from pathlib import Path

   from physiotwin4d import (
       RegisterImagesICON,
       SegmentChestTotalSegmentatorWithContrast,
       WorkflowConvertImageToUSD,
   )

   frame_files = sorted(Path("data/Slicer-Heart-CT").glob("slice_???.mha"))
   time_series_images = [itk.imread(str(path)) for path in frame_files]
   reference_image = time_series_images[int(0.7 * len(time_series_images))]

   workflow = WorkflowConvertImageToUSD(
       time_series_images=time_series_images,
       reference_image=reference_image,
       output_directory="./results",
       usd_project_name="cardiac_model",
       registration_method=RegisterImagesICON(),
       segmentation_method=SegmentChestTotalSegmentatorWithContrast(),
       save_assets=True,
   )
   results = workflow.process()

   print(f"USD model saved to: {results['dynamic']}")

That is the whole pipeline: 4D input to 3D frames, registration between
phases, AI segmentation of the reference, contour transformation across time,
and an animated USD scene. A 4D ``.seq.nrrd`` needs splitting into frames
first — use ``physiotwin4d-convert-image-4d-to-3d`` or
:class:`~physiotwin4d.ConvertImage4DTo3D`.

:doc:`tutorials` walks the same call through real data with screenshots, and
each tutorial section ends with the constants to change for your own scans.

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

Download Sample Datasets
-------------------------

``Slicer-Heart-CT``, ``KCL-Heart-Model``, ``CHOP-Valve4D``, and ``Chest-CT``
are all auto-downloadable, via the CLI:

.. code-block:: bash

   physiotwin4d-download-data Slicer-Heart-CT --directory data/Slicer-Heart-CT

or from Python:

.. code-block:: python

   from physiotwin4d import DataDownloadTools

   data_file = DataDownloadTools.DownloadSlicerHeartCTData("data/Slicer-Heart-CT")
   assert DataDownloadTools.VerifySlicerHeartCTData("data/Slicer-Heart-CT")

See :doc:`cli_scripts/download_data` for sizes, source URLs, and directory
layouts for every dataset.

DirLab-4DCT data is manual-only; see ``data/DirLab-4DCT/README.md``. It drives
the lung pipeline — Tutorials 2, 3, 6 (lung) and 8 — which then feeds the
AI-surrogate Tutorials 9 and 10. Those two additionally require the optional
``physicsnemo`` extra (``pip install "physiotwin4d[physicsnemo]"``, plus
``torch-geometric`` for the MeshGraphNet); PhysicsNeMo itself requires
Python >= 3.11.

Visualizing Results
===================

The workflows write OpenUSD scenes, and viewing them needs a USD viewer — use
an Omniverse Kit application with RTX rendering, which is what evaluates the
material properties assigned to each tissue. Note that the ``usd-core``
package installed with PhysioTwin4D provides the OpenUSD *libraries* only and
contains no viewer.

:doc:`viewing_usd` covers where to get it, how to set it up, and how to open a
PhysioTwin4D scene — including switching to the camera defined in the scene,
whose clipping planes are fitted to the anatomy's scale.

The intermediate meshes need no USD tooling:

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
