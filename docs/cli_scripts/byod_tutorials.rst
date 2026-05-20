.. _byod_tutorials:

Bring Your Own Data - DICOM, Images & VTK to USD
================================================

PhysioMotion4D lets you convert your own medical imaging data into OpenUSD for
interactive visualization in NVIDIA Omniverse. Image inputs may be a directory
of 3D or 4D DICOM data, a single 3D or 4D file in a common medical image format
such as MHA, NRRD, or NIfTI, or a list of 3D image files representing a time
series. VTK inputs may be one mesh file or a mesh sequence.

.. note::

   PhysioMotion4D is a research tool and has **not** been validated for
   clinical use. Outputs must not be used for diagnostic or therapeutic
   decisions without independent validation.

Installation
------------

Install the package with CUDA support (recommended for GPU acceleration)
or the CPU-only variant:

.. code-block:: bash

   # Recommended - CUDA-enabled
   pip install physiomotion4d[cuda13]

   # CPU-only
   pip install physiomotion4d

Verify that all three relevant CLI entry-points are available after installation:

.. code-block:: bash

   physiomotion4d-download-data --help
   physiomotion4d-convert-image-to-usd --help
   physiomotion4d-convert-vtk-to-usd --help

See :doc:`/installation` for prerequisites, CUDA version requirements, and
source-based installation.

Download Demonstration Data
---------------------------

Use the installed download CLI to fetch the public Slicer-Heart 4D CT sample:

.. code-block:: bash

   physiomotion4d-download-data

This stores ``TruncalValve_4DCT.seq.nrrd`` under
``data/Slicer-Heart-CT``. To choose a different location:

.. code-block:: bash

   physiomotion4d-download-data Slicer-Heart-CT \
       --directory path/to/Slicer-Heart-CT

DICOM and Medical Images to USD
-------------------------------

Image-to-USD conversion accepts DICOM directories directly. It also accepts
3D and 4D image files readable by ITK, including common formats such as
``.mha``, ``.nrrd``, ``.nii``, and ``.nii.gz``.

3D - Single DICOM Directory or Image File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass a DICOM series directory or a single 3D image file to produce a static USD
scene.

**CLI:**

.. code-block:: bash

   physiomotion4d-convert-image-to-usd \
       patient_dicom_dir \
       --output-dir ./results \
       --project-name patient_heart

   physiomotion4d-convert-image-to-usd \
       patient_ct.mha \
       --output-dir ./results \
       --project-name patient_heart

**Python API:**

.. code-block:: python

   import physiomotion4d as pm4d

   workflow = pm4d.WorkflowConvertImageToUSD(
       input_filenames=["patient_dicom_dir"],
       contrast_enhanced=False,
       output_directory="./results",
       project_name="patient_heart",
   )
   workflow.process()

The workflow writes ``<project_name>.dynamic_painted.usd``,
``<project_name>.static_painted.usd``, and ``<project_name>.all_painted.usd``
inside ``--output-dir``.

4D - DICOM Directory, 4D Image File, or 3D Image List
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass a 4D DICOM directory, a single 4D image file, or an explicit list of 3D
image files to produce an animated USD scene. Use ``--fps`` when you need to
set the animated USD playback rate. Use ``--reference-image`` only when you
need to provide a separate fixed image for registration; otherwise the workflow
selects its default reference frame internally.

**CLI:**

.. code-block:: bash

   physiomotion4d-convert-image-to-usd \
       gated_ct_dicom_dir \
       --output-dir ./results \
       --project-name heart_animated

   physiomotion4d-convert-image-to-usd \
       gated_ct_4d.nrrd \
       --output-dir ./results \
       --project-name heart_animated

   physiomotion4d-convert-image-to-usd \
       phase_000.mha phase_001.mha phase_002.mha \
       --output-dir ./results \
       --fps 30 \
       --project-name heart_animated

**Python API:**

.. code-block:: python

   import physiomotion4d as pm4d

   workflow = pm4d.WorkflowConvertImageToUSD(
       input_filenames=["phase_000.mha", "phase_001.mha", "phase_002.mha"],
       contrast_enhanced=False,
       output_directory="./results",
       project_name="heart_animated",
       times_per_second=30.0,
   )
   workflow.process()

The resulting USD file contains a time-sampled mesh sequence that plays back
when you press **Play** in Omniverse USD Composer.

VTK to USD
----------

3D - Single Mesh
~~~~~~~~~~~~~~~~

Pass a single ``.vtp`` file. Use ``--appearance`` to control material style and
``--no-split`` to skip the default connected-component split.

**CLI:**

.. code-block:: bash

   # Default - split by connected component, anatomy material
   physiomotion4d-convert-vtk-to-usd heart.vtp \
       --output heart.usd \
       --appearance anatomy \
       --anatomy-type heart

   # Solid color, no splitting
   physiomotion4d-convert-vtk-to-usd mesh.vtp \
       --output mesh_red.usd \
       --appearance solid \
       --color 0.8 0.1 0.1 \
       --no-split

**Python API:**

.. code-block:: python

   import physiomotion4d as pm4d

   workflow = pm4d.WorkflowConvertVTKToUSD(
       vtk_files=["heart.vtp"],
       output_usd="heart.usd",
       appearance="anatomy",
       anatomy_type="heart",
   )
   workflow.run()

4D - Mesh Time Series
~~~~~~~~~~~~~~~~~~~~~

Pass per-frame VTK files. The default workflow treats multiple files as a time
series when their names match ``.t<index>.vtk``, ``.t<index>.vtp``, or
``.t<index>.vtu``. The VTK-to-USD CLI supports ``--fps`` to control playback
rate. For scalar colormaps, combine ``--primvar``, ``--cmap``, and
``--intensity-range``.

**CLI:**

.. code-block:: bash

   # Animated mesh sequence
   physiomotion4d-convert-vtk-to-usd heart.t0.vtp heart.t1.vtp heart.t2.vtp \
       --output heart_animation.usd \
       --fps 30

   # Animated with scalar colormap (e.g. wall stress)
   physiomotion4d-convert-vtk-to-usd stress.t0.vtk stress.t1.vtk stress.t2.vtk \
       --output stress_animation.usd \
       --fps 30 \
       --appearance colormap \
       --primvar vtk_point_stress_c0 \
       --cmap viridis \
       --intensity-range 0 500

**Python API:**

.. code-block:: python

   import physiomotion4d as pm4d

   workflow = pm4d.WorkflowConvertVTKToUSD(
       vtk_files=["stress.t0.vtk", "stress.t1.vtk", "stress.t2.vtk"],
       output_usd="stress_animation.usd",
       times_per_second=30,
       appearance="colormap",
       colormap_primvar="vtk_point_stress_c0",
       colormap_name="viridis",
       colormap_intensity_range=(0, 500),
   )
   workflow.run()

**Lower-level in-memory conversion with ConvertVTKToUSD:**

For programmatic pipelines where meshes are already in memory, use the
lower-level :class:`physiomotion4d.ConvertVTKToUSD` class directly:

.. code-block:: python

   import pyvista as pv
   import physiomotion4d as pm4d

   # Load or construct meshes in memory
   meshes = [pv.read(f"frame_{i:04d}.vtp") for i in range(10)]

   converter = pm4d.ConvertVTKToUSD(
       data_basename="HeartAnimation",
       input_polydata=meshes,
       times_per_second=30,
   )
   converter.convert("output.usd")

Viewing Results
---------------

**Programmatic inspection:**

.. code-block:: python

   import physiomotion4d as pm4d

   mesh = pm4d.USDTools().load_usd_as_vtk("output.usd")
   print(mesh.n_points, mesh.n_cells)

PyVista reads the VTK input files used above, but local validation with
PyVista 0.48.4 shows that ``pyvista.read()`` / ``pyvista.get_reader()`` do not
support ``.usd``, ``.usda``, or ``.usdc`` output files directly.

**In NVIDIA Omniverse:**

Open **Omniverse USD Composer**, drag your ``.usd`` file onto the viewport,
then press **Play** (spacebar) to watch the animation. For 4D cardiac data,
use the **Timeline** panel to scrub through phases.

See Also
--------

- :doc:`/installation`
- :doc:`/quickstart`
- :doc:`/cli_scripts/heart_gated_ct`
- :doc:`/cli_scripts/vtk_to_usd`
- :doc:`/api/workflows`
- :doc:`/examples`
- :doc:`/troubleshooting`

.. _isaac_for_healthcare_assets:

4D Isaac for Healthcare Assets
------------------------------

PhysioMotion4D has been used to generate a number of 4D anatomic models for
Isaac for Healthcare. These datasets are intended to support visualization and
workflow development with time-varying anatomy in OpenUSD.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Asset
     - Download link
   * - Chest with cardiac motion
     -
   * - Chest with respiratory motion
     -
   * - Heart with cardiac motion
     -
   * - Lungs with respiratory motion
     -
