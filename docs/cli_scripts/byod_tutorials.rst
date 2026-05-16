.. _byod_tutorials:

Bring Your Own Data — DICOM & VTK to USD
=========================================

PhysioMotion4D lets you convert your own medical imaging data — whether
DICOM-derived NIfTI volumes or VTK surface meshes — into OpenUSD for
interactive visualization in NVIDIA Omniverse.  Both 3D (single
volume/mesh) and 4D (time-series) inputs are supported.  The CLI and
Python API are **identical** for 3D and 4D inputs; the only difference
is how many files you pass in.

.. note::

   PhysioMotion4D is a research tool and has **not** been validated for
   clinical use.  Outputs must not be used for diagnostic or therapeutic
   decisions without independent validation.

Installation
------------

Install the package with CUDA support (recommended for GPU acceleration)
or the CPU-only variant:

.. code-block:: bash

   # Recommended — CUDA-enabled
   pip install physiomotion4d[cuda13]

   # CPU-only
   pip install physiomotion4d

Verify that both relevant CLI entry-points are available after installation:

.. code-block:: bash

   physiomotion4d-convert-heart-gated-ct-to-usd --help
   physiomotion4d-convert-vtk-to-usd --help

See :doc:`/installation` for prerequisites, CUDA version requirements, and
source-based installation.

DICOM to USD
------------

Raw DICOM images must first be converted to NIfTI with a tool such as
`dcm2niix <https://github.com/rordenlab/dcm2niix>`_ before being passed
to PhysioMotion4D.

3D — Single Volume
~~~~~~~~~~~~~~~~~~

Pass a single ``.nii.gz`` file to produce a static USD scene.

**CLI:**

.. code-block:: bash

   physiomotion4d-convert-heart-gated-ct-to-usd \
       patient_ct.nii.gz \
       --output patient_heart.usd

**Python API:**

.. code-block:: python

   import physiomotion4d as pm4d

   wf = pm4d.WorkflowConvertHeartGatedCTToUSD()
   wf.input_image = "patient_ct.nii.gz"
   wf.output_file = "patient_heart.usd"
   wf.run_workflow()

.. note::

   If your source data is raw DICOM, run ``dcm2niix -z y -o output_dir
   dicom_dir/`` first to produce the ``.nii.gz`` input file expected by
   this command.

4D — Gated CT Time Series
~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass multiple per-phase volumes (glob or explicit list) to produce an
animated USD scene.  Use ``--fps`` to control playback rate and
``--reference-frame`` to choose the registration anchor phase (0-indexed).

**CLI:**

.. code-block:: bash

   physiomotion4d-convert-heart-gated-ct-to-usd \
       phase_*.nii.gz \
       --output heart_animated.usd \
       --fps 25 \
       --reference-frame 0

**Python API:**

.. code-block:: python

   import glob
   import physiomotion4d as pm4d

   wf = pm4d.WorkflowConvertHeartGatedCTToUSD()
   wf.input_images = sorted(glob.glob("phase_*.nii.gz"))
   wf.output_file  = "heart_animated.usd"
   wf.fps            = 25
   wf.reference_frame = 0
   wf.run_workflow()

The resulting USD file contains a time-sampled mesh sequence that plays
back when you press **Play** in Omniverse USD Composer.

VTK to USD
----------

3D — Single Mesh
~~~~~~~~~~~~~~~~

Pass a single ``.vtp`` file.  Use ``--appearance`` to control material
style and ``--no-split`` to skip the default connected-component split.

**CLI:**

.. code-block:: bash

   # Default — split by connected component, anatomy material
   physiomotion4d-convert-vtk-to-usd heart.vtp \
       --output heart.usd \
       --appearance anatomy \
       --anatomy-type heart

   # Solid colour, no splitting
   physiomotion4d-convert-vtk-to-usd mesh.vtp \
       --output mesh_red.usd \
       --appearance solid \
       --color 0.8 0.1 0.1 \
       --no-split

**Python API:**

.. code-block:: python

   import physiomotion4d as pm4d

   wf = pm4d.WorkflowConvertVTKToUSD()
   wf.input_files  = ["heart.vtp"]
   wf.output_file  = "heart.usd"
   wf.appearance   = "anatomy"
   wf.anatomy_type = "heart"
   wf.run()

4D — Mesh Time Series
~~~~~~~~~~~~~~~~~~~~~

Pass a glob of per-frame files.  The ``--fps`` flag controls playback
rate.  For scalar colormaps, combine ``--primvar``, ``--cmap``, and
``--intensity-range``.

**CLI:**

.. code-block:: bash

   # Animated mesh sequence
   physiomotion4d-convert-vtk-to-usd frame_*.vtp \
       --output heart_animation.usd \
       --fps 30

   # Animated with scalar colormap (e.g. wall stress)
   physiomotion4d-convert-vtk-to-usd frame_*.vtk \
       --output stress_animation.usd \
       --fps 30 \
       --appearance colormap \
       --primvar vtk_point_stress_c0 \
       --cmap viridis \
       --intensity-range 0 500

**Python API:**

.. code-block:: python

   import glob
   import physiomotion4d as pm4d

   wf = pm4d.WorkflowConvertVTKToUSD()
   wf.input_files      = sorted(glob.glob("frame_*.vtk"))
   wf.output_file      = "stress_animation.usd"
   wf.fps              = 30
   wf.appearance       = "colormap"
   wf.primvar          = "vtk_point_stress_c0"
   wf.cmap             = "viridis"
   wf.intensity_range  = (0, 500)
   wf.run()

**Lower-level in-memory conversion with** ``ConvertVTKToUSD``**:**

For programmatic pipelines where meshes are already in memory, use the
lower-level :class:`physiomotion4d.ConvertVTKToUSD` class directly:

.. code-block:: python

   import pyvista as pv
   import physiomotion4d as pm4d

   # Load or construct meshes in memory
   meshes = [pv.read(f"frame_{i:04d}.vtp") for i in range(10)]

   converter = pm4d.ConvertVTKToUSD(output_file="output.usd", fps=30)
   for i, mesh in enumerate(meshes):
       converter.add_frame(mesh, frame_index=i)
   converter.write()

Viewing Results
---------------

**Quick preview with PyVista (no Omniverse required):**

.. code-block:: python

   import pyvista as pv

   mesh = pv.read("output.usd")
   mesh.plot()

**In NVIDIA Omniverse:**

Open **Omniverse USD Composer**, drag your ``.usd`` file onto the
viewport, then press **Play** (spacebar) to watch the animation.  For
4D cardiac data, use the **Timeline** panel to scrub through phases.

See Also
--------

- :doc:`/installation`
- :doc:`/quickstart`
- :doc:`/cli_scripts/heart_gated_ct`
- :doc:`/cli_scripts/vtk_to_usd`
- :doc:`/api/workflows`
- :doc:`/examples`
- :doc:`/troubleshooting`
