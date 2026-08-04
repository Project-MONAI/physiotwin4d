==================
Viewing USD Files
==================

Every USD-producing workflow in PhysioTwin4D — Tutorials 1, 5 and 10, and the
``physiotwin4d-convert-image-to-usd`` and ``physiotwin4d-convert-vtk-to-usd``
commands — writes an OpenUSD scene: anatomy split into per-organ prims, painted
with OmniSurface materials, and time-sampled when the input was a series. To
see the motion you need a USD viewer.

Two are worth knowing. Use **usdview** for day-to-day inspection and debugging,
and an **Omniverse Kit application** when you want real-time ray tracing or to
build on the scene.

.. important::

   ``pip install physiotwin4d`` pulls in `usd-core
   <https://pypi.org/project/usd-core/>`_, which is the OpenUSD *libraries*
   only — enough to write and read stages, but it contains no viewer.
   ``usdview`` needs a build that includes USD Imaging, from one of the two
   routes below.

usdview
=======

``usdview`` is the canonical viewer that ships with OpenUSD. It is a
lightweight application for opening a stage, walking its scene graph,
inspecting prim properties and composition, scrubbing the timeline, and
switching between renderers — the tool to reach for when you want to know what
is actually in the file. See the `OpenUSD toolset documentation
<https://openusd.org/release/toolset.html>`_ for the full feature list.

Getting it: pre-built binaries
------------------------------

The quickest route is NVIDIA's pre-built OpenUSD libraries and tools, which
include ``usdview`` for Windows and Linux and are matched to specific Python
versions: https://developer.nvidia.com/usd

Download the package matching your Python version, unpack it, and put its
``bin`` and ``lib`` directories on your path. On Windows:

.. code-block:: bat

   set USD_ROOT=C:\usd
   set PATH=%USD_ROOT%\bin;%USD_ROOT%\lib;%PATH%
   set PYTHONPATH=%USD_ROOT%\lib\python;%PYTHONPATH%

   usdview tutorials\output\tutorial_01_heart\cardiac_model.usd

On Linux:

.. code-block:: bash

   export USD_ROOT=$HOME/usd
   export PATH=$USD_ROOT/bin:$PATH
   export PYTHONPATH=$USD_ROOT/lib/python:$PYTHONPATH

   usdview tutorials/output/tutorial_01_heart/cardiac_model.usd

Use a **separate environment** from the one PhysioTwin4D runs in, or at least
be deliberate about ordering: the ``PYTHONPATH`` above puts a second copy of
``pxr`` ahead of the ``usd-core`` wheel, and mixing two OpenUSD builds in one
interpreter causes import errors that are tedious to diagnose.

Getting it: building from source
--------------------------------

To build OpenUSD yourself — needed if no pre-built package matches your Python,
or you want a specific release — clone
https://github.com/PixarAnimationStudios/OpenUSD and run its build script:

.. code-block:: bash

   git clone https://github.com/PixarAnimationStudios/OpenUSD.git
   python OpenUSD/build_scripts/build_usd.py ~/usd

The script fetches and builds the dependencies as well, so expect it to take a
while. USD Imaging and ``usdview`` are included by default; ``usdview`` also
needs PySide and PyOpenGL in the Python environment you launch it from. The
repository's build instructions list the per-platform prerequisites.

Using it
--------

.. code-block:: bash

   usdview cardiac_model.usd

- The **viewport** opens on frame one. Press the play button, or scrub the
  timeline at the bottom, to see the cardiac or respiratory motion — a static
  scene means the workflow wrote a single time sample.
- The **scene graph** on the left is the anatomy hierarchy the workflow built
  (``/World/<name>/<group>/<organ>``). Select a prim to isolate an organ.
- The **property panel** shows the attributes on the selected prim, including
  the time-sampled ``points`` that carry the motion and the bound material.
- The **interpreter** (``Window > Interpreter``) gives you a Python prompt on
  the live stage, which is the fastest way to check an attribute's values at a
  given time code.

For a non-interactive sanity check that a file is valid USD, the same toolset
ships ``usdchecker``:

.. code-block:: bash

   usdchecker cardiac_model.usd

Omniverse Kit applications
==========================

NVIDIA Omniverse is built on OpenUSD and renders it with RTX in real time. Use
it when you want photorealistic playback of the anatomy, to compose a
PhysioTwin4D scene with other assets, or to drive a downstream simulation or
XR workflow rather than to inspect the file.

Download the Omniverse launcher and applications from
https://www.nvidia.com/en-us/omniverse/download/. The relevant Kit-based apps
are **USD Composer** for authoring and layout, and **USD Presenter** for
review and playback; the Kit SDK and the `USD Viewer
<https://docs.omniverse.nvidia.com/>`_ sample are the starting points if you
want to embed a viewer in your own tool.

Omniverse needs an RTX-capable NVIDIA GPU and a current driver, which is a
heavier requirement than ``usdview``'s GL preview — that is the main reason to
keep both around.

Opening a PhysioTwin4D scene:

1. Launch **USD Composer** (or **USD Presenter**).
2. ``File > Open`` and select the generated ``.usd`` file — for the tutorials,
   under ``tutorials/output/<tutorial_name>/``.
3. Press **Play** on the timeline to run the animation. The frame rate is the
   ``frames_per_second`` the workflow was given, so a value of ``1.0`` plays one
   phase per second; raise it for smoother playback.
4. Anatomy materials are already bound, so the organs arrive colored. Select a
   prim in the stage tree to adjust its material, or to hide organs that
   occlude the structure you care about.

If a scene opens but appears empty, check the units and the camera: the
workflows write millimetre-scale geometry in USD's right-handed Y-up frame, and
a viewport whose near plane is set for metre-scale content will clip it. See
:doc:`developer/usd_generation` for the conversion details.

Before USD: viewing the meshes directly
=======================================

The intermediate ``.vtp`` and ``.vtu`` files that Tutorials 4, 6, 7, 8 and 9
write need no USD tooling at all — PyVista, already a dependency, opens them:

.. code-block:: python

   import pyvista as pv

   pv.read("tutorials/output/tutorial_04_heart/patient_surfaces.vtp").plot()

This is usually the faster way to check a segmentation or a fitted shape model
before spending time on the USD export.

See Also
========

* :doc:`tutorials` — the workflows that produce these scenes
* :doc:`cli_scripts/vtk_to_usd` — converting existing meshes to USD
* :doc:`developer/usd_generation` — coordinate frames, materials, time samples
* :doc:`troubleshooting` — when a scene does not play or looks wrong
