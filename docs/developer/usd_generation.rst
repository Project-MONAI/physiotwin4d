========================
USD Generation
========================

PhysioMotion4D uses :class:`physiomotion4d.ConvertVTKToUSD` as the
application-level API for VTK-to-USD conversion. Workflows, command-line tools,
and experiments should use this class instead of importing
``physiomotion4d.vtk_to_usd`` directly.

``physiomotion4d.vtk_to_usd`` remains a public advanced low-level package for
external users who need file readers, data containers, or direct USD writer
primitives.

ConvertVTKToUSD
================

.. autoclass:: physiomotion4d.ConvertVTKToUSD
   :members:
   :undoc-members:
   :show-inheritance:

Single File
-----------

.. code-block:: python

   from physiomotion4d import ConvertVTKToUSD

   stage = ConvertVTKToUSD.from_files(
       data_basename='Heart',
       vtk_files=['heart.vtp'],
       extract_surface=True,
   ).convert('heart.usd')

Time Series
-----------

.. code-block:: python

   from physiomotion4d import ConvertVTKToUSD

   stage = ConvertVTKToUSD.from_files(
       data_basename='Heart',
       vtk_files=['heart_t0.vtp', 'heart_t1.vtp', 'heart_t2.vtp'],
       time_codes=[0.0, 1.0, 2.0],
       times_per_second=24.0,
   ).convert('animated_heart.usd')

In-Memory Meshes
----------------

.. code-block:: python

   from physiomotion4d import ConvertVTKToUSD
   import pyvista as pv

   meshes = [pv.read(path) for path in vtk_files]
   converter = ConvertVTKToUSD(
       data_basename='CardiacModel',
       input_polydata=meshes,
       compute_normals=True,
   )
   stage = converter.convert('cardiac_model.usd')

Colormaps
---------

.. code-block:: python

   converter.set_colormap(
       color_by_array='pressure',
       colormap='viridis',
       intensity_range=(0.0, 1.0),
   )
   stage = converter.convert('pressure.usd')

Advanced Low-Level Facade
=========================

Use the low-level facade only when the high-level class is not appropriate:

.. code-block:: python

   from physiomotion4d.vtk_to_usd import convert_vtk_file

   stage = convert_vtk_file('mesh.vtp', 'mesh.usd')
