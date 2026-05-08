=====================
VTK to USD Conversion
=====================

The ``physiomotion4d-convert-vtk-to-usd`` command converts VTK, VTP, or VTU
mesh files to USD for Omniverse visualization. Multiple input files are treated
as a time series.

Basic Usage
===========

.. code-block:: bash

   physiomotion4d-convert-vtk-to-usd heart.vtp \
       --output heart.usd

Time Series
===========

.. code-block:: bash

   physiomotion4d-convert-vtk-to-usd heart_*.vtp \
       --output heart_animation.usd \
       --fps 30

Appearance Options
==================

Solid color:

.. code-block:: bash

   physiomotion4d-convert-vtk-to-usd heart.vtp \
       --output heart_red.usd \
       --appearance solid \
       --color 1 0 0

Anatomy material:

.. code-block:: bash

   physiomotion4d-convert-vtk-to-usd heart.vtp \
       --output heart_material.usd \
       --appearance anatomy \
       --anatomy-type heart

Colormap from a VTK point data array:

.. code-block:: bash

   physiomotion4d-convert-vtk-to-usd frame_*.vtk \
       --output stress.usd \
       --appearance colormap \
       --primvar vtk_point_stress_c0 \
       --cmap viridis \
       --intensity-range 0 500

Splitting
=========

By default, meshes are split by connected component. Use ``--no-split`` to keep
one mesh, or ``--by-cell-type`` to split by cell type.

.. code-block:: bash

   physiomotion4d-convert-vtk-to-usd mesh.vtu \
       --output mesh.usd \
       --by-cell-type

Python API
==========

Use :class:`physiomotion4d.WorkflowConvertVTKToUSD` for the workflow API and
:class:`physiomotion4d.ConvertVTKToUSD` for direct in-memory conversion.

Related Pages
=============

* :doc:`overview`
* :doc:`../api/usd/vtk_conversion`
* :doc:`../developer/usd_generation`
