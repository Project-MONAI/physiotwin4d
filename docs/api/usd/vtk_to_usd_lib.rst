===============================
Low-Level vtk_to_usd Subpackage
===============================

.. module:: physiotwin4d.vtk_to_usd
.. currentmodule:: physiotwin4d.vtk_to_usd

``physiotwin4d.vtk_to_usd`` is a stable low-level API for advanced external
users. Inside this repository (experiments, workflows, CLIs, tutorials,
tests), use :class:`~physiotwin4d.ConvertVTKToUSD` from
:doc:`vtk_conversion` instead of importing this subpackage directly.

This subpackage exposes the readers, data containers, coordinate helpers, and
USD primitive writers that back ``ConvertVTKToUSD``. The public symbols are
documented in their defining submodules below; ``__init__`` re-exports them
for convenience.

File Facade
===========

.. automodule:: physiotwin4d.vtk_to_usd.converter
   :members:
   :undoc-members:
   :show-inheritance:

Data Structures
===============

.. automodule:: physiotwin4d.vtk_to_usd.data_structures
   :members:
   :undoc-members:
   :show-inheritance:

VTK Readers
===========

.. automodule:: physiotwin4d.vtk_to_usd.vtk_reader
   :members:
   :undoc-members:
   :show-inheritance:

USD Mesh Conversion
===================

.. automodule:: physiotwin4d.vtk_to_usd.usd_mesh_converter
   :members:
   :undoc-members:
   :show-inheritance:

Material Manager
================

.. automodule:: physiotwin4d.vtk_to_usd.material_manager
   :members:
   :undoc-members:
   :show-inheritance:

Mesh Utilities
==============

.. automodule:: physiotwin4d.vtk_to_usd.mesh_utils
   :members:
   :undoc-members:
   :show-inheritance:

USD Coordinate and Primvar Helpers
==================================

.. automodule:: physiotwin4d.vtk_to_usd.usd_utils
   :members:
   :undoc-members:
   :show-inheritance:

Primvar Derivations
===================

.. automodule:: physiotwin4d.vtk_to_usd.primvar_derivations
   :members:
   :undoc-members:
   :show-inheritance:

See Also
========

* :doc:`vtk_conversion`
* :doc:`../workflows`
