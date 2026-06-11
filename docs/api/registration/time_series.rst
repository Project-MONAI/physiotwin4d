========================
Time-Series Registration
========================

.. module:: physiomotion4d.register_time_series_images
.. currentmodule:: physiomotion4d

``RegisterTimeSeriesImages`` registers ordered 3D image phases to a reference
frame using Greedy, ICON, or combined ``Greedy_ICON`` methods.

Class Reference
===============

.. autoclass:: RegisterTimeSeriesImages
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: registration_method

Basic Usage
===========

.. code-block:: python

   import itk

   from physiomotion4d import RegisterTimeSeriesImages

   images = [itk.imread(f"phase_{idx:02d}.mha") for idx in range(10)]

   registrar = RegisterTimeSeriesImages(registration_method="Greedy")
   registrar.set_fixed_image(images[0])

   result = registrar.register_time_series(
       moving_images=images,
       reference_frame=0,
       register_reference=False,
   )
   forward_transforms = result["forward_transforms"]
   inverse_transforms = result["inverse_transforms"]

See Also
========

* :doc:`ants`
* :doc:`greedy`
* :doc:`icon`
* :doc:`../../cli_scripts/4dct_reconstruction`
