=================
ANTs Registration
=================

.. currentmodule:: physiomotion4d

``RegisterImagesANTs`` provides optimization-based deformable image
registration through ANTs.

Class Reference
===============

.. autoclass:: RegisterImagesANTs
   :members:
   :undoc-members:
   :show-inheritance:

Basic Registration
==================

.. code-block:: python

   import itk

   from physiomotion4d import RegisterImagesANTs

   fixed = itk.imread("reference.mha")
   moving = itk.imread("moving.mha")

   registrar = RegisterImagesANTs()
   registrar.set_modality("ct")
   registrar.set_transform_type("SyN")
   registrar.set_number_of_iterations([30, 15, 7])
   registrar.set_fixed_image(fixed)

   result = registrar.register(moving)

   forward_transform = result["forward_transform"]
   inverse_transform = result["inverse_transform"]
   registered = registrar.get_registered_image()

See Also
========

* :doc:`icon`
* :doc:`time_series`
