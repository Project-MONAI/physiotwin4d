==========================
Image Registration Modules
==========================

.. currentmodule:: physiotwin4d

PhysioTwin4D image registration classes align moving 3D images to a fixed
3D image and return transform dictionaries.

.. toctree::
   :maxdepth: 2

   base
   ants
   greedy
   icon
   time_series

Common Result Shape
===================

``RegisterImagesANTS.register()`` and ``RegisterImagesICON.register()`` return:

* ``forward_transform``
* ``inverse_transform``
* ``loss``

Use :meth:`RegisterImagesBase.get_registered_image` after ``register()`` when a
resampled moving image is needed.

Basic Example
=============

.. code-block:: python

   import itk

   from physiotwin4d import RegisterImagesANTS

   fixed = itk.imread("reference.mha")
   moving = itk.imread("moving.mha")

   registrar = RegisterImagesANTS()
   registrar.set_modality("ct")
   registrar.set_fixed_image(fixed)

   result = registrar.register(moving)
   registered = registrar.get_registered_image()

See Also
========

* :doc:`../model_registration/index`
* :doc:`../workflows`
