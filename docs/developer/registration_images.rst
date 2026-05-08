==================================
Image Registration Developer Guide
==================================

PhysioMotion4D image registration classes register a moving ITK image to a
fixed ITK image.

Basic Pattern
=============

.. code-block:: python

   import itk

   from physiomotion4d import RegisterImagesANTs

   fixed = itk.imread("fixed.mha")
   moving = itk.imread("moving.mha")

   registrar = RegisterImagesANTs()
   registrar.set_modality("ct")
   registrar.set_fixed_image(fixed)

   result = registrar.register(moving)
   registered = registrar.get_registered_image()

The result dictionary contains ``forward_transform``, ``inverse_transform``,
and ``loss``.

Time Series
===========

.. code-block:: python

   import itk

   from physiomotion4d import RegisterTimeSeriesImages

   images = [itk.imread(f"phase_{idx:02d}.mha") for idx in range(10)]

   registrar = RegisterTimeSeriesImages(registration_method="ants")
   registrar.set_fixed_image(images[0])
   result = registrar.register_time_series(
       moving_images=images,
       reference_frame=0,
       register_reference=False,
   )

Development Notes
=================

* Use masks when registration should focus on a specific anatomy.
* Check transform direction before applying transforms to contours or images.
* Use ``TransformTools.transform_image()`` for resampling images.
* Use ``TransformTools.transform_pvcontour()`` for PyVista contours.

See Also
========

* :doc:`../api/registration/index`
* :doc:`workflows`
