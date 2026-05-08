========================
Segmentation Base Class
========================

.. currentmodule:: physiomotion4d

``SegmentAnatomyBase`` defines the shared chest-anatomy segmentation contract
used by PhysioMotion4D segmentation implementations.

Class Reference
===============

.. autoclass:: SegmentAnatomyBase
   :members:
   :undoc-members:
   :show-inheritance:

Segmentation Contract
=====================

Concrete segmenters accept an ITK image and return a dictionary of ITK images:

.. code-block:: python

   import itk

   from physiomotion4d import SegmentChestTotalSegmentator

   image = itk.imread("chest_ct.nrrd")
   segmenter = SegmentChestTotalSegmentator()
   masks = segmenter.segment(image, contrast_enhanced_study=True)

   heart = masks["heart"]
   labelmap = masks["labelmap"]

Returned keys include ``labelmap``, ``lung``, ``heart``, ``major_vessels``,
``bone``, ``soft_tissue``, ``other``, and ``contrast``.

Extending Segmentation
======================

New runtime segmentation classes should inherit from ``SegmentAnatomyBase`` or
``PhysioMotion4DBase``, use ``log_info()`` / ``log_debug()``, and document the
returned mask keys. Keep synthetic tests small and mark real-data tests with
``requires_data``.

See Also
========

* :doc:`totalsegmentator`
* :doc:`index`
* :doc:`../../developer/segmentation`
