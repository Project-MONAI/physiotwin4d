============================
Segmentation Developer Guide
============================

Segmentation classes convert CT images into anatomy-group masks used by
workflows.

Current Segmentation Contract
=============================

.. code-block:: python

   import itk

   from physiomotion4d import SegmentChestTotalSegmentator

   image = itk.imread("chest_ct.nrrd")
   segmenter = SegmentChestTotalSegmentator()
   masks = segmenter.segment(image, contrast_enhanced_study=True)

   heart = masks["heart"]
   labelmap = masks["labelmap"]

Segmentation outputs are dictionaries of ITK images. Access masks by key rather
than by positional unpacking.

Implemented Segmenters
======================

* :class:`physiomotion4d.SegmentChestTotalSegmentator`
* :class:`physiomotion4d.SegmentHeartSimpleware`
* :class:`physiomotion4d.SegmentAnatomyBase`

Development Notes
=================

* New runtime segmenters should inherit from ``SegmentAnatomyBase`` or
  ``PhysioMotion4DBase``.
* Use ``log_info()`` and ``log_debug()`` inside runtime classes.
* Keep tests synthetic unless real model/data behavior is being validated.
* Mark real-data tests with ``requires_data``.

See Also
========

* :doc:`../api/segmentation/index`
* :doc:`workflows`
