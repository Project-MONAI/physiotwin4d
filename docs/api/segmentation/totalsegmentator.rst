================
TotalSegmentator
================

.. module:: physiomotion4d.segment_chest_total_segmentator
.. currentmodule:: physiomotion4d

``SegmentChestTotalSegmentator`` groups a TotalSegmentator labelmap into the
anatomy masks used by PhysioMotion4D workflows.

Class Reference
===============

.. autoclass:: SegmentChestTotalSegmentator
   :members:
   :undoc-members:
   :show-inheritance:

Basic Usage
===========

.. code-block:: python

   import itk

   from physiomotion4d import SegmentChestTotalSegmentator

   image = itk.imread("chest_ct.nrrd")
   segmenter = SegmentChestTotalSegmentator()

   masks = segmenter.segment(image, contrast_enhanced_study=True)

   heart = masks["heart"]
   lungs = masks["lung"]
   vessels = masks["major_vessels"]
   labelmap = masks["labelmap"]

   itk.imwrite(heart, "heart_mask.nrrd")
   itk.imwrite(lungs, "lung_mask.nrrd")
   itk.imwrite(vessels, "major_vessels_mask.nrrd")
   itk.imwrite(labelmap, "labelmap.nrrd")

Returned Keys
=============

For this segmenter, ``segment()`` returns a dictionary with the following
keys:

* ``labelmap``
* ``lung``
* ``heart``
* ``major_vessels``
* ``bone``
* ``soft_tissue``
* ``other``
* ``contrast``

The dictionary should be accessed by key. Do not unpack it positionally.
The exact key set is determined by the segmenter's :class:`AnatomyTaxonomy`
and may differ from other segmenters (see :doc:`base`). For
:class:`SegmentChestTotalSegmentator` specifically, all seven groups plus
``labelmap`` are always present; downstream code that targets a different
segmenter should check membership.

Operational Notes
=================

TotalSegmentator model inference may download model assets and can be slow on a
CPU-only environment. For repeatable workflows, prefer the tutorial scripts or
the ``physiomotion4d-convert-image-to-vtk`` CLI.

See Also
========

* :doc:`index`
* :doc:`../../cli_scripts/overview`
* :doc:`../../tutorials`
