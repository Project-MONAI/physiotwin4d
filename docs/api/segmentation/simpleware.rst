==========================
Simpleware Heart Segmenter
==========================

.. module:: physiotwin4d.segment_heart_simpleware
.. currentmodule:: physiotwin4d

``SegmentHeartSimpleware`` runs Synopsys Simpleware Medical's ASCardio module
as an external process and returns the resulting heart and major-vessel masks
as ITK images.

Class Reference
===============

.. autoclass:: SegmentHeartSimpleware
   :members:
   :undoc-members:
   :show-inheritance:

Basic Usage
===========

.. code-block:: python

   import itk

   from physiotwin4d import SegmentHeartSimpleware

   image = itk.imread("chest_ct.nrrd")
   segmenter = SegmentHeartSimpleware()
   masks = segmenter.segment(image)

   heart = masks["heart"]
   vessels = masks["major_vessels"]

Returned Keys
=============

The ASCardio module segments cardiac anatomy only, so this segmenter's
taxonomy registers a subset of the groups produced by
:class:`SegmentChestTotalSegmentator`. The returned dictionary contains:

* ``labelmap``
* ``heart``
* ``major_vessels``
* ``soft_tissue`` (base-class placeholder for label id 133)
* ``contrast`` (base-class placeholder for label id 135)
* ``other`` (all unclaimed label ids in [1, 256))

Keys such as ``lung`` and ``bone`` are **not** present. Callers that need
those groups must either use a different segmenter or check membership
(``"lung" in masks``) and handle the absence explicitly. See
:doc:`base` for the full taxonomy contract.

Installation Note
=================

``SegmentHeartSimpleware`` requires a licensed installation of Synopsys
Simpleware Medical and its ASCardio module. The class invokes the Simpleware
executable as a subprocess; the executable and helper script paths must be
configured on the host system.

See Also
========

* :doc:`base`
* :doc:`totalsegmentator`
* :doc:`index`
