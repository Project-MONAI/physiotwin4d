===============
NV-Segment-CTMR
===============

.. module:: physiotwin4d.segment_nv_segment_ct_mri
.. currentmodule:: physiotwin4d

``SegmentNVSegmentCTMRI`` runs NVIDIA's NV-Segment-CTMR model (a VISTA3D
derivative finetuned on 30K+ CT and MRI scans) and groups its 345-class
labelmap into the anatomy masks used by PhysioTwin4D workflows.

.. warning::

   The NV-Segment-CTMR *weights* are released under the NVIDIA OneWay
   Non-Commercial License (academic research use only); the surrounding bundle
   code is Apache 2.0. Use ``SegmentChestTotalSegmentator`` or NV-Segment-CT if
   you need a commercially licensed model.

Class Reference
===============

.. autoclass:: SegmentNVSegmentCTMRI
   :members:
   :undoc-members:
   :show-inheritance:

Basic Usage
===========

.. code-block:: python

   import itk

   from physiotwin4d import SegmentNVSegmentCTMRI

   image = itk.imread("chest_ct.nrrd")
   segmenter = SegmentNVSegmentCTMRI()

   masks = segmenter.segment(image)

   heart = masks["heart"]
   lungs = masks["lung"]
   labelmap = masks["labelmap"]

   itk.imwrite(labelmap, "labelmap.nrrd", compression=True)

For MR studies, select the matching modality before calling ``segment()``:

.. code-block:: python

   segmenter = SegmentNVSegmentCTMRI()
   segmenter.set_modality("MRI_BODY")   # or "CT_BODY", "MRI_BRAIN"

``MRI_BRAIN`` expects a T1 volume that has already been skull-stripped and
affinely aligned to the LUMIR template; this class does not perform that
preprocessing.

Returned Keys
=============

For this segmenter, ``segment()`` returns a dictionary with the following
keys:

* ``labelmap``
* ``heart``
* ``major_vessels``
* ``lung``
* ``bone``
* ``soft_tissue``
* ``brain_parcellation``
* ``other``

Label Ids
=========

Unlike the other segmenters, the labelmap is ``uint16``: label ids are the
model's own published class indices (see ``configs/label_dict.json`` in
https://github.com/NVIDIA-Medtech/NV-Segment-CTMR), which run to 345. For
example, 6 is the aorta and 115 the heart. The full group→id mapping is
available through the segmenter's ``taxonomy`` attribute
(``segmenter.taxonomy.labels_in_group("heart")``,
``segmenter.taxonomy.all_labels()``).

``brain_parcellation`` is a group name this segmenter introduces. It has no
entry in :data:`physiotwin4d.usd_anatomy_tools.DEFAULT_RENDER_PARAMS`, so it
falls back to the ``other`` OmniSurface look when rendered.

Operational Notes
=================

The first call to ``segment()`` downloads ~872 MB of model weights from
https://huggingface.co/nvidia/NV-Segment-CTMR into the Hugging Face cache
(override the destination with the ``model_cache_dir`` attribute). Inference
requires a CUDA GPU.

See Also
========

* :doc:`index`
* :doc:`totalsegmentator`
* :doc:`../../tutorials`
