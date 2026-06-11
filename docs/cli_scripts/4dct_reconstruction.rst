====================================
High-Resolution 4D CT Reconstruction
====================================

The ``physiomotion4d-reconstruct-highres-4d-ct`` command reconstructs a
high-resolution 4D CT time series from ordered phase images and a
high-resolution reference image.

Input Requirements
==================

* Ordered 3D phase images, such as ``.mha``, ``.mhd``, ``.nrrd``, or
  ``.nii.gz`` files.
* A fixed high-resolution reference image.
* Optional fixed and moving masks for registration focus.

DirLab-4DCT data cannot be downloaded automatically by PhysioMotion4D. Prepare
it manually before using the DirLab tutorial or examples.

Basic Usage
===========

.. code-block:: bash

   physiomotion4d-reconstruct-highres-4d-ct \
       --time-series-images frame_*.mha \
       --fixed-image highres_reference.mha \
       --output-dir ./results

With Upsampling
===============

.. code-block:: bash

   physiomotion4d-reconstruct-highres-4d-ct \
       --time-series-images frame_000.mha frame_001.mha frame_002.mha \
       --fixed-image highres_reference.mha \
       --reference-frame 0 \
       --upsample \
       --output-dir ./results

Registration Options
====================

.. code-block:: bash

   physiomotion4d-reconstruct-highres-4d-ct \
       --time-series-images frame_*.mha \
       --fixed-image highres_reference.mha \
       --registration-method Greedy_ICON \
       --Greedy-iterations 30 15 7 3 \
       --prior-weight 0.5 \
       --output-dir ./results

Outputs
=======

The command writes reconstructed images to ``--output-dir`` using
``--output-prefix`` as the filename prefix. Use ``--save-transforms`` and
``--save-losses`` when registration diagnostics are needed.

Python API
==========

Use :class:`physiomotion4d.WorkflowReconstructHighres4DCT` for programmatic
access.

Related Pages
=============

* :doc:`../tutorials`
* :doc:`overview`
* :doc:`../api/workflows`
