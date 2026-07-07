=====================
Download Example Data
=====================

The ``physiotwin4d-download-data`` command downloads example datasets used by
PhysioTwin4D tutorials and demos.

Supported Datasets
==================

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Data name
     - Description
   * - ``Slicer-Heart-CT``
     - Public 4D cardiac CT sample from SlicerHeart. This is currently the
       only dataset downloaded automatically by PhysioTwin4D.

Basic Usage
===========

Download the default dataset into the default location:

.. code-block:: bash

   physiotwin4d-download-data

This is equivalent to:

.. code-block:: bash

   physiotwin4d-download-data Slicer-Heart-CT \
       --directory data/Slicer-Heart-CT

Options
=======

.. code-block:: bash

   physiotwin4d-download-data [Slicer-Heart-CT] [--directory DIRECTORY]

``data_name``
   Dataset to download. The only accepted value is ``Slicer-Heart-CT``.

``--directory``
   Directory where the dataset is stored. Defaults to
   ``data/Slicer-Heart-CT``.

Output
======

For ``Slicer-Heart-CT``, the command downloads or reuses:

.. code-block:: text

   data/Slicer-Heart-CT/TruncalValve_4DCT.seq.nrrd

The command uses
:meth:`physiotwin4d.data_download_tools.DataDownloadTools.DownloadSlicerHeartCTData`,
so repeated runs reuse the existing non-empty file.

See Also
========

* :doc:`byod_tutorials`
* :doc:`heart_gated_ct`
* :doc:`overview`
