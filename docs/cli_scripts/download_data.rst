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
     - Public 4D cardiac CT sample from SlicerHeart.
   * - ``KCL-Heart-Model``
     - King's College London four-chamber heart model dataset: 20
       individual heart meshes plus an average mesh, from Zenodo.
   * - ``CHOP-Valve4D``
     - CHOP Jolley Lab transcatheter pulmonary valve model, converted from
       the original FEBio model to VTK/ITK and segmented with Simpleware,
       from the PhysioTwin4D GitHub release. See
       ``data/CHOP-Valve4D/README.md``.

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

   physiotwin4d-download-data [Slicer-Heart-CT|KCL-Heart-Model|CHOP-Valve4D] [--directory DIRECTORY]

``data_name``
   Dataset to download. One of ``Slicer-Heart-CT``, ``KCL-Heart-Model``, or
   ``CHOP-Valve4D``.

``--directory``
   Directory where the dataset is stored. Defaults to ``data/<data_name>``.

Output
======

For ``Slicer-Heart-CT``, the command downloads or reuses:

.. code-block:: text

   data/Slicer-Heart-CT/TruncalValve_4DCT.seq.nrrd

The command uses
:meth:`physiotwin4d.data_download_tools.DataDownloadTools.DownloadSlicerHeartCTData`,
so repeated runs reuse the existing non-empty file.

For ``KCL-Heart-Model``, the command downloads, extracts, and reuses:

.. code-block:: text

   data/KCL-Heart-Model/average_mesh.vtk
   data/KCL-Heart-Model/input_meshes/01.vtk ... 20.vtk

The command uses
:meth:`physiotwin4d.data_download_tools.DataDownloadTools.DownloadKCLHeartModelData`,
which fetches each per-model ``.tar.gz`` archive from Zenodo, extracts its
mesh, and skips archives whose target ``.vtk`` file is already present.

For ``CHOP-Valve4D``, the command downloads, extracts, and reuses:

.. code-block:: text

   data/CHOP-Valve4D/Alterra/   (valve mesh time series, >1 GB)
   data/CHOP-Valve4D/TPV25/     (valve mesh time series, >1 GB)
   data/CHOP-Valve4D/CT/        (source CT volume and Simpleware segmentation)

The command uses
:meth:`physiotwin4d.data_download_tools.DataDownloadTools.DownloadCHOPValve4DData`,
which fetches each subdirectory's zip archive from the PhysioTwin4D GitHub
release and skips a subdirectory once it has its expected extracted files
(the CT volume or Simpleware segmentation for ``CT/``, ``.vtk`` meshes for
``Alterra/`` and ``TPV25/``) — a subdirectory left behind by an interrupted
extraction is re-downloaded rather than treated as complete.

See Also
========

* :doc:`../tutorials` — Tutorials 1-4 use ``Slicer-Heart-CT`` and
  ``KCL-Heart-Model``; ``DirLab-4DCT`` (Tutorial 6) is manual-only, see
  ``data/DirLab-4DCT/README.md``.
* :doc:`byod_tutorials`
* :doc:`heart_gated_ct`
* :doc:`overview`
