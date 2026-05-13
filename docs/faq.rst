===
FAQ
===

Frequently Asked Questions about PhysioMotion4D.

General Questions
=================

What is PhysioMotion4D?
-----------------------

PhysioMotion4D is a medical imaging package that converts 4D CT scans into dynamic 
3D models for visualization in NVIDIA Omniverse.

What data formats are supported?
---------------------------------

* **Input**: NRRD, MHA, NIfTI, DICOM
* **Output**: USD (Universal Scene Description), VTK

Do I need NVIDIA Omniverse?
----------------------------

No, Omniverse is optional for visualization. You can also use:

* usdview (comes with usd-core)
* PyVista
* ParaView

Installation Questions
======================

Do I need a GPU?
----------------

No. A plain ``pip install physiomotion4d`` works without a GPU. At import time
a ``UserWarning`` is emitted (visible by default in all standard Python runs):

.. code-block:: text

   CuPy is not installed — GPU acceleration is unavailable and processing will be
   slow. Re-install with uv to get CuPy and CUDA-enabled PyTorch in one step
   (pip alone will not select the correct CUDA wheel):
     uv pip install 'physiomotion4d[cuda13]'  # CUDA 13

CPU-only mode is suitable for evaluation and small datasets. For production
workloads an NVIDIA GPU is strongly recommended.

Which CUDA version is required?
--------------------------------

CUDA 13 is supported. Install the CUDA 13 extra for GPU acceleration:

.. code-block:: bash

   uv pip install "physiomotion4d[cuda13]"

The extra installs CuPy. In uv-managed source environments, PyTorch,
torchvision, and torchaudio are sourced from
``https://download.pytorch.org/whl/cu130`` by default.

What Python version is required?
---------------------------------

Python 3.11 or 3.12 are supported.

Usage Questions
===============

How long does processing take?
-------------------------------

Typical processing time for 10-frame cardiac CT (with GPU):

* 4D to 3D conversion: ~1 minute
* Registration: ~5-10 minutes
* Segmentation: ~1-2 minutes
* USD creation: ~1 minute
* **Total**: ~10-15 minutes

Which segmentation method should I use?
----------------------------------------

* **TotalSegmentator**: Fast, good quality, general purpose
* **Simpleware**: Best quality for cardiac imaging, requires Simpleware Medical

See :doc:`api/segmentation/index` for comparison.

Which registration method should I use?
----------------------------------------

* **ICON**: Recommended for cardiac/lung (fast, GPU)
* **ANTs**: Best for brain imaging and general purpose

See :doc:`api/registration/index` for comparison.

Troubleshooting
===============

See :doc:`troubleshooting` for common issues and solutions.

More Questions?
===============

* Check the :doc:`cli_scripts/heart_gated_ct`
* Browse :doc:`examples`
* Open an issue on `GitHub <https://github.com/Project-MONAI/physiomotion4d/issues>`_

