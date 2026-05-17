===============
Troubleshooting
===============

Common issues and solutions for PhysioMotion4D.

Installation Issues
===================

CUDA Out of Memory
------------------

**Problem**: ``RuntimeError: CUDA out of memory``

**Solutions**:

1. Resample or crop the input image before running the workflow.
2. Use ``--registration-method ANTS`` when CUDA is unavailable.
3. Process fewer frames per run.

CUDA Version Mismatch
---------------------

**Problem**: Errors such as ``cupy`` failing to import, ``torch.cuda.is_available()``
returning ``False``, or runtime messages indicating a CUDA library version conflict.

**Cause**: The installed ``cupy`` or PyTorch wheel was built for a different CUDA
version than the one present on the system.

**Solution**: Install the CUDA 13 extra:

.. code-block:: bash

   uv pip install "physiomotion4d[cuda13]"

The extra installs CuPy. In uv-managed source environments, PyTorch resolves
from the CUDA 13.0 wheel index.

Verify the active CUDA version before reinstalling:

.. code-block:: bash

   nvidia-smi   # shows driver and CUDA version

.. note::
   If you have no NVIDIA GPU, a plain ``pip install physiomotion4d`` installs a
   CPU-only build. CuPy is absent and a ``UserWarning`` is emitted at import time.
   CPU execution of all operations is supported but will be significantly slower
   than a GPU-enabled install.

Import Errors
-------------

**Problem**: ``ImportError: No module named 'itk'``

**Solution**: Reinstall with all dependencies:

.. code-block:: bash

   pip install --upgrade physiomotion4d

Processing Issues
=================

Poor Segmentation Quality
-------------------------

**Problem**: Segmentation masks are inaccurate

**Solutions**:

1. Check if image is contrast-enhanced:

   .. code-block:: python

      workflow = WorkflowConvertImageToUSD(
          ...,
          contrast_enhanced=True  # or False
      )

2. Preprocess intensity, spacing, and field of view before invoking the workflow.

Registration Not Converging
---------------------------

**Problem**: Registration produces poor results

**Solutions**:

1. Increase ``--registration-iterations`` for the heart-gated CT CLI.

2. Try different method:

   .. code-block:: bash

      physiomotion4d-convert-image-to-usd cardiac_4d.nrrd --registration-method ANTS

3. Check image orientation and spacing

USD Issues
==========

USD Not Animating
-----------------

**Problem**: USD file loads but doesn't animate

**Solutions**:

1. Validate USD file:

   .. code-block:: bash

      usdchecker model.usd

2. Check time samples:

   .. code-block:: bash

      usdview model.usd

3. Verify that the generated USD contains time samples.

USD File Too Large
------------------

**Problem**: USD files are very large

**Solutions**:

1. Reduce mesh complexity before USD export.
2. Export fewer anatomy groups or fewer time points.

Performance Issues
==================

Slow Processing
---------------

**Problem**: Processing takes too long

**Solutions**:

1. Install ``physiomotion4d[cuda13]`` with uv for CUDA acceleration.
2. Reduce ``--registration-iterations`` during exploratory runs.
3. Run tutorial workflows with reduced frame counts where supported.

Getting Help
============

If you still have issues:

1. Check :doc:`faq`
2. Search `GitHub Issues <https://github.com/Project-MONAI/physiomotion4d/issues>`_
3. Open a new issue with:

   * Python version
   * CUDA version
   * Complete error message
   * Minimal code to reproduce

