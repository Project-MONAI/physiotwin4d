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

1. Reduce image size:

   .. code-block:: python

      processor.set_downsample_factor(2)

2. Use CPU instead:

   .. code-block:: python

      processor.set_registration_device('cpu')
      processor.set_segmentation_device('cpu')

3. Reduce batch size:

   .. code-block:: python

      processor.set_batch_size(1)

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

1. Try different segmentation method:

   .. code-block:: python

      processor.set_segmentation_method('simpleware_heart')

2. Check if image is contrast-enhanced:

   .. code-block:: python

      processor = ProcessHeartGatedCT(
          ...,
          contrast_enhanced=True  # or False
      )

3. Preprocess image:

   .. code-block:: python

      processor.set_intensity_normalization(True)
      processor.set_denoise(True)

Registration Not Converging
---------------------------

**Problem**: Registration produces poor results

**Solutions**:

1. Increase iterations:

   .. code-block:: python

      processor.set_registration_iterations(200)

2. Try different method:

   .. code-block:: python

      processor.set_registration_method('ants')

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

3. Verify frame rate:

   .. code-block:: python

      processor.set_frame_rate(30)

USD File Too Large
------------------

**Problem**: USD files are very large

**Solutions**:

1. Reduce mesh complexity:

   .. code-block:: python

      processor.set_decimation_target(5000)  # Fewer triangles

2. Flatten USD stage:

   .. code-block:: python

      processor.set_flatten_usd(True)

Performance Issues
==================

Slow Processing
---------------

**Problem**: Processing takes too long

**Solutions**:

1. Use GPU acceleration:

   .. code-block:: python

      processor.set_registration_device('cuda')
      processor.set_segmentation_device('cuda')

2. Use faster segmentation:

   .. code-block:: python

      processor.set_segmentation_method('totalsegmentator')

3. Reduce registration iterations:

   .. code-block:: python

      processor.set_registration_iterations(50)

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

