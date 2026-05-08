=======
Testing
=======

Use the fast, no-real-data test subset during development:

.. code-block:: bash

   pytest tests/ -m "not slow and not requires_data" -v

Test Categories
===============

PhysioMotion4D uses pytest markers and command-line flags to keep expensive
work separate from normal development tests.

.. code-block:: bash

   # Fast development signal
   pytest tests/ -m "not slow and not requires_data" -v

   # Include tutorial execution tests
   pytest tests/test_tutorials.py --run-tutorials -v

   # Include experiment tests
   pytest tests/ --run-experiments -v

   # CLI help smoke tests
   pytest tests/test_cli_smoke.py -v

   # Public import surface
   pytest tests/test_import_public_api.py -v

Specific Areas
==============

.. code-block:: bash

   pytest tests/test_convert_vtk_to_usd.py -v
   pytest tests/test_convert_nrrd_4d_to_3d.py -v
   pytest tests/test_contour_tools.py -v
   pytest tests/test_transform_tools.py -v
   pytest tests/test_image_tools.py -v

Real Data and GPU Tests
=======================

Tests that require downloaded or manually prepared datasets are marked
``requires_data``. Tutorial tests are opt-in through ``--run-tutorials`` and
preserve tutorial dependencies, such as Tutorial 4 consuming Tutorial 3 output.

Continuous Integration
======================

CI should run the fast subset by default and keep data-heavy tutorials and
experiments behind explicit flags.

See Also
========

* :doc:`contributing`
* :doc:`tutorials`
