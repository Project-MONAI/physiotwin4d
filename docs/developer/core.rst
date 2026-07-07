======================
Core Developer Guide
======================

This page summarizes the core conventions for extending PhysioTwin4D.

Base Class
==========

Runtime classes inherit from :class:`physiotwin4d.PhysioTwin4DBase` and use
the shared logging methods.

.. code-block:: python

   import logging

   from physiotwin4d import PhysioTwin4DBase

   class MyWorkflow(PhysioTwin4DBase):
       def __init__(self, input_file: str) -> None:
           super().__init__(class_name="MyWorkflow", log_level=logging.INFO)
           self.input_file = input_file

       def process(self) -> str:
           self.log_info("Processing %s", self.input_file)
           return self.input_file

Use ``log_info()``, ``log_debug()``, and ``log_warning()`` inside runtime
classes. Standalone scripts may use ``print()`` for command-line status.

Class Boundaries
================

* Workflows orchestrate complete pipelines.
* Registration classes estimate transforms.
* Segmentation classes return ITK images or dictionaries of ITK masks.
* ``TransformTools`` applies transforms to images and PyVista contours.
* ``ContourTools`` creates and transforms VTK/PyVista surface data.
* ``USDTools`` and ``USDAnatomyTools`` operate on USD stages and files.

Public APIs should be documented in ``docs/api`` and reflected in
``docs/API_MAP.md`` after regeneration.

Validation
==========

For most code changes, run:

.. code-block:: bash

   py -m pytest tests/ -v

(Slow / GPU / Simpleware / PhysicsNeMo / experiment / tutorial tests are
auto-skipped; opt in with ``--run-slow``, ``--run-gpu``, ``--run-simpleware``,
``--run-physicsnemo``, ``--run-experiments``, ``--run-tutorials``, or use
``--run-all`` to enable every bucket at once. Data-dependent tests download
their data through the session fixtures and run by default.)

After public API changes, regenerate the API map:

.. code-block:: bash

   py utils/generate_api_map.py

See Also
========

* :doc:`architecture`
* :doc:`workflows`
* :doc:`utilities`
