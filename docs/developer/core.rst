======================
Core Developer Guide
======================

This page summarizes the core conventions for extending PhysioMotion4D.

Base Class
==========

Runtime classes inherit from :class:`physiomotion4d.PhysioMotion4DBase` and use
the shared logging methods.

.. code-block:: python

   import logging

   from physiomotion4d import PhysioMotion4DBase

   class MyWorkflow(PhysioMotion4DBase):
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

   py -m pytest tests/ -m "not slow and not requires_data" -v

After public API changes, regenerate the API map:

.. code-block:: bash

   py utils/generate_api_map.py

See Also
========

* :doc:`architecture`
* :doc:`workflows`
* :doc:`utilities`
