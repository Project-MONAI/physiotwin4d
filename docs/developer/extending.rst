========================
Extending PhysioMotion4D
========================

PhysioMotion4D is an early-alpha toolkit. Prefer small, explicit extensions
that match the current class boundaries over large compatibility layers.

Where to Start
==============

* **Workflows**: add or modify complete pipelines.
* **CLI wrappers**: expose workflow classes for repeatable command-line use.
* **Utilities**: add focused image, contour, transform, or USD helpers.
* **Experiments**: keep exploratory research in ``experiments/``. Experiments
  document prior and ongoing research used to define the toolkit, but they are
  not examples for users or developers.

Runtime Class Pattern
=====================

.. code-block:: python

   import logging

   from physiomotion4d import PhysioMotion4DBase

   class MyWorkflow(PhysioMotion4DBase):
       def __init__(self, input_file: str, log_level: int | str = logging.INFO):
           super().__init__(class_name="MyWorkflow", log_level=log_level)
           self.input_file = input_file

       def process(self) -> str:
           self.log_info("Processing %s", self.input_file)
           return self.input_file

CLI Wrapper Pattern
===================

.. code-block:: python

   import argparse
   import sys

   from my_module import MyWorkflow

   def main() -> int:
       parser = argparse.ArgumentParser(description="Run my workflow")
       parser.add_argument("--input-file", required=True)
       args = parser.parse_args()

       workflow = MyWorkflow(input_file=args.input_file)
       workflow.process()
       return 0

   if __name__ == "__main__":
       sys.exit(main())

Documentation Requirements
==========================

* Update docstrings for changed public methods.
* Update or add examples that use the actual current API.
* Regenerate ``docs/API_MAP.md`` after public API changes.
* Avoid documenting planned APIs as if they are installed.

Testing Requirements
====================

Use synthetic ITK images and small PyVista meshes where possible. Mark tests
that require downloaded or manually prepared data with ``requires_data``.

.. code-block:: bash

   py -m pytest tests/ -m "not slow and not requires_data" -v

See Also
========

* :doc:`architecture`
* :doc:`workflows`
* :doc:`utilities`
