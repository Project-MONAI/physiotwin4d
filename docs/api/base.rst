====================================
Base Class
====================================

.. currentmodule:: physiomotion4d

``PhysioMotion4DBase`` provides the shared logging behavior used by workflow,
segmentation, registration, transform, contour, and USD helper classes.

Class Reference
===============

.. autoclass:: PhysioMotion4DBase
   :members:
   :undoc-members:
   :show-inheritance:

Logging
=======

Runtime classes should call ``log_info()``, ``log_debug()``, and
``log_warning()`` instead of printing directly. The base class also supports
global log filtering by class name.

.. code-block:: python

   import logging

   from physiomotion4d import PhysioMotion4DBase

   class MyProcessor(PhysioMotion4DBase):
       def __init__(self) -> None:
           super().__init__(class_name="MyProcessor", log_level=logging.INFO)

       def process(self) -> None:
           self.log_info("Starting processing")
           self.log_debug("Detailed diagnostic state")
           self.log_warning("Recoverable issue")

   processor = MyProcessor()
   processor.process()

   PhysioMotion4DBase.set_log_classes(["MyProcessor"])
   PhysioMotion4DBase.set_log_all_classes()

Extension Notes
===============

New runtime classes should inherit from ``PhysioMotion4DBase`` and pass a
``class_name`` plus ``log_level`` to ``super().__init__``. Standalone scripts,
data containers, and small pure utility functions do not need to inherit from
the base class.

See Also
========

* :doc:`workflows`
* :doc:`../developer/architecture`
* :doc:`../developer/extending`
