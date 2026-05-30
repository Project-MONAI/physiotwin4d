============
Contributing
============

Thank you for your interest in contributing to PhysioMotion4D! This guide will help you get started.

Ways to Contribute
===================

* Report bugs and issues
* Suggest new features
* Improve documentation
* Submit code contributions
* Share example workflows

Getting Started
===============

1. **Fork the repository** on GitHub
2. **Clone your fork**:

   .. code-block:: bash

      git clone https://github.com/YOUR_USERNAME/PhysioMotion4D.git
      cd PhysioMotion4D

3. **Create a virtual environment**:

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate

4. **Install in development mode**:

   .. code-block:: bash

      pip install -e ".[dev]"

   To install the full developer environment with CUDA 13, documentation, test,
   and development dependencies, use either of these equivalent uv forms:

   .. code-block:: bash

      uv pip install -e ".[cuda13,docs,test,dev]"
      uv pip install -e . --all-extras

5. **Install pre-commit hooks**:

   .. code-block:: bash

      pre-commit install

IDE Setup (VS Code / Cursor)
==============================

Recommended Extensions
----------------------

For the best development experience with VS Code or Cursor, install these extensions:

**Required:**

* `charliermarsh.ruff <https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff>`_ - Ruff linting and formatting
* `ms-python.python <https://marketplace.visualstudio.com/items?itemName=ms-python.python>`_ - Python language support
* `ms-python.vscode-pylance <https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance>`_ - IntelliSense and type checking

**Recommended:**

* `ms-python.debugpy <https://marketplace.visualstudio.com/items?itemName=ms-python.debugpy>`_ - Python debugger
* `njpwerner.autodocstring <https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring>`_ - Generate docstrings automatically

**Not Needed (Replaced by Ruff):**

* ms-python.black-formatter - No longer needed
* ms-python.isort - No longer needed
* ms-python.flake8 - No longer needed
* ms-python.pylint - No longer needed

VS Code Settings
----------------

The repository includes `.vscode/settings.json` with optimal configuration. Key settings:

.. code-block:: json

   {
     "[python]": {
       "editor.defaultFormatter": "charliermarsh.ruff",
       "editor.formatOnSave": true,
       "editor.codeActionsOnSave": {
         "source.fixAll": "explicit",
         "source.organizeImports": "explicit"
       },
       "editor.rulers": [88]
     },
     "ruff.enable": true,
     "python.analysis.typeCheckingMode": "basic"
   }

This configuration:

* Uses Ruff for all formatting and linting
* Automatically formats code on save
* Organizes imports automatically
* Shows a ruler at 88 characters (line length limit)
* Enables basic type checking with Pylance

Experiment Scripts
------------------

Experiments in the ``experiments/`` directory are ``# %%`` percent-format
Python scripts. They run end-to-end as plain Python (``python <script>.py``)
or cell-by-cell via the VS Code / Cursor Python extension's "Run Cell"
feature.

* Ruff formats these cell-separated scripts automatically
* Type checking is less strict in experiment scripts (expected for exploratory work)

First-Time Setup Checklist
---------------------------

After cloning the repository:

1. Install Python 3.11+ and create virtual environment
2. Install development dependencies: ``pip install -e ".[dev]"``,
   or install all extras with
   ``uv pip install -e ".[cuda13,docs,test,dev]"`` or
   ``uv pip install -e . --all-extras``
3. Install pre-commit hooks: ``pre-commit install``
4. Install Ruff extension in VS Code/Cursor
5. Remove old formatter extensions (black, isort, flake8, pylint)
6. Verify settings: Open a Python file and save to test auto-formatting
7. Run tests: ``pytest tests/ -m "not slow"`` to verify setup

Code Style
==========

PhysioMotion4D follows strict code quality standards using modern, fast tooling.

Formatting and Linting with Ruff
---------------------------------

We use **Ruff** for all formatting and linting (line length: 88, double quotes):

.. code-block:: bash

   # Check and fix linting issues
   ruff check . --fix

   # Format code
   ruff format .

   # Check without making changes
   ruff check . --diff
   ruff format --check .

Type Checking with mypy
------------------------

We use **mypy** for static type checking:

.. code-block:: bash

   # Run type checking
   mypy src/

Pre-commit Hooks
----------------

Run all checks automatically before committing:

.. code-block:: bash

   # Run on all files
   pre-commit run --all-files

   # Run on staged files only
   pre-commit run

The pre-commit hooks will automatically:

* Run Ruff linter with auto-fixes
* Run Ruff formatter
* Run mypy type checking (on push)
* Run fast unit tests (on push)

Development Workflow
====================

1. **Create a feature branch**:

   .. code-block:: bash

      git checkout -b feature/amazing-feature

2. **Make your changes** following code style guidelines

3. **Add tests** for new functionality

4. **Run tests**:

   .. code-block:: bash

      pytest tests/

5. **Commit your changes**:

   .. code-block:: bash

      git add .
      git commit -m "Add amazing feature"

6. **Push to your fork**:

   .. code-block:: bash

      git push origin feature/amazing-feature

7. **Open a Pull Request** on GitHub

Pull Request Guidelines
========================

* **Clear description**: Explain what and why
* **Reference issues**: Link related issues with #123
* **Pass all tests**: CI must pass
* **Update documentation**: Document new features
* **Add release note**: Document user-facing changes in the pull request

Testing
=======

Write Tests
-----------

Add tests in the ``tests/`` directory:

.. code-block:: python

   # tests/test_my_feature.py
   import pytest
   from physiomotion4d import MyNewFeature

   def test_my_feature():
       feature = MyNewFeature()
       result = feature.do_something()
       assert result == expected_value

Run Tests
---------

.. code-block:: bash

   # Run all tests
   pytest tests/

   # Run specific test file
   pytest tests/test_my_feature.py -v

   # Run with coverage
   pytest tests/ --cov=src/physiomotion4d --cov-report=html

   # Default invocation auto-skips slow/GPU/Simpleware/experiment/tutorial
   pytest tests/

   # Opt into specific buckets
   pytest tests/ --run-slow
   pytest tests/ --run-gpu --run-slow   # typical local GPU profile
   pytest tests/ --run-physicsnemo      # needs [physicsnemo] extra; requires Python >= 3.11
   # --run-all turns on every --run-* bucket at once (used by self-hosted CI):
   pytest tests/ --run-all

Documentation
=============

Documentation is built with Sphinx and hosted on ReadTheDocs.

Build Docs Locally
-------------------

.. code-block:: bash

   # Install documentation dependencies
   pip install -e ".[docs]"

   # Build HTML documentation
   cd docs
   make html

   # Open in browser
   open _build/html/index.html  # macOS
   xdg-open _build/html/index.html  # Linux
   start _build/html/index.html  # Windows

Documentation Style
-------------------

* Use **reStructuredText** (.rst) for documentation
* Follow existing structure and formatting
* Include code examples with proper syntax highlighting
* Add docstrings to all public classes and methods

Contributing Scripts vs Experiments
------------------------------------

When contributing new workflows or examples:

**Production Code (src/physiomotion4d/cli/):**

* **DO contribute here** for production-ready CLI implementations
* Must include proper error handling and validation
* Should follow all code style and testing requirements
* Serves as definitive usage examples for users
* Will be referenced in documentation

**Research Code (experiments/ directory):**

* **May contribute here** for exploratory research and design experiments
* Can have hardcoded paths and minimal error handling
* Should document what was learned and how it informed production code
* Helps others understand adaptation possibilities for new domains
* Should reference corresponding production implementation in CLI commands or ``src/physiomotion4d/cli/``

Docstring Format
----------------

Use Google-style docstrings:

.. code-block:: python

   def my_function(param1: str, param2: int) -> bool:
       """Brief description of function.

       Longer description with more details about what the function does,
       any important notes, and usage examples.

       Args:
           param1: Description of first parameter
           param2: Description of second parameter

       Returns:
           Description of return value

       Raises:
           ValueError: When something goes wrong
           RuntimeError: When something else fails

       Example:
           >>> result = my_function("test", 42)
           >>> print(result)
           True
       """
       return True

Code Review Process
===================

All contributions go through code review:

1. **Automated checks** run via GitHub Actions
2. **Maintainer review** for code quality and design
3. **Feedback** may request changes
4. **Approval** and merge when ready

Review Criteria
---------------

* **Correctness**: Does it work as intended?
* **Code quality**: Is it clean and well-structured?
* **Tests**: Are there adequate tests?
* **Documentation**: Is it properly documented?
* **Performance**: Are there any performance concerns?
* **Compatibility**: Does it maintain backwards compatibility?

Reporting Issues
================

Bug Reports
-----------

When reporting bugs, include:

* **Python version**
* **PhysioMotion4D version**
* **Operating system**
* **GPU/CUDA version** (if applicable)
* **Minimal code** to reproduce
* **Error messages** and stack traces
* **Expected vs actual behavior**

Feature Requests
----------------

When suggesting features:

* **Clear description** of the feature
* **Use cases** and motivation
* **Proposed API** or interface
* **Potential challenges** or limitations

Release Process
===============

Versioning
----------

PhysioMotion4D uses calendar versioning: ``YYYY.0M.PATCH``

* **YYYY**: Year
* **0M**: Zero-padded month
* **PATCH**: Patch number within month

Example: ``{{ pm4d_project_version }}``

Making a Release
----------------

Maintainers only:

.. code-block:: bash

   # Bump version
   bumpver update --patch

   # Build package
   python -m build

   # Upload to PyPI
   python -m twine upload dist/*

See ``docs/PYPI_RELEASE_GUIDE.md`` for detailed maintainer release
instructions.

Community Guidelines
====================

* **Be respectful** and professional
* **Be constructive** in feedback
* **Be patient** with reviews
* **Help others** in discussions
* **Share knowledge** and examples

Getting Help
============

* **GitHub Issues**: Report bugs and request features
* **GitHub Discussions**: Ask questions and share ideas
* **Documentation**: Check the docs first
* **Code of Conduct**: Follow community guidelines

License
=======

By contributing, you agree that your contributions will be licensed under the
Apache 2.0 License.

Acknowledgments
===============

Thank you to all contributors who help make PhysioMotion4D better!

See Also
========

* :doc:`architecture` - System architecture
* :doc:`testing` - Testing guide
* `GitHub Repository <https://github.com/Project-MONAI/physiomotion4d>`_
