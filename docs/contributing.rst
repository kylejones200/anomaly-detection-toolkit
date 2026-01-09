Contributing
============

Contributions are welcome! This document provides guidelines for contributing.

Getting Started
---------------

1. Fork the repository on GitHub
2. Clone your fork locally
3. Install in development mode:

.. code-block:: bash

   pip install -e ".[deep]"
   ./setup-pre-commit.sh

Development Setup
-----------------

The project uses pre-commit hooks for code quality:

.. code-block:: bash

   pip install pre-commit
   pre-commit install
   pre-commit install --hook-type pre-push

Code Style
----------

The project uses:

* **Black** for code formatting (line length: 100)
* **isort** for import sorting
* **flake8** for linting
* **mypy** for type checking (optional)

Run checks manually:

.. code-block:: bash

   black src/ tests/ examples/
   isort src/ tests/ examples/
   flake8 src/ tests/ examples/
   mypy src/

Testing
-------

Add tests for new features:

.. code-block:: bash

   pytest tests/
   pytest tests/ -v --cov=src/anomaly_detection_toolkit

Pull Request Process
--------------------

1. Create a feature branch from `main`
2. Make your changes
3. Ensure all tests pass
4. Ensure pre-commit hooks pass
5. Update documentation if needed
6. Submit a pull request

Documentation
-------------

When adding new features:

1. Add docstrings following Google/NumPy style
2. Update relevant user guide sections
3. Add examples if applicable
4. Update API reference (automatic with autodoc)

Reporting Issues
----------------

When reporting bugs:

1. Use the issue tracker on GitHub
2. Include a minimal reproducible example
3. Describe expected vs actual behavior
4. Include environment details (Python version, OS, etc.)
