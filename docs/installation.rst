============
Installation
============

Requirements
============

**Minimum Requirements:**

- Python ≥ 3.8
- NumPy ≥ 1.19.0
- SciPy ≥ 1.5.0

**Optional Dependencies:**

- Matplotlib ≥ 3.3.0 (for plotting)
- pytest ≥ 7.0.0 (for testing)
- sphinx ≥ 4.0.0 (for documentation)

Installation Methods
====================

1. Basic Installation
---------------------

Install core package only (minimal dependencies):

.. code-block:: bash

   pip install -e .

This installs PyMARS with NumPy and SciPy only.

2. Development Installation
----------------------------

Includes testing and code quality tools:

.. code-block:: bash

   pip install -e ".[dev]"

Installs:
   - pytest, pytest-cov (testing)
   - black (code formatting)
   - flake8 (linting)
   - mypy (type checking)

3. With Plotting Support
------------------------

Add visualization capabilities:

.. code-block:: bash

   pip install -e ".[plot]"

Adds matplotlib for all plotting functions.

4. Complete Installation
------------------------

Everything including documentation:

.. code-block:: bash

   pip install -e ".[dev,plot,docs]"

Installs all optional dependencies.

From GitHub
===========

Clone and install in development mode:

.. code-block:: bash

   git clone https://github.com/abder111/pymars.git
   cd pymars
   pip install -e .

For development:

.. code-block:: bash

   git clone https://github.com/abder111/pymars.git
   cd pymars
   pip install -e ".[dev,plot]"

Verification
============

Verify correct installation:

.. code-block:: python

   from pymars import MARS
   import numpy as np
   
   # Quick test
   X = np.random.randn(100, 3)
   y = X[:, 0] + np.sin(X[:, 1]) + np.random.randn(100)*0.1
   
   model = MARS(max_terms=10)
   model.fit(X, y)
   
   print(f"R² score: {model.score(X, y):.4f}")
   print("✅ PyMARS installed successfully!")

Expected output should show an R² score around 0.7-0.9 for this synthetic data.

Running Tests
=============

Run the test suite:

.. code-block:: bash

   cd pymars
   pytest tests/ -v

With coverage report:

.. code-block:: bash

   pytest tests/ --cov=pymars --cov-report=html

View coverage in your browser: ``open htmlcov/index.html``

Running Examples
================

Interactive demonstration:

.. code-block:: bash

   cd examples
   python demo_regression.py

This runs interactive examples with real and synthetic datasets.

Troubleshooting
===============

Import Error
------------

If you get ``ModuleNotFoundError: No module named 'pymars'``:

1. Verify installation in editable mode:

   .. code-block:: bash

      pip install -e .

2. Check your Python interpreter:

   .. code-block:: bash

      which python

3. List installed packages:

   .. code-block:: bash

      pip list | grep pymars

NumPy/SciPy Build Issues
------------------------

On Ubuntu/Debian:

.. code-block:: bash

   sudo apt-get install python3-dev
   pip install --upgrade numpy scipy

On macOS:

.. code-block:: bash

   brew install openblas
   pip install --upgrade numpy scipy

Test Failures
-------------

1. Check Python version (must be ≥ 3.8):

   .. code-block:: bash

      python --version

2. Clear pytest cache:

   .. code-block:: bash

      pytest --cache-clear tests/

3. Update all dependencies:

   .. code-block:: bash

      pip install --upgrade numpy scipy pytest

Conda Installation
==================

If using Conda (alternative):

.. code-block:: bash

   conda install numpy scipy matplotlib pytest
   git clone https://github.com/abder111/pymars.git
   cd pymars
   pip install -e .

Virtual Environment Setup
==========================

Recommended: Use virtual environment

.. code-block:: bash

   # Create virtual environment
   python -m venv pymars_env
   
   # Activate
   source pymars_env/bin/activate  # Linux/macOS
   # OR
   pymars_env\Scripts\activate  # Windows
   
   # Install
   pip install -e .

Uninstalling
============

Remove PyMARS:

.. code-block:: bash

   pip uninstall pymars

Development Setup
=================

For contributing to PyMARS:

.. code-block:: bash

   git clone https://github.com/abder111/pymars.git
   cd pymars
   
   # Install with all development tools
   pip install -e ".[dev,plot,docs]"
   
   # Create feature branch
   git checkout -b feature/your-feature
   
   # Make changes, run tests
   pytest tests/ -v
   
   # Format code
   black pymars/ tests/
   
   # Commit and push
   git add .
   git commit -m "Your message"
   git push origin feature/your-feature

Building Documentation Locally
===============================

Build HTML documentation:

.. code-block:: bash

   cd docs
   pip install sphinx sphinx-rtd-theme sphinxcontrib-bibtex
   make html

View in browser:

.. code-block:: bash

   open _build/html/index.html  # macOS
   xdg-open _build/html/index.html  # Linux
   start _build/html/index.html  # Windows
