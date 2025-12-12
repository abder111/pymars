==========================
Developer Guide
==========================

.. contents::
   :local:
   :backlinks: none

Architecture Overview
=====================

PyMARS is organized into focused modules, each with a clear responsibility:

.. code-block:: text

   pymars/
   ├── __init__.py           # Package exports
   ├── mars.py               # Main MARS class (fit/predict)
   ├── basis.py              # Basis function classes
   ├── model.py              # Forward/Backward algorithm implementations
   ├── gcv.py                # GCV calculation & selection
   ├── utils.py              # Utility functions (knots, solvers)
   ├── cubic.py              # Cubic spline extension
   ├── interactions.py       # ANOVA decomposition
   └── plots.py              # Visualization tools

Design Principles
=================

1. **Faithfulness to Friedman (1991)**
   - Every algorithm reproduced from paper
   - Mathematical formulas directly implemented
   - No algorithmic shortcuts

2. **Numerical Stability**
   - QR/Cholesky/pseudoinverse fallback chain
   - Standardization default-enabled
   - Condition number awareness

3. **scikit-learn Compatibility**
   - ``fit()`` / ``predict()`` / ``score()`` interface
   - Parameter validation
   - Transformer-compatible API

4. **Clarity Over Speed**
   - Readable code valued
   - Comments for non-obvious logic
   - Educational quality

5. **Modularity**
   - Each class has single responsibility
   - Composable components
   - Easy to extend

Core Classes
============

MARS Class
----------

**File:** ``pymars/mars.py``

**Responsibility:** Main user-facing class

.. code-block:: python

   class MARS:
       """Multivariate Adaptive Regression Splines."""
       
       def __init__(self, max_terms=30, max_degree=1, ...):
           """Initialize parameters."""
       
       def fit(self, X, y):
           """Forward + Backward pass."""
       
       def predict(self, X):
           """Evaluate basis functions."""
       
       def score(self, X, y):
           """R² score."""
       
       def summary(self):
           """Print detailed summary."""

**Key Methods:**
- ``fit()`` - Run forward & backward passes
- ``predict()`` - Evaluate model on new data
- ``score()`` - Compute R² score
- ``get_anova_decomposition()`` - ANOVA decomposition

**Attributes (after fitting):**
- ``basis_functions_`` - List of fitted bases
- ``coefficients_`` - Fitted coefficients
- ``gcv_score_`` - Final GCV score
- ``feature_importances_`` - Importance per feature

BasisFunction Class
--------------------

**File:** ``pymars/basis.py``

**Responsibility:** Represent a single basis function

.. code-block:: python

   class BasisFunction:
       """Product of hinge functions."""
       
       def __init__(self, hinges, basis_id):
           """Initialize with list of hinges."""
           self.hinges = hinges  # List of HingeFunction
           self.basis_id = basis_id  # Unique stable ID
       
       def evaluate(self, X):
           """Evaluate on data matrix X."""
           result = np.ones(len(X))
           for hinge in self.hinges:
               result *= hinge(X)
           return result
       
       def degree(self):
           """Number of hinges (interaction order)."""
           return len(self.hinges)

HingeFunction Class
-------------------

**File:** ``pymars/basis.py``

**Responsibility:** Single univariate hinge

.. code-block:: python

   class HingeFunction:
       """Univariate hinge: max(0, d*(x-t))"""
       
       def __init__(self, variable, knot, direction):
           self.variable = variable  # Which feature
           self.knot = knot          # Knot location
           self.direction = direction  # +1 or -1
       
       def __call__(self, X):
           """Evaluate on data."""
           x = X[:, self.variable]
           diff = self.direction * (x - self.knot)
           return np.maximum(diff, 0)

ForwardPass Class
-----------------

**File:** ``pymars/model.py``

**Responsibility:** Implement forward expansion

.. code-block:: python

   class ForwardPass:
       """Greedy basis expansion."""
       
       def run(self, X, y, max_terms, ...):
           """Execute forward pass."""
           
           # Initialize
           basis_functions = [self._constant_basis()]
           
           # Iterate
           for m in range(1, max_terms):
               # Find best split
               split = self._find_best_split(basis_functions, X, y)
               
               if split is None:
                   break
               
               # Add pair
               basis_functions.extend(split.pair)
           
           return basis_functions

BackwardPass Class
------------------

**File:** ``pymars/model.py``

**Responsibility:** Implement backward pruning

.. code-block:: python

   class BackwardPass:
       """GCV-based pruning."""
       
       def run(self, basis_functions, X, y, penalty):
           """Execute backward pass."""
           
           sequence = []  # Models at each step
           
           while len(basis_functions) > 1:
               # Try removing each basis
               best_removal = self._find_best_removal(
                   basis_functions, X, y
               )
               
               # Remove
               basis_functions.remove(best_removal)
               
               # Compute GCV
               gcv = self.gcv_calc.calculate(...)
               sequence.append((basis_functions, gcv))
           
           return sequence

GCVCalculator Class
-------------------

**File:** ``pymars/gcv.py``

**Responsibility:** GCV computation

.. code-block:: python

   class GCVCalculator:
       """Compute GCV scores."""
       
       def __init__(self, penalty=3.0):
           self.penalty = penalty
       
       def calculate(self, B, y):
           """GCV = RSS / [N(1 - df/N)²]"""
           
           c = solve_least_squares(B, y)
           rss = np.sum((y - B @ c)**2)
           df = self.complexity(B)
           
           N = len(y)
           return rss / (N * (1 - df/N)**2)
       
       def complexity(self, B):
           """df = trace(B(B^T B)^-1 B^T) + d*M"""
           
           BtB_inv = np.linalg.pinv(B.T @ B)
           trace_term = np.trace(B @ BtB_inv @ B.T)
           
           return trace_term + self.penalty * B.shape[1]

Development Workflow
====================

Setting Up Development Environment
-----------------------------------

.. code-block:: bash

   # Clone repository
   git clone https://github.com/abder111/pymars.git
   cd pymars
   
   # Create virtual environment
   python -m venv dev_env
   source dev_env/bin/activate  # Linux/macOS
   # OR
   dev_env\Scripts\activate  # Windows
   
   # Install in editable mode with dev deps
   pip install -e ".[dev,plot]"

Running Tests
-------------

.. code-block:: bash

   # Run all tests
   pytest tests/ -v
   
   # Run specific test
   pytest tests/test_mars.py::TestMARS::test_fit_predict -v
   
   # With coverage
   pytest tests/ --cov=pymars --cov-report=html
   
   # Open coverage report
   open htmlcov/index.html

Code Quality
------------

.. code-block:: bash

   # Format code
   black pymars/ tests/
   
   # Check style
   flake8 pymars/ tests/ --max-line-length=88
   
   # Type checking
   mypy pymars/

Building Documentation
-----------------------

.. code-block:: bash

   cd docs
   pip install sphinx sphinx-rtd-theme sphinxcontrib-bibtex
   make html
   
   # View locally
   open _build/html/index.html

Adding a New Feature
====================

Example: Adding L1 Regularization
----------------------------------

**Step 1:** Create new module

.. code-block:: bash

   touch pymars/regularization.py

**Step 2:** Implement class

.. code-block:: python

   # pymars/regularization.py
   
   class L1PenalizedMARSprinciples:
       """MARS with L1 penalty on coefficients."""
       
       def __init__(self, alpha=0.1):
           self.alpha = alpha
       
       def solve(self, B, y):
           """Solve with L1 + L2 (elastic net)."""
           from sklearn.linear_model import ElasticNet
           
           en = ElasticNet(alpha=self.alpha, l1_ratio=0.5)
           en.fit(B, y)
           return en.coef_

**Step 3:** Add tests

.. code-block:: python

   # tests/test_regularization.py
   
   import pytest
   from pymars.regularization import L1PenalizedMARS
   
   def test_l1_mars():
       """Test L1-regularized MARS."""
       
       X, y = generate_test_data()
       model = L1PenalizedMARS(alpha=0.1)
       # ... test implementation

**Step 4:** Update docs

.. code-block:: rst

   # docs/advanced_topics.rst
   
   L1 Regularization
   ------------------
   
   PyMARS supports optional L1 regularization...
   
   Example:
   
   .. code-block:: python
   
      model = L1PenalizedMARS(alpha=0.1)

**Step 5:** Update __init__.py

.. code-block:: python

   # pymars/__init__.py
   
   from .mars import MARS
   from .regularization import L1PenalizedMARS  # NEW
   
   __all__ = ['MARS', 'L1PenalizedMARS']

Testing Guidelines
==================

Test Structure
--------------

.. code-block:: python

   # tests/test_module.py
   
   import pytest
   import numpy as np
   from pymars import MARS
   
   @pytest.fixture
   def sample_data():
       """Provide test data."""
       X = np.random.randn(100, 5)
       y = X[:, 0] + np.sin(X[:, 1])
       return X, y
   
   class TestMARS:
       """Tests for MARS class."""
       
       def test_fit_predict(self, sample_data):
           """Test basic fit/predict."""
           X, y = sample_data
           
           model = MARS(max_terms=10)
           model.fit(X, y)
           y_pred = model.predict(X)
           
           assert y_pred.shape == y.shape
           assert not np.isnan(y_pred).any()

Test Categories
----------------

1. **Unit Tests** – Single function/method
2. **Integration Tests** – Multiple components together
3. **Regression Tests** – Known results (Friedman dataset)
4. **Edge Case Tests** – Boundary conditions
5. **Performance Tests** – Speed benchmarks (optional)

Writing Good Tests
-------------------

✅ **Good:**

.. code-block:: python

   def test_minspan_increases_with_sample_size():
       """Minspan should increase as N grows."""
       
       L_small = calculate_minspan(N=10, alpha=0.05)
       L_large = calculate_minspan(N=1000, alpha=0.05)
       
       assert L_large > L_small

❌ **Bad:**

.. code-block:: python

   def test_utils():
       """Test utils."""
       calculate_minspan(100, 0.05)  # No assertion!

Documentation Standards
=======================

Docstring Format
----------------

Use NumPy style docstrings:

.. code-block:: python

   def solve_least_squares(B, y):
       """Solve least squares system robustly.
       
       Uses QR decomposition with fallback to Cholesky
       and pseudoinverse for numerical stability.
       
       Parameters
       ----------
       B : ndarray
           Design matrix (n_samples, n_features).
       y : ndarray
           Target vector (n_samples,).
       
       Returns
       -------
       c : ndarray
           Coefficients (n_features,).
       
       Notes
       -----
       Uses three-tier fallback:
       1. QR (preferred, numerically stable)
       2. Cholesky (medium stability)
       3. Pseudoinverse (always works)
       
       Raises
       ------
       ValueError
           If dimensions don't match.
       
       Examples
       --------
       >>> B = np.array([[1, 0], [0, 1], [1, 1]])
       >>> y = np.array([1, 2, 3])
       >>> c = solve_least_squares(B, y)
       >>> np.allclose(B @ c, y, atol=1e-10)
       True
       """

Code Style
----------

* 4-space indentation
* Max 88 characters per line (Black standard)
* PEP 8 compliance (flake8)
* Type hints where helpful

.. code-block:: python

   def forward_pass(
       X: np.ndarray,
       y: np.ndarray,
       max_terms: int = 30
   ) -> list:
       """Run forward pass.
       
       Parameters
       ----------
       X : ndarray
           Features (n, d).
       y : ndarray
           Target (n,).
       max_terms : int
           Maximum iterations.
       
       Returns
       -------
       basis_functions : list
           Fitted basis functions.
       """

Contributing
============

How to Contribute
-----------------

1. **Fork** the repository
2. **Create** a feature branch: ``git checkout -b feature/my-feature``
3. **Make** changes and add tests
4. **Format** code: ``black pymars/ tests/``
5. **Test** thoroughly: ``pytest tests/ -v``
6. **Commit** with clear message: ``git commit -m "Add feature X"``
7. **Push** to branch: ``git push origin feature/my-feature``
8. **Submit** pull request

Commit Message Guidelines
--------------------------

.. code-block:: text

   Short description (max 50 chars)
   
   Longer explanation if needed. Describe:
   - What was changed
   - Why it was changed
   - How it works
   
   Fixes #123  (reference issues)

Performance Optimization
========================

Profiling
---------

.. code-block:: python

   import cProfile
   import pstats
   
   profiler = cProfile.Profile()
   profiler.enable()
   
   # ... code to profile ...
   
   profiler.disable()
   stats = pstats.Stats(profiler)
   stats.sort_stats('cumulative')
   stats.print_stats(10)

Common Bottlenecks
------------------

1. **Least squares solver** - O(N²M) per iteration
2. **Knot search** - Quadratic in candidate count
3. **Design matrix construction** - For large M

Optimization Ideas
-------------------

- Cache basis function evaluations
- Vectorize knot search (NumPy broadcasting)
- Use sparse matrices for large M
- Parallelize variable search

Release Checklist
=================

Before releasing version X.Y.Z:

- [ ] All tests pass locally
- [ ] Code review completed
- [ ] Documentation updated
- [ ] Changelog updated
- [ ] Version number bumped
- [ ] Commit tagged: ``git tag v0.X.Y``
- [ ] Built distributions: ``python -m build``
- [ ] Verified distributions: ``twine check dist/*``
- [ ] Pushed tag: ``git push origin v0.X.Y``
- [ ] Released on GitHub
- [ ] Uploaded to PyPI (when ready)

Troubleshooting Development
============================

ImportError: No module 'pymars'
-------------------------------

.. code-block:: bash

   pip install -e .

Test Failures
--------------

.. code-block:: bash

   # Clear cache
   pytest --cache-clear tests/
   
   # Run with verbose output
   pytest tests/ -vv

Documentation Build Fails
--------------------------

.. code-block:: bash

   cd docs
   rm -rf _build/
   pip install -r requirements.txt
   make html

Roadmap
=======

Planned Improvements
---------------------

* **v0.2:** Categorical support, missing values
* **v0.3:** Multi-output, classification, parallelization
* **v1.0:** Stable API, performance optimizations
* **1.x:** Extended features, domain extensions

Contributing to Roadmap
------------------------

See GitHub Issues and Discussions for community input on priorities.

Questions?
==========

* **Issues:** GitHub Issues
* **Discussions:** GitHub Discussions  
* **Email:** maintainers (in LICENSE)
* **Docs:** https://pymars.readthedocs.io

Thank you for contributing to PyMARS! 🎉
