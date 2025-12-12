===============================
API Reference
===============================

.. contents::
   :local:
   :backlinks: none

Core Module
===========

.. automodule:: pymars.mars
   :members:
   :undoc-members:
   :show-inheritance:

Basis Functions
===============

.. automodule:: pymars.basis
   :members:
   :undoc-members:
   :show-inheritance:

Model Selection (GCV)
=====================

.. automodule:: pymars.gcv
   :members:
   :undoc-members:
   :show-inheritance:

Utilities
=========

.. automodule:: pymars.utils
   :members:
   :undoc-members:
   :show-inheritance:

Model Algorithms
================

.. automodule:: pymars.model
   :members:
   :undoc-members:
   :show-inheritance:

Cubic Extension
===============

.. automodule:: pymars.cubic
   :members:
   :undoc-members:
   :show-inheritance:

Plotting
========

.. automodule:: pymars.plots
   :members:
   :undoc-members:
   :show-inheritance:

Interactions
============

.. automodule:: pymars.interactions
   :members:
   :undoc-members:
   :show-inheritance:

Quick API Summary
=================

MARS Class
----------

**Main entry point for all MARS operations.**

.. code-block:: python

   from pymars import MARS
   
   model = MARS(
       max_terms=30,
       max_degree=1,
       penalty=3.0,
       minspan='auto',
       endspan='auto',
       alpha=0.05,
       standardize=True,
       smooth=False,
       verbose=True
   )
   model.fit(X, y)
   y_pred = model.predict(X)
   score = model.score(X_test, y_test)
   model.summary()

Key Methods
-----------

**fit(X, y)**
   Fit the MARS model to data.

   :param X: Feature matrix (n_samples, n_features)
   :param y: Target vector (n_samples,)

**predict(X)**
   Make predictions on new data.

   :param X: Feature matrix
   :return: Predicted values

**score(X, y)**
   Compute R² score.

   :param X: Feature matrix
   :param y: Target values
   :return: R² score (0 to 1, higher is better)

**summary()**
   Print detailed model summary.

**get_anova_decomposition()**
   Get ANOVA decomposition by interaction order.

   :return: Dictionary {order: [basis_functions]}

Key Attributes
---------------

**basis_functions_**
   List of fitted basis functions.

**coefficients_**
   Fitted model coefficients (length = len(basis_functions_)).

**feature_importances_**
   Importance score for each feature (0 to 1).

**gcv_score_**
   Final GCV score of selected model.

**n_features_in_**
   Number of input features.

**n_basis_functions_**
   Number of basis functions in final model.

Basis Functions
---------------

.. code-block:: python

   from pymars.basis import HingeFunction, BasisFunction
   
   # Hinge: h(x, t, d) = max(0, d*(x - t))
   hinge = HingeFunction(variable=0, knot=1.5, direction=1)
   
   # Basis: product of hinges
   bf = BasisFunction(hinges=[hinge], basis_id=1)

GCV Calculator
--------------

.. code-block:: python

   from pymars.gcv import GCVCalculator
   
   gcv_calc = GCVCalculator(penalty=3.0)
   gcv_score = gcv_calc.calculate(B, y)  # Design matrix B, target y

Utilities
---------

.. code-block:: python

   from pymars.utils import (
       calculate_minspan,
       calculate_endspan,
       get_candidate_knots,
       solve_least_squares
   )
   
   L = calculate_minspan(N=200, alpha=0.05)
   Le = calculate_endspan(N=200, alpha=0.05)
   knots = get_candidate_knots(X_j, minspan=L, endspan=Le)
   c = solve_least_squares(B, y)

Cubic Splines
-------------

.. code-block:: python

   from pymars.cubic import convert_to_cubic
   
   # Convert fitted linear model to cubic
   model_cubic = convert_to_cubic(model)

Interactions
------------

.. code-block:: python

   from pymars.interactions import InteractionAnalysis
   
   ia = InteractionAnalysis(model)
   ia.summary()
   effects = ia.get_interaction_effects()

Plotting
--------

.. code-block:: python

   from pymars.plots import (
       plot_basis_functions,
       plot_predictions,
       plot_residuals,
       plot_feature_importance
   )
   
   plot_basis_functions(model, X, y)
   plot_feature_importance(model)

Parameter Details
=================

max_terms
---------

**Type:** int  
**Default:** 30  
**Range:** 5–500  

Maximum number of basis functions in forward pass. Higher values allow more complex models but increase fitting time.

- Small data: 10–20
- Medium data: 20–50
- Large data: 50–100

max_degree
----------

**Type:** int  
**Default:** 1  
**Range:** 1–5  

Maximum interaction order.

- 1: Main effects only
- 2: Up to 2-way interactions
- 3: Up to 3-way interactions

penalty
-------

**Type:** float  
**Default:** 3.0  
**Range:** 2.0–5.0  

GCV penalty per basis function. Controls pruning aggressiveness.

- 2.0: Aggressive (fewer basis functions)
- 3.0: Standard (balanced)
- 4.0: Conservative (more basis functions)

minspan & endspan
------------------

**Type:** int or 'auto'  
**Default:** 'auto'  

Knot spacing constraints. 'auto' uses Friedman formulas:

.. math::

   L = \lfloor -\log_2(\alpha/N) / 2.5 \rfloor

.. math::

   L_e = \lceil 3 - \log_2(\alpha/N) \rceil

alpha
-----

**Type:** float  
**Default:** 0.05  
**Range:** 0.01–0.10  

Significance level for span calculations.

standardize
-----------

**Type:** bool  
**Default:** True  

Standardize features to mean 0, std 1. Highly recommended for numerical stability.

smooth
------

**Type:** bool  
**Default:** False  

Use cubic splines instead of linear hinges. Produces smoother models with continuous 2nd derivatives.

verbose
-------

**Type:** bool  
**Default:** True  

Print iteration progress during fitting.

Common Patterns
===============

Pattern 1: Additive Model
--------------------------

No interactions, simple fit:

.. code-block:: python

   model = MARS(
       max_terms=20,
       max_degree=1,  # No interactions
       penalty=2.0    # More pruning
   )
   model.fit(X, y)

Pattern 2: With Interactions
-----------------------------

Allow 2-way interactions:

.. code-block:: python

   model = MARS(
       max_terms=40,
       max_degree=2,  # 2-way interactions
       penalty=3.0
   )
   model.fit(X, y)

Pattern 3: Smooth Model
-----------------------

Cubic splines:

.. code-block:: python

   model = MARS(
       max_terms=25,
       max_degree=1,
       smooth=True  # Cubic instead of linear
   )
   model.fit(X, y)

Pattern 4: High-Dimensional
----------------------------

Feature selection with many variables:

.. code-block:: python

   model = MARS(
       max_terms=50,   # Higher limit
       max_degree=1,   # Focus on main effects
       penalty=2.5     # Medium pruning
   )
   model.fit(X, y)
   
   # See selected features
   important = np.where(model.feature_importances_ > 0)[0]

Exceptions & Errors
===================

ValueError: max_terms must be >= 2
--------------------------------------

The model needs at least 2 basis functions (constant + 1 hinge).

**Solution:** Increase max_terms to ≥ 2.

ValueError: max_degree must be >= 1
--------------------------------------

Interactions require at least degree 1.

**Solution:** Set max_degree=1.

LinAlgError: Singular matrix
-----------------------------

Least-squares matrix is rank-deficient.

**Causes:**
   - Constant feature (all same value)
   - Collinear features
   - Too many basis functions

**Solutions:**
   - Remove constant/near-constant features
   - Standardize (default)
   - Reduce max_terms
   - Increase penalty

ConvergenceWarning
-------------------

Forward pass did not reach max_terms.

**Meaning:** RSS improvement plateaued before max_terms.

**This is normal behavior** – indicates model has reached optimal complexity.

Logging & Debugging
===================

Enable verbose output:

.. code-block:: python

   model = MARS(verbose=True)
   model.fit(X, y)

Check GCV progression:

.. code-block:: python

   print(f"GCV Score: {model.gcv_score_:.6f}")
   print(f"Training R²: {model.score(X, y):.6f}")

Inspect basis functions:

.. code-block:: python

   for i, bf in enumerate(model.basis_functions_):
       print(f"B_{i}: {bf}")
       print(f"  Coefficient: {model.coefficients_[i]:.6f}")

Performance Considerations
===========================

Fitting Time
------------

Typical execution times on standard hardware:

.. list-table::
   :header-rows: 1

   * - Samples
     - Features
     - Max Terms
     - Time
   * - 100
     - 5
     - 20
     - ~0.5 s
   * - 200
     - 10
     - 30
     - ~2 s
   * - 500
     - 15
     - 40
     - ~15 s
   * - 1000
     - 20
     - 50
     - ~60 s

Memory Usage
------------

Approximate peak memory (MB):

.. math::

   \text{Memory} \approx 8 \times N \times (d + M)

where N = samples, d = features, M = basis functions.

Prediction Speed
----------------

Once fitted, prediction is very fast:

.. code-block:: python

   import time
   
   start = time.time()
   y_pred = model.predict(X_test)  # 10,000 samples
   elapsed = time.time() - start
   print(f"Time: {elapsed:.3f}s, Rate: {len(X_test)/elapsed:.0f} samples/sec")

Typical: 50,000–100,000 predictions/second.
