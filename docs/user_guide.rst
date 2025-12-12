========================================
User Guide
========================================

.. contents::
   :local:
   :backlinks: none

Overview
========

PyMARS implements the complete MARS algorithm as described in Friedman (1991). This guide covers practical usage for fitting models, making predictions, interpreting results, and extending the functionality.

Core Workflow
=============

The typical MARS workflow consists of:

1. **Data Preparation** – Load and preprocess features and target
2. **Model Creation** – Instantiate MARS with desired parameters
3. **Model Fitting** – Execute forward and backward passes
4. **Evaluation** – Assess model fit with R², MSE, or custom metrics
5. **Interpretation** – Extract basis functions, feature importance, ANOVA decomposition
6. **Visualization** – Plot partial effects, basis functions, diagnostics

Basic Fitting
=============

Minimal Example
---------------

.. code-block:: python

   from pymars import MARS
   import numpy as np
   
   # 1. Generate synthetic data
   np.random.seed(42)
   X = np.random.uniform(-3, 3, (200, 2))
   y = np.sin(X[:, 0]) + 0.5 * X[:, 1]**2 + np.random.randn(200)*0.1
   
   # 2. Create and fit model
   model = MARS(max_terms=15, max_degree=1)
   model.fit(X, y)
   
   # 3. Predict
   y_pred = model.predict(X)
   
   # 4. Evaluate
   r2 = model.score(X, y)
   print(f"R² = {r2:.4f}")

Real Data Example
-----------------

With actual CSV data:

.. code-block:: python

   import pandas as pd
   from pymars import MARS
   
   # Load data
   data = pd.read_csv('data.csv')
   X = data.iloc[:, :-1].values
   y = data.iloc[:, -1].values
   
   # Fit model
   model = MARS(max_terms=30, max_degree=2)
   model.fit(X, y)
   
   # Predictions on new data
   X_new = pd.read_csv('new_data.csv').values
   y_new = model.predict(X_new)

Model Parameters
================

The MARS class accepts several key parameters:

.. list-table::
   :header-rows: 1
   :widths: 20 15 40 25

   * - Parameter
     - Type
     - Description
     - Default
   * - max_terms
     - int
     - Maximum basis functions (forward pass limit)
     - 30
   * - max_degree
     - int
     - Maximum interaction order (1=additive, 2=pairwise)
     - 1
   * - penalty
     - float
     - GCV penalty per basis function
     - 3.0
   * - minspan
     - int or 'auto'
     - Minimum observations between knots
     - 'auto'
   * - endspan
     - int or 'auto'
     - Minimum observations from endpoints
     - 'auto'
   * - alpha
     - float
     - Significance level for span calculations
     - 0.05
   * - standardize
     - bool
     - Standardize features (recommended)
     - True
   * - smooth
     - bool
     - Use cubic splines instead of linear hinges
     - False
   * - verbose
     - bool
     - Print fitting progress
     - True

Parameter Selection Guide
==========================

**max_terms**

Controls model complexity and training time.

- Small data (N < 100): max_terms = 10–20
- Medium data (100 < N < 500): max_terms = 20–40
- Large data (N > 500): max_terms = 40–100

.. code-block:: python

   # Additive model, many features
   model = MARS(max_terms=50, max_degree=1)
   
   # Interaction model, few features
   model = MARS(max_terms=20, max_degree=2)

**max_degree**

Controls interaction complexity:

- max_degree=1: Additive only (no interactions)
- max_degree=2: Pairwise interactions allowed
- max_degree=3: Up to 3-way interactions

.. code-block:: python

   # No interactions
   model = MARS(max_degree=1)
   
   # With interactions
   model = MARS(max_degree=2)

**penalty**

GCV penalty balances complexity vs. fit:

- Lower penalty → more complex models
- Higher penalty → simpler models
- Typical range: 2.0–4.0
- penalty=2 for additive models
- penalty=3 for general models

.. code-block:: python

   # More aggressive pruning
   model = MARS(penalty=2.0)
   
   # Less aggressive pruning
   model = MARS(penalty=4.0)

**minspan & endspan**

Control knot placement spacing:

- 'auto' (default): Uses Friedman formula from paper
- Integer value: Enforce specific spacing

.. code-block:: python

   # Friedman formula (automatic)
   model = MARS(minspan='auto', endspan='auto')
   
   # Manual spacing
   model = MARS(minspan=5, endspan=3)

**standardize**

Always recommended (improves numerical stability):

.. code-block:: python

   model = MARS(standardize=True)  # Default; recommended

**smooth**

Enable cubic splines (Section 3.7 of Friedman 1991):

.. code-block:: python

   # Linear hinge functions (default)
   model = MARS(smooth=False)
   
   # Cubic splines (smoother)
   model = MARS(smooth=True)

Making Predictions
==================

Once fitted, use predict():

.. code-block:: python

   # Single prediction
   y_hat = model.predict(X_test)
   
   # With shape handling
   X_single = np.array([[1.5, 2.3, -0.5]])
   y_single = model.predict(X_single)

Evaluating Performance
======================

R² Score
--------

.. code-block:: python

   r2 = model.score(X_test, y_test)
   print(f"R² = {r2:.4f}")  # Higher is better (max=1.0)

Mean Squared Error
------------------

.. code-block:: python

   from sklearn.metrics import mean_squared_error
   
   y_pred = model.predict(X_test)
   mse = mean_squared_error(y_test, y_pred)
   rmse = np.sqrt(mse)
   print(f"RMSE = {rmse:.4f}")

Custom Metrics
--------------

.. code-block:: python

   # Mean Absolute Error
   mae = np.mean(np.abs(y_test - y_pred))
   print(f"MAE = {mae:.4f}")
   
   # Mean Absolute Percentage Error
   mape = np.mean(np.abs((y_test - y_pred) / y_test))
   print(f"MAPE = {mape:.4f}")

Model Interpretation
====================

Summary Output
--------------

.. code-block:: python

   model.summary()

Produces:

.. code-block:: text

   ======================================================================
   MARS Model Summary
   ======================================================================
   Number of basis functions: 12
   Number of features: 5
   Maximum degree: 2
   GCV score: 0.02341
   Training MSE: 0.01523
   Training R²: 0.9521
   
   Feature Importances:
     x0: 0.4521
     x1: 0.3210
     x2: 0.2105
     x3: 0.0164
     x4: 0.0000
   
   Basis Functions:
     [0] coef=  1.2345  B_0 (constant)
     [1] coef=  2.3456  B_1 = h(x0, +, 0.523)
     [2] coef= -0.8765  B_2 = h(x1, -, 1.234)
     [3] coef=  1.1234  B_3 = B_1 * h(x2, +, -0.123)
     ...
   ======================================================================

Feature Importance
------------------

Get feature importance scores:

.. code-block:: python

   importances = model.feature_importances_
   
   # Plot importances
   import matplotlib.pyplot as plt
   
   plt.figure(figsize=(10, 4))
   plt.barh(range(len(importances)), importances)
   plt.xlabel('Importance')
   plt.ylabel('Feature')
   plt.title('Feature Importances')
   plt.show()

Basis Functions
---------------

Access fitted basis functions:

.. code-block:: python

   for i, bf in enumerate(model.basis_functions_):
       print(f"B_{i}: {bf}")

Coefficients
------------

.. code-block:: python

   coefs = model.coefficients_
   print(f"Number of coefficients: {len(coefs)}")
   print(f"Coefficients: {coefs}")

ANOVA Decomposition
-------------------

See :doc:`interactions` for full ANOVA details.

Quick access:

.. code-block:: python

   anova = model.get_anova_decomposition()
   
   # Main effects
   main_effects = anova.get(1, [])
   print(f"Main effects: {len(main_effects)}")
   
   # 2-way interactions
   interactions = anova.get(2, [])
   print(f"2-way interactions: {len(interactions)}")

GCV Score
---------

.. code-block:: python

   gcv = model.gcv_score_
   print(f"GCV score: {gcv:.6f}")

Visualization
=============

See :doc:`plots` for full details.

Quick examples:

.. code-block:: python

   from pymars.plots import plot_basis_functions, plot_predictions
   
   # Plot basis functions
   plot_basis_functions(model, X, y)
   
   # Plot fitted surface
   plot_predictions(model, X, y)

Advanced Topics
===============

Cubic Splines
-------------

For smoother models, use cubic extensions:

.. code-block:: python

   model = MARS(max_terms=20, smooth=True)
   model.fit(X, y)

See :doc:`cubic_extension` for details.

Interaction Analysis
--------------------

Extract and visualize interactions:

.. code-block:: python

   from pymars.interactions import InteractionAnalysis
   
   ia = InteractionAnalysis(model)
   ia.summary()

See :doc:`interactions` for full guide.

Model Selection
---------------

For hyperparameter tuning, see :doc:`model_selection`.

Common Use Cases
================

Case 1: Purely Additive Regression
-----------------------------------

When you expect no interactions:

.. code-block:: python

   model = MARS(
       max_terms=20,
       max_degree=1,      # No interactions
       penalty=2.0,       # More aggressive pruning
       standardize=True
   )
   model.fit(X, y)

Case 2: Interaction Model
---------------------------

When interactions are expected:

.. code-block:: python

   model = MARS(
       max_terms=40,
       max_degree=2,      # Allow 2-way interactions
       penalty=3.0,
       standardize=True,
       smooth=False       # Linear hinges
   )
   model.fit(X, y)

Case 3: Smooth Nonlinear Regression
-------------------------------------

For smooth, continuous predictions:

.. code-block:: python

   model = MARS(
       max_terms=25,
       max_degree=1,
       penalty=3.0,
       standardize=True,
       smooth=True        # Cubic splines
   )
   model.fit(X, y)

Case 4: High-Dimensional Selection
------------------------------------

When you have many features:

.. code-block:: python

   model = MARS(
       max_terms=50,      # Higher limit for feature selection
       max_degree=1,      # Focus on main effects
       penalty=2.5,       # Medium pruning
       standardize=True
   )
   model.fit(X, y)
   
   # See which features were selected
   important_features = np.where(model.feature_importances_ > 0)[0]
   print(f"Selected {len(important_features)} features")

Troubleshooting
===============

Model Not Improving
--------------------

If R² is low:

1. Increase max_terms
2. Increase max_degree
3. Decrease penalty
4. Check data quality and scaling

.. code-block:: python

   # More complex model
   model = MARS(max_terms=50, max_degree=2, penalty=2.5)

Overfitting
-----------

If training R² is high but test R² is low:

1. Increase penalty (more pruning)
2. Decrease max_terms
3. Decrease max_degree

.. code-block:: python

   # Simpler model
   model = MARS(max_terms=20, penalty=4.0)

Numerical Issues
----------------

If you get warnings about singular matrices:

1. Enable standardization (default=True)
2. Check for constant or near-constant features
3. Remove highly correlated features

.. code-block:: python

   model = MARS(standardize=True)

Slow Training
-------------

If fitting is very slow:

1. Reduce max_terms
2. Reduce max_degree
3. Reduce sample size
4. Use smaller minspan/endspan

.. code-block:: python

   model = MARS(max_terms=20, minspan=5, endspan=3)
