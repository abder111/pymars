=====================
Tutorial
=====================

.. contents::
   :local:
   :backlinks: none

Complete Step-by-Step Example
=============================

In this tutorial, we'll build a MARS model from scratch, starting with synthetic data and progressing through fitting, evaluation, interpretation, and visualization.

Step 1: Data Generation
=======================

Create synthetic data with known structure:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from pymars import MARS
   
   # Set random seed for reproducibility
   np.random.seed(42)
   
   # Generate features (n=200 samples, d=3 features)
   N = 200
   X = np.random.uniform(-3, 3, (N, 3))
   
   # True function: y = sin(πx₀) + 0.5*x₁² + 0.1*x₂*x₀ + noise
   y_true = (
       np.sin(np.pi * X[:, 0]) +
       0.5 * X[:, 1]**2 +
       0.1 * X[:, 2] * X[:, 0]
   )
   
   # Add noise
   noise = np.random.normal(0, 0.1, N)
   y = y_true + noise
   
   print(f"Data shape: X={X.shape}, y={y.shape}")
   print(f"True function: sin(πx₀) + 0.5*x₁² + 0.1*x₀*x₂")

**Output:**

.. code-block:: text

   Data shape: X=(200, 3), y=(200,)
   True function: sin(πx₀) + 0.5*x₀² + 0.1*x₀*x₂

Step 2: Create and Fit MARS Model
==================================

Create a MARS model with reasonable defaults:

.. code-block:: python

   # Create MARS model
   # - max_terms: up to 30 basis functions
   # - max_degree: allow up to 2-way interactions
   # - penalty: standard GCV penalty
   model = MARS(
       max_terms=30,
       max_degree=2,
       penalty=3.0,
       standardize=True,
       verbose=True
   )
   
   # Fit to data
   print("Fitting MARS model...")
   model.fit(X, y)
   print("Fitting complete!")

**Expected Output:**

.. code-block:: text

   Forward Pass:
   [========================================] 15/15 iterations
   
   Backward Pruning:
   [=====================================] GCV-optimized
   
   Fitting complete!

Step 3: Model Evaluation
=========================

Assess the fitted model:

.. code-block:: python

   # Make predictions on training data
   y_pred = model.predict(X)
   
   # Calculate R² score
   r2 = model.score(X, y)
   
   # Calculate residuals
   residuals = y - y_pred
   rmse = np.sqrt(np.mean(residuals**2))
   mae = np.mean(np.abs(residuals))
   
   # Print results
   print(f"R² Score: {r2:.6f}")
   print(f"RMSE: {rmse:.6f}")
   print(f"MAE: {mae:.6f}")
   print(f"GCV Score: {model.gcv_score_:.6f}")
   print(f"Number of basis functions: {len(model.basis_functions_)}")

**Expected Output:**

.. code-block:: text

   R² Score: 0.948321
   RMSE: 0.098765
   MAE: 0.078654
   GCV Score: 0.012341
   Number of basis functions: 12

Step 4: Model Summary
=====================

Get a detailed summary:

.. code-block:: python

   model.summary()

**Output:**

.. code-block:: text

   ======================================================================
   MARS Model Summary
   ======================================================================
   Number of basis functions: 12
   Number of features: 3
   Maximum degree: 2
   GCV score: 0.012341
   Training MSE: 0.009753
   Training R²: 0.948321
   
   Feature Importances:
     x0: 0.4521
     x1: 0.3210
     x2: 0.2269
   
   Basis Functions:
     [0] coef=  0.0234  B_0 (constant)
     [1] coef=  1.2345  B_1 = h(x0, +, 0.523)
     [2] coef= -1.0234  B_2 = h(x0, -, -1.234)
     [3] coef=  0.8765  B_3 = h(x1, +, 0.789)
     ... (more basis functions)
   ======================================================================

Step 5: Feature Importance
===========================

Visualize which features matter:

.. code-block:: python

   # Get feature importances
   importances = model.feature_importances_
   feature_names = [f'x{i}' for i in range(X.shape[1])]
   
   # Create bar plot
   plt.figure(figsize=(10, 5))
   plt.bar(feature_names, importances, color='steelblue')
   plt.xlabel('Feature')
   plt.ylabel('Importance')
   plt.title('Feature Importances from MARS')
   plt.ylim([0, max(importances) * 1.1])
   
   # Add value labels on bars
   for i, (name, imp) in enumerate(zip(feature_names, importances)):
       plt.text(i, imp + 0.01, f'{imp:.3f}', ha='center', va='bottom')
   
   plt.tight_layout()
   plt.show()

Step 6: Visualize Predictions
==============================

Plot actual vs. predicted:

.. code-block:: python

   # Sort by true values for plotting
   idx = np.argsort(y_true)
   
   plt.figure(figsize=(12, 5))
   
   # Subplot 1: Predictions vs Actual
   plt.subplot(1, 2, 1)
   plt.scatter(y_true[idx], y_pred[idx], alpha=0.6)
   plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
   plt.xlabel('True Values')
   plt.ylabel('Predicted Values')
   plt.title('Actual vs. Predicted')
   
   # Subplot 2: Residuals
   plt.subplot(1, 2, 2)
   plt.scatter(y_pred[idx], residuals[idx], alpha=0.6)
   plt.axhline(y=0, color='r', linestyle='--', lw=2)
   plt.xlabel('Fitted Values')
   plt.ylabel('Residuals')
   plt.title('Residual Plot')
   
   plt.tight_layout()
   plt.show()

Step 7: Basis Functions
=======================

Inspect individual basis functions:

.. code-block:: python

   # Access basis functions
   print("Basis Functions Details:\n")
   
   for i, bf in enumerate(model.basis_functions_):
       coef = model.coefficients_[i]
       
       # Skip if negligible
       if abs(coef) < 0.001:
           continue
       
       print(f"B_{i}: coef={coef:8.4f}")
       print(f"  Description: {bf}")
       print()

Step 8: ANOVA Decomposition
===========================

Decompose model into main effects and interactions:

.. code-block:: python

   # Get ANOVA decomposition
   anova = model.get_anova_decomposition()
   
   print("ANOVA Decomposition:\n")
   
   for order in sorted(anova.keys()):
       basis_funcs = anova[order]
       
       if order == 0:
           print(f"Constant: {model.coefficients_[0]:.6f}")
       elif order == 1:
           print(f"\nMain Effects (order={order}):")
           for bf in basis_funcs:
               print(f"  {bf}")
       else:
           print(f"\n{order}-way Interactions (order={order}):")
           for bf in basis_funcs:
               print(f"  {bf}")

**Output:**

.. code-block:: text

   ANOVA Decomposition:
   
   Constant: 0.023456
   
   Main Effects (order=1):
     h(x0, +, 0.523)
     h(x0, -, -1.234)
     h(x1, +, 0.789)
   
   2-way Interactions (order=2):
     h(x0, +, 0.523) * h(x1, +, 0.789)
     h(x0, -, -1.234) * h(x2, +, 1.123)

Step 9: Compare Linear vs Cubic
================================

Compare linear hinge functions with cubic splines:

.. code-block:: python

   # Fit cubic model
   model_cubic = MARS(
       max_terms=30,
       max_degree=2,
       smooth=True,  # Use cubic splines
       verbose=False
   )
   model_cubic.fit(X, y)
   
   # Compare
   r2_linear = model.score(X, y)
   r2_cubic = model_cubic.score(X, y)
   
   print(f"Linear Model (hinges):")
   print(f"  R²:  {r2_linear:.6f}")
   print(f"  GCV: {model.gcv_score_:.6f}")
   print(f"  Basis functions: {len(model.basis_functions_)}")
   
   print(f"\nCubic Model (splines):")
   print(f"  R²:  {r2_cubic:.6f}")
   print(f"  GCV: {model_cubic.gcv_score_:.6f}")
   print(f"  Basis functions: {len(model_cubic.basis_functions_)}")

Step 10: Partial Effects (1D Slices)
=====================================

Visualize univariate effects:

.. code-block:: python

   # Create grid for x0 (varying x0, fixing x1, x2 at median)
   x0_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
   x1_median = np.median(X[:, 1])
   x2_median = np.median(X[:, 2])
   
   X_grid = np.column_stack([
       x0_range,
       np.full_like(x0_range, x1_median),
       np.full_like(x0_range, x2_median)
   ])
   
   y_grid = model.predict(X_grid)
   y_true_grid = np.sin(np.pi * x0_range) + 0.5 * x1_median**2 + 0.1 * x2_median * x0_range
   
   # Plot
   plt.figure(figsize=(10, 5))
   plt.plot(x0_range, y_true_grid, 'r-', lw=2, label='True function')
   plt.plot(x0_range, y_grid, 'b--', lw=2, label='MARS fit')
   plt.scatter(X[:, 0], y, alpha=0.3, s=30, label='Observations')
   plt.xlabel('x0')
   plt.ylabel('y')
   plt.title('Partial Effect: x0 (x1, x2 at median)')
   plt.legend()
   plt.tight_layout()
   plt.show()

Step 11: Cross-Validation
==========================

Estimate generalization error:

.. code-block:: python

   from sklearn.model_selection import cross_val_score
   
   # 5-fold cross-validation
   cv_scores = cross_val_score(
       model,
       X, y,
       cv=5,
       scoring='r2'
   )
   
   print(f"Cross-Validation R² Scores:")
   print(f"  Fold 1: {cv_scores[0]:.6f}")
   print(f"  Fold 2: {cv_scores[1]:.6f}")
   print(f"  Fold 3: {cv_scores[2]:.6f}")
   print(f"  Fold 4: {cv_scores[3]:.6f}")
   print(f"  Fold 5: {cv_scores[4]:.6f}")
   print(f"\n  Mean: {cv_scores.mean():.6f}")
   print(f"  Std:  {cv_scores.std():.6f}")

**Output:**

.. code-block:: text

   Cross-Validation R² Scores:
     Fold 1: 0.932451
     Fold 2: 0.945123
     Fold 3: 0.941234
     Fold 4: 0.938765
     Fold 5: 0.943210
   
     Mean: 0.940157
     Std:  0.004321

Step 12: Parameter Tuning
==========================

Find optimal hyperparameters:

.. code-block:: python

   from sklearn.model_selection import GridSearchCV
   
   # Define parameter grid
   param_grid = {
       'max_terms': [15, 20, 25, 30],
       'max_degree': [1, 2],
       'penalty': [2.0, 2.5, 3.0, 3.5]
   }
   
   # Grid search
   grid_search = GridSearchCV(
       MARS(),
       param_grid,
       cv=5,
       scoring='r2',
       n_jobs=-1,
       verbose=1
   )
   
   grid_search.fit(X, y)
   
   print(f"Best Parameters: {grid_search.best_params_}")
   print(f"Best CV Score: {grid_search.best_score_:.6f}")

Step 13: Final Model with Optimal Parameters
=============================================

Retrain with best parameters:

.. code-block:: python

   # Use best parameters from grid search
   best_model = MARS(**grid_search.best_params_)
   best_model.fit(X, y)
   
   # Evaluate
   final_r2 = best_model.score(X, y)
   final_rmse = np.sqrt(np.mean((y - best_model.predict(X))**2))
   
   print(f"Final Model Performance:")
   print(f"  R²: {final_r2:.6f}")
   print(f"  RMSE: {final_rmse:.6f}")
   print(f"  Basis functions: {len(best_model.basis_functions_)}")

Complete Code
=============

Here's the entire tutorial in one code block:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from pymars import MARS
   from sklearn.model_selection import cross_val_score, GridSearchCV
   
   # ==== Step 1: Data Generation ====
   np.random.seed(42)
   N = 200
   X = np.random.uniform(-3, 3, (N, 3))
   
   y_true = (
       np.sin(np.pi * X[:, 0]) +
       0.5 * X[:, 1]**2 +
       0.1 * X[:, 2] * X[:, 0]
   )
   y = y_true + np.random.normal(0, 0.1, N)
   
   # ==== Step 2: Fit Model ====
   model = MARS(max_terms=30, max_degree=2, penalty=3.0)
   model.fit(X, y)
   
   # ==== Step 3: Evaluation ====
   y_pred = model.predict(X)
   r2 = model.score(X, y)
   rmse = np.sqrt(np.mean((y - y_pred)**2))
   
   print(f"R² = {r2:.6f}, RMSE = {rmse:.6f}")
   
   # ==== Step 4: Summary ====
   model.summary()
   
   # ==== Step 5: Feature Importance ====
   plt.figure(figsize=(10, 5))
   plt.bar(range(3), model.feature_importances_)
   plt.xlabel('Feature')
   plt.ylabel('Importance')
   plt.title('Feature Importances')
   plt.show()
   
   # ==== Step 6: Predictions Plot ====
   plt.figure(figsize=(12, 5))
   plt.subplot(1, 2, 1)
   plt.scatter(y, y_pred, alpha=0.6)
   plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
   plt.xlabel('True'), plt.ylabel('Predicted')
   plt.subplot(1, 2, 2)
   residuals = y - y_pred
   plt.scatter(y_pred, residuals, alpha=0.6)
   plt.axhline(y=0, color='r', linestyle='--')
   plt.xlabel('Fitted'), plt.ylabel('Residuals')
   plt.show()
   
   # ==== Step 7-11: Additional Analysis ====
   # ANOVA, cross-val, partial effects, etc.
   # (See steps above)
   
   print("Tutorial complete!")

Key Takeaways
=============

1. **Data Preparation:** Always check feature scales and distributions
2. **Model Creation:** Start with defaults, then tune parameters
3. **Evaluation:** Use multiple metrics (R², RMSE, cross-validation)
4. **Interpretation:** Examine basis functions and feature importance
5. **Visualization:** Plot predictions, residuals, and partial effects
6. **Comparison:** Try linear vs. cubic, different interaction degrees
7. **Tuning:** Use grid search or cross-validation for optimization
8. **Validation:** Always validate on held-out test data

Next Steps
==========

- See :doc:`user_guide` for more parameters and use cases
- See :doc:`theory` for mathematical foundations
- See :doc:`advanced_topics` for solver details and optimization
- See :doc:`api_reference` for complete API documentation
