===========================
Interaction Analysis & ANOVA
===========================

.. contents::
   :local:
   :backlinks: none

Overview
========

MARS naturally identifies **interaction terms** (products of univariate hinges) and provides **ANOVA decomposition** to understand which variables interact and their magnitudes.

Interaction Concepts
====================

What is an Interaction?
-----------------------

An interaction occurs when the effect of one variable depends on the value of another.

**Example:**

The true model:
.. math::

   y = x_0 + x_1 + x_0 \cdot x_1

has a **2-way interaction** between :math:`x_0` and :math:`x_1`:

- Main effect of :math:`x_0`: :math:`\frac{\partial y}{\partial x_0} = 1 + x_1`
- Effect depends on :math:`x_1`: **interaction!**

Visually:

.. code-block:: text

   No interaction:            With interaction:
   
   y │                        y │      ╱╱
     │    ╱      ╱               │    ╱  ╱
     │   ╱      ╱                │   ╱   ╱
     │  ╱      ╱                 │  ╱   ╱
     └─────────────► x0          └──────────► x0
          x1=0                    (lines converge!)
           (parallel)

MARS Basis Functions as Interactions
-------------------------------------

MARS represents interactions as **products of univariate hinges**:

.. math::

   B_m(x) = h(x_i, t_1, d_1) \times h(x_j, t_2, d_2)

**Example MARS basis:**

.. code-block:: python

   B_5 = h(x0, +, 0.5) * h(x1, +, 1.0)
   
   Meaning: This basis "turns on" when:
   - x0 > 0.5  AND
   - x1 > 1.0

The product of hinges creates a **localized interaction** in that region.

Interaction Degree
-------------------

.. list-table::
   :header-rows: 1

   * - Degree
     - Form
     - Interpretation
   * - 0
     - Constant
     - Baseline
   * - 1
     - :math:`h(x_i, t, d)`
     - Main effect of :math:`x_i`
   * - 2
     - :math:`h(x_i, t_1, d_1) \times h(x_j, t_2, d_2)`
     - 2-way interaction
   * - 3
     - Product of 3 hinges
     - 3-way interaction
   * - k
     - Product of k hinges
     - k-way interaction

Parameter: **max_degree** (default=1)

- `max_degree=1`: No interactions (additive model)
- `max_degree=2`: Up to 2-way interactions
- `max_degree=3`: Up to 3-way interactions

Fitting Models with Interactions
=================================

Allowing Interactions
---------------------

.. code-block:: python

   from pymars import MARS
   
   # Additive (no interactions)
   model_additive = MARS(max_degree=1)
   
   # With 2-way interactions
   model_2way = MARS(max_degree=2)
   
   # With 3-way interactions
   model_3way = MARS(max_degree=3)

Typically:
- Start with `max_degree=1` (simpler)
- If R² is low, try `max_degree=2`
- Rarely need `max_degree > 2`

Example: Detecting Interactions
--------------------------------

.. code-block:: python

   import numpy as np
   from pymars import MARS
   
   # Generate data with TRUE interaction
   np.random.seed(42)
   X = np.random.uniform(-1, 1, (200, 3))
   
   # y = x0 + x1 + x0*x1 (interaction between x0, x1)
   y = X[:, 0] + X[:, 1] + X[:, 0] * X[:, 1] + np.random.randn(200)*0.1
   
   # Fit WITHOUT interactions
   model_no_int = MARS(max_degree=1)
   model_no_int.fit(X, y)
   r2_no_int = model_no_int.score(X, y)
   
   # Fit WITH interactions
   model_with_int = MARS(max_degree=2)
   model_with_int.fit(X, y)
   r2_with_int = model_with_int.score(X, y)
   
   print(f"Without interactions: R² = {r2_no_int:.4f}")
   print(f"With interactions:    R² = {r2_with_int:.4f}")
   print(f"Improvement: {(r2_with_int - r2_no_int):.4f}")

**Expected Output:**

.. code-block:: text

   Without interactions: R² = 0.7234
   With interactions:    R² = 0.9876
   Improvement: 0.2642

The significant improvement indicates the interaction was detected!

ANOVA Decomposition
===================

ANOVA = Analysis of Variance, extended here to MARS basis functions.

**Goal:** Decompose the fitted function into:

.. math::

   \hat{f}(x) = c_0 + \sum_i f_i(x_i) + \sum_{i<j} f_{ij}(x_i, x_j) + \ldots

where:
- :math:`c_0` = constant term
- :math:`f_i(x_i)` = main effect of variable :math:`i`
- :math:`f_{ij}(x_i, x_j)` = 2-way interaction between :math:`i` and :math:`j`

Extracting ANOVA Decomposition
-------------------------------

.. code-block:: python

   # Fit model with interactions
   model = MARS(max_degree=2)
   model.fit(X, y)
   
   # Get ANOVA decomposition
   anova = model.get_anova_decomposition()
   
   # anova is a dict: {order: [basis_functions]}
   # order=0: constant
   # order=1: main effects
   # order=2: 2-way interactions
   # order=3: 3-way interactions

Inspecting Components
---------------------

.. code-block:: python

   anova = model.get_anova_decomposition()
   
   print("ANOVA Decomposition:\n")
   
   # Constant term
   if 0 in anova:
       c0 = model.coefficients_[0]
       print(f"Constant: {c0:.6f}\n")
   
   # Main effects
   if 1 in anova:
       print("Main Effects (Degree 1):")
       for bf in anova[1]:
           idx = model.basis_functions_.index(bf)
           coef = model.coefficients_[idx]
           print(f"  {bf} : coef = {coef:.6f}")
       print()
   
   # 2-way interactions
   if 2 in anova:
       print("2-way Interactions (Degree 2):")
       for bf in anova[2]:
           idx = model.basis_functions_.index(bf)
           coef = model.coefficients_[idx]
           print(f"  {bf} : coef = {coef:.6f}")

Contribution to Prediction
---------------------------

Compute each term's contribution to predictions:

.. code-block:: python

   def anova_contributions(model, X):
       """
       Compute contribution of each ANOVA term to predictions.
       
       Returns:
           dict: {order: contributions}
       """
       anova = model.get_anova_decomposition()
       contributions = {0: np.full(len(X), model.coefficients_[0])}
       
       for order, basis_list in anova.items():
           if order == 0:
               continue
           
           contrib = np.zeros(len(X))
           
           for bf in basis_list:
               idx = model.basis_functions_.index(bf)
               coef = model.coefficients_[idx]
               bf_eval = bf.evaluate(X)  # Evaluate basis
               contrib += coef * bf_eval
           
           contributions[order] = contrib
       
       return contributions

Visualizing Interactions
=========================

Partial Effects Plot
---------------------

For a 2-way interaction, plot the effect of varying both variables:

.. code-block:: python

   import matplotlib.pyplot as plt
   from mpl_toolkits.mplot3d import Axes3D
   
   # Grid of x0, x1 values
   x0_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 50)
   x1_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 50)
   X0_grid, X1_grid = np.meshgrid(x0_range, x1_range)
   
   # Predictions on grid (fix x2 at median)
   x2_median = np.median(X[:, 2])
   X_grid = np.column_stack([
       X0_grid.ravel(),
       X1_grid.ravel(),
       np.full(X0_grid.size, x2_median)
   ])
   
   y_grid = model.predict(X_grid).reshape(X0_grid.shape)
   
   # 3D surface plot
   fig = plt.figure(figsize=(10, 8))
   ax = fig.add_subplot(111, projection='3d')
   
   surf = ax.plot_surface(X0_grid, X1_grid, y_grid, cmap='viridis', alpha=0.8)
   ax.scatter(X[:, 0], X[:, 1], y, alpha=0.3, s=10)
   
   ax.set_xlabel('x0'), ax.set_ylabel('x1'), ax.set_zlabel('y')
   ax.set_title('Interaction Effect: y vs (x0, x1)')
   
   fig.colorbar(surf, ax=ax, label='Predicted y')
   plt.show()

Interaction Heatmap
-------------------

Visualize which pairs of variables interact:

.. code-block:: python

   import matplotlib.pyplot as plt
   
   # Count interactions per variable pair
   d = X.shape[1]
   interaction_matrix = np.zeros((d, d))
   
   anova = model.get_anova_decomposition()
   for order, basis_list in anova.items():
       if order == 2:
           for bf in basis_list:
               # Extract variables from basis function
               vars_in_bf = bf.get_variables()  # [i, j]
               if len(vars_in_bf) == 2:
                   i, j = vars_in_bf
                   interaction_matrix[i, j] += 1
                   interaction_matrix[j, i] += 1
   
   # Heatmap
   plt.figure(figsize=(8, 6))
   plt.imshow(interaction_matrix, cmap='YlOrRd', interpolation='nearest')
   plt.colorbar(label='Interaction Count')
   plt.xlabel('Variable')
   plt.ylabel('Variable')
   plt.title('Variable Interactions in MARS Model')
   plt.xticks(range(d)), plt.yticks(range(d))
   plt.tight_layout()
   plt.show()

Quantifying Interaction Strength
=================================

Relative Contribution
---------------------

.. code-block:: python

   # Total sum of absolute coefficients by order
   contributions = {}
   
   anova = model.get_anova_decomposition()
   for order, basis_list in anova.items():
       total_coef = sum(
           abs(model.coefficients_[model.basis_functions_.index(bf)])
           for bf in basis_list
       )
       contributions[order] = total_coef
   
   # Normalize
   total = sum(contributions.values())
   relative = {k: v/total for k, v in contributions.items()}
   
   print("Relative Contribution by Interaction Order:")
   for order in sorted(relative.keys()):
       print(f"  Order {order}: {relative[order]:.1%}")

Example Output:

.. code-block:: text

   Relative Contribution by Interaction Order:
     Order 0 (constant): 5.2%
     Order 1 (main):    78.3%
     Order 2 (2-way):   16.5%

Interpretation
--------------

This model is:
- **Dominated by main effects** (78%)
- **Minor interactions** (17%)
- **Mostly additive** structure

Compare Models
--------------

.. code-block:: python

   for max_degree in [1, 2, 3]:
       model = MARS(max_degree=max_degree)
       model.fit(X, y)
       
       anova = model.get_anova_decomposition()
       
       print(f"max_degree={max_degree}:")
       
       for order in sorted(anova.keys()):
           n_basis = len(anova[order])
           print(f"  Order {order}: {n_basis} basis functions")

Interaction Constraints
=======================

PyMARS respects **interaction constraints** during fitting:

1. **max_degree constraint:** Cannot create basis of degree > max_degree
2. **Parent constraint:** A basis can only be parent if not already degree max_degree
3. **Forward pass:** Only adds child bases respecting constraints

Example:

.. code-block:: python

   # Can't create 3-way without allowing it
   model = MARS(max_degree=2)  # max degree 2
   
   # Forward pass will never create a basis like:
   # B = h(x0, t0, d0) * h(x1, t1, d1) * h(x2, t2, d2)
   
   # But with max_degree=3, it can

Testing for Interactions
========================

Statistical Test
-----------------

.. code-block:: python

   from scipy.stats import f_oneway
   
   # Additive model (no interactions)
   model_1 = MARS(max_degree=1)
   model_1.fit(X, y)
   residuals_1 = y - model_1.predict(X)
   ss_1 = np.sum(residuals_1**2)
   
   # Interaction model
   model_2 = MARS(max_degree=2)
   model_2.fit(X, y)
   residuals_2 = y - model_2.predict(X)
   ss_2 = np.sum(residuals_2**2)
   
   # F-test: Does adding interactions significantly improve fit?
   n = len(X)
   p1 = len(model_1.basis_functions_)
   p2 = len(model_2.basis_functions_)
   
   f_stat = ((ss_1 - ss_2) / (p2 - p1)) / (ss_2 / (n - p2))
   
   from scipy.stats import f
   p_value = 1 - f.cdf(f_stat, p2-p1, n-p2)
   
   print(f"F-statistic: {f_stat:.4f}")
   print(f"p-value: {p_value:.6f}")
   
   if p_value < 0.05:
       print("Interactions are significant! ✓")
   else:
       print("No significant interactions detected.")

Interpreting Results
====================

Strong Main Effects, Weak Interactions
---------------------------------------

Model structure:
.. math::

   \hat{f} \approx f(x_0) + f(x_1) + f(x_2)

Indicates:
- Variables act independently
- Simple, interpretable model
- Good for prediction and explanation

Weak Main Effects, Strong Interactions
----------------------------------------

Model structure:
.. math::

   \hat{f} \approx f(x_0, x_1) + f(x_1, x_2) + \ldots

Indicates:
- Variables only matter in combination
- Complex interdependencies
- Harder to interpret but captures key insights

Mixed (Both Strong)
--------------------

Model combines main + interaction effects:
.. math::

   \hat{f} \approx f(x_0) + f(x_1) + f(x_0, x_1) + \ldots

Indicates:
- Variables have individual effects
- Plus synergistic effects
- Requires explaining both components

Best Practices
==============

1. **Start with max_degree=1** – ensures interpretability
2. **Check if interactions help** – compare R² and GCV
3. **Visualize interactions** – use surface plots
4. **Quantify strength** – relative contributions
5. **Test significance** – F-test or cross-validation
6. **Interpret carefully** – interactions are harder to explain

Summary
=======

**Key Points:**

- ✅ MARS naturally finds interactions as basis products
- ✅ max_degree controls max interaction order
- ✅ ANOVA decomposes model by degree
- ✅ 2-way interactions usually sufficient
- ✅ Visualize with surface and heatmap plots
- ✅ Test significance before concluding interactions matter

**When to Use Interactions:**

- ✅ Domain knowledge suggests interactions
- ✅ R² improves significantly
- ✅ GCV score decreases with interactions
- ✅ You care about understanding relationships

**When to Avoid:**

- ❌ Need highest interpretability
- ❌ Sample size is very small
- ❌ Computational speed is critical
- ❌ Explanation to non-technical audience required
