=============================
Model Selection & GCV
=============================

.. contents::
   :local:
   :backlinks: none

Overview
========

Model selection is a **critical component** of MARS. The algorithm automatically selects the "right-sized" model using **Generalized Cross-Validation (GCV)**, which balances fit quality against model complexity.

The GCV Problem
===============

**Central Question:** How many basis functions should we keep?

- **Too few:** Underfitting (high bias, poor R²)
- **Too many:** Overfitting (high variance, poor generalization)

**Solution:** Use a criterion that penalizes both:

1. **Training error (RSS)** – How well the model fits the data
2. **Model complexity (df)** – How many parameters we estimated

The GCV Formula
===============

**Generalized Cross-Validation** (Friedman 1991, Eq. 30):

.. math::

   \text{GCV}(M) = \frac{\text{RSS}(M)}{N \left(1 - \frac{C(M)}{N}\right)^2}

where:

**Residual Sum of Squares:**

.. math::

   \text{RSS}(M) = \sum_{i=1}^{N} (y_i - \hat{f}_M(x_i))^2

**Effective Degrees of Freedom:**

.. math::

   C(M) = \text{trace}(B(B^T B)^{-1}B^T) + d \cdot M

with:
- :math:`B` = :math:`N \times M` design matrix (M basis functions)
- :math:`d` = penalty per basis function (typically 2–3)
- :math:`M` = number of basis functions

Intuition
---------

.. math::

   \text{GCV} = \frac{\text{fit error}}{\text{complexity penalty}^2}

**As M increases:**
- Numerator (RSS) decreases → better fit
- Denominator penalty increases → penalizes complexity

**Optimal M:** Minimizes GCV score

**Example Curve:**

.. code-block:: text

   GCV Score
      ▲
      │     (underfitting)          (overfitting)
      │         ╱╲                 ╱
      │        ╱  ╲               ╱
      │       ╱    ╲             ╱
      │      ╱      ╲___________╱ ← minimum (optimal)
      │     ╱
      └─────────────────────────────► M (basis functions)
         0   5   10  15  20  25

The Forward Pass
================

Greedy basis expansion that usually overshoots the optimal model size.

**Process:**

1. Start with constant model (M=1)
2. Add basis function pairs sequentially
3. Continue until M = max_terms (overfit)
4. Result: Overly complex model with high variance

Example:

.. code-block:: text

   Iteration 1: M=1 (constant)     RSS = 10.5
   Iteration 2: M=3 (+ 2 hinges)   RSS = 5.2
   Iteration 3: M=5               RSS = 2.8
   Iteration 4: M=7               RSS = 1.5
   ...
   Iteration 15: M=29             RSS = 0.01 (overfitting!)

The Backward Pass
=================

**Goal:** Remove redundant basis functions using GCV.

**Process:**

1. Start with full model from forward pass (overfitted)
2. Try removing each basis function one at a time
3. Refit and compute GCV
4. Permanently remove function that minimizes GCV
5. Repeat until no improvement

**Example Sequence:**

.. code-block:: text

   Model 1: M=29  GCV=0.285
   Model 2: M=27  GCV=0.127  ← better! (removed 2 hinges)
   Model 3: M=25  GCV=0.089  ← better! (removed 2 more)
   Model 4: M=23  GCV=0.074  ← better!
   Model 5: M=21  GCV=0.062  ← better!
   Model 6: M=19  GCV=0.058  ← better!
   Model 7: M=17  GCV=0.063  ← worse! (stop pruning)
   
   ✓ Select Model 6 (M=19) with minimum GCV=0.058

Implementation
==============

GCVCalculator Class
-------------------

.. code-block:: python

   from pymars.gcv import GCVCalculator
   
   # Create calculator
   gcv_calc = GCVCalculator(penalty=3.0)
   
   # Compute GCV
   gcv_score = gcv_calc.calculate(B, y)
   
   where:
     B = Design matrix (N x M)
     y = Target vector (N,)
     penalty = d in formula above
     
     Returns: scalar GCV score

Computing GCV Step-by-Step
---------------------------

**Step 1: Fit model**

.. code-block:: python

   # Solve least squares: c = (B^T B)^-1 B^T y
   c = solve_least_squares(B, y)

**Step 2: Compute RSS**

.. code-block:: python

   residuals = y - B @ c
   RSS = np.sum(residuals**2)

**Step 3: Compute effective degrees of freedom**

.. code-block:: python

   # df = trace(B @ inv(B^T B) @ B^T) + d * M
   
   # Method 1 (explicit trace)
   BtB_inv = np.linalg.pinv(B.T @ B)
   trace_term = np.trace(B @ BtB_inv @ B.T)
   
   # Method 2 (via SVD, more stable)
   U, s, Vt = np.linalg.svd(B, full_matrices=False)
   trace_term = np.sum(s**2 / (s**2 + 1e-10))
   
   df = trace_term + penalty * B.shape[1]

**Step 4: Compute GCV**

.. code-block:: python

   N = B.shape[0]
   gcv = RSS / (N * (1 - df/N)**2)

Penalty Parameter
=================

The penalty :math:`d` in the GCV formula controls pruning severity.

**Common Values:**

.. list-table::
   :header-rows: 1

   * - Penalty
     - Model Type
     - Effect
   * - 2.0
     - Additive
     - Less aggressive pruning
   * - 2.5
     - Balanced
     - Medium pruning
   * - 3.0
     - General (default)
     - Standard pruning
   * - 4.0
     - Conservative
     - Very aggressive pruning

**How to Choose:**

.. code-block:: python

   # Additive model (no interactions)
   model = MARS(penalty=2.0)
   
   # General model with interactions
   model = MARS(penalty=3.0)  # default
   
   # Very parsimonious model
   model = MARS(penalty=4.0)

**Effect on Final Model Size:**

Higher penalty → fewer basis functions → simpler model

Example:

.. code-block:: python

   for penalty in [2.0, 3.0, 4.0]:
       model = MARS(penalty=penalty)
       model.fit(X, y)
       print(f"penalty={penalty}: {len(model.basis_functions_)} basis")
   
   # Output:
   # penalty=2.0: 18 basis
   # penalty=3.0: 12 basis
   # penalty=4.0: 8 basis

Alternative: BIC
================

Instead of GCV, you can use **Bayesian Information Criterion (BIC)**:

.. math::

   \text{BIC}(M) = N \log(\text{RSS}/N) + M \log(N)

**Comparison:**

.. list-table::
   :header-rows: 1

   * - Criterion
     - Formula
     - Penalty Type
     - Typical Result
   * - GCV
     - RSS / [N(1 - df/N)²]
     - Automatic (data-driven)
     - Intermediate complexity
   * - AIC
     - RSS + 2M
     - Linear (2 per term)
     - Slightly more complex
   * - BIC
     - RSS + M*log(N)
     - Increases with N
     - Less complex for large N
   * - Cp (Mallows)
     - RSS + 2σ²M
     - Requires σ² estimate
     - Similar to AIC

**When to use alternatives:**

- **BIC:** If you have large sample (N > 1000)
- **AIC:** If you prefer less aggressive pruning
- **Cp:** If you have reliable error variance estimate

Manual Model Selection
======================

You can also manually inspect the GCV sequence and choose:

.. code-block:: python

   from pymars.model import BackwardPass
   
   # Get full backward sequence
   bp = BackwardPass(basis_functions, coefficients, X, y, penalty=3.0)
   sequence = bp.run()
   
   # Print GCV sequence
   for i, (model_info, gcv_score) in enumerate(sequence):
       n_basis = len(model_info['basis_functions'])
       print(f"Model {i}: {n_basis} basis, GCV={gcv_score:.6f}")
   
   # Manually select (e.g., prefer sparsity)
   selected_idx = 5  # Pick model 5 instead of minimum GCV
   selected_model = sequence[selected_idx]

Cross-Validation vs GCV
=======================

**GCV Advantages:**

- ✅ Fast (no refitting)
- ✅ Deterministic (no randomness)
- ✅ Works well in practice
- ✅ Computationally efficient

**Cross-Validation Advantages:**

- ✅ Unbiased generalization estimate
- ✅ Works even if GCV assumption violated
- ✅ More flexible (any loss function)

**Comparison:**

.. code-block:: python

   from sklearn.model_selection import cross_val_score
   
   model = MARS(penalty=3.0)
   model.fit(X, y)
   
   # GCV score (automatic, from fitting)
   print(f"GCV: {model.gcv_score_:.6f}")
   
   # 5-fold cross-validation
   cv_scores = cross_val_score(model, X, y, cv=5)
   print(f"CV Mean: {cv_scores.mean():.6f}")
   print(f"CV Std: {cv_scores.std():.6f}")

**Rule of Thumb:** GCV and cross-validation usually agree (~within 10%).

Hyperparameter Tuning
=====================

Use GridSearchCV to find optimal penalty:

.. code-block:: python

   from sklearn.model_selection import GridSearchCV
   
   # Parameter grid
   params = {
       'penalty': [2.0, 2.5, 3.0, 3.5, 4.0],
       'max_terms': [20, 30, 40],
       'max_degree': [1, 2]
   }
   
   # Grid search with cross-validation
   grid = GridSearchCV(
       MARS(),
       params,
       cv=5,
       scoring='r2'
   )
   grid.fit(X, y)
   
   # Results
   print(f"Best params: {grid.best_params_}")
   print(f"Best CV score: {grid.best_score_:.6f}")
   
   # Use best model
   best_model = grid.best_estimator_
   best_model.fit(X, y)

Visualizing Model Selection
============================

Plot GCV vs model complexity:

.. code-block:: python

   import matplotlib.pyplot as plt
   from pymars.model import BackwardPass
   
   # Get backward sequence
   sequence = bp.run()
   
   n_basis_list = []
   gcv_scores = []
   
   for model_info, gcv in sequence:
       n_basis = len(model_info['basis_functions'])
       n_basis_list.append(n_basis)
       gcv_scores.append(gcv)
   
   # Plot
   plt.figure(figsize=(10, 6))
   plt.plot(n_basis_list, gcv_scores, 'o-', linewidth=2, markersize=8)
   
   # Mark minimum
   min_idx = np.argmin(gcv_scores)
   plt.scatter(n_basis_list[min_idx], gcv_scores[min_idx], 
               color='red', s=200, marker='*', zorder=5,
               label=f'Minimum at {n_basis_list[min_idx]} basis')
   
   plt.xlabel('Number of Basis Functions')
   plt.ylabel('GCV Score')
   plt.title('Model Selection: GCV vs Complexity')
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.tight_layout()
   plt.show()

Common Issues
=============

GCV Increases (No Improvement)
------------------------------

**Problem:** GCV goes up as you prune; backward pass doesn't remove anything.

**Causes:**
- Forward pass already found near-optimal model
- max_terms too low
- penalty too high

**Solution:**

.. code-block:: python

   # Reduce penalty (less aggressive pruning)
   model = MARS(penalty=2.5)

GCV Unstable
------------

**Problem:** GCV fluctuates wildly; hard to interpret.

**Causes:**
- Small sample size
- High condition number

**Solution:**

.. code-block:: python

   # Increase penalty (smooth out pruning)
   # Or increase max_terms to get better selection
   model = MARS(max_terms=50, penalty=3.5)

Overfitting Despite GCV
-----------------------

**Problem:** High training R² but low test R².

**Causes:**
- GCV assumption violated
- Noisy data with complicated true function

**Solution:**

.. code-block:: python

   # Use cross-validation penalty
   # Or increase penalty
   model = MARS(penalty=4.0)
   
   # Always validate on test set
   test_r2 = model.score(X_test, y_test)

Best Practices
==============

1. **Always standardize features** – improves numerical stability
2. **Use default penalty=3.0** – proven default for general models
3. **Validate on test set** – GCV is training estimate
4. **Try multiple penalties** – ensemble or sensitivity check
5. **Plot GCV sequence** – understand model selection process
6. **Check residuals** – ensure model assumptions met

Summary
=======

**Key Points:**

- ✅ GCV automatically balances fit vs. complexity
- ✅ Forward pass overshoots (overfits); backward prunes
- ✅ Penalty parameter controls pruning severity
- ✅ Default penalty=3.0 works well
- ✅ Always validate final model on test data
- ✅ Cross-validation provides independent estimate

**Formula to remember:**

.. math::

   \text{GCV} = \frac{\text{RSS}}{N(1 - C(M)/N)^2}

**Where:**

.. math::

   C(M) = \text{trace}(B(B^T B)^{-1}B^T) + dM
