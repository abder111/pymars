================
Advanced Topics
================

.. contents::
   :local:
   :backlinks: none

Numerical Solver Details
========================

Least Squares Stability Chain
-----------------------------

PyMARS uses a robust three-tier fallback strategy for solving least squares:

.. code-block:: python

   def solve_least_squares(B, y):
       """
       Solve: min_c ||y - B @ c||_2^2
       
       Uses fallback chain:
       1. QR decomposition (preferred, numerically stable)
       2. Cholesky on B^T B
       3. Pseudoinverse (most robust)
       """
       try:
           # Tier 1: QR decomposition
           Q, R = np.linalg.qr(B, mode='reduced')
           c = np.linalg.solve(R, Q.T @ y)
           
       except np.linalg.LinAlgError:
           try:
               # Tier 2: Cholesky
               L = np.linalg.cholesky(B.T @ B)
               c = np.linalg.solve(
                   L.T,
                   np.linalg.solve(L, B.T @ y)
               )
           except np.linalg.LinAlgError:
               # Tier 3: Pseudoinverse (always works)
               c = np.linalg.pinv(B) @ y
       
       return c

**Why This Matters:**

- QR: Fast, backward stable, preferred
- Cholesky: Medium speed, less stable
- Pseudoinverse: Slowest but always works

Condition Number Management
----------------------------

For ill-conditioned problems:

.. code-block:: python

   import numpy as np
   
   # Check condition number
   BtB = B.T @ B
   cond = np.linalg.cond(BtB)
   
   if cond > 1e10:
       print("WARNING: Ill-conditioned problem")
       print(f"Condition number: {cond:.2e}")
       
       # Solutions:
       # 1. Use pseudoinverse
       c = np.linalg.pinv(B) @ y
       
       # 2. Standardize features (default in MARS)
       # 3. Remove collinear features
       # 4. Increase regularization (not in MARS)

Computational Complexity
=========================

Algorithm Complexity Analysis
-----------------------------

**Forward Pass:**

.. math::

   O(M_{\max} \cdot d \cdot N \cdot |\text{knots}| \cdot N^2)

Breakdown:
- :math:`M_{\max}` iterations
- :math:`d` variables per iteration
- :math:`N` knot candidates per variable
- :math:`N^2` for least squares (per candidate)

**For typical MARS:**

- Forward: :math:`O(30 \times 10 \times 200 \times 30 \times 40000) \approx 7.2 \times 10^{11}` ops
- But: Knot search is pruned, least squares is fast with QR

**Backward Pass:**

.. math::

   O(M \cdot M \cdot N^2)

Faster than forward, dominated by least squares refitting.

**Total:** O(1-2 minutes) for typical dataset (N=1000, d=20)

Memory Usage
------------

Peak memory scales with:

.. math::

   \text{Memory} \approx 8 \times (N \times M + d \times M + \text{workspace})

For N=1000, M=50, d=20:
- Design matrix B: 1000 × 50 = 50k elements = 400 KB
- Coefficients: 50 elements = 400 B
- QR workspace: ~20 MB
- **Total: ~20-30 MB**

Optimization Strategies
=======================

For Large Datasets
-------------------

If N > 10,000:

.. code-block:: python

   model = MARS(
       max_terms=20,          # Reduce complexity
       minspan=50,            # Fewer knots to test
       endspan=10,            # Fewer endpoint candidates
       verbose=True           # Monitor progress
   )

For High-Dimensional Data
---------------------------

If d > 50:

.. code-block:: python

   # 1. Dimension reduction first
   from sklearn.decomposition import PCA
   pca = PCA(n_components=20)
   X_reduced = pca.fit_transform(X)
   
   # 2. Or use sparse MARS
   model = MARS(
       max_terms=30,
       max_degree=1,          # No interactions for speed
       penalty=2.0            # More pruning
   )

For Real-Time Prediction
------------------------

.. code-block:: python

   # Prediction is fast: O(M) per sample
   # M = basis functions (typically < 50)
   
   # Example: 100,000 predictions in < 1 second
   import time
   
   start = time.time()
   y_pred = model.predict(X_test)  # X_test.shape = (100000, 20)
   elapsed = time.time() - start
   
   rate = len(X_test) / elapsed
   print(f"Prediction rate: {rate:.0f} samples/sec")
   # Output: ~200,000-500,000 samples/sec

Caching Strategies
==================

If Refitting Models Repeatedly
-------------------------------

Cache basis function evaluations:

.. code-block:: python

   # Evaluate all bases once
   B_cache = {}
   for bf in model.basis_functions_:
       B_cache[bf.id] = bf.evaluate(X)
   
   # Now comparisons are O(M) instead of O(M*N)
   for penalty in [2.0, 2.5, 3.0]:
       # Use cached B instead of recomputing
       gcv = compute_gcv(B_cache, y, penalty)

Parallel Processing
===================

Knot Search Parallelization
----------------------------

Forward pass can be parallelized:

.. code-block:: python

   from joblib import Parallel, delayed
   
   def search_variable(X_j, y, current_basis):
       """Search all knots for one variable."""
       knots = get_candidate_knots(X_j, minspan, endspan)
       
       best_rss = float('inf')
       best_pair = None
       
       for t in knots:
           # Fit and evaluate pair
           # ... (as in forward pass)
       
       return best_pair, best_rss
   
   # Parallel loop over variables
   results = Parallel(n_jobs=-1)(
       delayed(search_variable)(X[:, j], y, current_basis)
       for j in range(X.shape[1])
   )

**Note:** PyMARS doesn't use built-in parallelization by default to maintain simplicity. Users can add it as shown above.

Debugging & Profiling
======================

Verbose Output
--------------

.. code-block:: python

   model = MARS(verbose=True)
   model.fit(X, y)
   
   # Shows iteration progress

Profiling Fit Time
------------------

.. code-block:: python

   import cProfile
   import pstats
   
   profiler = cProfile.Profile()
   profiler.enable()
   
   model = MARS()
   model.fit(X, y)
   
   profiler.disable()
   stats = pstats.Stats(profiler)
   stats.sort_stats('cumulative')
   stats.print_stats(10)  # Top 10 functions by time

Memory Profiling
----------------

.. code-block:: python

   from memory_profiler import profile
   
   @profile
   def fit_mars():
       model = MARS()
       model.fit(X, y)
       return model
   
   fit_mars()
   
   # Output shows memory usage per line

Edge Cases & Robustness
=======================

Constant Features
-----------------

If a feature has no variance:

.. code-block:: python

   # Detection
   if np.std(X[:, j]) < 1e-10:
       print(f"Feature {j} is constant, removing...")
       X = np.delete(X, j, axis=1)

Single Knot Problem
-------------------

If all data values are identical:

.. code-block:: python

   # Add small noise or handle specially
   if len(np.unique(X[:, j])) == 1:
       # Can't place knots; skip this variable
       pass

Singular Design Matrix
----------------------

If B^T B is singular:

.. code-block:: python

   # Automatic fallback to pseudoinverse
   # (handled in solve_least_squares)
   
   # Or manually use pseudoinverse
   c = np.linalg.pinv(B) @ y

Perfect Fit (RSS = 0)
---------------------

If model fits perfectly (rare):

.. code-block:: python

   # GCV formula becomes 0 / 0
   # Handled gracefully:
   
   if RSS < 1e-10:
       gcv = 0.0  # Perfect fit
   else:
       gcv = RSS / (N * (1 - df/N)**2)

Extension Points
================

Custom Basis Functions
----------------------

Implement your own by subclassing:

.. code-block:: python

   from pymars.basis import HingeFunction
   
   class SineHinge(HingeFunction):
       """Custom: sin(x) instead of hinge"""
       
       def __call__(self, X):
           x = X[:, self.variable]
           return np.sin(x - self.knot)

Custom Loss Functions
---------------------

MARS uses squared loss by default. For custom losses:

.. code-block:: python

   def fit_with_custom_loss(X, y, loss_fn, loss_grad):
       """
       Gradient descent instead of least squares.
       (Not in standard MARS)
       """
       # Iterative refinement using loss_grad
       pass

Custom Penalty Functions
------------------------

Instead of constant penalty :math:`d`:

.. code-block:: python

   def adaptive_penalty(M, n_interactions):
       """Penalty that increases with interactions."""
       base_penalty = 3.0
       interaction_penalty = 0.5 * n_interactions
       return base_penalty + interaction_penalty

Advanced: Fine-Tuning the Algorithm
====================================

Modifying Knot Density
-----------------------

Change how many knots are tested:

.. code-block:: python

   # Default: test all unique sorted values
   # Custom: test every k-th value
   
   def sparse_knot_candidates(X_j, k=5):
       """Test every k-th unique value."""
       unique_vals = np.unique(X_j)
       return unique_vals[::k]

Interaction Constraints
-----------------------

Add domain-specific constraints:

.. code-block:: python

   class ConstrainedMARS(MARS):
       """MARS with interaction constraints."""
       
       def _find_best_split(self, basis, y):
           """Override to add constraints."""
           
           # ... standard search ...
           
           # Check constraint before adding
           if self._is_valid_interaction(basis, new_hinge):
               # Keep candidate
               pass

Algorithm Variants
===================

Boosting MARS
-------------

Stack multiple MARS models on residuals:

.. code-block:: python

   def boost_mars(X, y, n_boosters=5, learning_rate=0.1):
       """Boosted MARS ensemble."""
       
       residuals = y.copy()
       models = []
       
       for i in range(n_boosters):
           model = MARS(max_terms=10)
           model.fit(X, residuals)
           
           pred = model.predict(X)
           residuals = residuals - learning_rate * pred
           
           models.append(model)
       
       return models
   
   def predict_boosted(models, X, learning_rate=0.1):
       """Predict with boosted ensemble."""
       
       y_pred = np.zeros(len(X))
       for model in models:
           y_pred += learning_rate * model.predict(X)
       
       return y_pred

Sparse MARS
-----------

Force sparsity via L1 penalty:

.. code-block:: python

   # Not built-in; requires modifying least squares
   # Use Elastic Net instead of OLS:
   
   from sklearn.linear_model import ElasticNet
   
   def solve_sparse(B, y, alpha=0.1, l1_ratio=0.5):
       """Solve with L1 + L2 penalty."""
       
       en = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
       en.fit(B, y)
       return en.coef_

Summary
=======

**Key Advanced Topics:**

- ✅ Numerical stability via fallback chain
- ✅ Complexity analysis and optimization
- ✅ Memory-efficient computation
- ✅ Caching and parallelization strategies
- ✅ Debugging and profiling tools
- ✅ Extension points for customization
- ✅ Algorithm variants (boosting, sparse)

**When to Use Advanced Features:**

- Large-scale datasets: Optimize memory and time
- Custom applications: Extend with domain knowledge
- Research: Implement algorithm variants
- Production: Add robustness and monitoring
