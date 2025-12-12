===============================
MARS Algorithms
===============================

.. contents::
   :local:
   :backlinks: none

Overview
========

This section details the three core algorithms in MARS (Friedman 1991):

1. **Algorithm 1** – Recursive partitioning basis generation
2. **Algorithm 2** – Forward pass (greedy basis expansion)
3. **Algorithm 3** – Backward pass (GCV-based pruning)

These are reproduced directly from the paper with implementation notes.

Algorithm 1: Recursive Partitioning
===================================

**Purpose:** Generate candidate basis functions through recursive partitioning.

**Conceptual Algorithm:**

.. code-block:: text

   RECURSIVEPARTITION(data, depth=0):
       if (depth >= max_depth) or (n_samples < min_samples):
           return {constant_function}
       
       best_split = None
       best_gain = 0
       
       for each variable j:
           for each candidate knot t in valid_knots(x_j):
               split_left = data where x_j <= t
               split_right = data where x_j > t
               gain = reduction_in_error(left, right)
               
               if gain > best_gain:
                   best_gain = gain
                   best_split = (j, t)
       
       if best_split is None:
           return {constant_function}
       
       left_basis = RECURSIVEPARTITION(split_left, depth+1)
       right_basis = RECURSIVEPARTITION(split_right, depth+1)
       
       return left_basis ∪ right_basis

**Notes:**

- Forms a binary tree of partitions
- Each partition becomes a basis function candidate
- MARS uses this conceptually but computes it more efficiently

**Connection to Implementation:**

In `pymars/basis.py`, the `HingeFunction` class represents these partitions as products of univariate hinges, avoiding explicit tree structures.

Algorithm 2: Forward Pass
==========================

**Purpose:** Greedily expand basis functions to fit the residuals.

**Formal Statement** (Friedman 1991, Algorithm 2):

.. code-block:: text

   FORWARD(X, y, M_max, d_max, minspan, endspan, α):
       /* Input: N×d data matrix X, response vector y
                 M_max = max basis functions
                 d_max = max interaction degree
                 minspan, endspan = spacing constraints
                 α = significance level
       */
       
       /* Initialize */
       B = {B_1 = 1}                    /* Constant basis */
       coef = solve_least_squares(B, y) /* Fit constant */
       RSS = sum((y - coef)^2)
       RSS_previous = ∞
       
       /* Main loop: add basis function pairs */
       m = 1
       while m < M_max:
           RSS_best = ∞
           pair_best = None
           
           /* For each existing basis function */
           for m' in 1...|B|:
               B_m' = B_{m'}
               
               /* For each variable */
               for j in 1..d:
                   /* Find valid knot locations */
                   knots = valid_knots(X[:, j], minspan, endspan)
                   
                   /* For each knot */
                   for t in knots:
                       /* Check degree constraint */
                       if degree(B_m') < d_max:
                           /* Try both directions */
                           for dir in {+, -}:
                               /* Create hinge pair */
                               h_plus = B_m' * h(X[:, j], t, +)
                               h_minus = B_m' * h(X[:, j], t, -)
                               
                               /* Temporary basis with new pair */
                               B_temp = B ∪ {h_plus, h_minus}
                               
                               /* Fit least squares */
                               coef_temp = solve_least_squares(B_temp, y)
                               RSS_temp = sum((y - coef_temp)^2)
                               
                               /* Track best */
                               if RSS_temp < RSS_best:
                                   RSS_best = RSS_temp
                                   pair_best = (h_plus, h_minus, j, t, dir)
           
           /* Add best pair */
           if pair_best is not None:
               B = B ∪ {pair_best.h_plus, pair_best.h_minus}
               coef = solve_least_squares(B, y)
               RSS = RSS_best
               m = m + 1
           
           /* Check improvement */
           if RSS >= RSS_previous:
               break                    /* No improvement */
           RSS_previous = RSS
       
       return B, coef

**Key Implementation Details:**

.. list-table::
   :header-rows: 1

   * - Component
     - Description
     - Implementation Location
   * - Valid knots
     - Respect minspan/endspan
     - `pymars/utils.py`
     - `get_candidate_knots()`
   * - Hinge function
     - :math:`h(x_j, t, d) = \max(0, d(x_j - t))`
     - `pymars/basis.py`
     - `HingeFunction.__call__()`
   * - Least squares
     - :math:`\hat{c} = (B^T B)^{-1} B^T y`
     - `pymars/utils.py`
     - `solve_least_squares()`
   * - RSS reduction
     - :math:`\Delta RSS = RSS_{\text{old}} - RSS_{\text{new}}`
     - `pymars/model.py`
     - `ForwardPass._find_best_split()`

**Complexity:**

- Per iteration: :math:`O(M \cdot d \cdot N \cdot |\text{knots}|)`
- Total (M iterations): :math:`O(M^2 \cdot d \cdot N \cdot |\text{knots}|)`
- Typically 5–30 iterations, so :math:`O(100 \text{ to } 1000) \times N \times d`

Algorithm 3: Backward Pruning
=============================

**Purpose:** Remove redundant basis functions using GCV criterion.

**Formal Statement** (Friedman 1991, Algorithm 3):

.. code-block:: text

   BACKWARD(B, X, y, d_penalty):
       /* Input: Basis functions B from forward pass
                 Data X, y
                 GCV penalty d_penalty (typically 2–3)
       */
       
       /* Initialize */
       gcv_best = ∞
       sequence = []
       
       /* Pruning loop */
       while |B| > 1:
           /* Try removing each basis */
           gcv_min = ∞
           b_remove = None
           
           for each b in B:
               /* Remove basis b */
               B_temp = B \ {b}
               
               /* Fit model */
               coef = solve_least_squares(B_temp, y)
               
               /* Compute GCV */
               RSS = sum((y - coef)^2)
               df = trace(B_temp * (B_temp^T * B_temp)^{-1} * B_temp^T) 
                    + d_penalty * |B_temp|
               
               gcv = RSS / (N * (1 - df/N)^2)
               
               /* Track minimum */
               if gcv < gcv_min:
                   gcv_min = gcv
                   b_remove = b
           
           /* Remove best candidate */
           if b_remove is not None:
               B = B \ {b_remove}
               
               /* Store state */
               sequence.append((B, gcv_min))
               
               /* Check if improved */
               if gcv_min >= gcv_best:
                   break                /* No improvement; restore previous */
               gcv_best = gcv_min
       
       return sequence

**Select Model:** Choose model with minimum GCV from sequence.

**Key Implementation Details:**

.. list-table::
   :header-rows: 1

   * - Component
     - Description
     - Location
   * - GCV score
     - :math:`\text{GCV} = \frac{\text{RSS}}{N(1 - \text{df}/N)^2}`
     - `pymars/gcv.py`
     - `GCVCalculator.calculate()`
   * - Degrees of freedom
     - :math:`\text{df} = \text{trace}(B(B^T B)^{-1}B^T) + dM`
     - `pymars/gcv.py`
     - `GCVCalculator.complexity()`
   * - Least squares
     - Solve for each candidate removal
     - `pymars/utils.py`
     - `solve_least_squares()`

**Complexity:**

- Per removal: :math:`O(M \cdot N^2 + M^3)` (least squares)
- Total (M removals): :math:`O(M^2 \cdot N^2 + M^4)`
- Typically fast (M ≤ 50)

Pseudocode: Complete MARS Algorithm
====================================

Combining the three algorithms:

.. code-block:: text

   MARS(X, y, M_max, d_max, penalty, minspan, endspan, α):
       
       /* Forward pass */
       B_forward, coef_forward = FORWARD(X, y, M_max, d_max, 
                                          minspan, endspan, α)
       
       /* Backward pass */
       B_sequence = BACKWARD(B_forward, X, y, penalty)
       
       /* Select best model */
       gcv_min = ∞
       B_best = None
       
       for (B_m, gcv_m) in B_sequence:
           if gcv_m < gcv_min:
               gcv_min = gcv_m
               B_best = B_m
       
       /* Fit final model */
       coef_best = solve_least_squares(B_best, y)
       
       return B_best, coef_best, gcv_min

Knot Selection Details
======================

Valid Knot Computation
----------------------

For variable :math:`j`, valid knots respect:

**Minspan constraint:**

.. math::

   \#\{i : x_{ij} < t\} \geq L \quad \text{and} \quad \#\{i : x_{ij} > t\} \geq L

where :math:`L = \lfloor -\log_2(\alpha/N) / 2.5 \rfloor`

**Endspan constraint:**

.. math::

   \min_i \{x_{ij}\} + L_e \leq t \leq \max_i \{x_{ij}\} - L_e

where :math:`L_e = \lceil 3 - \log_2(\alpha/N) \rceil`

**Implementation** (`pymars/utils.py`):

.. code-block:: python

   def get_candidate_knots(X_j, minspan, endspan):
       """
       Returns sorted array of valid knot locations.
       """
       x_min, x_max = X_j.min(), X_j.max()
       
       # Unique values in middle section
       x_sorted = np.sort(X_j)
       n = len(x_sorted)
       
       # Skip minspan obs from each end
       start_idx = minspan
       end_idx = n - minspan - 1
       
       # Apply endspan constraint
       if endspan is not None:
           x_min_valid = x_sorted[endspan - 1]
           x_max_valid = x_sorted[n - endspan]
       else:
           x_min_valid = x_min
           x_max_valid = x_max
       
       # Valid knot candidates
       knots = []
       for i in range(start_idx, end_idx + 1):
           t = (x_sorted[i] + x_sorted[i+1]) / 2
           if x_min_valid <= t <= x_max_valid:
               knots.append(t)
       
       return np.array(knots)

Least Squares Solver
====================

The core operation repeated many times:

.. code-block:: text

   Given: Design matrix B (N × M), response y (N,)
   Solve: min_c ||y - B @ c||_2^2
   Return: c = (B^T B)^{-1} B^T y
   
   Stability chain:
   1. Try: QR decomposition (most stable)
   2. Else: Cholesky on B^T B
   3. Else: Pseudoinverse (most robust)

Implementation (`pymars/utils.py`):

.. code-block:: python

   def solve_least_squares(B, y):
       """Stable least squares solver with fallback chain."""
       try:
           # First try: QR decomposition
           Q, R = np.linalg.qr(B)
           c = np.linalg.solve(R, Q.T @ y)
       except:
           try:
               # Second try: Cholesky
               L = np.linalg.cholesky(B.T @ B)
               c = np.linalg.solve(L.T, 
                   np.linalg.solve(L, B.T @ y))
           except:
               # Last resort: Pseudoinverse
               c = np.linalg.pinv(B) @ y
       return c

GCV Calculation
===============

**Step 1: Fit model**

.. code-block:: python

   coef = solve_least_squares(B, y)
   y_pred = B @ coef

**Step 2: Compute RSS**

.. code-block:: python

   residuals = y - y_pred
   RSS = np.sum(residuals**2)

**Step 3: Compute degrees of freedom**

.. code-block:: python

   # Effective degrees of freedom
   # df = trace(B @ inv(B^T @ B) @ B^T) + d * M
   
   def compute_df(B, d_penalty):
       BtB_inv = np.linalg.pinv(B.T @ B)
       trace_term = np.trace(B @ BtB_inv @ B.T)
       df = trace_term + d_penalty * B.shape[1]
       return df

**Step 4: Compute GCV**

.. code-block:: python

   df = compute_df(B, d_penalty)
   N = B.shape[0]
   gcv = RSS / (N * (1 - df/N)**2)

Implementation Location
=======================

.. list-table::
   :header-rows: 1

   * - Algorithm
     - File
     - Class/Function
   * - Forward Pass
     - `pymars/model.py`
     - `ForwardPass`
   * - Backward Pass
     - `pymars/model.py`
     - `BackwardPass`
     - Combined in `MARS.fit()`
   * - Knot Selection
     - `pymars/utils.py`
     - `get_candidate_knots()`
   * - Minspan/Endspan
     - `pymars/utils.py`
     - `calculate_minspan()`
     - `calculate_endspan()`
   * - Least Squares
     - `pymars/utils.py`
     - `solve_least_squares()`
   * - GCV
     - `pymars/gcv.py`
     - `GCVCalculator`
   * - Basis Functions
     - `pymars/basis.py`
     - `HingeFunction`
     - `BasisFunction`

Numerical Example
=================

Small Example: Friedman 1-D
---------------------------

Data: :math:`y = \sin(\pi x_1) + 0.5 x_2^2`, N=50

**Forward Pass:**

Iteration 1:
   - Try: h(x_1, +, 0.2), h(x_1, -, 0.2)
   - RSS drops from 10.5 to 5.2
   - RSS reduction = 5.3

   Iteration 2:
   - Try: h(x_2, +, 0.0), h(x_2, -, 0.0) [on residual]
   - RSS drops from 5.2 to 2.8
   - RSS reduction = 2.4

   Iteration 3: Continue until RSS plateaus

**Backward Pass:**

   Remove h(x_1, 0.8, -)?
       GCV_without = 1.2
   
   Remove h(x_2, 0.1, +)?
       GCV_without = 3.5 (worse)
   
   Keep all; GCV = 0.8 is minimum

**Result:**

   :math:`\hat{f}(x) = -0.2 + 1.1 h(x_1, +, 0.2) + 0.8 h(x_2, +, 0.1)`

References
==========

See :doc:`references` for full citations.

Primary: Friedman (1991) sections 3.4–3.9.
