====================================
Theory & Mathematical Foundation
====================================

.. contents::
   :local:
   :backlinks: none

Introduction
============

This section provides the mathematical foundations of MARS, based entirely on:

**Friedman, J.H. (1991).** "Multivariate Adaptive Regression Splines." *The Annals of Statistics*, 19(1), 1–67.

Every equation, algorithm, and formula in this documentation maps directly to the original paper. Implementation details are noted where applicable.

Multivariate Adaptive Regression Splines
=========================================

MARS is a **nonparametric regression method** that learns piecewise-linear (or piecewise-cubic) functions through adaptive knot placement.

The model has the form:

.. math::

   \hat{f}(x) = \sum_{m=1}^{M} c_m B_m(x)

where:
- :math:`M` = number of basis functions
- :math:`c_m` = fitted coefficients
- :math:`B_m(x)` = basis functions

Basis Functions
===============

Linear Hinges (Friedman 1991, Eq. 19)
-------------------------------------

Each basis function is a **product of univariate hinge functions**:

.. math::

   B_m(x) = \prod_{k=1}^{K_m} h_k(x)

where each hinge function :math:`h_k(x)` has the form:

.. math::

   h(x_j, t, d) = \max(0, d(x_j - t))

with:
- :math:`x_j` = :math:`j`-th predictor
- :math:`t` = knot location
- :math:`d \in \{-1, +1\}` = direction indicator
- :math:`(z)_+ = \max(0, z)` = positive part function

**Interpretation:**
- :math:`h(x_j, +, t) = (x_j - t)_+` = ReLU on right side of knot
- :math:`h(x_j, -, t) = (t - x_j)_+` = ReLU on left side of knot

The product creates a "roof function" that is zero outside :math:`[t_1, t_2]`.

Cubic Splines (Friedman 1991, Sec. 3.7)
-----------------------------------------

For smooth continuity, replace hinges with cubic basis:

.. math::

   h_{\text{cubic}}(x_j, t, d) = (d(x_j - t))_+^3

This gives continuous second derivatives.

Interaction Terms
-----------------

Products of hinges naturally create interaction terms:

.. math::

   B_m(x) = h(x_i, t_1, d_1) \cdot h(x_j, t_2, d_2)

This is a **2-way interaction** between variables :math:`i` and :math:`j`.

**max_degree** parameter limits interaction order:
- max_degree=1: Only main effects
- max_degree=2: Up to 2-way interactions
- max_degree=k: Up to k-way interactions

Knot Selection
==============

Minspan Formula (Friedman 1991, Eq. 43)
-----------------------------------------

Minimum observations between knots:

.. math::

   L = \left\lfloor -\frac{\log_2(\alpha / N)}{2.5} \right\rfloor

where:
- :math:`N` = sample size
- :math:`\alpha` = significance level (default 0.05)

**Interpretation:** Enforces a minimum gap between knots to avoid overfitting.

Endspan Formula (Friedman 1991, Eq. 45)
-----------------------------------------

Minimum observations from domain boundaries:

.. math::

   L_e = \left\lceil 3 - \log_2(\alpha / N) \right\rceil

**Interpretation:** Prevents knots very close to the endpoints of the feature range.

The Forward Pass
================

Algorithm Overview
------------------

The forward pass is a **greedy basis expansion** algorithm (Friedman 1991, Algorithm 2):

**Input:** Data :math:`(x_i, y_i)`, :math:`i=1,\ldots,N`

**Output:** Set of basis functions :math:`\mathcal{B}`

**Algorithm:**

.. code-block:: text

   Initialize: B = {1}  (constant term)
   
   For m = 1 to M_max:
       For each basis function b in B:
           For each variable j in 1..d:
               For each observation i in valid knots (respecting minspan/endspan):
                   For each direction d in {+1, -1}:
                       Fit pair: h^+ = h(x_j, t_i, +) * b
                                 h^- = h(x_j, t_i, -) * b
                       Compute residuals
                       Calculate RSS reduction
       
       Add pair with maximum RSS reduction to B
   
   Return B

**Key Points:**

1. **Greedy selection:** Always pick the pair reducing RSS most
2. **Pairwise expansion:** Adds two basis functions per iteration
3. **Knot respects constraints:** minspan and endspan enforced
4. **Continuation:** Continues until :math:`M_{\max}` basis functions

Least Squares Solve
-------------------

At each iteration, solve:

.. math::

   \hat{c} = \arg\min_c \sum_{i=1}^{N} \left(y_i - \sum_m c_m B_m(x_i)\right)^2

Using QR decomposition → Cholesky → pseudoinverse chain (see :doc:`advanced_topics`).

The Backward Pass
=================

Generalized Cross-Validation (GCV)
-----------------------------------

After forward pass, **prune basis functions** using GCV (Friedman 1991, Eq. 30):

.. math::

   \text{GCV}(M) = \frac{\text{RSS}(M)}{N(1 - C(M)/N)^2}

where **complexity penalty** is:

.. math::

   C(M) = \text{trace}(B(B^T B)^{-1}B^T) + dM

with:
- :math:`B` = :math:`N \times M` design matrix
- :math:`d` = penalty per basis function (typically 2–3)

**Interpretation:**

- Numerator: Residual sum of squares (fit quality)
- Denominator: Complexity adjustment (penalizes more terms)
- Lower GCV = better model selection

Backward Pruning Algorithm
---------------------------

**Algorithm** (Friedman 1991, Algorithm 3):

.. code-block:: text

   Start: B = set of all basis functions from forward pass
   
   While |B| > 1:
       For each basis function b in B:
           Remove b from B
           Fit model
           Compute GCV(B)
           Add b back
       
       b* = basis function whose removal minimizes GCV
       Remove b* from B
       
       If GCV(B) not improved:
           Restore all removed terms
           Break
   
   Return final B

**Result:** A sequence of nested models with decreasing complexity.

**Model Selection:** Choose model with minimum GCV score.

GCV Complexity Analysis
========================

Design Matrix
-------------

Let :math:`B` be the :math:`N \times M` design matrix where each column is a basis function.

Effective Degrees of Freedom
-----------------------------

The trace term in Eq. 32:

.. math::

   \text{df}(M) = \text{trace}(B(B^T B)^{-1}B^T)

represents the **effective degrees of freedom**.

This is between :math:`M` (if columns orthogonal) and :math:`MN` (saturated).

Penalty Parameter
------------------

The GCV penalty :math:`d` controls pruning aggressiveness:

- :math:`d = 2` (additive models): Less aggressive pruning
- :math:`d = 3` (general models): Standard pruning
- :math:`d = 4` (conservative): Very aggressive pruning

**Effect:** Higher :math:`d` → simpler models (more pruning).

Model Equations Summary
========================

**Fitted Model:**

.. math::

   \hat{f}(x) = c_0 + \sum_{m=1}^{M} c_m \prod_{k=1}^{K_m} h_k(x)

**Loss Function (RSS):**

.. math::

   \text{RSS} = \sum_{i=1}^{N} (y_i - \hat{f}(x_i))^2

**GCV Selection Criterion:**

.. math::

   \text{GCV} = \frac{\text{RSS}}{N(1 - C(M)/N)^2}

**Complexity Penalty:**

.. math::

   C(M) = \text{trace}(B(B^T B)^{-1}B^T) + dM

Cubic Spline Extension
======================

Cubic Basis (Friedman 1991, Sec. 3.7)
---------------------------------------

For smoother models with continuous 2nd derivatives, use cubic hinges:

.. math::

   h_{\text{cubic}}(x_j, t, d) = [d(x_j - t)]_+^3

This replaces linear hinges in the basis expansion.

**Continuity Properties:**

- :math:`C^0` continuity: Guaranteed (hinges are continuous)
- :math:`C^1` continuity: Guaranteed (first derivative continuous)
- :math:`C^2` continuity: Guaranteed (second derivative continuous)

**Motivation:** Cubic bases are smoother but still interpretable, avoiding overly wiggly functions.

Knot Placement (Cubic)
-----------------------

When converting to cubic, place "side knots" to ensure cubic smoothness:

For a hinge at knot :math:`t`, place side knots at:

.. math::

   t^- = \text{median}(x_j : x_j < t)

   t^+ = \text{median}(x_j : x_j > t)

This creates a cubic "power basis" with :math:`C^2` continuity everywhere.

The cubic coefficient becomes:

.. math::

   r^+ = \frac{2}{(t^+ - t^-)^3}

(Friedman 1991, Eq. 34–35)

Numerical Implementation
========================

Standardization
---------------

Features are standardized to mean 0, variance 1:

.. math::

   x_j^{(std)} = \frac{x_j - \bar{x}_j}{s_j}

where :math:`\bar{x}_j` and :math:`s_j` are sample mean and standard deviation.

**Benefits:**
- Improves numerical stability
- Makes knot locations comparable across scales
- Makes coefficients interpretable (standardized effect sizes)

Knot Validity
--------------

A knot at :math:`t` for variable :math:`j` is **valid** if:

.. math::

   \#\{i : x_{ij} < t\} \geq L \quad \text{and} \quad \#\{i : x_{ij} > t\} \geq L

where :math:`L` is minspan.

Similarly, **endspan** constraints ensure sufficient observations near domain boundaries.

Condition Number Management
----------------------------

To avoid singular least-squares problems:

1. Standardize features (default)
2. Use pseudoinverse when condition number is high
3. Avoid collinear basis functions (model pruning helps)

Connection to Paper Sections
=============================

.. list-table::
   :header-rows: 1

   * - Section
     - Topic
     - Formula/Algorithm
   * - 3.1
     - Recursive Partitioning
     - Basis function formulation
   * - 3.2
     - Continuity (q=1)
     - Linear hinge definition
   * - 3.3
     - Generalization
     - Interaction terms
   * - 3.4
     - MARS Algorithm
     - Algorithms 1, 2, 3
   * - 3.5
     - ANOVA
     - Decomposition analysis
   * - 3.6
     - GCV
     - Equations 30–32
   * - 3.7
     - Cubic Continuity
     - Equations 34–35
   * - 3.8
     - Knot Optimization
     - Minspan, Endspan
   * - 3.9
     - Computational
     - Efficiency strategies

References
==========

The complete references are in :doc:`references`.

Key citations:
- Friedman (1991) – Original MARS paper
- Friedman & Silverman (1989) – Flexible smoothing methods

Notation Reference
==================

.. list-table::
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`N`
     - Number of observations
   * - :math:`d`
     - Number of features
   * - :math:`M`
     - Number of basis functions
   * - :math:`x_i`
     - :math:`i`-th observation (vector)
   * - :math:`y_i`
     - :math:`i`-th target value
   * - :math:`B_m(x)`
     - :math:`m`-th basis function
   * - :math:`c_m`
     - Coefficient for basis :math:`m`
   * - :math:`t`
     - Knot location
   * - :math:`d`
     - Direction (+1 or -1)
   * - :math:`h(x, t, d)`
     - Hinge function
   * - :math:`\alpha`
     - Significance level
   * - :math:`L`
     - Minspan
   * - :math:`L_e`
     - Endspan
