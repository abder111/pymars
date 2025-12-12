===========
References
===========

Academic Papers
===============

Primary Reference
------------------

.. [Friedman1991]
   Friedman, J.H. (1991).
   "Multivariate Adaptive Regression Splines."
   *The Annals of Statistics*, 19(1), 1–67.
   
   **DOI:** 10.1214/aos/1176347963
   
   **Direct Link:** https://projecteuclid.org/euclid.aos/1176347963
   
   **Full PDF:** Available through Project Euclid
   
   **Significance:** The seminal paper introducing MARS. Every algorithm, equation, and formula in this implementation is derived from this work.

Foundational Theory
--------------------

.. [Friedman1989]
   Friedman, J.H. & Silverman, B.W. (1989).
   "Flexible Parsimonious Smoothing and Additive Modeling."
   *Technometrics*, 31(1), 3–21.
   
   **DOI:** 10.2307/1268578
   
   **Content:** Discusses B-spline bases and flexible regression that motivated MARS.

Complementary Methods
-----------------------

.. [Hastie2009]
   Hastie, T., Tibshirani, R., & Friedman, J. (2009).
   *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.).
   Springer.
   
   **Sections:** 9.3-9.4 (MARS algorithm overview)
   
   **Content:** Modern treatment of MARS within broader ML context.

.. [Breiman2001]
   Breiman, L. (2001).
   "Statistical Modeling: The Two Cultures."
   *Statistical Science*, 16(3), 199–231.
   
   **Content:** Philosophical context for nonparametric methods like MARS.

Spline Theory
--------------

.. [Wahba1990]
   Wahba, G. (1990).
   *Spline Models for Observational Data.*
   CBMS-NSF Regional Conference Series in Applied Mathematics.
   
   **Content:** Comprehensive spline theory; GCV selection criterion.

Numerical Methods
-------------------

.. [Golub1999]
   Golub, G.H. & Pereyra, V. (1973).
   "The Differentiation of Pseudo-Inverses and Nonlinear Least-Squares Problems
   Whose Variables Separate."
   *SIAM Journal on Numerical Analysis*, 10(2), 413–432.
   
   **Content:** Pseudoinverse properties (relevant to least-squares solver).

.. [Björck1996]
   Björck, Å. (1996).
   *Numerical Methods for Least Squares Problems.*
   SIAM.
   
   **Content:** Comprehensive reference on QR, Cholesky, and pseudoinverse methods.

Model Selection & Cross-Validation
------------------------------------

.. [Stone1974]
   Stone, M. (1974).
   "Cross-Validatory Choice and Assessment of Statistical Predictions."
   *Journal of the Royal Statistical Society*, 36(2), 111–147.
   
   **Content:** GCV theory foundation.

.. [Craven1978]
   Craven, P. & Wahba, G. (1978).
   "Smoothing Noisy Data with Spline Functions."
   *Numerische Mathematik*, 31, 377–403.
   
   **Content:** GCV formula derivation.

Related Methods
===============

Regression Splines
-------------------

* **Natural Cubic Splines** - Constrained cubic splines through data points
* **Smoothing Splines** - Penalized regression using smoothing parameter
* **Thin-plate Splines** - Multivariate generalization of smoothing splines
* **B-splines** - Basis functions with compact support (used in MARS)

Tree-Based Methods
-------------------

* **CART (Classification and Regression Trees)** - Recursive partitioning (conceptual basis for MARS)
* **Random Forests** - Ensemble of trees (similar flexibility, different approach)
* **Gradient Boosting** - Iterative tree fitting (can be combined with MARS)

Kernel Methods
---------------

* **Kernel Ridge Regression** - Nonlinear regression via kernel trick
* **Support Vector Regression** - Robust regression with sparsity
* **Gaussian Processes** - Bayesian nonparametric regression

Additive Models
----------------

* **Generalized Additive Models (GAMs)** - Flexible additive models
* **Local Polynomials** - LOESS/LOWESS local fitting
* **Projection Pursuit** - Greedy basis expansion (similar philosophy)

Online Resources
================

Documentation & Tutorials
--------------------------

* **Official PyMARS Documentation:** https://pymars.readthedocs.io
* **GitHub Repository:** https://github.com/abder111/pymars
* **Installation Guide:** https://pymars.readthedocs.io/installation.html
* **User Guide:** https://pymars.readthedocs.io/user_guide.html
* **API Reference:** https://pymars.readthedocs.io/api_reference.html

Implementations
----------------

* **MARS in R (earth package):** https://CRAN.R-project.org/package=earth
  
  Most feature-complete open-source MARS implementation

* **MARS in R (caret package):** https://topepo.github.io/caret/
  
  Integrated with machine learning framework

* **MARS in SAS:** SAS/STAT PROC TRANSREG
  
  Commercial implementation

* **Gensym G2 MARS:** Enterprise implementation

Mathematical Resources
-----------------------

* **Linear Algebra Review:** https://math.mit.edu/~gs/linearalgebra/
  
  Understanding matrices, QR, eigenvalues

* **Numerical Recipes:** https://numerical.recipes/
  
  Implementation details for numerical algorithms

* **University of Illinois Statistics:** https://www.stat.illinois.edu/
  
  Courses on splines and nonparametric methods

Books
======

Essential References
---------------------

1. **Hastie, Tibshirani, & Friedman (2009)**
   *The Elements of Statistical Learning*
   - Comprehensive ML reference
   - MARS coverage in Chapter 9
   - Also covers GAMs, trees, boosting

2. **Friedman (1991) Original Paper**
   *Multivariate Adaptive Regression Splines*
   - Must-read for understanding MARS deeply
   - All mathematics and algorithms detailed

3. **Wahba (1990)**
   *Spline Models for Observational Data*
   - Theoretical foundation for smoothing splines
   - GCV derivation and theory

4. **Björck (1996)**
   *Numerical Methods for Least Squares Problems*
   - Implementation details
   - Solver stability and alternatives

Related Reading
----------------

* De Boor, C. (1978). *A Practical Guide to Splines.* Springer.
* Hastie, T., Tibshirani, R. (1990). *Generalized Additive Models.* Chapman & Hall.
* Rippa, S. (1999). "An Algorithm for Selecting a Good Smoothing Parameter for Spline Smoothing." *Numerische Mathematik*.

Theory Resources
=================

GCV & Model Selection
---------------------

The GCV formula used in MARS:

.. math::

   \text{GCV}(M) = \frac{\text{RSS}(M)}{N(1 - C(M)/N)^2}

is derived in:

* Craven & Wahba (1978)
* Hastie & Tibshirani (1990), Chapter 3
* Rippa (1999)

Knot Selection
---------------

Minspan and Endspan formulas:

.. math::

   L = \lfloor -\log_2(\alpha/N) / 2.5 \rfloor

derived in Friedman (1991), Section 3.8

Based on statistical theory of:

* Confidence intervals
* Type I error rates
* Minimum cell sizes

Cubic Splines
--------------

Cubic basis functions:

.. math::

   h_{\text{cubic}}(x, t, d) = [d(x-t)]_+^3

Theory in:

* Wahba (1990)
* de Boor (1978)
* Friedman (1991), Section 3.7

Interaction Theory
-------------------

Interaction terms and ANOVA decomposition:

* Friedman & Silverman (1989)
* Friedman (1991), Section 3.5
* Hastie et al. (2009), Chapter 9.3

Software Dependencies
=====================

Core
----

* **NumPy** (≥1.19.0) - Numerical computing
* **SciPy** (≥1.5.0) - Scientific computing (linear algebra)

Optional
--------

* **Matplotlib** (≥3.3.0) - Visualization
* **scikit-learn** - Compatibility and utilities
* **Pandas** - Data I/O and manipulation

Development
-----------

* **pytest** - Testing framework
* **black** - Code formatting
* **flake8** - Code linting
* **mypy** - Static type checking
* **sphinx** - Documentation generation
* **sphinx-rtd-theme** - RTD theme

Citing This Work
================

**In Academic Papers:**

.. code-block:: bibtex

   @software{pymars2025,
     title={PyMARS: A Python Implementation of Multivariate Adaptive Regression Splines},
     author={ES-SAFI, Abderrahman and LAMGHARI, Yassine and CHAIBOU, Saidou Abdouye},
     year={2025},
     url={https://github.com/abder111/pymars}
   }

**In Text:**

  We used PyMARS (ES-SAFI et al., 2025), a Python implementation of the MARS algorithm (Friedman, 1991).

**Original Algorithm Citation:**

When using MARS, always cite Friedman (1991):

.. code-block:: bibtex

   @article{friedman1991mars,
     title={Multivariate adaptive regression splines},
     author={Friedman, Jerome H},
     journal={The Annals of Statistics},
     volume={19},
     number={1},
     pages={1--67},
     year={1991},
     publisher={JSTOR}
   }

Glossary
========

.. glossary::

   Basis Function
      A function used to build the MARS model; products of hinge functions.

   Forward Pass
      Phase where MARS greedily adds basis function pairs to fit the data.

   Backward Pass
      Phase where MARS prunes basis functions using GCV selection.

   GCV
      Generalized Cross-Validation; criterion for model selection balancing fit and complexity.

   Hinge Function
      Univariate piecewise-linear function: :math:`h(x, t, d) = \max(0, d(x-t))`.

   Interaction
      When the effect of one variable depends on another variable.

   Knot
      Location where the hinge function changes slope.

   Minspan
      Minimum observations between consecutive knots.

   Endspan
      Minimum observations from domain boundaries to first/last knot.

   MARS
      Multivariate Adaptive Regression Splines.

   Penalty
      Parameter :math:`d` in GCV that controls pruning severity.

   Smoothing Spline
      Penalized regression with smoothing parameter.

Index
=====

For an alphabetical index of all terms, see the Index page or search the documentation.
