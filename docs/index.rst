========================================
PyMARS: MARS Implementation (Friedman 1991)
========================================

.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.8+

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

**PyMARS** is a comprehensive, educational implementation of Multivariate Adaptive Regression Splines (MARS) based on Jerome Friedman's seminal 1991 paper, extended with cubic spline support and full interactive term analysis.

.. contents::
   :local:
   :backlinks: none

What is MARS?
=============

MARS is a **nonparametric regression technique** that:

- ✅ Automatically identifies relevant variables from high-dimensional data
- ✅ Detects two-way (or higher) interaction effects
- ✅ Produces continuous, differentiable models with hinge basis functions
- ✅ Provides interpretable decomposition via ANOVA
- ✅ Works optimally with 3–20 predictors and 50–1000 observations
- ✅ Applies data-driven knot selection with GCV-based model selection

MARS combines the **flexibility of recursive partitioning** (tree-based methods) with the **smoothness of piecewise regression** and cubic extensions.

Key Features
============

Core Capabilities
-----------------

.. list-table::
   :header-rows: 1

   * - Feature
     - Description
     - Status
   * - Forward Pass
     - Greedy basis pair addition
     - ✅ Complete
   * - Backward Pruning
     - GCV-optimized reduction
     - ✅ Complete
   * - Basis Functions
     - Hinge products: :math:`h(x,t,d) = (d(x-t))_+`
     - ✅ Complete
   * - Knot Selection
     - Minspan/Endspan from Friedman (1991)
     - ✅ Complete
   * - Interactions
     - Max-degree detection & visualization
     - ✅ Complete
   * - Cubic Splines
     - Continuous 2nd derivative
     - ✅ Complete
   * - ANOVA Decomposition
     - Univariate & interaction effects
     - ✅ Complete
   * - GCV Model Selection
     - Generalized cross-validation with penalty
     - ✅ Complete

Theory & References
-------------------

This implementation fully adheres to:

- **Friedman, J.H. (1991)** – "Multivariate Adaptive Regression Splines." *The Annals of Statistics*, 19(1), 1–67. [[PDF]](https://projecteuclid.org/euclid.aos/1176347963)
- **Friedman & Silverman (1989)** – Flexible parsimonious smoothing and additive modeling.

Every major algorithm, equation, and formula from the original paper is reproduced and verified.

Quick Start
===========

Installation
------------

.. code-block:: bash

   git clone https://github.com/abder111/pymars.git
   cd pymars
   pip install -e .

Basic Usage
-----------

.. code-block:: python

   from pymars import MARS
   import numpy as np

   # Generate data
   X = np.random.randn(200, 5)
   y = X[:, 0]**2 + 2*X[:, 1]*X[:, 2] + np.random.randn(200)*0.1

   # Fit MARS model
   model = MARS(max_terms=20, max_degree=2)
   model.fit(X, y)

   # Predictions
   y_pred = model.predict(X)

   # Model summary
   model.summary()

Documentation
==============

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   user_guide
   tutorial

.. toctree::
   :maxdepth: 2
   :caption: Theory & Algorithms

   theory
   algorithms
   model_selection

.. toctree::
   :maxdepth: 2
   :caption: Features

   cubic_extension
   interactions
   plots

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api_reference

.. toctree::
   :maxdepth: 2
   :caption: Advanced

   advanced_topics
   developer_guide
   changelog
   references

Project Status
==============

✅ **Production Ready** – Version 0.1.0

- ✅ All 11 bugs fixed & verified
- ✅ 55+ comprehensive tests (100% passing)
- ✅ Friedman 1991 compliance confirmed
- ✅ Cubic implementation tested
- ✅ Full documentation generated

Development Team
================

- **ES-SAFI ABDERRAHMAN** – Lead Developer
- **LAMGHARI YASSINE** – Core Developer
- **CHAIBOU SAIDOU ABDOUYE** – Core Developer

Repository
==========

- **GitHub:** https://github.com/abder111/pymars
- **License:** MIT (2025)

License
=======

.. code-block:: text

   MIT License
   
   Copyright (c) 2025 ES-SAFI ABDERRAHMAN, LAMGHARI YASSINE, CHAIBOU SAIDOU ABDOUYE
   
   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software...

See LICENSE file for full text.

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
