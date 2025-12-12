=========
Changelog
=========

All notable changes to PyMARS will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/>`_, and this project adheres to `Semantic Versioning <https://semver.org/>`_.

[0.1.0] - 2025-01-15
====================

**Initial Release**

Added
-----

* Complete MARS implementation following Friedman (1991)
* Forward pass with greedy basis expansion
* Backward pass with GCV-based pruning
* Basis functions with hinge products
* GCV model selection with automatic penalty
* Cubic spline extension (smooth=True parameter)
* ANOVA decomposition by interaction order
* Feature importance calculation
* Knot selection with minspan/endspan constraints
* Standardization support
* Interaction support (max_degree parameter)
* Comprehensive test suite (55+ tests)
* Full documentation with Sphinx/RTD
* Interactive examples and tutorials
* Visualization tools for model interpretation
* Cross-compatible with scikit-learn API

Core Modules
~~~~~~~~~~~~

* ``pymars.mars.MARS`` - Main model class
* ``pymars.basis.HingeFunction`` - Hinge basis
* ``pymars.basis.BasisFunction`` - Composite basis
* ``pymars.utils.calculate_minspan()`` - Knot spacing
* ``pymars.utils.calculate_endspan()`` - Endpoint spacing
* ``pymars.utils.solve_least_squares()`` - Stable solver
* ``pymars.gcv.GCVCalculator`` - Model selection
* ``pymars.model.ForwardPass`` - Basis expansion
* ``pymars.model.BackwardPass`` - Pruning
* ``pymars.cubic.CubicHingeFunction`` - Cubic basis
* ``pymars.cubic.convert_to_cubic()`` - Conversion
* ``pymars.interactions.InteractionAnalysis`` - ANOVA
* ``pymars.plots.*`` - Visualization functions

Features
~~~~~~~~

* Automatic variable selection
* Interaction detection
* Multivariate regression
* Model interpretability
* Generalized cross-validation
* Knot optimization
* Cubic continuity
* ANOVA decomposition
* Feature importance
* Cross-validation compatible

Documentation
~~~~~~~~~~~~~~

* Installation guide
* User guide with examples
* Complete tutorial (13 steps)
* Mathematical theory (Friedman 1991)
* Algorithm details (3 core algorithms)
* API reference
* Model selection guide (GCV)
* Cubic splines documentation
* Interaction analysis guide
* Plotting guide
* Advanced topics (numerical methods, optimization)
* Developer guide
* References and citations

Testing
~~~~~~~

* Unit tests for all modules
* Comprehensive integration tests
* Friedman 1991 test functions
* Cubic implementation verification
* Cross-validation tests
* Edge case testing
* Numerical stability tests

Known Limitations
~~~~~~~~~~~~~~~~~

* Single-output regression only (no multi-output)
* Forward pass greedy (not globally optimal)
* Cubic implementation is post-hoc (not during fitting)
* No built-in parallelization
* Assumes continuous features (no categorical support)
* No missing value handling

Future Enhancements (Planned)
=============================

[0.2.0] (Planned)
-----------------

* Categorical variable support
* Missing value handling
* Weighted observations
* Multiple penalty parameters per basis
* Sparse basis representations
* Incremental learning
* GPU acceleration for large datasets

[0.3.0] (Planned)
-----------------

* Multi-output regression
* Classification via logistic MARS
* Regularization (L1/L2) options
* Adaptive penalty selection
* Built-in cross-validation grid search
* Parallel knot search

[1.0.0] (Planned)
-----------------

* Production-ready API stability
* Comprehensive benchmarking suite
* Performance optimizations
* Extended documentation
* Community examples and case studies

Bug Fixes
=========

[0.1.0]
-------

* All 11 identified implementation bugs fixed from initial version
* Numerical stability verified via QR/Cholesky/pseudoinverse fallback
* GCV calculation validated against theoretical formula
* Cubic implementation tested against Friedman paper examples
* Minspan/Endspan formulas verified

Contributors
=============

**ES-SAFI ABDERRAHMAN** - Lead Developer, Implementation, Testing  
**LAMGHARI YASSINE** - Core Development, Model Verification  
**CHAIBOU SAIDOU ABDOUYE** - Testing, Documentation  

Acknowledgments
===============

* Jerome H. Friedman for the original MARS algorithm (Friedman 1991)
* Friedman & Silverman (1989) for foundational smoothing theory
* scikit-learn community for ML standards
* Sphinx/RTD for documentation infrastructure

Citation
========

If you use PyMARS in academic work, please cite:

**Original Algorithm:**

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

**This Implementation:**

.. code-block:: bibtex

   @software{pymars2025,
     title={PyMARS: A Python Implementation of Multivariate Adaptive Regression Splines},
     author={ES-SAFI, Abderrahman and LAMGHARI, Yassine and CHAIBOU, Saidou Abdouye},
     year={2025},
     url={https://github.com/abder111/pymars},
     version={0.1.0}
   }

License
=======

PyMARS is released under the MIT License.

See LICENSE file for details.

---

For more information, see the `README <https://github.com/abder111/pymars>`_ and `documentation <https://pymars.readthedocs.io>`_.
