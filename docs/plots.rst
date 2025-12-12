===========
Plotting & Visualization
===========

.. contents::
   :local:
   :backlinks: none

Overview
========

PyMARS includes comprehensive visualization tools for understanding model behavior, diagnostics, and interpreting results.

Basic Imports
=============

.. code-block:: python

   from pymars.plots import (
       plot_basis_functions,
       plot_predictions,
       plot_residuals,
       plot_feature_importance,
       plot_partial_effects,
       plot_anova_decomposition
   )

Feature Importance
==================

Visualize which variables matter most:

.. code-block:: python

   from pymars.plots import plot_feature_importance
   
   plot_feature_importance(model, figsize=(10, 5))
   plt.title('Feature Importances')
   plt.show()

Predictions vs Actual
=====================

Compare predictions with true values:

.. code-block:: python

   from pymars.plots import plot_predictions
   
   plot_predictions(model, X, y, figsize=(12, 5))
   plt.suptitle('Model Predictions')
   plt.show()

Residual Diagnostics
====================

Assess model assumptions:

.. code-block:: python

   from pymars.plots import plot_residuals
   
   plot_residuals(model, X, y, figsize=(15, 10))
   plt.suptitle('Residual Diagnostics')
   plt.show()

Partial Effects
===============

Univariate slice plots:

.. code-block:: python

   from pymars.plots import plot_partial_effects
   
   plot_partial_effects(model, X, features=[0, 1, 2], figsize=(12, 4))
   plt.suptitle('Partial Effects Plots')
   plt.show()

ANOVA Decomposition
===================

Visualize interaction effects:

.. code-block:: python

   from pymars.plots import plot_anova_decomposition
   
   plot_anova_decomposition(model, figsize=(10, 6))
   plt.show()

Full API Reference
==================

.. automodule:: pymars.plots
   :members:
   :undoc-members:
   :show-inheritance:
