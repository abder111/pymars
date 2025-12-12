"""
PyMARS: Pure Python implementation of Multivariate Adaptive Regression Splines
===============================================================================

A complete implementation of Jerome Friedman's MARS algorithm (1991) for
nonparametric regression with automatic variable selection and interaction detection.

Main classes:
    MARS: Main regression model
    BasisFunction: Individual basis function representation
    
Example:
    >>> from pymars import MARS
    >>> model = MARS(max_terms=20, max_degree=2)
    >>> model.fit(X_train, y_train)
    >>> y_pred = model.predict(X_test)
"""

__version__ = "0.1.0"
__author__ = "Es-safi abderrahman"
__email__ = "abd.essafi@edu.umi.ac.ma"

from .mars import MARS
from .basis import BasisFunction, HingeFunction

__all__ = ['MARS', 'BasisFunction', 'HingeFunction']