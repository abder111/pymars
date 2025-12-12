"""
Main MARS regression class
==========================

Public API for Multivariate Adaptive Regression Splines.
"""

import numpy as np
from typing import Optional, Dict, List
from .basis import BasisFunction, build_design_matrix
from .model import ForwardPass, BackwardPass
from .gcv import GCVCalculator, calculate_mse, calculate_r2
from .utils import (standardize_data, unstandardize_predictions,
                   calculate_minspan, calculate_endspan, solve_least_squares)


class MARS:
    """
    Multivariate Adaptive Regression Splines (MARS)
    
    Nonparametric regression method that automatically selects relevant
    variables and detects interactions through adaptive basis function
    construction.
    
    Parameters
    ----------
    max_terms : int, default=30
        Maximum number of basis functions to create in forward pass.
        The final model will typically have fewer due to pruning.
        
    max_degree : int, default=1
        Maximum interaction order (number of variables in a product).
        - max_degree=1: Additive model (no interactions)
        - max_degree=2: Pairwise interactions allowed
        - max_degree=n: Full interactions allowed
        
    penalty : float, default=3.0
        GCV penalty parameter (d in Friedman 1991).
        - penalty=2: Recommended for additive models
        - penalty=3: Recommended for general MARS
        Larger values produce smoother models.
        
    minspan : int or 'auto', default='auto'
        Minimum number of observations between knots.
        'auto' uses Friedman's formula: -log2(alpha/n) / 2.5
        
    endspan : int or 'auto', default='auto'
        Minimum observations from data endpoints.
        'auto' uses Friedman's formula: 3 - log2(alpha/n)
        
    alpha : float, default=0.05
        Significance level for automatic span calculation.
        Only used if minspan or endspan is 'auto'.
        
    standardize : bool, default=True
        Whether to standardize features before fitting.
        Recommended for numerical stability.
        
    verbose : bool, default=True
        Whether to print progress messages during fitting.
    
    Attributes
    ----------
    basis_functions_ : list of BasisFunction
        Selected basis functions after pruning
        
    coefficients_ : array
        Fitted coefficients for basis functions
        
    gcv_score_ : float
        Final GCV score
        
    n_features_in_ : int
        Number of input features
        
    feature_importances_ : array
        Importance score for each feature
    
    Examples
    --------
    >>> from pymars import MARS
    >>> import numpy as np
    >>> 
    >>> # Generate synthetic data
    >>> X = np.random.randn(100, 5)
    >>> y = X[:, 0]**2 + 2*X[:, 1] + np.random.randn(100)*0.1
    >>> 
    >>> # Fit MARS model
    >>> model = MARS(max_terms=20, max_degree=2)
    >>> model.fit(X, y)
    >>> 
    >>> # Make predictions
    >>> y_pred = model.predict(X)
    >>> 
    >>> # View summary
    >>> model.summary()
    """
    
    def __init__(self,
                 max_terms: int = 30,
                 max_degree: int = 1,
                 penalty: float = 3.0,
                 minspan: str = 'auto',
                 endspan: str = 'auto',
                 alpha: float = 0.05,
                 standardize: bool = True,
                 verbose: bool = True,smooth: bool = False
                 ):
        
        self.max_terms = max_terms
        self.max_degree = max_degree
        self.penalty = penalty
        self.minspan = minspan
        self.endspan = endspan
        self.alpha = alpha
        self.standardize = standardize
        self.verbose = verbose
        
        # Fitted attributes (set during fit)
        self.basis_functions_: Optional[List[BasisFunction]] = None
        self.coefficients_: Optional[np.ndarray] = None
        self.gcv_score_: Optional[float] = None
        self.n_features_in_: Optional[int] = None
        self.feature_importances_: Optional[np.ndarray] = None
        
        # Standardization parameters
        self._x_mean: Optional[np.ndarray] = None
        self._x_std: Optional[np.ndarray] = None
        self._y_mean: float = 0.0
        self._y_std: float = 1.0
        
        # Training metrics
        self._train_mse: Optional[float] = None
        self._train_r2: Optional[float] = None
        
        self.smooth = smooth  # Enable cubic conversion
        self.cubic_basis_ = None  # Store cubic basis if used
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MARS':
        """
        Fit MARS model to data
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training input features
        y : array-like, shape (n_samples,)
            Training target values
            
        Returns
        -------
        self : MARS
            Fitted estimator
        """
        # Validate and convert inputs
        X, y = self._validate_data(X, y)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"MARS Fitting")
            print(f"{'='*60}")
            print(f"Samples: {n_samples}, Features: {n_features}")
            print(f"Max terms: {self.max_terms}, Max degree: {self.max_degree}")
            print(f"Penalty: {self.penalty}")
        
        # Standardize if requested
        if self.standardize:
            X_scaled, self._x_mean, self._x_std = standardize_data(X)
            y_scaled, y_mean, y_std = standardize_data(y.reshape(-1, 1))
            y_scaled = y_scaled.ravel()
            self._y_mean = y_mean[0]
            self._y_std = y_std[0]
        else:
            X_scaled = X.copy()
            y_scaled = y.copy()
        
        # Calculate span parameters if auto
        if self.minspan == 'auto':
            minspan = calculate_minspan(n_samples, self.alpha)
        else:
            minspan = self.minspan
        
        if self.endspan == 'auto':
            endspan = calculate_endspan(n_samples, n_features, self.alpha)
        else:
            endspan = self.endspan
        
        if self.verbose:
            print(f"Minspan: {minspan}, Endspan: {endspan}")
            print()
        
        # Forward pass: build large model
        forward = ForwardPass(
            max_terms=self.max_terms,
            max_degree=self.max_degree,
            minspan=minspan,
            endspan=endspan
        )
        basis_functions, coefficients = forward.fit(X_scaled, y_scaled)
        
        # Backward pass: prune to optimal size
        gcv_calc = GCVCalculator(penalty=self.penalty)
        backward = BackwardPass(gcv_calculator=gcv_calc)
        
        self.basis_functions_, self.coefficients_, self.gcv_score_ = backward.prune(
            X_scaled, y_scaled, basis_functions, coefficients
        )
        
        # Calculate feature importances
        self.feature_importances_ = self._calculate_feature_importance()
        
        # Calculate training metrics
        y_pred = self.predict(X)
        self._train_mse = calculate_mse(y, y_pred)
        self._train_r2 = calculate_r2(y, y_pred)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Training complete")
            print(f"Final model: {len(self.basis_functions_)} basis functions")
            print(f"GCV score: {self.gcv_score_:.6f}")
            print(f"Training MSE: {self._train_mse:.6f}")
            print(f"Training R²: {self._train_r2:.6f}")
            print(f"{'='*60}\n")
        
        # NOUVEAU: Convert to cubic if requested
        if self.smooth:
            from .cubic import convert_to_cubic
            
            self.cubic_basis_, self.coefficients_ = convert_to_cubic(
                self.basis_functions_, X_scaled, y_scaled
            )
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using fitted MARS model
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input features (in original scale, not standardized)
            
        Returns
        -------
        y_pred : array, shape (n_samples,)
            Predicted values
            
        Notes
        -----
        All knot locations in basis_functions_ are stored in the standardized domain
        (after mean/std transformation if self.standardize=True). Input X is 
        automatically scaled to match before evaluation.
        """
        if self.basis_functions_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X = self._validate_input(X)
        
        # Standardize if needed (knots are in standardized domain)
        if self.standardize:
            X_scaled = (X - self._x_mean) / self._x_std
        else:
            X_scaled = X
        
        # Evaluate basis functions
        B = build_design_matrix(X_scaled, self.basis_functions_)
        y_pred = B @ self.coefficients_
        
        if self.smooth and self.cubic_basis_ is not None:
            # Use cubic basis
            from .cubic import evaluate_cubic_basis
            B = evaluate_cubic_basis(self.cubic_basis_, X_scaled)
        else:
            # Use linear basis
            B = build_design_matrix(X_scaled, self.basis_functions_)
        
        y_pred = B @ self.coefficients_
        
        # Unstandardize predictions
        if self.standardize:
            y_pred = unstandardize_predictions(y_pred, self._y_mean, self._y_std)
        
        return y_pred
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate R² score on test data
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Test features
        y : array-like, shape (n_samples,)
            True values
            
        Returns
        -------
        r2 : float
            R² score
        """
        y_pred = self.predict(X)
        return calculate_r2(y, y_pred)
    
    def summary(self) -> Dict:
        """
        Get model summary with statistics and basis functions
        
        Returns
        -------
        summary : dict
            Model information including:
            - n_basis: Number of basis functions
            - gcv_score: GCV score
            - train_mse: Training MSE (if available)
            - train_r2: Training R² (if available)
            - basis_info: List of basis function descriptions
            - feature_importances: Feature importance scores
        """
        if self.basis_functions_ is None or len(self.basis_functions_) == 0:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Collect basis function info
        basis_info = []
        for i, (basis, coef) in enumerate(zip(self.basis_functions_, 
                                              self.coefficients_)):
            info = {
                'index': i,
                'coefficient': float(coef),
                'degree': basis.degree,
                'variables': basis.variables,
                'description': str(basis)
            }
            basis_info.append(info)
        
        # Calculate max degree safely
        max_degree = max((b.degree for b in self.basis_functions_), default=0)
        
        # Handle optional training metrics safely
        train_mse = float(self._train_mse) if hasattr(self, '_train_mse') and self._train_mse is not None else None
        train_r2 = float(self._train_r2) if hasattr(self, '_train_r2') and self._train_r2 is not None else None
        
        summary = {
            'n_basis': len(self.basis_functions_),
            'max_degree_achieved': max_degree,
            'n_features': self.n_features_in_,
            'gcv_score': float(self.gcv_score_) if self.gcv_score_ is not None else None,
            'train_mse': train_mse,
            'train_r2': train_r2,
            'basis_functions': basis_info,
            'feature_importances': self.feature_importances_.tolist()
        }
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"MARS Model Summary")
        print(f"{'='*70}")
        print(f"Number of basis functions: {summary['n_basis']}")
        print(f"Number of features: {summary['n_features']}")
        print(f"Maximum degree: {summary['max_degree_achieved']}")
        if summary['gcv_score'] is not None:
            print(f"GCV score: {summary['gcv_score']:.6f}")
        if train_mse is not None and train_r2 is not None:
            print(f"Training MSE: {train_mse:.6f}")
            print(f"Training R²: {train_r2:.6f}")
            print(f"Training MSE: {summary['train_mse']:.6f}")
            print(f"Training R²: {summary['train_r2']:.6f}")
        
        print(f"\nFeature Importances:")
        for i, imp in enumerate(summary['feature_importances']):
            print(f"  x{i}: {imp:.4f}")
        
        print(f"\nBasis Functions:")
        for info in basis_info[:10]:  # Show first 10
            print(f"  [{info['index']}] coef={info['coefficient']:8.4f}  {info['description']}")
        if len(basis_info) > 10:
            print(f"  ... ({len(basis_info) - 10} more)")
        
        print(f"{'='*70}\n")
        
        return summary
    
    def get_anova_decomposition(self) -> Dict[int, List[BasisFunction]]:
        """
        Get ANOVA decomposition of model
        
        Returns functions grouped by interaction order:
        - order 0: constant
        - order 1: main effects (single variables)
        - order 2: two-way interactions
        - etc.
        
        Returns
        -------
        anova : dict
            Maps interaction order to list of basis functions
        """
        if self.basis_functions_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        anova = {}
        for basis in self.basis_functions_:
            order = basis.degree
            if order not in anova:
                anova[order] = []
            anova[order].append(basis)
        
        return anova
    
    def _calculate_feature_importance(self) -> np.ndarray:
        """
        Calculate feature importance scores
        
        Importance = sum of |coefficient| for all basis functions
        containing that variable. The constant term (first basis) is excluded.
        """
        if self.basis_functions_ is None or len(self.basis_functions_) == 0:
            return np.zeros(self.n_features_in_)
        
        importance = np.zeros(self.n_features_in_)
        
        # Skip the constant term (first basis function)
        for i, (basis, coef) in enumerate(zip(self.basis_functions_[1:], self.coefficients_[1:]), 1):
            # Only count non-zero coefficients for non-constant terms
            if abs(coef) > 0:
                for var in basis.variables:
                    importance[var] += abs(coef)
        
        # Normalize to [0, 1]
        if importance.sum() > 0:
            importance /= importance.sum()
        
        # Ensure shape matches n_features_in_
        importance = importance[:self.n_features_in_]
        
        return importance
    
    def _validate_data(self, X, y):
        """Validate and convert training data"""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if len(y) != X.shape[0]:
            raise ValueError(f"X and y must have same length: {X.shape[0]} vs {len(y)}")
        
        return X, y
    
    def _validate_input(self, X):
        """Validate prediction input"""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")
        return X