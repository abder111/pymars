"""
Generalized Cross-Validation (GCV) for MARS
===========================================

Implements the GCV criterion for model selection:
    GCV = RSS / [N * (1 - C(M)/N)^2]

Where C(M) is the complexity penalty accounting for both fitted
parameters and the adaptive basis function selection.
"""

import numpy as np
from typing import Optional


class GCVCalculator:
    """
    Calculate Generalized Cross-Validation scores
    
    Parameters
    ----------
    penalty : float, default=3.0
        Penalty parameter d for each basis function optimization
        Friedman recommends:
            - d = 2 for additive models
            - d = 3 for general MARS
    """
    
    def __init__(self, penalty: float = 3.0):
        self.penalty = penalty
    
    def complexity(self, n_basis: int, design_matrix: np.ndarray) -> float:
        """
        Calculate complexity cost C(M)
        
        C(M) = trace[B(B'B)^-1 B'] + d*M
        
        The first term is the effective number of parameters in linear fit (trace).
        The second term accounts for adaptive basis selection.
        
        Parameters
        ----------
        n_basis : int
            Number of basis functions (including constant)
        design_matrix : array, shape (n_samples, n_cols)
            Design matrix B
            
        Returns
        -------
        complexity : float
            Total complexity cost
        """
        B = np.asarray(design_matrix, dtype=float)
        n_samples, n_cols = B.shape
        
        # Effective parameters from linear fit: trace[B @ pinv(B)]
        try:
            B_pinv = np.linalg.pinv(B, rcond=1e-15)
            c_linear = float(np.trace(B @ B_pinv))
        except (np.linalg.LinAlgError, ValueError):
            # Fallback: use SVD rank
            try:
                _, s, _ = np.linalg.svd(B, full_matrices=False)
                tol = s.max() * max(B.shape) * np.finfo(float).eps
                c_linear = float(np.sum(s > tol))
            except Exception:
                c_linear = float(min(n_samples, n_cols))
        
        # Additional penalty for adaptive basis selection
        # M = n_basis - 1 (don't count constant term)
        M = max(0, n_basis - 1)
        c_adaptive = self.penalty * M
        
        return c_linear + c_adaptive
    
    def calculate(self, residuals: np.ndarray, 
                  design_matrix: np.ndarray,
                  n_basis: int) -> float:
        """
        Calculate GCV score
        
        GCV = (RSS/N) / (1 - C(M)/N)^2
        
        Parameters
        ----------
        residuals : array, shape (n_samples,)
            Residuals from model fit
        design_matrix : array, shape (n_samples, n_basis)
            Design matrix
        n_basis : int
            Number of basis functions (including constant)
            
        Returns
        -------
        gcv_score : float
            GCV criterion value (lower is better)
        """
        n_samples = len(residuals)
        
        # RSS (Residual Sum of Squares)
        rss = np.sum(residuals ** 2)
        
        # Complexity penalty
        c = self.complexity(n_basis, design_matrix)
        
        # GCV formula with numerical stability
        eps = 1e-15
        denominator = n_samples * (1.0 - c / n_samples) ** 2
        
        # Protect against overfitting (c >= N means no degrees of freedom)
        if c >= n_samples * (1.0 - eps):
            return np.inf
        
        gcv = rss / denominator
        return gcv
    
    def calculate_from_fit(self, y_true: np.ndarray,
                          y_pred: np.ndarray,
                          design_matrix: np.ndarray,
                          n_basis: int) -> float:
        """
        Calculate GCV from predictions
        
        Parameters
        ----------
        y_true : array, shape (n_samples,)
            True target values
        y_pred : array, shape (n_samples,)
            Predicted values
        design_matrix : array, shape (n_samples, n_basis)
            Design matrix
        n_basis : int
            Number of basis functions
            
        Returns
        -------
        gcv_score : float
            GCV criterion value
        """
        residuals = y_true - y_pred
        return self.calculate(residuals, design_matrix, n_basis)


class CVCalculator:
    """
    Standard k-fold cross-validation calculator
    
    Parameters
    ----------
    n_folds : int, default=10
        Number of CV folds
    random_state : int, optional
        Random seed for fold generation
    """
    
    def __init__(self, n_folds: int = 10, 
                 random_state: Optional[int] = None):
        self.n_folds = n_folds
        self.random_state = random_state
    
    def split_folds(self, n_samples: int) -> list:
        """
        Generate fold indices
        
        Parameters
        ----------
        n_samples : int
            Number of samples
            
        Returns
        -------
        folds : list of tuples
            Each tuple contains (train_idx, test_idx)
        """
        rng = np.random.RandomState(self.random_state)
        indices = np.arange(n_samples)
        rng.shuffle(indices)
        
        fold_sizes = np.full(self.n_folds, n_samples // self.n_folds, dtype=int)
        fold_sizes[:n_samples % self.n_folds] += 1
        
        current = 0
        folds = []
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_idx = indices[start:stop]
            train_idx = np.concatenate([indices[:start], indices[stop:]])
            folds.append((train_idx, test_idx))
            current = stop
        
        return folds


def calculate_rss(residuals: np.ndarray) -> float:
    """
    Calculate Residual Sum of Squares
    
    Parameters
    ----------
    residuals : array-like
        Residuals from model fit
        
    Returns
    -------
    rss : float
        Sum of squared residuals
    """
    return float(np.sum(np.asarray(residuals) ** 2))


def calculate_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Squared Error
    
    Parameters
    ----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
        
    Returns
    -------
    mse : float
        Mean squared error
    """
    return float(np.mean((y_true - y_pred) ** 2))


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate R-squared score
    
    Parameters
    ----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
        
    Returns
    -------
    r2 : float
        R-squared score
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    
    return 1.0 - (ss_res / ss_tot)