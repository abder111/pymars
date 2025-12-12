"""
Utility functions for MARS
===========================

Matrix operations, knot selection, and helper functions.
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy import linalg


def solve_least_squares(A: np.ndarray, b: np.ndarray, 
                       ridge: float = 1e-10) -> np.ndarray:
    """
    Solve least squares problem: min ||Ax - b||^2
    
    Uses np.linalg.lstsq for robustness (handles under/overdetermined systems).
    Falls back to regularized normal equations if needed.
    
    Parameters
    ----------
    A : array, shape (n_samples, n_features)
        Design matrix
    b : array, shape (n_samples,)
        Target values
    ridge : float, default=1e-10
        Ridge parameter for fallback regularization
        
    Returns
    -------
    x : array, shape (n_features,)
        Least squares solution
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).ravel()
    
    # Primary: use numpy's lstsq (handles all cases robustly)
    try:
        x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        return x
    except (np.linalg.LinAlgError, ValueError):
        pass  # Fall through to ridge regression
    
    # Fallback 1: Regularized normal equations with Cholesky
    try:
        ATA = A.T @ A
        if ridge > 0:
            ATA = ATA + ridge * np.eye(ATA.shape[0])
        ATb = A.T @ b
        
        # Try Cholesky for speed
        L = linalg.cholesky(ATA, lower=True)
        x = linalg.solve_triangular(L, ATb, lower=True)
        x = linalg.solve_triangular(L.T, x, lower=False)
        return x
    except linalg.LinAlgError:
        pass  # Fall through to pseudoinverse
    
    # Fallback 2: Pseudoinverse (most robust but slow)
    try:
        return np.linalg.pinv(A, rcond=1e-15) @ b
    except Exception as e:
        raise ValueError(f"solve_least_squares failed: {e}")


def get_candidate_knots(X: np.ndarray, variable: int,
                       minspan: int = 0,
                       existing_knots: Optional[List[float]] = None) -> np.ndarray:
    """
    Get candidate knot locations for a variable
    
    Selects knot candidates ensuring minimum spacing in sorted data.
    minspan is the minimum number of sorted samples between consecutive knots.
    
    Parameters
    ----------
    X : array, shape (n_samples, n_features) or (n_samples,)
        Input data
    variable : int
        Variable index (0 if X is 1D)
    minspan : int
        Minimum number of sorted observations between knots
    existing_knots : list of float, optional
        Already used knot locations to avoid duplicates
        
    Returns
    -------
    knots : array
        Candidate knot locations (unique, sorted)
    """
    # Extract variable
    if X.ndim == 1:
        x_var = X
    else:
        x_var = X[:, variable]
    
    # Sort and get unique values with their first occurrence positions
    order = np.argsort(x_var)
    x_sorted = x_var[order]
    unique_vals, idx_first = np.unique(x_sorted, return_index=True)
    
    if len(unique_vals) <= 1:
        return np.array([])
    
    # Apply minspan constraint: keep knots separated by at least minspan sorted positions
    if minspan <= 0:
        knots = unique_vals.copy()
    else:
        kept = []
        last_pos = -np.inf
        for pos, val in zip(idx_first, unique_vals):
            if pos - last_pos >= minspan:
                kept.append(val)
                last_pos = pos
        knots = np.array(kept)
    
    # Remove duplicates with existing knots
    if existing_knots is not None and len(existing_knots) > 0:
        tol = 1e-12
        mask = np.ones(len(knots), dtype=bool)
        for ek in existing_knots:
            mask &= np.abs(knots - ek) > tol
        knots = knots[mask]
    
    return knots


def calculate_minspan(n_samples: int, 
                     alpha: float = 0.05) -> int:
    """
    Calculate minimum span between knots
    
    From Friedman (1991):
        L = -log2(alpha/n) / 2.5
    
    Parameters
    ----------
    n_samples : int
        Number of samples
    alpha : float, default=0.05
        Significance level for run resistance
        
    Returns
    -------
    minspan : int
        Minimum number of observations between knots
    """
    if n_samples < 10:
        return 0
    
    # Friedman's formula (page 94): L = -log2(alpha/n) / 2.5
    # where n = n_samples (NOT n_features)
    l_star = -np.log2(alpha / n_samples) / 2.5
    minspan = max(0, int(np.floor(l_star)))
    
    return minspan


def calculate_endspan(n_samples: int, n_features: int, alpha: float = 0.05) -> int:
    """
    Calculate minimum span from endpoints
    
    From Friedman (1991):
        Le = 3 - log2(alpha/n)
    
    Parameters
    ----------
    n_samples : int
        Number of samples
    n_features : int
        Number of features
    alpha : float, default=0.05
        Significance level
        
    Returns
    -------
    endspan : int
        Minimum observations from endpoints
    """
    # Friedman's formula (page 94): Le = 3 - log2(alpha/n)
    # where n = n_samples (NOT n_features)
    le = 3 - np.log2(alpha / n_samples)
    endspan = max(1, int(np.ceil(le)))
    return endspan


def apply_endspan_constraint(knots: np.ndarray, 
                            x_values: np.ndarray,
                            endspan: int) -> np.ndarray:
    """
    Remove knots too close to data endpoints
    
    Ensures knots are at least endspan observations from the start and end
    of the sorted data.
    
    Parameters
    ----------
    knots : array
        Candidate knot locations
    x_values : array
        Data values for this variable
    endspan : int
        Minimum observations from endpoints
        
    Returns
    -------
    valid_knots : array
        Knots satisfying endspan constraint (strictly between boundaries)
    """
    if len(knots) == 0 or endspan <= 0:
        return knots
    
    x_sorted = np.sort(x_values)
    n = len(x_sorted)
    
    # Not enough data
    if n <= 2 * endspan:
        return np.array([])
    
    # Boundaries: knots must be strictly between these values
    # Allow endspan samples on each side
    lower_bound = x_sorted[endspan]        # endspan observations from start
    upper_bound = x_sorted[n - endspan - 1]  # endspan observations from end
    
    # Filter knots: must be strictly inside the boundaries
    mask = (knots > lower_bound) & (knots < upper_bound)
    return knots[mask]


def standardize_data(X: np.ndarray, 
                    mean: Optional[np.ndarray] = None,
                    std: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize features to zero mean and unit variance
    
    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        Input data
    mean : array, optional
        Pre-computed means (for transform)
    std : array, optional
        Pre-computed standard deviations (for transform)
        
    Returns
    -------
    X_scaled : array
        Standardized data
    mean : array
        Feature means
    std : array
        Feature standard deviations
    """
    if mean is None:
        mean = np.mean(X, axis=0)
    if std is None:
        std = np.std(X, axis=0)
        std[std == 0] = 1.0  # Avoid division by zero
    
    X_scaled = (X - mean) / std
    return X_scaled, mean, std


def unstandardize_predictions(y: np.ndarray,
                             y_mean: float,
                             y_std: float) -> np.ndarray:
    """
    Transform predictions back to original scale
    
    Parameters
    ----------
    y : array
        Standardized predictions
    y_mean : float
        Original target mean
    y_std : float
        Original target std
        
    Returns
    -------
    y_original : array
        Predictions in original scale
    """
    return y * y_std + y_mean


def linear_extrapolation(x: float, x1: float, y1: float, 
                        x2: float, y2: float) -> float:
    """
    Linear extrapolation through two points
    
    Parameters
    ----------
    x : float
        Point to extrapolate to
    x1, y1 : float
        First point
    x2, y2 : float
        Second point
        
    Returns
    -------
    y : float
        Extrapolated value
    """
    if abs(x2 - x1) < 1e-10:
        return y1
    
    slope = (y2 - y1) / (x2 - x1)
    return y1 + slope * (x - x1)


def cubic_interpolation(x: float, knots: List[float], 
                       values: List[float]) -> float:
    """
    Piecewise cubic interpolation with continuous derivatives
    
    Parameters
    ----------
    x : float
        Point to interpolate
    knots : list of float
        Knot locations (sorted)
    values : list of float
        Function values at knots
        
    Returns
    -------
    y : float
        Interpolated value
    """
    if len(knots) < 2:
        return values[0] if values else 0.0
    
    # Find interval
    idx = np.searchsorted(knots, x)
    
    if idx == 0:
        # Before first knot - linear extrapolation
        return linear_extrapolation(x, knots[0], values[0], 
                                    knots[1], values[1])
    elif idx >= len(knots):
        # After last knot - linear extrapolation
        return linear_extrapolation(x, knots[-2], values[-2],
                                    knots[-1], values[-1])
    else:
        # Cubic Hermite interpolation
        t = (x - knots[idx-1]) / (knots[idx] - knots[idx-1])
        t2 = t * t
        t3 = t2 * t
        
        h00 = 2*t3 - 3*t2 + 1
        h10 = t3 - 2*t2 + t
        h01 = -2*t3 + 3*t2
        h11 = t3 - t2
        
        # Estimate derivatives (finite differences)
        if idx == 1:
            m0 = (values[1] - values[0]) / (knots[1] - knots[0])
        else:
            m0 = (values[idx] - values[idx-2]) / (knots[idx] - knots[idx-2])
        
        if idx == len(knots) - 1:
            m1 = (values[idx] - values[idx-1]) / (knots[idx] - knots[idx-1])
        else:
            m1 = (values[idx+1] - values[idx-1]) / (knots[idx+1] - knots[idx-1])
        
        dx = knots[idx] - knots[idx-1]
        return (h00 * values[idx-1] + h10 * dx * m0 + 
                h01 * values[idx] + h11 * dx * m1)