# pymars/cubic.py
"""
Cubic Spline Conversion for MARS
=================================

Converts piecewise-linear MARS model to cubic splines 
with continuous first derivatives.
"""

import numpy as np
from typing import List, Tuple, Dict
from .basis import BasisFunction, HingeFunction


class CubicHingeFunction:
    """
    Cubic truncated power hinge function with continuous derivatives
    
    C(x; s, t-, t, t+) where:
    - t is the central knot
    - t-, t+ are side knots for smoothness
    """
    
    def __init__(self, variable: int, knot_center: float, 
                 knot_left: float, knot_right: float, direction: int):
        self.variable = variable
        self.knot_center = knot_center  # t
        self.knot_left = knot_left      # t-
        self.knot_right = knot_right    # t+
        self.direction = direction
        
        # Calculate cubic coefficient for continuity
        self.r_plus = 2.0 / ((knot_right - knot_left) ** 3)
    
    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Evaluate cubic hinge function"""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        x_var = X[:, self.variable]
        result = np.zeros(len(x_var))
        
        t_minus = self.knot_left
        t = self.knot_center
        t_plus = self.knot_right
        
        if self.direction == 1:  # Right hinge
            # x <= t-: linear
            mask1 = x_var <= t_minus
            result[mask1] = x_var[mask1]
            
            # t- < x < t+: cubic
            mask2 = (x_var > t_minus) & (x_var < t_plus)
            dx = x_var[mask2] - t_minus
            result[mask2] = t_minus + self.r_plus * (dx ** 3)
            
            # x >= t+: linear
            mask3 = x_var >= t_plus
            result[mask3] = x_var[mask3]
            
        else:  # Left hinge (direction == -1)
            # Mirror case
            # x <= t-: linear
            mask1 = x_var <= t_minus
            result[mask1] = t_plus - x_var[mask1]
            
            # t- < x < t+: cubic
            mask2 = (x_var > t_minus) & (x_var < t_plus)
            dx = t_plus - x_var[mask2]
            result[mask2] = self.r_plus * (dx ** 3)
            
            # x >= t+: constant (0)
            mask3 = x_var >= t_plus
            result[mask3] = 0.0
        
        return result


class CubicBasisFunction:
    """Basis function composed of cubic hinges"""
    
    def __init__(self, cubic_hinges: List[CubicHingeFunction]):
        self.cubic_hinges = cubic_hinges
        self._degree = len(cubic_hinges)
        self._variables = [h.variable for h in cubic_hinges]
    
    @property
    def degree(self) -> int:
        return self._degree
    
    @property
    def variables(self) -> List[int]:
        return self._variables
    
    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Evaluate cubic basis function"""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        result = np.ones(X.shape[0])
        
        for hinge in self.cubic_hinges:
            result *= hinge.evaluate(X)
        
        return result


def place_side_knots(central_knots: List[float], 
                     x_min: float, x_max: float) -> Dict[float, Tuple[float, float]]:
    """
    Place side knots (t-, t+) for each central knot
    
    Side knots are placed at midpoints between adjacent central knots
    
    Parameters
    ----------
    central_knots : list of float
        Central knot locations (sorted)
    x_min, x_max : float
        Data range on this variable
        
    Returns
    -------
    side_knots : dict
        Maps central_knot -> (knot_left, knot_right)
    """
    if len(central_knots) == 0:
        return {}
    
    sorted_knots = sorted(set(central_knots))
    side_knots = {}
    
    for i, t in enumerate(sorted_knots):
        # Left side knot
        if i == 0:
            t_minus = (x_min + t) / 2.0
        else:
            t_minus = (sorted_knots[i-1] + t) / 2.0
        
        # Right side knot
        if i == len(sorted_knots) - 1:
            t_plus = (t + x_max) / 2.0
        else:
            t_plus = (t + sorted_knots[i+1]) / 2.0
        
        side_knots[t] = (t_minus, t_plus)
    
    return side_knots


def convert_to_cubic(linear_basis: List[BasisFunction], 
                    X: np.ndarray, y: np.ndarray) -> Tuple[List[CubicBasisFunction], np.ndarray]:
    """
    Convert linear MARS model to cubic splines with continuous derivatives
    
    Parameters
    ----------
    linear_basis : list of BasisFunction
        Linear basis functions from MARS
    X : array
        Training data
    y : array
        Training targets
        
    Returns
    -------
    cubic_basis : list of CubicBasisFunction
        Cubic basis functions
    coefficients : array
        New coefficients fitted to cubic basis
    """
    n_features = X.shape[1]
    
    # Collect all central knots by variable
    knots_by_variable = {v: [] for v in range(n_features)}
    
    for basis in linear_basis:
        for hinge in basis.hinges:
            knots_by_variable[hinge.variable].append(hinge.knot)
    
    # Calculate side knots for each variable
    side_knots_by_var = {}
    for v in range(n_features):
        if len(knots_by_variable[v]) > 0:
            x_min = X[:, v].min()
            x_max = X[:, v].max()
            side_knots_by_var[v] = place_side_knots(
                knots_by_variable[v], x_min, x_max
            )
    
    # Convert each linear basis function to cubic
    cubic_basis_list = []
    
    for linear_bf in linear_basis:
        if linear_bf.degree == 0:
            # Constant basis - stays as is
            cubic_basis_list.append(CubicBasisFunction([]))
        else:
            # Convert each hinge to cubic
            cubic_hinges = []
            
            for hinge in linear_bf.hinges:
                v = hinge.variable
                t = hinge.knot
                s = hinge.direction
                
                # Get side knots
                if t in side_knots_by_var[v]:
                    t_minus, t_plus = side_knots_by_var[v][t]
                else:
                    # Fallback if not found
                    x_min = X[:, v].min()
                    x_max = X[:, v].max()
                    t_minus = (x_min + t) / 2.0
                    t_plus = (t + x_max) / 2.0
                
                cubic_hinge = CubicHingeFunction(
                    variable=v,
                    knot_center=t,
                    knot_left=t_minus,
                    knot_right=t_plus,
                    direction=s
                )
                cubic_hinges.append(cubic_hinge)
            
            cubic_basis_list.append(CubicBasisFunction(cubic_hinges))
    
    # Build design matrix with cubic basis
    from .utils import solve_least_squares
    B_cubic = np.column_stack([bf.evaluate(X) for bf in cubic_basis_list])
    
    # Refit coefficients
    coefficients = solve_least_squares(B_cubic, y)
    
    return cubic_basis_list, coefficients


def evaluate_cubic_basis(cubic_basis: List[CubicBasisFunction], 
                         X: np.ndarray) -> np.ndarray:
    """
    Evaluate cubic basis functions on data
    
    Parameters
    ----------
    cubic_basis : list
        Cubic basis functions
    X : array
        Data to evaluate on
        
    Returns
    -------
    B : array
        Design matrix
    """
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    B = np.column_stack([bf.evaluate(X) for bf in cubic_basis])
    return B