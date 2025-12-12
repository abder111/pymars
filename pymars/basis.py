"""
Basis functions for MARS
========================

Implements hinge functions and basis function combinations for building
multivariate adaptive regression splines.
"""

import numpy as np
from typing import Optional, List, Tuple


# Global basis ID counter for stable parent-child tracking
_basis_id_counter = {"value": 0}

def _next_basis_id() -> int:
    """Generate next unique basis function ID"""
    _basis_id_counter["value"] += 1
    return _basis_id_counter["value"]


class HingeFunction:
    """
    Single hinge function: [s(x - t)]+ where s ∈ {-1, +1}
    
    The function equals:
        - s(x - t) if s(x - t) > 0
        - 0 otherwise
    
    Parameters
    ----------
    variable : int
        Index of the variable this hinge acts on
    knot : float
        Knot location t
    direction : int
        Sign s: +1 for right hinge, -1 for left hinge
    """
    
    def __init__(self, variable: int, knot: float, direction: int):
        if direction not in [-1, 1]:
            raise ValueError("Direction must be -1 or 1")
        
        self.variable = variable
        self.knot = knot
        self.direction = direction
    
    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate hinge function on data matrix X
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data
            
        Returns
        -------
        values : array, shape (n_samples,)
            Hinge function values
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        x_var = X[:, self.variable]
        result = self.direction * (x_var - self.knot)
        return np.maximum(0, result)
    
    def __repr__(self):
        sign = "+" if self.direction == 1 else "-"
        return f"h(x{self.variable}, {sign}, {self.knot:.3f})"


class BasisFunction:
    """
    MARS basis function: product of hinge functions
    
    B_m(x) = ∏ [s_k(x_v(k,m) - t_k)]+ 
    
    Parameters
    ----------
    hinges : list of HingeFunction
        Individual hinge functions to multiply
    parent_id : int, optional
        ID of parent basis function (for tracking hierarchy)
    basis_id : int, optional
        Unique stable ID for this basis (generated if not provided)
    """
    
    def __init__(self, hinges: Optional[List[HingeFunction]] = None, 
                 parent_id: Optional[int] = None,
                 basis_id: Optional[int] = None):
        self.hinges = hinges if hinges is not None else []
        self.parent_id = parent_id
        self.basis_id = basis_id if basis_id is not None else _next_basis_id()
        self._degree = len(self.hinges)
        self._variables = [h.variable for h in self.hinges]
    
    @property
    def degree(self) -> int:
        """Number of hinges (interaction order)"""
        return self._degree
    
    @property
    def variables(self) -> List[int]:
        """Variables involved in this basis function"""
        return self._variables
    
    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate basis function on data matrix X
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data
            
        Returns
        -------
        values : array, shape (n_samples,)
            Basis function values
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Start with ones (constant)
        result = np.ones(X.shape[0])
        
        # Multiply by each hinge
        for hinge in self.hinges:
            result *= hinge.evaluate(X)
        
        return result
    
    def add_hinge(self, hinge: HingeFunction) -> 'BasisFunction':
        """
        Create new basis function by adding a hinge
        
        Parameters
        ----------
        hinge : HingeFunction
            Hinge to add to product
            
        Returns
        -------
        new_basis : BasisFunction
            New basis function with added hinge
        """
        new_hinges = self.hinges.copy()
        new_hinges.append(hinge)
        return BasisFunction(hinges=new_hinges, parent_id=self.basis_id)
    
    def get_knot_info(self) -> List[Tuple[int, float, int]]:
        """
        Get information about all knots in this basis
        
        Returns
        -------
        knots : list of tuples
            Each tuple: (variable, knot_value, direction)
        """
        return [(h.variable, h.knot, h.direction) for h in self.hinges]
    
    def __repr__(self):
        if len(self.hinges) == 0:
            return f"B_{self.basis_id} (constant)"
        hinge_strs = [str(h) for h in self.hinges]
        hinges_product = " × ".join(hinge_strs)
        return f"B_{self.basis_id}({hinges_product})"


class InteractionConstraint:
    """
    Manages constraints on variable interactions
    
    Parameters
    ----------
    max_degree : int
        Maximum interaction order (number of variables in a product)
    allowed_interactions : list of tuples, optional
        Specific allowed variable combinations
    forbidden_variables : list of int, optional
        Variables that cannot interact with others
    """
    
    def __init__(self, max_degree: int = 1,
                 allowed_interactions: Optional[List[Tuple[int, ...]]] = None,
                 forbidden_variables: Optional[List[int]] = None):
        self.max_degree = max_degree
        self.allowed_interactions = allowed_interactions
        self.forbidden_variables = forbidden_variables or []
    
    def is_valid(self, basis: BasisFunction, new_variable: int) -> bool:
        """
        Check if adding new_variable to basis is valid
        
        Parameters
        ----------
        basis : BasisFunction
            Current basis function
        new_variable : int
            Variable to potentially add
            
        Returns
        -------
        valid : bool
            True if the combination is allowed
        """
        # Check if variable already in basis (no repeated variables)
        if new_variable in basis.variables:
            return False
        
        # Check degree constraint
        if basis.degree >= self.max_degree:
            return False
        
        # Check forbidden variables
        if new_variable in self.forbidden_variables:
            if basis.degree > 0:  # Can only appear alone
                return False
        
        # Check specific allowed interactions
        if self.allowed_interactions is not None:
            new_vars = tuple(sorted(basis.variables + [new_variable]))
            if new_vars not in self.allowed_interactions:
                return False
        
        return True


def build_design_matrix(X: np.ndarray, 
                       basis_functions: List[BasisFunction]) -> np.ndarray:
    """
    Build design matrix from basis functions
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data
    basis_functions : list of BasisFunction
        Basis functions to evaluate
        
    Returns
    -------
    B : array, shape (n_samples, n_basis)
        Design matrix where each column is a basis function evaluation
    """
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    n_samples = X.shape[0]
    n_basis = len(basis_functions)
    
    B = np.empty((n_samples, n_basis))
    
    for j, basis in enumerate(basis_functions):
        B[:, j] = basis.evaluate(X)
    
    return B