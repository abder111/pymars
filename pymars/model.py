"""
Forward and backward stepwise algorithms for MARS
=================================================

Implements the core forward selection and backward pruning procedures.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from .basis import BasisFunction, HingeFunction, build_design_matrix, InteractionConstraint
from .gcv import GCVCalculator
from .utils import solve_least_squares, get_candidate_knots, apply_endspan_constraint


class ForwardPass:
    """
    Forward stepwise basis function selection
    
    Builds an overfitted model by iteratively adding basis function pairs
    that best reduce the residual sum of squares.
    
    Parameters
    ----------
    max_terms : int
        Maximum number of basis functions to create
    max_degree : int
        Maximum interaction order
    minspan : int
        Minimum samples between knots
    endspan : int
        Minimum samples from data endpoints
    """
    
    def __init__(self, max_terms: int = 30,
                 max_degree: int = 1,
                 minspan: int = 0,
                 endspan: int = 0):
        self.max_terms = max_terms
        self.max_degree = max_degree
        self.minspan = minspan
        self.endspan = endspan
        self.constraint = InteractionConstraint(max_degree=max_degree)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Tuple[List[BasisFunction], np.ndarray]:
        """
        Run forward pass to build basis function set
        
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Input features
        y : array, shape (n_samples,)
            Target values
            
        Returns
        -------
        basis_functions : list of BasisFunction
            Selected basis functions (including constant)
        coefficients : array
            Fitted coefficients
        """
        n_samples, n_features = X.shape
        
        # Initialize with constant term
        constant_basis = BasisFunction(hinges=[])
        basis_functions = [constant_basis]
        
        # Track best RSS
        B = build_design_matrix(X, basis_functions)
        coef = solve_least_squares(B, y)
        y_pred = B @ coef
        best_rss = np.sum((y - y_pred) ** 2)
        
        print(f"Forward pass: Initial RSS = {best_rss:.6f}")
        
        # Iteratively add basis function pairs
        # Note: basis_functions[0] is the constant, so (len - 1) = number of non-constant bases
        iteration = 0
        while (len(basis_functions) - 1) < self.max_terms:
            iteration += 1
            
            # Find best basis pair to add
            result = self._find_best_split(X, y, basis_functions)
            
            if result is None:
                print(f"Forward pass terminated: No improvement found")
                break
            
            new_basis_left, new_basis_right, new_rss, parent_idx, var_idx, knot = result
            
            # Check for improvement
            improvement = best_rss - new_rss
            if improvement <= 0:
                print(f"Forward pass terminated: No RSS improvement")
                break
            
            # Add both basis functions
            basis_functions.append(new_basis_left)
            basis_functions.append(new_basis_right)
            best_rss = new_rss
            
            print(f"Iter {iteration}: Added basis pair on x{var_idx} at {knot:.3f}, "
                  f"RSS = {best_rss:.6f}, improvement = {improvement:.6f}")
            
            if len(basis_functions) >= self.max_terms + 1:
                break
        
        # Final fit
        B = build_design_matrix(X, basis_functions)
        coefficients = solve_least_squares(B, y)
        
        print(f"Forward pass complete: {len(basis_functions)} basis functions")
        
        return basis_functions, coefficients
    
    def _find_best_split(self, X: np.ndarray, y: np.ndarray,
                        current_basis: List[BasisFunction]) -> Optional[Tuple]:
        """
        Find best basis function pair to add
        
        Returns
        -------
        result : tuple or None
            (left_basis, right_basis, new_rss, parent_idx, var_idx, knot)
        """
        n_samples, n_features = X.shape
        best_rss = np.inf
        best_result = None
        
        # Try splitting each current basis function
        for parent_idx, parent_basis in enumerate(current_basis):
            
            # Evaluate parent on data
            parent_values = parent_basis.evaluate(X)
            nonzero_mask = parent_values > 1e-10
            
            if np.sum(nonzero_mask) < 2 * max(self.minspan, 1):
                continue  # Not enough samples
            
            # Try each variable
            for var_idx in range(n_features):
                
                # Check if variable can be added (interaction constraints)
                if not self.constraint.is_valid(parent_basis, var_idx):
                    continue
                
                # Get candidate knots
                X_subset = X[nonzero_mask]
                knots = get_candidate_knots(X_subset, var_idx, 
                                          minspan=self.minspan)
                
                # Apply endspan constraint
                if self.endspan > 0:
                    knots = apply_endspan_constraint(
                        knots, X_subset[:, var_idx], self.endspan
                    )
                
                if len(knots) == 0:
                    continue
                
                # Try each knot
                for knot in knots:
                    rss = self._evaluate_split(
                        X, y, current_basis, parent_basis, 
                        var_idx, knot, parent_idx
                    )
                    
                    if rss < best_rss:
                        best_rss = rss
                        
                        # Create new basis functions
                        hinge_left = HingeFunction(var_idx, knot, direction=-1)
                        hinge_right = HingeFunction(var_idx, knot, direction=1)
                        
                        left_basis = parent_basis.add_hinge(hinge_left)
                        right_basis = parent_basis.add_hinge(hinge_right)
                        
                        best_result = (left_basis, right_basis, best_rss, 
                                     parent_idx, var_idx, knot)
        
        return best_result
    
    def _evaluate_split(self, X: np.ndarray, y: np.ndarray,
                       current_basis: List[BasisFunction],
                       parent_basis: BasisFunction,
                       var_idx: int, knot: float,
                       parent_idx: int) -> float:
        """
        Evaluate RSS for a candidate split
        
        Returns
        -------
        rss : float
            Residual sum of squares with this split
        """
        # Create candidate basis functions
        hinge_left = HingeFunction(var_idx, knot, direction=-1)
        hinge_right = HingeFunction(var_idx, knot, direction=1)
        
        left_basis = parent_basis.add_hinge(hinge_left)
        right_basis = parent_basis.add_hinge(hinge_right)
        
        # Build design matrix with new basis
        trial_basis = current_basis + [left_basis, right_basis]
        B = build_design_matrix(X, trial_basis)
        
        # Fit and compute RSS
        try:
            coef = solve_least_squares(B, y)
            y_pred = B @ coef
            rss = np.sum((y - y_pred) ** 2)
        except (np.linalg.LinAlgError, ValueError):
            rss = np.inf
        
        return rss


class BackwardPass:
    """
    Backward stepwise basis function pruning
    
    Removes basis functions one at a time to optimize GCV score.
    
    Parameters
    ----------
    gcv_calculator : GCVCalculator
        GCV calculator instance
    """
    
    def __init__(self, gcv_calculator: GCVCalculator):
        self.gcv_calculator = gcv_calculator
    
    def prune(self, X: np.ndarray, y: np.ndarray,
              basis_functions: List[BasisFunction],
              coefficients: np.ndarray) -> Tuple[List[BasisFunction], np.ndarray, float]:
        """
        Prune basis functions to minimize GCV
        
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Input features
        y : array, shape (n_samples,)
            Target values
        basis_functions : list
            Initial (overfitted) basis functions
        coefficients : array
            Initial coefficients
            
        Returns
        -------
        best_basis : list
            Pruned basis functions
        best_coef : array
            Pruned coefficients
        best_gcv : float
            Best GCV score achieved
        """
        print(f"\nBackward pass: Starting with {len(basis_functions)} basis functions")
        
        # Calculate initial GCV
        B = build_design_matrix(X, basis_functions)
        y_pred = B @ coefficients
        current_gcv = self.gcv_calculator.calculate_from_fit(
            y, y_pred, B, len(basis_functions)
        )
        
        print(f"Initial GCV = {current_gcv:.6f}")
        
        best_basis = basis_functions.copy()
        best_coef = coefficients.copy()
        best_gcv = current_gcv
        
        current_basis = basis_functions.copy()
        
        # Iteratively remove basis functions
        iteration = 0
        while len(current_basis) > 1:  # Keep at least constant term
            iteration += 1
            
            # Try removing each basis (except constant)
            min_gcv = np.inf
            remove_idx = None
            
            for idx in range(1, len(current_basis)):  # Skip constant at idx=0
                # Create candidate basis set
                trial_basis = current_basis[:idx] + current_basis[idx+1:]
                
                # Refit
                B_trial = build_design_matrix(X, trial_basis)
                coef_trial = solve_least_squares(B_trial, y)
                y_pred_trial = B_trial @ coef_trial
                
                # Calculate GCV
                gcv = self.gcv_calculator.calculate_from_fit(
                    y, y_pred_trial, B_trial, len(trial_basis)
                )
                
                if gcv < min_gcv:
                    min_gcv = gcv
                    remove_idx = idx
            
            # Remove the basis function that gave best GCV
            if remove_idx is not None:
                removed = current_basis.pop(remove_idx)
                
                # Refit with removed basis
                B_new = build_design_matrix(X, current_basis)
                coef_new = solve_least_squares(B_new, y)
                
                print(f"Iter {iteration}: Removed basis {remove_idx}, "
                      f"GCV = {min_gcv:.6f}, {len(current_basis)} basis remaining")
                
                # Track best model
                if min_gcv < best_gcv:
                    best_gcv = min_gcv
                    best_basis = current_basis.copy()
                    best_coef = coef_new.copy()
            else:
                break
        
        print(f"\nBackward pass complete: Best model has {len(best_basis)} basis functions")
        print(f"Best GCV = {best_gcv:.6f}")
        
        return best_basis, best_coef, best_gcv