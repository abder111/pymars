"""
Advanced interaction analysis for MARS
======================================

Tools for analyzing and visualizing variable interactions in MARS models.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from itertools import combinations
from .basis import BasisFunction


class InteractionAnalyzer:
    """
    Analyze interactions in fitted MARS model
    
    Parameters
    ----------
    model : MARS
        Fitted MARS model
    """
    
    def __init__(self, model):
        self.model = model
        if model.basis_functions_ is None:
            raise ValueError("Model must be fitted first")
    
    def get_interaction_strength(self) -> Dict[Tuple[int, ...], float]:
        """
        Calculate strength of each variable interaction
        
        Returns sum of absolute coefficients for each unique variable set.
        
        Returns
        -------
        interactions : dict
            Maps variable tuple to interaction strength
        """
        interactions = {}
        
        for basis, coef in zip(self.model.basis_functions_, 
                              self.model.coefficients_):
            if basis.degree == 0:
                continue  # Skip constant
            
            var_tuple = tuple(sorted(basis.variables))
            
            if var_tuple in interactions:
                interactions[var_tuple] += abs(coef)
            else:
                interactions[var_tuple] = abs(coef)
        
        return interactions
    
    def get_pairwise_interactions(self) -> np.ndarray:
        """
        Get matrix of pairwise interaction strengths
        
        Returns
        -------
        matrix : array, shape (n_features, n_features)
            Symmetric matrix where [i,j] is strength of x_i * x_j interaction
        """
        n_features = self.model.n_features_in_
        matrix = np.zeros((n_features, n_features))
        
        for basis, coef in zip(self.model.basis_functions_, 
                              self.model.coefficients_):
            if basis.degree == 2:  # Two-way interaction
                vars = sorted(basis.variables)
                i, j = vars[0], vars[1]
                matrix[i, j] += abs(coef)
                matrix[j, i] += abs(coef)
        
        return matrix
    
    def rank_interactions(self, top_k: int = 10) -> List[Tuple]:
        """
        Rank interactions by strength
        
        Parameters
        ----------
        top_k : int
            Number of top interactions to return
            
        Returns
        -------
        ranked : list of tuples
            Each tuple: (variables, strength)
        """
        interactions = self.get_interaction_strength()
        
        # Sort by strength
        ranked = sorted(interactions.items(), key=lambda x: x[1], reverse=True)
        
        return ranked[:top_k]
    
    def find_pure_additive_effects(self) -> List[int]:
        """
        Find variables that only appear in additive terms (degree=1)
        and never in interactions (degree > 1)
        
        Returns
        -------
        variables : list of int
            Variable indices with only additive effects (sorted)
        """
        # Collect variables that appear in interactions (degree > 1)
        interaction_vars = set()
        for basis in self.model.basis_functions_:
            if basis.degree > 1:
                interaction_vars.update(basis.variables)
        
        # Find variables that appear only in degree=1 terms
        # and are NOT in any interaction
        additive_only = set()
        for basis in self.model.basis_functions_:
            if basis.degree == 1:
                var = basis.variables[0]
                if var not in interaction_vars:
                    additive_only.add(var)
        
        return sorted(additive_only)
    
    def decompose_prediction(self, x: np.ndarray) -> Dict[str, float]:
        """
        Decompose a single prediction into contributions
        
        Parameters
        ----------
        x : array, shape (n_features,)
            Single input vector (in original, non-standardized scale)
            
        Returns
        -------
        contributions : dict
            Maps component name to its contribution value
            
        Notes
        -----
        If model.standardize=True, input x is automatically standardized
        to match the domain where basis functions were trained.
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # Standardize input if model was trained with standardization
        if hasattr(self.model, 'standardize') and self.model.standardize:
            x_eval = (x - self.model._x_mean) / self.model._x_std
        else:
            x_eval = x
        
        contributions = {'constant': 0.0}
        
        for basis, coef in zip(self.model.basis_functions_, 
                              self.model.coefficients_):
            value = basis.evaluate(x_eval)[0]
            contrib = coef * value
            
            if basis.degree == 0:
                contributions['constant'] = contrib
            elif basis.degree == 1:
                var = basis.variables[0]
                key = f'x{var}'
                contributions[key] = contributions.get(key, 0.0) + contrib
            else:
                var_str = 'x' + '*x'.join(map(str, sorted(basis.variables)))
                key = f'interaction_{var_str}'
                contributions[key] = contributions.get(key, 0.0) + contrib
        
        return contributions
    
    def interaction_test(self, var1: int, var2: int,
                        X: np.ndarray, y: np.ndarray) -> float:
        """
        Test if interaction between two variables improves fit
        
        Compares model with and without the specific interaction.
        
        Parameters
        ----------
        var1, var2 : int
            Variable indices
        X, y : arrays
            Test data
            
        Returns
        -------
        improvement : float
            R² improvement from including interaction
        """
        from .mars import MARS
        
        # Fit additive model
        model_add = MARS(max_terms=20, max_degree=1, verbose=False)
        model_add.fit(X, y)
        r2_add = model_add.score(X, y)
        
        # Fit with interactions
        model_int = MARS(max_terms=20, max_degree=2, verbose=False)
        model_int.fit(X, y)
        r2_int = model_int.score(X, y)
        
        # Check if the specific interaction was selected
        has_interaction = False
        for basis in model_int.basis_functions_:
            if basis.degree == 2:
                vars_in_basis = set(basis.variables)
                if vars_in_basis == {var1, var2}:
                    has_interaction = True
                    break
        
        if has_interaction:
            return r2_int - r2_add
        else:
            return 0.0
    
    def hierarchical_interaction_map(self) -> Dict[int, Set[Tuple[int, ...]]]:
        """
        Create hierarchical map of interactions
        
        Returns
        -------
        hierarchy : dict
            Maps degree to set of variable tuples at that degree
        """
        hierarchy = {}
        
        for basis in self.model.basis_functions_:
            degree = basis.degree
            if degree == 0:
                continue
            
            var_tuple = tuple(sorted(basis.variables))
            
            if degree not in hierarchy:
                hierarchy[degree] = set()
            hierarchy[degree].add(var_tuple)
        
        return hierarchy
    
    def compute_h_statistic(self, var1: int, var2: int,
                           X: np.ndarray) -> float:
        """
        Compute Friedman's H-statistic for interaction strength
        
        H measures the proportion of variance in the joint effect
        that cannot be explained by the sum of main effects.
        
        Parameters
        ----------
        var1, var2 : int
            Variable indices
        X : array
            Input data
            
        Returns
        -------
        h_stat : float
            H-statistic (0 = no interaction, 1 = pure interaction)
        """
        n_samples = X.shape[0]
        
        # Get predictions varying both variables
        f_12 = self.model.predict(X)
        
        # Get predictions varying only var1 (fix var2 at median)
        X_1 = X.copy()
        X_1[:, var2] = np.median(X[:, var2])
        f_1 = self.model.predict(X_1)
        
        # Get predictions varying only var2 (fix var1 at median)
        X_2 = X.copy()
        X_2[:, var1] = np.median(X[:, var1])
        f_2 = self.model.predict(X_2)
        
        # Get baseline (both fixed)
        X_0 = X.copy()
        X_0[:, var1] = np.median(X[:, var1])
        X_0[:, var2] = np.median(X[:, var2])
        f_0 = self.model.predict(X_0)
        
        # Compute H-statistic
        interaction_effect = f_12 - f_1 - f_2 + f_0
        total_effect = f_12 - f_0
        
        h_numerator = np.sum(interaction_effect ** 2)
        h_denominator = np.sum(total_effect ** 2)
        
        if h_denominator < 1e-10:
            return 0.0
        
        h_stat = h_numerator / h_denominator
        return np.clip(h_stat, 0.0, 1.0)


def analyze_interactions_full(model, X: np.ndarray, 
                             threshold: float = 0.01) -> Dict:
    """
    Comprehensive interaction analysis
    
    Parameters
    ----------
    model : MARS
        Fitted model
    X : array
        Input data for analysis
    threshold : float
        Minimum strength threshold
        
    Returns
    -------
    analysis : dict
        Complete interaction analysis results
    """
    analyzer = InteractionAnalyzer(model)
    
    # Get all interactions
    interactions = analyzer.get_interaction_strength()
    
    # Filter by threshold
    significant = {k: v for k, v in interactions.items() 
                  if v >= threshold}
    
    # Rank them
    ranked = analyzer.rank_interactions(top_k=20)
    
    # Find additive effects
    additive = analyzer.find_pure_additive_effects()
    
    # Get pairwise matrix
    pairwise = analyzer.get_pairwise_interactions()
    
    # Hierarchy
    hierarchy = analyzer.hierarchical_interaction_map()
    
    analysis = {
        'all_interactions': interactions,
        'significant_interactions': significant,
        'top_interactions': ranked,
        'pure_additive_vars': additive,
        'pairwise_matrix': pairwise,
        'hierarchy': hierarchy,
        'max_degree': max(hierarchy.keys()) if hierarchy else 0,
        'n_interactions': len(significant),
        'n_additive': len(additive)
    }
    
    return analysis


def print_interaction_report(analysis: Dict):
    """
    Print formatted interaction analysis report
    
    Parameters
    ----------
    analysis : dict
        Output from analyze_interactions_full
    """
    print("\n" + "="*70)
    print("MARS Interaction Analysis Report")
    print("="*70)
    
    print(f"\nMaximum interaction degree: {analysis['max_degree']}")
    print(f"Number of significant interactions: {analysis['n_interactions']}")
    print(f"Number of purely additive variables: {analysis['n_additive']}")
    
    if analysis['pure_additive_vars']:
        print(f"\nPurely additive variables: {analysis['pure_additive_vars']}")
    
    print(f"\nTop {min(10, len(analysis['top_interactions']))} Interactions:")
    for i, (vars, strength) in enumerate(analysis['top_interactions'][:10], 1):
        var_str = ', '.join(f'x{v}' for v in vars)
        degree = len(vars)
        print(f"  {i}. ({var_str}) - degree={degree}, strength={strength:.4f}")
    
    print(f"\nInteraction Hierarchy:")
    for degree in sorted(analysis['hierarchy'].keys()):
        interactions = analysis['hierarchy'][degree]
        print(f"  Degree {degree}: {len(interactions)} unique combinations")
    
    print("="*70 + "\n")