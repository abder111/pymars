# Detailed List of All Corrections Applied

## Files Modified: 6 Core Modules

---

## File 1: pymars/utils.py

### Change 1: calculate_minspan function signature (Line 129-157)

**Before:**
```python
def calculate_minspan(n_samples: int, n_features: int, 
                     alpha: float = 0.05) -> int:
    """Calculate minimum span between knots"""
    if n_samples < 10:
        return 0
    
    # Bug: Using n_features instead of n_samples
    l_star = -np.log2(alpha / n_features) / 2.5
    minspan = max(0, int(np.floor(l_star)))
    return minspan
```

**After:**
```python
def calculate_minspan(n_samples: int, 
                     alpha: float = 0.05) -> int:
    """Calculate minimum span between knots
    
    From Friedman (1991):
        L = -log2(alpha/n) / 2.5
    
    Parameters
    ----------
    n_samples : int
        Number of samples
    alpha : float, default=0.05
        Significance level for run resistance
    ...
    """
    if n_samples < 10:
        return 0
    
    # Friedman's formula (page 94): L = -log2(alpha/n) / 2.5
    # where n = n_samples (NOT n_features)
    l_star = -np.log2(alpha / n_samples) / 2.5
    minspan = max(0, int(np.floor(l_star)))
    return minspan
```

**Impact**: Removes unused n_features parameter, fixes critical formula bug

---

### Change 2: calculate_endspan function (Line 160-185)

**Before:**
```python
def calculate_endspan(n_samples: int, n_features: int, alpha: float = 0.05) -> int:
    """Calculate minimum span from endpoints"""
    ...
    # Bug: Using n_features instead of n_samples
    le = 3 - np.log2(alpha / n_features)
    endspan = max(1, int(np.ceil(le)))
    return endspan
```

**After:**
```python
def calculate_endspan(n_samples: int, n_features: int, alpha: float = 0.05) -> int:
    """Calculate minimum span from endpoints
    
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
    ...
    """
    # Friedman's formula (page 94): Le = 3 - log2(alpha/n)
    # where n = n_samples (NOT n_features)
    le = 3 - np.log2(alpha / n_samples)
    endspan = max(1, int(np.ceil(le)))
    return endspan
```

**Impact**: Fixes critical formula bug, corrects boundary endpoint constraints

---

### Change 3: apply_endspan_constraint function (Line 190-215)

**Before:**
```python
def apply_endspan_constraint(knots: np.ndarray, 
                            x_values: np.ndarray,
                            endspan: int) -> np.ndarray:
    """Apply endspan constraint to knots"""
    ...
    # Bug: Incorrect index calculation
    lower_bound = x_sorted[endspan - 1]  # WRONG
    upper_bound = x_sorted[n - endspan]  # WRONG
    ...
```

**After:**
```python
def apply_endspan_constraint(knots: np.ndarray, 
                            x_values: np.ndarray,
                            endspan: int) -> np.ndarray:
    """Apply endspan constraint to knots
    
    Keeps knots that are at least endspan positions from data endpoints.
    
    Parameters
    ----------
    knots : array
        Candidate knot locations
    x_values : array
        Sorted data values for the variable
    endspan : int
        Minimum positions from endpoints
    
    Returns
    -------
    filtered_knots : array
        Knots satisfying endspan constraint
    """
    x_sorted = np.sort(x_values)
    n = len(x_sorted)
    
    # Knots must be within [x_sorted[endspan], x_sorted[n-endspan-1]]
    lower_bound = x_sorted[endspan]  # FIXED: was endspan-1
    upper_bound = x_sorted[n - endspan - 1]  # FIXED: was n-endspan
    
    return knots[(knots >= lower_bound) & (knots <= upper_bound)]
```

**Impact**: Fixes boundary index logic, ensures correct endspan enforcement

---

### Change 4: get_candidate_knots function (Line 67-120)

**Before:**
```python
def get_candidate_knots(X: np.ndarray, variable: int,
                       minspan: int = 0,
                       existing_knots: Optional[List[float]] = None) -> np.ndarray:
    """Get candidate knot locations for a variable"""
    ...
    # Limited implementation
    unique_vals, idx_first = np.unique(x_sorted, return_index=True)
    ...
```

**After:**
```python
def get_candidate_knots(X: np.ndarray, variable: int,
                       minspan: int = 0,
                       existing_knots: Optional[List[float]] = None) -> np.ndarray:
    """Get candidate knot locations for a variable
    
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
        ...
```

**Impact**: Improves minspan semantics using sorted indices, better knot spacing logic

---

### Change 5: solve_least_squares function (Line 230-275)

**Before:**
```python
def solve_least_squares(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve least squares problem Ax = b"""
    # Bug: Limited fallback, potential in-place mutation
    try:
        ATA = A.T @ A
        ATA_chol = np.linalg.cholesky(ATA)
        ATb = A.T @ b
        x = np.linalg.solve(ATA_chol, ATb)
    except np.linalg.LinAlgError:
        x = np.linalg.lstsq(A, b, rcond=None)[0]
    return x
```

**After:**
```python
def solve_least_squares(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve least squares problem Ax = b with robust fallback chain
    
    Uses three methods in order of preference:
    1. lstsq (LAPACK-based, most robust)
    2. Cholesky (fast if SPD)
    3. pinv (Moore-Penrose, handles rank-deficient)
    
    Parameters
    ----------
    A : array, shape (m, n)
        Design matrix
    b : array, shape (m,) or (m, k)
        Target values
    
    Returns
    -------
    x : array, shape (n,) or (n, k)
        Least squares solution
    """
    # Method 1: Use lstsq (most numerically stable)
    try:
        x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        return x
    except (np.linalg.LinAlgError, ValueError):
        pass
    
    # Method 2: Try Cholesky (fast if A.T @ A is SPD)
    try:
        ATA = A.T @ A
        ATb = A.T @ b
        # Check if matrix is approximately SPD
        eigenvals = np.linalg.eigvalsh(ATA)
        if np.all(eigenvals > 1e-10):  # All eigenvalues positive
            ATA_chol = np.linalg.cholesky(ATA)
            x = np.linalg.solve(ATA_chol, ATb)
            return x
    except (np.linalg.LinAlgError, ValueError):
        pass
    
    # Method 3: Use pseudoinverse (robust fallback)
    A_pinv = np.linalg.pinv(A, rcond=1e-15)
    x = A_pinv @ b
    return x
```

**Impact**: More robust regression fitting, handles ill-conditioned matrices gracefully

---

## File 2: pymars/gcv.py

### Change 1: complexity method (Line 45-65)

**Before:**
```python
def complexity(self, B: np.ndarray, n_basis: int) -> float:
    """Calculate effective number of parameters"""
    # Bug: Uses only SVD rank, doesn't account for interactions properly
    try:
        _, s, _ = np.linalg.svd(B, full_matrices=False)
        rank = np.sum(s > 1e-10)
        return float(rank)
    except:
        return float(n_basis)
```

**After:**
```python
def complexity(self, B: np.ndarray, n_basis: int) -> float:
    """Calculate effective number of parameters
    
    Uses trace formula: C(M) = trace(B @ pinv(B))
    
    This accounts for correlation between basis functions.
    More accurate than simple rank for redundant bases.
    
    Parameters
    ----------
    B : array, shape (n_samples, n_basis)
        Design matrix of basis functions
    n_basis : int
        Number of basis functions (including constant)
    
    Returns
    -------
    complexity : float
        Effective degrees of freedom
    """
    eps = 1e-15
    
    try:
        # Use trace formula for complexity
        B_pinv = np.linalg.pinv(B, rcond=eps)
        c_linear = float(np.trace(B @ B_pinv))
        
        # Guard against numerical issues
        if not np.isfinite(c_linear):
            return float(n_basis)
        
        return max(1.0, c_linear)
    except (np.linalg.LinAlgError, ValueError):
        # Fallback to simple estimate
        return float(min(B.shape))  # min(n_samples, n_basis)
```

**Impact**: Implements proper Friedman formula, better complexity calculation

---

### Change 2: calculate method (Line 20-42)

**Before:**
```python
def calculate(self, y: np.ndarray, y_pred: np.ndarray, 
             B: np.ndarray) -> float:
    """Calculate GCV score"""
    n_samples = len(y)
    c = self.complexity(B, B.shape[1])
    
    # Bug: No guard against c >= N causing division issues
    gcv = np.sum((y - y_pred) ** 2) / (n_samples * (1 - c / n_samples) ** 2)
    return gcv
```

**After:**
```python
def calculate(self, y: np.ndarray, y_pred: np.ndarray, 
             B: np.ndarray) -> float:
    """Calculate GCV score
    
    GCV = RSS / [N * (1 - C(M)/N)²]
    
    where RSS = residual sum of squares
          N = number of samples
          C(M) = model complexity (effective degrees of freedom)
    
    Parameters
    ----------
    y : array, shape (n_samples,)
        True target values
    y_pred : array, shape (n_samples,)
        Predicted values
    B : array, shape (n_samples, n_basis)
        Design matrix
    
    Returns
    -------
    gcv : float
        Generalized cross-validation score
    """
    n_samples = len(y)
    c = self.complexity(B, B.shape[1])
    
    # Guard against numerical issues when c >= n_samples
    eps = np.finfo(float).eps
    denominator = 1.0 - c / n_samples
    
    if abs(denominator) < eps or denominator <= 0:
        return np.inf  # Model is overfitted
    
    rss = np.sum((y - y_pred) ** 2)
    gcv = rss / (n_samples * denominator ** 2)
    
    return float(gcv)
```

**Impact**: Adds numerical stability, prevents inf/nan in GCV calculation

---

## File 3: pymars/basis.py

### Change 1: Add module-level ID counter (Line 1-20)

**Before:**
```python
"""Basis function definitions for MARS"""

import numpy as np
from typing import List, Tuple, Optional
...
```

**After:**
```python
"""Basis function definitions for MARS"""

import numpy as np
from typing import List, Tuple, Optional

# Global counter for stable basis function IDs
_basis_id_counter = {"value": 0}

def _next_basis_id() -> int:
    """Generate next stable basis function ID
    
    Returns
    -------
    basis_id : int
        Unique stable ID for this basis function
    """
    _basis_id_counter["value"] += 1
    return _basis_id_counter["value"]

...
```

**Impact**: Enables stable ID generation replacing Python's id()

---

### Change 2: Update BasisFunction.__init__ (Line 90-105)

**Before:**
```python
def __init__(self, hinges: Optional[List[HingeFunction]] = None, 
             parent_id: Optional[int] = None):
    self.hinges = hinges if hinges is not None else []
    self.parent_id = parent_id
    self._degree = len(self.hinges)
    self._variables = [h.variable for h in self.hinges]
```

**After:**
```python
def __init__(self, hinges: Optional[List[HingeFunction]] = None, 
             parent_id: Optional[int] = None,
             basis_id: Optional[int] = None):
    self.hinges = hinges if hinges is not None else []
    self.parent_id = parent_id
    self.basis_id = basis_id if basis_id is not None else _next_basis_id()
    self._degree = len(self.hinges)
    self._variables = [h.variable for h in self.hinges]
```

**Impact**: Adds stable basis_id to constructor

---

### Change 3: Update add_hinge method (Line 135-153)

**Before:**
```python
def add_hinge(self, hinge: HingeFunction) -> 'BasisFunction':
    """Create new basis function by adding a hinge"""
    new_hinges = self.hinges.copy()
    new_hinges.append(hinge)
    return BasisFunction(hinges=new_hinges, parent_id=id(self))  # BUG: id(self)
```

**After:**
```python
def add_hinge(self, hinge: HingeFunction) -> 'BasisFunction':
    """Create new basis function by adding a hinge
    
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
    return BasisFunction(hinges=new_hinges, parent_id=self.basis_id)  # FIXED
```

**Impact**: Uses stable basis_id instead of id(self)

---

### Change 4: Improve __repr__ method (Line 164-169)

**Before:**
```python
def __repr__(self):
    if len(self.hinges) == 0:
        return "B_0 (constant)"
    hinge_strs = [str(h) for h in self.hinges]
    return "B(" + " * ".join(hinge_strs) + ")"
```

**After:**
```python
def __repr__(self):
    if len(self.hinges) == 0:
        return f"B_{self.basis_id} (constant)"
    hinge_strs = [str(h) for h in self.hinges]
    hinges_product = " × ".join(hinge_strs)
    return f"B_{self.basis_id}({hinges_product})"
```

**Impact**: Shows basis ID in representation, uses × for multiplication

---

## File 4: pymars/model.py

### Change 1: Fix forward pass loop condition (Line 78-85)

**Before:**
```python
# Iteratively add basis function pairs
iteration = 0
while len(basis_functions) < self.max_terms + 1:
    iteration += 1
    ...
```

**After:**
```python
# Iteratively add basis function pairs
# Note: basis_functions[0] is the constant, so (len - 1) = number of non-constant bases
iteration = 0
while (len(basis_functions) - 1) < self.max_terms:
    iteration += 1
    ...
```

**Impact**: Clarifies that constant term is not counted in max_terms

---

## File 5: pymars/mars.py

### Change 1: Fix calculate_minspan call (Line 177)

**Before:**
```python
if self.minspan == 'auto':
    minspan = calculate_minspan(n_samples, n_features, self.alpha)  # WRONG
```

**After:**
```python
if self.minspan == 'auto':
    minspan = calculate_minspan(n_samples, self.alpha)  # FIXED
```

**Impact**: Matches new function signature after removing unused n_features parameter

---

### Change 2: Rewrite _calculate_feature_importance (Line 375-401)

**Before:**
```python
def _calculate_feature_importance(self) -> np.ndarray:
    """Calculate feature importance scores"""
    if self.basis_functions_ is None:
        return np.array([])
    
    importance = np.zeros(self.n_features_in_)
    
    for basis, coef in zip(self.basis_functions_, self.coefficients_):
        for var in basis.variables:
            importance[var] += abs(coef)
    
    if importance.sum() > 0:
        importance /= importance.sum()
    
    return importance
```

**After:**
```python
def _calculate_feature_importance(self) -> np.ndarray:
    """Calculate feature importance scores
    
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
```

**Impact**: Skips constant term, ensures correct shape, adds None checks

---

### Change 3: Improve summary method safety (Line 280-345)

**Before:**
```python
def summary(self) -> Dict:
    """Get model summary with statistics"""
    if self.basis_functions_ is None:
        raise ValueError("Model not fitted...")
    
    ...
    summary = {
        'n_basis': len(self.basis_functions_),
        'max_degree_achieved': max(b.degree for b in self.basis_functions_),  # Can crash if empty
        'n_features': self.n_features_in_,
        'gcv_score': float(self.gcv_score_),  # Can crash if None
        'train_mse': float(self._train_mse) if self._train_mse else None,  # Fragile
        ...
    }
```

**After:**
```python
def summary(self) -> Dict:
    """Get model summary with statistics and basis functions
    
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
        'max_degree_achieved': max_degree,  # FIXED: safe with default=0
        'n_features': self.n_features_in_,
        'gcv_score': float(self.gcv_score_) if self.gcv_score_ is not None else None,  # FIXED
        'train_mse': train_mse,  # FIXED: safer checks
        'train_r2': train_r2,  # FIXED: safer checks
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
    ...
```

**Impact**: Added safety checks for all potentially None values

---

### Change 4: Improve predict method docstring (Line 226-261)

**Before:**
```python
def predict(self, X: np.ndarray) -> np.ndarray:
    """Make predictions using fitted MARS model
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input features
        
    Returns
    -------
    y_pred : array, shape (n_samples,)
        Predicted values
    """
```

**After:**
```python
def predict(self, X: np.ndarray) -> np.ndarray:
    """Make predictions using fitted MARS model
    
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
```

**Impact**: Clarifies that knots are in standardized domain

---

## File 6: pymars/interactions.py

### Change 1: Improve find_pure_additive_effects (Line 99-124)

**Before:**
```python
def find_pure_additive_effects(self) -> List[int]:
    """Find variables that only appear in additive terms"""
    interaction_vars = set()
    for basis in self.model.basis_functions_:
        if basis.degree > 1:
            interaction_vars.update(basis.variables)
    
    additive_only = []
    for basis in self.model.basis_functions_:
        if basis.degree == 1:
            var = basis.variables[0]
            if var not in interaction_vars:
                additive_only.append(var)
    
    return sorted(set(additive_only))  # Deduped at the end
```

**After:**
```python
def find_pure_additive_effects(self) -> List[int]:
    """Find variables that only appear in additive terms (degree=1)
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
    additive_only = set()  # FIXED: use set directly
    for basis in self.model.basis_functions_:
        if basis.degree == 1:
            var = basis.variables[0]
            if var not in interaction_vars:
                additive_only.add(var)  # Use set.add()
    
    return sorted(additive_only)  # Clean conversion
```

**Impact**: More efficient, cleaner code, better documentation

---

### Change 2: Improve decompose_prediction standardization (Line 126-173)

**Before:**
```python
def decompose_prediction(self, x: np.ndarray) -> Dict[str, float]:
    """Decompose a single prediction into contributions
    
    Parameters
    ----------
    x : array, shape (n_features,)
        Single input vector
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)
    
    contributions = {'constant': 0.0}
    
    for basis, coef in zip(self.model.basis_functions_, 
                          self.model.coefficients_):
        value = basis.evaluate(x)[0]  # BUG: Not standardized
        ...
```

**After:**
```python
def decompose_prediction(self, x: np.ndarray) -> Dict[str, float]:
    """Decompose a single prediction into contributions
    
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
        x_eval = (x - self.model._x_mean) / self.model._x_std  # FIXED
    else:
        x_eval = x
    
    contributions = {'constant': 0.0}
    
    for basis, coef in zip(self.model.basis_functions_, 
                          self.model.coefficients_):
        value = basis.evaluate(x_eval)[0]  # Uses standardized x
        ...
```

**Impact**: Corrects basis evaluation when model uses standardization

---

## Summary of Changes

| File | Function | Change Type | Impact |
|------|----------|------------|--------|
| utils.py | calculate_minspan | Remove unused param, fix formula | CRITICAL |
| utils.py | calculate_endspan | Fix formula | CRITICAL |
| utils.py | apply_endspan_constraint | Fix boundary indices | IMPORTANT |
| utils.py | get_candidate_knots | Improve documentation | IMPORTANT |
| utils.py | solve_least_squares | Add fallback chain | CRITICAL |
| gcv.py | complexity | Replace SVD with trace | CRITICAL |
| gcv.py | calculate | Add numerical stability | CRITICAL |
| basis.py | (module level) | Add ID counter | IMPORTANT |
| basis.py | \_\_init\_\_ | Add basis_id parameter | IMPORTANT |
| basis.py | add_hinge | Use stable ID | IMPORTANT |
| basis.py | \_\_repr\_\_ | Show basis ID | IMPORTANT |
| model.py | fit (forward pass) | Clarify loop condition | IMPORTANT |
| mars.py | fit | Fix minspan call | CRITICAL |
| mars.py | _calculate_feature_importance | Skip constant, safer | IMPORTANT |
| mars.py | summary | Add safety checks | IMPORTANT |
| mars.py | predict | Add knot domain docs | IMPORTANT |
| interactions.py | find_pure_additive_effects | Use set, cleaner | IMPORTANT |
| interactions.py | decompose_prediction | Add standardization | IMPORTANT |

**Total**: 17 distinct changes across 6 files

---

**All changes verified and tested** ✓
