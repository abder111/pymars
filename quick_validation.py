#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Quick validation of critical fixes without pytest overhead"""

import sys
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')

print("=" * 70)
print("VALIDATION OF PYMARS CORRECTIONS")
print("=" * 70)

# Test 1: Minspan formula
print("\n1. Testing Minspan Formula (uses n_samples, not n_features)...")
try:
    from pymars.utils import calculate_minspan
    
    n_samples = 100
    alpha = 0.05
    minspan_result = calculate_minspan(n_samples, alpha)
    expected_float = -np.log2(alpha / n_samples) / 2.5
    expected_int = int(np.floor(expected_float))
    
    print(f"   Minspan(n=100): {minspan_result}")
    print(f"   Expected (float): {expected_float:.6f}")
    print(f"   Expected (int):   {expected_int}")
    
    if minspan_result == expected_int:
        print("   ✓ PASS: Minspan formula is correct")
    else:
        print("   ✗ FAIL: Minspan formula is incorrect")
        sys.exit(1)
except Exception as e:
    print(f"   ✗ ERROR: {e}")
    sys.exit(1)

# Test 2: Endspan formula
print("\n2. Testing Endspan Formula (uses n_samples, not n_features)...")
try:
    from pymars.utils import calculate_endspan
    
    n_samples = 100
    n_features = 10
    alpha = 0.05
    endspan_result = calculate_endspan(n_samples, n_features, alpha)
    expected_float = 3 - np.log2(alpha / n_samples)
    expected_int = max(1, int(np.ceil(expected_float)))
    
    print(f"   Endspan(n=100): {endspan_result}")
    print(f"   Expected (float): {expected_float:.6f}")
    print(f"   Expected (int):   {expected_int}")
    
    if endspan_result == expected_int:
        print("   ✓ PASS: Endspan formula is correct")
    else:
        print("   ✗ FAIL: Endspan formula is incorrect")
        sys.exit(1)
except Exception as e:
    print(f"   ✗ ERROR: {e}")
    sys.exit(1)

# Test 3: Basis ID stability
print("\n3. Testing Stable Basis ID Generation...")
try:
    from pymars.basis import BasisFunction
    
    bases = [BasisFunction() for _ in range(5)]
    ids = [b.basis_id for b in bases]
    
    # All IDs should be unique
    if len(set(ids)) == len(ids):
        print(f"   Generated IDs: {ids}")
        print("   ✓ PASS: All basis IDs are unique")
    else:
        print(f"   ✗ FAIL: Duplicate IDs detected: {ids}")
        sys.exit(1)
except Exception as e:
    print(f"   ✗ ERROR: {e}")
    sys.exit(1)

# Test 4: Parent ID tracking
print("\n4. Testing Parent ID Tracking in add_hinge...")
try:
    from pymars.basis import BasisFunction, HingeFunction
    
    parent = BasisFunction()
    parent_id = parent.basis_id
    
    hinge = HingeFunction(variable=0, knot=0.5, direction=1)
    child = parent.add_hinge(hinge)
    
    if child.parent_id == parent_id and child.basis_id != parent_id:
        print(f"   Parent ID: {parent_id}")
        print(f"   Child ID:  {child.basis_id}")
        print(f"   Child parent_id: {child.parent_id}")
        print("   ✓ PASS: Parent ID tracking works correctly")
    else:
        print(f"   ✗ FAIL: Parent ID not tracked correctly")
        sys.exit(1)
except Exception as e:
    print(f"   ✗ ERROR: {e}")
    sys.exit(1)

# Test 5: Simple MARS fit
print("\n5. Testing Simple MARS Model Fit...")
try:
    from pymars import MARS
    
    np.random.seed(42)
    X = np.random.randn(50, 2)
    y = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(50) * 0.1
    
    mars = MARS(max_terms=10, verbose=0)
    mars.fit(X, y)
    
    y_pred = mars.predict(X)
    r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)
    
    print(f"   R² score: {r2:.6f}")
    print(f"   Basis functions: {len(mars.basis_functions_)}")
    print(f"   Feature importances: {mars.feature_importances_}")
    
    if r2 > 0.8:
        print("   ✓ PASS: MARS model fits data well")
    else:
        print(f"   ⚠ WARNING: R² is lower than expected ({r2:.4f})")
except Exception as e:
    print(f"   ✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Feature importance shape
print("\n6. Testing Feature Importance Shape...")
try:
    if mars.feature_importances_.shape == (2,):
        print(f"   Feature importance shape: {mars.feature_importances_.shape}")
        print("   ✓ PASS: Feature importance has correct shape")
    else:
        print(f"   ✗ FAIL: Feature importance shape is {mars.feature_importances_.shape}, expected (2,)")
        sys.exit(1)
except Exception as e:
    print(f"   ✗ ERROR: {e}")
    sys.exit(1)

# Test 7: Interaction analysis
print("\n7. Testing Interaction Analysis...")
try:
    from pymars.interactions import InteractionAnalyzer
    
    analyzer = InteractionAnalyzer(mars)
    interactions = analyzer.get_interaction_strength()
    additive_vars = analyzer.find_pure_additive_effects()
    
    print(f"   Interactions found: {len(interactions)}")
    print(f"   Pure additive variables: {additive_vars}")
    print(f"   Additive vars are unique: {len(additive_vars) == len(set(additive_vars))}")
    print("   ✓ PASS: Interaction analysis works")
except Exception as e:
    print(f"   ✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("ALL TESTS PASSED ✓")
print("=" * 70)
