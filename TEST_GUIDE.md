# Running Tests for PyMARS Corrections

## Quick Validation (Recommended)

For a quick test of all critical corrections:

```bash
python quick_validation.py
```

**Expected Output**: All 7 tests PASS, including:
- Minspan formula (n_samples dependency)
- Endspan formula (n_samples dependency)
- Basis ID uniqueness
- Parent-child tracking
- MARS model fit accuracy (R² > 0.98)
- Feature importance shape
- Interaction analysis

**Time**: ~30 seconds

---

## Comprehensive Test Suite

For full pytest coverage (requires pytest):

```bash
pip install pytest
python -m pytest test_comprehensive_fixes.py -v
```

**Test Classes**:
1. `TestMinspanEndspanFormulas` - Minspan and endspan formula correctness
2. `TestGCVComplexity` - GCV complexity calculation and stability
3. `TestBasisIDTracking` - Stable ID generation and parent tracking
4. `TestFeatureImportance` - Feature importance calculations
5. `TestInteractionDetection` - Interaction and pure additive detection
6. `TestPredictAccuracy` - Model accuracy on benchmark functions
7. `TestSummaryMethod` - Summary method robustness

**Total Tests**: 20+ individual test cases

**Time**: ~2-5 minutes (depending on system)

---

## Manual Validation Checklist

### 1. Formula Correctness
```python
from pymars.utils import calculate_minspan, calculate_endspan
import numpy as np

# Test minspan
n = 100
alpha = 0.05
minspan = calculate_minspan(n, alpha)
expected = int(np.floor(-np.log2(alpha/n) / 2.5))
assert minspan == expected, f"Minspan failed: {minspan} vs {expected}"

# Test endspan
endspan = calculate_endspan(n, 10, alpha)
expected = max(1, int(np.ceil(3 - np.log2(alpha/n))))
assert endspan == expected, f"Endspan failed: {endspan} vs {expected}"

print("✓ Formulas correct")
```

### 2. Basis ID Stability
```python
from pymars.basis import BasisFunction, HingeFunction

b1 = BasisFunction()
b2 = BasisFunction()
b3 = BasisFunction()

assert b1.basis_id != b2.basis_id != b3.basis_id
print(f"✓ Basis IDs unique: {b1.basis_id}, {b2.basis_id}, {b3.basis_id}")

# Test parent tracking
hinge = HingeFunction(0, 0.5, 1)
child = b1.add_hinge(hinge)
assert child.parent_id == b1.basis_id
assert child.basis_id != b1.basis_id
print("✓ Parent-child tracking works")
```

### 3. Model Fit
```python
from pymars import MARS
import numpy as np

np.random.seed(42)
X = np.random.randn(50, 2)
y = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(50) * 0.1

mars = MARS(max_terms=10, verbose=0)
mars.fit(X, y)

y_pred = mars.predict(X)
r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)

assert r2 > 0.8, f"R² too low: {r2}"
assert mars.feature_importances_.shape == (2,)
print(f"✓ Model fit: R² = {r2:.6f}, importances = {mars.feature_importances_}")
```

### 4. Interaction Analysis
```python
from pymars.interactions import InteractionAnalyzer

analyzer = InteractionAnalyzer(mars)
interactions = analyzer.get_interaction_strength()
additive_vars = analyzer.find_pure_additive_effects()

assert len(set(additive_vars)) == len(additive_vars), "Duplicates in additive vars"
print(f"✓ Interactions: {len(interactions)} found")
print(f"✓ Additive variables: {additive_vars} (unique)")
```

### 5. GCV Stability
```python
from pymars.gcv import GCVCalculator
from pymars.basis import build_design_matrix
import numpy as np

X = np.random.randn(20, 5)
y = np.random.randn(20)

from pymars.basis import BasisFunction
b = BasisFunction()
B = build_design_matrix(X, [b])

gcv = GCVCalculator()
score = gcv.calculate(y, y, B)

assert np.isfinite(score), "GCV score is not finite"
print(f"✓ GCV stable: score = {score:.6f}")
```

---

## Automated CI/CD Testing

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.8', '3.9', '3.10', '3.11', '3.12']
    
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install numpy scipy matplotlib pytest
    
    - name: Run quick validation
      run: python quick_validation.py
    
    - name: Run comprehensive tests
      run: python -m pytest test_comprehensive_fixes.py -v
```

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'pytest'`
**Solution**: 
```bash
pip install pytest
```

### Issue: `UnicodeEncodeError` on Windows
**Solution**: Already fixed in `quick_validation.py` (uses ASCII symbols)

### Issue: Test timeout
**Solution**: Run `quick_validation.py` instead (faster)

### Issue: Import errors
**Solution**: Ensure you're in the pymars directory:
```bash
cd /path/to/pymars
python quick_validation.py
```

---

## Expected Test Output

### quick_validation.py
```
======================================
VALIDATION OF PYMARS CORRECTIONS
======================================

1. Testing Minspan Formula (uses n_samples, not n_features)...
   Minspan(n=100): 4
   Expected (float): 4.386314
   Expected (int):   4
   [PASS] Minspan formula is correct

2. Testing Endspan Formula (uses n_samples, not n_features)...
   Endspan(n=100): 14
   Expected (float): 13.965784
   Expected (int):   14
   [PASS] Endspan formula is correct

3. Testing Stable Basis ID Generation...
   Generated IDs: [1, 2, 3, 4, 5]
   [PASS] All basis IDs are unique

4. Testing Parent ID Tracking in add_hinge...
   Parent ID: 6
   Child ID:  7
   Child parent_id: 6
   [PASS] Parent ID tracking works correctly

5. Testing Simple MARS Model Fit...
   R² score: 0.989515
   Basis functions: 5
   Feature importances: [0.62745338 0.37254662]
   [PASS] MARS model fits data well

6. Testing Feature Importance Shape...
   Feature importance shape: (2,)
   [PASS] Feature importance has correct shape

7. Testing Interaction Analysis...
   Interactions found: 2
   Pure additive variables: [0, 1]
   Additive vars are unique: True
   [PASS] Interaction analysis works

======================================
ALL TESTS PASSED [OK]
======================================
```

---

## Performance Benchmarks

Expected performance on typical hardware:

| Test | Time | Status |
|------|------|--------|
| quick_validation.py | ~20-30s | ✓ Fast |
| test_comprehensive_fixes.py | ~2-5 min | ✓ Reasonable |
| Single minspan test | <100ms | ✓ Instant |
| MARS fit (50 samples) | ~5-10s | ✓ Good |

---

## Regression Testing

To ensure fixes don't break existing functionality:

```bash
# Run with different random seeds
for seed in 1 2 3 4 5; do
    python -c "
import numpy as np
from pymars import MARS

np.random.seed($seed)
X = np.random.randn(100, 5)
y = X[:, 0] + np.random.randn(100) * 0.1

mars = MARS(verbose=0)
mars.fit(X, y)
print(f'Seed {$seed}: R² = {mars.score(X, y):.4f}')
"
done
```

---

**Last Updated**: December 11, 2025
**Status**: All tests passing ✓
