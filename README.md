# PyMARS: Multivariate Adaptive Regression Splines

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A pure Python implementation of **Multivariate Adaptive Regression Splines (MARS)** based on Jerome Friedman's 1991 paper.

## What is MARS?

MARS is a nonparametric regression technique that:

- ✅ **Automatically selects relevant variables** from high-dimensional data
- ✅ **Detects interactions** between variables  
- ✅ **Produces continuous models** with interpretable basis functions
- ✅ **Works well with 3-20 variables** and moderate sample sizes (50-1000)
- ✅ **Provides additive decomposition** (ANOVA) for interpretation

MARS combines the flexibility of recursive partitioning (decision trees) with the smoothness of spline fitting.

## Installation

### From source

```bash
git clone https://github.com/abder111/pymars.git
cd pymars
pip install -e .
```

### For development

```bash
pip install -e ".[dev]"
```

### With plotting support

```bash
pip install -e ".[plot]"
```

## Quick Start

```python
from pymars import MARS
import numpy as np

# Generate sample data
X = np.random.randn(200, 5)
y = X[:, 0]**2 + 2*X[:, 1]*X[:, 2] + np.random.randn(200)*0.1

# Fit MARS model
model = MARS(max_terms=20, max_degree=2)
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# View model summary
model.summary()

# Get feature importances
print("Feature importances:", model.feature_importances_)
```

## Key Features

### 1. Automatic Variable Selection

MARS automatically identifies which variables are important:

```python
# Only first 3 variables matter
X = np.random.randn(100, 10)
y = X[:, 0] + 2*X[:, 1] - X[:, 2] + noise

model = MARS(max_terms=20, max_degree=1)
model.fit(X, y)

# MARS will focus on variables 0, 1, 2
print(model.feature_importances_)
# Output: [0.35, 0.45, 0.20, 0.0, 0.0, 0.0, ...]
```

### 2. Interaction Detection

MARS can find two-way (or higher) interactions:

```python
# y depends on interaction between x0 and x1
X = np.random.randn(100, 3)
y = X[:, 0] * X[:, 1] + noise

model = MARS(max_terms=20, max_degree=2)
model.fit(X, y)

# View ANOVA decomposition
anova = model.get_anova_decomposition()
print(f"Main effects: {len(anova.get(1, []))}")
print(f"2-way interactions: {len(anova.get(2, []))}")
```

### 3. Additive Modeling

Force a purely additive model (no interactions):

```python
model = MARS(max_terms=20, max_degree=1)  # max_degree=1 → additive only
model.fit(X, y)
```

### 4. Model Interpretability

```python
model.summary()
```

Output:
```
======================================================================
MARS Model Summary
======================================================================
Number of basis functions: 8
Number of features: 5
Maximum degree: 2
GCV score: 0.012456
Training MSE: 0.010234
Training R²: 0.954321

Feature Importances:
  x0: 0.4521
  x1: 0.3210
  x2: 0.2105
  x3: 0.0164
  x4: 0.0000

Basis Functions:
  [0] coef=  2.1234  B_0 (constant)
  [1] coef=  1.5432  B(h(x0, +, 0.523))
  [2] coef= -0.8765  B(h(x1, -, 1.234))
  [3] coef=  2.3456  B(h(x0, +, 0.523) * h(x1, +, 0.789))
  ...
======================================================================
```

## API Reference

### MARS Class

```python
MARS(
    max_terms=30,        # Maximum basis functions in forward pass
    max_degree=1,        # Maximum interaction order (1=additive, 2=pairwise, etc.)
    penalty=3.0,         # GCV penalty (2 for additive, 3 for general)
    minspan='auto',      # Min observations between knots
    endspan='auto',      # Min observations from endpoints
    alpha=0.05,          # Significance for span calculation
    standardize=True,  # Standardize features (recommended)
    smooth=False    # Cubic splines
    verbose=True         # Print progress
)
```

### Methods

- **`fit(X, y)`**: Fit model to training data
- **`predict(X)`**: Make predictions
- **`score(X, y)`**: Calculate R² score
- **`summary()`**: Print detailed model summary
- **`get_anova_decomposition()`**: Get functions grouped by interaction order

### Attributes (after fitting)

- **`basis_functions_`**: List of selected basis functions
- **`coefficients_`**: Fitted coefficients
- **`gcv_score_`**: Final GCV score
- **`feature_importances_`**: Importance scores for each feature
- **`n_features_in_`**: Number of input features

## Examples

### Example 1: Simple Regression

```python
import numpy as np
from pymars import MARS

# Generate data
np.random.seed(42)
X = np.random.uniform(-3, 3, (200, 1))
y = np.sin(X).ravel() + np.random.randn(200) * 0.1

# Fit
model = MARS(max_terms=15)
model.fit(X, y)

# Evaluate
print(f"R² score: {model.score(X, y):.4f}")
```

### Example 2: Friedman's Test Function

```python
# Friedman (1991) test function
def friedman_function(X):
    return (10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 
            20 * (X[:, 2] - 0.5)**2 + 
            10 * X[:, 3] + 
            5 * X[:, 4])

X = np.random.uniform(0, 1, (100, 10))  # 10 features, only 5 matter
y = friedman_function(X) + np.random.randn(100)

model = MARS(max_terms=30, max_degree=2)
model.fit(X, y)

# MARS should identify features 0-4 as important
print(model.feature_importances_[:5])  # High values
print(model.feature_importances_[5:])  # Near zero
```

### Example 3: Custom Configuration

```python
model = MARS(
    max_terms=50,        # Allow more basis functions
    max_degree=3,        # Allow 3-way interactions
    penalty=2.5,         # Less aggressive pruning
    minspan=5,           # At least 5 obs between knots
    endspan=5,           # At least 5 obs from endpoints
    standardize=True,    # Standardize for numerical stability
    verbose=True
)
model.fit(X, y)
```

## Algorithm Overview

MARS works in two phases:

### Phase 1: Forward Pass
1. Start with constant model
2. For each iteration:
   - Try adding pairs of basis functions (left/right hinges)
   - Test all variables and all possible knot locations
   - Select pair that best reduces RSS
3. Continue until `max_terms` reached (overfit)

### Phase 2: Backward Pass
1. Remove basis functions one at a time
2. For each removal:
   - Refit model
   - Calculate GCV score
3. Keep removal that minimizes GCV
4. Stop when GCV stops improving

## Performance

Typical execution times on moderate hardware:

| Samples | Features | Max Terms | Time     |
|---------|----------|-----------|----------|
| 100     | 5        | 20        | ~1s      |
| 200     | 10       | 30        | ~5s      |
| 500     | 20       | 50        | ~30s     |
| 1000    | 20       | 50        | ~2min    |

## When to Use MARS

**MARS excels when:**
- You have 3-20 predictor variables
- Sample size is 50-1000
- Functional form is unknown
- Interactions may exist
- Interpretability matters

**Consider alternatives when:**
- Very small samples (< 50) → Use linear regression
- Very high dimensions (> 50) → Use random forests/gradient boosting
- Purely predictive accuracy matters → Use XGBoost/neural networks
- Time series with strong autocorrelation → Use specialized methods

## Testing

Run tests with pytest:

```bash
pytest tests/ -v
```

With coverage:

```bash
pytest tests/ --cov=pymars --cov-report=html
```

## References

1. **Friedman, J. H. (1991)**. "Multivariate Adaptive Regression Splines." *The Annals of Statistics*, 19(1), 1-67.
   
2. **Friedman, J. H., & Silverman, B. W. (1989)**. "Flexible Parsimonious Smoothing and Additive Modeling." *Technometrics*, 31(1), 3-21.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use PyMARS in academic work, please cite Friedman's original paper:

```bibtex
@article{friedman1991mars,
  title={Multivariate adaptive regression splines},
  author={Friedman, Jerome H},
  journal={The annals of statistics},
  pages={1--67},
  year={1991},
  publisher={JSTOR}
}
```

## Contact

- GitHub Issues: https://github.com/abder111/pymars/issues
- GitHub Repository: https://github.com/abder111/pymars
- Documentation: https://pymars.readthedocs.io