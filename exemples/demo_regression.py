"""
MARS Demonstration Examples
===========================

Examples showing various use cases for PyMARS.
"""

import numpy as np
import sys
sys.path.insert(0, '..')

from pymars import MARS


def example_simple_univariate():
    """Example 1: Simple univariate regression"""
    print("\n" + "="*70)
    print("Example 1: Univariate Nonlinear Function")
    print("="*70)
    
    # Generate data: y = sin(x) + noise
    np.random.seed(42)
    X = np.random.uniform(-3, 3, (200, 1))
    y = np.sin(X).ravel() + np.random.randn(200) * 0.1
    
    # Fit MARS
    model = MARS(max_terms=15, max_degree=1, verbose=True)
    model.fit(X, y)
    
    # Evaluate
    y_pred = model.predict(X)
    r2 = model.score(X, y)
    
    print(f"\nFinal R² score: {r2:.4f}")
    print(f"Number of basis functions: {len(model.basis_functions_)}")


def example_friedman():
    """Example 2: Friedman's test function"""
    print("\n" + "="*70)
    print("Example 2: Friedman Test Function (1991)")
    print("="*70)
    
    def friedman_function(X):
        """
        Friedman's test function from the 1991 paper:
        y = 10*sin(π*x1*x2) + 20*(x3-0.5)² + 10*x4 + 5*x5
        """
        return (10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 
                20 * (X[:, 2] - 0.5)**2 + 
                10 * X[:, 3] + 
                5 * X[:, 4])
    
    # Generate data
    np.random.seed(42)
    n_samples = 200
    n_features = 10  # Only first 5 matter
    
    X = np.random.uniform(0, 1, (n_samples, n_features))
    y_true = friedman_function(X)
    noise = np.random.randn(n_samples) * 1.0
    y = y_true + noise
    
    print(f"\nData: {n_samples} samples, {n_features} features")
    print(f"True function depends on first 5 features only")
    print(f"Signal-to-noise ratio: {np.std(y_true)/np.std(noise):.2f}")
    
    # Fit MARS with interactions
    model = MARS(max_terms=30, max_degree=2, verbose=True)
    model.fit(X, y)
    
    # Get summary
    summary = model.summary()
    
    # Check which features were identified
    print("\nFeature Selection Results:")
    for i in range(n_features):
        imp = model.feature_importances_[i]
        status = "✓ IMPORTANT" if imp > 0.01 else "✗ ignored"
        print(f"  Feature x{i}: importance={imp:.4f}  {status}")


def example_additive_vs_interactions():
    """Example 3: Comparing additive and interaction models"""
    print("\n" + "="*70)
    print("Example 3: Additive vs. Interaction Models")
    print("="*70)
    
    # Generate data with interaction
    np.random.seed(42)
    X = np.random.randn(200, 3)
    y = X[:, 0] + 2*X[:, 1] + 3*X[:, 0]*X[:, 1] + np.random.randn(200)*0.5
    
    print("\nTrue model: y = x0 + 2*x1 + 3*x0*x1 + noise")
    print(f"Data: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Fit additive model
    print("\n--- Additive Model (max_degree=1) ---")
    model_add = MARS(max_terms=20, max_degree=1, verbose=False)
    model_add.fit(X, y)
    r2_add = model_add.score(X, y)
    print(f"R² score: {r2_add:.4f}")
    print(f"Basis functions: {len(model_add.basis_functions_)}")
    
    # Fit interaction model
    print("\n--- Interaction Model (max_degree=2) ---")
    model_int = MARS(max_terms=20, max_degree=2, verbose=False)
    model_int.fit(X, y)
    r2_int = model_int.score(X, y)
    print(f"R² score: {r2_int:.4f}")
    print(f"Basis functions: {len(model_int.basis_functions_)}")
    
    # Compare
    print("\n--- Comparison ---")
    print(f"R² improvement: {r2_int - r2_add:.4f}")
    print(f"Interaction model captures the x0*x1 interaction!")
    
    # Show ANOVA decomposition
    anova = model_int.get_anova_decomposition()
    print(f"\nANOVA decomposition:")
    print(f"  Constant: {len(anova.get(0, []))} terms")
    print(f"  Main effects (degree=1): {len(anova.get(1, []))} terms")
    print(f"  Interactions (degree=2): {len(anova.get(2, []))} terms")


def example_variable_selection():
    """Example 4: Automatic variable selection"""
    print("\n" + "="*70)
    print("Example 4: Automatic Variable Selection")
    print("="*70)
    
    # Generate data where only 3 of 15 variables matter
    np.random.seed(42)
    n_samples = 150
    n_features = 15
    n_informative = 3
    
    X = np.random.randn(n_samples, n_features)
    
    # Only first 3 features affect y
    y = (2*X[:, 0] + 3*X[:, 1]**2 - 1.5*X[:, 2] + 
         np.random.randn(n_samples)*0.2)
    
    print(f"\nData: {n_samples} samples, {n_features} features")
    print(f"True model uses only features 0, 1, 2")
    print(f"Remaining {n_features - n_informative} features are noise")
    
    # Fit MARS
    model = MARS(max_terms=25, max_degree=1, verbose=False)
    model.fit(X, y)
    
    # Analyze results
    print("\n--- Variable Selection Results ---")
    r2 = model.score(X, y)
    print(f"R² score: {r2:.4f}")
    
    imp = model.feature_importances_
    important_vars = np.where(imp > 0.01)[0]
    
    print(f"\nFeatures identified as important (importance > 0.01):")
    for var in important_vars:
        print(f"  x{var}: {imp[var]:.4f}")
    
    print(f"\nTotal features selected: {len(important_vars)}/{n_features}")
    print(f"Correctly identified: {np.sum(important_vars < n_informative)}/{n_informative}")


def example_small_sample():
    """Example 5: Small sample size"""
    print("\n" + "="*70)
    print("Example 5: Small Sample Size")
    print("="*70)
    
    # Small dataset
    np.random.seed(42)
    n_samples = 30
    X = np.random.randn(n_samples, 3)
    y = X[:, 0]**2 + X[:, 1] + np.random.randn(n_samples)*0.2
    
    print(f"\nData: {n_samples} samples (very small!)")
    
    # Fit with appropriate parameters for small sample
    model = MARS(
        max_terms=10,      # Fewer terms for small sample
        max_degree=1,      # Avoid interactions to reduce overfitting
        penalty=4.0,       # Higher penalty for small sample
        verbose=False
    )
    model.fit(X, y)
    
    r2 = model.score(X, y)
    print(f"R² score: {r2:.4f}")
    print(f"Basis functions: {len(model.basis_functions_)}")
    print(f"GCV score: {model.gcv_score_:.6f}")
    
    print("\nNote: With small samples, simpler models often work best!")


def example_perfect_linear():
    """Example 6: Perfect linear relationship"""
    print("\n" + "="*70)
    print("Example 6: Perfect Linear Relationship")
    print("="*70)
    
    # Generate perfectly linear data
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = 2*X[:, 0] + 3*X[:, 1] - X[:, 2] + 5.0  # No noise
    
    print("\nTrue model: y = 2*x0 + 3*x1 - x2 + 5 (no noise)")
    
    # Fit MARS
    model = MARS(max_terms=15, max_degree=1, verbose=False)
    model.fit(X, y)
    
    # Check accuracy
    y_pred = model.predict(X)
    mse = np.mean((y - y_pred)**2)
    r2 = model.score(X, y)
    
    print(f"\nMSE: {mse:.2e} (should be near zero)")
    print(f"R²: {r2:.10f} (should be 1.0)")
    print(f"Basis functions: {len(model.basis_functions_)}")
    
    print("\nMARS correctly identifies and fits linear relationships!")


def main():
    """Run all examples"""
    examples = [
        ("Simple Univariate", example_simple_univariate),
        ("Friedman Function", example_friedman),
        ("Additive vs Interactions", example_additive_vs_interactions),
        ("Variable Selection", example_variable_selection),
        ("Small Sample", example_small_sample),
        ("Perfect Linear", example_perfect_linear),
    ]
    
    print("\n" + "="*70)
    print("PyMARS Demonstration Examples")
    print("="*70)
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    print(f"  0. Run all examples")
    
    try:
        choice = input("\nSelect example (0-6): ").strip()
        choice = int(choice)
        
        if choice == 0:
            for name, func in examples:
                func()
        elif 1 <= choice <= len(examples):
            examples[choice-1][1]()
        else:
            print("Invalid choice")
    except (ValueError, KeyboardInterrupt):
        print("\nExiting...")
    
    print("\n" + "="*70)
    print("Examples complete!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()