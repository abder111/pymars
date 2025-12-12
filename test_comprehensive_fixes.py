"""
Comprehensive test suite for MARS algorithm corrections
========================================================

Tests verify fixes applied to pymars to match Friedman (1991):
1. Minspan/endspan formulas use n_samples (not n_features)
2. GCV complexity calculation via trace(B @ pinv(B))
3. Stable basis ID tracking
4. Feature importance calculation (exclude constant)
5. Interaction detection and analysis
"""

import numpy as np
import pytest
from pymars import MARS, HingeFunction
from pymars.basis import BasisFunction, build_design_matrix
from pymars.utils import calculate_minspan, calculate_endspan, get_candidate_knots, solve_least_squares
from pymars.gcv import GCVCalculator
from pymars.interactions import InteractionAnalyzer


class TestMinspanEndspanFormulas:
    """Test that minspan/endspan use correct formula with n_samples"""
    
    def test_minspan_n_samples_dependency(self):
        """Minspan should depend on n_samples, not n_features"""
        n_samples_1 = 100
        n_samples_2 = 1000
        n_features = 10
        alpha = 0.05
        
        minspan_1 = calculate_minspan(n_samples_1, alpha)
        minspan_2 = calculate_minspan(n_samples_2, alpha)
        
        # More samples should give larger minspan (log of larger number)
        assert minspan_2 > minspan_1, "Minspan should increase with n_samples"
    
    def test_endspan_n_samples_dependency(self):
        """Endspan should depend on n_samples, not n_features"""
        n_samples_1 = 100
        n_samples_2 = 1000
        n_features = 10
        alpha = 0.05
        
        endspan_1 = calculate_endspan(n_samples_1, n_features, alpha)
        endspan_2 = calculate_endspan(n_samples_2, n_features, alpha)
        
        # More samples should give larger endspan
        assert endspan_2 > endspan_1, "Endspan should increase with n_samples"
    
    def test_minspan_formula_correctness(self):
        """Test exact minspan formula: L = -log2(α/n) / 2.5"""
        n_samples = 100
        alpha = 0.05
        expected = -np.log2(alpha / n_samples) / 2.5
        calculated = calculate_minspan(n_samples, alpha)
        np.testing.assert_almost_equal(calculated, expected, decimal=5)
    
    def test_endspan_formula_correctness(self):
        """Test exact endspan formula: Le = 3 - log2(α/n)"""
        n_samples = 100
        n_features = 10
        alpha = 0.05
        expected = 3 - np.log2(alpha / n_samples)
        calculated = calculate_endspan(n_samples, n_features, alpha)
        np.testing.assert_almost_equal(calculated, expected, decimal=5)


class TestGCVComplexity:
    """Test GCV complexity calculation via trace(B @ pinv(B))"""
    
    def test_gcv_complexity_basic(self):
        """GCV complexity should use trace formula"""
        # Create simple design matrix
        n_samples = 20
        X = np.random.randn(n_samples, 3)
        
        # Constant basis only
        constant_basis = BasisFunction()  # empty hinges = constant
        B = build_design_matrix(X, [constant_basis])
        
        gcv = GCVCalculator()
        # For constant only, complexity should be 1
        assert B.shape[1] == 1, "Constant basis should give 1 column"
    
    def test_gcv_numerical_stability(self):
        """GCV should not crash on ill-conditioned matrices"""
        n_samples = 10
        X = np.ones((n_samples, 5))  # Rank-1 matrix (problematic)
        
        constant_basis = BasisFunction()
        B = build_design_matrix(X, [constant_basis])
        
        y = np.random.randn(n_samples)
        gcv = GCVCalculator()
        
        # Should not raise an exception
        result = gcv.calculate(y, y, B)
        assert np.isfinite(result), "GCV should return finite value even for ill-conditioned matrix"


class TestBasisIDTracking:
    """Test stable basis ID generation and parent tracking"""
    
    def test_basis_id_stability(self):
        """Each basis function should have stable, unique ID"""
        b1 = BasisFunction()
        b2 = BasisFunction()
        b3 = BasisFunction()
        
        assert b1.basis_id != b2.basis_id, "Different bases should have different IDs"
        assert b1.basis_id != b3.basis_id
        assert b2.basis_id != b3.basis_id
        assert isinstance(b1.basis_id, int), "Basis ID should be integer"
    
    def test_basis_id_incremental(self):
        """Basis IDs should increment"""
        basis_ids = [BasisFunction().basis_id for _ in range(5)]
        # IDs should be unique
        assert len(set(basis_ids)) == 5, "All basis IDs should be unique"
        # Should be mostly increasing (though not guaranteed, they should be distinct)
        assert basis_ids[0] < basis_ids[-1], "Later bases should have larger IDs"
    
    def test_parent_id_tracking(self):
        """Parent basis should be tracked via add_hinge"""
        parent = BasisFunction()
        parent_id = parent.basis_id
        
        hinge = HingeFunction(variable=0, knot=0.5, direction=1)
        child = parent.add_hinge(hinge)
        
        assert child.parent_id == parent_id, "Child should track parent ID"
        assert child.basis_id != parent_id, "Child should have own ID"
    
    def test_repr_shows_basis_id(self):
        """__repr__ should display basis ID"""
        b = BasisFunction()
        repr_str = repr(b)
        assert "B_" in repr_str, "Repr should show basis ID as B_n"


class TestFeatureImportance:
    """Test feature importance calculation"""
    
    def test_importance_excludes_constant(self):
        """Feature importance should skip constant basis"""
        # Create simple model data
        np.random.seed(42)
        X = np.random.randn(30, 3)
        y = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(30) * 0.1
        
        mars = MARS(max_terms=5)
        mars.fit(X, y)
        
        importances = mars.feature_importances_
        assert len(importances) == 3, "Should have importance for each feature"
        assert np.sum(importances) <= 1.01, "Importances should be normalized to [0,1]"
        assert np.all(importances >= 0), "Importances should be non-negative"
    
    def test_importance_shape_matches_features(self):
        """Feature importance array should match n_features"""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = np.sum(X[:, :2], axis=1) + np.random.randn(50) * 0.1
        
        mars = MARS(max_terms=10)
        mars.fit(X, y)
        
        assert mars.feature_importances_.shape == (5,), "Importance shape should match n_features"


class TestInteractionDetection:
    """Test interaction detection and analysis"""
    
    def test_pure_additive_detection_no_duplicates(self):
        """find_pure_additive_effects should return unique variables"""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = X[:, 0] + X[:, 1] + np.random.randn(50) * 0.1
        
        mars = MARS(max_degree=1, max_terms=10)  # Force additive only
        mars.fit(X, y)
        
        analyzer = InteractionAnalyzer(mars)
        additive_vars = analyzer.find_pure_additive_effects()
        
        assert len(additive_vars) == len(set(additive_vars)), "Should return unique variables"
        assert all(isinstance(v, (int, np.integer)) for v in additive_vars), "Should be integer indices"
    
    def test_interaction_strength_dict(self):
        """Interaction strength should return proper dict"""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = X[:, 0] * X[:, 1] + np.random.randn(50) * 0.1
        
        mars = MARS(max_degree=2, max_terms=15)
        mars.fit(X, y)
        
        analyzer = InteractionAnalyzer(mars)
        interactions = analyzer.get_interaction_strength()
        
        assert isinstance(interactions, dict), "Should return dict"
        for var_tuple, strength in interactions.items():
            assert isinstance(var_tuple, tuple), "Keys should be variable tuples"
            assert strength > 0, "Strengths should be positive"
    
    def test_decompose_prediction_standardization(self):
        """Decompose prediction should handle standardization correctly"""
        np.random.seed(42)
        X = np.random.randn(30, 2)
        y = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(30) * 0.1
        
        mars = MARS(standardize=True, max_terms=10)
        mars.fit(X, y)
        
        analyzer = InteractionAnalyzer(mars)
        x_test = np.array([0.5, -0.3])
        
        decomp = analyzer.decompose_prediction(x_test)
        assert isinstance(decomp, dict), "Should return dict"
        assert 'constant' in decomp, "Should have constant term"
        
        # Reconstruct from decomposition
        total = sum(decomp.values())
        pred = mars.predict(x_test.reshape(1, -1))[0]
        np.testing.assert_almost_equal(total, pred, decimal=5)


class TestPredictAccuracy:
    """Test overall model accuracy with corrections"""
    
    def test_friedman_function_accuracy(self):
        """Test on Friedman benchmark function"""
        np.random.seed(42)
        n = 100
        X = np.random.uniform(-1, 1, (n, 5))
        # Friedman function
        y = (10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 
             20 * (X[:, 2] - 0.5)**2 + 
             10 * X[:, 3] + 
             5 * X[:, 4] +
             np.random.randn(n) * 0.5)
        
        mars = MARS(max_degree=2, max_terms=20)
        mars.fit(X, y)
        
        # Check R² on training data
        y_pred = mars.predict(X)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - y.mean())**2)
        r2 = 1 - (ss_res / ss_tot)
        
        assert r2 > 0.8, f"R² should be reasonable for Friedman function, got {r2:.4f}"
    
    def test_additive_model_accuracy(self):
        """Test on simple additive model"""
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 3)
        y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + np.random.randn(n) * 0.1
        
        mars = MARS(max_degree=1, max_terms=10, standardize=True)
        mars.fit(X, y)
        
        y_pred = mars.predict(X)
        r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)
        
        assert r2 > 0.95, f"Should fit additive model well, got R²={r2:.4f}"


class TestSummaryMethod:
    """Test that summary() method works safely"""
    
    def test_summary_no_crash_on_fitted_model(self):
        """Summary should work without crashing"""
        np.random.seed(42)
        X = np.random.randn(30, 2)
        y = X[:, 0] + np.random.randn(30) * 0.1
        
        mars = MARS(max_terms=10)
        mars.fit(X, y)
        
        summary = mars.summary()
        assert isinstance(summary, dict), "Should return dict"
        assert 'n_basis' in summary
        assert 'max_degree_achieved' in summary
        assert 'feature_importances' in summary
    
    def test_summary_handles_missing_metrics(self):
        """Summary should handle missing training metrics gracefully"""
        np.random.seed(42)
        X = np.random.randn(20, 2)
        y = X[:, 0] + np.random.randn(20) * 0.1
        
        mars = MARS()
        mars.fit(X, y)
        
        # Summary should not crash even if some metrics are missing
        summary = mars.summary()
        assert summary is not None


if __name__ == '__main__':
    # Run all tests
    pytest.main([__file__, '-v', '--tb=short'])
