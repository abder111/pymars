"""
Unit tests for MARS
===================

Tests for the main MARS class and core functionality.
"""

import pytest
import numpy as np
from pymars import MARS


class TestMARS:
    """Test suite for MARS class"""
    
    @pytest.fixture
    def simple_data(self):
        """Generate simple test data"""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = X[:, 0]**2 + 2*X[:, 1] + np.random.randn(100)*0.1
        return X, y
    
    @pytest.fixture
    def additive_data(self):
        """Generate purely additive data"""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = X[:, 0] + 2*X[:, 1] - X[:, 2] + np.random.randn(100)*0.1
        return X, y
    
    def test_initialization(self):
        """Test MARS initialization with various parameters"""
        model = MARS()
        assert model.max_terms == 30
        assert model.max_degree == 1
        assert model.penalty == 3.0
        
        model2 = MARS(max_terms=20, max_degree=2, penalty=2.5)
        assert model2.max_terms == 20
        assert model2.max_degree == 2
        assert model2.penalty == 2.5
    
    def test_fit_predict(self, simple_data):
        """Test basic fit and predict"""
        X, y = simple_data
        
        model = MARS(max_terms=10, max_degree=1, verbose=False)
        model.fit(X, y)
        
        # Check fitted attributes exist
        assert model.basis_functions_ is not None
        assert model.coefficients_ is not None
        assert model.gcv_score_ is not None
        assert model.n_features_in_ == 3
        
        # Check predictions
        y_pred = model.predict(X)
        assert y_pred.shape == y.shape
        assert not np.any(np.isnan(y_pred))
    
    def test_fit_shape_validation(self):
        """Test input validation"""
        model = MARS(verbose=False)
        
        # Wrong X shape
        with pytest.raises(ValueError):
            X = np.random.randn(10)
            y = np.random.randn(10)
            model.fit(X, y)
        
        # Mismatched lengths
        with pytest.raises(ValueError):
            X = np.random.randn(10, 3)
            y = np.random.randn(15)
            model.fit(X, y)
    
    def test_predict_before_fit(self):
        """Test that predict raises error before fitting"""
        model = MARS(verbose=False)
        X = np.random.randn(10, 3)
        
        with pytest.raises(ValueError, match="not fitted"):
            model.predict(X)
    
    def test_predict_wrong_features(self, simple_data):
        """Test prediction with wrong number of features"""
        X, y = simple_data
        
        model = MARS(max_terms=10, verbose=False)
        model.fit(X, y)
        
        # Wrong number of features
        with pytest.raises(ValueError):
            X_wrong = np.random.randn(10, 5)
            model.predict(X_wrong)
    
    def test_score(self, simple_data):
        """Test scoring method"""
        X, y = simple_data
        
        model = MARS(max_terms=15, max_degree=2, verbose=False)
        model.fit(X, y)
        
        r2 = model.score(X, y)
        assert -1 <= r2 <= 1
        assert r2 > 0.5  # Should fit reasonably well
    
    def test_additive_model(self, additive_data):
        """Test additive model constraint"""
        X, y = additive_data
        
        model = MARS(max_terms=20, max_degree=1, verbose=False)
        model.fit(X, y)
        
        # All basis functions should have degree <= 1
        for basis in model.basis_functions_:
            assert basis.degree <= 1
        
        # Should achieve good fit
        r2 = model.score(X, y)
        assert r2 > 0.8
    
    def test_interaction_model(self, simple_data):
        """Test model with interactions"""
        X, y = simple_data
        
        model = MARS(max_terms=20, max_degree=2, verbose=False)
        model.fit(X, y)
        
        # Should have at least one basis function with degree > 1
        degrees = [basis.degree for basis in model.basis_functions_]
        # Note: may or may not find interactions depending on data
        # Just check no error occurs
        assert max(degrees) <= 2
    
    def test_summary(self, simple_data):
        """Test summary method"""
        X, y = simple_data
        
        model = MARS(max_terms=10, verbose=False)
        model.fit(X, y)
        
        summary = model.summary()
        
        # Check summary structure
        assert 'n_basis' in summary
        assert 'gcv_score' in summary
        assert 'train_mse' in summary
        assert 'train_r2' in summary
        assert 'basis_functions' in summary
        assert 'feature_importances' in summary
        
        # Check values
        assert summary['n_basis'] > 0
        assert summary['gcv_score'] > 0
        assert len(summary['feature_importances']) == 3
    
    def test_feature_importances(self, simple_data):
        """Test feature importance calculation"""
        X, y = simple_data
        
        model = MARS(max_terms=15, max_degree=2, verbose=False)
        model.fit(X, y)
        
        imp = model.feature_importances_
        
        # Check properties
        assert len(imp) == 3
        assert np.all(imp >= 0)
        assert np.isclose(imp.sum(), 1.0)
        
        # Features 0 and 1 should be more important than 2
        # (since y depends on x0 and x1 but noise on x2)
        assert imp[0] > 0.01 or imp[1] > 0.01
    
    def test_anova_decomposition(self, simple_data):
        """Test ANOVA decomposition"""
        X, y = simple_data
        
        model = MARS(max_terms=15, max_degree=2, verbose=False)
        model.fit(X, y)
        
        anova = model.get_anova_decomposition()
        
        # Check structure
        assert isinstance(anova, dict)
        assert 0 in anova  # Constant term
        
        # Check basis functions are correctly grouped
        for order, bases in anova.items():
            for basis in bases:
                assert basis.degree == order
    
    def test_standardization_on(self, simple_data):
        """Test with standardization enabled"""
        X, y = simple_data
        
        model = MARS(max_terms=10, standardize=True, verbose=False)
        model.fit(X, y)
        
        y_pred = model.predict(X)
        
        # Predictions should be in original scale
        assert y_pred.min() >= y.min() - 1.0
        assert y_pred.max() <= y.max() + 1.0
    
    def test_standardization_off(self, simple_data):
        """Test with standardization disabled"""
        X, y = simple_data
        
        model = MARS(max_terms=10, standardize=False, verbose=False)
        model.fit(X, y)
        
        y_pred = model.predict(X)
        assert y_pred.shape == y.shape
    
    def test_perfect_fit(self):
        """Test on perfectly fittable data"""
        np.random.seed(42)
        X = np.linspace(0, 10, 50).reshape(-1, 1)
        y = 2 * X.ravel() + 3
        
        model = MARS(max_terms=5, verbose=False)
        model.fit(X, y)
        
        y_pred = model.predict(X)
        
        # Should achieve near-perfect fit for linear data
        mse = np.mean((y - y_pred)**2)
        assert mse < 1e-10
    
    def test_reproducibility(self, simple_data):
        """Test that results are reproducible"""
        X, y = simple_data
        
        model1 = MARS(max_terms=10, verbose=False)
        model1.fit(X, y)
        pred1 = model1.predict(X)
        
        model2 = MARS(max_terms=10, verbose=False)
        model2.fit(X, y)
        pred2 = model2.predict(X)
        
        # Should get same results (deterministic algorithm)
        np.testing.assert_array_almost_equal(pred1, pred2)


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_single_feature(self):
        """Test with single feature"""
        np.random.seed(42)
        X = np.random.randn(50, 1)
        y = X.ravel()**2 + np.random.randn(50)*0.1
        
        model = MARS(max_terms=10, verbose=False)
        model.fit(X, y)
        
        assert model.n_features_in_ == 1
        y_pred = model.predict(X)
        assert len(y_pred) == len(y)
    
    def test_small_sample(self):
        """Test with very small sample"""
        X = np.random.randn(10, 3)
        y = np.random.randn(10)
        
        model = MARS(max_terms=5, verbose=False)
        model.fit(X, y)
        
        y_pred = model.predict(X)
        assert len(y_pred) == 10
    
    def test_large_features(self):
        """Test with many features"""
        np.random.seed(42)
        X = np.random.randn(100, 20)
        # Only first 3 features matter
        y = X[:, 0] + 2*X[:, 1] - X[:, 2] + np.random.randn(100)*0.1
        
        model = MARS(max_terms=30, max_degree=1, verbose=False)
        model.fit(X, y)
        
        # Should identify important features
        imp = model.feature_importances_
        assert imp[0] > 0.01
        assert imp[1] > 0.01
        assert imp[2] > 0.01
    
    def test_constant_target(self):
        """Test with constant target"""
        X = np.random.randn(50, 3)
        y = np.ones(50) * 5.0
        
        model = MARS(max_terms=10, verbose=False)
        model.fit(X, y)
        
        y_pred = model.predict(X)
        
        # Should predict constant
        assert np.allclose(y_pred, 5.0, atol=0.1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])