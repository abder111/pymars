"""
Tests for basis functions
=========================
"""

import pytest
import numpy as np
from pymars.basis import HingeFunction, BasisFunction, build_design_matrix


class TestHingeFunction:
    """Tests for HingeFunction"""
    
    def test_initialization(self):
        """Test hinge function creation"""
        h = HingeFunction(variable=0, knot=0.5, direction=1)
        assert h.variable == 0
        assert h.knot == 0.5
        assert h.direction == 1
    
    def test_invalid_direction(self):
        """Test that invalid direction raises error"""
        with pytest.raises(ValueError):
            HingeFunction(variable=0, knot=0.5, direction=2)
    
    def test_evaluate_right_hinge(self):
        """Test right hinge: max(0, x - t)"""
        h = HingeFunction(variable=0, knot=0.5, direction=1)
        
        X = np.array([[0.0], [0.5], [1.0]])
        result = h.evaluate(X)
        
        expected = np.array([0.0, 0.0, 0.5])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_evaluate_left_hinge(self):
        """Test left hinge: max(0, t - x)"""
        h = HingeFunction(variable=0, knot=0.5, direction=-1)
        
        X = np.array([[0.0], [0.5], [1.0]])
        result = h.evaluate(X)
        
        expected = np.array([0.5, 0.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_evaluate_multivariate(self):
        """Test hinge on multivariate data"""
        h = HingeFunction(variable=1, knot=0.0, direction=1)
        
        X = np.array([
            [1.0, -1.0],
            [2.0,  0.0],
            [3.0,  1.0]
        ])
        result = h.evaluate(X)
        
        expected = np.array([0.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_repr(self):
        """Test string representation"""
        h1 = HingeFunction(variable=2, knot=1.5, direction=1)
        assert 'x2' in repr(h1)
        assert '1.5' in repr(h1)
        assert '+' in repr(h1)
        
        h2 = HingeFunction(variable=0, knot=0.5, direction=-1)
        assert '-' in repr(h2)


class TestBasisFunction:
    """Tests for BasisFunction"""
    
    def test_constant_basis(self):
        """Test constant basis function"""
        b = BasisFunction()
        
        assert b.degree == 0
        assert len(b.variables) == 0
        
        X = np.random.randn(10, 3)
        result = b.evaluate(X)
        
        np.testing.assert_array_equal(result, np.ones(10))
    
    def test_single_hinge(self):
        """Test basis with single hinge"""
        h = HingeFunction(variable=0, knot=0.0, direction=1)
        b = BasisFunction(hinges=[h])
        
        assert b.degree == 1
        assert b.variables == [0]
        
        X = np.array([[-1.0], [0.0], [1.0]])
        result = b.evaluate(X)
        
        expected = np.array([0.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_product_of_hinges(self):
        """Test product of multiple hinges"""
        h1 = HingeFunction(variable=0, knot=0.0, direction=1)
        h2 = HingeFunction(variable=1, knot=0.5, direction=1)
        b = BasisFunction(hinges=[h1, h2])
        
        assert b.degree == 2
        assert set(b.variables) == {0, 1}
        
        X = np.array([
            [1.0, 0.0],  # h1=1.0, h2=0.0 -> product=0.0
            [1.0, 1.0],  # h1=1.0, h2=0.5 -> product=0.5
            [2.0, 2.0],  # h1=2.0, h2=1.5 -> product=3.0
        ])
        result = b.evaluate(X)
        
        expected = np.array([0.0, 0.5, 3.0])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_add_hinge(self):
        """Test adding hinge to basis"""
        h1 = HingeFunction(variable=0, knot=0.0, direction=1)
        b1 = BasisFunction(hinges=[h1])
        
        h2 = HingeFunction(variable=1, knot=0.5, direction=-1)
        b2 = b1.add_hinge(h2)
        
        # Original unchanged
        assert b1.degree == 1
        
        # New basis has both hinges
        assert b2.degree == 2
        assert set(b2.variables) == {0, 1}
    
    def test_get_knot_info(self):
        """Test knot information extraction"""
        h1 = HingeFunction(variable=2, knot=1.5, direction=1)
        h2 = HingeFunction(variable=3, knot=0.5, direction=-1)
        b = BasisFunction(hinges=[h1, h2])
        
        knots = b.get_knot_info()
        
        assert len(knots) == 2
        assert knots[0] == (2, 1.5, 1)
        assert knots[1] == (3, 0.5, -1)


class TestBuildDesignMatrix:
    """Tests for design matrix construction"""
    
    def test_constant_only(self):
        """Test design matrix with constant only"""
        b = BasisFunction()
        X = np.random.randn(5, 3)
        
        B = build_design_matrix(X, [b])
        
        assert B.shape == (5, 1)
        np.testing.assert_array_equal(B[:, 0], np.ones(5))
    
    def test_multiple_basis(self):
        """Test with multiple basis functions"""
        b0 = BasisFunction()  # Constant
        
        h1 = HingeFunction(variable=0, knot=0.0, direction=1)
        b1 = BasisFunction(hinges=[h1])
        
        h2 = HingeFunction(variable=1, knot=0.0, direction=1)
        b2 = BasisFunction(hinges=[h2])
        
        X = np.array([
            [-1.0, -1.0],
            [0.0, 0.0],
            [1.0, 1.0]
        ])
        
        B = build_design_matrix(X, [b0, b1, b2])
        
        expected = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0]
        ])
        
        np.testing.assert_array_almost_equal(B, expected)
    
    def test_shape(self):
        """Test output shape"""
        basis_list = [BasisFunction() for _ in range(5)]
        X = np.random.randn(10, 3)
        
        B = build_design_matrix(X, basis_list)
        
        assert B.shape == (10, 5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])