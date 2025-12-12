"""
Visualization tools for MARS models
===================================

Functions for plotting basis functions, model predictions, and diagnostics.
"""

import numpy as np
from typing import Optional, List, Tuple
import warnings

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("matplotlib not available. Plotting functions will not work.")


def plot_univariate_effects(model, X: np.ndarray, feature_idx: int,
                           n_points: int = 100,
                           ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot the effect of a single feature on predictions
    
    Parameters
    ----------
    model : MARS
        Fitted MARS model
    X : array
        Reference data for other features
    feature_idx : int
        Index of feature to plot
    n_points : int
        Number of points to evaluate
    ax : matplotlib axes, optional
        Axes to plot on
        
    Returns
    -------
    ax : matplotlib axes
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for plotting")
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    # Get feature range
    x_min, x_max = X[:, feature_idx].min(), X[:, feature_idx].max()
    margin = (x_max - x_min) * 0.1
    x_plot = np.linspace(x_min - margin, x_max + margin, n_points)
    
    # Use median values for other features
    X_plot = np.tile(np.median(X, axis=0), (n_points, 1))
    X_plot[:, feature_idx] = x_plot
    
    # Predict
    y_plot = model.predict(X_plot)
    
    # Plot
    ax.plot(x_plot, y_plot, 'b-', linewidth=2, label='MARS prediction')
    ax.scatter(X[:, feature_idx], model.predict(X), 
              alpha=0.3, s=20, c='gray', label='Training data')
    
    ax.set_xlabel(f'x{feature_idx}', fontsize=12)
    ax.set_ylabel('Prediction', fontsize=12)
    ax.set_title(f'Effect of Feature {feature_idx}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_bivariate_effect(model, X: np.ndarray,
                         feature1: int, feature2: int,
                         n_points: int = 50,
                         plot_type: str = 'contour',
                         ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot interaction effect between two features
    
    Parameters
    ----------
    model : MARS
        Fitted MARS model
    X : array
        Reference data
    feature1, feature2 : int
        Indices of features to plot
    n_points : int
        Grid resolution
    plot_type : str
        'contour' or 'surface'
    ax : matplotlib axes, optional
        
    Returns
    -------
    ax : matplotlib axes
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for plotting")
    
    # Create grid
    x1_min, x1_max = X[:, feature1].min(), X[:, feature1].max()
    x2_min, x2_max = X[:, feature2].min(), X[:, feature2].max()
    
    x1_grid = np.linspace(x1_min, x1_max, n_points)
    x2_grid = np.linspace(x2_min, x2_max, n_points)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    
    # Prepare data
    X_plot = np.tile(np.median(X, axis=0), (n_points**2, 1))
    X_plot[:, feature1] = X1.ravel()
    X_plot[:, feature2] = X2.ravel()
    
    # Predict
    y_pred = model.predict(X_plot).reshape(n_points, n_points)
    
    # Plot
    if plot_type == 'contour':
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        contour = ax.contourf(X1, X2, y_pred, levels=20, cmap='viridis')
        plt.colorbar(contour, ax=ax, label='Prediction')
        ax.contour(X1, X2, y_pred, levels=10, colors='white', 
                  alpha=0.3, linewidths=0.5)
        
        ax.set_xlabel(f'x{feature1}', fontsize=12)
        ax.set_ylabel(f'x{feature2}', fontsize=12)
        ax.set_title(f'Interaction: x{feature1} × x{feature2}', 
                    fontsize=14, fontweight='bold')
        
    elif plot_type == 'surface':
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        
        surf = ax.plot_surface(X1, X2, y_pred, cmap='viridis', 
                              alpha=0.8, edgecolor='none')
        plt.colorbar(surf, ax=ax, shrink=0.5, label='Prediction')
        
        ax.set_xlabel(f'x{feature1}', fontsize=10)
        ax.set_ylabel(f'x{feature2}', fontsize=10)
        ax.set_zlabel('Prediction', fontsize=10)
        ax.set_title(f'Interaction: x{feature1} × x{feature2}', 
                    fontsize=12, fontweight='bold')
    else:
        raise ValueError("plot_type must be 'contour' or 'surface'")
    
    return ax


def plot_basis_functions(model, X: np.ndarray, max_plot: int = 6,
                        figsize: Tuple[int, int] = (12, 8)):
    """
    Plot individual basis functions
    
    Parameters
    ----------
    model : MARS
        Fitted MARS model
    X : array
        Data to evaluate on
    max_plot : int
        Maximum number of basis functions to plot
    figsize : tuple
        Figure size
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for plotting")
    
    n_basis = min(len(model.basis_functions_), max_plot)
    n_cols = 3
    n_rows = (n_basis + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.ravel() if n_basis > 1 else [axes]
    
    for i in range(n_basis):
        basis = model.basis_functions_[i]
        coef = model.coefficients_[i]
        
        # Evaluate basis
        values = basis.evaluate(X)
        
        ax = axes[i]
        ax.hist(values, bins=30, edgecolor='black', alpha=0.7)
        ax.set_title(f'Basis {i}: coef={coef:.3f}\n{basis}', fontsize=9)
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_basis, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig


def plot_feature_importances(model, feature_names: Optional[List[str]] = None,
                            figsize: Tuple[int, int] = (8, 5)) -> plt.Figure:
    """
    Bar plot of feature importances
    
    Parameters
    ----------
    model : MARS
        Fitted model
    feature_names : list of str, optional
        Names for features
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig : matplotlib figure
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for plotting")
    
    importances = model.feature_importances_
    n_features = len(importances)
    
    if feature_names is None:
        feature_names = [f'x{i}' for i in range(n_features)]
    
    # Sort by importance
    idx = np.argsort(importances)[::-1]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['green' if imp > 0.01 else 'lightgray' for imp in importances[idx]]
    ax.barh(range(n_features), importances[idx], color=colors, edgecolor='black')
    ax.set_yticks(range(n_features))
    ax.set_yticklabels([feature_names[i] for i in idx])
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title('Feature Importances', fontsize=14, fontweight='bold')
    ax.axvline(0.01, color='red', linestyle='--', linewidth=1, alpha=0.5,
              label='Threshold (0.01)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    return fig


def plot_predictions(model, X: np.ndarray, y: np.ndarray,
                    figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
    """
    Scatter plot of predictions vs actual values
    
    Parameters
    ----------
    model : MARS
        Fitted model
    X, y : arrays
        Data to predict on
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig : matplotlib figure
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for plotting")
    
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Predictions vs actual
    ax1.scatter(y, y_pred, alpha=0.5, s=20)
    lims = [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())]
    ax1.plot(lims, lims, 'r--', linewidth=2, label='Perfect prediction')
    ax1.set_xlabel('Actual', fontsize=12)
    ax1.set_ylabel('Predicted', fontsize=12)
    ax1.set_title('Predictions vs Actual', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Residuals
    ax2.scatter(y_pred, residuals, alpha=0.5, s=20)
    ax2.axhline(0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Predicted', fontsize=12)
    ax2.set_ylabel('Residuals', fontsize=12)
    ax2.set_title('Residual Plot', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_anova_summary(model, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Summary plot of ANOVA decomposition
    
    Parameters
    ----------
    model : MARS
        Fitted model
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig : matplotlib figure
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for plotting")
    
    anova = model.get_anova_decomposition()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Count by degree
    degrees = sorted(anova.keys())
    counts = [len(anova[d]) for d in degrees]
    
    ax1.bar(degrees, counts, edgecolor='black', color='steelblue', alpha=0.7)
    ax1.set_xlabel('Interaction Order', fontsize=12)
    ax1.set_ylabel('Number of Terms', fontsize=12)
    ax1.set_title('Terms by Interaction Order', fontsize=13, fontweight='bold')
    ax1.set_xticks(degrees)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Coefficient magnitudes by degree
    for degree in degrees:
        basis_list = anova[degree]
        indices = [model.basis_functions_.index(b) for b in basis_list]
        coefs = [abs(model.coefficients_[i]) for i in indices]
        
        if coefs:
            positions = [degree] * len(coefs)
            ax2.scatter(positions, coefs, s=50, alpha=0.6)
    
    ax2.set_xlabel('Interaction Order', fontsize=12)
    ax2.set_ylabel('|Coefficient|', fontsize=12)
    ax2.set_title('Coefficient Magnitudes', fontsize=13, fontweight='bold')
    ax2.set_xticks(degrees)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig