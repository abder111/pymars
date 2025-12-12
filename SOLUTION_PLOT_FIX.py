"""
EXEMPLE DE CORRECTION - Comment utiliser plot_univariate_effects correctement
==============================================================================
"""

from pymars import MARS
from pymars.plots import plot_univariate_effects
import numpy as np
import matplotlib.pyplot as plt

# Générer des données
np.random.seed(42)
X = np.random.uniform(-3, 3, (200, 3))
y = np.sin(X[:, 0]) + 2*X[:, 1]**2 + 0.5*X[:, 2] + np.random.randn(200)*0.2

# Fit MARS
print("Fitting MARS model...")
model = MARS(max_terms=20, max_degree=2, verbose=False)
model.fit(X, y)
print(f"✓ Model fitted with {len(model.basis_functions_)} basis functions")

# ============================================================================
# ❌ MAUVAIS - Ce qui cause l'erreur:
# ============================================================================
print("\n❌ ERREUR (à ne pas faire):")
print("   plot_univariate_effects(model, X, y)  # y n'est pas un feature_idx!")

# Cela donnerait:
# IndexError: arrays used as indices must be of integer (or boolean) type


# ============================================================================
# ✓ CORRECT - Solution 1: Spécifier feature_idx comme entier
# ============================================================================
print("\n✓ SOLUTION 1: Spécifier feature_idx")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i in range(3):
    ax = axes[i]
    plot_univariate_effects(model, X, feature_idx=i, ax=ax)

plt.tight_layout()
plt.savefig('partial_effects_solution1.png')
print("   ✓ Sauvegardé: partial_effects_solution1.png")
plt.close()


# ============================================================================
# ✓ CORRECT - Solution 2: Utiliser une boucle
# ============================================================================
print("\n✓ SOLUTION 2: Boucler sur toutes les features")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i in range(X.shape[1]):
    plot_univariate_effects(model, X, feature_idx=i, ax=axes[i])

plt.suptitle("Univariate Effects of All Features")
plt.tight_layout()
plt.savefig('partial_effects_solution2.png')
print("   ✓ Sauvegardé: partial_effects_solution2.png")
plt.close()


# ============================================================================
# ✓ CORRECT - Solution 3: Pour une seule feature
# ============================================================================
print("\n✓ SOLUTION 3: Une seule feature")

fig, ax = plt.subplots(figsize=(8, 5))
plot_univariate_effects(model, X, feature_idx=0, ax=ax)
plt.savefig('partial_effect_feature0.png')
print("   ✓ Sauvegardé: partial_effect_feature0.png")
plt.close()


# ============================================================================
# Information supplémentaire
# ============================================================================
print("\n" + "="*70)
print("EXPLICATION:")
print("="*70)
print("""
La fonction plot_univariate_effects a cette signature:

  plot_univariate_effects(model, X, feature_idx, n_points=100, ax=None)
  
Paramètres:
  - model: MARS object (fitted)
  - X: array des données (n_samples, n_features)
  - feature_idx: int - Index de la feature à plotter (0, 1, 2, ...)
  - n_points: int - Nombre de points pour l'évaluation (default 100)
  - ax: matplotlib axes (optionnel)

Exemple CORRECT:
  plot_univariate_effects(model, X, feature_idx=0)

Exemple FAUX:
  plot_univariate_effects(model, X, y)  # ❌ y est un array, pas un int!
  plot_univariate_effects(model, X, np.array(...))  # ❌ Idem!
""")

print("✓ Toutes les visualisations ont été créées avec succès!")
print("="*70)
