"""Test MARS avec la vraie fonction Friedman 1991"""
import numpy as np
from pymars import MARS
from pymars.interactions import InteractionAnalyzer

def friedman(X):
    """y = 10*sin(π*x1*x2) + 20*(x3-0.5)² + 10*x4 + 5*x5"""
    return (10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 
            20 * (X[:, 2] - 0.5)**2 + 
            10 * X[:, 3] + 
            5 * X[:, 4])

print("\n" + "="*70)
print("TEST: MARS avec Fonction Friedman 1991")
print("="*70)

# Génération
np.random.seed(123)
n_samples = 300
n_features = 10
X = np.random.uniform(0, 1, (n_samples, n_features))
y_true = friedman(X)
y = y_true + np.random.randn(n_samples) * 1.0

print(f"\nDonnées: {n_samples} samples, {n_features} features")
print(f"Fonction vraie: dépend de x0, x1, x2, x3, x4 seulement")
print(f"SNR = {np.std(y_true)/np.std(y - y_true):.2f}")

# Fit MARS
print("\nFitting MARS model...")
model = MARS(max_terms=40, max_degree=2, penalty=3.0, verbose=False)
model.fit(X, y)

# Résultats
y_pred = model.predict(X)
r2 = model.score(X, y)
print(f"\n✓ Model R²: {r2:.4f}")
print(f"✓ Basis functions: {len(model.basis_functions_)}")

# Feature importance
print("\nFEATURE IMPORTANCE:")
important_vars = []
for i in range(n_features):
    imp = model.feature_importances_[i]
    marker = "← IMPORTANT" if imp > 0.02 else ""
    print(f"  x{i}: {imp:.4f} {marker}")
    if imp > 0.02:
        important_vars.append(i)

print(f"\n✓ Variables détectées: {sorted(important_vars)}")
print(f"✓ Variables attendues:  [0, 1, 2, 3, 4]")

# Check accuracy
expected = {0, 1, 2, 3, 4}
detected = set(important_vars)
correct = expected & detected
missed = expected - detected
false_pos = detected - expected

if not false_pos and not missed:
    print("✓✓✓ PARFAIT: Toutes les variables détectées correctement!")
else:
    if missed:
        print(f"  ⚠ Variables manquées: {sorted(missed)}")
    if false_pos:
        print(f"  ⚠ Faux positifs: {sorted(false_pos)}")

# Interactions
print("\nINTERACTIONS DÉTECTÉES (Top 5):")
try:
    analyzer = InteractionAnalyzer(model)
    interactions = analyzer.rank_interactions(top_k=5)
    for vars, strength in interactions:
        print(f"  Variables {vars}: strength={strength:.4f}")
except Exception as e:
    print(f"  Erreur: {e}")

print("\n" + "="*70)
