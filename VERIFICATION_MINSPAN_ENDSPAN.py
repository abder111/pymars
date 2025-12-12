"""
VÉRIFICATION DÉTAILLÉE: Formules Minspan et Endspan
====================================================

Comparaison avec Friedman (1991) - Multivariate Adaptive Regression Splines

Source: Friedman, J.H. (1991). "Multivariate Adaptive Regression Splines"
Annals of Statistics, Vol. 19, No. 1, pp. 1-141
"""

import numpy as np

print("="*80)
print("ANALYSE: Formules Minspan et Endspan (Friedman 1991)")
print("="*80)

# ============================================================================
# FORMULE MINSPAN - Page 94 du papier Friedman
# ============================================================================

print("\n1️⃣  MINSPAN FORMULA")
print("-" * 80)

print("\n📖 FRIEDMAN 1991 (Page 94, Equation 3.8):")
print("   L = -log₂(α/n) / 2.5")
print("   où:")
print("      α = significance level (default 0.05)")
print("      n = NUMBER OF SAMPLES (n_samples)")

print("\n❌ CODE ACTUEL (pymars/utils.py, ligne 141):")
print("""
    def calculate_minspan(n_samples: int, n_features: int, alpha: float = 0.05) -> int:
        l_star = -np.log2(alpha / n_features) / 2.5  # ← ERREUR: n_features au lieu de n_samples!
        minspan = max(0, int(np.floor(l_star)))
        return minspan
""")

print("\n✓ CODE CORRECT devrait être:")
print("""
    def calculate_minspan(n_samples: int, n_features: int, alpha: float = 0.05) -> int:
        l_star = -np.log2(alpha / n_samples) / 2.5   # ← CORRECT: n_samples
        minspan = max(0, int(np.floor(l_star)))
        return minspan
""")

# ============================================================================
# IMPACT NUMÉRIQUES
# ============================================================================

print("\n📊 IMPACT NUMÉRIQUES:")
print("-" * 80)

alpha = 0.05
n_samples = 200
n_features = 10

# Calcul FAUX
minspan_wrong = -np.log2(alpha / n_features) / 2.5
minspan_wrong = max(0, int(np.floor(minspan_wrong)))

# Calcul CORRECT
minspan_correct = -np.log2(alpha / n_samples) / 2.5
minspan_correct = max(0, int(np.floor(minspan_correct)))

print(f"\nAvec n_samples = {n_samples}, n_features = {n_features}, α = {alpha}:")
print(f"\n❌ FAUX  (alpha/n_features): minspan = {minspan_wrong}")
print(f"✓ CORRECT (alpha/n_samples): minspan = {minspan_correct}")
print(f"\nDifférence: {abs(minspan_wrong - minspan_correct)} observations")

# Autres exemples
print("\n\n📈 Autres exemples:")
print(f"{'n_samples':<12} {'n_features':<12} {'FAUX':<10} {'CORRECT':<10} {'Différence':<12}")
print("-" * 60)

for n_samples in [100, 200, 500, 1000]:
    for n_features in [5, 10, 20]:
        wrong = max(0, int(np.floor(-np.log2(0.05 / n_features) / 2.5)))
        correct = max(0, int(np.floor(-np.log2(0.05 / n_samples) / 2.5)))
        diff = abs(wrong - correct)
        print(f"{n_samples:<12} {n_features:<12} {wrong:<10} {correct:<10} {diff:<12}")

# ============================================================================
# FORMULE ENDSPAN - Page 94 du papier Friedman
# ============================================================================

print("\n\n2️⃣  ENDSPAN FORMULA")
print("-" * 80)

print("\n📖 FRIEDMAN 1991 (Page 94, Equation 3.9):")
print("   Le = 3 - log₂(α/n)")
print("   où:")
print("      α = significance level (default 0.05)")
print("      n = NUMBER OF SAMPLES (n_samples)")

print("\n❌ CODE ACTUEL (pymars/utils.py, ligne 158):")
print("""
    def calculate_endspan(n_features: int, alpha: float = 0.05) -> int:
        le = 3 - np.log2(alpha / n_features)  # ← ERREUR: n_features au lieu de n_samples!
        endspan = max(1, int(np.ceil(le)))
        return endspan
""")

print("\n⚠️  PROBLÈME: Signature de la fonction ne reçoit que n_features, pas n_samples!")
print("   La fonction ne peut donc pas utiliser n_samples même si elle le voulait.")

print("\n✓ CODE CORRECT devrait être:")
print("""
    def calculate_endspan(n_samples: int, n_features: int, alpha: float = 0.05) -> int:
        le = 3 - np.log2(alpha / n_samples)  # ← CORRECT: n_samples
        endspan = max(1, int(np.ceil(le)))
        return endspan
""")

# Impact numériques endspan
print("\n📊 IMPACT NUMÉRIQUES (ENDSPAN):")
print("-" * 80)

alpha = 0.05
n_samples = 200
n_features = 10

# Calcul FAUX
endspan_wrong = 3 - np.log2(alpha / n_features)
endspan_wrong = max(1, int(np.ceil(endspan_wrong)))

# Calcul CORRECT
endspan_correct = 3 - np.log2(alpha / n_samples)
endspan_correct = max(1, int(np.ceil(endspan_correct)))

print(f"\nAvec n_samples = {n_samples}, n_features = {n_features}, α = {alpha}:")
print(f"\n❌ FAUX  (alpha/n_features): endspan = {endspan_wrong}")
print(f"✓ CORRECT (alpha/n_samples): endspan = {endspan_correct}")
print(f"\nDifférence: {abs(endspan_wrong - endspan_correct)} observations")

# ============================================================================
# APPELS DANS LE CODE
# ============================================================================

print("\n\n3️⃣  APPELS DANS LE CODE")
print("-" * 80)

print("\n📍 En mars.py, ligne 176-179:")
print("""
    if self.minspan == 'auto':
        minspan = calculate_minspan(n_samples, n_features, self.alpha)  # OK: passe n_samples
    
    if self.endspan == 'auto':
        endspan = calculate_endspan(n_features, self.alpha)  # ❌ OUBLIE n_samples!
""")

print("\n✓ Devrait être:")
print("""
    if self.minspan == 'auto':
        minspan = calculate_minspan(n_samples, n_features, self.alpha)
    
    if self.endspan == 'auto':
        endspan = calculate_endspan(n_samples, n_features, self.alpha)  # ← Ajouter n_samples
""")

# ============================================================================
# RÉSUMÉ
# ============================================================================

print("\n\n" + "="*80)
print("📋 RÉSUMÉ DES CORRECTIONS NÉCESSAIRES")
print("="*80)

print("""
❌ PROBLÈME 1: minspan utilise alpha/n_features au lieu de alpha/n_samples
   Location: pymars/utils.py, ligne 141
   Fix: Changer l_star = -np.log2(alpha / n_features) / 2.5
        en     l_star = -np.log2(alpha / n_samples) / 2.5

❌ PROBLÈME 2: endspan utilise alpha/n_features au lieu de alpha/n_samples
   Location: pymars/utils.py, ligne 158
   Fix: Changer le = 3 - np.log2(alpha / n_features)
        en     le = 3 - np.log2(alpha / n_samples)

❌ PROBLÈME 3: calculate_endspan() ne reçoit pas n_samples
   Location: pymars/utils.py, ligne 148 (signature)
   Fix: Changer def calculate_endspan(n_features: int, alpha: float = 0.05)
        en     def calculate_endspan(n_samples: int, n_features: int, alpha: float = 0.05)

❌ PROBLÈME 4: L'appel à calculate_endspan() ne passe pas n_samples
   Location: pymars/mars.py, ligne 178
   Fix: Changer endspan = calculate_endspan(n_features, self.alpha)
        en     endspan = calculate_endspan(n_samples, n_features, self.alpha)
""")

print("\n" + "="*80)
print("SÉVÉRITÉ: ⚠️  MOYENNE")
print("="*80)
print("""
Ces erreurs affectent les paramètres de régularisation du modèle MARS.
Avec les valeurs incorrectes, le modèle peut être:
  - Trop restrictif (minspan/endspan trop grands) → modèle sous-ajusté
  - Trop permissif (minspan/endspan trop petits) → surajustement

Impact: Les nœuds ne sont pas sélectionnés de manière optimale.
""")

print("\n" + "="*80)
