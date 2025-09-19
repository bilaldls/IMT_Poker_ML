import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Chargement du dataset encodé
df = pd.read_csv("../CSV/Encoded/OneHot/poker_encoded_onehot(100).csv", header=None)

# Séparation des données
X = df.iloc[:, :119]
y_preflop = df.iloc[:, 119]
y_fullboard = df.iloc[:, 120]

# Vérification (optionnelle) : conversion des types
y_preflop = pd.to_numeric(y_preflop, errors='coerce').dropna()
y_fullboard = pd.to_numeric(y_fullboard, errors='coerce').dropna()

# Définir les bornes des bins entre 0 et 1
bins = np.linspace(0, 1, 21)  # 20 intervalles : 0.0, 0.05, ..., 1.0

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(y_preflop, bins=bins, color='skyblue', edgecolor='black')
plt.title("Distribution - Preflop Win Rate")
plt.xlabel("Probabilité (Win Rate)")
plt.ylabel("Fréquence")
plt.xticks(bins, rotation=45)

plt.subplot(1, 2, 2)
plt.hist(y_fullboard, bins=bins, color='salmon', edgecolor='black')
plt.title("Distribution - Fullboard Win Rate")
plt.xlabel("Probabilité (Win Rate)")
plt.ylabel("Fréquence")
plt.xticks(bins, rotation=45)

plt.tight_layout()
plt.show()

# 2. Vérification du nombre d'exemples uniques
unique_inputs = X.drop_duplicates().shape[0]
total_examples = X.shape[0]

print("🔎 Nombre d'exemples totaux :", total_examples)
print("📊 Nombre d'exemples uniques (features) :", unique_inputs)
print("📉 Ratio d'unicité :", round(unique_inputs / total_examples, 3))

# 3. Statistiques descriptives sur les cibles
print("\n📈 Statistiques des taux de victoire :")

def show_stats(label, y):
    print(f"➡️ {label}")
    print("  Moyenne :", round(y.mean(), 4))
    print("  Médiane :", round(y.median(), 4))
    print("  Écart-type :", round(y.std(), 4))
    print("  Min :", round(y.min(), 4))
    print("  Max :", round(y.max(), 4))
    print()

show_stats("Preflop Win Rate", y_preflop)
show_stats("Fullboard Win Rate", y_fullboard)