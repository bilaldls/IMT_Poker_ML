import pandas as pd

# Chargement du dataset (modifie le chemin si besoin)
df = pd.read_csv("/Poker_Finie/Code/CSV/poker_winrates(Monte_carlo=100).csv", header=None)

# Index des deux dernières colonnes
preflop_col = df.columns[-2]
fullboard_col = df.columns[-1]

# Conversion explicite en float (important !)
df[preflop_col] = pd.to_numeric(df[preflop_col], errors='coerce')
df[fullboard_col] = pd.to_numeric(df[fullboard_col], errors='coerce')

# Filtrage : garder uniquement les lignes avec des win rates valides entre 0 et 1
df_clean = df[
    (df[preflop_col] >= 0) & (df[preflop_col] <= 1) &
    (df[fullboard_col] >= 0) & (df[fullboard_col] <= 1)
    ]

# Affichage du résultat
print(f"Lignes initiales : {len(df)}")
print(f"Lignes après nettoyage : {len(df_clean)}")
print(f"Lignes supprimées : {len(df) - len(df_clean)}")

# Sauvegarde
df_clean.to_csv("poker_winrates1(Monte_carlo=100).csv", index=False, header=False)