import pandas as pd

# === Définition des cartes ===
value_map = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
    '7': 7, '8': 8, '9': 9, 'T': 10,
    'J': 11, 'Q': 12, 'K': 13, 'A': 14
}
suits = ['s', 'h', 'd', 'c']

# === Fonction d'encodage : valeur brute + couleur one-hot ===
def encode_card(card):
    card = card.strip()
    value, suit = card[0], card[1]

    value_encoded = [value_map[value]]  # valeur brute (1 seule colonne)
    suit_encoded = [1 if s == suit else 0 for s in suits]  # one-hot couleur

    return value_encoded + suit_encoded  # total 5 colonnes

# === Chargement du fichier CSV brut ===
df = pd.read_csv("/Users/leovasseur/IdeaProjects/Poker1Git//Poker_Finie/Code/CSV/poker_winrates(Monte_Carlo=100).csv", sep=',')

# Suppression éventuelle de la première ligne (mauvais header)
df = df.drop(index=0)

# Nom des colonnes manuellement
df.columns = [
    "hand_1", "hand_2",
    "board_1", "board_2", "board_3", "board_4", "board_5",
    "preflop_win_rate", "fullboard_win_rate"
]

# === Encodage de chaque ligne ===
encoded_rows = []

for _, row in df.iterrows():
    row_encoded = []
    # On encode toutes les cartes : 2 main + 5 board
    for col in ["hand_1", "hand_2", "board_1", "board_2", "board_3", "board_4", "board_5"]:
        row_encoded.extend(encode_card(row[col]))
    # Ajout des winrates
    row_encoded.append(float(row["preflop_win_rate"]))
    row_encoded.append(float(row["fullboard_win_rate"]))
    encoded_rows.append(row_encoded)

# === Création du DataFrame final ===
encoded_df = pd.DataFrame(encoded_rows)

# === Sauvegarde ===
filename = "../CSV/Encoded/Couleur_OneHot/poker_encoded_couleurOneHot(100).csv"
encoded_df.to_csv(filename, index=False)
print(f"✅ Encodage terminé. Fichier sauvegardé : {filename}")