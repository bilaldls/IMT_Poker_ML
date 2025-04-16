import pandas as pd

# Mapping des valeurs
value_map = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
    '7': 7, '8': 8, '9': 9, 'T': 10,
    'J': 11, 'Q': 12, 'K': 13, 'A': 14
}
suits = ['s', 'h', 'd', 'c']

# Fonction d'encodage d'une carte
def encode_card(card):
    card = card.strip()
    value = value_map[card[0]]
    suit = card[1]

    suit_encoding = [1 if s == suit else 0 for s in suits]
    return [value] + suit_encoding  # Valeur + One-hot de la couleur

# Lecture du CSV avec le bon séparateur
df = pd.read_csv("poker_winrates.csv", sep=',')

df.columns = [
    "hand_1", "hand_2",
    "board_1", "board_2", "board_3", "board_4", "board_5",
    "preflop_win_rate", "fullboard_win_rate"
]

# Encodage des cartes
encoded_rows = []

for _, row in df.iterrows():
    row_encoded = []
    for col in ["hand_1", "hand_2", "board_1", "board_2", "board_3", "board_4", "board_5"]:
        row_encoded.extend(encode_card(row[col]))
    row_encoded.append(row["preflop_win_rate"])
    row_encoded.append(row["fullboard_win_rate"])
    encoded_rows.append(row_encoded)

# Création du DataFrame final encodé
encoded_df = pd.DataFrame(encoded_rows)

# Sauvegarde
encoded_df.to_csv("poker_encoded.csv", index=False)
print("✅ Encodage terminé. Fichier sauvegardé : poker_encoded.csv")