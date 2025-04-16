import pandas as pd

# Liste des valeurs et couleurs pour l'encodage one-hot
value_list = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
suit_list = ['s', 'h', 'd', 'c']

# Fonction d'encodage d'une carte (one-hot sur valeur + couleur)
def encode_card(card):
    card = card.strip()
    value = card[0]
    suit = card[1]

    # One-hot pour la valeur
    value_encoding = [1 if v == value else 0 for v in value_list]

    # One-hot pour la couleur
    suit_encoding = [1 if s == suit else 0 for s in suit_list]

    return value_encoding + suit_encoding  # concatène les deux encodages

# Lecture du CSV
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
encoded_df.to_csv("poker_encoded_onehot(10000).csv", index=False)
print("✅ Encodage one-hot terminé. Fichier sauvegardé : poker_encoded_onehot.csv")