import csv
from treys import Card, Deck, Evaluator
import random
from tqdm import tqdm

# Fonction Monte Carlo : estime le taux de victoire
def simulate_win_rate(player_hand_str, board_str=[], nb_simulations=1000):
    evaluator = Evaluator()
    wins = 0
    ties = 0

    player_hand = [Card.new(c) for c in player_hand_str]
    board = [Card.new(c) for c in board_str]
    known_cards = player_hand + board

    for _ in range(nb_simulations):
        deck = Deck()
        for card in known_cards:
            deck.cards.remove(card)

        opp_hand = deck.draw(2)

        # Compléter le board jusqu’à 5 cartes
        full_board = board + deck.draw(5 - len(board))

        p_score = evaluator.evaluate(full_board, player_hand)
        o_score = evaluator.evaluate(full_board, opp_hand)

        if p_score < o_score:
            wins += 1
        elif p_score == o_score:
            ties += 1

    return round((wins + 0.5 * ties) / nb_simulations, 4)

# Génère un exemple complet avec preflop + board complet
def generate_example(nb_simulations):
    deck = Deck()
    player_hand = deck.draw(2)
    board = deck.draw(5)

    hand_str = [Card.int_to_str(c) for c in player_hand]
    board_str = [Card.int_to_str(c) for c in board]

    preflop_win_rate = simulate_win_rate(hand_str, board_str=[], nb_simulations=nb_simulations)
    fullboard_win_rate = simulate_win_rate(hand_str, board_str=board_str, nb_simulations=nb_simulations)

    return {
        "hand_1": hand_str[0],
        "hand_2": hand_str[1],
        "board_1": board_str[0],
        "board_2": board_str[1],
        "board_3": board_str[2],
        "board_4": board_str[3],
        "board_5": board_str[4],
        "preflop_win_rate": preflop_win_rate,
        "fullboard_win_rate": fullboard_win_rate
    }

# Génération CSV
def generate_dataset_csv(n=1000, nb_simulations=1000, output_file="poker_winrates.csv"):
    fieldnames = [
        "hand_1", "hand_2",
        "board_1", "board_2", "board_3", "board_4", "board_5",
        "preflop_win_rate", "fullboard_win_rate"
    ]

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for _ in tqdm(range(n), desc="Génération du dataset"):
            row = generate_example(nb_simulations)
            writer.writerow(row)

    print(f"\n✅ Dataset complet sauvegardé dans {output_file}")

# Lancement
if __name__ == "__main__":
    generate_dataset_csv(n=10000, nb_simulations=10000)