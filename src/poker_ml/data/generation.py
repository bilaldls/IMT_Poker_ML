"""Utilities to generate poker win-rate datasets using Monte Carlo."""

from __future__ import annotations

from csv import DictWriter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence
from treys import Card, Deck, Evaluator
from tqdm import tqdm

from poker_ml.utils import RAW_DATA_DIR, ensure_directory

__all__ = [
    "SimulationConfig",
    "simulate_win_rate",
    "generate_example",
    "generate_dataset",
]


def _normalize_card(card: str) -> str:
    card = card.strip().upper()
    if len(card) != 2:
        raise ValueError(f"Invalid card representation: {card!r}")
    return card


@dataclass(slots=True)
class SimulationConfig:
    """Configuration for dataset generation."""

    simulations: int = 1_000
    dataset_size: int = 1_000
    output_path: Path = RAW_DATA_DIR / "poker_winrates.csv"

    def resolved_output_path(self) -> Path:
        return ensure_directory(self.output_path.parent) / self.output_path.name


_evaluator = Evaluator()


def simulate_win_rate(
    player_hand: Sequence[str],
    board: Sequence[str] | None = None,
    *,
    simulations: int,
) -> float:
    """Estimate the win rate of ``player_hand`` against a random opponent."""

    normalized_hand = [_normalize_card(c) for c in player_hand]
    normalized_board = [_normalize_card(c) for c in board or []]

    player_cards = [Card.new(card) for card in normalized_hand]
    board_cards = [Card.new(card) for card in normalized_board]

    wins = 0
    ties = 0

    for _ in range(simulations):
        deck = Deck()
        for card in player_cards + board_cards:
            deck.cards.remove(card)

        opponent_hand = deck.draw(2)
        full_board = board_cards + deck.draw(5 - len(board_cards))

        player_score = _evaluator.evaluate(full_board, player_cards)
        opponent_score = _evaluator.evaluate(full_board, opponent_hand)

        if player_score < opponent_score:
            wins += 1
        elif player_score == opponent_score:
            ties += 1

    return round((wins + 0.5 * ties) / simulations, 4)


def generate_example(*, simulations: int) -> dict[str, str | float]:
    """Generate a single training example with pre-flop and full board win-rates."""

    deck = Deck()
    player_hand = deck.draw(2)
    board_cards = deck.draw(5)

    hand_str = [Card.int_to_str(card) for card in player_hand]
    board_str = [Card.int_to_str(card) for card in board_cards]

    preflop_win_rate = simulate_win_rate(hand_str, simulations=simulations)
    fullboard_win_rate = simulate_win_rate(hand_str, board_str, simulations=simulations)

    return {
        "hand_1": hand_str[0],
        "hand_2": hand_str[1],
        "board_1": board_str[0],
        "board_2": board_str[1],
        "board_3": board_str[2],
        "board_4": board_str[3],
        "board_5": board_str[4],
        "preflop_win_rate": preflop_win_rate,
        "fullboard_win_rate": fullboard_win_rate,
    }


def generate_dataset(config: SimulationConfig) -> Path:
    """Generate a dataset based on ``config`` and return the output path."""

    output_path = config.resolved_output_path()
    field_names: List[str] = [
        "hand_1",
        "hand_2",
        "board_1",
        "board_2",
        "board_3",
        "board_4",
        "board_5",
        "preflop_win_rate",
        "fullboard_win_rate",
    ]

    with output_path.open("w", newline="") as csv_file:
        writer = DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()
        for _ in tqdm(range(config.dataset_size), desc="Generating dataset"):
            writer.writerow(generate_example(simulations=config.simulations))

    return output_path
