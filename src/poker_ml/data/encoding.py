"""Helpers to transform card representations into machine learning features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import pandas as pd

from poker_ml.utils import PROCESSED_DATA_DIR, ensure_directory

__all__ = [
    "EncodingConfig",
    "encode_card_value_suit",
    "encode_card_full_one_hot",
    "encode_dataframe",
]

VALUE_ORDER = "23456789TJQKA"
SUITS = "shdc"
DEFAULT_CARD_COLUMNS = (
    "hand_1",
    "hand_2",
    "board_1",
    "board_2",
    "board_3",
    "board_4",
    "board_5",
)
DEFAULT_TARGET_COLUMNS = ("preflop_win_rate", "fullboard_win_rate")
DEFAULT_COLUMNS = DEFAULT_CARD_COLUMNS + DEFAULT_TARGET_COLUMNS


@dataclass(slots=True)
class EncodingConfig:
    """Describe how a raw dataset should be encoded."""

    input_path: Path
    output_path: Path = PROCESSED_DATA_DIR / "poker_encoded.csv"
    encoding: Literal["value_suit", "full_one_hot"] = "value_suit"
    card_columns: Sequence[str] = DEFAULT_CARD_COLUMNS
    target_columns: Sequence[str] = DEFAULT_TARGET_COLUMNS
    header: int | None = None
    column_names: Sequence[str] = DEFAULT_COLUMNS
    drop_rows: int = 0

    def resolved_input_path(self) -> Path:
        return self.input_path.expanduser().resolve()

    def resolved_output_path(self) -> Path:
        return ensure_directory(self.output_path.parent) / self.output_path.name

    def card_encoding_size(self) -> int:
        return 5 if self.encoding == "value_suit" else 17


def _normalize_card(card: str) -> str:
    card = card.strip()
    if len(card) != 2:
        raise ValueError(f"Invalid card value: {card!r}")
    value, suit = card[0].upper(), card[1].lower()
    if value not in VALUE_ORDER or suit not in SUITS:
        raise ValueError(f"Unknown card provided: {card!r}")
    return value + suit


def encode_card_value_suit(card: str) -> list[int]:
    """Encode a card as the numeric value and suit one-hot vector (5 features)."""

    value, suit = _normalize_card(card)
    value_score = VALUE_ORDER.index(value) + 2
    suit_one_hot = [1 if candidate == suit else 0 for candidate in SUITS]
    return [value_score, *suit_one_hot]


def encode_card_full_one_hot(card: str) -> list[int]:
    """Encode a card using full one-hot representations for value and suit (17 features)."""

    value, suit = _normalize_card(card)
    value_one_hot = [1 if candidate == value else 0 for candidate in VALUE_ORDER]
    suit_one_hot = [1 if candidate == suit else 0 for candidate in SUITS]
    return [*value_one_hot, *suit_one_hot]


def _select_encoding_function(encoding: str):  # type: ignore[no-untyped-def]
    if encoding == "value_suit":
        return encode_card_value_suit
    if encoding == "full_one_hot":
        return encode_card_full_one_hot
    raise ValueError(f"Unsupported encoding: {encoding}")


def encode_dataframe(df: pd.DataFrame, config: EncodingConfig) -> pd.DataFrame:
    """Return an encoded view of ``df`` based on ``config``."""

    encoder = _select_encoding_function(config.encoding)
    encoded_rows = []

    for _, row in df.iterrows():
        features: list[int | float] = []
        for column in config.card_columns:
            features.extend(encoder(row[column]))
        for column in config.target_columns:
            features.append(float(row[column]))
        encoded_rows.append(features)

    encoded_df = pd.DataFrame(encoded_rows)
    return encoded_df


def encode_dataset(config: EncodingConfig) -> Path:
    """Load, encode and persist a dataset according to ``config``."""

    path = config.resolved_input_path()
    df = pd.read_csv(path, header=config.header)
    if config.drop_rows:
        df = df.iloc[config.drop_rows :].reset_index(drop=True)
    df.columns = list(config.column_names)

    encoded_df = encode_dataframe(df, config)
    output_path = config.resolved_output_path()
    encoded_df.to_csv(output_path, index=False, header=False)
    return output_path


__all__.append("encode_dataset")
