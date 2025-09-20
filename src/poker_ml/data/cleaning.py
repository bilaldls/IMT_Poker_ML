"""Data loading and cleaning helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from poker_ml.utils import PROCESSED_DATA_DIR, ensure_directory

__all__ = [
    "CleaningConfig",
    "load_dataset",
    "clean_win_rates",
    "save_dataset",
]


@dataclass(slots=True)
class CleaningConfig:
    """Configuration for the :func:`clean_win_rates` pipeline."""

    input_path: Path
    output_path: Path = PROCESSED_DATA_DIR / "poker_winrates_clean.csv"
    win_rate_columns: Sequence[int | str] = (-2, -1)
    min_win_rate: float = 0.0
    max_win_rate: float = 1.0

    def resolved_input_path(self) -> Path:
        return self.input_path.expanduser().resolve()

    def resolved_output_path(self) -> Path:
        return ensure_directory(self.output_path.parent) / self.output_path.name


def load_dataset(path: Path, *, header: int | None = None) -> pd.DataFrame:
    """Load a dataset from ``path`` into a :class:`pandas.DataFrame`."""

    return pd.read_csv(path, header=header)


def clean_win_rates(df: pd.DataFrame, config: CleaningConfig) -> pd.DataFrame:
    """Return a copy of ``df`` where the win-rate columns are numeric and clipped."""

    cleaned = df.copy()
    columns = [cleaned.columns[index] if isinstance(index, int) else index for index in config.win_rate_columns]

    for column in columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    mask = (cleaned[columns] >= config.min_win_rate) & (cleaned[columns] <= config.max_win_rate)
    return cleaned[mask.all(axis=1)].reset_index(drop=True)


def save_dataset(df: pd.DataFrame, path: Path, *, header: bool = False) -> Path:
    """Persist ``df`` to ``path`` and return the resolved path."""

    resolved_path = ensure_directory(path.parent) / path.name
    df.to_csv(resolved_path, index=False, header=header)
    return resolved_path
