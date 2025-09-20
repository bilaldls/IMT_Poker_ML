"""Visualisation helpers for the encoded datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all__ = [
    "EncodedDataset",
    "load_encoded_dataset",
    "plot_win_rate_histograms",
    "describe_win_rates",
]


@dataclass(slots=True)
class EncodedDataset:
    features: pd.DataFrame
    preflop: pd.Series
    fullboard: pd.Series


def load_encoded_dataset(
    path: Path,
    *,
    card_feature_count: int,
    has_header: bool = False,
) -> EncodedDataset:
    """Load an encoded dataset and split the target columns."""

    header = 0 if has_header else None
    df = pd.read_csv(path, header=header)

    x = df.iloc[:, :card_feature_count]
    preflop = df.iloc[:, card_feature_count]
    fullboard = df.iloc[:, card_feature_count + 1]

    preflop = pd.to_numeric(preflop, errors="coerce").dropna()
    fullboard = pd.to_numeric(fullboard, errors="coerce").dropna()

    return EncodedDataset(x, preflop, fullboard)


def plot_win_rate_histograms(dataset: EncodedDataset, *, bins: int = 20) -> None:
    """Display histograms of the pre-flop and full-board win rates."""

    bin_edges = np.linspace(0, 1, bins + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(dataset.preflop, bins=bin_edges, color="skyblue", edgecolor="black")
    plt.title("Distribution - Preflop Win Rate")
    plt.xlabel("Win Rate")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    plt.hist(dataset.fullboard, bins=bin_edges, color="salmon", edgecolor="black")
    plt.title("Distribution - Fullboard Win Rate")
    plt.xlabel("Win Rate")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()


def describe_win_rates(values: Iterable[float]) -> dict[str, float]:
    """Return basic descriptive statistics for ``values``."""

    series = pd.Series(list(values))
    return {
        "mean": float(series.mean()),
        "median": float(series.median()),
        "std": float(series.std()),
        "min": float(series.min()),
        "max": float(series.max()),
    }
