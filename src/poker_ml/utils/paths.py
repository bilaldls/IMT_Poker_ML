"""Utility helpers for locating important project directories."""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


def ensure_directory(path: Path) -> Path:
    """Create *path* if it does not already exist and return it.

    The helper mirrors :func:`pathlib.Path.mkdir` with ``parents=True`` and
    ``exist_ok=True`` so that calling code does not have to worry about
    whether a directory already exists.
    """

    path.mkdir(parents=True, exist_ok=True)
    return path


__all__ = [
    "PACKAGE_ROOT",
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "MODELS_DIR",
    "ensure_directory",
]
