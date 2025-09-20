"""Regression utilities used across the project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

try:  # pragma: no cover - optional dependency
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover - fallback when xgboost is unavailable
    XGBRegressor = None  # type: ignore[misc]

__all__ = [
    "RegressionMetrics",
    "RegressionSummary",
    "RegressionTask",
    "cross_validate_regressor",
    "evaluate_regressor_on_tasks",
    "XGBRegressor",
]


@dataclass(slots=True)
class RegressionMetrics:
    mse: float
    rmse: float
    mae: float
    r2: float


@dataclass(slots=True)
class RegressionSummary:
    model_name: str
    metrics: list[RegressionMetrics]

    def mean(self, attribute: str) -> float:
        return float(np.mean([getattr(metric, attribute) for metric in self.metrics]))

    @property
    def mean_mse(self) -> float:
        return self.mean("mse")

    @property
    def mean_rmse(self) -> float:
        return self.mean("rmse")

    @property
    def mean_mae(self) -> float:
        return self.mean("mae")

    @property
    def mean_r2(self) -> float:
        return self.mean("r2")


@dataclass(slots=True)
class RegressionTask:
    name: str
    features: pd.DataFrame | np.ndarray
    target: pd.Series | np.ndarray


def _prepare_arrays(X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    features = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
    targets = y.values if isinstance(y, pd.Series) else np.asarray(y)
    return features, targets


def cross_validate_regressor(
    estimator: RegressorMixin,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    *,
    n_splits: int = 5,
    random_state: int | None = 0,
) -> RegressionSummary:
    """Cross-validate ``estimator`` and return aggregate metrics."""

    X_arr, y_arr = _prepare_arrays(X, y)
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    metrics: list[RegressionMetrics] = []
    for train_idx, test_idx in kfold.split(X_arr):
        model = clone(estimator)
        model.fit(X_arr[train_idx], y_arr[train_idx])
        predictions = model.predict(X_arr[test_idx])

        mse = mean_squared_error(y_arr[test_idx], predictions)
        metrics.append(
            RegressionMetrics(
                mse=mse,
                rmse=float(np.sqrt(mse)),
                mae=mean_absolute_error(y_arr[test_idx], predictions),
                r2=r2_score(y_arr[test_idx], predictions),
            )
        )

    return RegressionSummary(type(estimator).__name__, metrics)


def evaluate_regressor_on_tasks(
    estimator: RegressorMixin,
    tasks: Sequence[RegressionTask],
    *,
    n_splits: int = 5,
    random_state: int | None = 0,
) -> dict[str, RegressionSummary]:
    """Evaluate ``estimator`` on each of ``tasks``."""

    results: dict[str, RegressionSummary] = {}
    for task in tasks:
        results[task.name] = cross_validate_regressor(
            estimator,
            task.features,
            task.target,
            n_splits=n_splits,
            random_state=random_state,
        )
    return results
