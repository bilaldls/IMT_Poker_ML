"""Reusable training routines for the classification experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import label_binarize

__all__ = [
    "FoldMetrics",
    "ClassificationSummary",
    "discretize_probabilities",
    "binarize_probabilities",
    "evaluate_multiclass_logistic_regression",
    "evaluate_naive_bayes",
    "evaluate_random_forest",
]


@dataclass(slots=True)
class FoldMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None = None


@dataclass(slots=True)
class ClassificationSummary:
    model_name: str
    metrics: list[FoldMetrics]
    confusion_matrices: list[np.ndarray]

    def mean(self, attribute: str) -> float:
        return float(np.mean([getattr(metric, attribute) for metric in self.metrics]))

    @property
    def mean_accuracy(self) -> float:
        return self.mean("accuracy")

    @property
    def mean_precision(self) -> float:
        return self.mean("precision")

    @property
    def mean_recall(self) -> float:
        return self.mean("recall")

    @property
    def mean_f1(self) -> float:
        return self.mean("f1")

    @property
    def mean_roc_auc(self) -> float | None:
        values = [metric.roc_auc for metric in self.metrics if metric.roc_auc is not None]
        if not values:
            return None
        return float(np.mean(values))

    @property
    def mean_confusion_matrix(self) -> np.ndarray:
        return np.mean(self.confusion_matrices, axis=0)


Boundaries = Sequence[float]


def discretize_probabilities(values: Iterable[float], boundaries: Boundaries) -> np.ndarray:
    """Map probabilities to integer classes according to ``boundaries``."""

    sorted_boundaries = sorted(boundaries)
    classes = []
    for value in values:
        for index, boundary in enumerate(sorted_boundaries):
            if value < boundary:
                classes.append(index)
                break
        else:
            classes.append(len(sorted_boundaries))
    return np.asarray(classes)


def binarize_probabilities(values: Iterable[float], threshold: float) -> np.ndarray:
    return (np.asarray(list(values)) > threshold).astype(int)


def _prepare_arrays(X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    features = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
    targets = y.values if isinstance(y, pd.Series) else np.asarray(y)
    return features, targets


def evaluate_multiclass_logistic_regression(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    *,
    use_smote: bool = False,
    n_splits: int = 5,
    random_state: int | None = 0,
    max_iter: int = 1_000,
) -> ClassificationSummary:
    """Evaluate a multinomial logistic regression model using cross-validation."""

    X_arr, y_arr = _prepare_arrays(X, y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    metrics: list[FoldMetrics] = []
    confusion_matrices: list[np.ndarray] = []
    classes = np.unique(y_arr)

    for train_idx, test_idx in skf.split(X_arr, y_arr):
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        y_train, y_test = y_arr[train_idx], y_arr[test_idx]

        if use_smote:
            smote = SMOTE(random_state=random_state)
            X_train, y_train = smote.fit_resample(X_train, y_train)

        model = LogisticRegression(
            max_iter=max_iter,
            multi_class="multinomial",
            solver="lbfgs",
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        metric = FoldMetrics(
            accuracy=accuracy_score(y_test, y_pred),
            precision=precision_score(y_test, y_pred, average="macro", zero_division=0),
            recall=recall_score(y_test, y_pred, average="macro", zero_division=0),
            f1=f1_score(y_test, y_pred, average="macro", zero_division=0),
            roc_auc=roc_auc_score(
                label_binarize(y_test, classes=classes),
                y_proba,
                multi_class="ovr",
                average="macro",
            ),
        )
        metrics.append(metric)
        confusion_matrices.append(confusion_matrix(y_test, y_pred))

    return ClassificationSummary("Logistic Regression", metrics, confusion_matrices)


def evaluate_naive_bayes(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    *,
    n_splits: int = 5,
    random_state: int | None = 0,
) -> ClassificationSummary:
    """Evaluate a Gaussian Naive Bayes classifier with cross-validation."""

    X_arr, y_arr = _prepare_arrays(X, y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    metrics: list[FoldMetrics] = []
    confusion_matrices: list[np.ndarray] = []

    for train_idx, test_idx in skf.split(X_arr, y_arr):
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        y_train, y_test = y_arr[train_idx], y_arr[test_idx]

        model = GaussianNB()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if model.classes_.shape[0] == 2 else None

        roc_auc = None
        if y_proba is not None:
            roc_auc = roc_auc_score(y_test, y_proba)

        metric = FoldMetrics(
            accuracy=accuracy_score(y_test, y_pred),
            precision=precision_score(y_test, y_pred, zero_division=0),
            recall=recall_score(y_test, y_pred, zero_division=0),
            f1=f1_score(y_test, y_pred, zero_division=0),
            roc_auc=roc_auc,
        )
        metrics.append(metric)
        confusion_matrices.append(confusion_matrix(y_test, y_pred))

    return ClassificationSummary("Gaussian Naive Bayes", metrics, confusion_matrices)


def evaluate_random_forest(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    *,
    n_splits: int = 5,
    random_state: int | None = 0,
    n_estimators: int = 100,
) -> ClassificationSummary:
    """Evaluate a random forest classifier."""

    X_arr, y_arr = _prepare_arrays(X, y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    metrics: list[FoldMetrics] = []
    confusion_matrices: list[np.ndarray] = []
    classes = np.unique(y_arr)

    for train_idx, test_idx in skf.split(X_arr, y_arr):
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        y_train, y_test = y_arr[train_idx], y_arr[test_idx]

        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        metric = FoldMetrics(
            accuracy=accuracy_score(y_test, y_pred),
            precision=precision_score(y_test, y_pred, average="macro", zero_division=0),
            recall=recall_score(y_test, y_pred, average="macro", zero_division=0),
            f1=f1_score(y_test, y_pred, average="macro", zero_division=0),
            roc_auc=roc_auc_score(
                label_binarize(y_test, classes=classes),
                y_proba,
                multi_class="ovr",
                average="macro",
            ),
        )
        metrics.append(metric)
        confusion_matrices.append(confusion_matrix(y_test, y_pred))

    return ClassificationSummary("Random Forest", metrics, confusion_matrices)
