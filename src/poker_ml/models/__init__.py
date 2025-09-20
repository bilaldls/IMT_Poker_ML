"""Model evaluation utilities."""

from .classification import (
    ClassificationSummary,
    FoldMetrics,
    binarize_probabilities,
    discretize_probabilities,
    evaluate_multiclass_logistic_regression,
    evaluate_naive_bayes,
    evaluate_random_forest,
)
from .regression import (
    RegressionMetrics,
    RegressionSummary,
    RegressionTask,
    XGBRegressor,
    cross_validate_regressor,
    evaluate_regressor_on_tasks,
)

__all__ = [
    "ClassificationSummary",
    "FoldMetrics",
    "binarize_probabilities",
    "discretize_probabilities",
    "evaluate_multiclass_logistic_regression",
    "evaluate_naive_bayes",
    "evaluate_random_forest",
    "RegressionMetrics",
    "RegressionSummary",
    "RegressionTask",
    "XGBRegressor",
    "cross_validate_regressor",
    "evaluate_regressor_on_tasks",
]
