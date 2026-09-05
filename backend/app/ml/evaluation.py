"""Evaluation utilities for Refund Sentinel ML models.

Provides deterministic binary-classification metrics for evaluating
trained risk models against labeled feature rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from backend.app.ml.model import LogisticRiskModel


class EvaluationError(ValueError):
    """Raised when model evaluation cannot be performed safely."""


@dataclass(frozen=True)
class ConfusionMatrix:
    """Binary-classification confusion matrix."""

    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int

    def __post_init__(self) -> None:
        values = (
            self.true_positive,
            self.true_negative,
            self.false_positive,
            self.false_negative,
        )

        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            for value in values
        ):
            raise TypeError(
                "Confusion matrix values must be integers"
            )

        if any(value < 0 for value in values):
            raise EvaluationError(
                "Confusion matrix values must not be negative"
            )

    @property
    def total(self) -> int:
        """Return the total number of evaluated examples."""

        return (
            self.true_positive
            + self.true_negative
            + self.false_positive
            + self.false_negative
        )


@dataclass(frozen=True)
class EvaluationMetrics:
    """Summary metrics for binary classification."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: ConfusionMatrix

    def __post_init__(self) -> None:
        for name in (
            "accuracy",
            "precision",
            "recall",
            "f1_score",
        ):
            value = getattr(self, name)

            if isinstance(value, bool) or not isinstance(
                value,
                (int, float),
            ):
                raise TypeError(
                    f"{name} must be numeric"
                )

            numeric_value = float(value)

            if not 0.0 <= numeric_value <= 1.0:
                raise EvaluationError(
                    f"{name} must be between 0.0 and 1.0"
                )

            object.__setattr__(
                self,
                name,
                numeric_value,
            )


def evaluate_model(
    *,
    model: LogisticRiskModel,
    feature_rows: Sequence[Sequence[float]],
    labels: Sequence[int],
    classification_threshold: float = 0.5,
) -> EvaluationMetrics:
    """Evaluate a trained binary risk model.

    Args:
        model:
            Trained logistic risk model.

        feature_rows:
            Model-ready feature rows. Each row must match the
            model's expected feature width.

        labels:
            Binary ground-truth labels where 1 represents the
            positive/high-risk class and 0 represents the negative class.

        classification_threshold:
            Probability threshold for positive classification.

    Returns:
        EvaluationMetrics containing standard binary-classification metrics.
    """

    if not isinstance(model, LogisticRiskModel):
        raise TypeError(
            "model must be a LogisticRiskModel"
        )

    _validate_threshold(
        classification_threshold
    )

    if len(feature_rows) != len(labels):
        raise EvaluationError(
            "Feature rows and labels must have the same length"
        )

    if not feature_rows:
        raise EvaluationError(
            "Cannot evaluate a model without feature rows"
        )

    threshold = float(
        classification_threshold
    )

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for row_index, (
        feature_row,
        label,
    ) in enumerate(
        zip(
            feature_rows,
            labels,
        )
    ):
        _validate_label(
            label=label,
            row_index=row_index,
        )

        probability = (
            model.predict_probability(
                feature_row
            )
        )

        predicted_positive = (
            probability >= threshold
        )

        actual_positive = label == 1

        if predicted_positive and actual_positive:
            true_positive += 1
        elif predicted_positive:
            false_positive += 1
        elif actual_positive:
            false_negative += 1
        else:
            true_negative += 1

    confusion_matrix = ConfusionMatrix(
        true_positive=true_positive,
        true_negative=true_negative,
        false_positive=false_positive,
        false_negative=false_negative,
    )

    accuracy = _safe_divide(
        numerator=(
            true_positive
            + true_negative
        ),
        denominator=confusion_matrix.total,
    )

    precision = _safe_divide(
        numerator=true_positive,
        denominator=(
            true_positive
            + false_positive
        ),
    )

    recall = _safe_divide(
        numerator=true_positive,
        denominator=(
            true_positive
            + false_negative
        ),
    )

    f1_score = _safe_divide(
        numerator=2.0 * precision * recall,
        denominator=precision + recall,
    )

    return EvaluationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        confusion_matrix=confusion_matrix,
    )


def compute_binary_metrics(
    *,
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> EvaluationMetrics:
    """Compute binary classification metrics from true and predicted binary labels."""
    if len(y_true) != len(y_pred):
        raise EvaluationError("y_true and y_pred must have the same length")

    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    cm = ConfusionMatrix(
        true_positive=tp,
        true_negative=tn,
        false_positive=fp,
        false_negative=fn,
    )

    accuracy = _safe_divide(numerator=tp + tn, denominator=cm.total)
    precision = _safe_divide(numerator=tp, denominator=tp + fp)
    recall = _safe_divide(numerator=tp, denominator=tp + fn)
    f1_score = _safe_divide(numerator=2.0 * precision * recall, denominator=precision + recall)

    return EvaluationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        confusion_matrix=cm,
    )


def _validate_threshold(
    threshold: float,
) -> None:
    """Validate a binary classification threshold."""

    if isinstance(threshold, bool) or not isinstance(
        threshold,
        (int, float),
    ):
        raise TypeError(
            "classification_threshold must be numeric"
        )

    numeric_threshold = float(
        threshold
    )

    if not 0.0 <= numeric_threshold <= 1.0:
        raise EvaluationError(
            "classification_threshold must be between "
            "0.0 and 1.0"
        )


def _validate_label(
    *,
    label: object,
    row_index: int,
) -> None:
    """Validate one binary ground-truth label."""

    if isinstance(label, bool) or not isinstance(
        label,
        int,
    ):
        raise TypeError(
            f"Label at row {row_index} must be an integer"
        )

    if label not in (
        0,
        1,
    ):
        raise EvaluationError(
            f"Label at row {row_index} must be 0 or 1"
        )


def _safe_divide(
    *,
    numerator: float,
    denominator: float,
) -> float:
    """Divide safely, returning 0.0 for a zero denominator."""

    if denominator == 0:
        return 0.0

    return float(
        numerator / denominator
    )