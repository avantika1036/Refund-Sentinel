"""Unit tests for ML model evaluation."""

from __future__ import annotations

import pytest

from backend.app.ml.evaluation import (
    ConfusionMatrix,
    EvaluationError,
    EvaluationMetrics,
    evaluate_model,
)
from backend.app.ml.model import LogisticRiskModel


def _create_model() -> LogisticRiskModel:
    """Create a simple deterministic logistic model for testing."""

    return LogisticRiskModel(
        feature_names=("signal",),
        coefficients=(1.0,),
        intercept=0.0,
    )


def test_confusion_matrix_total() -> None:
    """Confusion matrix should report its total example count."""

    matrix = ConfusionMatrix(
        true_positive=3,
        true_negative=4,
        false_positive=2,
        false_negative=1,
    )

    assert matrix.total == 10


@pytest.mark.parametrize(
    "value",
    [
        -1,
        -10,
    ],
)
def test_confusion_matrix_rejects_negative_values(
    value: int,
) -> None:
    """Confusion matrix values cannot be negative."""

    with pytest.raises(
        EvaluationError,
        match="must not be negative",
    ):
        ConfusionMatrix(
            true_positive=value,
            true_negative=0,
            false_positive=0,
            false_negative=0,
        )


def test_confusion_matrix_rejects_boolean_values() -> None:
    """Boolean values must not be treated as counts."""

    with pytest.raises(
        TypeError,
        match="must be integers",
    ):
        ConfusionMatrix(
            true_positive=True,
            true_negative=0,
            false_positive=0,
            false_negative=0,
        )


def test_evaluation_metrics_rejects_invalid_score() -> None:
    """Metric values must remain within [0, 1]."""

    with pytest.raises(
        EvaluationError,
        match="accuracy must be between",
    ):
        EvaluationMetrics(
            accuracy=1.1,
            precision=0.5,
            recall=0.5,
            f1_score=0.5,
            confusion_matrix=ConfusionMatrix(
                true_positive=1,
                true_negative=1,
                false_positive=0,
                false_negative=0,
            ),
        )


def test_evaluate_model_perfect_predictions() -> None:
    """Perfect predictions should produce perfect metrics."""

    model = _create_model()

    metrics = evaluate_model(
        model=model,
        feature_rows=[
            (-10.0,),
            (-5.0,),
            (5.0,),
            (10.0,),
        ],
        labels=[
            0,
            0,
            1,
            1,
        ],
    )

    assert metrics.accuracy == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1_score == 1.0

    assert metrics.confusion_matrix == (
        ConfusionMatrix(
            true_positive=2,
            true_negative=2,
            false_positive=0,
            false_negative=0,
        )
    )


def test_evaluate_model_mixed_predictions() -> None:
    """Evaluation should calculate mixed confusion-matrix metrics."""

    model = _create_model()

    metrics = evaluate_model(
        model=model,
        feature_rows=[
            (-10.0,),
            (10.0,),
            (10.0,),
            (-10.0,),
        ],
        labels=[
            0,
            0,
            1,
            1,
        ],
    )

    assert metrics.confusion_matrix == (
        ConfusionMatrix(
            true_positive=1,
            true_negative=1,
            false_positive=1,
            false_negative=1,
        )
    )

    assert metrics.accuracy == 0.5
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    assert metrics.f1_score == 0.5


def test_evaluate_model_handles_zero_positive_predictions() -> None:
    """Precision and F1 should safely become zero when undefined."""

    model = LogisticRiskModel(
        feature_names=("signal",),
        coefficients=(0.0,),
        intercept=-10.0,
    )

    metrics = evaluate_model(
        model=model,
        feature_rows=[
            (1.0,),
            (2.0,),
        ],
        labels=[
            1,
            0,
        ],
    )

    assert metrics.confusion_matrix == (
        ConfusionMatrix(
            true_positive=0,
            true_negative=1,
            false_positive=0,
            false_negative=1,
        )
    )

    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1_score == 0.0


def test_evaluate_model_rejects_mismatched_lengths() -> None:
    """Feature rows and labels must align exactly."""

    with pytest.raises(
        EvaluationError,
        match="same length",
    ):
        evaluate_model(
            model=_create_model(),
            feature_rows=[
                (1.0,),
            ],
            labels=[
                1,
                0,
            ],
        )


def test_evaluate_model_rejects_empty_data() -> None:
    """Evaluation requires at least one example."""

    with pytest.raises(
        EvaluationError,
        match="without feature rows",
    ):
        evaluate_model(
            model=_create_model(),
            feature_rows=[],
            labels=[],
        )


@pytest.mark.parametrize(
    "threshold",
    [
        -0.1,
        1.1,
    ],
)
def test_evaluate_model_rejects_invalid_threshold(
    threshold: float,
) -> None:
    """Classification threshold must remain inside [0, 1]."""

    with pytest.raises(
        EvaluationError,
        match="between 0.0 and 1.0",
    ):
        evaluate_model(
            model=_create_model(),
            feature_rows=[
                (1.0,),
            ],
            labels=[
                1,
            ],
            classification_threshold=threshold,
        )


def test_evaluate_model_rejects_boolean_threshold() -> None:
    """Boolean values must not be accepted as thresholds."""

    with pytest.raises(
        TypeError,
        match="classification_threshold must be numeric",
    ):
        evaluate_model(
            model=_create_model(),
            feature_rows=[
                (1.0,),
            ],
            labels=[
                1,
            ],
            classification_threshold=True,
        )


def test_evaluate_model_rejects_invalid_label() -> None:
    """Labels must be binary integers."""

    with pytest.raises(
        EvaluationError,
        match="must be 0 or 1",
    ):
        evaluate_model(
            model=_create_model(),
            feature_rows=[
                (1.0,),
            ],
            labels=[
                2,
            ],
        )