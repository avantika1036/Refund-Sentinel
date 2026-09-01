"""Unit tests for ML logistic risk model training and prediction."""

import pytest

from backend.app.ml.dataset import create_dataset
from backend.app.ml.model import (
    LogisticRiskModel,
    ModelError,
    TrainingConfig,
    train_logistic_risk_model,
)


def test_model_predicts_probability_in_valid_range() -> None:
    """Predicted probabilities must always be in [0, 1]."""

    model = LogisticRiskModel(
        feature_names=("risk_signal",),
        coefficients=(2.0,),
        intercept=-1.0,
    )

    probability = model.predict_probability([1.0])

    assert 0.0 <= probability <= 1.0


def test_model_prediction_respects_threshold() -> None:
    """Binary predictions should use the supplied threshold."""

    model = LogisticRiskModel(
        feature_names=("signal",),
        coefficients=(2.0,),
        intercept=-1.0,
    )

    assert model.predict([1.0], threshold=0.5) == 1
    assert model.predict([0.0], threshold=0.5) == 0


def test_model_rejects_wrong_feature_count() -> None:
    """Prediction rows must match the trained feature schema."""

    model = LogisticRiskModel(
        feature_names=("a", "b"),
        coefficients=(1.0, 1.0),
        intercept=0.0,
    )

    with pytest.raises(
        ModelError,
        match="Expected 2 feature values",
    ):
        model.predict_probability([1.0])


def test_model_predicts_multiple_rows() -> None:
    """The model should support batch prediction."""

    model = LogisticRiskModel(
        feature_names=("signal",),
        coefficients=(2.0,),
        intercept=-1.0,
    )

    probabilities = model.predict_probabilities(
        [
            [0.0],
            [1.0],
            [2.0],
        ]
    )

    assert len(probabilities) == 3
    assert probabilities[0] < probabilities[1] < probabilities[2]


def test_training_learns_simple_separable_pattern() -> None:
    """Training should learn a simple binary relationship."""

    dataset = create_dataset(
        feature_names=("signal",),
        feature_rows=[
            [0.0],
            [0.1],
            [0.2],
            [0.8],
            [0.9],
            [1.0],
        ],
        labels=[
            0,
            0,
            0,
            1,
            1,
            1,
        ],
    )

    result = train_logistic_risk_model(
        dataset,
        config=TrainingConfig(
            learning_rate=0.5,
            epochs=2_000,
        ),
    )

    assert result.epochs_completed == 2_000
    assert result.training_loss >= 0.0

    assert result.model.predict([0.0]) == 0
    assert result.model.predict([1.0]) == 1


def test_training_requires_positive_examples() -> None:
    """A classifier cannot train meaningfully without positives."""

    dataset = create_dataset(
        feature_names=("signal",),
        feature_rows=[
            [0.0],
            [1.0],
        ],
        labels=[
            0,
            0,
        ],
    )

    with pytest.raises(
        ModelError,
        match="positive example",
    ):
        train_logistic_risk_model(dataset)


def test_training_requires_negative_examples() -> None:
    """A classifier cannot train meaningfully without negatives."""

    dataset = create_dataset(
        feature_names=("signal",),
        feature_rows=[
            [0.0],
            [1.0],
        ],
        labels=[
            1,
            1,
        ],
    )

    with pytest.raises(
        ModelError,
        match="negative example",
    ):
        train_logistic_risk_model(dataset)


def test_training_is_deterministic() -> None:
    """Identical data and configuration should produce identical models."""

    dataset = create_dataset(
        feature_names=("signal",),
        feature_rows=[
            [0.0],
            [0.2],
            [0.8],
            [1.0],
        ],
        labels=[
            0,
            0,
            1,
            1,
        ],
    )

    config = TrainingConfig(
        learning_rate=0.3,
        epochs=500,
    )

    first_result = train_logistic_risk_model(
        dataset,
        config=config,
    )

    second_result = train_logistic_risk_model(
        dataset,
        config=config,
    )

    assert first_result.model == second_result.model
    assert (
        first_result.training_loss
        == second_result.training_loss
    )


def test_training_config_rejects_invalid_values() -> None:
    """Invalid training configuration should fail clearly."""

    with pytest.raises(
        ModelError,
        match="learning_rate must be greater than zero",
    ):
        TrainingConfig(learning_rate=0)

    with pytest.raises(
        ModelError,
        match="epochs must be greater than zero",
    ):
        TrainingConfig(epochs=0)

    with pytest.raises(
        ModelError,
        match="must not be negative",
    ):
        TrainingConfig(l2_regularization=-0.1)


def test_model_rejects_invalid_threshold() -> None:
    """Prediction thresholds must remain in the probability range."""

    model = LogisticRiskModel(
        feature_names=("signal",),
        coefficients=(1.0,),
        intercept=0.0,
    )

    with pytest.raises(
        ModelError,
        match="threshold must be between",
    ):
        model.predict([1.0], threshold=1.5)