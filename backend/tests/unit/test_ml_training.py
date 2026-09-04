"""Unit tests for end-to-end ML training orchestration."""

import pytest

from backend.app.ml.dataset import create_dataset
from backend.app.ml.model import TrainingConfig
from backend.app.ml.training import (
    TrainingPipelineError,
    train_and_validate_model,
)


def _create_dataset():
    """Create a balanced deterministic dataset for training tests."""

    return create_dataset(
        feature_names=(
            "risk_signal",
            "velocity_signal",
        ),
        feature_rows=[
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [0.3, 0.2],
            [0.7, 0.8],
            [0.8, 0.7],
            [0.9, 0.9],
            [1.0, 1.0],
        ],
        labels=[
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
        ],
    )


def test_training_pipeline_returns_complete_result() -> None:
    """The pipeline should train and return validation results."""

    result = train_and_validate_model(
        _create_dataset(),
        validation_fraction=0.25,
        training_config=TrainingConfig(
            learning_rate=0.3,
            epochs=500,
        ),
    )

    assert result.training_row_count > 0
    assert result.validation_row_count > 0

    assert (
        result.training_row_count
        + result.validation_row_count
        == 8
    )

    assert (
        result.model.feature_names
        == (
            "risk_signal",
            "velocity_signal",
        )
    )

    assert result.validation_metrics.total_count == (
        result.validation_row_count
    )


def test_training_and_validation_both_preserve_classes() -> None:
    """Stratified splitting should preserve both classes for training."""

    result = train_and_validate_model(
        _create_dataset(),
        validation_fraction=0.25,
        training_config=TrainingConfig(
            learning_rate=0.3,
            epochs=500,
        ),
    )

    metrics = result.validation_metrics

    assert (
        metrics.true_positives
        + metrics.false_negatives
        > 0
    )

    assert (
        metrics.true_negatives
        + metrics.false_positives
        > 0
    )


def test_training_pipeline_is_deterministic() -> None:
    """Identical data and seed should produce identical results."""

    dataset = _create_dataset()

    config = TrainingConfig(
        learning_rate=0.3,
        epochs=500,
    )

    first_result = train_and_validate_model(
        dataset,
        random_seed=123,
        training_config=config,
    )

    second_result = train_and_validate_model(
        dataset,
        random_seed=123,
        training_config=config,
    )

    assert first_result.model == second_result.model
    assert (
        first_result.validation_metrics
        == second_result.validation_metrics
    )


def test_training_pipeline_fits_preprocessor_on_training_data() -> None:
    """The returned preprocessor should match the model feature schema."""

    result = train_and_validate_model(
        _create_dataset(),
        training_config=TrainingConfig(
            learning_rate=0.3,
            epochs=500,
        ),
    )

    assert (
        result.preprocessor.feature_names
        == result.model.feature_names
    )


def test_validation_metrics_are_in_valid_ranges() -> None:
    """All reported validation metrics should be probabilities."""

    result = train_and_validate_model(
        _create_dataset(),
        training_config=TrainingConfig(
            learning_rate=0.3,
            epochs=500,
        ),
    )

    metrics = result.validation_metrics

    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.precision <= 1.0
    assert 0.0 <= metrics.recall <= 1.0


def test_training_rejects_dataset_with_too_few_rows() -> None:
    """Training and validation require enough examples."""

    dataset = create_dataset(
        feature_names=("signal",),
        feature_rows=[
            [0.0],
            [1.0],
            [0.5],
        ],
        labels=[
            0,
            1,
            1,
        ],
    )

    with pytest.raises(
        TrainingPipelineError,
        match="at least 4 rows",
    ):
        train_and_validate_model(dataset)


def test_training_rejects_insufficient_positive_examples() -> None:
    """Both partitions require positive examples."""

    dataset = create_dataset(
        feature_names=("signal",),
        feature_rows=[
            [0.0],
            [0.1],
            [0.2],
            [1.0],
        ],
        labels=[
            0,
            0,
            0,
            1,
        ],
    )

    with pytest.raises(
        TrainingPipelineError,
        match="at least 2 positive examples",
    ):
        train_and_validate_model(dataset)


def test_training_rejects_insufficient_negative_examples() -> None:
    """Both partitions require negative examples."""

    dataset = create_dataset(
        feature_names=("signal",),
        feature_rows=[
            [0.0],
            [0.8],
            [0.9],
            [1.0],
        ],
        labels=[
            0,
            1,
            1,
            1,
        ],
    )

    with pytest.raises(
        TrainingPipelineError,
        match="at least 2 negative examples",
    ):
        train_and_validate_model(dataset)


@pytest.mark.parametrize(
    "validation_fraction",
    [
        0.0,
        1.0,
        -0.1,
        1.1,
    ],
)
def test_training_rejects_invalid_validation_fraction(
    validation_fraction: float,
) -> None:
    """Validation fraction must be strictly between zero and one."""

    with pytest.raises(
        TrainingPipelineError,
        match="between 0.0 and 1.0",
    ):
        train_and_validate_model(
            _create_dataset(),
            validation_fraction=validation_fraction,
        )


def test_training_rejects_non_integer_random_seed() -> None:
    """The deterministic random seed must be an integer."""

    with pytest.raises(
        TypeError,
        match="random_seed must be an integer",
    ):
        train_and_validate_model(
            _create_dataset(),
            random_seed=1.5,
        )


@pytest.mark.parametrize(
    "threshold",
    [
        -0.1,
        1.1,
    ],
)
def test_training_rejects_invalid_classification_threshold(
    threshold: float,
) -> None:
    """Validation classification threshold must be valid."""

    with pytest.raises(
        TrainingPipelineError,
        match="between 0.0 and 1.0",
    ):
        train_and_validate_model(
            _create_dataset(),
            classification_threshold=threshold,
        )

def test_group_aware_split_keeps_groups_isolated() -> None:
    """No group may appear in both training and validation partitions."""

    dataset = create_dataset(
        feature_names=("signal",),
        feature_rows=[
            [0.0], [0.1], [0.2], [0.3],
            [0.8], [0.9], [1.0], [1.1],
        ],
        labels=[0, 0, 0, 0, 1, 1, 1, 1],
    )
    groups = ["n1", "n1", "n2", "n2", "p1", "p1", "p2", "p2"]

    result = train_and_validate_model(
        dataset,
        groups=groups,
        random_seed=7,
        training_config=TrainingConfig(learning_rate=0.2, epochs=200),
    )

    assert result.training_row_count + result.validation_row_count == 8


def test_group_aware_split_rejects_mixed_label_groups() -> None:
    """A group identifier must represent one class for leakage-safe splitting."""

    dataset = create_dataset(
        feature_names=("signal",),
        feature_rows=[[0.0], [0.1], [0.9], [1.0]],
        labels=[0, 1, 0, 1],
    )

    with pytest.raises(TrainingPipelineError, match="exactly one class"):
        train_and_validate_model(
            dataset,
            groups=["mixed", "mixed", "n2", "p2"],
        )
