"""End-to-end machine learning training orchestration.

Connects dataset splitting, preprocessing, model training, and validation
evaluation into one deterministic training pipeline.

The pipeline deliberately fits preprocessing statistics only on the training
partition to avoid validation-data leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from random import Random

from backend.app.ml.dataset import MLDataset, create_dataset
from backend.app.ml.model import (
    LogisticRiskModel,
    ModelTrainingResult,
    TrainingConfig,
    train_logistic_risk_model,
)
from backend.app.ml.preprocessing import (
    MLPreprocessor,
    fit_preprocessor,
)


class TrainingPipelineError(ValueError):
    """Raised when end-to-end model training cannot be performed safely."""


@dataclass(frozen=True)
class ValidationMetrics:
    """Classification metrics computed on a validation dataset.

    Attributes:
        accuracy:
            Fraction of validation examples classified correctly.

        precision:
            Fraction of predicted positives that are true positives.
            Defined as 0.0 when there are no predicted positives.

        recall:
            Fraction of actual positives that are correctly detected.
            Defined as 0.0 when there are no actual positives.

        true_positives:
            Number of correctly predicted positive examples.

        true_negatives:
            Number of correctly predicted negative examples.

        false_positives:
            Number of negative examples predicted as positive.

        false_negatives:
            Number of positive examples predicted as negative.
    """

    accuracy: float
    precision: float
    recall: float

    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int

    def __post_init__(self) -> None:
        for name, value in (
            ("accuracy", self.accuracy),
            ("precision", self.precision),
            ("recall", self.recall),
        ):
            if not isinstance(value, (int, float)):
                raise TypeError(
                    f"{name} must be numeric"
                )

            if isinstance(value, bool):
                raise TypeError(
                    f"{name} must be numeric"
                )

            numeric_value = float(value)

            if not isfinite(numeric_value):
                raise TrainingPipelineError(
                    f"{name} must be finite"
                )

            if not 0.0 <= numeric_value <= 1.0:
                raise TrainingPipelineError(
                    f"{name} must be between 0.0 and 1.0"
                )

        for name, value in (
            ("true_positives", self.true_positives),
            ("true_negatives", self.true_negatives),
            ("false_positives", self.false_positives),
            ("false_negatives", self.false_negatives),
        ):
            if isinstance(value, bool) or not isinstance(
                value,
                int,
            ):
                raise TypeError(
                    f"{name} must be an integer"
                )

            if value < 0:
                raise TrainingPipelineError(
                    f"{name} must not be negative"
                )

    @property
    def total_count(self) -> int:
        """Return the number of evaluated examples."""

        return (
            self.true_positives
            + self.true_negatives
            + self.false_positives
            + self.false_negatives
        )


@dataclass(frozen=True)
class TrainingPipelineResult:
    """Complete result produced by the ML training pipeline.

    Attributes:
        model:
            Trained logistic regression model.

        preprocessor:
            Preprocessor fitted exclusively on training data.

        training_result:
            Low-level optimization result returned by model training.

        validation_metrics:
            Classification metrics computed on held-out validation data.

        training_row_count:
            Number of examples used for training.

        validation_row_count:
            Number of held-out validation examples.
    """

    model: LogisticRiskModel
    preprocessor: MLPreprocessor
    training_result: ModelTrainingResult
    validation_metrics: ValidationMetrics

    training_row_count: int
    validation_row_count: int

    def __post_init__(self) -> None:
        for name, value in (
            ("training_row_count", self.training_row_count),
            ("validation_row_count", self.validation_row_count),
        ):
            if isinstance(value, bool) or not isinstance(
                value,
                int,
            ):
                raise TypeError(
                    f"{name} must be an integer"
                )

            if value <= 0:
                raise TrainingPipelineError(
                    f"{name} must be greater than zero"
                )


def train_and_validate_model(
    dataset: MLDataset,
    *,
    validation_fraction: float = 0.25,
    random_seed: int = 42,
    training_config: TrainingConfig | None = None,
    classification_threshold: float = 0.5,
) -> TrainingPipelineResult:
    """Train and validate a logistic refund-risk model.

    The dataset is deterministically shuffled using ``random_seed`` and split
    into training and validation partitions.

    Missing-value replacement statistics are fitted exclusively on the
    training partition before being applied to validation data.

    Args:
        dataset:
            Validated supervised dataset.

        validation_fraction:
            Fraction of examples reserved for validation.

        random_seed:
            Seed used for deterministic shuffling.

        training_config:
            Optional logistic regression training configuration.

        classification_threshold:
            Probability threshold used for validation classification.

    Returns:
        Complete training pipeline result.

    Raises:
        TrainingPipelineError:
            If splitting or pipeline configuration is invalid.
    """

    _validate_training_dataset(dataset)

    _validate_validation_fraction(validation_fraction)

    _validate_random_seed(random_seed)

    _validate_classification_threshold(
        classification_threshold
    )

    training_indices, validation_indices = (
        _stratified_split_indices(
            labels=dataset.labels,
            validation_fraction=validation_fraction,
            random_seed=random_seed,
        )
    )

    training_dataset = _subset_dataset(
        dataset=dataset,
        indices=training_indices,
    )

    validation_dataset = _subset_dataset(
        dataset=dataset,
        indices=validation_indices,
    )

    preprocessor = fit_preprocessor(
        feature_names=training_dataset.feature_names,
        feature_rows=training_dataset.feature_matrix(),
    )

    processed_training_rows = preprocessor.transform(
        feature_names=training_dataset.feature_names,
        feature_rows=training_dataset.feature_matrix(),
    )

    processed_validation_rows = preprocessor.transform(
        feature_names=validation_dataset.feature_names,
        feature_rows=validation_dataset.feature_matrix(),
    )

    processed_training_dataset = create_dataset(
        feature_names=training_dataset.feature_names,
        feature_rows=processed_training_rows,
        labels=training_dataset.label_vector(),
    )

    training_result = train_logistic_risk_model(
        processed_training_dataset,
        config=training_config,
    )

    validation_predictions = (
        training_result.model.predict_many(
            processed_validation_rows,
            threshold=classification_threshold,
        )
    )

    validation_metrics = _calculate_validation_metrics(
        actual_labels=validation_dataset.label_vector(),
        predicted_labels=validation_predictions,
    )

    return TrainingPipelineResult(
        model=training_result.model,
        preprocessor=preprocessor,
        training_result=training_result,
        validation_metrics=validation_metrics,
        training_row_count=training_dataset.row_count,
        validation_row_count=validation_dataset.row_count,
    )


def _validate_training_dataset(
    dataset: object,
) -> None:
    """Validate that the supplied object is an ML dataset."""

    if not isinstance(dataset, MLDataset):
        raise TypeError(
            "dataset must be an MLDataset"
        )

    if dataset.row_count < 4:
        raise TrainingPipelineError(
            "Dataset must contain at least 4 rows "
            "for training and validation"
        )

    if dataset.positive_count < 2:
        raise TrainingPipelineError(
            "Dataset must contain at least 2 positive examples "
            "for training and validation"
        )

    if dataset.negative_count < 2:
        raise TrainingPipelineError(
            "Dataset must contain at least 2 negative examples "
            "for training and validation"
        )


def _validate_validation_fraction(
    validation_fraction: float,
) -> None:
    """Validate the requested validation fraction."""

    if isinstance(validation_fraction, bool) or not isinstance(
        validation_fraction,
        (int, float),
    ):
        raise TypeError(
            "validation_fraction must be numeric"
        )

    numeric_value = float(validation_fraction)

    if not isfinite(numeric_value):
        raise TrainingPipelineError(
            "validation_fraction must be finite"
        )

    if not 0.0 < numeric_value < 1.0:
        raise TrainingPipelineError(
            "validation_fraction must be between 0.0 and 1.0"
        )


def _validate_random_seed(
    random_seed: int,
) -> None:
    """Validate the deterministic shuffle seed."""

    if isinstance(random_seed, bool) or not isinstance(
        random_seed,
        int,
    ):
        raise TypeError(
            "random_seed must be an integer"
        )


def _validate_classification_threshold(
    classification_threshold: float,
) -> None:
    """Validate the validation classification threshold."""

    if isinstance(
        classification_threshold,
        bool,
    ) or not isinstance(
        classification_threshold,
        (int, float),
    ):
        raise TypeError(
            "classification_threshold must be numeric"
        )

    numeric_value = float(classification_threshold)

    if not isfinite(numeric_value):
        raise TrainingPipelineError(
            "classification_threshold must be finite"
        )

    if not 0.0 <= numeric_value <= 1.0:
        raise TrainingPipelineError(
            "classification_threshold must be between "
            "0.0 and 1.0"
        )


def _split_indices(
    *,
    row_count: int,
    validation_fraction: float,
    random_seed: int,
) -> tuple[list[int], list[int]]:
    """Create deterministic stratified train/validation indices.

    Both classes are represented in both partitions.
    """

    if row_count < 4:
        raise TrainingPipelineError(
            "Dataset must contain at least 4 rows"
        )

    raise RuntimeError(
        "_split_indices requires class labels and "
        "should not be called directly"
    )


def _stratified_split_indices(
    *,
    labels: tuple[int, ...],
    validation_fraction: float,
    random_seed: int,
) -> tuple[list[int], list[int]]:
    """Create a deterministic stratified train/validation split."""

    positive_indices = [
        index
        for index, label in enumerate(labels)
        if label == 1
    ]

    negative_indices = [
        index
        for index, label in enumerate(labels)
        if label == 0
    ]

    random_generator = Random(random_seed)

    random_generator.shuffle(positive_indices)
    random_generator.shuffle(negative_indices)

    validation_positive_count = _validation_count(
        len(positive_indices),
        validation_fraction,
    )

    validation_negative_count = _validation_count(
        len(negative_indices),
        validation_fraction,
    )

    validation_indices = (
        positive_indices[:validation_positive_count]
        + negative_indices[:validation_negative_count]
    )

    training_indices = (
        positive_indices[validation_positive_count:]
        + negative_indices[validation_negative_count:]
    )

    random_generator.shuffle(training_indices)
    random_generator.shuffle(validation_indices)

    return training_indices, validation_indices


def _validation_count(
    class_count: int,
    validation_fraction: float,
) -> int:
    """Calculate a class validation count while preserving both partitions."""

    requested_count = round(
        class_count * validation_fraction
    )

    return max(
        1,
        min(
            class_count - 1,
            requested_count,
        ),
    )


def _subset_dataset(
    *,
    dataset: MLDataset,
    indices: list[int],
) -> MLDataset:
    """Create a validated dataset containing selected rows."""

    return create_dataset(
        feature_names=dataset.feature_names,
        feature_rows=[
            dataset.feature_rows[index]
            for index in indices
        ],
        labels=[
            dataset.labels[index]
            for index in indices
        ],
    )


def _calculate_validation_metrics(
    *,
    actual_labels: list[int],
    predicted_labels: list[int],
) -> ValidationMetrics:
    """Calculate binary classification metrics."""

    if len(actual_labels) != len(predicted_labels):
        raise TrainingPipelineError(
            "Actual and predicted label counts must match"
        )

    if not actual_labels:
        raise TrainingPipelineError(
            "Validation labels must not be empty"
        )

    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    for actual, predicted in zip(
        actual_labels,
        predicted_labels,
    ):
        if actual == 1 and predicted == 1:
            true_positives += 1
        elif actual == 0 and predicted == 0:
            true_negatives += 1
        elif actual == 0 and predicted == 1:
            false_positives += 1
        elif actual == 1 and predicted == 0:
            false_negatives += 1
        else:
            raise TrainingPipelineError(
                "Labels must be binary"
            )

    total_count = len(actual_labels)

    accuracy = (
        true_positives + true_negatives
    ) / total_count

    predicted_positive_count = (
        true_positives + false_positives
    )

    actual_positive_count = (
        true_positives + false_negatives
    )

    precision = (
        true_positives / predicted_positive_count
        if predicted_positive_count
        else 0.0
    )

    recall = (
        true_positives / actual_positive_count
        if actual_positive_count
        else 0.0
    )

    return ValidationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        true_positives=true_positives,
        true_negatives=true_negatives,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )