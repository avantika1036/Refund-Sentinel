"""Machine learning model training and prediction.

Provides a small, dependency-free logistic regression implementation for
binary refund-risk classification.

The model operates on already-preprocessed numeric feature data. Missing-value
handling belongs to ``backend.app.ml.preprocessing`` and dataset validation
belongs to ``backend.app.ml.dataset``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite
from typing import Sequence

from backend.app.ml.dataset import MLDataset


class ModelError(ValueError):
    """Raised when model training or prediction cannot be performed."""


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for logistic regression training.

    Attributes:
        learning_rate:
            Gradient-descent step size.

        epochs:
            Number of complete passes over the training dataset.

        l2_regularization:
            L2 regularization strength. The intercept is not regularized.
    """

    learning_rate: float = 0.1
    epochs: int = 1_000
    l2_regularization: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.learning_rate, (int, float)):
            raise TypeError(
                "learning_rate must be numeric"
            )

        if isinstance(self.learning_rate, bool):
            raise TypeError(
                "learning_rate must be numeric"
            )

        if not isfinite(float(self.learning_rate)):
            raise ModelError(
                "learning_rate must be finite"
            )

        if self.learning_rate <= 0:
            raise ModelError(
                "learning_rate must be greater than zero"
            )

        if isinstance(self.epochs, bool) or not isinstance(
            self.epochs,
            int,
        ):
            raise TypeError(
                "epochs must be an integer"
            )

        if self.epochs <= 0:
            raise ModelError(
                "epochs must be greater than zero"
            )

        if isinstance(self.l2_regularization, bool) or not isinstance(
            self.l2_regularization,
            (int, float),
        ):
            raise TypeError(
                "l2_regularization must be numeric"
            )

        if not isfinite(float(self.l2_regularization)):
            raise ModelError(
                "l2_regularization must be finite"
            )

        if self.l2_regularization < 0:
            raise ModelError(
                "l2_regularization must not be negative"
            )


@dataclass(frozen=True)
class LogisticRiskModel:
    """Trained binary logistic regression model.

    Attributes:
        feature_names:
            Ordered feature schema expected during prediction.

        coefficients:
            One coefficient for each feature.

        intercept:
            Bias term of the logistic regression model.
    """

    feature_names: tuple[str, ...]
    coefficients: tuple[float, ...]
    intercept: float

    def __post_init__(self) -> None:
        if not self.feature_names:
            raise ModelError(
                "Model must contain at least one feature"
            )

        if len(self.feature_names) != len(self.coefficients):
            raise ModelError(
                "Number of coefficients must match number of features"
            )

        if len(self.feature_names) != len(set(self.feature_names)):
            raise ModelError(
                "Model feature names must be unique"
            )

        for feature_name in self.feature_names:
            if not isinstance(feature_name, str):
                raise TypeError(
                    "Feature names must be strings"
                )

            if not feature_name.strip():
                raise ModelError(
                    "Feature names must not be empty"
                )

        for coefficient in self.coefficients:
            _validate_finite_number(
                coefficient,
                "Model coefficients must be finite",
            )

        _validate_finite_number(
            self.intercept,
            "Model intercept must be finite",
        )

    @property
    def feature_count(self) -> int:
        """Return the number of features expected by the model."""

        return len(self.feature_names)

    def predict_probability(
        self,
        feature_values: Sequence[float],
    ) -> float:
        """Predict the positive-class probability for one example."""

        if len(feature_values) != self.feature_count:
            raise ModelError(
                f"Expected {self.feature_count} feature values "
                f"but received {len(feature_values)}"
            )

        linear_score = float(self.intercept)

        for coefficient, value in zip(
            self.coefficients,
            feature_values,
        ):
            numeric_value = _validate_finite_number(
                value,
                "Prediction feature values must be finite",
            )

            linear_score += coefficient * numeric_value

        return _sigmoid(linear_score)

    def predict(
        self,
        feature_values: Sequence[float],
        *,
        threshold: float = 0.5,
    ) -> int:
        """Predict a binary risk label.

        A probability greater than or equal to ``threshold`` is classified
        as the positive class.
        """

        _validate_threshold(threshold)

        probability = self.predict_probability(feature_values)

        return int(probability >= threshold)

    def predict_probabilities(
        self,
        feature_rows: Sequence[Sequence[float]],
    ) -> list[float]:
        """Predict positive-class probabilities for multiple examples."""

        return [
            self.predict_probability(row)
            for row in feature_rows
        ]

    def predict_many(
        self,
        feature_rows: Sequence[Sequence[float]],
        *,
        threshold: float = 0.5,
    ) -> list[int]:
        """Predict binary labels for multiple examples."""

        _validate_threshold(threshold)

        return [
            self.predict(
                row,
                threshold=threshold,
            )
            for row in feature_rows
        ]


@dataclass(frozen=True)
class ModelTrainingResult:
    """Result produced by model training.

    Attributes:
        model:
            The trained logistic regression model.

        training_loss:
            Binary cross-entropy loss after the final training epoch.

        epochs_completed:
            Number of optimization epochs completed.
    """

    model: LogisticRiskModel
    training_loss: float
    epochs_completed: int

    def __post_init__(self) -> None:
        _validate_finite_number(
            self.training_loss,
            "training_loss must be finite",
        )

        if self.training_loss < 0:
            raise ModelError(
                "training_loss must not be negative"
            )

        if isinstance(self.epochs_completed, bool) or not isinstance(
            self.epochs_completed,
            int,
        ):
            raise TypeError(
                "epochs_completed must be an integer"
            )

        if self.epochs_completed <= 0:
            raise ModelError(
                "epochs_completed must be greater than zero"
            )


def train_logistic_risk_model(
    dataset: MLDataset,
    *,
    config: TrainingConfig | None = None,
) -> ModelTrainingResult:
    """Train a logistic regression model on a validated ML dataset.

    Training uses batch gradient descent with optional L2 regularization.

    The dataset must contain both positive and negative examples because a
    classifier trained on only one class cannot learn a meaningful decision
    boundary.
    """

    if dataset.positive_count == 0:
        raise ModelError(
            "Training dataset must contain at least one positive example"
        )

    if dataset.negative_count == 0:
        raise ModelError(
            "Training dataset must contain at least one negative example"
        )

    training_config = config or TrainingConfig()

    feature_matrix = dataset.feature_matrix()
    labels = dataset.label_vector()

    row_count = dataset.row_count
    feature_count = dataset.feature_count

    coefficients = [0.0] * feature_count
    intercept = 0.0

    for _ in range(training_config.epochs):
        coefficient_gradients = [0.0] * feature_count
        intercept_gradient = 0.0

        for row, label in zip(feature_matrix, labels):
            linear_score = intercept

            for coefficient, value in zip(
                coefficients,
                row,
            ):
                linear_score += coefficient * value

            probability = _sigmoid(linear_score)
            error = probability - label

            intercept_gradient += error

            for index, value in enumerate(row):
                coefficient_gradients[index] += error * value

        for index in range(feature_count):
            gradient = (
                coefficient_gradients[index] / row_count
            )

            if training_config.l2_regularization > 0:
                gradient += (
                    training_config.l2_regularization
                    * coefficients[index]
                )

            coefficients[index] -= (
                training_config.learning_rate
                * gradient
            )

        intercept -= (
            training_config.learning_rate
            * (intercept_gradient / row_count)
        )

    model = LogisticRiskModel(
        feature_names=dataset.feature_names,
        coefficients=tuple(coefficients),
        intercept=intercept,
    )

    training_loss = _binary_cross_entropy_loss(
        model=model,
        feature_rows=feature_matrix,
        labels=labels,
    )

    return ModelTrainingResult(
        model=model,
        training_loss=training_loss,
        epochs_completed=training_config.epochs,
    )


def _sigmoid(value: float) -> float:
    """Compute a numerically stable sigmoid."""

    if value >= 0:
        exponent = exp(-value)
        return 1.0 / (1.0 + exponent)

    exponent = exp(value)
    return exponent / (1.0 + exponent)


def _binary_cross_entropy_loss(
    *,
    model: LogisticRiskModel,
    feature_rows: Sequence[Sequence[float]],
    labels: Sequence[int],
) -> float:
    """Calculate average binary cross-entropy loss."""

    total_loss = 0.0
    epsilon = 1e-15

    for row, label in zip(feature_rows, labels):
        probability = model.predict_probability(row)

        probability = min(
            max(probability, epsilon),
            1.0 - epsilon,
        )

        if label == 1:
            total_loss -= _natural_log(probability)
        else:
            total_loss -= _natural_log(1.0 - probability)

    return total_loss / len(labels)


def _natural_log(value: float) -> float:
    """Return the natural logarithm."""

    from math import log

    return log(value)


def _validate_finite_number(
    value: object,
    message: str,
) -> float:
    """Validate and normalize a finite numeric value."""

    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise TypeError(message)

    numeric_value = float(value)

    if not isfinite(numeric_value):
        raise ModelError(message)

    return numeric_value


def _validate_threshold(threshold: float) -> None:
    """Validate a binary classification threshold."""

    numeric_threshold = _validate_finite_number(
        threshold,
        "threshold must be a finite number",
    )

    if not 0.0 <= numeric_threshold <= 1.0:
        raise ModelError(
            "threshold must be between 0.0 and 1.0"
        )