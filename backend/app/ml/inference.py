"""Runtime inference for Refund Sentinel ML risk predictions.

Transforms a RiskAssessment into the ML feature representation, applies the
fitted preprocessor, and produces a prediction from a trained model.
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.app.ml.features import build_feature_vector
from backend.app.ml.persistence import PersistedModelBundle
from backend.app.risk.assessment import RiskAssessment


@dataclass(frozen=True)
class MLPrediction:
    """Result produced by the ML inference pipeline.

    Attributes:
        probability:
            Predicted probability in the inclusive range [0.0, 1.0].

        is_high_risk:
            Whether the probability meets or exceeds the configured
            classification threshold.
    """

    probability: float
    is_high_risk: bool

    def __post_init__(self) -> None:
        if not isinstance(self.probability, (int, float)):
            raise TypeError(
                "probability must be numeric"
            )

        probability = float(self.probability)

        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                "probability must be between 0.0 and 1.0"
            )

        if not isinstance(self.is_high_risk, bool):
            raise TypeError(
                "is_high_risk must be a boolean"
            )

        object.__setattr__(
            self,
            "probability",
            probability,
        )


class MLInferenceService:
    """Perform ML inference for risk assessments.

    The service owns no training state. It receives a validated persisted
    model bundle containing both the trained model and the exact preprocessor
    required by that model.

    The deterministic RiskAssessment remains the source of explainable risk
    evidence. This service produces a separate learned probability signal.
    """

    def __init__(
        self,
        bundle: PersistedModelBundle,
        *,
        classification_threshold: float = 0.5,
    ) -> None:
        if not isinstance(
            bundle,
            PersistedModelBundle,
        ):
            raise TypeError(
                "bundle must be a PersistedModelBundle"
            )

        self._validate_threshold(
            classification_threshold
        )

        self._bundle = bundle
        self._classification_threshold = float(
            classification_threshold
        )

    @property
    def classification_threshold(self) -> float:
        """Return the probability threshold used for classification."""

        return self._classification_threshold

    def predict(
        self,
        assessment: RiskAssessment,
    ) -> MLPrediction:
        """Produce an ML prediction for one risk assessment.

        Args:
            assessment:
                Existing deterministic risk assessment.

        Returns:
            An MLPrediction containing probability and classification.
        """

        if not isinstance(
            assessment,
            RiskAssessment,
        ):
            raise TypeError(
                "assessment must be a RiskAssessment"
            )

        feature_vector = build_feature_vector(
            assessment
        )

        self._validate_feature_schema(
            feature_vector.feature_names
        )

        processed_rows = (
            self._bundle.preprocessor.transform(
                feature_names=feature_vector.feature_names,
                feature_rows=[
                    feature_vector.values,
                ],
            )
        )

        processed_features = processed_rows[0]

        probability = (
            self._bundle.model.predict_probability(
                processed_features
            )
        )

        return MLPrediction(
            probability=probability,
            is_high_risk=(
                probability
                >= self._classification_threshold
            ),
        )

    def _validate_feature_schema(
        self,
        feature_names: tuple[str, ...],
    ) -> None:
        """Ensure inference features match the trained model schema."""

        expected_feature_names = (
            self._bundle.model.feature_names
        )

        if feature_names != expected_feature_names:
            raise ValueError(
                "Inference feature schema does not match "
                "the trained model schema"
            )

    @staticmethod
    def _validate_threshold(
        threshold: float,
    ) -> None:
        """Validate the classification threshold."""

        if isinstance(threshold, bool) or not isinstance(
            threshold,
            (int, float),
        ):
            raise TypeError(
                "classification_threshold must be numeric"
            )

        numeric_threshold = float(threshold)

        if not 0.0 <= numeric_threshold <= 1.0:
            raise ValueError(
                "classification_threshold must be between "
                "0.0 and 1.0"
            )