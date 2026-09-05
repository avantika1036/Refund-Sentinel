"""Runtime inference for Refund Sentinel ML risk predictions.

Transforms a RiskAssessment into the ML feature representation, applies the
fitted preprocessor, and produces a prediction from a trained model.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isnan

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


@dataclass(frozen=True)
class MLFeatureContribution:
    """Signed local contribution of one standardized feature to the ML logit."""

    feature_name: str
    raw_value: float
    contribution: float


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

    def explain_features(
        self,
        assessment: RiskAssessment,
        *,
        limit: int = 5,
    ) -> list[MLFeatureContribution]:
        """Return top local linear-model contributions for one assessment.

        Contributions are coefficient × standardized feature value and are
        therefore evidence about this exact prediction, not global importance.
        """
        if limit <= 0:
            return []
        vector = build_feature_vector(assessment)
        self._validate_feature_schema(vector.feature_names)
        processed = self._bundle.preprocessor.transform(
            feature_names=vector.feature_names,
            feature_rows=[vector.values],
        )[0]
        contributions = [
            MLFeatureContribution(
                feature_name=name,
                raw_value=(0.0 if isinstance(raw, float) and isnan(raw) else float(raw)),
                contribution=float(coefficient * value),
            )
            for name, raw, value, coefficient in zip(
                vector.feature_names,
                vector.values,
                processed,
                self._bundle.model.coefficients,
            )
        ]
        return sorted(
            contributions,
            key=lambda item: abs(item.contribution),
            reverse=True,
        )[:limit]

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