"""Unit tests for ML inference."""

from __future__ import annotations

import pytest

from backend.app.ml.inference import (
    MLInferenceService,
    MLPrediction,
)
from backend.app.ml.model import LogisticRiskModel
from backend.app.ml.persistence import (
    PersistedModelBundle,
)
from backend.app.ml.preprocessing import MLPreprocessor
from backend.app.risk.assessment import RiskAssessment


def _create_bundle(
    feature_names: tuple[str, ...],
) -> PersistedModelBundle:
    """Create a model bundle matching the supplied feature schema."""

    model = LogisticRiskModel(
        feature_names=feature_names,
        coefficients=tuple(
            0.1
            for _ in feature_names
        ),
        intercept=0.0,
    )

    preprocessor = MLPreprocessor(
        feature_names=feature_names,
        replacement_values=tuple(
            0.0
            for _ in feature_names
        ),
    )

    return PersistedModelBundle(
        model=model,
        preprocessor=preprocessor,
    )


def _create_matching_service(
    assessment: RiskAssessment,
) -> MLInferenceService:
    """Create a service whose model matches an assessment feature schema."""

    from backend.app.ml.features import (
        build_feature_vector,
    )

    feature_vector = build_feature_vector(
        assessment
    )

    bundle = _create_bundle(
        feature_vector.feature_names
    )

    return MLInferenceService(bundle)


def test_prediction_requires_valid_probability() -> None:
    """Predictions reject probabilities outside [0, 1]."""

    with pytest.raises(
        ValueError,
        match="between 0.0 and 1.0",
    ):
        MLPrediction(
            probability=1.1,
            is_high_risk=True,
        )


def test_prediction_rejects_non_boolean_classification() -> None:
    """Predictions require an explicit boolean classification."""

    with pytest.raises(
        TypeError,
        match="is_high_risk must be a boolean",
    ):
        MLPrediction(
            probability=0.5,
            is_high_risk=1,
        )


def test_service_rejects_invalid_bundle() -> None:
    """Inference requires a persisted model bundle."""

    with pytest.raises(
        TypeError,
        match="PersistedModelBundle",
    ):
        MLInferenceService(
            "not a bundle",
        )


@pytest.mark.parametrize(
    "threshold",
    [
        -0.1,
        1.1,
    ],
)
def test_service_rejects_invalid_threshold(
    threshold: float,
) -> None:
    """Threshold must remain inside [0, 1]."""

    bundle = _create_bundle(
        ("feature",)
    )

    with pytest.raises(
        ValueError,
        match="between 0.0 and 1.0",
    ):
        MLInferenceService(
            bundle,
            classification_threshold=threshold,
        )


def test_service_rejects_boolean_threshold() -> None:
    """Boolean values must not be treated as numeric thresholds."""

    bundle = _create_bundle(
        ("feature",)
    )

    with pytest.raises(
        TypeError,
        match="classification_threshold must be numeric",
    ):
        MLInferenceService(
            bundle,
            classification_threshold=True,
        )


def test_predict_rejects_invalid_assessment() -> None:
    """Inference requires a RiskAssessment."""

    bundle = _create_bundle(
        ("feature",)
    )

    service = MLInferenceService(
        bundle
    )

    with pytest.raises(
        TypeError,
        match="RiskAssessment",
    ):
        service.predict(
            "not an assessment"
        )


def test_prediction_classification_uses_threshold(
    monkeypatch,
) -> None:
    """Predictions should classify using the configured threshold."""

    from backend.app.ml import inference

    model = LogisticRiskModel(
        feature_names=("feature_a",),
        coefficients=(0.0,),
        intercept=0.0,
    )

    preprocessor = MLPreprocessor(
        feature_names=("feature_a",),
        replacement_values=(0.0,),
    )

    bundle = PersistedModelBundle(
        model=model,
        preprocessor=preprocessor,
    )

    service = MLInferenceService(
        bundle,
        classification_threshold=0.5,
    )

    monkeypatch.setattr(
        inference,
        "build_feature_vector",
        lambda assessment: type(
            "FeatureVector",
            (),
            {
                "feature_names": (
                    "feature_a",
                ),
                "values": (
                    0.5,
                ),
            },
        )(),
    )

    prediction = service.predict(
        object.__new__(
            RiskAssessment
        )
    )

    assert prediction.probability == pytest.approx(
        0.5
    )

    assert prediction.is_high_risk is True


def test_feature_schema_mismatch_is_rejected(
    monkeypatch,
) -> None:
    """Inference must reject features from a different schema."""

    from backend.app.ml import inference

    bundle = _create_bundle(
        ("expected_feature",)
    )

    service = MLInferenceService(
        bundle
    )

    monkeypatch.setattr(
        inference,
        "build_feature_vector",
        lambda assessment: type(
            "FeatureVector",
            (),
            {
                "feature_names": (
                    "different_feature",
                ),
                "values": (
                    0.5,
                ),
            },
        )(),
    )

    with pytest.raises(
        ValueError,
        match="feature schema",
    ):
        service.predict(
            object.__new__(
                RiskAssessment
            )
        )