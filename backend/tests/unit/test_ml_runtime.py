"""Unit tests for application-level ML runtime construction."""

from __future__ import annotations

import pytest

from backend.app.config import Settings
from backend.app.ml.inference import MLInferenceService
from backend.app.ml.model import LogisticRiskModel
from backend.app.ml.persistence import (
    ModelPersistenceError,
    PersistedModelBundle,
    save_model_bundle,
)
from backend.app.ml.preprocessing import MLPreprocessor
from backend.app.ml.runtime import (
    create_ml_inference_service,
)


def _create_settings(
    **overrides,
) -> Settings:
    """Create test application settings."""

    values = {
        "database_url": "postgresql://test",
    }

    values.update(overrides)

    return Settings(**values)


def _create_bundle() -> PersistedModelBundle:
    """Create a minimal valid model bundle."""

    model = LogisticRiskModel(
        feature_names=(
            "feature_a",
            "feature_b",
        ),
        coefficients=(
            1.0,
            -0.5,
        ),
        intercept=0.0,
    )

    preprocessor = MLPreprocessor(
        feature_names=(
            "feature_a",
            "feature_b",
        ),
        replacement_values=(
            0.0,
            0.0,
        ),
    )

    return PersistedModelBundle(
        model=model,
        preprocessor=preprocessor,
    )


def test_runtime_returns_none_without_model_path() -> None:
    """ML inference should remain optional."""

    settings = _create_settings(
        ml_model_path="",
    )

    service = create_ml_inference_service(
        settings
    )

    assert service is None


def test_runtime_ignores_whitespace_model_path() -> None:
    """Whitespace-only paths should be treated as unconfigured."""

    settings = _create_settings(
        ml_model_path="   ",
    )

    service = create_ml_inference_service(
        settings
    )

    assert service is None


def test_runtime_loads_configured_model(
    tmp_path,
) -> None:
    """A configured model artifact should create inference."""

    artifact_path = (
        tmp_path / "model.json"
    )

    save_model_bundle(
        _create_bundle(),
        artifact_path,
    )

    settings = _create_settings(
        ml_model_path=str(artifact_path),
    )

    service = create_ml_inference_service(
        settings
    )

    assert isinstance(
        service,
        MLInferenceService,
    )

    assert service.classification_threshold == 0.5


def test_runtime_uses_configured_threshold(
    tmp_path,
) -> None:
    """The configured classification threshold should be used."""

    artifact_path = (
        tmp_path / "model.json"
    )

    save_model_bundle(
        _create_bundle(),
        artifact_path,
    )

    settings = _create_settings(
        ml_model_path=str(artifact_path),
        ml_classification_threshold=0.8,
    )

    service = create_ml_inference_service(
        settings
    )

    assert service is not None

    assert service.classification_threshold == 0.8


def test_runtime_rejects_missing_configured_model(
    tmp_path,
) -> None:
    """A configured missing artifact should fail clearly."""

    settings = _create_settings(
        ml_model_path=str(
            tmp_path / "missing.json"
        ),
    )

    with pytest.raises(
        ModelPersistenceError,
        match="does not exist",
    ):
        create_ml_inference_service(
            settings
        )


def test_runtime_rejects_invalid_settings() -> None:
    """Runtime construction requires Settings."""

    with pytest.raises(
        TypeError,
        match="settings must be a Settings instance",
    ):
        create_ml_inference_service(
            object()
        )