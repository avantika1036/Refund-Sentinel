"""Unit tests for ML model persistence."""

from __future__ import annotations

import json

import pytest

from backend.app.ml.model import LogisticRiskModel
from backend.app.ml.persistence import (
    ModelPersistenceError,
    PersistedModelBundle,
    load_model_bundle,
    save_model_bundle,
)
from backend.app.ml.preprocessing import MLPreprocessor


def _create_bundle() -> PersistedModelBundle:
    """Create a representative model bundle for testing."""

    model = LogisticRiskModel(
        feature_names=(
            "risk_signal",
            "velocity_signal",
        ),
        coefficients=(
            1.25,
            -0.75,
        ),
        intercept=0.2,
    )

    preprocessor = MLPreprocessor(
        feature_names=(
            "risk_signal",
            "velocity_signal",
        ),
        replacement_values=(
            0.5,
            0.25,
        ),
    )

    return PersistedModelBundle(
        model=model,
        preprocessor=preprocessor,
    )


def test_save_and_load_model_bundle_round_trip(
    tmp_path,
) -> None:
    """A saved bundle should load back unchanged."""

    original_bundle = _create_bundle()

    artifact_path = tmp_path / "risk_model.json"

    save_model_bundle(
        original_bundle,
        artifact_path,
    )

    loaded_bundle = load_model_bundle(
        artifact_path
    )

    assert loaded_bundle == original_bundle


def test_save_creates_parent_directories(
    tmp_path,
) -> None:
    """Saving should create missing artifact directories."""

    artifact_path = (
        tmp_path
        / "models"
        / "production"
        / "risk_model.json"
    )

    saved_path = save_model_bundle(
        _create_bundle(),
        artifact_path,
    )

    assert saved_path == artifact_path
    assert artifact_path.exists()


def test_saved_artifact_is_valid_json(
    tmp_path,
) -> None:
    """Persisted artifacts should remain inspectable JSON."""

    artifact_path = tmp_path / "model.json"

    save_model_bundle(
        _create_bundle(),
        artifact_path,
    )

    payload = json.loads(
        artifact_path.read_text(
            encoding="utf-8"
        )
    )

    assert payload["artifact_version"] == 2

    assert payload["model"]["feature_names"] == [
        "risk_signal",
        "velocity_signal",
    ]

    assert payload["preprocessor"][
        "replacement_values"
    ] == [
        0.5,
        0.25,
    ]


def test_loaded_model_can_predict(
    tmp_path,
) -> None:
    """A loaded artifact should remain usable for prediction."""

    artifact_path = tmp_path / "model.json"

    original_bundle = _create_bundle()

    save_model_bundle(
        original_bundle,
        artifact_path,
    )

    loaded_bundle = load_model_bundle(
        artifact_path
    )

    original_probability = (
        original_bundle.model.predict_probability(
            [0.5, 0.25]
        )
    )

    loaded_probability = (
        loaded_bundle.model.predict_probability(
            [0.5, 0.25]
        )
    )

    assert loaded_probability == pytest.approx(
        original_probability
    )


def test_bundle_rejects_mismatched_schemas() -> None:
    """Model and preprocessor schemas must match."""

    model = LogisticRiskModel(
        feature_names=("feature_a",),
        coefficients=(1.0,),
        intercept=0.0,
    )

    preprocessor = MLPreprocessor(
        feature_names=("feature_b",),
        replacement_values=(0.0,),
    )

    with pytest.raises(
        ModelPersistenceError,
        match="feature schemas must match",
    ):
        PersistedModelBundle(
            model=model,
            preprocessor=preprocessor,
        )


def test_save_rejects_invalid_bundle(
    tmp_path,
) -> None:
    """Saving requires a validated model bundle."""

    with pytest.raises(
        TypeError,
        match="PersistedModelBundle",
    ):
        save_model_bundle(
            "not a bundle",
            tmp_path / "model.json",
        )


def test_save_rejects_empty_path() -> None:
    """Artifact paths must not be empty."""

    with pytest.raises(
        ModelPersistenceError,
        match="must not be empty",
    ):
        save_model_bundle(
            _create_bundle(),
            "",
        )


def test_load_rejects_missing_artifact(
    tmp_path,
) -> None:
    """Loading a missing artifact should fail clearly."""

    artifact_path = (
        tmp_path / "missing.json"
    )

    with pytest.raises(
        ModelPersistenceError,
        match="does not exist",
    ):
        load_model_bundle(
            artifact_path
        )


def test_load_rejects_invalid_json(
    tmp_path,
) -> None:
    """Malformed JSON artifacts should not be loaded."""

    artifact_path = tmp_path / "broken.json"

    artifact_path.write_text(
        "{not valid json",
        encoding="utf-8",
    )

    with pytest.raises(
        ModelPersistenceError,
        match="invalid JSON",
    ):
        load_model_bundle(
            artifact_path
        )


def test_load_rejects_unsupported_version(
    tmp_path,
) -> None:
    """Unknown artifact versions must fail safely."""

    artifact_path = tmp_path / "future.json"

    artifact_path.write_text(
        json.dumps(
            {
                "artifact_version": 999,
                "model": {
                    "feature_names": [
                        "feature",
                    ],
                    "coefficients": [
                        1.0,
                    ],
                    "intercept": 0.0,
                },
                "preprocessor": {
                    "feature_names": [
                        "feature",
                    ],
                    "replacement_values": [
                        0.0,
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ModelPersistenceError,
        match="Unsupported artifact version",
    ):
        load_model_bundle(
            artifact_path
        )


def test_load_rejects_missing_required_keys(
    tmp_path,
) -> None:
    """Incomplete artifacts should not be accepted."""

    artifact_path = tmp_path / "incomplete.json"

    artifact_path.write_text(
        json.dumps(
            {
                "artifact_version": 1,
                "model": {},
                "preprocessor": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ModelPersistenceError,
        match="missing required keys",
    ):
        load_model_bundle(
            artifact_path
        )


def test_load_rejects_unexpected_keys(
    tmp_path,
) -> None:
    """Unexpected artifact fields should fail schema validation."""

    artifact_path = tmp_path / "unexpected.json"

    artifact_path.write_text(
        json.dumps(
            {
                "artifact_version": 1,
                "model": {
                    "feature_names": [
                        "feature",
                    ],
                    "coefficients": [
                        1.0,
                    ],
                    "intercept": 0.0,
                    "unexpected": "value",
                },
                "preprocessor": {
                    "feature_names": [
                        "feature",
                    ],
                    "replacement_values": [
                        0.0,
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ModelPersistenceError,
        match="unexpected keys",
    ):
        load_model_bundle(
            artifact_path
        )


def test_load_rejects_non_matching_schemas(
    tmp_path,
) -> None:
    """Persisted model and preprocessor schemas must match."""

    artifact_path = tmp_path / "mismatch.json"

    artifact_path.write_text(
        json.dumps(
            {
                "artifact_version": 1,
                "model": {
                    "feature_names": [
                        "feature_a",
                    ],
                    "coefficients": [
                        1.0,
                    ],
                    "intercept": 0.0,
                },
                "preprocessor": {
                    "feature_names": [
                        "feature_b",
                    ],
                    "replacement_values": [
                        0.0,
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ModelPersistenceError,
        match="schemas",
    ):
        load_model_bundle(
            artifact_path
        )


def test_load_rejects_non_finite_numbers(
    tmp_path,
) -> None:
    """Non-finite numeric model values must not be accepted."""

    artifact_path = tmp_path / "invalid_number.json"

    artifact_path.write_text(
        json.dumps(
            {
                "artifact_version": 1,
                "model": {
                    "feature_names": [
                        "feature",
                    ],
                    "coefficients": [
                        1.0,
                    ],
                    "intercept": "NaN",
                },
                "preprocessor": {
                    "feature_names": [
                        "feature",
                    ],
                    "replacement_values": [
                        0.0,
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ModelPersistenceError,
        match="must be numeric",
    ):
        load_model_bundle(
            artifact_path
        )