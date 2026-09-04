"""Integration tests for ML-enabled investigation API behavior.

Verifies the production-style path:

Persisted model artifact
    -> FastAPI application startup
    -> ML runtime initialization
    -> persisted events
    -> reconstruction
    -> investigation
    -> ML inference
    -> API response
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete

from backend.app.config import settings
from backend.app.main import app
from backend.app.ml.features import build_feature_vector
from backend.app.ml.model import LogisticRiskModel
from backend.app.ml.persistence import (
    PersistedModelBundle,
    save_model_bundle,
)
from backend.app.ml.preprocessing import fit_preprocessor
from backend.app.persistence.database import SessionLocal
from backend.app.persistence.ingestion_service import IngestionService
from backend.app.persistence.models import (
    EventModel,
    IngestionRecordModel,
)
from backend.app.persistence.reconstruction import (
    ReconstructionService,
)
from backend.app.risk.assessment import RiskAssessor
from backend.app.simulator.scenarios import (
    AS01_DENSE_COORDINATED_REFUND_RING,
)


def _cleanup_database() -> None:
    """Remove data created by the integration tests."""

    with SessionLocal() as session:
        session.execute(delete(EventModel))
        session.execute(delete(IngestionRecordModel))
        session.commit()


def _persist_scenario_events():
    """Generate and persist a deterministic scenario.

    Returns:
        A refund identifier belonging to the generated scenario.
    """

    generator = AS01_DENSE_COORDINATED_REFUND_RING(
        seed=42,
    )

    output = generator.generate(
        num_customers=3,
        orders_per_customer=2,
    )

    ingestion_service = IngestionService()

    for event in output.events:
        ingestion_service.ingest(event)

    abuse_labels = [
        label
        for label in output.get_abuse_labels()
        if label.refund_id is not None
    ]

    assert abuse_labels, (
        "AS01 scenario should generate at least "
        "one labeled refund"
    )

    target_refund_id = abuse_labels[0].refund_id

    assert target_refund_id is not None

    return target_refund_id


def _create_model_artifact(
    *,
    artifact_path: Path,
    refund_id,
) -> None:
    """Create a real persisted model matching the refund feature schema."""

    with SessionLocal() as session:
        snapshot = ReconstructionService(
            session
        ).reconstruct()

    assessment = RiskAssessor(
        snapshot
    ).assess(refund_id)

    feature_vector = build_feature_vector(
        assessment
    )

    preprocessor = fit_preprocessor(
        feature_names=feature_vector.feature_names,
        feature_rows=[
            feature_vector.values,
        ],
    )

    model = LogisticRiskModel(
        feature_names=feature_vector.feature_names,
        coefficients=tuple(
            0.0
            for _ in feature_vector.feature_names
        ),
        intercept=0.0,
    )

    bundle = PersistedModelBundle(
        model=model,
        preprocessor=preprocessor,
    )

    save_model_bundle(
        bundle,
        artifact_path,
    )


@pytest.mark.integration
def test_investigation_api_returns_ml_prediction(
    tmp_path,
    monkeypatch,
) -> None:
    """Configured ML model should produce an API prediction."""

    _cleanup_database()

    try:
        target_refund_id = _persist_scenario_events()

        artifact_path = (
            tmp_path / "refund_sentinel_model.json"
        )

        _create_model_artifact(
            artifact_path=artifact_path,
            refund_id=target_refund_id,
        )

        monkeypatch.setattr(
            settings,
            "ml_model_path",
            str(artifact_path),
        )

        monkeypatch.setattr(
            settings,
            "ml_classification_threshold",
            0.5,
        )

        monkeypatch.setattr(
            settings,
            "app_api_key",
            "",
        )

        with TestClient(app) as client:
            response = client.get(
                "/api/v1/investigations/"
                f"{target_refund_id}"
            )

        assert response.status_code == 200

        payload = response.json()

        assert payload["ml_prediction"] is not None

        prediction = payload["ml_prediction"]

        assert prediction["probability"] == 0.5

        assert prediction["is_high_risk"] is True

    finally:
        _cleanup_database()


@pytest.mark.integration
def test_investigation_api_returns_no_ml_prediction_without_model(
    monkeypatch,
) -> None:
    """Application should remain functional when ML is disabled."""

    _cleanup_database()

    try:
        target_refund_id = _persist_scenario_events()

        monkeypatch.setattr(
            settings,
            "ml_model_path",
            "",
        )

        monkeypatch.setattr(
            settings,
            "app_api_key",
            "",
        )

        with TestClient(app) as client:
            response = client.get(
                "/api/v1/investigations/"
                f"{target_refund_id}"
            )

        assert response.status_code == 200

        payload = response.json()

        assert payload["ml_prediction"] is None

    finally:
        _cleanup_database()