"""Unit tests for the Refund Sentinel API."""

from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from backend.app.config import settings

from backend.app.main import app
from backend.app.persistence.database import get_db


def _fake_db() -> Generator[object, None, None]:
    """Provide a placeholder database dependency for API validation tests."""

    yield object()

@pytest.fixture
def client(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[TestClient, None, None]:
    """Create an isolated API test client without an ML model."""

    monkeypatch.setattr(
        settings,
        "app_api_key",
        "",
    )

    # Explicitly disable ML model loading for these unit tests.
    # The test below verifies that the application can initialize
    # successfully when no ML model is configured.
    monkeypatch.setattr(
        settings,
        "ml_model_path",
        "",
    )

    app.dependency_overrides[get_db] = _fake_db

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()


def test_application_initializes_without_ml_model(
    client: TestClient,
) -> None:
    """Application should start when no ML model is configured."""

    assert (
        client.app.state.ml_inference_service
        is None
    )


def test_health_endpoint_returns_ok(
    client: TestClient,
) -> None:
    """Health endpoint reports that the application is running."""

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json() == {
        "status": "ok",
    }


def test_invalid_refund_id_returns_bad_request(
    client: TestClient,
) -> None:
    """Assessment endpoint rejects malformed refund identifiers."""

    response = client.get(
        "/api/v1/assessments/not-a-valid-uuid"
    )

    assert response.status_code == 400

    assert response.json() == {
        "detail": "Invalid refund ID format",
    }


def test_invalid_investigation_refund_id_returns_bad_request(
    client: TestClient,
) -> None:
    """Investigation endpoint rejects malformed refund identifiers."""

    response = client.get(
        "/api/v1/investigations/not-a-valid-uuid"
    )

    assert response.status_code == 400

    assert response.json() == {
        "detail": "Invalid refund ID format",
    }