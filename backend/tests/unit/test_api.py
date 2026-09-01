"""Unit tests for the Refund Sentinel API."""

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.persistence.database import get_db


def _fake_db():
    """Provide a placeholder database dependency for API validation tests."""

    yield object()


app.dependency_overrides[get_db] = _fake_db

client = TestClient(app)


def test_health_endpoint_returns_ok() -> None:
    """Health endpoint reports that the application is running."""

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json() == {
        "status": "ok",
    }


def test_invalid_refund_id_returns_bad_request() -> None:
    """Assessment endpoint rejects malformed refund identifiers."""

    response = client.get(
        "/api/v1/assessments/not-a-valid-uuid"
    )

    assert response.status_code == 400

    assert response.json() == {
        "detail": "Invalid refund ID format",
    }