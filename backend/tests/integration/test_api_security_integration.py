"""Integration tests for Refund Sentinel API authentication."""

from __future__ import annotations

from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from backend.app.config import settings
from backend.app.main import app
from backend.app.persistence.database import get_db


def _fake_db() -> Generator[object, None, None]:
    """Provide a placeholder database dependency."""

    yield object()


@pytest.fixture
def authenticated_client(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[TestClient, None, None]:
    """Create an isolated client with API authentication enabled."""

    monkeypatch.setattr(
        settings,
        "app_api_key",
        "test-secret-key",
    )

    app.dependency_overrides[get_db] = _fake_db

    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()


def test_health_endpoint_remains_public(
    authenticated_client: TestClient,
) -> None:
    """Health endpoint should not require an API key."""

    response = authenticated_client.get(
        "/health"
    )

    assert response.status_code == 200

    assert response.json() == {
        "status": "ok",
    }


def test_protected_assessment_endpoint_rejects_missing_key(
    authenticated_client: TestClient,
) -> None:
    """Protected assessment endpoint should reject missing API keys."""

    response = authenticated_client.get(
        "/api/v1/assessments/not-a-valid-uuid"
    )

    assert response.status_code == 401

    assert response.json() == {
        "detail": "Invalid or missing API key",
    }


def test_protected_investigation_endpoint_rejects_missing_key(
    authenticated_client: TestClient,
) -> None:
    """Protected investigation endpoint should reject missing API keys."""

    response = authenticated_client.get(
        "/api/v1/investigations/not-a-valid-uuid"
    )

    assert response.status_code == 401

    assert response.json() == {
        "detail": "Invalid or missing API key",
    }


def test_protected_endpoint_rejects_invalid_key(
    authenticated_client: TestClient,
) -> None:
    """Protected endpoint should reject an incorrect API key."""

    response = authenticated_client.get(
        "/api/v1/assessments/not-a-valid-uuid",
        headers={
            "X-API-Key": "wrong-key",
        },
    )

    assert response.status_code == 401

    assert response.json() == {
        "detail": "Invalid or missing API key",
    }


def test_valid_api_key_allows_request_to_reach_route(
    authenticated_client: TestClient,
) -> None:
    """Valid API key should allow the request past authentication."""

    response = authenticated_client.get(
        "/api/v1/assessments/not-a-valid-uuid",
        headers={
            "X-API-Key": "test-secret-key",
        },
    )

    assert response.status_code == 400

    assert response.json() == {
        "detail": "Invalid refund ID format",
    }