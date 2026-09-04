"""Unit tests for Refund Sentinel API security."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from backend.app.api.security import require_api_key
from backend.app.config import settings


def test_api_key_authentication_is_disabled_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Requests should be allowed when no API key is configured."""

    monkeypatch.setattr(
        settings,
        "app_api_key",
        "",
    )

    require_api_key(
        x_api_key=None,
    )


def test_matching_api_key_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configured API key should be accepted."""

    monkeypatch.setattr(
        settings,
        "app_api_key",
        "secret-key",
    )

    require_api_key(
        x_api_key="secret-key",
    )


def test_missing_api_key_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing API key should be rejected when authentication is enabled."""

    monkeypatch.setattr(
        settings,
        "app_api_key",
        "secret-key",
    )

    with pytest.raises(
        HTTPException,
    ) as exc_info:
        require_api_key(
            x_api_key=None,
        )

    assert exc_info.value.status_code == 401

    assert exc_info.value.detail == (
        "Invalid or missing API key"
    )


def test_invalid_api_key_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Incorrect API key should be rejected."""

    monkeypatch.setattr(
        settings,
        "app_api_key",
        "secret-key",
    )

    with pytest.raises(
        HTTPException,
    ) as exc_info:
        require_api_key(
            x_api_key="wrong-key",
        )

    assert exc_info.value.status_code == 401

    assert exc_info.value.detail == (
        "Invalid or missing API key"
    )