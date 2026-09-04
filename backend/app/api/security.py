"""API security dependencies for Refund Sentinel."""

from __future__ import annotations

from fastapi import (
    Header,
    HTTPException,
    status,
)

from backend.app.config import settings


def require_api_key(
    x_api_key: str | None = Header(
        default=None,
        alias="X-API-Key",
    ),
) -> None:
    """Validate the configured API key.

    Authentication is disabled when no application API key is configured.
    This allows local development without requiring authentication while
    enabling protection in deployed environments.

    Raises:
        HTTPException:
            If API authentication is enabled and the supplied key is missing
            or invalid.
    """

    configured_api_key = settings.app_api_key.strip()

    if not configured_api_key:
        return

    if x_api_key != configured_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )