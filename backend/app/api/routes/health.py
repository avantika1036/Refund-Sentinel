"""Health check API routes."""

from fastapi import APIRouter

from backend.app.api.schemas import HealthResponse


router = APIRouter(
    tags=["system"],
)


@router.get(
    "/health",
    response_model=HealthResponse,
)
def health_check() -> HealthResponse:
    """Return application health status."""

    return HealthResponse()