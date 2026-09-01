"""Refund Sentinel FastAPI application."""

from fastapi import FastAPI

from backend.app.api.routes.assessments import (
    router as assessments_router,
)
from backend.app.api.routes.health import (
    router as health_router,
)


app = FastAPI(
    title="Refund Sentinel API",
    version="0.1.0",
    description=(
        "Explainable refund-risk assessment "
        "and investigation API."
    ),
)

app.include_router(health_router)

app.include_router(assessments_router)