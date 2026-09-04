"""Refund Sentinel FastAPI application."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes.assessments import (
    router as assessments_router,
)
from backend.app.api.routes.health import (
    router as health_router,
)
from backend.app.api.routes.investigations import (
    router as investigations_router,
)
from backend.app.api.routes.model_evaluation import (
    router as model_evaluation_router,
)
from backend.app.config import settings
from backend.app.ml.runtime import (
    create_ml_inference_service,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize application-wide runtime services."""

    app.state.ml_inference_service = (
        create_ml_inference_service(settings)
    )

    yield

    app.state.ml_inference_service = None


app = FastAPI(
    title="Refund Sentinel API",
    version="0.1.0",
    description=(
        "Explainable refund-risk assessment "
        "and investigation API."
    ),
    lifespan=lifespan,
)

# Configure CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5000",
        "http://127.0.0.1:5000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)

app.include_router(assessments_router)

app.include_router(model_evaluation_router)

app.include_router(investigations_router)