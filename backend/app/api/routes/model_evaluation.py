"""Runtime model status and evaluation metadata routes."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, Request

from backend.app.api.schemas import ModelEvaluationResponse
from backend.app.api.security import require_api_key
from backend.app.config import settings


router = APIRouter(
    prefix="/api/v1/model-evaluation",
    tags=["model"],
    dependencies=[Depends(require_api_key)],
)


@router.get(
    "",
    response_model=ModelEvaluationResponse,
)
def get_model_evaluation(request: Request) -> ModelEvaluationResponse:
    """Expose only model metadata that is genuinely available at runtime."""

    model_service = request.app.state.ml_inference_service
    if model_service is None:
        return ModelEvaluationResponse(
            model_available=False,
            status="unavailable",
            evaluation_metrics_available=False,
            data_note=(
                "No ML model is configured. Deterministic risk scoring "
                "remains available."
            ),
        )

    artifact_path = Path(settings.ml_model_path)
    artifact_version: int | None = None
    feature_count: int | None = None
    persisted_metrics: dict[str, float] = {}

    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        artifact_version = payload.get("artifact_version")
        feature_names = payload.get("model", {}).get("feature_names", [])
        feature_count = len(feature_names) if isinstance(feature_names, list) else None
        training_metadata = payload.get("training_metadata", {})
        candidate_metrics = (
            training_metadata.get("metrics", {})
            if isinstance(training_metadata, dict)
            else {}
        )
        if isinstance(candidate_metrics, dict):
            persisted_metrics = {
                str(key): float(value)
                for key, value in candidate_metrics.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
    except (OSError, ValueError, TypeError, AttributeError):
        # The runtime service has already validated the model for inference.
        # Metadata is optional and must not make the status page fail.
        pass

    return ModelEvaluationResponse(
        model_available=True,
        status="loaded",
        artifact_version=artifact_version,
        feature_count=feature_count,
        evaluation_metrics_available=bool(persisted_metrics),
        metrics=persisted_metrics,
        data_note=(
            "This model is an analytical risk signal trained from the project's "
            "available dataset; it is not fraud ground truth."
            if not persisted_metrics
            else "Evaluation metadata persisted with the deployed artifact."
        ),
    )