"""Runtime model status and comparative evaluation metadata routes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Request

from backend.app.api.schemas import (
    ComparativeBaselineMetricsResponse,
    ModelEvaluationResponse,
)
from backend.app.api.security import require_api_key
from backend.app.config import settings


router = APIRouter(
    prefix="/api/v1/model-evaluation",
    tags=["model"],
    dependencies=[Depends(require_api_key)],
)

_PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else _PROJECT_ROOT / path


def _load_benchmark_results() -> tuple[
    dict[str, ComparativeBaselineMetricsResponse],
    dict[str, object],
]:
    """Load only validated held-out benchmark data if results are available."""
    path = _resolve_project_path(settings.evaluation_results_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_summary = payload.get("summary", {})
        raw_protocol = payload.get("evaluation_protocol", {})
        if not isinstance(raw_summary, dict):
            return {}, {}
        benchmark_summary: dict[str, ComparativeBaselineMetricsResponse] = {}
        for name, metrics in raw_summary.items():
            if not isinstance(metrics, dict):
                continue
            benchmark_summary[str(name)] = ComparativeBaselineMetricsResponse(
                precision=metrics.get("precision", 0.0),
                recall=metrics.get("recall", 0.0),
                f1_score=metrics.get("f1_score", 0.0),
                accuracy=metrics.get("accuracy", 0.0),
                true_positive=metrics.get("true_positive", 0),
                true_negative=metrics.get("true_negative", 0),
                false_positive=metrics.get("false_positive", 0),
                false_negative=metrics.get("false_negative", 0),
                review_volume=metrics.get("review_volume", 0),
                loss_prevented_inr=metrics.get("loss_prevented_inr", 0.0),
                false_positive_flagged_amount_inr=metrics.get(
                    "false_positive_flagged_amount_inr",
                    0.0,
                ),
                total_flagged_amount_inr=metrics.get(
                    "total_flagged_amount_inr",
                    0.0,
                ),
                operating_threshold=metrics.get("operating_threshold"),
            )
        protocol = (
            {str(key): value for key, value in raw_protocol.items()}
            if isinstance(raw_protocol, dict)
            else {}
        )
        return benchmark_summary, protocol
    except (OSError, ValueError, TypeError):
        return {}, {}


@router.get(
    "",
    response_model=ModelEvaluationResponse,
)
def get_model_evaluation(request: Request) -> ModelEvaluationResponse:
    """Expose runtime status plus honest, persisted held-out benchmark results."""
    benchmark_summary, benchmark_protocol = _load_benchmark_results()

    model_service = request.app.state.ml_inference_service
    if model_service is None:
        return ModelEvaluationResponse(
            model_available=False,
            status="unavailable",
            evaluation_metrics_available=False,
            benchmark_available=bool(benchmark_summary),
            benchmark_summary=benchmark_summary,
            benchmark_protocol=benchmark_protocol,
            data_note=(
                "No ML model is configured. Deterministic risk scoring remains "
                "available. Comparative benchmark data is shown only when a "
                "persisted held-out evaluation result is available."
            ),
        )

    artifact_path = _resolve_project_path(settings.ml_model_path)
    artifact_version: int | None = None
    feature_count: int | None = None
    persisted_metrics: dict[str, float] = {}

    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        artifact_version = payload.get("artifact_version")
        feature_names = payload.get("model", {}).get("feature_names", [])
        feature_count = (
            len(feature_names) if isinstance(feature_names, list) else None
        )
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
                if isinstance(value, (int, float))
                and not isinstance(value, bool)
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
        benchmark_available=bool(benchmark_summary),
        benchmark_summary=benchmark_summary,
        benchmark_protocol=benchmark_protocol,
        data_note=(
            "The analytical model is a risk signal, not fraud ground truth. "
            "Comparative results use held-out scenario families when available."
        ),
    )
