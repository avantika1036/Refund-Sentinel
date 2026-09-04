"""Pydantic schemas for Refund Sentinel API responses."""

from __future__ import annotations

from pydantic import BaseModel, Field

from backend.app.risk.decision import DecisionAction, RiskLevel
from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"


class RuleEvidenceResponse(BaseModel):
    """Serialized deterministic rule evidence."""

    rule_id: str
    triggered: bool
    evidence_type: str
    evidence_value: float | int | bool | None
    evidence_threshold: float | int | None
    base_signal_weight: float
    notes: str


class RiskScoreResponse(BaseModel):
    """Serialized risk score components."""

    rule_signal_component: float = Field(
        ge=0.0,
        le=1.0,
    )

    behavioral_confirmation_score: float = Field(
        ge=0.0,
        le=1.0,
    )

    cluster_signal_component: float = Field(
        ge=0.0,
        le=1.0,
    )

    final_score: float = Field(
        ge=0.0,
        le=1.0,
    )


class AssessmentResponse(BaseModel):
    """Complete API response for one refund assessment."""

    refund_id: str
    customer_id: str
    component_id: str

    risk_level: RiskLevel
    action: DecisionAction

    triggered_rule_ids: list[str]

    behavioral_confirmation_score: float

    risk_score: RiskScoreResponse

    rule_outputs: list[RuleEvidenceResponse]

    explanation: str


class FinancialExposureResponse(BaseModel):
    """Serialized financial exposure for an investigation."""

    realized_suspicious_amount_paise: int = Field(
        ge=0,
    )

    pending_refund_exposure_paise: int = Field(
        ge=0,
    )

    remaining_refundable_exposure_paise: int = Field(
        ge=0,
    )


class MLPredictionResponse(BaseModel):
    """Serialized machine-learning prediction."""

    probability: float = Field(
        ge=0.0,
        le=1.0,
    )

    is_high_risk: bool


class InvestigationResponse(BaseModel):
    """Complete API response for a refund investigation."""

    assessment: AssessmentResponse

    exposure: FinancialExposureResponse

    component_refund_ids: list[str]

    ml_prediction: MLPredictionResponse | None = None


class QueueCaseResponse(BaseModel):
    """A real refund candidate prepared for queue prioritization."""

    refund_id: str
    customer_id: str
    component_id: str
    status: str
    requested_at: str
    requested_amount_paise: int = Field(ge=0)
    risk_level: RiskLevel
    action: DecisionAction
    risk_score: float = Field(ge=0.0, le=1.0)
    triggered_rule_ids: list[str]
    component_refund_count: int = Field(ge=1)
    exposure: FinancialExposureResponse
    ml_prediction: MLPredictionResponse | None = None


class QueueMetricsResponse(BaseModel):
    """Aggregate, non-overlapping operational metrics for the queue."""

    open_case_count: int = Field(ge=0)
    high_risk_count: int = Field(ge=0)
    medium_risk_count: int = Field(ge=0)
    low_risk_count: int = Field(ge=0)
    triggered_case_count: int = Field(ge=0)
    clustered_case_count: int = Field(ge=0)
    pending_refund_exposure_paise: int = Field(ge=0)
    realized_suspicious_amount_paise: int = Field(ge=0)


class InvestigationQueueResponse(BaseModel):
    """Prioritized refund candidates and operational queue metrics."""

    cases: list[QueueCaseResponse]
    metrics: QueueMetricsResponse


class ModelEvaluationResponse(BaseModel):
    """Honest runtime model status and available evaluation metadata."""

    model_config = ConfigDict(
        protected_namespaces=()
    )

    model_available: bool
    status: str
    artifact_version: int | None = None
    feature_count: int | None = Field(default=None, ge=0)
    evaluation_metrics_available: bool
    metrics: dict[str, float] = {}
    data_note: str