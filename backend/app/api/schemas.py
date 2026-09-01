"""Pydantic schemas for Refund Sentinel API responses."""

from __future__ import annotations

from pydantic import BaseModel, Field

from backend.app.risk.decision import DecisionAction, RiskLevel


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