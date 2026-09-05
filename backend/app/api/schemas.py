"""Pydantic schemas for Refund Sentinel API responses."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

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


class CustomerProfileResponse(BaseModel):
    """Serialized customer profile for investigation."""

    customer_id: str
    email: str | None = None
    phone: str | None = None
    created_at: str | None = None
    total_order_count: int = 0
    total_refund_count: int = 0
    total_paid_paise: int = 0
    total_refunded_paise: int = 0
    refund_rate_by_count: float = 0.0
    refund_rate_by_amount: float = 0.0


class GraphTopologyResponse(BaseModel):
    """Serialized graph cluster topology evidence."""

    cluster_id: str
    cluster_size: int = 1
    connected_customer_ids: list[str] = Field(default_factory=list)
    connected_refund_ids: list[str] = Field(default_factory=list)
    shared_ip_addresses: list[str] = Field(default_factory=list)
    shared_shipping_addresses: list[str] = Field(default_factory=list)
    shared_device_fingerprints: list[str] = Field(default_factory=list)
    shared_bank_accounts: list[str] = Field(default_factory=list)
    is_multi_entity_cluster: bool = False


class FeatureContributionResponse(BaseModel):
    """Serialized feature signal contribution."""

    feature_name: str
    value: float
    direction: str
    description: str


class EvidenceBundleResponse(BaseModel):
    """Unified Section 9 Evidence Bundle."""

    refund_id: str
    assessed_at: str
    risk_level: str
    action: str
    final_risk_score: float
    behavioral_confirmation_score: float
    customer_profile: CustomerProfileResponse
    financial_exposure: FinancialExposureResponse
    graph_topology: GraphTopologyResponse
    rule_violations: list[RuleEvidenceResponse]
    feature_contributions: list[FeatureContributionResponse]


class InvestigationExplanationResponse(BaseModel):
    """Synthesized investigation analyst explanation."""

    headline: str
    narrative_summary: str
    key_risk_drivers: list[str]
    suggested_action_rationale: str
    is_llm_generated: bool = False


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

    evidence_bundle: EvidenceBundleResponse | None = None

    explanation_summary: InvestigationExplanationResponse | None = None


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



class ComparativeBaselineMetricsResponse(BaseModel):
    """One held-out benchmark baseline summary."""

    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1_score: float = Field(ge=0.0, le=1.0)
    accuracy: float = Field(ge=0.0, le=1.0)
    true_positive: int = Field(ge=0)
    true_negative: int = Field(ge=0)
    false_positive: int = Field(ge=0)
    false_negative: int = Field(ge=0)
    review_volume: int = Field(ge=0)
    # Primary benchmark exposure metric. This is measured abuse exposure
    # captured by the operating point, not a claim of financial recovery.
    abuse_exposure_captured_inr: float = Field(default=0.0, ge=0.0)
    abuse_exposure_capture_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    total_test_abuse_exposure_inr: float = Field(default=0.0, ge=0.0)
    false_positive_flagged_amount_inr: float = Field(ge=0.0)
    total_flagged_amount_inr: float = Field(ge=0.0)
    operating_threshold: float | int | None = None

class ModelEvaluationResponse(BaseModel):
    """Honest runtime model status and available evaluation metadata."""

    model_config = ConfigDict(
        protected_namespaces=()
    )

    model_available: bool
    status: str

    artifact_version: int | None = None

    feature_count: int | None = Field(
        default=None,
        ge=0,
    )

    evaluation_metrics_available: bool

    metrics: dict[str, float] = Field(
        default_factory=dict,
    )

    benchmark_available: bool = False

    benchmark_summary: dict[
        str,
        ComparativeBaselineMetricsResponse,
    ] = Field(
        default_factory=dict,
    )

    benchmark_protocol: dict[str, object] = Field(
        default_factory=dict,
    )

    data_note: str