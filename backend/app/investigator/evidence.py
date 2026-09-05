"""Evidence bundle builder for fraud and refund risk investigations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from backend.app.domain.identifiers import CustomerId, RefundId
from backend.app.finance.exposure import FinancialExposure
from backend.app.finance.types import ReconstructionSnapshot
from backend.app.graph.builder import StructuralGraphBuilder
from backend.app.graph.model import EdgeType
from backend.app.ml.inference import MLInferenceService
from backend.app.risk.assessment import RiskAssessment
from backend.app.risk.decision import RiskDecision


@dataclass(frozen=True)
class CustomerProfileSnapshot:
    """Customer profile and historical behavioral metrics."""

    customer_id: str
    email: str | None
    phone: str | None
    created_at: str | None
    total_order_count: int
    total_refund_count: int
    total_paid_paise: int
    total_refunded_paise: int
    refund_rate_by_count: float
    refund_rate_by_amount: float


@dataclass(frozen=True)
class GraphTopologyEvidence:
    """Graph cluster connections and shared identity indicators."""

    cluster_id: str
    cluster_size: int
    connected_customer_ids: list[str]
    connected_refund_ids: list[str]
    shared_ip_addresses: list[str]
    shared_shipping_addresses: list[str]
    shared_device_fingerprints: list[str]
    shared_bank_accounts: list[str]
    is_multi_entity_cluster: bool


@dataclass(frozen=True)
class FeatureContributionEvidence:
    """Top feature signals contributing to the risk score."""

    feature_name: str
    value: float
    direction: str  # 'HIGH_RISK' or 'NORMAL'
    description: str


@dataclass(frozen=True)
class RuleViolationEvidence:
    """Detailed deterministic rule breach."""

    rule_id: str
    triggered: bool
    evidence_type: str
    evidence_value: Any
    evidence_threshold: Any
    notes: str


@dataclass(frozen=True)
class FinancialExposureEvidence:
    """Quantitative financial exposure metrics."""

    requested_amount_paise: int
    realized_suspicious_amount_paise: int
    pending_refund_exposure_paise: int
    remaining_refundable_exposure_paise: int
    net_retained_revenue_paise: int


@dataclass(frozen=True)
class EvidenceBundle:
    """Comprehensive investigation bundle synthesizing all detection signals."""

    refund_id: str
    assessed_at: str
    risk_level: str
    action: str
    final_risk_score: float
    behavioral_confirmation_score: float
    customer_profile: CustomerProfileSnapshot
    financial_exposure: FinancialExposureEvidence
    graph_topology: GraphTopologyEvidence
    rule_violations: list[RuleViolationEvidence]
    feature_contributions: list[FeatureContributionEvidence]

    def to_dict(self) -> dict[str, Any]:
        """Serialize bundle to dictionary."""
        return asdict(self)


class EvidenceBundleBuilder:
    """Builds unified EvidenceBundle from reconstruction snapshot and assessment."""

    @staticmethod
    def build(
        snapshot: ReconstructionSnapshot,
        refund_id: RefundId,
        assessment: RiskAssessment,
        decision: RiskDecision,
        exposure: FinancialExposure,
        component_refund_ids: tuple[RefundId, ...] | list[RefundId],
        ml_inference_service: MLInferenceService | None = None,
    ) -> EvidenceBundle:
        """Assemble a complete evidence bundle."""
        refund = snapshot.refunds[refund_id]

        # 1. Customer profile from order and refund aggregates
        customer_refunds = [
            r for r in snapshot.refunds.values()
            if r.customer_id == refund.customer_id
        ]
        customer_orders = [
            o for o in snapshot.orders.values()
            if o.customer_id == refund.customer_id
        ]

        total_paid_paise = sum(
            o.amount.amount_paise for o in customer_orders
        )
        total_refunded_paise = sum(
            r.requested_amount.amount_paise for r in customer_refunds
            if r.status.value in ("processed", "completed")
        )

        refund_rate_count = (
            len(customer_refunds) / len(customer_orders)
            if customer_orders else 0.0
        )
        refund_rate_amount = (
            total_refunded_paise / total_paid_paise
            if total_paid_paise > 0 else 0.0
        )

        earliest_order_time = (
            min(o.created_at.value for o in customer_orders).isoformat()
            if customer_orders else None
        )

        customer_profile = CustomerProfileSnapshot(
            customer_id=str(refund.customer_id),
            email=None,
            phone=None,
            created_at=earliest_order_time,
            total_order_count=len(customer_orders),
            total_refund_count=len(customer_refunds),
            total_paid_paise=total_paid_paise,
            total_refunded_paise=total_refunded_paise,
            refund_rate_by_count=round(refund_rate_count, 4),
            refund_rate_by_amount=round(refund_rate_amount, 4),
        )

        # 2. Financial exposure
        financial_exposure = FinancialExposureEvidence(
            requested_amount_paise=refund.requested_amount.amount_paise,
            realized_suspicious_amount_paise=exposure.realized_suspicious_amount.amount_paise,
            pending_refund_exposure_paise=exposure.pending_refund_exposure.amount_paise,
            remaining_refundable_exposure_paise=exposure.remaining_refundable_exposure.amount_paise,
            net_retained_revenue_paise=max(0, total_paid_paise - total_refunded_paise),
        )

        # 3. Graph topology
        connected_customers: set[str] = set()
        for c_ref_id in component_refund_ids:
            c_ref = snapshot.refunds.get(c_ref_id)
            if c_ref:
                connected_customers.add(str(c_ref.customer_id))

        graph = StructuralGraphBuilder().build(snapshot)
        component_customer_ids = set(connected_customers)
        device_to_customers: dict[str, set[str]] = {}
        for edge in graph.get_edges_by_type(EdgeType.CUSTOMER_USES_DEVICE):
            if edge.source_node_id in component_customer_ids:
                device_to_customers.setdefault(edge.target_node_id, set()).add(edge.source_node_id)
        shared_devices = sorted(
            device_id
            for device_id, customer_ids in device_to_customers.items()
            if len(customer_ids) > 1
        )

        order_to_customer = {
            str(order.order_id): str(order.customer_id)
            for order in snapshot.orders.values()
            if str(order.customer_id) in component_customer_ids
        }
        address_to_customers: dict[str, set[str]] = {}
        for edge in graph.get_edges_by_type(EdgeType.ORDER_SHIPS_TO_ADDRESS):
            customer_id = order_to_customer.get(edge.source_node_id)
            if customer_id is not None:
                address_to_customers.setdefault(edge.target_node_id, set()).add(customer_id)
        shared_addresses = sorted(
            address_id
            for address_id, customer_ids in address_to_customers.items()
            if len(customer_ids) > 1
        )

        graph_topology = GraphTopologyEvidence(
            cluster_id=assessment.component_id,
            cluster_size=len(component_refund_ids),
            connected_customer_ids=sorted(component_customer_ids),
            connected_refund_ids=sorted(str(rid) for rid in component_refund_ids),
            # IP and bank-account identifiers are not currently represented by
            # the domain model, so they remain empty rather than being fabricated.
            shared_ip_addresses=[],
            shared_shipping_addresses=shared_addresses,
            shared_device_fingerprints=shared_devices,
            shared_bank_accounts=[],
            is_multi_entity_cluster=len(component_customer_ids) > 1,
        )

        # 4. Rule violations
        rule_violations = [
            RuleViolationEvidence(
                rule_id=output.rule_id.value,
                triggered=output.triggered,
                evidence_type=output.evidence_type.value,
                evidence_value=output.evidence_value,
                evidence_threshold=output.evidence_threshold,
                notes=output.notes,
            )
            for output in assessment.rule_outputs
        ]

        # 5. Feature contributions. Prefer local ML contributions when a model
        # is configured; otherwise expose the deterministic score components.
        if ml_inference_service is not None:
            feature_contributions = [
                FeatureContributionEvidence(
                    feature_name=contribution.feature_name,
                    value=round(contribution.contribution, 4),
                    direction=("HIGH_RISK" if contribution.contribution > 0 else "LOWER_RISK"),
                    description=(
                        f"Raw feature value {contribution.raw_value:.4g}; local model contribution "
                        f"{contribution.contribution:+.4f} to the prediction logit"
                    ),
                )
                for contribution in ml_inference_service.explain_features(assessment, limit=5)
            ]
        else:
            feature_contributions = [
                FeatureContributionEvidence(
                    feature_name="Deterministic rule signal",
                    value=round(assessment.risk_score.rule_signal_component, 4),
                    direction="HIGH_RISK" if assessment.risk_score.rule_signal_component > 0 else "NORMAL",
                    description="Aggregated deterministic rule evidence used by the operational risk score",
                ),
                FeatureContributionEvidence(
                    feature_name="Behavioral confirmation",
                    value=round(assessment.risk_score.behavioral_confirmation_score, 4),
                    direction="HIGH_RISK" if assessment.risk_score.behavioral_confirmation_score > 0 else "NORMAL",
                    description="Behavioral pattern confirmation used by the operational risk score",
                ),
                FeatureContributionEvidence(
                    feature_name="Cluster signal",
                    value=round(assessment.risk_score.cluster_signal_component, 4),
                    direction="HIGH_RISK" if assessment.risk_score.cluster_signal_component > 0 else "NORMAL",
                    description="Structural coordination signal from the connected component",
                ),
            ]

        return EvidenceBundle(
            refund_id=str(refund_id),
            assessed_at=datetime.now(timezone.utc).isoformat(),
            risk_level=decision.risk_level.value,
            action=decision.action.value,
            final_risk_score=round(assessment.risk_score.final_score, 4),
            behavioral_confirmation_score=round(
                assessment.risk_score.behavioral_confirmation_score, 4
            ),
            customer_profile=customer_profile,
            financial_exposure=financial_exposure,
            graph_topology=graph_topology,
            rule_violations=rule_violations,
            feature_contributions=feature_contributions,
        )
