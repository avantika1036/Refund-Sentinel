"""Deterministic rule-based evidence signals.

Implements the six rules from the Refund Sentinel architecture.
Rules produce signals, NOT fraud probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from backend.app.domain.identifiers import CustomerId
from backend.app.domain.value_objects import UTCDateTime
from backend.app.finance.aggregates import PaymentState
from backend.app.finance.types import ReconstructionSnapshot
from backend.app.graph.components import ConnectedComponent
from backend.app.graph.model import NodeType

if TYPE_CHECKING:
    from backend.app.risk.features.individual import IndividualFeatures


class RuleId(str, Enum):
    """Identifiers for the six deterministic rules."""

    R01_RAPID_REFUND_AFTER_CAPTURE = "R01_RAPID_REFUND_AFTER_CAPTURE"
    R02_REFUND_BEFORE_DELIVERY = "R02_REFUND_BEFORE_DELIVERY"
    R03_FULL_REFUND = "R03_FULL_REFUND"
    R04_CUSTOMER_REFUND_RATE_ANOMALY = "R04_CUSTOMER_REFUND_RATE_ANOMALY"
    R05_REFUND_VELOCITY_SPIKE = "R05_REFUND_VELOCITY_SPIKE"
    R06_MULTIPLE_ACCOUNTS_FLAGGED_IN_CLUSTER = (
        "R06_MULTIPLE_ACCOUNTS_FLAGGED_IN_CLUSTER"
    )


class EvidenceType(str, Enum):
    """Types of evidence produced by rules."""

    LATENCY_HOURS = "latency_hours"
    BOOLEAN = "boolean"
    RATE = "rate"
    COUNT = "count"
    CLUSTER_FLAG_COUNT = "cluster_flag_count"


@dataclass(frozen=True)
class RuleOutput:
    """Output from a rule evaluation.

    Contains the evidence signal, not a fraud probability.
    """

    rule_id: RuleId
    triggered: bool
    evidence_type: EvidenceType
    evidence_value: float | int | bool | None
    evidence_threshold: float | int | None
    base_signal_weight: float
    notes: str


class RuleEngine:
    """Evaluates deterministic rules on features and state.

    IMPORTANT: Rules produce signals, NOT fraud probabilities.

    The rule_signal_component is a fraction of rule signal strength
    that fired. It is NOT a fraud probability.
    """

    # Default thresholds
    R01_THRESHOLD_HOURS = 4.0
    R04_K_MULTIPLIER = 3.0
    R05_VELOCITY_THRESHOLD = 3
    R06_CLUSTER_FLAG_THRESHOLD = 2

    # Base signal weights
    R01_WEIGHT = 0.3
    R02_WEIGHT = 0.4
    R03_WEIGHT = 0.3
    R04_WEIGHT = 0.5
    R05_WEIGHT = 0.4
    R06_WEIGHT = 0.6

    def __init__(self, snapshot: ReconstructionSnapshot) -> None:
        self._snapshot = snapshot

    def evaluate_r01_rapid_refund(
        self,
        features: IndividualFeatures,
        payment_state: PaymentState,
    ) -> RuleOutput:
        """R-01: Rapid refund after capture.

        Trigger when capture-to-refund-request latency is below
        the configured threshold.

        Default threshold: 4 hours.
        """
        latency = features.capture_to_refund_latency_hrs
        threshold = self.R01_THRESHOLD_HOURS

        triggered = latency is not None and latency < threshold

        return RuleOutput(
            rule_id=RuleId.R01_RAPID_REFUND_AFTER_CAPTURE,
            triggered=triggered,
            evidence_type=EvidenceType.LATENCY_HOURS,
            evidence_value=latency,
            evidence_threshold=threshold,
            base_signal_weight=self.R01_WEIGHT,
            notes=(
                f"Capture-to-refund latency: {latency} hours "
                f"(threshold: {threshold} hours)"
                if latency is not None
                else "Capture time not available"
            ),
        )

    def evaluate_r02_refund_before_delivery(
        self,
        features: IndividualFeatures,
        order_delivered_at: UTCDateTime | None,
    ) -> RuleOutput:
        """R-02: Refund requested before delivery confirmation.

        Trigger only when delivery data exists AND the refund was
        requested before delivery.

        Absence of delivery data does NOT trigger this rule.
        """
        if order_delivered_at is None:
            return RuleOutput(
                rule_id=RuleId.R02_REFUND_BEFORE_DELIVERY,
                triggered=False,
                evidence_type=EvidenceType.BOOLEAN,
                evidence_value=False,
                evidence_threshold=None,
                base_signal_weight=self.R02_WEIGHT,
                notes="Delivery data not available, rule not applicable",
            )

        latency = features.delivery_to_refund_latency_hrs
        triggered = latency is not None and latency < 0

        return RuleOutput(
            rule_id=RuleId.R02_REFUND_BEFORE_DELIVERY,
            triggered=triggered,
            evidence_type=EvidenceType.LATENCY_HOURS,
            evidence_value=latency,
            evidence_threshold=0.0,
            base_signal_weight=self.R02_WEIGHT,
            notes=(
                f"Delivery-to-refund latency: {latency} hours "
                "(negative = before delivery)"
                if latency is not None
                else "Delivery-to-refund latency not available"
            ),
        )

    def evaluate_r03_full_refund(
        self,
        features: IndividualFeatures,
        payment_state: PaymentState,
    ) -> RuleOutput:
        """R-03: Full refund.

        Trigger when:

        1. The refund amount is effectively the full captured amount.
        2. This payment has exactly one refund.

        This rule intentionally uses the reconstructed snapshot as the
        source of truth for the refund count.
        """
        is_full = features.is_full_refund
        fraction = features.refund_amount_fraction

        refund_count = sum(
            1
            for refund_state in self._snapshot.refunds.values()
            if refund_state.payment_id == payment_state.payment_id
        )

        exactly_one_refund = refund_count == 1

        triggered = (
            is_full is True
            and fraction is not None
            and fraction >= 0.99
            and exactly_one_refund
        )

        return RuleOutput(
            rule_id=RuleId.R03_FULL_REFUND,
            triggered=triggered,
            evidence_type=EvidenceType.BOOLEAN,
            evidence_value=is_full,
            evidence_threshold=0.99,
            base_signal_weight=self.R03_WEIGHT,
            notes=(
                f"Full refund: {is_full}, "
                f"fraction: {fraction}, "
                f"refund count for payment: {refund_count}, "
                f"exactly one refund: {exactly_one_refund}"
            ),
        )

    def evaluate_r04_refund_rate_anomaly(
        self,
        features: IndividualFeatures,
        merchant_baseline_rate: float = 0.1,
    ) -> RuleOutput:
        """R-04: Customer refund rate anomaly.

        Trigger when:

            customer_refund_rate_90d >
            K × merchant_baseline_rate

        Default K = 3.

        The feature layer is responsible for enforcing the minimum
        historical-order requirement. When the feature is None, this
        rule is not applicable and does not trigger.
        """
        rate = features.customer_refund_rate_90d
        threshold = self.R04_K_MULTIPLIER * merchant_baseline_rate

        if rate is None:
            return RuleOutput(
                rule_id=RuleId.R04_CUSTOMER_REFUND_RATE_ANOMALY,
                triggered=False,
                evidence_type=EvidenceType.RATE,
                evidence_value=None,
                evidence_threshold=threshold,
                base_signal_weight=self.R04_WEIGHT,
                notes="Insufficient history, rule not applicable",
            )

        triggered = rate > threshold

        return RuleOutput(
            rule_id=RuleId.R04_CUSTOMER_REFUND_RATE_ANOMALY,
            triggered=triggered,
            evidence_type=EvidenceType.RATE,
            evidence_value=rate,
            evidence_threshold=threshold,
            base_signal_weight=self.R04_WEIGHT,
            notes=(
                f"Customer refund rate: {rate:.3f} "
                f"(threshold: {threshold:.3f})"
            ),
        )

    def evaluate_r05_refund_velocity_spike(
        self,
        features: IndividualFeatures,
    ) -> RuleOutput:
        """R-05: Refund velocity spike.

        Trigger when customer refunds in the previous 7 days are
        greater than or equal to the configured threshold.
        """
        velocity = features.customer_refund_velocity_7d
        threshold = self.R05_VELOCITY_THRESHOLD

        triggered = velocity is not None and velocity >= threshold

        return RuleOutput(
            rule_id=RuleId.R05_REFUND_VELOCITY_SPIKE,
            triggered=triggered,
            evidence_type=EvidenceType.COUNT,
            evidence_value=velocity,
            evidence_threshold=threshold,
            base_signal_weight=self.R05_WEIGHT,
            notes=(
                f"Refund velocity (7d): {velocity} "
                f"(threshold: {threshold})"
                if velocity is not None
                else "Refund velocity data not available"
            ),
        )

    def evaluate_r06_cluster_flags(
        self,
        component: ConnectedComponent,
        member_rule_outputs: dict[CustomerId, list[RuleOutput]],
    ) -> RuleOutput:
        """R-06: Multiple accounts flagged in same cluster.

        Trigger when at least the configured number of distinct
        customers in the component have at least one triggered rule.

        This rule is evaluated only after individual member rules
        have been evaluated.
        """
        flagged_customers = 0

        for customer_id, outputs in member_rule_outputs.items():
            customer_in_component = any(
                node.node_type == NodeType.CUSTOMER
                and node.node_id == str(customer_id)
                for node in component.nodes
            )

            if customer_in_component and any(
                output.triggered for output in outputs
            ):
                flagged_customers += 1

        threshold = self.R06_CLUSTER_FLAG_THRESHOLD
        triggered = flagged_customers >= threshold

        return RuleOutput(
            rule_id=RuleId.R06_MULTIPLE_ACCOUNTS_FLAGGED_IN_CLUSTER,
            triggered=triggered,
            evidence_type=EvidenceType.CLUSTER_FLAG_COUNT,
            evidence_value=flagged_customers,
            evidence_threshold=threshold,
            base_signal_weight=self.R06_WEIGHT,
            notes=(
                f"Flagged customers in cluster: {flagged_customers} "
                f"(threshold: {threshold})"
            ),
        )


def compute_rule_signal_component(
    rule_outputs: list[RuleOutput],
) -> float:
    """Compute the rule signal component.

    Formula:

        weighted sum of base_signal_weight for triggered rules
        --------------------------------------------------------
        sum of base_signal_weight for all applicable rules

    Returns a value in [0, 1].

    IMPORTANT:
    This value means "fraction of rule signal strength that fired".
    It is NOT a fraud probability.
    """
    if not rule_outputs:
        return 0.0

    triggered_weight = sum(
        output.base_signal_weight
        for output in rule_outputs
        if output.triggered
    )

    total_weight = sum(
        output.base_signal_weight
        for output in rule_outputs
    )

    if total_weight == 0:
        return 0.0

    return triggered_weight / total_weight