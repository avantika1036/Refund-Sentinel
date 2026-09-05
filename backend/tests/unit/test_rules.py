from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.app.domain.identifiers import CustomerId, PaymentId
from backend.app.domain.value_objects import UTCDateTime
from backend.app.finance.types import ReconstructionSnapshot
from backend.app.graph.components import ConnectedComponent
from backend.app.graph.model import GraphNode, NodeType
from backend.app.risk.features.individual import IndividualFeatures
from backend.app.risk.rules import (
    EvidenceType,
    RuleEngine,
    RuleId,
    RuleOutput,
    compute_rule_signal_component,
)


def make_features(
    *,
    capture_latency: float | None = 2.0,
    delivery_latency: float | None = 10.0,
    refund_fraction: float | None = 1.0,
    is_full_refund: bool | None = True,
    refund_rate: float | None = 0.0,
    velocity: int | None = 0,
) -> IndividualFeatures:
    """Create focused feature fixtures for rule tests."""
    return IndividualFeatures(
        capture_to_refund_latency_hrs=capture_latency,
        order_to_refund_latency_hrs=10.0,
        delivery_to_refund_latency_hrs=delivery_latency,
        refund_requested_before_delivery=(None if delivery_latency is None else delivery_latency < 0),
        refund_amount_fraction=refund_fraction,
        is_full_refund=is_full_refund,
        customer_refund_rate_90d=refund_rate,
        customer_refund_velocity_7d=velocity,
        refund_reason_code="DEFECTIVE",
        reason_rotation_count_90d=1,
        account_age_at_refund_days=None,
        prior_successful_orders_no_refund=0,
    )


def make_engine(
    refunds: dict | None = None,
) -> RuleEngine:
    """Create a rule engine with the minimum required snapshot."""
    snapshot = ReconstructionSnapshot(
        payments={},
        refunds=refunds or {},
        orders={},
        reconstruction_ordinal=1,
        event_count=0,
    )
    return RuleEngine(snapshot)


def test_r01_triggers_below_four_hours() -> None:
    """R-01 triggers when refund occurs less than four hours after capture."""
    engine = make_engine()
    features = make_features(capture_latency=3.99)

    payment = SimpleNamespace(payment_id=PaymentId.generate())

    result = engine.evaluate_r01_rapid_refund(features, payment)

    assert result.rule_id is RuleId.R01_RAPID_REFUND_AFTER_CAPTURE
    assert result.triggered is True
    assert result.evidence_type is EvidenceType.LATENCY_HOURS
    assert result.evidence_value == 3.99
    assert result.evidence_threshold == 4.0


def test_r01_does_not_trigger_at_four_hour_boundary() -> None:
    """R-01 uses a strict less-than comparison."""
    engine = make_engine()
    features = make_features(capture_latency=4.0)

    payment = SimpleNamespace(payment_id=PaymentId.generate())

    result = engine.evaluate_r01_rapid_refund(features, payment)

    assert result.triggered is False


def test_r02_requires_delivery_data() -> None:
    """R-02 does not trigger when delivery information is unavailable."""
    engine = make_engine()
    features = make_features(delivery_latency=None)

    result = engine.evaluate_r02_refund_before_delivery(
        features,
        order_delivered_at=None,
    )

    assert result.rule_id is RuleId.R02_REFUND_BEFORE_DELIVERY
    assert result.triggered is False
    assert result.evidence_type is EvidenceType.BOOLEAN
    assert result.evidence_value is False


def test_r02_triggers_when_refund_precedes_delivery() -> None:
    """R-02 triggers when delivery-to-refund latency is negative."""
    engine = make_engine()
    features = make_features(delivery_latency=-2.0)

    delivered_at = UTCDateTime.now()

    result = engine.evaluate_r02_refund_before_delivery(
        features,
        order_delivered_at=delivered_at,
    )

    assert result.triggered is True
    assert result.evidence_type is EvidenceType.BOOLEAN
    assert result.evidence_value is True


def test_r03_requires_exactly_one_refund_for_payment() -> None:
    """R-03 requires a full refund and exactly one refund."""
    payment_id = PaymentId.generate()

    refund = SimpleNamespace(payment_id=payment_id)

    engine = make_engine(
        refunds={
            "refund-1": refund,
        }
    )

    features = make_features(
        refund_fraction=1.0,
        is_full_refund=True,
    )

    payment = SimpleNamespace(payment_id=payment_id)

    result = engine.evaluate_r03_full_refund(
        features,
        payment,
    )

    assert result.triggered is True
    assert result.evidence_type is EvidenceType.BOOLEAN


def test_r03_does_not_trigger_when_payment_has_multiple_refunds() -> None:
    """R-03 must not fire when the payment has multiple refunds."""
    payment_id = PaymentId.generate()

    refunds = {
        "refund-1": SimpleNamespace(payment_id=payment_id),
        "refund-2": SimpleNamespace(payment_id=payment_id),
    }

    engine = make_engine(refunds=refunds)

    features = make_features(
        refund_fraction=1.0,
        is_full_refund=True,
    )

    payment = SimpleNamespace(payment_id=payment_id)

    result = engine.evaluate_r03_full_refund(
        features,
        payment,
    )

    assert result.triggered is False


def test_r04_does_not_trigger_without_sufficient_history() -> None:
    """R-04 is not applicable when the feature layer has insufficient history."""
    engine = make_engine()

    features = make_features(refund_rate=None)

    result = engine.evaluate_r04_refund_rate_anomaly(
        features,
        merchant_baseline_rate=0.1,
    )

    assert result.rule_id is RuleId.R04_CUSTOMER_REFUND_RATE_ANOMALY
    assert result.triggered is False
    assert result.evidence_value is None
    assert result.evidence_threshold == pytest.approx(0.3)


def test_r04_triggers_above_three_times_merchant_baseline() -> None:
    """R-04 triggers when customer refund rate exceeds 3x baseline."""
    engine = make_engine()

    features = make_features(refund_rate=0.31)

    result = engine.evaluate_r04_refund_rate_anomaly(
        features,
        merchant_baseline_rate=0.1,
    )

    assert result.triggered is True
    assert result.evidence_type is EvidenceType.RATE
    assert result.evidence_value == 0.31
    assert result.evidence_threshold == pytest.approx(0.3)


def test_r05_triggers_at_velocity_threshold() -> None:
    """R-05 triggers when seven-day refund velocity reaches three."""
    engine = make_engine()

    features = make_features(velocity=3)

    result = engine.evaluate_r05_refund_velocity_spike(features)

    assert result.rule_id is RuleId.R05_REFUND_VELOCITY_SPIKE
    assert result.triggered is True
    assert result.evidence_type is EvidenceType.COUNT
    assert result.evidence_value == 3
    assert result.evidence_threshold == 3


def test_r05_handles_missing_velocity() -> None:
    """R-05 does not trigger when velocity data is unavailable."""
    engine = make_engine()

    features = make_features(velocity=None)

    result = engine.evaluate_r05_refund_velocity_spike(features)

    assert result.triggered is False
    assert result.evidence_value is None


def test_r06_triggers_when_two_cluster_members_are_flagged() -> None:
    """R-06 triggers when at least two customers have triggered rules."""
    customer1 = CustomerId.generate()
    customer2 = CustomerId.generate()

    component = ConnectedComponent(
        component_id=str(customer1),
        nodes=frozenset(
            {
                GraphNode(str(customer1), NodeType.CUSTOMER),
                GraphNode(str(customer2), NodeType.CUSTOMER),
            }
        ),
        edges=frozenset(),
    )

    triggered_output = RuleOutput(
        rule_id=RuleId.R01_RAPID_REFUND_AFTER_CAPTURE,
        triggered=True,
        evidence_type=EvidenceType.LATENCY_HOURS,
        evidence_value=2.0,
        evidence_threshold=4.0,
        base_signal_weight=0.3,
        notes="triggered",
    )

    engine = make_engine()

    result = engine.evaluate_r06_cluster_flags(
        component,
        {
            customer1: [triggered_output],
            customer2: [triggered_output],
        },
    )

    assert result.rule_id is RuleId.R06_MULTIPLE_ACCOUNTS_FLAGGED_IN_CLUSTER
    assert result.triggered is True
    assert result.evidence_value == 2
    assert result.evidence_threshold == 2


def test_rule_signal_component_uses_triggered_weights() -> None:
    """Rule signal component is the fraction of weighted rule strength that fired."""
    outputs = [
        RuleOutput(
            rule_id=RuleId.R01_RAPID_REFUND_AFTER_CAPTURE,
            triggered=True,
            evidence_type=EvidenceType.LATENCY_HOURS,
            evidence_value=2.0,
            evidence_threshold=4.0,
            base_signal_weight=0.3,
            notes="triggered",
        ),
        RuleOutput(
            rule_id=RuleId.R02_REFUND_BEFORE_DELIVERY,
            triggered=False,
            evidence_type=EvidenceType.LATENCY_HOURS,
            evidence_value=5.0,
            evidence_threshold=0.0,
            base_signal_weight=0.4,
            notes="not triggered",
        ),
        RuleOutput(
            rule_id=RuleId.R03_FULL_REFUND,
            triggered=True,
            evidence_type=EvidenceType.BOOLEAN,
            evidence_value=True,
            evidence_threshold=0.99,
            base_signal_weight=0.3,
            notes="triggered",
        ),
    ]

    result = compute_rule_signal_component(outputs)

    assert result == 0.6


def test_rule_signal_component_is_zero_for_empty_input() -> None:
    """No rules means no rule signal."""
    assert compute_rule_signal_component([]) == 0.0