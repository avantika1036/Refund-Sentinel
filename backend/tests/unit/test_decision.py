"""Unit tests for risk decision classification."""

from datetime import datetime, timedelta, timezone

from backend.app.domain.enums import PaymentStatus, RefundStatus
from backend.app.domain.identifiers import (
    CustomerId,
    OrderId,
    PaymentId,
    RefundId,
)
from backend.app.domain.value_objects import Money, UTCDateTime
from backend.app.finance.aggregates import (
    OrderState,
    PaymentState,
    RefundState,
)
from backend.app.finance.types import ReconstructionSnapshot
from backend.app.risk.assessment import RiskAssessor
from backend.app.risk.decision import (
    DecisionAction,
    RiskDecisionEngine,
    RiskLevel,
)


def _build_snapshot() -> tuple[
    ReconstructionSnapshot,
    RefundId,
]:
    """Build a deterministic snapshot for decision testing."""

    customer_id = CustomerId.generate()
    merchant_id = CustomerId.generate()

    order_id = OrderId.generate()
    payment_id = PaymentId.generate()
    refund_id = RefundId.generate()

    now = datetime.now(timezone.utc)

    snapshot = ReconstructionSnapshot(
        payments={
            payment_id: PaymentState(
                payment_id=payment_id,
                merchant_id=merchant_id,
                order_id=order_id,
                customer_id=customer_id,
                authorised_amount=Money.of_paise(1000),
                status=PaymentStatus.CAPTURED,
                created_at=UTCDateTime(
                    value=now - timedelta(hours=3)
                ),
                captured_amount=Money.of_paise(1000),
                captured_at=UTCDateTime(
                    value=now - timedelta(hours=2)
                ),
            )
        },
        refunds={
            refund_id: RefundState(
                refund_id=refund_id,
                payment_id=payment_id,
                merchant_id=merchant_id,
                customer_id=customer_id,
                order_id=order_id,
                requested_amount=Money.of_paise(1000),
                status=RefundStatus.PROCESSED,
                requested_at=UTCDateTime(value=now),
                processed_at=UTCDateTime(value=now),
                processed_amount=Money.of_paise(1000),
            )
        },
        orders={
            order_id: OrderState(
                order_id=order_id,
                merchant_id=merchant_id,
                customer_id=customer_id,
                amount=Money.of_paise(1000),
                created_at=UTCDateTime(
                    value=now - timedelta(hours=4)
                ),
            )
        },
        reconstruction_ordinal=1,
        event_count=3,
    )

    return snapshot, refund_id


def test_decide_returns_decision_for_assessment() -> None:
    """Decision engine converts an assessment into a decision."""

    snapshot, refund_id = _build_snapshot()

    assessment = RiskAssessor(snapshot).assess(refund_id)

    decision = RiskDecisionEngine().decide(assessment)

    assert decision.refund_id == refund_id

    assert 0.0 <= decision.final_score <= 1.0

    assert decision.risk_level in {
        RiskLevel.LOW,
        RiskLevel.MEDIUM,
        RiskLevel.HIGH,
    }

    assert decision.action in {
        DecisionAction.ALLOW,
        DecisionAction.REVIEW,
        DecisionAction.INVESTIGATE,
    }


def test_low_risk_maps_to_allow() -> None:
    """Low risk assessments recommend allowing the refund."""

    snapshot, refund_id = _build_snapshot()

    assessment = RiskAssessor(snapshot).assess(refund_id)

    decision = RiskDecisionEngine().decide(assessment)

    if decision.risk_level is RiskLevel.LOW:
        assert decision.action is DecisionAction.ALLOW


def test_medium_risk_maps_to_review() -> None:
    """Medium risk maps to manual review."""

    engine = RiskDecisionEngine()

    assert (
        engine._classify_risk_level(0.50)
        is RiskLevel.MEDIUM
    )

    assert (
        engine._select_action(RiskLevel.MEDIUM)
        is DecisionAction.REVIEW
    )


def test_high_risk_maps_to_investigate() -> None:
    """High risk maps to investigation."""

    engine = RiskDecisionEngine()

    assert (
        engine._classify_risk_level(0.90)
        is RiskLevel.HIGH
    )

    assert (
        engine._select_action(RiskLevel.HIGH)
        is DecisionAction.INVESTIGATE
    )


def test_low_risk_maps_to_allow_directly() -> None:
    """Low score maps directly to the allow action."""

    engine = RiskDecisionEngine()

    assert (
        engine._classify_risk_level(0.20)
        is RiskLevel.LOW
    )

    assert (
        engine._select_action(RiskLevel.LOW)
        is DecisionAction.ALLOW
    )


def test_triggered_rules_are_preserved() -> None:
    """Decision exposes triggered rule identifiers."""

    snapshot, refund_id = _build_snapshot()

    assessment = RiskAssessor(snapshot).assess(refund_id)

    decision = RiskDecisionEngine().decide(assessment)

    expected = tuple(
        output.rule_id
        for output in assessment.rule_outputs
        if output.triggered
    )

    assert decision.triggered_rule_ids == expected


def test_decision_explanation_contains_key_information() -> None:
    """Decision explanation includes score and evidence information."""

    snapshot, refund_id = _build_snapshot()

    assessment = RiskAssessor(snapshot).assess(refund_id)

    decision = RiskDecisionEngine().decide(assessment)

    assert str(refund_id) in decision.explanation
    assert "score" in decision.explanation.lower()
    assert "action" in decision.explanation.lower()