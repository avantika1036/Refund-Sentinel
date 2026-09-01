"""Unit tests for complete refund investigations."""

from datetime import datetime, timezone

import pytest

from backend.app.domain.enums import (
    PaymentStatus,
    RefundStatus,
)
from backend.app.domain.identifiers import (
    CustomerId,
    MerchantId,
    OrderId,
    PaymentId,
    RefundId,
)
from backend.app.domain.value_objects import (
    Money,
    UTCDateTime,
)
from backend.app.finance.aggregates import (
    OrderState,
    PaymentState,
    RefundState,
)
from backend.app.finance.types import (
    ReconstructionSnapshot,
)
from backend.app.risk.decision import (
    DecisionAction,
    RiskLevel,
)
from backend.app.risk.investigation import (
    InvestigationService,
)


def _build_snapshot() -> tuple[
    ReconstructionSnapshot,
    RefundId,
    RefundId,
    RefundId,
]:
    """Build a connected two-customer refund scenario."""

    now = datetime.now(timezone.utc)

    merchant_id = MerchantId.generate()

    customer1_id = CustomerId.generate()
    customer2_id = CustomerId.generate()

    payment1_id = PaymentId.generate()
    payment2_id = PaymentId.generate()

    order1_id = OrderId.generate()
    order2_id = OrderId.generate()

    refund1_id = RefundId.generate()
    refund2_id = RefundId.generate()
    unrelated_refund_id = RefundId.generate()

    payment1 = PaymentState(
        payment_id=payment1_id,
        merchant_id=merchant_id,
        order_id=order1_id,
        customer_id=customer1_id,
        authorised_amount=Money.of_paise(10000),
        status=PaymentStatus.CAPTURED,
        created_at=UTCDateTime(value=now),
        captured_amount=Money.of_paise(10000),
        captured_at=UTCDateTime(value=now),
    )

    payment2 = PaymentState(
        payment_id=payment2_id,
        merchant_id=merchant_id,
        order_id=order2_id,
        customer_id=customer2_id,
        authorised_amount=Money.of_paise(8000),
        status=PaymentStatus.CAPTURED,
        created_at=UTCDateTime(value=now),
        captured_amount=Money.of_paise(8000),
        captured_at=UTCDateTime(value=now),
    )

    order1 = OrderState(
        order_id=order1_id,
        merchant_id=merchant_id,
        customer_id=customer1_id,
        amount=Money.of_paise(10000),
        created_at=UTCDateTime(value=now),
    )

    order2 = OrderState(
        order_id=order2_id,
        merchant_id=merchant_id,
        customer_id=customer2_id,
        amount=Money.of_paise(8000),
        created_at=UTCDateTime(value=now),
    )

    refund1 = RefundState(
        refund_id=refund1_id,
        payment_id=payment1_id,
        merchant_id=merchant_id,
        customer_id=customer1_id,
        order_id=order1_id,
        requested_amount=Money.of_paise(4000),
        status=RefundStatus.PROCESSED,
        requested_at=UTCDateTime(value=now),
        processed_at=UTCDateTime(value=now),
        processed_amount=Money.of_paise(4000),
    )

    refund2 = RefundState(
        refund_id=refund2_id,
        payment_id=payment2_id,
        merchant_id=merchant_id,
        customer_id=customer2_id,
        order_id=order2_id,
        requested_amount=Money.of_paise(3000),
        status=RefundStatus.REQUESTED,
        requested_at=UTCDateTime(value=now),
    )

    unrelated_customer_id = CustomerId.generate()
    unrelated_payment_id = PaymentId.generate()
    unrelated_order_id = OrderId.generate()

    unrelated_payment = PaymentState(
        payment_id=unrelated_payment_id,
        merchant_id=merchant_id,
        order_id=unrelated_order_id,
        customer_id=unrelated_customer_id,
        authorised_amount=Money.of_paise(5000),
        status=PaymentStatus.CAPTURED,
        created_at=UTCDateTime(value=now),
        captured_amount=Money.of_paise(5000),
        captured_at=UTCDateTime(value=now),
    )

    unrelated_order = OrderState(
        order_id=unrelated_order_id,
        merchant_id=merchant_id,
        customer_id=unrelated_customer_id,
        amount=Money.of_paise(5000),
        created_at=UTCDateTime(value=now),
    )

    unrelated_refund = RefundState(
        refund_id=unrelated_refund_id,
        payment_id=unrelated_payment_id,
        merchant_id=merchant_id,
        customer_id=unrelated_customer_id,
        order_id=unrelated_order_id,
        requested_amount=Money.of_paise(1000),
        status=RefundStatus.PROCESSED,
        requested_at=UTCDateTime(value=now),
        processed_at=UTCDateTime(value=now),
        processed_amount=Money.of_paise(1000),
    )

    snapshot = ReconstructionSnapshot(
        payments={
            payment1_id: payment1,
            payment2_id: payment2,
            unrelated_payment_id: unrelated_payment,
        },
        refunds={
            refund1_id: refund1,
            refund2_id: refund2,
            unrelated_refund_id: unrelated_refund,
        },
        orders={
            order1_id: order1,
            order2_id: order2,
            unrelated_order_id: unrelated_order,
        },
        reconstruction_ordinal=1,
        event_count=3,
    )

    return (
        snapshot,
        refund1_id,
        refund2_id,
        unrelated_refund_id,
    )


def test_investigation_contains_assessment_and_decision() -> None:
    """Investigation should include assessment and decision results."""

    snapshot, refund1_id, _, _ = _build_snapshot()

    service = InvestigationService(snapshot)

    investigation = service.investigate(refund1_id)

    assert investigation.assessment.refund_id == refund1_id
    assert investigation.decision.refund_id == refund1_id

    assert 0.0 <= investigation.decision.final_score <= 1.0


def test_investigation_decision_matches_assessment_score() -> None:
    """Decision score should match the underlying assessment."""

    snapshot, refund1_id, _, _ = _build_snapshot()

    investigation = InvestigationService(
        snapshot
    ).investigate(refund1_id)

    assert (
        investigation.decision.final_score
        == investigation.assessment.risk_score.final_score
    )


def test_component_refunds_include_target_refund() -> None:
    """Target refund must always be included in investigation scope."""

    snapshot, refund1_id, _, _ = _build_snapshot()

    investigation = InvestigationService(
        snapshot
    ).investigate(refund1_id)

    assert refund1_id in investigation.component_refund_ids


def test_component_refunds_exclude_unrelated_refunds() -> None:
    """Refunds outside the target component must be excluded."""

    (
        snapshot,
        refund1_id,
        _,
        unrelated_refund_id,
    ) = _build_snapshot()

    investigation = InvestigationService(
        snapshot
    ).investigate(refund1_id)

    assert unrelated_refund_id not in (
        investigation.component_refund_ids
    )


def test_exposure_amounts_are_non_negative() -> None:
    """All exposure categories must be non-negative."""

    snapshot, refund1_id, _, _ = _build_snapshot()

    investigation = InvestigationService(
        snapshot
    ).investigate(refund1_id)

    exposure = investigation.exposure

    assert (
        exposure.realized_suspicious_amount.amount_paise
        >= 0
    )

    assert (
        exposure.pending_refund_exposure.amount_paise
        >= 0
    )

    assert (
        exposure.remaining_refundable_exposure.amount_paise
        >= 0
    )


def test_investigation_returns_valid_decision() -> None:
    """Every investigation should produce a valid decision."""

    snapshot, refund1_id, _, _ = _build_snapshot()

    investigation = InvestigationService(
        snapshot
    ).investigate(refund1_id)

    assert investigation.decision.risk_level in {
        RiskLevel.LOW,
        RiskLevel.MEDIUM,
        RiskLevel.HIGH,
    }

    assert investigation.decision.action in {
        DecisionAction.ALLOW,
        DecisionAction.REVIEW,
        DecisionAction.INVESTIGATE,
    }


def test_missing_refund_raises_value_error() -> None:
    """Investigating a nonexistent refund should fail clearly."""

    snapshot, _, _, _ = _build_snapshot()

    service = InvestigationService(snapshot)

    with pytest.raises(ValueError):
        service.investigate(RefundId.generate())