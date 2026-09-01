"""Unit tests for end-to-end risk assessment orchestration."""

from datetime import datetime, timedelta, timezone

import pytest

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
from backend.app.risk.assessment import RiskAssessment, RiskAssessor


def _build_single_refund_snapshot() -> tuple[
    ReconstructionSnapshot,
    RefundId,
    CustomerId,
]:
    """Build a minimal valid snapshot containing one refund."""

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
                    value=now - timedelta(hours=2)
                ),
                captured_amount=Money.of_paise(1000),
                captured_at=UTCDateTime(
                    value=now - timedelta(hours=1)
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
                    value=now - timedelta(hours=3)
                ),
            )
        },
        reconstruction_ordinal=1,
        event_count=3,
    )

    return snapshot, refund_id, customer_id


def test_assess_returns_complete_risk_assessment() -> None:
    """RiskAssessor produces a complete assessment for a valid refund."""

    snapshot, refund_id, customer_id = (
        _build_single_refund_snapshot()
    )

    assessor = RiskAssessor(snapshot)

    assessment = assessor.assess(refund_id)

    assert isinstance(assessment, RiskAssessment)

    assert assessment.refund_id == refund_id
    assert assessment.customer_id == customer_id

    assert assessment.component_id

    assert assessment.individual_features is not None
    assert assessment.cluster_features is not None
    assert assessment.relationship_features is not None

    assert len(assessment.rule_outputs) == 6

    assert 0.0 <= assessment.behavioral_confirmation_score <= 1.0

    assert assessment.risk_score is not None


def test_assess_returns_r01_to_r06_outputs() -> None:
    """Assessment evaluates all six deterministic rules."""

    snapshot, refund_id, _ = _build_single_refund_snapshot()

    assessment = RiskAssessor(snapshot).assess(refund_id)

    assert len(assessment.rule_outputs) == 6

    rule_ids = {
        output.rule_id
        for output in assessment.rule_outputs
    }

    assert len(rule_ids) == 6


def test_singleton_cluster_has_zero_behavioral_confirmation() -> None:
    """A single-customer component cannot demonstrate coordination."""

    snapshot, refund_id, _ = _build_single_refund_snapshot()

    assessment = RiskAssessor(snapshot).assess(refund_id)

    assert assessment.cluster_features.cluster_size == 1
    assert assessment.behavioral_confirmation_score == 0.0


def test_assessment_is_deterministic_for_same_snapshot() -> None:
    """Repeated assessments of the same refund produce consistent results."""

    snapshot, refund_id, _ = _build_single_refund_snapshot()

    assessor = RiskAssessor(snapshot)

    first = assessor.assess(refund_id)
    second = assessor.assess(refund_id)

    assert first.refund_id == second.refund_id
    assert first.customer_id == second.customer_id
    assert first.component_id == second.component_id

    assert (
        first.behavioral_confirmation_score
        == second.behavioral_confirmation_score
    )

    assert first.risk_score == second.risk_score


def test_assess_unknown_refund_raises_value_error() -> None:
    """Assessing a refund absent from the snapshot fails clearly."""

    snapshot, _, _ = _build_single_refund_snapshot()

    unknown_refund_id = RefundId.generate()

    assessor = RiskAssessor(snapshot)

    with pytest.raises(
        ValueError,
        match="not found in snapshot",
    ):
        assessor.assess(unknown_refund_id)