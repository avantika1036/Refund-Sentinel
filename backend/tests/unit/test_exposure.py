"""Unit tests for financial exposure estimation."""

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
    PaymentState,
    RefundState,
)
from backend.app.finance.exposure import (
    compute_financial_exposure,
)
from backend.app.finance.types import (
    ReconstructionSnapshot,
)


def _build_snapshot() -> tuple[
    ReconstructionSnapshot,
    PaymentId,
    RefundId,
    RefundId,
    RefundId,
]:
    """Build a snapshot containing processed and pending refunds."""

    now = datetime.now(timezone.utc)

    merchant_id = MerchantId.generate()
    customer_id = CustomerId.generate()

    payment_id = PaymentId.generate()
    order_id = OrderId.generate()

    processed_refund_id = RefundId.generate()
    requested_refund_id = RefundId.generate()
    failed_refund_id = RefundId.generate()

    payment = PaymentState(
        payment_id=payment_id,
        merchant_id=merchant_id,
        order_id=order_id,
        customer_id=customer_id,
        authorised_amount=Money.of_paise(10000),
        status=PaymentStatus.CAPTURED,
        created_at=UTCDateTime(value=now),
        captured_amount=Money.of_paise(10000),
        captured_at=UTCDateTime(value=now),
        cumulative_refunded=Money.of_paise(3000),
    )

    processed_refund = RefundState(
        refund_id=processed_refund_id,
        payment_id=payment_id,
        merchant_id=merchant_id,
        customer_id=customer_id,
        order_id=order_id,
        requested_amount=Money.of_paise(3000),
        status=RefundStatus.PROCESSED,
        requested_at=UTCDateTime(value=now),
        processed_at=UTCDateTime(value=now),
        processed_amount=Money.of_paise(3000),
    )

    requested_refund = RefundState(
        refund_id=requested_refund_id,
        payment_id=payment_id,
        merchant_id=merchant_id,
        customer_id=customer_id,
        order_id=order_id,
        requested_amount=Money.of_paise(2000),
        status=RefundStatus.REQUESTED,
        requested_at=UTCDateTime(value=now),
    )

    failed_refund = RefundState(
        refund_id=failed_refund_id,
        payment_id=payment_id,
        merchant_id=merchant_id,
        customer_id=customer_id,
        order_id=order_id,
        requested_amount=Money.of_paise(1000),
        status=RefundStatus.FAILED,
        requested_at=UTCDateTime(value=now),
    )

    snapshot = ReconstructionSnapshot(
        payments={
            payment_id: payment,
        },
        refunds={
            processed_refund_id: processed_refund,
            requested_refund_id: requested_refund,
            failed_refund_id: failed_refund,
        },
        orders={},
        reconstruction_ordinal=1,
        event_count=3,
    )

    return (
        snapshot,
        payment_id,
        processed_refund_id,
        requested_refund_id,
        failed_refund_id,
    )


def test_processed_refund_contributes_to_realized_exposure() -> None:
    """Processed amounts are counted as realized exposure."""

    (
        snapshot,
        _,
        processed_refund_id,
        _,
        _,
    ) = _build_snapshot()

    exposure = compute_financial_exposure(
        snapshot=snapshot,
        refund_ids=[processed_refund_id],
        risk_score=1.0,
    )

    assert (
        exposure.realized_suspicious_amount.amount_paise
        == 3000
    )

    assert (
        exposure.pending_refund_exposure.amount_paise
        == 0
    )


def test_pending_refund_contributes_to_pending_exposure() -> None:
    """Requested refunds contribute to pending exposure."""

    (
        snapshot,
        _,
        _,
        requested_refund_id,
        _,
    ) = _build_snapshot()

    exposure = compute_financial_exposure(
        snapshot=snapshot,
        refund_ids=[requested_refund_id],
        risk_score=1.0,
    )

    assert (
        exposure.realized_suspicious_amount.amount_paise
        == 0
    )

    assert (
        exposure.pending_refund_exposure.amount_paise
        == 2000
    )


def test_failed_refund_does_not_contribute_to_realized_or_pending() -> None:
    """Failed refunds have no realized or pending exposure."""

    (
        snapshot,
        _,
        _,
        _,
        failed_refund_id,
    ) = _build_snapshot()

    exposure = compute_financial_exposure(
        snapshot=snapshot,
        refund_ids=[failed_refund_id],
        risk_score=1.0,
    )

    assert (
        exposure.realized_suspicious_amount.amount_paise
        == 0
    )

    assert (
        exposure.pending_refund_exposure.amount_paise
        == 0
    )


def test_risk_score_weights_all_exposure_categories() -> None:
    """All exposure categories are multiplied by risk score."""

    (
        snapshot,
        _,
        processed_refund_id,
        requested_refund_id,
        _,
    ) = _build_snapshot()

    exposure = compute_financial_exposure(
        snapshot=snapshot,
        refund_ids=[
            processed_refund_id,
            requested_refund_id,
        ],
        risk_score=0.5,
    )

    # Processed: 3000 × 0.5 = 1500
    assert (
        exposure.realized_suspicious_amount.amount_paise
        == 1500
    )

    # Pending: 2000 × 0.5 = 1000
    assert (
        exposure.pending_refund_exposure.amount_paise
        == 1000
    )

    # Payment remaining:
    # 10000 captured - 3000 processed = 7000
    # minus 2000 pending = 5000 future refundable
    # 5000 × 0.5 = 2500
    assert (
        exposure.remaining_refundable_exposure.amount_paise
        == 2500
    )


def test_remaining_exposure_does_not_double_count_pending_refunds() -> None:
    """Already-requested refunds are excluded from future exposure."""

    (
        snapshot,
        _,
        _,
        requested_refund_id,
        _,
    ) = _build_snapshot()

    exposure = compute_financial_exposure(
        snapshot=snapshot,
        refund_ids=[requested_refund_id],
        risk_score=1.0,
    )

    # 10000 captured - 3000 processed = 7000 remaining on payment.
    # The existing 2000 requested refund is already pending, so only
    # 5000 remains as future not-yet-requested exposure.
    assert (
        exposure.remaining_refundable_exposure.amount_paise
        == 5000
    )


def test_empty_refund_set_returns_zero_exposure() -> None:
    """No selected refunds produces zero exposure."""

    snapshot, _, _, _, _ = _build_snapshot()

    exposure = compute_financial_exposure(
        snapshot=snapshot,
        refund_ids=[],
        risk_score=0.75,
    )

    assert (
        exposure.realized_suspicious_amount.amount_paise
        == 0
    )

    assert (
        exposure.pending_refund_exposure.amount_paise
        == 0
    )

    assert (
        exposure.remaining_refundable_exposure.amount_paise
        == 0
    )


def test_invalid_risk_score_raises_value_error() -> None:
    """Risk score must remain in the valid scoring range."""

    snapshot, _, processed_refund_id, _, _ = (
        _build_snapshot()
    )

    with pytest.raises(ValueError):
        compute_financial_exposure(
            snapshot=snapshot,
            refund_ids=[processed_refund_id],
            risk_score=1.1,
        )


def test_missing_refund_raises_value_error() -> None:
    """Selected refunds must exist in the reconstruction snapshot."""

    snapshot, _, _, _, _ = _build_snapshot()

    with pytest.raises(ValueError):
        compute_financial_exposure(
            snapshot=snapshot,
            refund_ids=[RefundId.generate()],
            risk_score=0.5,
        )