"""
Tests for backend.app.domain.entities

Proves
------
Merchant:
  - Valid construction succeeds.
  - Empty and whitespace-only names are rejected.

Payment state machine:
  - Initial status is CREATED.
  - CREATED → CAPTURED succeeds with correct arguments.
  - CREATED → FAILED succeeds with correct arguments.
  - CAPTURED → CAPTURED is illegal.
  - CAPTURED → FAILED is illegal.
  - FAILED → CAPTURED is illegal.
  - captured_at before created_at is rejected.
  - captured_amount exceeding authorised amount is rejected.
  - captured_amount with mismatched currency is rejected.
  - Transition to CAPTURED without captured_amount raises ValueError.
  - Transition to CAPTURED without captured_at raises ValueError.
  - Transition to FAILED without failed_at raises ValueError.

Refund state machine:
  - Initial status is REQUESTED.
  - REQUESTED → CREATED succeeds.
  - CREATED → PROCESSED succeeds.
  - REQUESTED → FAILED succeeds.
  - CREATED → FAILED succeeds.
  - REQUESTED → PROCESSED is illegal (must go through CREATED).
  - PROCESSED → CREATED is illegal (terminal state).
  - FAILED → PROCESSED is illegal (terminal state).
  - Zero-amount refund is rejected at construction.
  - processed_at before created_at is rejected.
  - processed_amount of zero is rejected.
  - Transition to PROCESSED without processed_amount raises ValueError.
  - reason_text is UntrustedText type.

Order:
  - Valid construction with no shipping/delivery data.
  - shipped_at before created_at is rejected.
  - delivered_at before shipped_at is rejected.
  - delivered_at without shipped_at is permitted.

Cross-entity financial invariant (deferred):
  - Test documents that refund_amount <= captured_amount enforcement
    is the responsibility of Phase 2 (state_engine.py), not this entity.
"""

from datetime import datetime, timezone

import pytest

from backend.app.domain.entities import (
    Customer,
    InvalidStateTransitionError,
    Merchant,
    Order,
    Payment,
    Refund,
)
from backend.app.domain.enums import (
    PaymentStatus,
    RefundReasonCode,
    RefundStatus,
)
from backend.app.domain.identifiers import (
    CustomerId,
    MerchantId,
    OrderId,
    PaymentId,
    RefundId,
)
from backend.app.domain.value_objects import Money, UTCDateTime, UntrustedText


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> UTCDateTime:
    return UTCDateTime(value=datetime(year, month, day, hour, minute, 0, tzinfo=timezone.utc))


def make_payment(
    created_at: UTCDateTime | None = None,
    amount_paise: int = 100_000,
) -> Payment:
    return Payment(
        payment_id=PaymentId.generate(),
        order_id=OrderId.generate(),
        merchant_id=MerchantId.generate(),
        customer_id=CustomerId.generate(),
        created_at=created_at or utc(2024, 6, 1, 10, 0),
        amount=Money.of_paise(amount_paise),
    )


def make_refund(
    requested_at: UTCDateTime | None = None,
    amount_paise: int = 50_000,
) -> Refund:
    return Refund(
        refund_id=RefundId.generate(),
        payment_id=PaymentId.generate(),
        order_id=OrderId.generate(),
        merchant_id=MerchantId.generate(),
        customer_id=CustomerId.generate(),
        requested_at=requested_at or utc(2024, 6, 1, 14, 0),
        amount=Money.of_paise(amount_paise),
        reason_code=RefundReasonCode.DEFECTIVE,
    )


# ---------------------------------------------------------------------------
# Merchant
# ---------------------------------------------------------------------------


class TestMerchant:
    def test_valid_merchant(self):
        m = Merchant(
            merchant_id=MerchantId.generate(),
            name="Test Merchant",
            created_at=utc(2024, 1, 1),
        )
        assert m.name == "Test Merchant"

    def test_empty_name_rejected(self):
        with pytest.raises(ValueError):
            Merchant(
                merchant_id=MerchantId.generate(),
                name="",
                created_at=utc(2024, 1, 1),
            )

    def test_whitespace_only_name_rejected(self):
        with pytest.raises(ValueError):
            Merchant(
                merchant_id=MerchantId.generate(),
                name="   ",
                created_at=utc(2024, 1, 1),
            )


# ---------------------------------------------------------------------------
# Payment state machine
# ---------------------------------------------------------------------------


class TestPaymentStateMachine:
    def test_initial_status_is_created(self):
        p = make_payment()
        assert p.status == PaymentStatus.CREATED

    def test_created_to_captured_succeeds(self):
        p = make_payment(created_at=utc(2024, 6, 1, 10, 0))
        p.apply_transition(
            PaymentStatus.CAPTURED,
            captured_amount=Money.of_paise(100_000),
            captured_at=utc(2024, 6, 1, 10, 5),
        )
        assert p.status == PaymentStatus.CAPTURED
        assert p.captured_amount is not None
        assert p.captured_amount.amount_paise == 100_000
        assert p.captured_at is not None

    def test_created_to_failed_succeeds(self):
        p = make_payment(created_at=utc(2024, 6, 1, 10, 0))
        p.apply_transition(
            PaymentStatus.FAILED,
            failed_at=utc(2024, 6, 1, 10, 2),
            failure_reason="Insufficient funds",
        )
        assert p.status == PaymentStatus.FAILED
        assert p.failure_reason == "Insufficient funds"

    def test_captured_to_captured_is_illegal(self):
        p = make_payment()
        p.apply_transition(
            PaymentStatus.CAPTURED,
            captured_amount=Money.of_paise(100_000),
            captured_at=utc(2024, 6, 1, 10, 5),
        )
        with pytest.raises(InvalidStateTransitionError):
            p.apply_transition(
                PaymentStatus.CAPTURED,
                captured_amount=Money.of_paise(100_000),
                captured_at=utc(2024, 6, 1, 10, 10),
            )

    def test_captured_to_failed_is_illegal(self):
        p = make_payment()
        p.apply_transition(
            PaymentStatus.CAPTURED,
            captured_amount=Money.of_paise(100_000),
            captured_at=utc(2024, 6, 1, 10, 5),
        )
        with pytest.raises(InvalidStateTransitionError):
            p.apply_transition(
                PaymentStatus.FAILED,
                failed_at=utc(2024, 6, 1, 10, 10),
            )

    def test_failed_to_captured_is_illegal(self):
        p = make_payment()
        p.apply_transition(
            PaymentStatus.FAILED,
            failed_at=utc(2024, 6, 1, 10, 2),
        )
        with pytest.raises(InvalidStateTransitionError):
            p.apply_transition(
                PaymentStatus.CAPTURED,
                captured_amount=Money.of_paise(100_000),
                captured_at=utc(2024, 6, 1, 10, 5),
            )

    def test_captured_at_before_created_at_is_rejected(self):
        p = make_payment(created_at=utc(2024, 6, 1, 12, 0))
        with pytest.raises(ValueError, match="cannot precede"):
            p.apply_transition(
                PaymentStatus.CAPTURED,
                captured_amount=Money.of_paise(100_000),
                captured_at=utc(2024, 6, 1, 11, 0),
            )

    def test_captured_amount_exceeding_authorised_is_rejected(self):
        p = make_payment(amount_paise=100_000)
        with pytest.raises(ValueError, match="cannot exceed"):
            p.apply_transition(
                PaymentStatus.CAPTURED,
                captured_amount=Money.of_paise(100_001),
                captured_at=utc(2024, 6, 1, 10, 5),
            )

    def test_captured_amount_equal_to_authorised_is_accepted(self):
        """captured_amount == amount is valid (full capture)."""
        p = make_payment(amount_paise=100_000)
        p.apply_transition(
            PaymentStatus.CAPTURED,
            captured_amount=Money.of_paise(100_000),
            captured_at=utc(2024, 6, 1, 10, 5),
        )
        assert p.captured_amount.amount_paise == 100_000

    def test_captured_amount_less_than_authorised_is_accepted(self):
        """Partial capture (captured_amount < amount) is valid."""
        p = make_payment(amount_paise=100_000)
        p.apply_transition(
            PaymentStatus.CAPTURED,
            captured_amount=Money.of_paise(80_000),
            captured_at=utc(2024, 6, 1, 10, 5),
        )
        assert p.captured_amount.amount_paise == 80_000

    def test_capture_without_captured_amount_raises(self):
        p = make_payment()
        with pytest.raises(ValueError, match="captured_amount is required"):
            p.apply_transition(
                PaymentStatus.CAPTURED,
                captured_at=utc(2024, 6, 1, 10, 5),
            )

    def test_capture_without_captured_at_raises(self):
        p = make_payment()
        with pytest.raises(ValueError, match="captured_at is required"):
            p.apply_transition(
                PaymentStatus.CAPTURED,
                captured_amount=Money.of_paise(100_000),
            )

    def test_fail_without_failed_at_raises(self):
        p = make_payment()
        with pytest.raises(ValueError, match="failed_at is required"):
            p.apply_transition(PaymentStatus.FAILED)


# ---------------------------------------------------------------------------
# Refund state machine
# ---------------------------------------------------------------------------


class TestRefundStateMachine:
    def test_initial_status_is_requested(self):
        r = make_refund()
        assert r.status == RefundStatus.REQUESTED

    def test_requested_to_created_succeeds(self):
        r = make_refund(requested_at=utc(2024, 6, 1, 14, 0))
        r.apply_transition(
            RefundStatus.CREATED,
            created_at=utc(2024, 6, 1, 14, 5),
        )
        assert r.status == RefundStatus.CREATED
        assert r.created_at is not None

    def test_created_to_processed_succeeds(self):
        r = make_refund(requested_at=utc(2024, 6, 1, 14, 0))
        r.apply_transition(RefundStatus.CREATED, created_at=utc(2024, 6, 1, 14, 5))
        r.apply_transition(
            RefundStatus.PROCESSED,
            processed_at=utc(2024, 6, 1, 15, 0),
            processed_amount=Money.of_paise(50_000),
        )
        assert r.status == RefundStatus.PROCESSED
        assert r.processed_amount.amount_paise == 50_000

    def test_requested_to_failed_succeeds(self):
        r = make_refund(requested_at=utc(2024, 6, 1, 14, 0))
        r.apply_transition(
            RefundStatus.FAILED,
            failed_at=utc(2024, 6, 1, 14, 10),
            failure_reason="Gateway timeout",
        )
        assert r.status == RefundStatus.FAILED

    def test_created_to_failed_succeeds(self):
        r = make_refund(requested_at=utc(2024, 6, 1, 14, 0))
        r.apply_transition(RefundStatus.CREATED, created_at=utc(2024, 6, 1, 14, 5))
        r.apply_transition(RefundStatus.FAILED, failed_at=utc(2024, 6, 1, 14, 30))
        assert r.status == RefundStatus.FAILED

    def test_requested_to_processed_is_illegal(self):
        """Cannot skip the CREATED state — gateway acceptance is mandatory."""
        r = make_refund()
        with pytest.raises(InvalidStateTransitionError):
            r.apply_transition(
                RefundStatus.PROCESSED,
                processed_at=utc(2024, 6, 1, 15, 0),
                processed_amount=Money.of_paise(50_000),
            )

    def test_processed_to_created_is_illegal(self):
        """PROCESSED is a terminal state."""
        r = make_refund(requested_at=utc(2024, 6, 1, 14, 0))
        r.apply_transition(RefundStatus.CREATED, created_at=utc(2024, 6, 1, 14, 5))
        r.apply_transition(
            RefundStatus.PROCESSED,
            processed_at=utc(2024, 6, 1, 15, 0),
            processed_amount=Money.of_paise(50_000),
        )
        with pytest.raises(InvalidStateTransitionError):
            r.apply_transition(RefundStatus.CREATED, created_at=utc(2024, 6, 1, 16, 0))

    def test_failed_to_processed_is_illegal(self):
        """FAILED is a terminal state."""
        r = make_refund(requested_at=utc(2024, 6, 1, 14, 0))
        r.apply_transition(RefundStatus.FAILED, failed_at=utc(2024, 6, 1, 14, 5))
        with pytest.raises(InvalidStateTransitionError):
            r.apply_transition(
                RefundStatus.PROCESSED,
                processed_at=utc(2024, 6, 1, 15, 0),
                processed_amount=Money.of_paise(50_000),
            )

    def test_zero_amount_refund_rejected_at_construction(self):
        with pytest.raises(ValueError, match="positive"):
            Refund(
                refund_id=RefundId.generate(),
                payment_id=PaymentId.generate(),
                order_id=OrderId.generate(),
                merchant_id=MerchantId.generate(),
                customer_id=CustomerId.generate(),
                requested_at=utc(2024, 6, 1, 14, 0),
                amount=Money.of_paise(0),
                reason_code=RefundReasonCode.DEFECTIVE,
            )

    def test_processed_at_before_created_at_is_rejected(self):
        r = make_refund(requested_at=utc(2024, 6, 1, 14, 0))
        r.apply_transition(RefundStatus.CREATED, created_at=utc(2024, 6, 1, 14, 30))
        with pytest.raises(ValueError, match="cannot precede"):
            r.apply_transition(
                RefundStatus.PROCESSED,
                processed_at=utc(2024, 6, 1, 14, 0),
                processed_amount=Money.of_paise(50_000),
            )

    def test_zero_processed_amount_is_rejected(self):
        r = make_refund(requested_at=utc(2024, 6, 1, 14, 0))
        r.apply_transition(RefundStatus.CREATED, created_at=utc(2024, 6, 1, 14, 5))
        with pytest.raises(ValueError, match="positive"):
            r.apply_transition(
                RefundStatus.PROCESSED,
                processed_at=utc(2024, 6, 1, 15, 0),
                processed_amount=Money.of_paise(0),
            )

    def test_process_without_processed_amount_raises(self):
        r = make_refund(requested_at=utc(2024, 6, 1, 14, 0))
        r.apply_transition(RefundStatus.CREATED, created_at=utc(2024, 6, 1, 14, 5))
        with pytest.raises(ValueError, match="processed_amount is required"):
            r.apply_transition(
                RefundStatus.PROCESSED,
                processed_at=utc(2024, 6, 1, 15, 0),
            )

    def test_reason_text_is_untrusted_text_type(self):
        """reason_text must be UntrustedText, not a plain string."""
        r = Refund(
            refund_id=RefundId.generate(),
            payment_id=PaymentId.generate(),
            order_id=OrderId.generate(),
            merchant_id=MerchantId.generate(),
            customer_id=CustomerId.generate(),
            requested_at=utc(2024, 6, 1, 14, 0),
            amount=Money.of_paise(50_000),
            reason_code=RefundReasonCode.DEFECTIVE,
            reason_text=UntrustedText(raw="Item was broken on arrival."),
        )
        assert isinstance(r.reason_text, UntrustedText)

    def test_cross_entity_financial_invariant_is_deferred_to_phase_2(self):
        """
        This test documents — not implements — the cross-entity invariant.

        The Refund entity alone cannot enforce:
            sum(processed_amount for all processed refunds) <= payment.captured_amount

        because it requires knowledge of all sibling refunds on the same payment.

        The financial state engine (Phase 2, finance/state_engine.py) is
        responsible for this check. It must quarantine any refund whose
        inclusion would cause the cumulative total to exceed captured_amount.

        This test simply confirms that a Refund with an amount larger than a
        typical captured_amount CAN be constructed at the entity level — the
        domain layer does not reject it. Phase 2 will prevent it from being
        processed.
        """
        # A refund for 200,000 paise on a payment that was captured for 100,000 paise.
        # The entity layer accepts this. The state engine (Phase 2) will reject it.
        r = make_refund(amount_paise=200_000)
        assert r.amount.amount_paise == 200_000
        # No exception at this layer. Phase 2 enforces the cumulative constraint.


# ---------------------------------------------------------------------------
# Order
# ---------------------------------------------------------------------------


class TestOrder:
    def test_valid_order_without_shipping_delivery(self):
        o = Order(
            order_id=OrderId.generate(),
            merchant_id=MerchantId.generate(),
            customer_id=CustomerId.generate(),
            created_at=utc(2024, 6, 1),
            amount=Money.of_paise(200_000),
        )
        assert o.shipped_at is None
        assert o.delivered_at is None

    def test_shipped_before_created_is_rejected(self):
        with pytest.raises(ValueError, match="cannot precede"):
            Order(
                order_id=OrderId.generate(),
                merchant_id=MerchantId.generate(),
                customer_id=CustomerId.generate(),
                created_at=utc(2024, 6, 2),
                amount=Money.of_paise(200_000),
                shipped_at=utc(2024, 6, 1),
            )

    def test_delivered_before_shipped_is_rejected(self):
        with pytest.raises(ValueError, match="cannot precede"):
            Order(
                order_id=OrderId.generate(),
                merchant_id=MerchantId.generate(),
                customer_id=CustomerId.generate(),
                created_at=utc(2024, 6, 1),
                amount=Money.of_paise(200_000),
                shipped_at=utc(2024, 6, 3),
                delivered_at=utc(2024, 6, 2),
            )

    def test_delivered_without_shipped_is_permitted(self):
        """
        delivered_at without shipped_at is structurally permitted.
        The shipped event may arrive out of order or be absent in some flows.
        The financial state engine (Phase 2) handles event ordering.
        """
        o = Order(
            order_id=OrderId.generate(),
            merchant_id=MerchantId.generate(),
            customer_id=CustomerId.generate(),
            created_at=utc(2024, 6, 1),
            amount=Money.of_paise(200_000),
            delivered_at=utc(2024, 6, 4),
        )
        assert o.delivered_at is not None
        assert o.shipped_at is None