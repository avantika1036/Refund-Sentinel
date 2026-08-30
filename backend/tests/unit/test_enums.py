"""
Tests for backend.app.domain.enums

Proves
------
- All enum members have the expected string values and serialise correctly.
- Enum members are constructible from their string values (used in deserialisation).
- Invalid values raise ValueError, confirming enumerations are closed.
- PaymentStatus and RefundStatus contain exactly the expected members,
  so an accidental future addition is caught.
"""

import pytest

from backend.app.domain.enums import (
    Currency,
    DataSource,
    EventType,
    PaymentMethod,
    PaymentStatus,
    RefundReasonCode,
    RefundStatus,
)


class TestCurrency:
    def test_inr_value(self):
        assert Currency.INR.value == "INR"

    def test_construct_from_string_value(self):
        assert Currency("INR") == Currency.INR

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            Currency("USD")


class TestPaymentStatus:
    def test_exact_member_set(self):
        """PaymentStatus must have exactly these members — no more, no fewer."""
        assert {s.value for s in PaymentStatus} == {"created", "captured", "failed"}

    def test_construct_from_string_value(self):
        assert PaymentStatus("created") == PaymentStatus.CREATED
        assert PaymentStatus("captured") == PaymentStatus.CAPTURED
        assert PaymentStatus("failed") == PaymentStatus.FAILED

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            PaymentStatus("pending")

    def test_invalid_value_raises_authorised(self):
        """'authorised' is a Razorpay-specific state not used in our domain model."""
        with pytest.raises(ValueError):
            PaymentStatus("authorised")


class TestRefundStatus:
    def test_exact_member_set(self):
        """RefundStatus must have exactly these members."""
        assert {s.value for s in RefundStatus} == {
            "requested", "created", "processed", "failed"
        }

    def test_construct_from_string_value(self):
        assert RefundStatus("requested") == RefundStatus.REQUESTED
        assert RefundStatus("created") == RefundStatus.CREATED
        assert RefundStatus("processed") == RefundStatus.PROCESSED
        assert RefundStatus("failed") == RefundStatus.FAILED

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            RefundStatus("approved")

    def test_invalid_value_raises_pending(self):
        with pytest.raises(ValueError):
            RefundStatus("pending")


class TestRefundReasonCode:
    def test_unspecified_exists_and_is_distinct_from_unknown(self):
        """
        UNSPECIFIED means no reason was supplied.
        There is no generic UNKNOWN fallback.
        """
        assert RefundReasonCode.UNSPECIFIED.value == "unspecified"
        with pytest.raises(ValueError):
            RefundReasonCode("unknown")

    def test_exact_member_set(self):
        expected = {
            "not_delivered",
            "defective",
            "wrong_item",
            "duplicate_order",
            "changed_mind",
            "damaged_in_transit",
            "partial_delivery",
            "quality_not_as_described",
            "unspecified",
        }
        assert {r.value for r in RefundReasonCode} == expected

    def test_construct_from_string_value(self):
        assert RefundReasonCode("defective") == RefundReasonCode.DEFECTIVE

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            RefundReasonCode("fraud")


class TestEventType:
    def test_order_created_value(self):
        assert EventType.ORDER_CREATED.value == "order.created"

    def test_refund_processed_value(self):
        assert EventType.REFUND_PROCESSED.value == "refund.processed"

    def test_simulator_only_events_present(self):
        """
        ORDER_SHIPPED and ORDER_DELIVERED are simulator-only events.
        Their presence in the enum is intentional and must not be removed.
        """
        assert EventType.ORDER_SHIPPED.value == "order.shipped"
        assert EventType.ORDER_DELIVERED.value == "order.delivered"

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            EventType("payment.blocked")

    def test_all_core_payment_refund_events_present(self):
        values = {e.value for e in EventType}
        for expected in [
            "order.created",
            "payment.created",
            "payment.captured",
            "payment.failed",
            "refund.requested",
            "refund.created",
            "refund.processed",
            "refund.failed",
        ]:
            assert expected in values, f"Missing EventType value: {expected}"


class TestDataSource:
    def test_all_members_present(self):
        values = {s.value for s in DataSource}
        assert "simulator" in values
        assert "razorpay_webhook" in values
        assert "manual" in values

    def test_construct_from_string_value(self):
        assert DataSource("simulator") == DataSource.SIMULATOR


class TestPaymentMethod:
    def test_exact_member_set(self):
        expected = {"card", "upi", "netbanking", "wallet", "emi", "other"}
        assert {m.value for m in PaymentMethod} == expected

    def test_other_is_a_catch_all_for_payment_methods(self):
        """
        PaymentMethod.OTHER exists for methods not yet enumerated.
        Note: RefundStatus and PaymentStatus do NOT have an OTHER/UNKNOWN
        fallback — their domains are closed. PaymentMethod is more open-ended.
        """
        assert PaymentMethod.OTHER.value == "other"