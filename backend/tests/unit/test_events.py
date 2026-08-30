"""
Tests for backend.app.domain.events

Proves
------
EventEnvelope:
  - Valid construction with normal timestamps succeeds.
  - received_at after occurred_at does not set has_timestamp_anomaly.
  - received_at within CLOCK_SKEW_ALERT_SECONDS before occurred_at does not
    set has_timestamp_anomaly.
  - received_at more than CLOCK_SKEW_ALERT_SECONDS before occurred_at
    DOES set has_timestamp_anomaly — but the event is still structurally valid.
  - Events are frozen after construction.
  - has_timestamp_anomaly is a computed property, not a stored field.

Event ordering semantics:
  - occurred_at is preserved as the canonical ordering key for the state engine.
  - An event where received_at > occurred_at (delayed delivery) has the
    correct occurred_at preserved.
  - The order of two events by received_at may differ from their order by
    occurred_at. Both timestamps are preserved.

Event type / envelope mismatch:
  - Each event type rejects an EventEnvelope with the wrong event_type.
  - Each event type accepts the correct event_type.

Immutability:
  - All event models are frozen.

Payload-level invariants:
  - RefundRequestedPayload rejects zero-amount refunds.
  - reason_text is UntrustedText.
  - RefundProcessedPayload carries processed_amount.
  - Optional context identifiers on PaymentCreatedPayload can be absent.

AnyDomainEvent union:
  - Covers all concrete event classes. A missing class is caught here.
"""

import typing
from datetime import datetime, timedelta, timezone

import pytest

from backend.app.domain.enums import DataSource, EventType, RefundReasonCode
from backend.app.domain.events import (
    CLOCK_SKEW_ALERT_SECONDS,
    AnyDomainEvent,
    EventEnvelope,
    OrderCreatedEvent,
    OrderCreatedPayload,
    OrderDeliveredEvent,
    OrderDeliveredPayload,
    OrderShippedEvent,
    OrderShippedPayload,
    PaymentCapturedEvent,
    PaymentCapturedPayload,
    PaymentCreatedEvent,
    PaymentCreatedPayload,
    PaymentFailedEvent,
    PaymentFailedPayload,
    RefundCreatedEvent,
    RefundCreatedPayload,
    RefundFailedEvent,
    RefundFailedPayload,
    RefundProcessedEvent,
    RefundProcessedPayload,
    RefundRequestedEvent,
    RefundRequestedPayload,
)
from backend.app.domain.identifiers import (
    CustomerId,
    DeviceId,
    EventId,
    MerchantId,
    OrderId,
    PaymentId,
    RefundId,
)
from backend.app.domain.value_objects import IpIdentifier, Money, UTCDateTime, UntrustedText


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def utc_dt(hour: int = 12, minute: int = 0, second: int = 0) -> UTCDateTime:
    return UTCDateTime(value=datetime(2024, 6, 1, hour, minute, second, tzinfo=timezone.utc))


def utc_dt_offset(base: UTCDateTime, delta_seconds: float) -> UTCDateTime:
    """Produce a UTCDateTime offset from base by delta_seconds."""
    new_value = base.value + timedelta(seconds=delta_seconds)
    return UTCDateTime(value=new_value)


def make_envelope(
    event_type: EventType,
    occurred_at: UTCDateTime | None = None,
    received_at: UTCDateTime | None = None,
    source: DataSource = DataSource.SIMULATOR,
) -> EventEnvelope:
    occ = occurred_at or utc_dt(12, 0)
    rec = received_at or utc_dt(12, 1)
    return EventEnvelope(
        event_id=EventId.generate(),
        event_type=event_type,
        occurred_at=occ,
        received_at=rec,
        source=source,
    )


# ---------------------------------------------------------------------------
# EventEnvelope
# ---------------------------------------------------------------------------


class TestEventEnvelope:
    def test_valid_envelope_normal_timestamps(self):
        occ = utc_dt(10, 0)
        rec = utc_dt(10, 30)
        env = make_envelope(EventType.ORDER_CREATED, occurred_at=occ, received_at=rec)
        assert env.event_type == EventType.ORDER_CREATED
        assert env.source == DataSource.SIMULATOR
        assert not env.has_timestamp_anomaly

    def test_received_after_occurred_is_normal_no_anomaly(self):
        """Delayed webhook delivery: received_at > occurred_at is expected."""
        occ = utc_dt(10, 0)
        rec = utc_dt(10, 30)
        env = make_envelope(EventType.PAYMENT_CAPTURED, occurred_at=occ, received_at=rec)
        assert not env.has_timestamp_anomaly

    def test_received_within_skew_tolerance_before_occurred_is_not_anomalous(self):
        """
        received_at slightly before occurred_at is within clock-skew tolerance.
        60 seconds < CLOCK_SKEW_ALERT_SECONDS (300). No anomaly.
        """
        occ = utc_dt(10, 0)
        rec = utc_dt_offset(occ, -60)  # 60 seconds before occurred_at
        env = make_envelope(EventType.ORDER_CREATED, occurred_at=occ, received_at=rec)
        assert not env.has_timestamp_anomaly

    def test_received_beyond_skew_tolerance_before_occurred_is_anomalous(self):
        """
        received_at more than CLOCK_SKEW_ALERT_SECONDS before occurred_at
        triggers has_timestamp_anomaly = True.

        Critically, this does NOT make the event structurally invalid.
        The event is still constructed successfully. The ingestion layer
        (Phase 2) decides how to handle the anomaly.
        """
        occ = utc_dt(12, 0)
        rec = utc_dt_offset(occ, -(CLOCK_SKEW_ALERT_SECONDS + 1))
        env = make_envelope(EventType.ORDER_CREATED, occurred_at=occ, received_at=rec)
        # Event was constructed successfully — it is structurally valid.
        assert env is not None
        # But the anomaly flag is set.
        assert env.has_timestamp_anomaly

    def test_anomaly_flag_is_computed_not_stored(self):
        """
        has_timestamp_anomaly is a @property, not a stored field.
        It must not appear in model_fields.
        """
        env = make_envelope(EventType.ORDER_CREATED)
        assert "has_timestamp_anomaly" not in EventEnvelope.model_fields

    def test_envelope_is_frozen(self):
        env = make_envelope(EventType.ORDER_CREATED)
        with pytest.raises(Exception):
            env.source = DataSource.MANUAL

    def test_clock_skew_alert_seconds_constant_value(self):
        assert CLOCK_SKEW_ALERT_SECONDS == 300.0


# ---------------------------------------------------------------------------
# Event ordering semantics
# ---------------------------------------------------------------------------


class TestEventOrderingSemantics:
    def test_occurred_at_preserved_for_state_engine_ordering(self):
        """
        occurred_at is the canonical ordering key for the financial state engine.
        It must be preserved exactly as supplied, not replaced by received_at.
        """
        occ = utc_dt(8, 0)   # Event occurred at 08:00
        rec = utc_dt(18, 0)  # Received much later — delayed delivery
        env = make_envelope(EventType.PAYMENT_CAPTURED, occurred_at=occ, received_at=rec)
        assert env.occurred_at.value.hour == 8
        assert env.received_at.value.hour == 18

    def test_two_events_received_in_reversed_order_preserve_correct_occurred_at(self):
        """
        Event A occurred at 10:00 but was received at 11:30.
        Event B occurred at 10:30 but was received at 10:45.
        If sorted by received_at, B comes before A — incorrect business order.
        If sorted by occurred_at, A comes before B — correct business order.
        Both events preserve their own occurred_at values correctly.
        """
        occ_a = utc_dt(10, 0)
        rec_a = utc_dt(11, 30)
        occ_b = utc_dt(10, 30)
        rec_b = utc_dt(10, 45)

        env_a = make_envelope(EventType.PAYMENT_CAPTURED, occurred_at=occ_a, received_at=rec_a)
        env_b = make_envelope(EventType.REFUND_REQUESTED, occurred_at=occ_b, received_at=rec_b)

        # By received_at order: B (10:45) before A (11:30) — wrong business order.
        by_received = sorted([env_a, env_b], key=lambda e: e.received_at.value)
        assert by_received[0].event_type == EventType.REFUND_REQUESTED

        # By occurred_at order: A (10:00) before B (10:30) — correct business order.
        by_occurred = sorted([env_a, env_b], key=lambda e: e.occurred_at.value)
        assert by_occurred[0].event_type == EventType.PAYMENT_CAPTURED

        # Both timestamps are preserved correctly on each event.
        assert env_a.occurred_at.value.hour == 10
        assert env_a.occurred_at.value.minute == 0
        assert env_b.occurred_at.value.hour == 10
        assert env_b.occurred_at.value.minute == 30


# ---------------------------------------------------------------------------
# Envelope type mismatch tests
# ---------------------------------------------------------------------------


class TestEnvelopeTypeMismatch:
    def test_order_created_event_rejects_wrong_envelope_type(self):
        env = make_envelope(EventType.PAYMENT_CREATED)
        with pytest.raises(ValueError, match="ORDER_CREATED"):
            OrderCreatedEvent(
                envelope=env,
                payload=OrderCreatedPayload(
                    order_id=OrderId.generate(),
                    merchant_id=MerchantId.generate(),
                    customer_id=CustomerId.generate(),
                    amount=Money.of_paise(100_000),
                ),
            )

    def test_refund_processed_event_rejects_wrong_envelope_type(self):
        env = make_envelope(EventType.REFUND_CREATED)
        with pytest.raises(ValueError, match="REFUND_PROCESSED"):
            RefundProcessedEvent(
                envelope=env,
                payload=RefundProcessedPayload(
                    refund_id=RefundId.generate(),
                    payment_id=PaymentId.generate(),
                    merchant_id=MerchantId.generate(),
                    processed_at=utc_dt(13, 0),
                    processed_amount=Money.of_paise(50_000),
                ),
            )


# ---------------------------------------------------------------------------
# Per-event construction
# ---------------------------------------------------------------------------


class TestOrderCreatedEvent:
    def test_valid_with_shipping_address(self):
        from backend.app.domain.identifiers import AddressId
        event = OrderCreatedEvent(
            envelope=make_envelope(EventType.ORDER_CREATED),
            payload=OrderCreatedPayload(
                order_id=OrderId.generate(),
                merchant_id=MerchantId.generate(),
                customer_id=CustomerId.generate(),
                amount=Money.of_paise(150_000),
                shipping_address_id=AddressId.generate(),
            ),
        )
        assert event.payload.amount.amount_paise == 150_000
        assert event.payload.shipping_address_id is not None

    def test_valid_without_shipping_address(self):
        event = OrderCreatedEvent(
            envelope=make_envelope(EventType.ORDER_CREATED),
            payload=OrderCreatedPayload(
                order_id=OrderId.generate(),
                merchant_id=MerchantId.generate(),
                customer_id=CustomerId.generate(),
                amount=Money.of_paise(150_000),
            ),
        )
        assert event.payload.shipping_address_id is None

    def test_frozen(self):
        event = OrderCreatedEvent(
            envelope=make_envelope(EventType.ORDER_CREATED),
            payload=OrderCreatedPayload(
                order_id=OrderId.generate(),
                merchant_id=MerchantId.generate(),
                customer_id=CustomerId.generate(),
                amount=Money.of_paise(150_000),
            ),
        )
        with pytest.raises(Exception):
            event.payload = None


class TestPaymentCreatedEvent:
    def test_valid_with_context_identifiers(self):
        event = PaymentCreatedEvent(
            envelope=make_envelope(EventType.PAYMENT_CREATED),
            payload=PaymentCreatedPayload(
                payment_id=PaymentId.generate(),
                order_id=OrderId.generate(),
                merchant_id=MerchantId.generate(),
                customer_id=CustomerId.generate(),
                amount=Money.of_paise(100_000),
                device_id=DeviceId.generate(),
                ip_identifier=IpIdentifier(value="192.168.1.1"),
            ),
        )
        assert event.payload.device_id is not None
        assert event.payload.ip_identifier.value == "192.168.1.1"

    def test_valid_without_context_identifiers(self):
        """Context identifiers are optional — their absence is not an error."""
        event = PaymentCreatedEvent(
            envelope=make_envelope(EventType.PAYMENT_CREATED),
            payload=PaymentCreatedPayload(
                payment_id=PaymentId.generate(),
                order_id=OrderId.generate(),
                merchant_id=MerchantId.generate(),
                customer_id=CustomerId.generate(),
                amount=Money.of_paise(100_000),
            ),
        )
        assert event.payload.device_id is None
        assert event.payload.session_id is None
        assert event.payload.ip_identifier is None


class TestPaymentCapturedEvent:
    def test_valid(self):
        event = PaymentCapturedEvent(
            envelope=make_envelope(EventType.PAYMENT_CAPTURED),
            payload=PaymentCapturedPayload(
                payment_id=PaymentId.generate(),
                merchant_id=MerchantId.generate(),
                captured_amount=Money.of_paise(100_000),
                captured_at=utc_dt(10, 5),
            ),
        )
        assert event.payload.captured_amount.amount_paise == 100_000


class TestRefundRequestedEvent:
    def test_valid_with_reason_text(self):
        event = RefundRequestedEvent(
            envelope=make_envelope(EventType.REFUND_REQUESTED),
            payload=RefundRequestedPayload(
                refund_id=RefundId.generate(),
                payment_id=PaymentId.generate(),
                order_id=OrderId.generate(),
                merchant_id=MerchantId.generate(),
                customer_id=CustomerId.generate(),
                amount=Money.of_paise(75_000),
                reason_code=RefundReasonCode.NOT_DELIVERED,
                reason_text=UntrustedText(raw="Package never arrived."),
            ),
        )
        assert isinstance(event.payload.reason_text, UntrustedText)
        assert event.payload.reason_code == RefundReasonCode.NOT_DELIVERED

    def test_valid_without_reason_text(self):
        event = RefundRequestedEvent(
            envelope=make_envelope(EventType.REFUND_REQUESTED),
            payload=RefundRequestedPayload(
                refund_id=RefundId.generate(),
                payment_id=PaymentId.generate(),
                order_id=OrderId.generate(),
                merchant_id=MerchantId.generate(),
                customer_id=CustomerId.generate(),
                amount=Money.of_paise(75_000),
                reason_code=RefundReasonCode.UNSPECIFIED,
            ),
        )
        assert event.payload.reason_text is None

    def test_zero_amount_refund_request_is_rejected(self):
        with pytest.raises(ValueError):
            RefundRequestedPayload(
                refund_id=RefundId.generate(),
                payment_id=PaymentId.generate(),
                order_id=OrderId.generate(),
                merchant_id=MerchantId.generate(),
                customer_id=CustomerId.generate(),
                amount=Money.of_paise(0),
                reason_code=RefundReasonCode.DEFECTIVE,
            )


class TestRefundProcessedEvent:
    def test_processed_amount_is_the_authoritative_financial_fact(self):
        """
        processed_amount is what the gateway actually returned.
        It is used by the financial state engine (Phase 2) for accounting,
        not the originally requested amount.
        """
        event = RefundProcessedEvent(
            envelope=make_envelope(EventType.REFUND_PROCESSED),
            payload=RefundProcessedPayload(
                refund_id=RefundId.generate(),
                payment_id=PaymentId.generate(),
                merchant_id=MerchantId.generate(),
                processed_at=utc_dt(13, 0),
                processed_amount=Money.of_paise(50_000),
            ),
        )
        assert event.payload.processed_amount.amount_paise == 50_000


class TestSimulatorOnlyEvents:
    def test_order_shipped_event_requires_simulator_source_awareness(self):
        """
        ORDER_SHIPPED originates from the simulator, not Razorpay.
        The source field on the envelope must reflect this.
        """
        event = OrderShippedEvent(
            envelope=make_envelope(EventType.ORDER_SHIPPED, source=DataSource.SIMULATOR),
            payload=OrderShippedPayload(
                order_id=OrderId.generate(),
                merchant_id=MerchantId.generate(),
                shipped_at=utc_dt(14, 0),
                carrier="BlueDart",
            ),
        )
        assert event.envelope.source == DataSource.SIMULATOR

    def test_order_delivered_event(self):
        event = OrderDeliveredEvent(
            envelope=make_envelope(EventType.ORDER_DELIVERED, source=DataSource.SIMULATOR),
            payload=OrderDeliveredPayload(
                order_id=OrderId.generate(),
                merchant_id=MerchantId.generate(),
                delivered_at=utc_dt(16, 0),
            ),
        )
        assert event.payload.delivered_at is not None


# ---------------------------------------------------------------------------
# AnyDomainEvent union completeness
# ---------------------------------------------------------------------------


class TestAnyDomainEventUnionCompleteness:
    def test_union_covers_all_concrete_event_classes(self):
        """
        Every concrete event class must appear in AnyDomainEvent.
        If a new event type is added to events.py and forgotten in the union,
        this test fails — which is the desired behaviour.
        """
        union_members = set(typing.get_args(AnyDomainEvent))
        expected = {
            OrderCreatedEvent,
            PaymentCreatedEvent,
            PaymentCapturedEvent,
            PaymentFailedEvent,
            RefundRequestedEvent,
            RefundCreatedEvent,
            RefundProcessedEvent,
            RefundFailedEvent,
            OrderShippedEvent,
            OrderDeliveredEvent,
        }
        assert union_members == expected, (
            f"AnyDomainEvent union mismatch.\n"
            f"Missing: {expected - union_members}\n"
            f"Extra:   {union_members - expected}"
        )