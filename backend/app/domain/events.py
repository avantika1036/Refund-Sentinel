"""
Domain event definitions for Refund Sentinel.

Events are immutable records of facts that have occurred. The financial state
engine (Phase 2) processes ordered sequences of events to reconstruct the
authoritative financial state of each payment.

Event ordering
--------------
The financial state engine uses occurred_at — the time the fact occurred in
the originating system — to reconstruct the canonical order of events for a
given payment. received_at is ingestion provenance only. These two timestamps
must never be conflated:

  occurred_at: when the business fact happened (authoritative for ordering)
  received_at: when our ingestion layer saw the event (provenance/audit)

A batch of simulator events generated and ingested at the same wall-clock
moment may have occurred_at values spread across hours or days. Sorting by
received_at in that case would produce a fundamentally incorrect event order.

The state engine (Phase 2) is responsible for sorting by occurred_at when
reconstructing per-payment event sequences.

Timestamp anomaly vs domain validity
-------------------------------------
A suspicious relationship between received_at and occurred_at (e.g.
received_at appears significantly before occurred_at, beyond plausible clock
skew) is a provenance anomaly, not a domain invalidity. The business fact
still occurred. Rejecting the event entirely at the domain layer based on
this anomaly would be wrong — the event is real, even if its timestamps are
surprising.

Instead, EventEnvelope exposes has_timestamp_anomaly (a computed property)
and the CLOCK_SKEW_ALERT_SECONDS constant. The ingestion layer (Phase 2)
decides how to handle flagged events: accept with a warning, quarantine for
review, or escalate. That decision belongs to the ingestion boundary, not
the domain model.

Immutability
------------
All event models are frozen. A recorded event must not be mutated. If a
correction is needed, a new corrective event is appended to the event
sequence — not a mutation of an existing event.

Simulator-only events
---------------------
ORDER_SHIPPED and ORDER_DELIVERED originate from the simulator. They are not
claimed to come from Razorpay webhooks. See the architecture contract.
"""

from __future__ import annotations

from typing import Final, Union

from pydantic import BaseModel, model_validator

from backend.app.domain.enums import DataSource, EventType, RefundReasonCode
from backend.app.domain.identifiers import (
    AddressId,
    CustomerId,
    DeviceId,
    EventId,
    MerchantId,
    OrderId,
    PaymentId,
    RefundId,
    SessionId,
)
from backend.app.domain.value_objects import IpIdentifier, Money, UTCDateTime, UntrustedText


# ---------------------------------------------------------------------------
# Clock-skew alert threshold
# ---------------------------------------------------------------------------

# If received_at precedes occurred_at by more than this many seconds, the
# EventEnvelope will report has_timestamp_anomaly = True.
# This threshold is exposed for use by the ingestion layer (Phase 2).
# It does NOT cause events to be rejected at the domain layer.
CLOCK_SKEW_ALERT_SECONDS: Final[float] = 300.0


# ---------------------------------------------------------------------------
# Event envelope
# ---------------------------------------------------------------------------


class EventEnvelope(BaseModel):
    """
    Provenance metadata carried by every domain event.

    Fields
    ------
    event_id    : UUID idempotency key. Two events with the same event_id
                  are duplicates; the ingestion layer must process only the
                  first and record the second as a duplicate.
    event_type  : The canonical EventType for this event.
    occurred_at : When the business fact occurred (authoritative for ordering).
    received_at : When our ingestion layer received the event (provenance).
    source      : DataSource — determines validation rules at the ingestion
                  boundary (e.g. webhook signature verification for
                  RAZORPAY_WEBHOOK sources).

    has_timestamp_anomaly
    ---------------------
    A computed property (not stored) that returns True when received_at
    precedes occurred_at by more than CLOCK_SKEW_ALERT_SECONDS. This
    indicates a suspicious provenance relationship but does NOT make the
    event invalid. The ingestion layer decides how to handle anomalous events.

    Legitimate causes of timestamp anomalies:
    - Simulator generating past-time events and loading them in batch
    - Delayed webhook delivery followed by a corrected timestamp
    - Clock skew between the originating system and our ingestion host
    """

    model_config = {"frozen": True}

    event_id: EventId
    event_type: EventType
    occurred_at: UTCDateTime
    received_at: UTCDateTime
    source: DataSource

    @property
    def has_timestamp_anomaly(self) -> bool:
        """
        Returns True if received_at precedes occurred_at by more than
        CLOCK_SKEW_ALERT_SECONDS, indicating a suspicious provenance
        relationship.

        This is an observation, not a rejection. The event is structurally
        valid regardless of this flag. The ingestion layer (Phase 2) is
        responsible for deciding how to handle flagged events.
        """
        if self.received_at < self.occurred_at:
            skew_seconds = self.occurred_at.seconds_since(self.received_at)
            return skew_seconds > CLOCK_SKEW_ALERT_SECONDS
        return False


# ---------------------------------------------------------------------------
# Per-event payload models
# ---------------------------------------------------------------------------


class OrderCreatedPayload(BaseModel):
    model_config = {"frozen": True}

    order_id: OrderId
    merchant_id: MerchantId
    customer_id: CustomerId
    amount: Money
    # May be absent if the customer has not yet confirmed the delivery address.
    shipping_address_id: AddressId | None = None


class PaymentCreatedPayload(BaseModel):
    model_config = {"frozen": True}

    payment_id: PaymentId
    order_id: OrderId
    merchant_id: MerchantId
    customer_id: CustomerId
    amount: Money
    # Optional context identifiers for graph edge construction.
    device_id: DeviceId | None = None
    session_id: SessionId | None = None
    ip_identifier: IpIdentifier | None = None


class PaymentCapturedPayload(BaseModel):
    model_config = {"frozen": True}

    payment_id: PaymentId
    merchant_id: MerchantId
    captured_amount: Money
    captured_at: UTCDateTime


class PaymentFailedPayload(BaseModel):
    model_config = {"frozen": True}

    payment_id: PaymentId
    merchant_id: MerchantId
    failed_at: UTCDateTime
    failure_reason: str | None = None


class RefundRequestedPayload(BaseModel):
    model_config = {"frozen": True}

    refund_id: RefundId
    payment_id: PaymentId
    order_id: OrderId
    merchant_id: MerchantId
    customer_id: CustomerId
    amount: Money
    reason_code: RefundReasonCode
    # Customer-supplied free text. Explicitly typed as untrusted.
    reason_text: UntrustedText | None = None

    @model_validator(mode="after")
    def refund_amount_must_be_positive(self) -> "RefundRequestedPayload":
        if self.amount.is_zero():
            raise ValueError(
                "Refund amount in RefundRequestedPayload must be positive. "
                "Zero-amount refund requests are not valid."
            )
        return self


class RefundCreatedPayload(BaseModel):
    """
    The payment gateway has accepted the refund request.

    This does not mean funds have been returned. The PROCESSED event
    confirms actual fund transfer.
    """

    model_config = {"frozen": True}

    refund_id: RefundId
    payment_id: PaymentId
    merchant_id: MerchantId
    created_at: UTCDateTime


class RefundProcessedPayload(BaseModel):
    """
    Funds have been returned to the customer.

    processed_amount is the gateway-confirmed amount. The financial state
    engine uses processed_amount — not the originally requested amount —
    for all accounting. This is the authoritative financial fact for this
    refund event.
    """

    model_config = {"frozen": True}

    refund_id: RefundId
    payment_id: PaymentId
    merchant_id: MerchantId
    processed_at: UTCDateTime
    processed_amount: Money


class RefundFailedPayload(BaseModel):
    model_config = {"frozen": True}

    refund_id: RefundId
    payment_id: PaymentId
    merchant_id: MerchantId
    failed_at: UTCDateTime
    failure_reason: str | None = None


class OrderShippedPayload(BaseModel):
    """
    Simulator-generated event. NOT a Razorpay webhook event.
    Represents the merchant's OMS confirming shipment.
    """

    model_config = {"frozen": True}

    order_id: OrderId
    merchant_id: MerchantId
    shipped_at: UTCDateTime
    carrier: str | None = None


class OrderDeliveredPayload(BaseModel):
    """
    Simulator-generated event. NOT a Razorpay webhook event.
    Represents delivery confirmation from the merchant's OMS.
    """

    model_config = {"frozen": True}

    order_id: OrderId
    merchant_id: MerchantId
    delivered_at: UTCDateTime


# ---------------------------------------------------------------------------
# Typed domain events
# ---------------------------------------------------------------------------


class DomainEvent(BaseModel):
    """
    Base class for all typed domain events.

    Each concrete subclass pairs an EventEnvelope with a typed payload.
    The event_type on the envelope must match the concrete event class —
    enforced by a model_validator on each subclass.
    """

    model_config = {"frozen": True}

    envelope: EventEnvelope


class OrderCreatedEvent(DomainEvent):
    model_config = {"frozen": True}
    payload: OrderCreatedPayload

    @model_validator(mode="after")
    def envelope_type_matches(self) -> "OrderCreatedEvent":
        if self.envelope.event_type != EventType.ORDER_CREATED:
            raise ValueError(
                f"EventEnvelope.event_type must be ORDER_CREATED for OrderCreatedEvent, "
                f"got {self.envelope.event_type.value!r}."
            )
        return self


class PaymentCreatedEvent(DomainEvent):
    model_config = {"frozen": True}
    payload: PaymentCreatedPayload

    @model_validator(mode="after")
    def envelope_type_matches(self) -> "PaymentCreatedEvent":
        if self.envelope.event_type != EventType.PAYMENT_CREATED:
            raise ValueError(
                f"EventEnvelope.event_type must be PAYMENT_CREATED for PaymentCreatedEvent, "
                f"got {self.envelope.event_type.value!r}."
            )
        return self


class PaymentCapturedEvent(DomainEvent):
    model_config = {"frozen": True}
    payload: PaymentCapturedPayload

    @model_validator(mode="after")
    def envelope_type_matches(self) -> "PaymentCapturedEvent":
        if self.envelope.event_type != EventType.PAYMENT_CAPTURED:
            raise ValueError(
                f"EventEnvelope.event_type must be PAYMENT_CAPTURED for PaymentCapturedEvent, "
                f"got {self.envelope.event_type.value!r}."
            )
        return self


class PaymentFailedEvent(DomainEvent):
    model_config = {"frozen": True}
    payload: PaymentFailedPayload

    @model_validator(mode="after")
    def envelope_type_matches(self) -> "PaymentFailedEvent":
        if self.envelope.event_type != EventType.PAYMENT_FAILED:
            raise ValueError(
                f"EventEnvelope.event_type must be PAYMENT_FAILED for PaymentFailedEvent, "
                f"got {self.envelope.event_type.value!r}."
            )
        return self


class RefundRequestedEvent(DomainEvent):
    model_config = {"frozen": True}
    payload: RefundRequestedPayload

    @model_validator(mode="after")
    def envelope_type_matches(self) -> "RefundRequestedEvent":
        if self.envelope.event_type != EventType.REFUND_REQUESTED:
            raise ValueError(
                f"EventEnvelope.event_type must be REFUND_REQUESTED for RefundRequestedEvent, "
                f"got {self.envelope.event_type.value!r}."
            )
        return self


class RefundCreatedEvent(DomainEvent):
    model_config = {"frozen": True}
    payload: RefundCreatedPayload

    @model_validator(mode="after")
    def envelope_type_matches(self) -> "RefundCreatedEvent":
        if self.envelope.event_type != EventType.REFUND_CREATED:
            raise ValueError(
                f"EventEnvelope.event_type must be REFUND_CREATED for RefundCreatedEvent, "
                f"got {self.envelope.event_type.value!r}."
            )
        return self


class RefundProcessedEvent(DomainEvent):
    model_config = {"frozen": True}
    payload: RefundProcessedPayload

    @model_validator(mode="after")
    def envelope_type_matches(self) -> "RefundProcessedEvent":
        if self.envelope.event_type != EventType.REFUND_PROCESSED:
            raise ValueError(
                f"EventEnvelope.event_type must be REFUND_PROCESSED for RefundProcessedEvent, "
                f"got {self.envelope.event_type.value!r}."
            )
        return self


class RefundFailedEvent(DomainEvent):
    model_config = {"frozen": True}
    payload: RefundFailedPayload

    @model_validator(mode="after")
    def envelope_type_matches(self) -> "RefundFailedEvent":
        if self.envelope.event_type != EventType.REFUND_FAILED:
            raise ValueError(
                f"EventEnvelope.event_type must be REFUND_FAILED for RefundFailedEvent, "
                f"got {self.envelope.event_type.value!r}."
            )
        return self


class OrderShippedEvent(DomainEvent):
    model_config = {"frozen": True}
    payload: OrderShippedPayload

    @model_validator(mode="after")
    def envelope_type_matches(self) -> "OrderShippedEvent":
        if self.envelope.event_type != EventType.ORDER_SHIPPED:
            raise ValueError(
                f"EventEnvelope.event_type must be ORDER_SHIPPED for OrderShippedEvent, "
                f"got {self.envelope.event_type.value!r}."
            )
        return self


class OrderDeliveredEvent(DomainEvent):
    model_config = {"frozen": True}
    payload: OrderDeliveredPayload

    @model_validator(mode="after")
    def envelope_type_matches(self) -> "OrderDeliveredEvent":
        if self.envelope.event_type != EventType.ORDER_DELIVERED:
            raise ValueError(
                f"EventEnvelope.event_type must be ORDER_DELIVERED for OrderDeliveredEvent, "
                f"got {self.envelope.event_type.value!r}."
            )
        return self


# ---------------------------------------------------------------------------
# Union type for the ingestion layer
# ---------------------------------------------------------------------------

AnyDomainEvent = Union[
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
]