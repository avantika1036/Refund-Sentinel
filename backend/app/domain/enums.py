"""
Closed enumeration types for the Refund Sentinel domain.

All enumerations are string-valued so that they serialise predictably to JSON
and are readable in database records and log output.

Design rules
------------
- Enumerations are closed. New values require an explicit code change and a
  database migration. Unknown values from external systems are rejected at
  the ingestion boundary, not silently defaulted.
- Do not add a generic UNKNOWN or OTHER fallback value to PaymentStatus or
  RefundStatus. Unknown states from external systems must surface as
  ingestion errors, not be hidden.
- RefundReasonCode includes UNSPECIFIED because the reason field is optional
  in some payment gateway responses. UNSPECIFIED means the merchant or
  gateway did not supply a reason — it is different from "unknown reason".
"""

from enum import Enum


class Currency(str, Enum):
    """
    Supported currencies.

    Only INR is supported in the initial implementation. The Money value
    object carries currency explicitly, so adding a new currency in a future
    phase does not require changes to Money's structure — only to this enum
    and the relevant financial calculations.
    """

    INR = "INR"


class PaymentStatus(str, Enum):
    """
    Lifecycle states of a Payment.

    State machine (enforced in Payment.apply_transition):
        created ──► captured
        created ──► failed

    No other transitions are legal. A captured payment cannot be failed.
    A failed payment cannot be captured.
    """

    CREATED = "created"
    CAPTURED = "captured"
    FAILED = "failed"


class RefundStatus(str, Enum):
    """
    Lifecycle states of a Refund.

    State machine (enforced in Refund.apply_transition):
        requested ──► created ──► processed
        requested ──► failed
        created   ──► failed

    A processed refund is terminal. A failed refund is terminal.
    Skipping the 'created' state (requested → processed directly) is illegal
    because it would bypass the gateway acceptance step.
    """

    REQUESTED = "requested"
    CREATED = "created"
    PROCESSED = "processed"
    FAILED = "failed"


class RefundReasonCode(str, Enum):
    """
    Standardised reason codes for refund requests.

    Raw strings from payment gateways or customers must be mapped to these
    codes at the ingestion boundary. Unmappable codes are rejected or
    classified as UNSPECIFIED.

    UNSPECIFIED is the only permissible catch-all.
    """

    NOT_DELIVERED = "not_delivered"
    DEFECTIVE = "defective"
    WRONG_ITEM = "wrong_item"
    DUPLICATE_ORDER = "duplicate_order"
    CHANGED_MIND = "changed_mind"
    DAMAGED_IN_TRANSIT = "damaged_in_transit"
    PARTIAL_DELIVERY = "partial_delivery"
    QUALITY_NOT_AS_DESCRIBED = "quality_not_as_described"
    UNSPECIFIED = "unspecified"


class EventType(str, Enum):
    """
    Canonical event type names used in the domain event envelope.

    These are internal canonical names, not necessarily Razorpay webhook
    event names. The Razorpay integration layer (Phase 11) maps Razorpay
    event names to these canonical names.
    """

    ORDER_CREATED = "order.created"
    PAYMENT_CREATED = "payment.created"
    PAYMENT_CAPTURED = "payment.captured"
    PAYMENT_FAILED = "payment.failed"
    REFUND_REQUESTED = "refund.requested"
    REFUND_CREATED = "refund.created"
    REFUND_PROCESSED = "refund.processed"
    REFUND_FAILED = "refund.failed"
    # Shipping and delivery events originate from the simulator only.
    # They are NOT claimed to come from Razorpay webhooks.
    ORDER_SHIPPED = "order.shipped"
    ORDER_DELIVERED = "order.delivered"


class DataSource(str, Enum):
    """
    The origin of an ingested event.

    Used for provenance tracking in the event envelope. Affects how the
    ingestion layer validates the event (e.g. webhook signature verification
    is required for RAZORPAY_WEBHOOK sources only).
    """

    SIMULATOR = "simulator"
    RAZORPAY_WEBHOOK = "razorpay_webhook"
    MANUAL = "manual"


class PaymentMethod(str, Enum):
    """
    High-level payment method categories.

    Granular instrument details (card network, UPI VPA) are captured in
    PaymentInstrument, not here.
    """

    CARD = "card"
    UPI = "upi"
    NETBANKING = "netbanking"
    WALLET = "wallet"
    EMI = "emi"
    OTHER = "other"