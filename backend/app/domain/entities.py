"""
Core domain entities for Refund Sentinel.

Entities have identity (expressed through typed IDs) and mutable lifecycle
state governed by explicit state machines. The state machines live here,
not in the database or API layers.

Design decisions
----------------
- Entities are Pydantic models with frozen=False because they have explicit
  state transitions. Mutation is only permitted through named transition
  methods, never through direct field assignment on the caller's side.
- Derived / computed fields (e.g. remaining_refundable) are NOT stored here.
  They are computed by the financial state engine in Phase 2.
- Risk scores, fraud labels, and ML features are NOT stored here. The domain
  layer must not know about the risk layer.
- ContextIdentifiers bundles the optional device/session/network signals
  that support graph construction in Phase 4.

Cross-entity financial invariant — PHASE 2 RESPONSIBILITY
----------------------------------------------------------
Refund.amount is validated to be positive (non-zero) here. However, the
invariant:

    sum(processed_amount for all processed refunds on a payment)
        <= payment.captured_amount

cannot be enforced by Refund alone because it requires knowledge of all
sibling refunds on the same payment. This cumulative invariant MUST be
enforced by the financial state engine in Phase 2 (finance/state_engine.py).
The state engine is the authoritative source for whether a new refund would
cause the cumulative total to exceed the captured amount. Any refund that
would violate this invariant must be quarantined by the state engine, not
silently accepted.
"""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, field_validator, model_validator

from backend.app.domain.enums import (
    PaymentMethod,
    PaymentStatus,
    RefundReasonCode,
    RefundStatus,
)
from backend.app.domain.identifiers import (
    AddressId,
    CustomerId,
    DeviceId,
    InstrumentId,
    MerchantId,
    OrderId,
    PaymentId,
    RefundId,
    SessionId,
)
from backend.app.domain.value_objects import IpIdentifier, Money, UTCDateTime, UntrustedText


# ---------------------------------------------------------------------------
# State machine transition tables
# ---------------------------------------------------------------------------

_PAYMENT_LEGAL_TRANSITIONS: Final[frozenset[tuple[PaymentStatus, PaymentStatus]]] = frozenset(
    {
        (PaymentStatus.CREATED, PaymentStatus.CAPTURED),
        (PaymentStatus.CREATED, PaymentStatus.FAILED),
    }
)

_REFUND_LEGAL_TRANSITIONS: Final[frozenset[tuple[RefundStatus, RefundStatus]]] = frozenset(
    {
        (RefundStatus.REQUESTED, RefundStatus.CREATED),
        (RefundStatus.REQUESTED, RefundStatus.FAILED),
        (RefundStatus.CREATED, RefundStatus.PROCESSED),
        (RefundStatus.CREATED, RefundStatus.FAILED),
    }
)


class InvalidStateTransitionError(Exception):
    """
    Raised when an entity receives a state transition that is not permitted
    by the domain's state machine definition.

    This is a domain error, not an infrastructure error. It must not be
    silently caught and normalised upstream.
    """

    def __init__(self, entity: str, from_status: str, to_status: str) -> None:
        super().__init__(
            f"Invalid state transition for {entity}: "
            f"{from_status!r} → {to_status!r} is not a legal transition."
        )
        self.entity = entity
        self.from_status = from_status
        self.to_status = to_status


# ---------------------------------------------------------------------------
# Merchant
# ---------------------------------------------------------------------------


class Merchant(BaseModel):
    """
    A merchant account on the Razorpay platform.

    MerchantId is included so the schema supports future multi-tenancy
    (P2) without a breaking change.
    """

    model_config = {"frozen": True}

    merchant_id: MerchantId
    name: str
    created_at: UTCDateTime

    @field_validator("name")
    @classmethod
    def name_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Merchant name must not be empty.")
        return v.strip()


# ---------------------------------------------------------------------------
# Customer
# ---------------------------------------------------------------------------


class Customer(BaseModel):
    """
    A customer who places orders with a merchant.

    Graph edges to Device, Address, and email identifiers are established
    through their presence on Payment and Order entities, not stored directly
    on Customer. This keeps Customer clean and avoids denormalisation.

    email_hash and phone_hash are hashed identifiers used for graph edge
    construction. Raw values are not stored in the domain model. Hashing is
    performed at the ingestion boundary. We acknowledge that hashing alone
    does not guarantee privacy for low-cardinality spaces (see LIMITATIONS.md)
    but it reduces sensitivity of data in the risk database.
    """

    model_config = {"frozen": True}

    customer_id: CustomerId
    merchant_id: MerchantId
    registered_at: UTCDateTime
    email_hash: str | None = None
    phone_hash: str | None = None

    @field_validator("email_hash", "phone_hash", mode="before")
    @classmethod
    def hash_must_be_nonempty_if_present(cls, v: object) -> object:
        if v is not None and (not isinstance(v, str) or not v.strip()):
            raise ValueError("Hashed identifier must be a non-empty string if provided.")
        return v


# ---------------------------------------------------------------------------
# PaymentInstrument
# ---------------------------------------------------------------------------


class PaymentInstrument(BaseModel):
    """
    The payment instrument used for a specific payment.

    instrument_hash is a hashed token — never the raw card number or UPI VPA.
    """

    model_config = {"frozen": True}

    instrument_id: InstrumentId
    method: PaymentMethod
    instrument_hash: str | None = None


# ---------------------------------------------------------------------------
# ContextIdentifiers
# ---------------------------------------------------------------------------


class ContextIdentifiers(BaseModel):
    """
    Device, session, and network identifiers captured at payment time.

    These are the raw materials for graph edge construction in Phase 4.
    All fields are optional because not all payment channels supply all
    identifiers. Missing fields reduce graph richness but do not invalidate
    the payment record.

    None of these identifiers is a high-confidence fraud signal on its own.
    They become meaningful only when combined with behavioral evidence
    (Phase 4 graph analysis + behavioral confirmation gate).
    """

    model_config = {"frozen": True}

    device_id: DeviceId | None = None
    session_id: SessionId | None = None
    ip_identifier: IpIdentifier | None = None


# ---------------------------------------------------------------------------
# Order
# ---------------------------------------------------------------------------


class Order(BaseModel):
    """
    A merchant order placed by a customer.

    Shipping and delivery timestamps are supplied by the simulator (not by
    Razorpay webhooks — see architecture contract). They are optional because
    not all test scenarios include delivery confirmation.

    shipping_address_id identifies the delivery address for graph edge
    construction. Two customers shipping to the same AddressId are connected
    in the structural entity graph.
    """

    model_config = {"frozen": True}

    order_id: OrderId
    merchant_id: MerchantId
    customer_id: CustomerId
    created_at: UTCDateTime
    amount: Money
    shipping_address_id: AddressId | None = None
    shipped_at: UTCDateTime | None = None
    delivered_at: UTCDateTime | None = None

    @model_validator(mode="after")
    def shipped_not_before_created(self) -> "Order":
        if self.shipped_at is not None and self.shipped_at < self.created_at:
            raise ValueError(
                f"shipped_at ({self.shipped_at}) cannot precede "
                f"order created_at ({self.created_at})."
            )
        return self

    @model_validator(mode="after")
    def delivered_not_before_shipped(self) -> "Order":
        if self.delivered_at is not None and self.shipped_at is not None:
            if self.delivered_at < self.shipped_at:
                raise ValueError(
                    f"delivered_at ({self.delivered_at}) cannot precede "
                    f"shipped_at ({self.shipped_at})."
                )
        return self


# ---------------------------------------------------------------------------
# Payment
# ---------------------------------------------------------------------------


class Payment(BaseModel):
    """
    A payment attempt associated with an order.

    State machine (enforced by apply_transition):
        created ──► captured
        created ──► failed

    captured_amount is the amount actually captured, set when status
    transitions to CAPTURED. It may be less than or equal to amount (the
    authorised amount) but must never exceed it.

    captured_at must not precede created_at. This ordering is critical for
    the capture_to_refund_latency feature computed in Phase 5.
    """

    model_config = {"frozen": False}  # Mutable to allow controlled state transitions.

    payment_id: PaymentId
    order_id: OrderId
    merchant_id: MerchantId
    customer_id: CustomerId
    created_at: UTCDateTime
    amount: Money
    status: PaymentStatus = PaymentStatus.CREATED

    captured_amount: Money | None = None
    captured_at: UTCDateTime | None = None
    failed_at: UTCDateTime | None = None
    failure_reason: str | None = None

    instrument: PaymentInstrument | None = None
    context: ContextIdentifiers | None = None

    def apply_transition(
        self,
        new_status: PaymentStatus,
        *,
        captured_amount: Money | None = None,
        captured_at: UTCDateTime | None = None,
        failed_at: UTCDateTime | None = None,
        failure_reason: str | None = None,
    ) -> None:
        """
        Transition this payment to a new lifecycle status.

        Raises InvalidStateTransitionError for illegal transitions.
        Raises ValueError for missing required arguments per transition type.
        """
        if (self.status, new_status) not in _PAYMENT_LEGAL_TRANSITIONS:
            raise InvalidStateTransitionError(
                entity="Payment",
                from_status=self.status.value,
                to_status=new_status.value,
            )

        if new_status == PaymentStatus.CAPTURED:
            if captured_amount is None:
                raise ValueError(
                    "captured_amount is required when transitioning Payment to CAPTURED."
                )
            if captured_at is None:
                raise ValueError(
                    "captured_at is required when transitioning Payment to CAPTURED."
                )
            if captured_amount.currency != self.amount.currency:
                raise ValueError(
                    f"captured_amount currency ({captured_amount.currency.value}) "
                    f"must match payment amount currency ({self.amount.currency.value})."
                )
            if captured_amount > self.amount:
                raise ValueError(
                    f"captured_amount ({captured_amount.amount_paise} paise) cannot exceed "
                    f"the authorised amount ({self.amount.amount_paise} paise)."
                )
            if captured_at < self.created_at:
                raise ValueError(
                    f"captured_at ({captured_at}) cannot precede "
                    f"payment created_at ({self.created_at})."
                )
            self.captured_amount = captured_amount
            self.captured_at = captured_at

        elif new_status == PaymentStatus.FAILED:
            if failed_at is None:
                raise ValueError(
                    "failed_at is required when transitioning Payment to FAILED."
                )
            if failed_at < self.created_at:
                raise ValueError(
                    f"failed_at ({failed_at}) cannot precede "
                    f"payment created_at ({self.created_at})."
                )
            self.failed_at = failed_at
            self.failure_reason = failure_reason

        self.status = new_status


# ---------------------------------------------------------------------------
# Refund
# ---------------------------------------------------------------------------

_MAX_FAILURE_REASON_LEN: Final[int] = 300


class Refund(BaseModel):
    """
    A refund request against a captured payment.

    State machine (enforced by apply_transition):
        requested ──► created ──► processed
        requested ──► failed
        created   ──► failed

    Financial invariants enforced HERE
    -----------------------------------
    - amount must be positive (zero-amount refunds have no financial meaning).

    Financial invariant deferred to Phase 2 (state_engine.py)
    ----------------------------------------------------------
    - The cumulative sum of all processed refund amounts on a given payment
      must not exceed that payment's captured_amount.

      This invariant requires access to all sibling refunds on the same
      payment. Refund cannot enforce it alone. The financial state engine
      in Phase 2 MUST check this before marking any refund as processed
      and MUST quarantine any refund that would violate it.

    reason_text is UntrustedText because it is customer-supplied and may
    contain any content. It must not be used as an unsanitised ML feature
    or interpolated into LLM prompt instructions.
    """

    model_config = {"frozen": False}  # Mutable to allow controlled state transitions.

    refund_id: RefundId
    payment_id: PaymentId
    order_id: OrderId
    merchant_id: MerchantId
    customer_id: CustomerId
    requested_at: UTCDateTime
    amount: Money
    reason_code: RefundReasonCode
    status: RefundStatus = RefundStatus.REQUESTED

    reason_text: UntrustedText | None = None
    created_at: UTCDateTime | None = None
    processed_at: UTCDateTime | None = None
    processed_amount: Money | None = None
    failed_at: UTCDateTime | None = None
    failure_reason: str | None = None

    @field_validator("amount")
    @classmethod
    def refund_amount_must_be_positive(cls, v: Money) -> Money:
        if v.is_zero():
            raise ValueError(
                "Refund amount must be positive. "
                "Zero-amount refunds are not meaningful in the domain model."
            )
        return v

    def apply_transition(
        self,
        new_status: RefundStatus,
        *,
        created_at: UTCDateTime | None = None,
        processed_at: UTCDateTime | None = None,
        processed_amount: Money | None = None,
        failed_at: UTCDateTime | None = None,
        failure_reason: str | None = None,
    ) -> None:
        """
        Transition this refund to a new lifecycle status.

        Raises InvalidStateTransitionError for illegal transitions.
        Raises ValueError for missing required arguments per transition type.
        """
        if (self.status, new_status) not in _REFUND_LEGAL_TRANSITIONS:
            raise InvalidStateTransitionError(
                entity="Refund",
                from_status=self.status.value,
                to_status=new_status.value,
            )

        if new_status == RefundStatus.CREATED:
            if created_at is None:
                raise ValueError(
                    "created_at is required when transitioning Refund to CREATED."
                )
            if created_at < self.requested_at:
                raise ValueError(
                    f"created_at ({created_at}) cannot precede "
                    f"requested_at ({self.requested_at})."
                )
            self.created_at = created_at

        elif new_status == RefundStatus.PROCESSED:
            if processed_at is None:
                raise ValueError(
                    "processed_at is required when transitioning Refund to PROCESSED."
                )
            if self.created_at is not None and processed_at < self.created_at:
                raise ValueError(
                    f"processed_at ({processed_at}) cannot precede "
                    f"created_at ({self.created_at})."
                )
            if processed_amount is None:
                raise ValueError(
                    "processed_amount is required when transitioning Refund to PROCESSED."
                )
            if processed_amount.currency != self.amount.currency:
                raise ValueError(
                    f"processed_amount currency ({processed_amount.currency.value}) "
                    f"must match the requested amount currency ({self.amount.currency.value})."
                )
            if processed_amount.is_zero():
                raise ValueError(
                    "processed_amount must be positive. "
                    "A processed refund of zero paise is not a valid outcome."
                )
            self.processed_at = processed_at
            self.processed_amount = processed_amount

        elif new_status == RefundStatus.FAILED:
            if failed_at is None:
                raise ValueError(
                    "failed_at is required when transitioning Refund to FAILED."
                )
            if failed_at < self.requested_at:
                raise ValueError(
                    f"failed_at ({failed_at}) cannot precede "
                    f"requested_at ({self.requested_at})."
                )
            if failure_reason is not None and len(failure_reason) > _MAX_FAILURE_REASON_LEN:
                raise ValueError(
                    f"failure_reason exceeds the maximum length of "
                    f"{_MAX_FAILURE_REASON_LEN} characters."
                )
            self.failed_at = failed_at
            self.failure_reason = failure_reason

        self.status = new_status