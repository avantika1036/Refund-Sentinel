from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from backend.app.domain.enums import PaymentStatus, RefundStatus
from backend.app.domain.identifiers import (
    CustomerId, EventId, MerchantId, OrderId, PaymentId, RefundId,
)
from backend.app.domain.value_objects import Money, UTCDateTime

if TYPE_CHECKING:
    from backend.app.domain.events import AnyDomainEvent


@dataclass
class PaymentState:
    payment_id: PaymentId
    merchant_id: MerchantId
    order_id: OrderId
    customer_id: CustomerId
    authorised_amount: Money
    status: PaymentStatus
    created_at: UTCDateTime
    captured_amount: Money | None = None
    captured_at: UTCDateTime | None = None
    failed_at: UTCDateTime | None = None
    failure_reason: str | None = None
    cumulative_refunded: Money | None = None
    event_history: list[tuple[UTCDateTime, "AnyDomainEvent"]] = field(default_factory=list)
    applied_event_ids: set[EventId] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.cumulative_refunded is None:
            self.cumulative_refunded = Money.zero(self.authorised_amount.currency)

    @property
    def remaining_refundable(self) -> Money:
        if self.captured_amount is None:
            return Money.zero(self.authorised_amount.currency)
        return self.captured_amount - self.cumulative_refunded

    @property
    def is_fully_refunded(self) -> bool:
        return self.captured_amount is not None and self.cumulative_refunded >= self.captured_amount


@dataclass
class RefundState:
    refund_id: RefundId
    payment_id: PaymentId
    merchant_id: MerchantId
    customer_id: CustomerId
    order_id: OrderId
    requested_amount: Money
    status: RefundStatus
    requested_at: UTCDateTime
    created_at: UTCDateTime | None = None
    processed_at: UTCDateTime | None = None
    processed_amount: Money | None = None
    failed_at: UTCDateTime | None = None
    failure_reason: str | None = None
    event_history: list[tuple[UTCDateTime, "AnyDomainEvent"]] = field(default_factory=list)
    applied_event_ids: set[EventId] = field(default_factory=set)


@dataclass
class OrderState:
    order_id: OrderId
    merchant_id: MerchantId
    customer_id: CustomerId
    amount: Money
    created_at: UTCDateTime
    payment_id: PaymentId | None = None
    shipping_address_id: object | None = None
    shipped_at: UTCDateTime | None = None
    delivered_at: UTCDateTime | None = None
    event_history: list[tuple[UTCDateTime, "AnyDomainEvent"]] = field(default_factory=list)
    applied_event_ids: set[EventId] = field(default_factory=set)
