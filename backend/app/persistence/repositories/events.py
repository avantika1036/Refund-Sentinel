"""Persistence boundary for the immutable domain event ledger."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.domain.enums import EventType
from backend.app.domain.events import (
    AnyDomainEvent,
    OrderCreatedEvent,
    OrderDeliveredEvent,
    OrderShippedEvent,
    PaymentCapturedEvent,
    PaymentCreatedEvent,
    PaymentFailedEvent,
    RefundCreatedEvent,
    RefundFailedEvent,
    RefundProcessedEvent,
    RefundRequestedEvent,
)
from backend.app.domain.identifiers import EventId
from backend.app.persistence.models import EventModel


class EventSaveOutcome(str, Enum):
    INSERTED = "inserted"
    DUPLICATE = "duplicate"
    CONFLICT = "conflict"


@dataclass(frozen=True)
class EventSaveResult:
    """Result of an idempotent event-save attempt."""

    outcome: EventSaveOutcome
    event: EventModel


_EVENT_CLASSES: dict[EventType, type[AnyDomainEvent]] = {
    EventType.ORDER_CREATED: OrderCreatedEvent,
    EventType.PAYMENT_CREATED: PaymentCreatedEvent,
    EventType.PAYMENT_CAPTURED: PaymentCapturedEvent,
    EventType.PAYMENT_FAILED: PaymentFailedEvent,
    EventType.REFUND_REQUESTED: RefundRequestedEvent,
    EventType.REFUND_CREATED: RefundCreatedEvent,
    EventType.REFUND_PROCESSED: RefundProcessedEvent,
    EventType.REFUND_FAILED: RefundFailedEvent,
    EventType.ORDER_SHIPPED: OrderShippedEvent,
    EventType.ORDER_DELIVERED: OrderDeliveredEvent,
}

_DATETIME_FIELDS = frozenset({
    "occurred_at", "received_at", "captured_at", "failed_at", "created_at",
    "processed_at", "shipped_at", "delivered_at",
})


def calculate_payload_hash(event: AnyDomainEvent) -> str:
    """Return the stable SHA-256 hash of business payload data only."""
    payload = event.payload.model_dump(mode="json")
    canonical_json = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def deserialize_event(payload: dict[str, object]) -> AnyDomainEvent:
    """Rebuild the concrete immutable domain event stored in an EventModel."""
    restored_payload = _restore_datetime_values(payload)
    envelope = restored_payload.get("envelope")
    if not isinstance(envelope, dict) or not isinstance(envelope.get("event_type"), str):
        raise ValueError("Stored event payload has no valid envelope.event_type.")
    event_type = EventType(envelope["event_type"])
    return _EVENT_CLASSES[event_type].model_validate(restored_payload)


def _restore_datetime_values(value: object, field_name: str | None = None) -> object:
    """Convert JSON event timestamps back to values accepted by UTCDateTime."""
    if isinstance(value, list):
        return [_restore_datetime_values(item) for item in value]
    if not isinstance(value, dict):
        return value

    restored = {
        key: _restore_datetime_values(item, key)
        for key, item in value.items()
    }
    if field_name in _DATETIME_FIELDS and isinstance(restored.get("value"), str):
        restored["value"] = datetime.fromisoformat(restored["value"])
    return restored


class EventRepository:
    """Stores append-only domain events without owning transaction commits."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def save(self, event: AnyDomainEvent) -> EventSaveResult:
        """Stage an event row, or explicitly classify a repeated event ID.

        The caller owns commit/rollback. ``flush()`` assigns the insert to the
        current transaction and surfaces database errors without committing it.
        """
        event_id = event.envelope.event_id.value
        event_hash = calculate_payload_hash(event)
        existing = self._session.get(EventModel, event_id)
        if existing is not None:
            outcome = (
                EventSaveOutcome.DUPLICATE
                if existing.payload_hash == event_hash
                else EventSaveOutcome.CONFLICT
            )
            return EventSaveResult(outcome=outcome, event=existing)

        row = EventModel(
            event_id=event_id,
            event_type=event.envelope.event_type.value,
            occurred_at=event.envelope.occurred_at.value,
            received_at=event.envelope.received_at.value,
            source=event.envelope.source.value,
            payload_hash=event_hash,
            payload=event.model_dump(mode="json"),
            created_at=datetime.now(timezone.utc),
        )
        self._session.add(row)
        self._session.flush()
        return EventSaveResult(outcome=EventSaveOutcome.INSERTED, event=row)

    def get(self, event_id: EventId) -> EventModel | None:
        return self._session.get(EventModel, event_id.value)

    def exists(self, event_id: EventId) -> bool:
        return self.get(event_id) is not None

    def list_by_occurred_at(self) -> list[EventModel]:
        """Return canonical ledger order, with event_id as a stable tie-breaker."""
        statement = select(EventModel).order_by(EventModel.occurred_at, EventModel.event_id)
        return list(self._session.scalars(statement).all())
