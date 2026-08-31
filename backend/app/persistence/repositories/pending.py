"""Persistence boundary for events awaiting prerequisites."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.domain.events import AnyDomainEvent
from backend.app.domain.identifiers import EventId
from backend.app.persistence.models import PendingEventModel
from backend.app.persistence.repositories.events import calculate_payload_hash


class PendingSaveOutcome(str, Enum):
    INSERTED = "inserted"
    DUPLICATE = "duplicate"
    CONFLICT = "conflict"


@dataclass(frozen=True)
class PendingSaveResult:
    outcome: PendingSaveOutcome
    event: PendingEventModel


class PendingEventRepository:
    """Stores pending events without owning the caller's transaction."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def save(self, event: AnyDomainEvent) -> PendingSaveResult:
        event_id = event.envelope.event_id.value
        event_hash = calculate_payload_hash(event)
        existing = self._session.get(PendingEventModel, event_id)
        if existing is not None:
            outcome = (
                PendingSaveOutcome.DUPLICATE
                if existing.payload_hash == event_hash
                else PendingSaveOutcome.CONFLICT
            )
            return PendingSaveResult(outcome, existing)

        row = PendingEventModel(
            event_id=event_id,
            event_type=event.envelope.event_type.value,
            payload_hash=event_hash,
            payload=event.model_dump(mode="json"),
            occurred_at=event.envelope.occurred_at.value,
            received_at=event.envelope.received_at.value,
            source=event.envelope.source.value,
            created_at=datetime.now(timezone.utc),
        )
        self._session.add(row)
        self._session.flush()
        return PendingSaveResult(PendingSaveOutcome.INSERTED, row)

    def get(self, event_id: EventId) -> PendingEventModel | None:
        return self._session.get(PendingEventModel, event_id.value)

    def exists(self, event_id: EventId) -> bool:
        return self.get(event_id) is not None

    def remove(self, event_id: EventId) -> bool:
        row = self.get(event_id)
        if row is None:
            return False
        self._session.delete(row)
        self._session.flush()
        return True

    def list_by_occurred_at(self) -> list[PendingEventModel]:
        statement = select(PendingEventModel).order_by(
            PendingEventModel.occurred_at, PendingEventModel.event_id
        )
        return list(self._session.scalars(statement).all())
