"""Persistence boundary for events isolated from canonical and pending stores."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.domain.events import AnyDomainEvent
from backend.app.domain.identifiers import EventId
from backend.app.persistence.models import QuarantineRecordModel
from backend.app.persistence.repositories.events import calculate_payload_hash


class QuarantineSaveOutcome(str, Enum):
    INSERTED = "inserted"
    DUPLICATE = "duplicate"
    CONFLICT = "conflict"


@dataclass(frozen=True)
class QuarantineSaveResult:
    outcome: QuarantineSaveOutcome
    record: QuarantineRecordModel


def _reason_value(reason_code: object) -> str:
    return str(getattr(reason_code, "value", reason_code))


class QuarantineRepository:
    """Stores quarantined events without replacing earlier quarantine records."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def save(
        self, event: AnyDomainEvent, *, reason_code: object, detail: str | None = None
    ) -> QuarantineSaveResult:
        event_hash = calculate_payload_hash(event)
        existing = self.get(event.envelope.event_id)
        if existing is not None:
            outcome = (
                QuarantineSaveOutcome.DUPLICATE
                if existing.payload_hash == event_hash
                else QuarantineSaveOutcome.CONFLICT
            )
            return QuarantineSaveResult(outcome, existing)

        row = QuarantineRecordModel(
            event_id=event.envelope.event_id.value,
            event_type=event.envelope.event_type.value,
            payload_hash=event_hash,
            reason_code=_reason_value(reason_code),
            detail=detail,
            payload=event.model_dump(mode="json"),
            created_at=datetime.now(timezone.utc),
        )
        self._session.add(row)
        self._session.flush()
        return QuarantineSaveResult(QuarantineSaveOutcome.INSERTED, row)

    def get(self, event_id: EventId) -> QuarantineRecordModel | None:
        statement = select(QuarantineRecordModel).where(
            QuarantineRecordModel.event_id == event_id.value
        ).order_by(QuarantineRecordModel.id)
        return self._session.scalars(statement).first()

    def list_by_created_at(self) -> list[QuarantineRecordModel]:
        statement = select(QuarantineRecordModel).order_by(
            QuarantineRecordModel.created_at, QuarantineRecordModel.id
        )
        return list(self._session.scalars(statement).all())
