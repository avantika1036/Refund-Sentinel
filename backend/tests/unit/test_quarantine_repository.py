from __future__ import annotations

from datetime import datetime, timezone

from backend.app.domain.enums import DataSource, EventType
from backend.app.domain.events import EventEnvelope, PaymentCreatedEvent, PaymentCreatedPayload
from backend.app.domain.identifiers import CustomerId, EventId, MerchantId, OrderId, PaymentId
from backend.app.domain.value_objects import Money, UTCDateTime
from backend.app.finance.processing import ReasonCode
from backend.app.persistence.models import PendingEventModel
from backend.app.persistence.repositories.quarantine import QuarantineRepository, QuarantineSaveOutcome


def event(*, event_id: EventId | None = None, amount: int = 100) -> PaymentCreatedEvent:
    time = UTCDateTime(value=datetime(2024, 1, 1, tzinfo=timezone.utc))
    return PaymentCreatedEvent(envelope=EventEnvelope(event_id=event_id or EventId.generate(), event_type=EventType.PAYMENT_CREATED, occurred_at=time, received_at=time, source=DataSource.SIMULATOR), payload=PaymentCreatedPayload(payment_id=PaymentId.generate(), order_id=OrderId.generate(), merchant_id=MerchantId.generate(), customer_id=CustomerId.generate(), amount=Money.of_paise(amount)))


class Result:
    def __init__(self, rows): self.rows = rows
    def first(self): return self.rows[0] if self.rows else None
    def all(self): return self.rows


class Session:
    def __init__(self): self.rows = []; self.flushes = 0
    def add(self, row): row.id = len(self.rows) + 1; self.rows.append(row)
    def flush(self): self.flushes += 1
    def scalars(self, statement):
        values = statement.compile().params.values()
        event_id = next((value for value in values if hasattr(value, "int")), None)
        rows = [row for row in self.rows if event_id is None or row.event_id == event_id]
        return Result(rows)


def test_save_get_duplicate_conflict_and_reason_preservation() -> None:
    session = Session(); repository = QuarantineRepository(session); original = event()
    inserted = repository.save(original, reason_code=ReasonCode.RECONSTRUCTION_ANOMALY, detail="bad data")
    assert inserted.outcome is QuarantineSaveOutcome.INSERTED
    assert repository.get(original.envelope.event_id).reason_code == ReasonCode.RECONSTRUCTION_ANOMALY.value
    assert repository.get(EventId.generate()) is None
    assert repository.save(PaymentCreatedEvent(envelope=original.envelope, payload=original.payload), reason_code=ReasonCode.UNKNOWN_PAYMENT).outcome is QuarantineSaveOutcome.DUPLICATE
    assert repository.save(event(event_id=original.envelope.event_id, amount=99), reason_code=ReasonCode.UNKNOWN_PAYMENT).outcome is QuarantineSaveOutcome.CONFLICT
    assert len(session.rows) == 1 and session.rows[0].detail == "bad data"


def test_quarantine_is_isolated_from_pending_storage() -> None:
    session = Session(); quarantined = event()
    QuarantineRepository(session).save(quarantined, reason_code=ReasonCode.UNKNOWN_PAYMENT)
    assert session.rows[0].__tablename__ == "quarantine_records"
    assert not isinstance(session.rows[0], PendingEventModel)


def test_quarantine_lists_records_deterministically() -> None:
    session = Session(); repository = QuarantineRepository(session)
    first, second = event(), event()
    repository.save(first, reason_code=ReasonCode.UNKNOWN_PAYMENT)
    repository.save(second, reason_code=ReasonCode.UNKNOWN_REFUND)
    assert [record.event_id for record in repository.list_by_created_at()] == [
        first.envelope.event_id.value, second.envelope.event_id.value,
    ]
