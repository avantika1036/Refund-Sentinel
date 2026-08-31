from __future__ import annotations

from datetime import datetime, timezone

from backend.app.domain.enums import DataSource, EventType
from backend.app.domain.events import EventEnvelope, PaymentCreatedEvent, PaymentCreatedPayload
from backend.app.domain.identifiers import CustomerId, EventId, MerchantId, OrderId, PaymentId
from backend.app.domain.value_objects import Money, UTCDateTime
from backend.app.persistence.models import PendingEventModel
from backend.app.persistence.repositories.pending import PendingEventRepository, PendingSaveOutcome


def event(*, event_id: EventId | None = None, hour: int = 10, amount: int = 100) -> PaymentCreatedEvent:
    timestamp = lambda h: UTCDateTime(value=datetime(2024, 1, 1, h, tzinfo=timezone.utc))
    return PaymentCreatedEvent(
        envelope=EventEnvelope(event_id=event_id or EventId.generate(), event_type=EventType.PAYMENT_CREATED, occurred_at=timestamp(hour), received_at=timestamp(20 - hour), source=DataSource.SIMULATOR),
        payload=PaymentCreatedPayload(payment_id=PaymentId.generate(), order_id=OrderId.generate(), merchant_id=MerchantId.generate(), customer_id=CustomerId.generate(), amount=Money.of_paise(amount)),
    )


class Result:
    def __init__(self, rows): self.rows = rows
    def all(self): return self.rows


class Session:
    def __init__(self): self.rows = {}; self.flushes = 0; self.deleted = []
    def get(self, _, key): return self.rows.get(key)
    def add(self, row): self.rows[row.event_id] = row
    def delete(self, row): self.deleted.append(row); self.rows.pop(row.event_id)
    def flush(self): self.flushes += 1
    def scalars(self, _): return Result(sorted(self.rows.values(), key=lambda r: (r.occurred_at, r.event_id.int)))


def test_save_get_exists_remove_and_missing() -> None:
    session, repository = Session(), None
    repository = PendingEventRepository(session)
    pending = event()
    assert repository.save(pending).outcome is PendingSaveOutcome.INSERTED
    assert repository.get(pending.envelope.event_id).event_id == pending.envelope.event_id.value
    assert repository.exists(pending.envelope.event_id)
    assert repository.get(EventId.generate()) is None
    assert repository.remove(pending.envelope.event_id) is True
    assert repository.remove(pending.envelope.event_id) is False
    assert session.flushes == 2


def test_pending_ordering_duplicate_and_conflict_preserve_original() -> None:
    session = Session(); repository = PendingEventRepository(session)
    later, earlier = event(hour=12), event(hour=9)
    repository.save(later); repository.save(earlier)
    assert [row.event_id for row in repository.list_by_occurred_at()] == [earlier.envelope.event_id.value, later.envelope.event_id.value]
    event_id = EventId.generate(); original = event(event_id=event_id, amount=100)
    repository.save(original)
    assert repository.save(PaymentCreatedEvent(envelope=original.envelope, payload=original.payload)).outcome is PendingSaveOutcome.DUPLICATE
    assert repository.save(event(event_id=event_id, amount=99)).outcome is PendingSaveOutcome.CONFLICT
    assert repository.get(event_id).payload == original.model_dump(mode="json")
