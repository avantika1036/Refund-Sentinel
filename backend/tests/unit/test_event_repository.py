from __future__ import annotations

from datetime import datetime, timezone

import pytest

from backend.app.domain.enums import DataSource, EventType
from backend.app.domain.events import (
    EventEnvelope,
    PaymentCapturedEvent,
    PaymentCapturedPayload,
    PaymentCreatedEvent,
    PaymentCreatedPayload,
)
from backend.app.domain.identifiers import CustomerId, EventId, MerchantId, OrderId, PaymentId
from backend.app.domain.value_objects import Money, UTCDateTime
from backend.app.persistence.models import EventModel
from backend.app.persistence.repositories.events import (
    EventRepository,
    EventSaveOutcome,
    calculate_payload_hash,
    deserialize_event,
)


def timestamp(hour: int) -> UTCDateTime:
    return UTCDateTime(value=datetime(2024, 6, 1, hour, tzinfo=timezone.utc))


def payment_created(*, event_id: EventId | None = None, hour: int = 10, amount: int = 100_000) -> PaymentCreatedEvent:
    return PaymentCreatedEvent(
        envelope=EventEnvelope(
            event_id=event_id or EventId.generate(),
            event_type=EventType.PAYMENT_CREATED,
            occurred_at=timestamp(hour),
            received_at=timestamp(20 - hour),
            source=DataSource.SIMULATOR,
        ),
        payload=PaymentCreatedPayload(
            payment_id=PaymentId.generate(), order_id=OrderId.generate(),
            merchant_id=MerchantId.generate(), customer_id=CustomerId.generate(),
            amount=Money.of_paise(amount),
        ),
    )


class _ScalarResult:
    def __init__(self, rows: list[EventModel]) -> None:
        self._rows = rows

    def all(self) -> list[EventModel]:
        return self._rows


class FakeSession:
    def __init__(self) -> None:
        self.rows: dict[object, EventModel] = {}
        self.added: list[EventModel] = []
        self.flush_count = 0
        self.committed = False
        self.statement = None

    def get(self, model: type[EventModel], event_id: object) -> EventModel | None:
        assert model is EventModel
        return self.rows.get(event_id)

    def add(self, row: EventModel) -> None:
        self.added.append(row)
        self.rows[row.event_id] = row

    def flush(self) -> None:
        self.flush_count += 1

    def scalars(self, statement: object) -> _ScalarResult:
        self.statement = statement
        rows = sorted(self.rows.values(), key=lambda row: (row.occurred_at, row.event_id.int))
        return _ScalarResult(rows)


@pytest.fixture
def session() -> FakeSession:
    return FakeSession()


def test_insert_event(session: FakeSession) -> None:
    event = payment_created()
    result = EventRepository(session).save(event)

    assert result.outcome is EventSaveOutcome.INSERTED
    assert result.event.event_id == event.envelope.event_id.value
    assert result.event.payload == event.model_dump(mode="json")
    assert result.event.payload_hash == calculate_payload_hash(event)
    assert session.flush_count == 1
    assert session.committed is False


def test_get_by_event_id_and_missing_event(session: FakeSession) -> None:
    repository = EventRepository(session)
    event = payment_created()
    repository.save(event)

    assert repository.get(event.envelope.event_id).event_id == event.envelope.event_id.value
    assert repository.get(EventId.generate()) is None


def test_exists(session: FakeSession) -> None:
    repository = EventRepository(session)
    event = payment_created()
    repository.save(event)

    assert repository.exists(event.envelope.event_id) is True
    assert repository.exists(EventId.generate()) is False


def test_events_are_ordered_by_occurred_at_not_received_at(session: FakeSession) -> None:
    repository = EventRepository(session)
    later = payment_created(hour=12)
    earlier = payment_created(hour=9)
    repository.save(later)
    repository.save(earlier)

    assert [row.event_id for row in repository.list_by_occurred_at()] == [
        earlier.envelope.event_id.value, later.envelope.event_id.value
    ]
    assert "ORDER BY events.occurred_at, events.event_id" in str(session.statement)


def test_same_event_id_and_payload_is_an_idempotent_duplicate(session: FakeSession) -> None:
    repository = EventRepository(session)
    event = payment_created()
    repository.save(event)

    result = repository.save(event)

    assert result.outcome is EventSaveOutcome.DUPLICATE
    assert len(session.rows) == 1
    assert session.flush_count == 1


def test_same_event_id_and_different_payload_is_a_conflict(session: FakeSession) -> None:
    repository = EventRepository(session)
    event_id = EventId.generate()
    repository.save(payment_created(event_id=event_id, amount=100_000))

    result = repository.save(payment_created(event_id=event_id, amount=90_000))

    assert result.outcome is EventSaveOutcome.CONFLICT
    assert len(session.rows) == 1


def test_events_remain_immutable_after_save(session: FakeSession) -> None:
    event = payment_created()
    EventRepository(session).save(event)

    with pytest.raises(Exception):
        event.payload.amount = Money.of_paise(1)
    assert session.rows[event.envelope.event_id.value].payload["payload"]["amount"]["amount_paise"] == 100_000


def test_multiple_event_types_and_json_round_trip(session: FakeSession) -> None:
    created = payment_created()
    captured = PaymentCapturedEvent(
        envelope=EventEnvelope(
            event_id=EventId.generate(), event_type=EventType.PAYMENT_CAPTURED,
            occurred_at=timestamp(11), received_at=timestamp(8), source=DataSource.SIMULATOR,
        ),
        payload=PaymentCapturedPayload(
            payment_id=created.payload.payment_id, merchant_id=created.payload.merchant_id,
            captured_amount=Money.of_paise(100_000), captured_at=timestamp(11),
        ),
    )
    repository = EventRepository(session)
    repository.save(created)
    repository.save(captured)

    restored_created = deserialize_event(repository.get(created.envelope.event_id).payload)
    restored_captured = deserialize_event(repository.get(captured.envelope.event_id).payload)
    assert restored_created == created
    assert restored_captured == captured
