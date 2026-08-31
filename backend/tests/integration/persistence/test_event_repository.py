from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone

import pytest
from sqlalchemy.orm import Session

from backend.app.domain.enums import DataSource, EventType
from backend.app.domain.events import (
    EventEnvelope,
    PaymentCreatedEvent,
    PaymentCreatedPayload,
)
from backend.app.domain.identifiers import CustomerId, EventId, MerchantId, OrderId, PaymentId
from backend.app.domain.value_objects import Money, UTCDateTime
from backend.app.persistence.database import SessionLocal, engine
from backend.app.persistence.models import EventModel
from backend.app.persistence.repositories.events import (
    EventRepository,
    EventSaveOutcome,
    deserialize_event,
)


def timestamp(hour: int) -> UTCDateTime:
    return UTCDateTime(value=datetime(2024, 6, 1, hour, tzinfo=timezone.utc))


def payment_created(
    *, event_id: EventId | None = None, hour: int = 10, received_hour: int = 10,
    amount: int = 100_000,
) -> PaymentCreatedEvent:
    return PaymentCreatedEvent(
        envelope=EventEnvelope(
            event_id=event_id or EventId.generate(),
            event_type=EventType.PAYMENT_CREATED,
            occurred_at=timestamp(hour),
            received_at=timestamp(received_hour),
            source=DataSource.SIMULATOR,
        ),
        payload=PaymentCreatedPayload(
            payment_id=PaymentId.generate(),
            order_id=OrderId.generate(),
            merchant_id=MerchantId.generate(),
            customer_id=CustomerId.generate(),
            amount=Money.of_paise(amount),
        ),
    )


@pytest.fixture(scope="module", autouse=True)
def event_table() -> None:
    """Create only the ledger table when it has not yet been migrated."""
    EventModel.__table__.create(bind=engine, checkfirst=True)


@pytest.fixture
def session() -> Iterator[Session]:
    """Give each test a real database transaction that is always rolled back."""
    connection = engine.connect()
    transaction = connection.begin()
    db = SessionLocal(bind=connection)
    try:
        yield db
    finally:
        db.close()
        transaction.rollback()
        connection.close()


@pytest.mark.integration
def test_save_get_and_jsonb_round_trip(session: Session) -> None:
    event = payment_created()
    repository = EventRepository(session)

    result = repository.save(event)
    session.expunge_all()
    stored = repository.get(event.envelope.event_id)

    assert result.outcome is EventSaveOutcome.INSERTED
    assert stored is not None
    assert stored.payload == event.model_dump(mode="json")
    assert deserialize_event(stored.payload) == event


@pytest.mark.integration
def test_duplicate_and_conflict_are_classified_against_postgres(session: Session) -> None:
    event_id = EventId.generate()
    original = payment_created(event_id=event_id, amount=100_000)
    same_submission = PaymentCreatedEvent(
        envelope=original.envelope,
        payload=original.payload,
    )
    conflicting_submission = PaymentCreatedEvent(
        envelope=original.envelope,
        payload=PaymentCreatedPayload(
            payment_id=original.payload.payment_id,
            order_id=original.payload.order_id,
            merchant_id=original.payload.merchant_id,
            customer_id=original.payload.customer_id,
            amount=Money.of_paise(90_000),
        ),
    )
    repository = EventRepository(session)

    repository.save(original)
    duplicate = repository.save(same_submission)
    conflict = repository.save(conflicting_submission)

    assert duplicate.outcome is EventSaveOutcome.DUPLICATE
    assert conflict.outcome is EventSaveOutcome.CONFLICT
    assert repository.get(event_id).payload == original.model_dump(mode="json")


@pytest.mark.integration
def test_events_are_ordered_by_occurred_at_against_postgres(session: Session) -> None:
    repository = EventRepository(session)
    later_occurred = payment_created(hour=12, received_hour=8)
    earlier_occurred = payment_created(hour=9, received_hour=16)

    repository.save(later_occurred)
    repository.save(earlier_occurred)
    session.expunge_all()

    assert [row.event_id for row in repository.list_by_occurred_at()] == [
        earlier_occurred.envelope.event_id.value,
        later_occurred.envelope.event_id.value,
    ]
