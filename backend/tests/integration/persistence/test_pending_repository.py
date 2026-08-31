from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone

import pytest
from sqlalchemy.orm import Session

from backend.app.domain.enums import DataSource, EventType
from backend.app.domain.events import EventEnvelope, PaymentCreatedEvent, PaymentCreatedPayload
from backend.app.domain.identifiers import CustomerId, EventId, MerchantId, OrderId, PaymentId
from backend.app.domain.value_objects import Money, UTCDateTime
from backend.app.persistence.database import SessionLocal, engine
from backend.app.persistence.models import PendingEventModel
from backend.app.persistence.repositories.pending import PendingEventRepository, PendingSaveOutcome


def event(*, event_id: EventId | None = None, hour: int = 10, received_hour: int = 10, amount: int = 100) -> PaymentCreatedEvent:
    ts = lambda value: UTCDateTime(value=datetime(2024, 1, 1, value, tzinfo=timezone.utc))
    return PaymentCreatedEvent(envelope=EventEnvelope(event_id=event_id or EventId.generate(), event_type=EventType.PAYMENT_CREATED, occurred_at=ts(hour), received_at=ts(received_hour), source=DataSource.SIMULATOR), payload=PaymentCreatedPayload(payment_id=PaymentId.generate(), order_id=OrderId.generate(), merchant_id=MerchantId.generate(), customer_id=CustomerId.generate(), amount=Money.of_paise(amount)))


@pytest.fixture(scope="module", autouse=True)
def pending_table() -> None:
    PendingEventModel.__table__.create(bind=engine, checkfirst=True)


@pytest.fixture
def session() -> Iterator[Session]:
    connection = engine.connect(); transaction = connection.begin(); db = SessionLocal(bind=connection)
    try: yield db
    finally: db.close(); transaction.rollback(); connection.close()


@pytest.mark.integration
def test_pending_insert_get_duplicate_conflict_and_jsonb_round_trip(session: Session) -> None:
    repository = PendingEventRepository(session); original = event()
    assert repository.save(original).outcome is PendingSaveOutcome.INSERTED
    session.expunge_all(); stored = repository.get(original.envelope.event_id)
    assert stored is not None and stored.payload == original.model_dump(mode="json")
    assert repository.save(PaymentCreatedEvent(envelope=original.envelope, payload=original.payload)).outcome is PendingSaveOutcome.DUPLICATE
    assert repository.save(event(event_id=original.envelope.event_id, amount=99)).outcome is PendingSaveOutcome.CONFLICT


@pytest.mark.integration
def test_pending_order_and_removal_against_postgres(session: Session) -> None:
    repository = PendingEventRepository(session); later = event(hour=12, received_hour=8); earlier = event(hour=9, received_hour=16)
    repository.save(later); repository.save(earlier); session.expunge_all()
    assert [row.event_id for row in repository.list_by_occurred_at()] == [earlier.envelope.event_id.value, later.envelope.event_id.value]
    assert repository.remove(earlier.envelope.event_id) is True
    assert repository.get(earlier.envelope.event_id) is None
