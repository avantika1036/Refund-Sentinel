from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone

import pytest
from sqlalchemy.orm import Session

from backend.app.domain.enums import DataSource, EventType
from backend.app.domain.events import EventEnvelope, PaymentCreatedEvent, PaymentCreatedPayload
from backend.app.domain.identifiers import CustomerId, EventId, MerchantId, OrderId, PaymentId
from backend.app.domain.value_objects import Money, UTCDateTime
from backend.app.finance.processing import ReasonCode
from backend.app.persistence.database import SessionLocal, engine
from backend.app.persistence.models import PendingEventModel, QuarantineRecordModel
from backend.app.persistence.repositories.quarantine import QuarantineRepository, QuarantineSaveOutcome


def event(*, event_id: EventId | None = None, amount: int = 100) -> PaymentCreatedEvent:
    time = UTCDateTime(value=datetime(2024, 1, 1, tzinfo=timezone.utc))
    return PaymentCreatedEvent(envelope=EventEnvelope(event_id=event_id or EventId.generate(), event_type=EventType.PAYMENT_CREATED, occurred_at=time, received_at=time, source=DataSource.SIMULATOR), payload=PaymentCreatedPayload(payment_id=PaymentId.generate(), order_id=OrderId.generate(), merchant_id=MerchantId.generate(), customer_id=CustomerId.generate(), amount=Money.of_paise(amount)))


@pytest.fixture(scope="module", autouse=True)
def quarantine_table() -> None:
    QuarantineRecordModel.__table__.create(bind=engine, checkfirst=True)


@pytest.fixture
def session() -> Iterator[Session]:
    connection = engine.connect(); transaction = connection.begin(); db = SessionLocal(bind=connection)
    try: yield db
    finally: db.close(); transaction.rollback(); connection.close()


@pytest.mark.integration
def test_quarantine_jsonb_reason_duplicate_and_conflict_against_postgres(session: Session) -> None:
    repository = QuarantineRepository(session); original = event()
    assert repository.save(original, reason_code=ReasonCode.RECONSTRUCTION_ANOMALY, detail="unsafe").outcome is QuarantineSaveOutcome.INSERTED
    session.expunge_all(); stored = repository.get(original.envelope.event_id)
    assert stored is not None and stored.payload == original.model_dump(mode="json")
    assert stored.reason_code == ReasonCode.RECONSTRUCTION_ANOMALY.value and stored.detail == "unsafe"
    assert repository.save(PaymentCreatedEvent(envelope=original.envelope, payload=original.payload), reason_code=ReasonCode.UNKNOWN_PAYMENT).outcome is QuarantineSaveOutcome.DUPLICATE
    assert repository.save(event(event_id=original.envelope.event_id, amount=99), reason_code=ReasonCode.UNKNOWN_PAYMENT).outcome is QuarantineSaveOutcome.CONFLICT
    later = event()
    repository.save(later, reason_code=ReasonCode.UNKNOWN_REFUND)
    session.expunge_all()
    assert [row.event_id for row in repository.list_by_created_at()] == [
        original.envelope.event_id.value, later.envelope.event_id.value,
    ]


@pytest.mark.integration
def test_quarantine_is_separate_from_pending_table(session: Session) -> None:
    repository = QuarantineRepository(session); quarantined = event()
    repository.save(quarantined, reason_code=ReasonCode.UNKNOWN_PAYMENT)
    assert repository.get(quarantined.envelope.event_id) is not None
    assert session.get(PendingEventModel, quarantined.envelope.event_id.value) is None
