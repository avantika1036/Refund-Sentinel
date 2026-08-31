from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.domain.enums import EventType
from backend.app.domain.identifiers import EventId
from backend.app.domain.value_objects import UTCDateTime
from backend.app.finance.types import IngestionOutcome, IngestionRecord
from backend.app.persistence.database import SessionLocal, engine
from backend.app.persistence.models import IngestionRecordModel
from backend.app.persistence.repositories.ingestion import IngestionRepository


def record(event_id: EventId | None = None, ordinal: int = 1, *, pending: bool = True) -> IngestionRecord:
    time = UTCDateTime(value=datetime(2024, 1, 1, tzinfo=timezone.utc))
    return IngestionRecord(event_id=event_id or EventId.generate(), event_type=EventType.PAYMENT_CREATED, occurred_at=time, received_at=time, submission_ordinal=ordinal, payload_hash="a" * 64, ingestion_outcome=IngestionOutcome.PENDING if pending else IngestionOutcome.RETAINED, pending=pending, retained=not pending)


@pytest.fixture(scope="module", autouse=True)
def ingestion_table() -> None:
    IngestionRecordModel.__table__.create(bind=engine, checkfirst=True)


@pytest.fixture
def session() -> Iterator[Session]:
    connection = engine.connect(); transaction = connection.begin(); db = SessionLocal(bind=connection)
    try: yield db
    finally: db.close(); transaction.rollback(); connection.close()


@pytest.mark.integration
def test_ingestion_append_update_and_order_against_postgres(session: Session) -> None:
    repository = IngestionRepository(session); second, first = record(ordinal=2), record(ordinal=1)
    repository.append(second); repository.append(first)
    promoted = record(first.event_id, 1, pending=False)
    promoted.promoted_at = UTCDateTime(value=datetime(2024, 1, 2, tzinfo=timezone.utc))
    repository.update(promoted); session.expunge_all()
    stored = repository.get(first.event_id, 1)
    assert stored is not None and stored.ingestion_outcome == IngestionOutcome.RETAINED.value
    assert len(repository.list_by_submission_order()) == 2
    assert [row.submission_ordinal for row in repository.list_by_submission_order()] == [1, 2]


@pytest.mark.integration
def test_ingestion_unique_submission_constraint_against_postgres(session: Session) -> None:
    repository = IngestionRepository(session); original = record()
    repository.append(original)
    with pytest.raises(IntegrityError):
        with session.begin_nested(): repository.append(record(original.event_id, original.submission_ordinal))
