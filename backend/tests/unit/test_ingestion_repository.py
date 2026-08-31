from __future__ import annotations

from datetime import datetime, timezone

import pytest

from backend.app.domain.enums import DataSource, EventType
from backend.app.domain.identifiers import EventId
from backend.app.domain.value_objects import UTCDateTime
from backend.app.finance.types import IngestionOutcome, IngestionRecord
from backend.app.persistence.repositories.ingestion import IngestionRepository


def record(event_id: EventId | None = None, ordinal: int = 1, *, pending: bool = True) -> IngestionRecord:
    time = UTCDateTime(value=datetime(2024, 1, 1, tzinfo=timezone.utc))
    return IngestionRecord(event_id=event_id or EventId.generate(), event_type=EventType.PAYMENT_CREATED, occurred_at=time, received_at=time, submission_ordinal=ordinal, payload_hash="a" * 64, ingestion_outcome=IngestionOutcome.PENDING if pending else IngestionOutcome.RETAINED, pending=pending, retained=not pending)


class Result:
    def __init__(self, rows): self.rows = rows
    def one_or_none(self): return self.rows[0] if len(self.rows) == 1 else None
    def all(self): return self.rows


class Session:
    def __init__(self): self.rows = []; self.flushes = 0
    def add(self, row): row.id = len(self.rows) + 1; self.rows.append(row)
    def flush(self): self.flushes += 1
    def scalars(self, _): return Result(sorted(self.rows, key=lambda r: (r.submission_ordinal, r.id)))


def test_append_get_missing_and_list_submission_order() -> None:
    session = Session(); repository = IngestionRepository(session)
    second, first = record(ordinal=2), record(ordinal=1)
    repository.append(first)
    assert repository.get(first.event_id, 1).event_id == first.event_id.value
    repository.append(second)
    assert repository.get(EventId.generate(), 1) is None
    assert [row.submission_ordinal for row in repository.list_by_submission_order()] == [1, 2]


def test_update_promotes_existing_record_without_second_row() -> None:
    session = Session(); repository = IngestionRepository(session); pending = record()
    repository.append(pending)
    promoted = record(pending.event_id, pending.submission_ordinal, pending=False)
    promoted.promoted_at = UTCDateTime(value=datetime(2024, 1, 2, tzinfo=timezone.utc))
    promoted.triggered_reconstruction = True; promoted.reconstruction_ordinal = 3
    row = repository.update(promoted)
    assert len(session.rows) == 1
    assert row.ingestion_outcome == IngestionOutcome.RETAINED.value
    assert row.pending is False and row.promoted_at == promoted.promoted_at.value


def test_update_missing_record_is_explicit() -> None:
    with pytest.raises(LookupError): IngestionRepository(Session()).update(record())
