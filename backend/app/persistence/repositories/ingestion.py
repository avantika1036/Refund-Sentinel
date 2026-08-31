"""Persistence boundary for immutable-per-submission ingestion audit records."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.domain.identifiers import EventId
from backend.app.finance.types import IngestionRecord
from backend.app.persistence.models import IngestionRecordModel


def _reason_value(reason_code: object | None) -> str | None:
    if reason_code is None:
        return None
    value = getattr(reason_code, "value", reason_code)
    return str(value)


class IngestionRepository:
    """Persists one audit row per submission without committing transactions."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def append(self, record: IngestionRecord) -> IngestionRecordModel:
        row = IngestionRecordModel(
            event_id=record.event_id.value,
            submission_ordinal=record.submission_ordinal,
            ingestion_outcome=record.ingestion_outcome.value,
            reason_code=_reason_value(record.reason_code),
            detail=record.detail,
            retained=record.retained,
            pending=record.pending,
            promoted_at=record.promoted_at.value if record.promoted_at else None,
            triggered_reconstruction=record.triggered_reconstruction,
            reconstruction_ordinal=record.reconstruction_ordinal,
            original_submission_ordinal=record.original_submission_ordinal,
        )
        self._session.add(row)
        self._session.flush()
        return row

    def get(self, event_id: EventId, submission_ordinal: int) -> IngestionRecordModel | None:
        statement = select(IngestionRecordModel).where(
            IngestionRecordModel.event_id == event_id.value,
            IngestionRecordModel.submission_ordinal == submission_ordinal,
        )
        return self._session.scalars(statement).one_or_none()

    def list_by_submission_order(self) -> list[IngestionRecordModel]:
        statement = select(IngestionRecordModel).order_by(
            IngestionRecordModel.submission_ordinal, IngestionRecordModel.id
        )
        return list(self._session.scalars(statement).all())

    def update(self, record: IngestionRecord) -> IngestionRecordModel:
        """Update the audit row for an existing submission, e.g. on promotion."""
        row = self.get(record.event_id, record.submission_ordinal)
        if row is None:
            raise LookupError("Cannot update an ingestion record that was not appended.")
        row.ingestion_outcome = record.ingestion_outcome.value
        row.reason_code = _reason_value(record.reason_code)
        row.detail = record.detail
        row.retained = record.retained
        row.pending = record.pending
        row.promoted_at = record.promoted_at.value if record.promoted_at else None
        row.triggered_reconstruction = record.triggered_reconstruction
        row.reconstruction_ordinal = record.reconstruction_ordinal
        row.original_submission_ordinal = record.original_submission_ordinal
        self._session.flush()
        return row
