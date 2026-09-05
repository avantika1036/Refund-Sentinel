"""Transactional application boundary for financial event ingestion."""

from __future__ import annotations

from collections.abc import Callable
from threading import Lock

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.domain.events import AnyDomainEvent
from backend.app.finance.processing import ReasonCode
from backend.app.finance.state_engine import FinancialStateEngine
from backend.app.finance.types import IngestionOutcome, IngestionRecord
from backend.app.persistence.database import SessionLocal
from backend.app.persistence.repositories.events import EventRepository, EventSaveOutcome, calculate_payload_hash
from backend.app.persistence.repositories.ingestion import IngestionRepository
from backend.app.persistence.repositories.pending import PendingEventRepository
from backend.app.persistence.repositories.quarantine import QuarantineRepository


SessionFactory = Callable[[], Session]

_sqlite_ordinal_lock = Lock()
_sqlite_ordinal_counter: int | None = None


class IngestionService:
    """Coordinates repositories and the state engine in one database transaction."""

    def __init__(
        self,
        session_factory: SessionFactory = SessionLocal,
        state_engine: FinancialStateEngine | None = None,
        after_event_saved: Callable[[], None] | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._engine = state_engine or FinancialStateEngine()
        self._after_event_saved = after_event_saved

    def ingest(self, event: AnyDomainEvent) -> IngestionRecord:
        with self._session_factory() as session:
            with session.begin():
                return self._ingest_in_transaction(session, event)

    def _ingest_in_transaction(self, session: Session, event: AnyDomainEvent) -> IngestionRecord:
        events = EventRepository(session)
        ingestion = IngestionRepository(session)
        pending = PendingEventRepository(session)
        quarantine = QuarantineRepository(session)
        ordinal = self._next_submission_ordinal(session)
        if session.get_bind().dialect.name == "sqlite":
            # SQLite is the local/test backend. Avoid nested SAVEPOINT semantics
            # here because the sqlite driver can implicitly alter transaction
            # boundaries around DDL/locking. Production PostgreSQL keeps the
            # savepoint path below for concurrent idempotency handling.
            saved = events.save(event)
        else:
            try:
                with session.begin_nested():
                    saved = events.save(event)
            except IntegrityError:
                # The savepoint keeps the outer audit transaction usable after a
                # concurrent winner inserted the same event_id.
                session.expire_all()
                existing = events.get(event.envelope.event_id)
                if existing is None:
                    raise
                outcome = EventSaveOutcome.DUPLICATE if existing.payload_hash == calculate_payload_hash(event) else EventSaveOutcome.CONFLICT
                saved = type("Save", (), {"outcome": outcome, "event": existing})()

        if saved.outcome is not EventSaveOutcome.INSERTED:
            result = self._audit_duplicate_or_conflict(event, ordinal, saved.outcome, ingestion)
            return result

        if self._after_event_saved is not None:
            self._after_event_saved()

        record = self._engine.ingest(event)
        record.submission_ordinal = ordinal
        if record.ingestion_outcome is IngestionOutcome.PENDING:
            pending.save(event)
        elif record.ingestion_outcome is IngestionOutcome.QUARANTINED:
            quarantine.save(event, reason_code=record.reason_code, detail=record.detail)
        ingestion.append(record)
        self._sync_promotions(ingestion, pending)
        return record

    def _audit_duplicate_or_conflict(self, event: AnyDomainEvent, ordinal: int, outcome: EventSaveOutcome, repository: IngestionRepository) -> IngestionRecord:
        original = repository.get_by_event_id(event.envelope.event_id)
        is_duplicate = outcome is EventSaveOutcome.DUPLICATE
        record = IngestionRecord(
            event_id=event.envelope.event_id, event_type=event.envelope.event_type,
            occurred_at=event.envelope.occurred_at, received_at=event.envelope.received_at,
            submission_ordinal=ordinal, payload_hash=calculate_payload_hash(event),
            ingestion_outcome=IngestionOutcome.DUPLICATE if is_duplicate else IngestionOutcome.CONFLICT,
            reason_code=ReasonCode.DUPLICATE_EVENT_ID if is_duplicate else ReasonCode.CONFLICTING_EVENT_ID_PAYLOAD,
            detail="Same EventId and payload." if is_duplicate else "Same EventId was submitted with a different payload.",
            original_submission_ordinal=original.submission_ordinal if original else None,
        )
        repository.append(record)
        return record

    def _sync_promotions(self, ingestion: IngestionRepository, pending: PendingEventRepository) -> None:
        for record in self._engine.get_ingestion_log():
            if record.promoted_at is None:
                continue
            stored = ingestion.get_by_event_id(record.event_id)
            if stored is None or stored.pending is False:
                continue
            record.submission_ordinal = stored.submission_ordinal
            ingestion.update(record)
            pending.remove(record.event_id)

    @staticmethod
    def _next_submission_ordinal(session: Session) -> int:
        """Reserve a durable submission ordinal with intentional gaps.

        PostgreSQL uses a native sequence because sequence advancement is
        non-transactional. SQLite has no comparable portable sequence
        primitive, so local/test runs reserve an autoincrement row through an
        independent committed transaction.
        """
        dialect = session.get_bind().dialect.name
        if dialect == "postgresql":
            return int(
                session.execute(
                    text(
                        "SELECT nextval(pg_get_serial_sequence('ingestion_records', 'id'))"
                    )
                ).scalar_one()
            )

        # SQLite is a local/test backend, so reserve the ordinal outside the
        # database transaction with a process-wide lock. This preserves the
        # important sequence semantics (uniqueness + gaps on rollback) without
        # taking a second SQLite write transaction while the ingestion
        # transaction is active. PostgreSQL remains the production path above.
        global _sqlite_ordinal_counter
        with _sqlite_ordinal_lock:
            if _sqlite_ordinal_counter is None:
                current = session.execute(
                    text("SELECT COALESCE(MAX(submission_ordinal), 0) FROM ingestion_records")
                ).scalar_one()
                _sqlite_ordinal_counter = int(current)
            _sqlite_ordinal_counter += 1
            return _sqlite_ordinal_counter
