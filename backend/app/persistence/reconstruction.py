"""Read-only reconstruction of financial state from the durable event ledger."""

from __future__ import annotations

from sqlalchemy.orm import Session

from backend.app.domain.enums import EventType
from backend.app.domain.identifiers import EventId
from backend.app.finance.state_engine import FinancialStateEngine
from backend.app.finance.types import ReconstructionAnomaly, ReconstructionSnapshot
from backend.app.persistence.repositories.events import EventRepository, deserialize_event


class ReconstructionService:
    """Rebuild a snapshot from persisted canonical events without writing data."""

    def __init__(self, session: Session) -> None:
        self._events = EventRepository(session)

    def reconstruct(self) -> ReconstructionSnapshot:
        events = []
        anomalies: list[ReconstructionAnomaly] = []
        for row in self._events.list_by_occurred_at():
            try:
                # The deserializer creates new immutable domain objects and
                # never changes JSONB dictionaries held by SQLAlchemy.
                events.append(deserialize_event(row.payload))
            except (TypeError, ValueError) as exc:
                anomalies.append(
                    ReconstructionAnomaly(
                        EventId(row.event_id), EventType(row.event_type),
                        f"Persisted event could not be deserialized: {exc}",
                    )
                )
        snapshot = FinancialStateEngine().reconstruct_from(events)
        return ReconstructionSnapshot(
            snapshot.payments, snapshot.refunds, snapshot.orders,
            snapshot.reconstruction_ordinal, snapshot.event_count,
            snapshot.anomalies + tuple(anomalies),
        )
