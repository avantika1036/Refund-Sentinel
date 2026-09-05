from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.persistence.database import Base


class EventModel(Base):
    """Durable immutable event ledger row.

    The payload is retained exactly as submitted. Domain reconstruction remains
    responsible for interpreting the event; this model does not duplicate
    domain state-machine logic.
    """

    __tablename__ = "events"

    event_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    event_type: Mapped[str] = mapped_column(String(100), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    payload_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON().with_variant(JSONB, "postgresql"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )




class IngestionRecordModel(Base):
    """Durable audit row for exactly one ingestion submission."""

    __tablename__ = "ingestion_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    submission_ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    ingestion_outcome: Mapped[str] = mapped_column(String(50), nullable=False)
    reason_code: Mapped[str | None] = mapped_column(String(100), nullable=True)
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    retained: Mapped[bool] = mapped_column(nullable=False, default=False)
    pending: Mapped[bool] = mapped_column(nullable=False, default=False)
    promoted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    triggered_reconstruction: Mapped[bool] = mapped_column(nullable=False, default=False)
    reconstruction_ordinal: Mapped[int | None] = mapped_column(Integer, nullable=True)
    original_submission_ordinal: Mapped[int | None] = mapped_column(Integer, nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "event_id",
            "submission_ordinal",
            name="uq_ingestion_event_submission",
        ),
    )


class PendingEventModel(Base):
    """Durable pending-event record.

    One event ID can have at most one pending representation.
    """

    __tablename__ = "pending_events"

    event_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    event_type: Mapped[str] = mapped_column(String(100), nullable=False)
    payload_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON().with_variant(JSONB, "postgresql"), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class QuarantineRecordModel(Base):
    """Durable record of an event that could not safely enter canonical state."""

    __tablename__ = "quarantine_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    event_type: Mapped[str] = mapped_column(String(100), nullable=False)
    payload_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    reason_code: Mapped[str] = mapped_column(String(100), nullable=False)
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON().with_variant(JSONB, "postgresql"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
