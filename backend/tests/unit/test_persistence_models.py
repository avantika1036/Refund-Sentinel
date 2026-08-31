from __future__ import annotations

from sqlalchemy import inspect

from backend.app.persistence.database import Base
from backend.app.persistence.models import (
    EventModel,
    IngestionRecordModel,
    PendingEventModel,
    QuarantineRecordModel,
)


def test_all_persistence_models_are_registered_with_metadata():
    tables = set(Base.metadata.tables)
    assert {
        "events",
        "ingestion_records",
        "pending_events",
        "quarantine_records",
    } <= tables


def test_event_model_is_keyed_by_event_id():
    primary_key = inspect(EventModel).primary_key
    assert [column.name for column in primary_key] == ["event_id"]


def test_pending_event_model_is_keyed_by_event_id():
    primary_key = inspect(PendingEventModel).primary_key
    assert [column.name for column in primary_key] == ["event_id"]


def test_ingestion_record_has_unique_event_submission_constraint():
    constraints = {
        constraint.name
        for constraint in IngestionRecordModel.__table__.constraints
        if constraint.name is not None
    }
    assert "uq_ingestion_event_submission" in constraints


def test_quarantine_is_separate_from_pending():
    assert QuarantineRecordModel.__tablename__ != PendingEventModel.__tablename__
