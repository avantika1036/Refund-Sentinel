"""Integration test fixtures ensuring a clean persistence state for each test."""

from __future__ import annotations

import pytest
from sqlalchemy import delete

from backend.app.persistence.database import Base, SessionLocal, engine
from backend.app.persistence.models import (
    EventModel,
    IngestionRecordModel,
    PendingEventModel,
    QuarantineRecordModel,
)


@pytest.fixture(autouse=True)
def clean_database():
    """Ensure database tables are created and clean before and after each integration test."""
    Base.metadata.create_all(engine, checkfirst=True)
    with SessionLocal() as session:
        session.execute(delete(QuarantineRecordModel))
        session.execute(delete(PendingEventModel))
        session.execute(delete(IngestionRecordModel))
        session.execute(delete(EventModel))
        session.commit()
    yield
    with SessionLocal() as session:
        session.execute(delete(QuarantineRecordModel))
        session.execute(delete(PendingEventModel))
        session.execute(delete(IngestionRecordModel))
        session.execute(delete(EventModel))
        session.commit()
