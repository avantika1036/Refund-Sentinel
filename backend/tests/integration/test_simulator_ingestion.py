"""Integration test for simulator event ingestion.

Verifies that simulator-generated events can pass through the existing
IngestionService without errors.
"""

import pytest

from backend.app.persistence.database import SessionLocal
from backend.app.persistence.ingestion_service import IngestionService
from backend.app.persistence.models import EventModel, IngestionRecordModel
from backend.app.simulator.background import BackgroundPopulationGenerator
from backend.app.simulator.scenarios import (
    AS01_DENSE_COORDINATED_REFUND_RING,
    LL01_LEGITIMATE_FAMILY,
)
from sqlalchemy import delete


@pytest.mark.integration
def test_background_events_pass_through_ingestion_service() -> None:
    """Background events can be ingested through the existing IngestionService."""
    gen = BackgroundPopulationGenerator(seed=42)
    output = gen.generate(num_customers=3, num_orders_per_customer=2)

    ingestion_service = IngestionService()

    for event in output.events:
        result = ingestion_service.ingest(event)
        # Should not raise any errors
        assert result is not None

    # Cleanup
    with SessionLocal() as session:
        session.execute(delete(EventModel))
        session.execute(delete(IngestionRecordModel))
        session.commit()


@pytest.mark.integration
def test_as01_events_pass_through_ingestion_service() -> None:
    """AS-01 events can be ingested through the existing IngestionService."""
    as01_gen = AS01_DENSE_COORDINATED_REFUND_RING(seed=42)
    output = as01_gen.generate(num_customers=3, orders_per_customer=2)

    ingestion_service = IngestionService()

    for event in output.events:
        result = ingestion_service.ingest(event)
        # Should not raise any errors
        assert result is not None

    # Cleanup
    with SessionLocal() as session:
        session.execute(delete(EventModel))
        session.execute(delete(IngestionRecordModel))
        session.commit()


@pytest.mark.integration
def test_ll01_events_pass_through_ingestion_service() -> None:
    """LL-01 events can be ingested through the existing IngestionService."""
    ll01_gen = LL01_LEGITIMATE_FAMILY(seed=42)
    output = ll01_gen.generate(num_family_members=3, orders_per_member=2)

    ingestion_service = IngestionService()

    for event in output.events:
        result = ingestion_service.ingest(event)
        # Should not raise any errors
        assert result is not None

    # Cleanup
    with SessionLocal() as session:
        session.execute(delete(EventModel))
        session.execute(delete(IngestionRecordModel))
        session.commit()
