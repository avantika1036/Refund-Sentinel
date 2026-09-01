"""Integration tests for the complete refund investigation pipeline.

Verifies the real pipeline:

Simulator
    -> IngestionService
    -> PostgreSQL
    -> ReconstructionService
    -> InvestigationService
"""

from sqlalchemy import delete

import pytest

from backend.app.persistence.database import SessionLocal
from backend.app.persistence.ingestion_service import IngestionService
from backend.app.persistence.models import (
    EventModel,
    IngestionRecordModel,
)
from backend.app.persistence.reconstruction import ReconstructionService
from backend.app.risk.investigation import InvestigationService
from backend.app.simulator.scenarios import (
    AS01_DENSE_COORDINATED_REFUND_RING,
)


def _cleanup_database() -> None:
    """Remove test data created by integration tests."""

    with SessionLocal() as session:
        session.execute(delete(EventModel))
        session.execute(delete(IngestionRecordModel))
        session.commit()


@pytest.mark.integration
def test_complete_investigation_pipeline_from_persisted_events() -> None:
    """Run a complete investigation from persisted simulator events.

    This verifies that a refund can travel through the entire production
    backend pipeline without manually constructing a snapshot.
    """

    _cleanup_database()

    try:
        generator = AS01_DENSE_COORDINATED_REFUND_RING(seed=42)

        output = generator.generate(
            num_customers=3,
            orders_per_customer=2,
        )

        ingestion_service = IngestionService()

        for event in output.events:
            ingestion_service.ingest(event)

        abuse_labels = [
            label
            for label in output.get_abuse_labels()
            if label.refund_id is not None
        ]

        assert abuse_labels, (
            "AS01 scenario should generate at least one labeled refund"
        )

        target_refund_id = abuse_labels[0].refund_id

        assert target_refund_id is not None

        with SessionLocal() as session:
            snapshot = ReconstructionService(
                session
            ).reconstruct()

        assert target_refund_id in snapshot.refunds

        investigation = InvestigationService(
            snapshot
        ).investigate(target_refund_id)

        assert (
            investigation.assessment.refund_id
            == target_refund_id
        )

        assert (
            investigation.decision.final_score
            == investigation.assessment.risk_score.final_score
        )

        assert target_refund_id in (
            investigation.component_refund_ids
        )

        assert (
            investigation.exposure
            .realized_suspicious_amount
            .amount_paise
            >= 0
        )

        assert (
            investigation.exposure
            .pending_refund_exposure
            .amount_paise
            >= 0
        )

        assert (
            investigation.exposure
            .remaining_refundable_exposure
            .amount_paise
            >= 0
        )

    finally:
        _cleanup_database()


@pytest.mark.integration
def test_investigation_is_deterministic_for_same_snapshot() -> None:
    """The same reconstructed state should produce the same investigation."""

    _cleanup_database()

    try:
        generator = AS01_DENSE_COORDINATED_REFUND_RING(seed=42)

        output = generator.generate(
            num_customers=3,
            orders_per_customer=2,
        )

        ingestion_service = IngestionService()

        for event in output.events:
            ingestion_service.ingest(event)

        abuse_labels = [
            label
            for label in output.get_abuse_labels()
            if label.refund_id is not None
        ]

        assert abuse_labels

        target_refund_id = abuse_labels[0].refund_id

        assert target_refund_id is not None

        with SessionLocal() as session:
            snapshot = ReconstructionService(
                session
            ).reconstruct()

        service = InvestigationService(snapshot)

        first = service.investigate(
            target_refund_id
        )

        second = service.investigate(
            target_refund_id
        )

        assert (
            first.assessment.risk_score.final_score
            == second.assessment.risk_score.final_score
        )

        assert (
            first.decision.risk_level
            == second.decision.risk_level
        )

        assert (
            first.decision.action
            == second.decision.action
        )

        assert (
            first.component_refund_ids
            == second.component_refund_ids
        )

        assert first.exposure == second.exposure

    finally:
        _cleanup_database()