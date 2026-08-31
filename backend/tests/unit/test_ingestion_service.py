from __future__ import annotations

from unittest.mock import Mock

from backend.app.persistence.ingestion_service import IngestionService


def test_submission_ordinal_uses_durable_database_sequence() -> None:
    result = Mock(); result.scalar_one.return_value = 42
    session = Mock(); session.execute.return_value = result

    assert IngestionService._next_submission_ordinal(session) == 42
    session.execute.assert_called_once()


def test_service_exposes_small_ingest_boundary() -> None:
    assert callable(IngestionService.ingest)
