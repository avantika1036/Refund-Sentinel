from __future__ import annotations

from datetime import datetime, timezone
from threading import Barrier, Thread

import pytest
from sqlalchemy import delete

from backend.app.domain.enums import DataSource, EventType
from backend.app.domain.events import EventEnvelope, PaymentCapturedEvent, PaymentCapturedPayload, PaymentCreatedEvent, PaymentCreatedPayload, RefundCreatedEvent, RefundCreatedPayload, RefundProcessedEvent, RefundProcessedPayload, RefundRequestedEvent, RefundRequestedPayload
from backend.app.domain.identifiers import CustomerId, EventId, MerchantId, OrderId, PaymentId, RefundId
from backend.app.domain.value_objects import Money, UTCDateTime
from backend.app.domain.enums import RefundReasonCode
from backend.app.finance.processing import ReasonCode
from backend.app.finance.types import IngestionOutcome
from backend.app.persistence.database import Base, SessionLocal, engine
from backend.app.persistence.ingestion_service import IngestionService
from backend.app.persistence.models import EventModel, IngestionRecordModel, PendingEventModel, QuarantineRecordModel

POSTGRES_ONLY = pytest.mark.skipif(engine.dialect.name != "postgresql", reason="requires PostgreSQL transaction/concurrency semantics")


def created(payment_id: PaymentId, order_id: OrderId, merchant_id: MerchantId, customer_id: CustomerId, *, event_id: EventId | None = None) -> PaymentCreatedEvent:
    time = UTCDateTime(value=datetime(2024, 1, 1, 10, tzinfo=timezone.utc))
    return PaymentCreatedEvent(envelope=EventEnvelope(event_id=event_id or EventId.generate(), event_type=EventType.PAYMENT_CREATED, occurred_at=time, received_at=time, source=DataSource.SIMULATOR), payload=PaymentCreatedPayload(payment_id=payment_id, order_id=order_id, merchant_id=merchant_id, customer_id=customer_id, amount=Money.of_paise(100)))


@pytest.fixture(scope="module", autouse=True)
def tables() -> None:
    Base.metadata.create_all(engine, checkfirst=True)


@pytest.mark.integration
def test_service_persists_retained_duplicate_conflict_and_audits() -> None:
    payment, order, merchant, customer, event_id = PaymentId.generate(), OrderId.generate(), MerchantId.generate(), CustomerId.generate(), EventId.generate()
    original = created(payment, order, merchant, customer, event_id=event_id)
    conflict = PaymentCreatedEvent(envelope=original.envelope, payload=PaymentCreatedPayload(payment_id=payment, order_id=order, merchant_id=merchant, customer_id=customer, amount=Money.of_paise(99)))
    service = IngestionService()
    try:
        assert service.ingest(original).ingestion_outcome is IngestionOutcome.RETAINED
        assert service.ingest(PaymentCreatedEvent(envelope=original.envelope, payload=original.payload)).ingestion_outcome is IngestionOutcome.DUPLICATE
        assert service.ingest(conflict).ingestion_outcome is IngestionOutcome.CONFLICT
        with SessionLocal() as session:
            assert session.get(EventModel, event_id.value) is not None
            assert len(session.query(IngestionRecordModel).filter_by(event_id=event_id.value).all()) == 3
    finally:
        with SessionLocal.begin() as session:
            session.execute(delete(IngestionRecordModel).where(IngestionRecordModel.event_id == event_id.value))
            session.execute(delete(PendingEventModel).where(PendingEventModel.event_id == event_id.value))
            session.execute(delete(QuarantineRecordModel).where(QuarantineRecordModel.event_id == event_id.value))
            session.execute(delete(EventModel).where(EventModel.event_id == event_id.value))


@pytest.mark.integration
def test_service_pending_event_is_promoted_without_second_audit_row() -> None:
    payment, order, merchant, customer = PaymentId.generate(), OrderId.generate(), MerchantId.generate(), CustomerId.generate()
    pending_id, created_id = EventId.generate(), EventId.generate()
    captured_time = UTCDateTime(value=datetime(2024, 1, 1, 11, tzinfo=timezone.utc))
    captured = PaymentCapturedEvent(envelope=EventEnvelope(event_id=pending_id, event_type=EventType.PAYMENT_CAPTURED, occurred_at=captured_time, received_at=captured_time, source=DataSource.SIMULATOR), payload=PaymentCapturedPayload(payment_id=payment, merchant_id=merchant, captured_amount=Money.of_paise(100), captured_at=captured_time))
    service = IngestionService()
    try:
        assert service.ingest(captured).ingestion_outcome is IngestionOutcome.PENDING
        service.ingest(created(payment, order, merchant, customer, event_id=created_id))
        with SessionLocal() as session:
            audit = session.query(IngestionRecordModel).filter_by(event_id=pending_id.value).one()
            assert audit.retained is True and audit.pending is False
            assert session.get(PendingEventModel, pending_id.value) is None
    finally:
        with SessionLocal.begin() as session:
            for event_id in (pending_id, created_id):
                session.execute(delete(IngestionRecordModel).where(IngestionRecordModel.event_id == event_id.value))
                session.execute(delete(PendingEventModel).where(PendingEventModel.event_id == event_id.value))
                session.execute(delete(QuarantineRecordModel).where(QuarantineRecordModel.event_id == event_id.value))
                session.execute(delete(EventModel).where(EventModel.event_id == event_id.value))


@pytest.mark.integration
def test_service_rolls_back_event_when_later_step_fails() -> None:
    payment, order, merchant, customer = PaymentId.generate(), OrderId.generate(), MerchantId.generate(), CustomerId.generate()
    event = created(payment, order, merchant, customer)

    def fail_after_event_saved() -> None:
        raise RuntimeError("test-only persistence failure")

    with pytest.raises(RuntimeError, match="test-only"):
        IngestionService(after_event_saved=fail_after_event_saved).ingest(event)
    with SessionLocal() as session:
        assert session.get(EventModel, event.envelope.event_id.value) is None
        assert session.query(IngestionRecordModel).filter_by(event_id=event.envelope.event_id.value).count() == 0
        assert session.get(PendingEventModel, event.envelope.event_id.value) is None


@pytest.mark.integration
@POSTGRES_ONLY
def test_service_handles_real_concurrent_identical_submissions() -> None:
    payment, order, merchant, customer, event_id = PaymentId.generate(), OrderId.generate(), MerchantId.generate(), CustomerId.generate(), EventId.generate()
    event = created(payment, order, merchant, customer, event_id=event_id)
    barrier, outcomes, failures = Barrier(2), [], []

    def submit() -> None:
        try:
            barrier.wait()
            outcomes.append(IngestionService().ingest(event).ingestion_outcome)
        except BaseException as exc:
            failures.append(exc)

    threads = [Thread(target=submit), Thread(target=submit)]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    try:
        assert failures == []
        assert sorted(outcome.value for outcome in outcomes) == ["duplicate", "retained"]
        with SessionLocal() as session:
            assert session.query(EventModel).filter_by(event_id=event_id.value).count() == 1
            assert session.query(IngestionRecordModel).filter_by(event_id=event_id.value).count() == 2
    finally:
        with SessionLocal.begin() as session:
            session.execute(delete(IngestionRecordModel).where(IngestionRecordModel.event_id == event_id.value))
            session.execute(delete(PendingEventModel).where(PendingEventModel.event_id == event_id.value))
            session.execute(delete(QuarantineRecordModel).where(QuarantineRecordModel.event_id == event_id.value))
            session.execute(delete(EventModel).where(EventModel.event_id == event_id.value))


def refund_flow(service: IngestionService):
    payment, order, merchant, customer, refund = PaymentId.generate(), OrderId.generate(), MerchantId.generate(), CustomerId.generate(), RefundId.generate()
    ids = [EventId.generate() for _ in range(5)]
    service.ingest(created(payment, order, merchant, customer, event_id=ids[0]))
    ts = lambda hour: UTCDateTime(value=datetime(2024, 1, 1, hour, tzinfo=timezone.utc))
    service.ingest(PaymentCapturedEvent(envelope=EventEnvelope(event_id=ids[1], event_type=EventType.PAYMENT_CAPTURED, occurred_at=ts(11), received_at=ts(11), source=DataSource.SIMULATOR), payload=PaymentCapturedPayload(payment_id=payment, merchant_id=merchant, captured_amount=Money.of_paise(100), captured_at=ts(11))))
    service.ingest(RefundRequestedEvent(envelope=EventEnvelope(event_id=ids[2], event_type=EventType.REFUND_REQUESTED, occurred_at=ts(12), received_at=ts(12), source=DataSource.SIMULATOR), payload=RefundRequestedPayload(refund_id=refund, payment_id=payment, order_id=order, merchant_id=merchant, customer_id=customer, amount=Money.of_paise(50), reason_code=RefundReasonCode.DEFECTIVE)))
    service.ingest(RefundCreatedEvent(envelope=EventEnvelope(event_id=ids[3], event_type=EventType.REFUND_CREATED, occurred_at=ts(13), received_at=ts(13), source=DataSource.SIMULATOR), payload=RefundCreatedPayload(refund_id=refund, payment_id=payment, merchant_id=merchant, created_at=ts(13))))
    processed = RefundProcessedEvent(envelope=EventEnvelope(event_id=ids[4], event_type=EventType.REFUND_PROCESSED, occurred_at=ts(14), received_at=ts(14), source=DataSource.SIMULATOR), payload=RefundProcessedPayload(refund_id=refund, payment_id=payment, merchant_id=merchant, processed_at=ts(14), processed_amount=Money.of_paise(51)))
    return processed, ids


@pytest.mark.integration
def test_service_persists_engine_quarantine_and_rolls_back_on_quarantine_failure(monkeypatch) -> None:
    service = IngestionService(); processed, ids = refund_flow(service)
    try:
        result = service.ingest(processed)
        assert result.ingestion_outcome is IngestionOutcome.QUARANTINED
        with SessionLocal() as session:
            assert session.get(PendingEventModel, processed.envelope.event_id.value) is None
            quarantine = session.query(QuarantineRecordModel).filter_by(event_id=processed.envelope.event_id.value).one()
            assert quarantine.reason_code == ReasonCode.PROCESSED_AMOUNT_EXCEEDS_REQUESTED.value
            assert session.query(IngestionRecordModel).filter_by(event_id=processed.envelope.event_id.value).count() == 1
        failing_service = IngestionService(); failing_processed, failing_ids = refund_flow(failing_service)
        from backend.app.persistence.repositories.quarantine import QuarantineRepository
        monkeypatch.setattr(QuarantineRepository, "save", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("quarantine failure")))
        with pytest.raises(RuntimeError, match="quarantine failure"):
            failing_service.ingest(failing_processed)
        with SessionLocal() as session:
            assert session.get(EventModel, failing_processed.envelope.event_id.value) is None
            assert session.query(IngestionRecordModel).filter_by(event_id=failing_processed.envelope.event_id.value).count() == 0
        ids.extend(failing_ids)
    finally:
        with SessionLocal.begin() as session:
            for event_id in ids:
                session.execute(delete(IngestionRecordModel).where(IngestionRecordModel.event_id == event_id.value))
                session.execute(delete(PendingEventModel).where(PendingEventModel.event_id == event_id.value))
                session.execute(delete(QuarantineRecordModel).where(QuarantineRecordModel.event_id == event_id.value))
                session.execute(delete(EventModel).where(EventModel.event_id == event_id.value))
