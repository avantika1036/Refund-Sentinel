from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import delete

from backend.app.domain.enums import (
    DataSource,
    EventType,
    RefundReasonCode,
    RefundStatus,
)
from backend.app.domain.events import (
    EventEnvelope,
    PaymentCapturedEvent,
    PaymentCapturedPayload,
    PaymentCreatedEvent,
    PaymentCreatedPayload,
    RefundCreatedEvent,
    RefundCreatedPayload,
    RefundProcessedEvent,
    RefundProcessedPayload,
    RefundRequestedEvent,
    RefundRequestedPayload,
)
from backend.app.domain.identifiers import CustomerId, EventId, MerchantId, OrderId, PaymentId, RefundId
from backend.app.finance.types import IngestionOutcome
from backend.app.domain.value_objects import Money, UTCDateTime
from backend.app.persistence.database import Base, SessionLocal, engine
from backend.app.persistence.ingestion_service import IngestionService
from backend.app.persistence.models import EventModel, IngestionRecordModel, PendingEventModel, QuarantineRecordModel
from backend.app.persistence.reconstruction import ReconstructionService


@pytest.mark.integration
def test_reconstruction_uses_postgres_after_service_restart() -> None:
    Base.metadata.create_all(engine, checkfirst=True)
    payment, order, merchant, customer, event_id = PaymentId.generate(), OrderId.generate(), MerchantId.generate(), CustomerId.generate(), EventId.generate()
    time = UTCDateTime(value=datetime(2024, 1, 1, 10, tzinfo=timezone.utc))
    event = PaymentCreatedEvent(envelope=EventEnvelope(event_id=event_id, event_type=EventType.PAYMENT_CREATED, occurred_at=time, received_at=time, source=DataSource.SIMULATOR), payload=PaymentCreatedPayload(payment_id=payment, order_id=order, merchant_id=merchant, customer_id=customer, amount=Money.of_paise(100)))
    try:
        before = IngestionService().ingest(event)
        assert before.retained is True
        with SessionLocal() as new_session:
            snapshot = ReconstructionService(new_session).reconstruct()
        assert snapshot.event_count == 1
        assert snapshot.payments[payment].authorised_amount == Money.of_paise(100)
    finally:
        with SessionLocal.begin() as session:
            session.execute(delete(IngestionRecordModel).where(IngestionRecordModel.event_id == event_id.value))
            session.execute(delete(PendingEventModel).where(PendingEventModel.event_id == event_id.value))
            session.execute(delete(QuarantineRecordModel).where(QuarantineRecordModel.event_id == event_id.value))
            session.execute(delete(EventModel).where(EventModel.event_id == event_id.value))


@pytest.mark.integration
def test_restart_reconstruction_uses_occurred_at_after_reverse_delivery() -> None:
    """A captured fact arriving first is pending, then reconstructs chronologically."""
    Base.metadata.create_all(engine, checkfirst=True)
    payment, order, merchant, customer = PaymentId.generate(), OrderId.generate(), MerchantId.generate(), CustomerId.generate()
    captured_id, created_id = EventId.generate(), EventId.generate()
    created_at = UTCDateTime(value=datetime(2024, 1, 1, 10, tzinfo=timezone.utc))
    captured_at = UTCDateTime(value=datetime(2024, 1, 1, 11, tzinfo=timezone.utc))
    captured = PaymentCapturedEvent(envelope=EventEnvelope(event_id=captured_id, event_type=EventType.PAYMENT_CAPTURED, occurred_at=captured_at, received_at=created_at, source=DataSource.SIMULATOR), payload=PaymentCapturedPayload(payment_id=payment, merchant_id=merchant, captured_amount=Money.of_paise(100), captured_at=captured_at))
    created_event = PaymentCreatedEvent(envelope=EventEnvelope(event_id=created_id, event_type=EventType.PAYMENT_CREATED, occurred_at=created_at, received_at=captured_at, source=DataSource.SIMULATOR), payload=PaymentCreatedPayload(payment_id=payment, order_id=order, merchant_id=merchant, customer_id=customer, amount=Money.of_paise(100)))
    try:
        service = IngestionService()
        assert service.ingest(captured).ingestion_outcome is IngestionOutcome.PENDING
        assert service.ingest(created_event).ingestion_outcome is IngestionOutcome.RETAINED
        with SessionLocal() as session:
            snapshot = ReconstructionService(session).reconstruct()
            assert snapshot.payments[payment].status.value == "captured"
            assert snapshot.event_count == 2
            assert session.get(PendingEventModel, captured_id.value) is None
    finally:
        with SessionLocal.begin() as session:
            for event_id in (captured_id, created_id):
                session.execute(delete(IngestionRecordModel).where(IngestionRecordModel.event_id == event_id.value))
                session.execute(delete(PendingEventModel).where(PendingEventModel.event_id == event_id.value))
                session.execute(delete(QuarantineRecordModel).where(QuarantineRecordModel.event_id == event_id.value))
                session.execute(delete(EventModel).where(EventModel.event_id == event_id.value))


@pytest.mark.integration
def test_duplicate_before_restart_reconstructs_one_fact() -> None:
    Base.metadata.create_all(engine, checkfirst=True)
    payment, order, merchant, customer, event_id = PaymentId.generate(), OrderId.generate(), MerchantId.generate(), CustomerId.generate(), EventId.generate()
    event = PaymentCreatedEvent(envelope=EventEnvelope(event_id=event_id, event_type=EventType.PAYMENT_CREATED, occurred_at=UTCDateTime(value=datetime(2024, 1, 1, 10, tzinfo=timezone.utc)), received_at=UTCDateTime(value=datetime(2024, 1, 1, 10, tzinfo=timezone.utc)), source=DataSource.SIMULATOR), payload=PaymentCreatedPayload(payment_id=payment, order_id=order, merchant_id=merchant, customer_id=customer, amount=Money.of_paise(100)))
    try:
        service = IngestionService(); service.ingest(event)
        assert service.ingest(PaymentCreatedEvent(envelope=event.envelope, payload=event.payload)).ingestion_outcome is IngestionOutcome.DUPLICATE
        with SessionLocal() as session:
            assert session.query(EventModel).filter_by(event_id=event_id.value).count() == 1
            assert session.query(IngestionRecordModel).filter_by(event_id=event_id.value).count() == 2
            snapshot = ReconstructionService(session).reconstruct()
            assert snapshot.event_count == 1 and payment in snapshot.payments
    finally:
        with SessionLocal.begin() as session:
            session.execute(delete(IngestionRecordModel).where(IngestionRecordModel.event_id == event_id.value))
            session.execute(delete(PendingEventModel).where(PendingEventModel.event_id == event_id.value))
            session.execute(delete(QuarantineRecordModel).where(QuarantineRecordModel.event_id == event_id.value))
            session.execute(delete(EventModel).where(EventModel.event_id == event_id.value))

@pytest.mark.integration
def test_late_prerequisite_promotion_survives_restart() -> None:
    """Pending dependency chain is promoted, then rebuilt from durable events."""
    Base.metadata.create_all(engine, checkfirst=True)

    payment = PaymentId.generate()
    order = OrderId.generate()
    merchant = MerchantId.generate()
    customer = CustomerId.generate()
    refund = RefundId.generate()

    created_id = EventId.generate()
    captured_id = EventId.generate()
    requested_id = EventId.generate()
    refund_created_id = EventId.generate()
    processed_id = EventId.generate()

    created_at = UTCDateTime(
        value=datetime(2024, 6, 1, 10, tzinfo=timezone.utc)
    )
    captured_at = UTCDateTime(
        value=datetime(2024, 6, 1, 11, tzinfo=timezone.utc)
    )
    requested_at = UTCDateTime(
        value=datetime(2024, 6, 1, 12, tzinfo=timezone.utc)
    )
    refund_created_at = UTCDateTime(
        value=datetime(2024, 6, 1, 13, tzinfo=timezone.utc)
    )
    processed_at = UTCDateTime(
        value=datetime(2024, 6, 1, 14, tzinfo=timezone.utc)
    )

    created = PaymentCreatedEvent(
        envelope=EventEnvelope(
            event_id=created_id,
            event_type=EventType.PAYMENT_CREATED,
            occurred_at=created_at,
            received_at=created_at,
            source=DataSource.SIMULATOR,
        ),
        payload=PaymentCreatedPayload(
            payment_id=payment,
            order_id=order,
            merchant_id=merchant,
            customer_id=customer,
            amount=Money.of_paise(100_000),
        ),
    )

    captured = PaymentCapturedEvent(
        envelope=EventEnvelope(
            event_id=captured_id,
            event_type=EventType.PAYMENT_CAPTURED,
            occurred_at=captured_at,
            received_at=captured_at,
            source=DataSource.SIMULATOR,
        ),
        payload=PaymentCapturedPayload(
            payment_id=payment,
            merchant_id=merchant,
            captured_amount=Money.of_paise(100_000),
            captured_at=captured_at,
        ),
    )

    requested = RefundRequestedEvent(
        envelope=EventEnvelope(
            event_id=requested_id,
            event_type=EventType.REFUND_REQUESTED,
            occurred_at=requested_at,
            received_at=requested_at,
            source=DataSource.SIMULATOR,
        ),
        payload=RefundRequestedPayload(
            refund_id=refund,
            payment_id=payment,
            order_id=order,
            merchant_id=merchant,
            customer_id=customer,
            amount=Money.of_paise(50_000),
            reason_code=RefundReasonCode.DEFECTIVE,
        ),
    )

    refund_created = RefundCreatedEvent(
        envelope=EventEnvelope(
            event_id=refund_created_id,
            event_type=EventType.REFUND_CREATED,
            occurred_at=refund_created_at,
            received_at=refund_created_at,
            source=DataSource.SIMULATOR,
        ),
        payload=RefundCreatedPayload(
            refund_id=refund,
            payment_id=payment,
            merchant_id=merchant,
            created_at=refund_created_at,
        ),
    )

    processed = RefundProcessedEvent(
        envelope=EventEnvelope(
            event_id=processed_id,
            event_type=EventType.REFUND_PROCESSED,
            occurred_at=processed_at,
            received_at=processed_at,
            source=DataSource.SIMULATOR,
        ),
        payload=RefundProcessedPayload(
            refund_id=refund,
            payment_id=payment,
            merchant_id=merchant,
            processed_at=processed_at,
            processed_amount=Money.of_paise(50_000),
        ),
    )

    # Deliberately reverse the dependency order.
    reverse_order = [
        processed,
        refund_created,
        requested,
        captured,
        created,
    ]

    try:
        service = IngestionService()

        for event in reverse_order[:4]:
            result = service.ingest(event)
            assert result.ingestion_outcome is IngestionOutcome.PENDING
            assert result.pending is True

        final_result = service.ingest(reverse_order[4])

        assert final_result.ingestion_outcome is IngestionOutcome.RETAINED
        assert final_result.triggered_reconstruction is True

        with SessionLocal() as session:
            assert session.query(PendingEventModel).count() == 0

            audit = (
                session.query(IngestionRecordModel)
                .filter(
                    IngestionRecordModel.event_id.in_(
                        [
                            created_id.value,
                            captured_id.value,
                            requested_id.value,
                            refund_created_id.value,
                            processed_id.value,
                        ]
                    )
                )
                .all()
            )
            assert len(audit) == 5

            # Simulate a process restart: reconstruct using a fresh session.
            with SessionLocal() as restarted_session:
                snapshot = ReconstructionService(
                    restarted_session
                ).reconstruct()

                assert snapshot.event_count == 5
                assert snapshot.payments[payment].cumulative_refunded.amount_paise == 50_000
                assert snapshot.refunds[refund].status is RefundStatus.PROCESSED

    finally:
        with SessionLocal.begin() as session:
            event_ids = [
                created_id,
                captured_id,
                requested_id,
                refund_created_id,
                processed_id,
            ]

            for event_id in event_ids:
                session.execute(
                    delete(IngestionRecordModel).where(
                        IngestionRecordModel.event_id == event_id.value
                    )
                )
                session.execute(
                    delete(PendingEventModel).where(
                        PendingEventModel.event_id == event_id.value
                    )
                )
                session.execute(
                    delete(QuarantineRecordModel).where(
                        QuarantineRecordModel.event_id == event_id.value
                    )
                )
                session.execute(
                    delete(EventModel).where(
                        EventModel.event_id == event_id.value
                    )
                )
