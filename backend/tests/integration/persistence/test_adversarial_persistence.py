"""Batch 3F: Adversarial database tests for persistence boundary hardening."""

from __future__ import annotations

from datetime import datetime, timezone
from threading import Barrier, Thread
from typing import Callable

import pytest
from sqlalchemy import delete, text
from sqlalchemy.exc import IntegrityError

from backend.app.domain.enums import DataSource, EventType
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
from backend.app.domain.identifiers import CustomerId, EventId, MerchantId, OrderId, PaymentId
from backend.app.domain.value_objects import Money, UTCDateTime
from backend.app.finance.types import IngestionOutcome
from backend.app.persistence.database import Base, SessionLocal, engine
from backend.app.persistence.ingestion_service import IngestionService
from backend.app.persistence.models import (
    EventModel,
    IngestionRecordModel,
    PendingEventModel,
    QuarantineRecordModel,
)
from backend.app.persistence.reconstruction import ReconstructionService
from backend.app.persistence.repositories.events import EventRepository, calculate_payload_hash


@pytest.fixture(scope="module", autouse=True)
def tables() -> None:
    Base.metadata.create_all(engine, checkfirst=True)


def payment_created(
    payment_id: PaymentId,
    order_id: OrderId,
    merchant_id: MerchantId,
    customer_id: CustomerId,
    *,
    event_id: EventId | None = None,
    amount: int = 100,
) -> PaymentCreatedEvent:
    time = UTCDateTime(value=datetime(2024, 1, 1, 10, tzinfo=timezone.utc))
    return PaymentCreatedEvent(
        envelope=EventEnvelope(
            event_id=event_id or EventId.generate(),
            event_type=EventType.PAYMENT_CREATED,
            occurred_at=time,
            received_at=time,
            source=DataSource.SIMULATOR,
        ),
        payload=PaymentCreatedPayload(
            payment_id=payment_id,
            order_id=order_id,
            merchant_id=merchant_id,
            customer_id=customer_id,
            amount=Money.of_paise(amount),
        ),
    )


def cleanup_event(event_id: EventId) -> None:
    with SessionLocal.begin() as session:
        session.execute(delete(IngestionRecordModel).where(IngestionRecordModel.event_id == event_id.value))
        session.execute(delete(PendingEventModel).where(PendingEventModel.event_id == event_id.value))
        session.execute(delete(QuarantineRecordModel).where(QuarantineRecordModel.event_id == event_id.value))
        session.execute(delete(EventModel).where(EventModel.event_id == event_id.value))


@pytest.mark.integration
def test_concurrent_identical_event_submission() -> None:
    """Two independent PostgreSQL sessions submit the same event_id and identical payload concurrently.
    
    Exactly one canonical EventModel exists.
    Exactly one submission is RETAINED/INSERTED and the other DUPLICATE.
    Both ingestion audit records survive.
    Losing transaction remains usable.
    No uncaught IntegrityError escapes the ingestion service.
    """
    payment, order, merchant, customer, event_id = (
        PaymentId.generate(),
        OrderId.generate(),
        MerchantId.generate(),
        CustomerId.generate(),
        EventId.generate(),
    )
    event = payment_created(payment, order, merchant, customer, event_id=event_id)
    barrier = Barrier(2)
    outcomes = []
    failures = []

    def submit() -> None:
        try:
            barrier.wait()
            result = IngestionService().ingest(event)
            outcomes.append(result.ingestion_outcome)
        except BaseException as exc:
            failures.append(exc)

    threads = [Thread(target=submit), Thread(target=submit)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    try:
        assert failures == [], f"Unexpected failures: {failures}"
        assert sorted(outcome.value for outcome in outcomes) == ["duplicate", "retained"]
        
        with SessionLocal() as session:
            # Exactly one canonical event exists
            canonical_events = session.query(EventModel).filter_by(event_id=event_id.value).all()
            assert len(canonical_events) == 1, f"Expected 1 canonical event, got {len(canonical_events)}"
            
            # Both ingestion audit records survive
            audit_records = session.query(IngestionRecordModel).filter_by(event_id=event_id.value).all()
            assert len(audit_records) == 2, f"Expected 2 audit records, got {len(audit_records)}"
            
            # One is retained, one is duplicate
            outcomes_sorted = sorted([r.ingestion_outcome for r in audit_records])
            assert outcomes_sorted == ["duplicate", "retained"]
            
            # Verify the canonical event has the correct payload
            canonical = canonical_events[0]
            assert canonical.payload_hash == calculate_payload_hash(event)
            assert canonical.payload["payload"]["amount"]["amount_paise"] == 100
    finally:
        cleanup_event(event_id)


@pytest.mark.integration
def test_concurrent_same_event_id_different_payloads() -> None:
    """Two independent sessions concurrently submit the same event_id but different business payloads.
    
    Exactly one canonical event survives.
    The winner is retained.
    The loser is classified CONFLICT after reading the durable winner.
    Both submissions have exactly one ingestion audit record.
    No canonical event is overwritten.
    """
    payment, order, merchant, customer, event_id = (
        PaymentId.generate(),
        OrderId.generate(),
        MerchantId.generate(),
        CustomerId.generate(),
        EventId.generate(),
    )
    
    event1 = payment_created(payment, order, merchant, customer, event_id=event_id, amount=100)
    event2 = payment_created(payment, order, merchant, customer, event_id=event_id, amount=200)
    
    barrier = Barrier(2)
    outcomes = []
    failures = []
    winner_payload = None

    def submit(event: PaymentCreatedEvent) -> None:
        nonlocal winner_payload
        try:
            barrier.wait()
            result = IngestionService().ingest(event)
            outcomes.append(result.ingestion_outcome)
            if result.ingestion_outcome == IngestionOutcome.RETAINED:
                winner_payload = event.payload.amount.amount_paise
        except BaseException as exc:
            failures.append(exc)

    threads = [Thread(target=submit, args=(event1,)), Thread(target=submit, args=(event2,))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    try:
        assert failures == [], f"Unexpected failures: {failures}"
        assert sorted(outcome.value for outcome in outcomes) == ["conflict", "retained"]
        
        with SessionLocal() as session:
            # Exactly one canonical event exists
            canonical_events = session.query(EventModel).filter_by(event_id=event_id.value).all()
            assert len(canonical_events) == 1, f"Expected 1 canonical event, got {len(canonical_events)}"
            
            # Both ingestion audit records survive
            audit_records = session.query(IngestionRecordModel).filter_by(event_id=event_id.value).all()
            assert len(audit_records) == 2, f"Expected 2 audit records, got {len(audit_records)}"
            
            # One is retained, one is conflict
            outcomes_sorted = sorted([r.ingestion_outcome for r in audit_records])
            assert outcomes_sorted == ["conflict", "retained"]
            
            # Verify the canonical event was NOT overwritten
            canonical = canonical_events[0]
            canonical_amount = canonical.payload["payload"]["amount"]["amount_paise"]
            assert canonical_amount in [100, 200], f"Unexpected canonical amount: {canonical_amount}"
            
            # The winner's payload is preserved
            assert canonical_amount == winner_payload
    finally:
        cleanup_event(event_id)


@pytest.mark.integration
def test_rollback_atomicity_after_event_saved() -> None:
    """Force a failure after canonical event persistence has been staged.
    
    Verify the transaction leaves no partial canonical event, pending row, 
    quarantine row, or ingestion audit row.
    Verify the same database session can still be used appropriately after the rollback.
    """
    payment, order, merchant, customer = (
        PaymentId.generate(),
        OrderId.generate(),
        MerchantId.generate(),
        CustomerId.generate(),
    )
    event = payment_created(payment, order, merchant, customer)

    def fail_after_event_saved() -> None:
        raise RuntimeError("test-only persistence failure")

    with pytest.raises(RuntimeError, match="test-only"):
        IngestionService(after_event_saved=fail_after_event_saved).ingest(event)

    with SessionLocal() as session:
        # No partial canonical event
        assert session.get(EventModel, event.envelope.event_id.value) is None
        
        # No pending row
        assert session.get(PendingEventModel, event.envelope.event_id.value) is None
        
        # No quarantine row
        quarantine = session.query(QuarantineRecordModel).filter_by(event_id=event.envelope.event_id.value).first()
        assert quarantine is None
        
        # No ingestion audit row
        audit = session.query(IngestionRecordModel).filter_by(event_id=event.envelope.event_id.value).first()
        assert audit is None
        
        # Verify the session can still be used
        test_event = payment_created(
            PaymentId.generate(),
            OrderId.generate(),
            MerchantId.generate(),
            CustomerId.generate(),
        )
        result = IngestionService().ingest(test_event)
        assert result.ingestion_outcome == IngestionOutcome.RETAINED
        cleanup_event(test_event.envelope.event_id)


@pytest.mark.integration
def test_pending_promotion_atomicity() -> None:
    """Create an event that initially becomes pending.
    
    Submit its prerequisite.
    Verify promotion does not create a second ingestion audit row for the original pending submission.
    Verify the pending row is removed.
    Verify exactly one canonical event exists for each retained event.
    """
    payment, order, merchant, customer = (
        PaymentId.generate(),
        OrderId.generate(),
        MerchantId.generate(),
        CustomerId.generate(),
    )
    pending_id, created_id = EventId.generate(), EventId.generate()
    
    captured_time = UTCDateTime(value=datetime(2024, 1, 1, 11, tzinfo=timezone.utc))
    captured = PaymentCapturedEvent(
        envelope=EventEnvelope(
            event_id=pending_id,
            event_type=EventType.PAYMENT_CAPTURED,
            occurred_at=captured_time,
            received_at=captured_time,
            source=DataSource.SIMULATOR,
        ),
        payload=PaymentCapturedPayload(
            payment_id=payment,
            merchant_id=merchant,
            captured_amount=Money.of_paise(100),
            captured_at=captured_time,
        ),
    )
    
    service = IngestionService()
    
    try:
        # Submit captured event first - should be pending
        pending_result = service.ingest(captured)
        assert pending_result.ingestion_outcome == IngestionOutcome.PENDING
        assert pending_result.pending is True
        
        # Get the initial audit record count
        with SessionLocal() as session:
            initial_audit_count = session.query(IngestionRecordModel).filter_by(event_id=pending_id.value).count()
            assert initial_audit_count == 1
        
        # Submit the prerequisite (PaymentCreated)
        created_result = service.ingest(payment_created(payment, order, merchant, customer, event_id=created_id))
        assert created_result.ingestion_outcome == IngestionOutcome.RETAINED
        
        with SessionLocal() as session:
            # Verify promotion did NOT create a second audit row
            final_audit_count = session.query(IngestionRecordModel).filter_by(event_id=pending_id.value).count()
            assert final_audit_count == 1, f"Expected 1 audit row after promotion, got {final_audit_count}"
            
            # Verify the pending row was removed
            assert session.get(PendingEventModel, pending_id.value) is None
            
            # Verify the audit row was updated to reflect promotion
            audit = session.query(IngestionRecordModel).filter_by(event_id=pending_id.value).one()
            assert audit.retained is True
            assert audit.pending is False
            assert audit.promoted_at is not None
            
            # Verify exactly one canonical event exists for each retained event
            canonical_count = session.query(EventModel).filter_by(event_id=pending_id.value).count()
            assert canonical_count == 1
            
            canonical_created_count = session.query(EventModel).filter_by(event_id=created_id.value).count()
            assert canonical_created_count == 1
    finally:
        cleanup_event(pending_id)
        cleanup_event(created_id)


@pytest.mark.integration
def test_duplicate_conflict_immutability() -> None:
    """Insert an event.
    
    Submit the same event again and verify the canonical row is unchanged.
    Submit the same event_id with a changed payload and verify the canonical row is still unchanged.
    Audit records may differ, but canonical business data must not be overwritten.
    """
    payment, order, merchant, customer, event_id = (
        PaymentId.generate(),
        OrderId.generate(),
        MerchantId.generate(),
        CustomerId.generate(),
        EventId.generate(),
    )
    
    original = payment_created(payment, order, merchant, customer, event_id=event_id, amount=100)
    service = IngestionService()
    
    try:
        # Insert original event
        original_result = service.ingest(original)
        assert original_result.ingestion_outcome == IngestionOutcome.RETAINED
        
        with SessionLocal() as session:
            original_canonical = session.get(EventModel, event_id.value)
            original_payload_hash = original_canonical.payload_hash
            original_amount = original_canonical.payload["payload"]["amount"]["amount_paise"]
        
        # Submit duplicate (same event_id, same payload)
        duplicate_result = service.ingest(original)
        assert duplicate_result.ingestion_outcome == IngestionOutcome.DUPLICATE
        
        with SessionLocal() as session:
            duplicate_canonical = session.get(EventModel, event_id.value)
            assert duplicate_canonical.payload_hash == original_payload_hash
            assert duplicate_canonical.payload["payload"]["amount"]["amount_paise"] == original_amount
        
        # Submit conflict (same event_id, different payload)
        conflict = payment_created(payment, order, merchant, customer, event_id=event_id, amount=999)
        conflict_result = service.ingest(conflict)
        assert conflict_result.ingestion_outcome == IngestionOutcome.CONFLICT
        
        with SessionLocal() as session:
            conflict_canonical = session.get(EventModel, event_id.value)
            # Canonical row must still be unchanged
            assert conflict_canonical.payload_hash == original_payload_hash
            assert conflict_canonical.payload["payload"]["amount"]["amount_paise"] == original_amount
            
            # Verify we have 3 audit records but only 1 canonical event
            audit_count = session.query(IngestionRecordModel).filter_by(event_id=event_id.value).count()
            assert audit_count == 3
            canonical_count = session.query(EventModel).filter_by(event_id=event_id.value).count()
            assert canonical_count == 1
    finally:
        cleanup_event(event_id)


@pytest.mark.integration
def test_database_uniqueness_enforcement() -> None:
    """Verify the relevant primary-key/unique constraints are actually enforced by PostgreSQL.
    
    Keep this focused on persistence invariants.
    """
    payment, order, merchant, customer, event_id = (
        PaymentId.generate(),
        OrderId.generate(),
        MerchantId.generate(),
        CustomerId.generate(),
        EventId.generate(),
    )
    
    event = payment_created(payment, order, merchant, customer, event_id=event_id)
    
    try:
        # Insert the event normally
        service = IngestionService()
        service.ingest(event)
        
        with SessionLocal() as session:
            # Try to insert a duplicate event_id directly into EventModel
            # This should violate the primary key constraint
            duplicate_row = EventModel(
                event_id=event_id.value,
                event_type=event.envelope.event_type.value,
                occurred_at=event.envelope.occurred_at.value,
                received_at=event.envelope.received_at.value,
                source=event.envelope.source.value,
                payload_hash=calculate_payload_hash(event),
                payload=event.model_dump(mode="json"),
                created_at=datetime.now(timezone.utc),
            )
            session.add(duplicate_row)
            
            with pytest.raises(IntegrityError):
                session.flush()
            
            session.rollback()
            
            # Verify ingestion_records unique constraint on (event_id, submission_ordinal)
            # Try to insert a duplicate (event_id, submission_ordinal) combination
            first_audit = session.query(IngestionRecordModel).filter_by(event_id=event_id.value).one()
            duplicate_audit = IngestionRecordModel(
                event_id=first_audit.event_id,
                submission_ordinal=first_audit.submission_ordinal,
                ingestion_outcome=first_audit.ingestion_outcome,
                reason_code=first_audit.reason_code,
                detail=first_audit.detail,
                retained=first_audit.retained,
                pending=first_audit.pending,
                promoted_at=first_audit.promoted_at,
                triggered_reconstruction=first_audit.triggered_reconstruction,
                reconstruction_ordinal=first_audit.reconstruction_ordinal,
                original_submission_ordinal=first_audit.original_submission_ordinal,
            )
            session.add(duplicate_audit)
            
            with pytest.raises(IntegrityError):
                session.flush()
    finally:
        cleanup_event(event_id)


@pytest.mark.integration
def test_reconstruction_after_adversarial_ingestion() -> None:
    """After adversarial scenarios, reconstruct from a fresh SQLAlchemy session.
    
    Verify reconstruction uses only the durable canonical event ledger.
    Verify duplicates/conflicts/pending/quarantine audit rows do not become duplicate domain facts.
    """
    payment, order, merchant, customer, event_id = (
        PaymentId.generate(),
        OrderId.generate(),
        MerchantId.generate(),
        CustomerId.generate(),
        EventId.generate(),
    )
    
    event = payment_created(payment, order, merchant, customer, event_id=event_id)
    service = IngestionService()
    
    try:
        # Ingest the event
        service.ingest(event)
        
        # Ingest a duplicate
        service.ingest(event)
        
        # Ingest a conflict
        conflict = payment_created(payment, order, merchant, customer, event_id=event_id, amount=999)
        service.ingest(conflict)
        
        # Reconstruct from a fresh session
        with SessionLocal() as session:
            reconstruction_service = ReconstructionService(session)
            snapshot = reconstruction_service.reconstruct()
            
            # Verify reconstruction uses only canonical events
            assert snapshot.event_count == 1
            assert payment in snapshot.payments
            
            # Verify the canonical amount is the original, not the conflict
            assert snapshot.payments[payment].authorised_amount.amount_paise == 100
            
            # Verify audit rows don't create duplicate domain facts
            audit_count = session.query(IngestionRecordModel).filter_by(event_id=event_id.value).count()
            assert audit_count == 3  # original + duplicate + conflict
            assert snapshot.event_count == 1  # only one domain fact
    finally:
        cleanup_event(event_id)


@pytest.mark.integration
def test_deterministic_ordering_with_inverted_received_at() -> None:
    """Insert events with intentionally inverted received_at values and identical occurred_at values.
    
    Verify canonical reconstruction order remains deterministic according to 
    the established occurred_at + event_id ordering.
    """
    payment, order, merchant, customer = (
        PaymentId.generate(),
        OrderId.generate(),
        MerchantId.generate(),
        CustomerId.generate(),
    )
    
    event_id1, event_id2 = EventId.generate(), EventId.generate()
    occurred_time = UTCDateTime(value=datetime(2024, 1, 1, 10, tzinfo=timezone.utc))
    
    # Event 1: occurred_at=10:00, received_at=12:00 (later received)
    event1 = PaymentCreatedEvent(
        envelope=EventEnvelope(
            event_id=event_id1,
            event_type=EventType.PAYMENT_CREATED,
            occurred_at=occurred_time,
            received_at=UTCDateTime(value=datetime(2024, 1, 1, 12, tzinfo=timezone.utc)),
            source=DataSource.SIMULATOR,
        ),
        payload=PaymentCreatedPayload(
            payment_id=payment,
            order_id=order,
            merchant_id=merchant,
            customer_id=customer,
            amount=Money.of_paise(100),
        ),
    )
    
    # Event 2: occurred_at=10:00, received_at=09:00 (earlier received)
    event2 = PaymentCreatedEvent(
        envelope=EventEnvelope(
            event_id=event_id2,
            event_type=EventType.PAYMENT_CREATED,
            occurred_at=occurred_time,
            received_at=UTCDateTime(value=datetime(2024, 1, 1, 9, tzinfo=timezone.utc)),
            source=DataSource.SIMULATOR,
        ),
        payload=PaymentCreatedPayload(
            payment_id=PaymentId.generate(),  # Different payment
            order_id=order,
            merchant_id=merchant,
            customer_id=customer,
            amount=Money.of_paise(200),
        ),
    )
    
    try:
        service = IngestionService()
        service.ingest(event1)
        service.ingest(event2)
        
        with SessionLocal() as session:
            reconstruction_service = ReconstructionService(session)
            events = EventRepository(session).list_by_occurred_at()
            
            # Verify ordering is by occurred_at, then event_id (deterministic)
            # Not by received_at
            assert len(events) == 2
            assert events[0].occurred_at == events[1].occurred_at  # Same occurred_at
            # event_id should be the tiebreaker
            event_ids = [events[0].event_id, events[1].event_id]
            assert event_ids == sorted(event_ids)  # Sorted by event_id
    finally:
        cleanup_event(event_id1)
        cleanup_event(event_id2)


@pytest.mark.integration
def test_quarantine_isolation() -> None:
    """Verify quarantined events remain separate from pending events.
    
    Verify quarantined events remain represented in the canonical/audit model 
    according to the existing ingestion design.
    Verify quarantine metadata survives a fresh database session.
    """
    from backend.app.finance.processing import ReasonCode
    from backend.app.domain.identifiers import RefundId
    from backend.app.domain.enums import RefundReasonCode
    
    payment, order, merchant, customer = (
        PaymentId.generate(),
        OrderId.generate(),
        MerchantId.generate(),
        CustomerId.generate(),
    )
    refund = RefundId.generate()
    
    # Set up a payment flow
    created_id = EventId.generate()
    captured_id = EventId.generate()
    requested_id = EventId.generate()
    refund_created_id = EventId.generate()
    processed_id = EventId.generate()
    
    time = lambda hour: UTCDateTime(value=datetime(2024, 1, 1, hour, tzinfo=timezone.utc))
    
    try:
        service = IngestionService()
        
        # Create payment flow
        service.ingest(
            PaymentCreatedEvent(
                envelope=EventEnvelope(
                    event_id=created_id,
                    event_type=EventType.PAYMENT_CREATED,
                    occurred_at=time(10),
                    received_at=time(10),
                    source=DataSource.SIMULATOR,
                ),
                payload=PaymentCreatedPayload(
                    payment_id=payment,
                    order_id=order,
                    merchant_id=merchant,
                    customer_id=customer,
                    amount=Money.of_paise(100),
                ),
            )
        )
        
        service.ingest(
            PaymentCapturedEvent(
                envelope=EventEnvelope(
                    event_id=captured_id,
                    event_type=EventType.PAYMENT_CAPTURED,
                    occurred_at=time(11),
                    received_at=time(11),
                    source=DataSource.SIMULATOR,
                ),
                payload=PaymentCapturedPayload(
                    payment_id=payment,
                    merchant_id=merchant,
                    captured_amount=Money.of_paise(100),
                    captured_at=time(11),
                ),
            )
        )
        
        service.ingest(
            RefundRequestedEvent(
                envelope=EventEnvelope(
                    event_id=requested_id,
                    event_type=EventType.REFUND_REQUESTED,
                    occurred_at=time(12),
                    received_at=time(12),
                    source=DataSource.SIMULATOR,
                ),
                payload=RefundRequestedPayload(
                    refund_id=refund,
                    payment_id=payment,
                    order_id=order,
                    merchant_id=merchant,
                    customer_id=customer,
                    amount=Money.of_paise(50),
                    reason_code=RefundReasonCode.DEFECTIVE,
                ),
            )
        )
        
        service.ingest(
            RefundCreatedEvent(
                envelope=EventEnvelope(
                    event_id=refund_created_id,
                    event_type=EventType.REFUND_CREATED,
                    occurred_at=time(13),
                    received_at=time(13),
                    source=DataSource.SIMULATOR,
                ),
                payload=RefundCreatedPayload(
                    refund_id=refund,
                    payment_id=payment,
                    merchant_id=merchant,
                    created_at=time(13),
                ),
            )
        )
        
        # Process a refund with amount exceeding requested (should be quarantined)
        processed = RefundProcessedEvent(
            envelope=EventEnvelope(
                event_id=processed_id,
                event_type=EventType.REFUND_PROCESSED,
                occurred_at=time(14),
                received_at=time(14),
                source=DataSource.SIMULATOR,
            ),
            payload=RefundProcessedPayload(
                refund_id=refund,
                payment_id=payment,
                merchant_id=merchant,
                processed_at=time(14),
                processed_amount=Money.of_paise(51),  # Exceeds requested 50
            ),
        )
        
        result = service.ingest(processed)
        assert result.ingestion_outcome == IngestionOutcome.QUARANTINED
        
        with SessionLocal() as session:
            # Verify quarantined event is NOT in pending
            assert session.get(PendingEventModel, processed_id.value) is None
            
            # Verify quarantined event IS in canonical events (current design saves to both)
            canonical = session.get(EventModel, processed_id.value)
            assert canonical is not None
            
            # Verify quarantined event IS in quarantine records
            quarantine = session.query(QuarantineRecordModel).filter_by(event_id=processed_id.value).one()
            assert quarantine.reason_code == ReasonCode.PROCESSED_AMOUNT_EXCEEDS_REQUESTED.value
            
            # Verify quarantine metadata survives a fresh session
        with SessionLocal() as fresh_session:
            fresh_quarantine = fresh_session.query(QuarantineRecordModel).filter_by(event_id=processed_id.value).one()
            assert fresh_quarantine.reason_code == ReasonCode.PROCESSED_AMOUNT_EXCEEDS_REQUESTED.value
            assert fresh_quarantine.detail is not None
            
            # Verify reconstruction includes quarantined events in canonical ledger
            # (they are saved to EventModel but flagged as quarantined in audit)
            reconstruction_service = ReconstructionService(fresh_session)
            snapshot = reconstruction_service.reconstruct()
            
            # The quarantined event IS in the canonical ledger but reconstruction
            # should handle it appropriately based on the state engine
            # For this test, we verify the quarantine metadata is preserved
            assert refund in snapshot.refunds
    finally:
        for event_id in [created_id, captured_id, requested_id, refund_created_id, processed_id]:
            cleanup_event(event_id)


@pytest.mark.integration
def test_sequence_ordinal_uniqueness_across_sessions() -> None:
    """Verify ingestion submission ordinals remain unique across independent sessions.
    
    Verify rollback/failed submissions do not cause duplicate ordinals.
    Sequence gaps are acceptable; uniqueness and durable ordering are the requirements.
    """
    payment, order, merchant, customer = (
        PaymentId.generate(),
        OrderId.generate(),
        MerchantId.generate(),
        CustomerId.generate(),
    )
    
    event1 = payment_created(payment, order, merchant, customer)
    event2 = payment_created(payment, order, merchant, customer)
    event3 = payment_created(payment, order, merchant, customer)
    
    try:
        service = IngestionService()
        
        # Successful ingestion
        result1 = service.ingest(event1)
        ordinal1 = result1.submission_ordinal
        
        # Failed ingestion (rollback)
        def fail_after_event_saved() -> None:
            raise RuntimeError("test-only failure")
        
        with pytest.raises(RuntimeError, match="test-only"):
            IngestionService(after_event_saved=fail_after_event_saved).ingest(event2)
        
        # Another successful ingestion
        result3 = service.ingest(event3)
        ordinal3 = result3.submission_ordinal
        
        with SessionLocal() as session:
            # Verify ordinals are unique
            all_ordinals = [r.submission_ordinal for r in session.query(IngestionRecordModel).all()]
            assert len(all_ordinals) == len(set(all_ordinals)), "Ordinals must be unique"
            
            # Verify ordinals are in durable order
            sorted_ordinals = sorted(all_ordinals)
            assert all_ordinals == sorted_ordinals, "Ordinals should be in submission order"
            
            # Verify there's a gap (the failed submission consumed an ordinal)
            assert ordinal3 > ordinal1 + 1, "Failed submission should have consumed an ordinal, creating a gap"
            
            # Verify the two successful events have different ordinals
            assert ordinal1 != ordinal3
    finally:
        cleanup_event(event1.envelope.event_id)
        cleanup_event(event2.envelope.event_id)
        cleanup_event(event3.envelope.event_id)


@pytest.mark.integration
def test_concurrent_ordinal_reservation() -> None:
    """Verify concurrent submissions reserve unique PostgreSQL sequence ordinals.

    Concurrent completion order is intentionally nondeterministic. The durable
    ordering guarantee is represented by submission_ordinal itself and can be
    recovered explicitly from the database.
    """
    payment, order, merchant, customer = (
        PaymentId.generate(),
        OrderId.generate(),
        MerchantId.generate(),
        CustomerId.generate(),
    )

    event1 = payment_created(payment, order, merchant, customer)
    event2 = payment_created(payment, order, merchant, customer)
    event3 = payment_created(payment, order, merchant, customer)

    events = [event1, event2, event3]

    barrier = Barrier(3)
    ordinals: list[int] = []
    failures: list[BaseException] = []

    def submit(event: PaymentCreatedEvent) -> None:
        try:
            barrier.wait()
            result = IngestionService().ingest(event)
            ordinals.append(result.submission_ordinal)
        except BaseException as exc:
            failures.append(exc)

    threads = [
        Thread(target=submit, args=(event1,)),
        Thread(target=submit, args=(event2,)),
        Thread(target=submit, args=(event3,)),
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    try:
        assert failures == [], f"Unexpected failures: {failures}"

        # PostgreSQL sequence values must be unique across concurrent sessions.
        assert len(ordinals) == 3
        assert len(ordinals) == len(set(ordinals))

        with SessionLocal() as session:
            records = (
                session.query(IngestionRecordModel)
                .filter(
                    IngestionRecordModel.event_id.in_(
                        [event.envelope.event_id.value for event in events]
                    )
                )
                .order_by(IngestionRecordModel.submission_ordinal)
                .all()
            )

        # Every successful submission must have a durable audit record.
        assert len(records) == 3

        persisted_ordinals = [record.submission_ordinal for record in records]

        # Durable ordering is explicitly recoverable from submission_ordinal.
        assert persisted_ordinals == sorted(ordinals)

    finally:
        for event in events:
            cleanup_event(event.envelope.event_id)