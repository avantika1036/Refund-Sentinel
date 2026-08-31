from __future__ import annotations

from datetime import datetime, timezone

from backend.app.domain.enums import DataSource, EventType, PaymentStatus, RefundReasonCode, RefundStatus
from backend.app.domain.events import (
    EventEnvelope,
    PaymentCapturedEvent, PaymentCapturedPayload,
    PaymentCreatedEvent, PaymentCreatedPayload,
    PaymentFailedEvent, PaymentFailedPayload,
    RefundCreatedEvent, RefundCreatedPayload,
    RefundProcessedEvent, RefundProcessedPayload,
    RefundRequestedEvent, RefundRequestedPayload,
)
from backend.app.domain.identifiers import CustomerId, EventId, MerchantId, OrderId, PaymentId, RefundId
from backend.app.domain.value_objects import Money, UTCDateTime
from backend.app.finance.state_engine import FinancialStateEngine
from backend.app.finance.types import IngestionOutcome


def ts(hour: int, minute: int = 0) -> UTCDateTime:
    return UTCDateTime(value=datetime(2024, 6, 1, hour, minute, tzinfo=timezone.utc))


def env(event_type: EventType, hour: int, event_id: EventId | None = None) -> EventEnvelope:
    return EventEnvelope(
        event_id=event_id or EventId.generate(),
        event_type=event_type,
        occurred_at=ts(hour),
        received_at=ts(hour, 30),
        source=DataSource.SIMULATOR,
    )


def payment_created(pid, oid, mid, cid, *, hour=10, eid=None, amount=100_000):
    return PaymentCreatedEvent(
        envelope=env(EventType.PAYMENT_CREATED, hour, eid),
        payload=PaymentCreatedPayload(
            payment_id=pid, order_id=oid, merchant_id=mid,
            customer_id=cid, amount=Money.of_paise(amount),
        ),
    )


def payment_captured(pid, mid, *, hour=11, eid=None, amount=100_000):
    return PaymentCapturedEvent(
        envelope=env(EventType.PAYMENT_CAPTURED, hour, eid),
        payload=PaymentCapturedPayload(
            payment_id=pid, merchant_id=mid,
            captured_amount=Money.of_paise(amount), captured_at=ts(hour),
        ),
    )


def refund_requested(rid, pid, oid, mid, cid, *, hour=12, eid=None, amount=50_000):
    return RefundRequestedEvent(
        envelope=env(EventType.REFUND_REQUESTED, hour, eid),
        payload=RefundRequestedPayload(
            refund_id=rid, payment_id=pid, order_id=oid,
            merchant_id=mid, customer_id=cid,
            amount=Money.of_paise(amount),
            reason_code=RefundReasonCode.DEFECTIVE,
        ),
    )


def refund_created(rid, pid, mid, *, hour=13, eid=None):
    return RefundCreatedEvent(
        envelope=env(EventType.REFUND_CREATED, hour, eid),
        payload=RefundCreatedPayload(
            refund_id=rid, payment_id=pid, merchant_id=mid,
            created_at=ts(hour),
        ),
    )


def refund_processed(rid, pid, mid, *, hour=14, eid=None, amount=50_000):
    return RefundProcessedEvent(
        envelope=env(EventType.REFUND_PROCESSED, hour, eid),
        payload=RefundProcessedPayload(
            refund_id=rid, payment_id=pid, merchant_id=mid,
            processed_at=ts(hour),
            processed_amount=Money.of_paise(amount),
        ),
    )


def identifiers():
    return (
        PaymentId.generate(), OrderId.generate(),
        MerchantId.generate(), CustomerId.generate(),
    )


def test_full_reverse_arrival_order_eventually_reaches_processed():
    pid, oid, mid, cid = identifiers()
    rid = RefundId.generate()

    events = [
        refund_processed(rid, pid, mid),
        refund_created(rid, pid, mid),
        refund_requested(rid, pid, oid, mid, cid),
        payment_captured(pid, mid),
        payment_created(pid, oid, mid, cid),
    ]

    engine = FinancialStateEngine()

    # Each event initially arrives before its prerequisite.
    for expected_pending_count, event in enumerate(events[:4], start=1):
        record = engine.ingest(event)

        assert record.ingestion_outcome is IngestionOutcome.PENDING
        assert record.pending is True
        assert len(engine.get_pending_events()) == expected_pending_count

    # The final prerequisite arrives and should trigger fixed-point
    # promotion of the entire dependency chain.
    final_record = engine.ingest(events[4])

    assert final_record.ingestion_outcome is IngestionOutcome.RETAINED
    assert final_record.retained is True
    assert final_record.triggered_reconstruction is True

    payment = engine.get_payment(pid)
    refund = engine.get_refund(rid)

    assert payment is not None
    assert payment.status is PaymentStatus.CAPTURED
    assert payment.cumulative_refunded.amount_paise == 50_000

    assert refund is not None
    assert refund.status is RefundStatus.PROCESSED

    assert len(engine.get_pending_events()) == 0

    # Five submissions must remain five audit records.
    audit = engine.get_ingestion_log()
    assert len(audit) == 5

    # The four previously-pending events were promoted rather than
    # creating duplicate submission records.
    for entry in audit[:4]:
        assert entry.retained is True
        assert entry.pending is False
        assert entry.promoted_at is not None
        assert entry.triggered_reconstruction is True


def test_late_prerequisite_reconstruction_does_not_erase_existing_refund():
    pid, oid, mid, cid = identifiers()
    rid = RefundId.generate()
    events = [
        payment_created(pid, oid, mid, cid),
        payment_captured(pid, mid),
        refund_requested(rid, pid, oid, mid, cid),
        refund_created(rid, pid, mid),
        refund_processed(rid, pid, mid),
    ]

    engine = FinancialStateEngine()
    engine.ingest_many(events)

    before = engine.get_payment(pid)
    assert before is not None
    assert before.cumulative_refunded.amount_paise == 50_000

    snapshot = engine.get_snapshot()
    replay = engine.get_snapshot()

    assert snapshot.payments[pid].cumulative_refunded.amount_paise == 50_000
    assert snapshot.refunds[rid].status is RefundStatus.PROCESSED
    assert replay.payments[pid].cumulative_refunded.amount_paise == 50_000


def test_duplicate_pending_event_does_not_create_two_pending_entries():
    pid, oid, mid, cid = identifiers()
    event = payment_captured(pid, mid)
    engine = FinancialStateEngine()

    first = engine.ingest(event)
    second = engine.ingest(event)

    assert first.ingestion_outcome is IngestionOutcome.PENDING
    assert second.ingestion_outcome is IngestionOutcome.DUPLICATE
    assert len(engine.get_pending_events()) == 1
    assert len(engine.get_ingestion_log()) == 2


def test_conflicting_pending_event_does_not_replace_original():
    pid, oid, mid, cid = identifiers()
    eid = EventId.generate()

    first = payment_captured(pid, mid, eid=eid, amount=100_000)
    conflict = payment_captured(pid, mid, eid=eid, amount=99_999)

    engine = FinancialStateEngine()
    first_record = engine.ingest(first)
    second_record = engine.ingest(conflict)

    assert first_record.ingestion_outcome is IngestionOutcome.PENDING
    assert second_record.ingestion_outcome is IngestionOutcome.CONFLICT
    assert len(engine.get_pending_events()) == 1
    assert engine.get_pending_events()[0].event.payload.captured_amount.amount_paise == 100_000


def test_refund_requested_against_failed_payment_is_rejected():
    pid, oid, mid, cid = identifiers()
    rid = RefundId.generate()

    failed = PaymentFailedEvent(
        envelope=env(EventType.PAYMENT_FAILED, 11),
        payload=PaymentFailedPayload(
            payment_id=pid, merchant_id=mid,
            failed_at=ts(11), failure_reason="issuer_declined",
        ),
    )

    engine = FinancialStateEngine()
    engine.ingest(payment_created(pid, oid, mid, cid))
    engine.ingest(failed)

    result = engine.ingest(refund_requested(rid, pid, oid, mid, cid))

    assert result.ingestion_outcome is IngestionOutcome.REJECTED
    assert engine.get_refund(rid) is None


def test_refund_processed_cannot_exceed_requested_amount():
    pid, oid, mid, cid = identifiers()
    rid = RefundId.generate()
    engine = FinancialStateEngine()

    engine.ingest_many([
        payment_created(pid, oid, mid, cid),
        payment_captured(pid, mid),
        refund_requested(rid, pid, oid, mid, cid, amount=50_000),
        refund_created(rid, pid, mid),
    ])

    result = engine.ingest(refund_processed(rid, pid, mid, amount=50_001))

    assert result.ingestion_outcome is IngestionOutcome.QUARANTINED
    assert engine.get_refund(rid).status is RefundStatus.CREATED
    assert engine.get_payment(pid).cumulative_refunded.amount_paise == 0
    assert len(engine.get_quarantine()) == 1


def test_two_different_events_with_same_occurred_at_have_deterministic_order():
    pid, oid, mid, cid = identifiers()
    e1_id, e2_id = EventId.generate(), EventId.generate()

    e1 = payment_created(pid, oid, mid, cid, hour=10, eid=e1_id)
    e2 = payment_captured(pid, mid, hour=10, eid=e2_id)

    engine_a = FinancialStateEngine()
    engine_b = FinancialStateEngine()

    snapshot_a = engine_a.reconstruct_from([e1, e2])
    snapshot_b = engine_b.reconstruct_from([e2, e1])

    assert snapshot_a.payments[pid].status is snapshot_b.payments[pid].status
    assert snapshot_a.payments[pid].captured_amount == snapshot_b.payments[pid].captured_amount


def test_reconstruction_does_not_mutate_input_event_list():
    pid, oid, mid, cid = identifiers()
    events = [
        payment_captured(pid, mid),
        payment_created(pid, oid, mid, cid),
    ]
    original_ids = [e.envelope.event_id for e in events]

    FinancialStateEngine().reconstruct_from(events)

    assert [e.envelope.event_id for e in events] == original_ids


def test_every_submission_has_exactly_one_audit_entry():
    pid, oid, mid, cid = identifiers()
    e1 = payment_created(pid, oid, mid, cid)
    e2 = payment_captured(pid, mid)
    e3 = payment_captured(pid, mid)

    engine = FinancialStateEngine()
    engine.ingest(e1)
    engine.ingest(e2)
    engine.ingest(e2)
    engine.ingest(e3)

    audit = engine.get_ingestion_log()
    assert len(audit) == 4
    assert len({entry.submission_ordinal for entry in audit}) == 4
