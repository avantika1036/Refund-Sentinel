from __future__ import annotations

from datetime import datetime, timezone

from backend.app.domain.enums import DataSource, EventType, PaymentStatus, RefundReasonCode, RefundStatus
from backend.app.domain.events import (
    EventEnvelope, PaymentCreatedEvent, PaymentCreatedPayload,
    PaymentCapturedEvent, PaymentCapturedPayload,
    RefundRequestedEvent, RefundRequestedPayload,
    RefundCreatedEvent, RefundCreatedPayload,
    RefundProcessedEvent, RefundProcessedPayload,
)
from backend.app.domain.identifiers import CustomerId, EventId, MerchantId, OrderId, PaymentId, RefundId
from backend.app.domain.value_objects import Money, UTCDateTime
from backend.app.finance.state_engine import FinancialStateEngine, payload_hash
from backend.app.finance.types import IngestionOutcome


def ts(hour: int, minute: int = 0) -> UTCDateTime:
    return UTCDateTime(value=datetime(2024, 6, 1, hour, minute, tzinfo=timezone.utc))


def envelope(event_type: EventType, occurred_at: UTCDateTime, event_id: EventId | None = None) -> EventEnvelope:
    return EventEnvelope(
        event_id=event_id or EventId.generate(),
        event_type=event_type,
        occurred_at=occurred_at,
        received_at=occurred_at,
        source=DataSource.SIMULATOR,
    )


def payment_created(pid: PaymentId, oid: OrderId, mid: MerchantId, cid: CustomerId, *, eid=None):
    return PaymentCreatedEvent(
        envelope=envelope(EventType.PAYMENT_CREATED, ts(10), eid),
        payload=PaymentCreatedPayload(
            payment_id=pid, order_id=oid, merchant_id=mid, customer_id=cid,
            amount=Money.of_paise(100_000),
        ),
    )


def payment_captured(pid: PaymentId, mid: MerchantId, *, eid=None, hour=11):
    return PaymentCapturedEvent(
        envelope=envelope(EventType.PAYMENT_CAPTURED, ts(hour), eid),
        payload=PaymentCapturedPayload(
            payment_id=pid, merchant_id=mid,
            captured_amount=Money.of_paise(100_000), captured_at=ts(hour),
        ),
    )


def refund_requested(rid, pid, oid, mid, cid, *, eid=None, hour=12, amount=50_000):
    return RefundRequestedEvent(
        envelope=envelope(EventType.REFUND_REQUESTED, ts(hour), eid),
        payload=RefundRequestedPayload(
            refund_id=rid, payment_id=pid, order_id=oid, merchant_id=mid,
            customer_id=cid, amount=Money.of_paise(amount),
            reason_code=RefundReasonCode.DEFECTIVE,
        ),
    )


def refund_created(rid, pid, mid, *, eid=None, hour=13):
    return RefundCreatedEvent(
        envelope=envelope(EventType.REFUND_CREATED, ts(hour), eid),
        payload=RefundCreatedPayload(
            refund_id=rid, payment_id=pid, merchant_id=mid, created_at=ts(hour),
        ),
    )


def refund_processed(rid, pid, mid, amount=50_000, *, eid=None, hour=14):
    return RefundProcessedEvent(
        envelope=envelope(EventType.REFUND_PROCESSED, ts(hour), eid),
        payload=RefundProcessedPayload(
            refund_id=rid, payment_id=pid, merchant_id=mid,
            processed_at=ts(hour), processed_amount=Money.of_paise(amount),
        ),
    )


def test_payment_captured_before_payment_created_is_pending_then_promoted():
    pid, oid, mid, cid = PaymentId.generate(), OrderId.generate(), MerchantId.generate(), CustomerId.generate()
    engine = FinancialStateEngine()
    first = engine.ingest(payment_captured(pid, mid))
    assert first.ingestion_outcome is IngestionOutcome.PENDING
    assert engine.get_payment(pid) is None

    second = engine.ingest(payment_created(pid, oid, mid, cid))
    assert second.ingestion_outcome is IngestionOutcome.RETAINED
    assert engine.get_payment(pid) is not None
    assert engine.get_payment(pid).status is PaymentStatus.CAPTURED
    assert len(engine.get_ingestion_log()) == 2
    assert engine.get_ingestion_record(first.event_id).retained is True
    assert engine.get_ingestion_record(first.event_id).pending is False


def test_same_event_id_same_payload_is_duplicate():
    pid, oid, mid, cid = PaymentId.generate(), OrderId.generate(), MerchantId.generate(), CustomerId.generate()
    event = payment_created(pid, oid, mid, cid)
    engine = FinancialStateEngine()
    first = engine.ingest(event)
    equivalent_submission = PaymentCreatedEvent(
        envelope=event.envelope,
        payload=event.payload,
    )
    second = engine.ingest(equivalent_submission)
    assert first.ingestion_outcome is IngestionOutcome.RETAINED
    assert second.ingestion_outcome is IngestionOutcome.DUPLICATE
    assert second.original_submission_ordinal == first.submission_ordinal
    assert engine.get_snapshot().reconstruction_ordinal == 1
    assert len(engine.get_ingestion_log()) == 2


def test_same_event_id_different_payload_is_conflict():
    pid, oid, mid, cid = PaymentId.generate(), OrderId.generate(), MerchantId.generate(), CustomerId.generate()
    event_id = EventId.generate()
    first = payment_created(pid, oid, mid, cid, eid=event_id)
    conflicting = PaymentCreatedEvent(
        envelope=envelope(EventType.PAYMENT_CREATED, ts(10), event_id),
        payload=PaymentCreatedPayload(
            payment_id=pid, order_id=oid, merchant_id=mid, customer_id=cid,
            amount=Money.of_paise(90_000),
        ),
    )
    engine = FinancialStateEngine()
    engine.ingest(first)
    result = engine.ingest(conflicting)
    assert result.ingestion_outcome is IngestionOutcome.CONFLICT
    assert engine.get_payment(pid).authorised_amount.amount_paise == 100_000
    assert engine.get_snapshot().reconstruction_ordinal == 1


def test_partial_refund_uses_processed_amount_for_cumulative():
    pid, oid, mid, cid, rid = PaymentId.generate(), OrderId.generate(), MerchantId.generate(), CustomerId.generate(), RefundId.generate()
    engine = FinancialStateEngine()
    engine.ingest_many([
        payment_created(pid, oid, mid, cid),
        payment_captured(pid, mid),
        refund_requested(rid, pid, oid, mid, cid, amount=50_000),
        refund_created(rid, pid, mid),
        refund_processed(rid, pid, mid, amount=45_000),
    ])
    payment = engine.get_payment(pid)
    refund = engine.get_refund(rid)
    assert refund.status is RefundStatus.PROCESSED
    assert refund.processed_amount.amount_paise == 45_000
    assert payment.cumulative_refunded.amount_paise == 45_000
    assert payment.remaining_refundable.amount_paise == 55_000


def test_exact_refund_boundary_is_retained():
    pid, oid, mid, cid = PaymentId.generate(), OrderId.generate(), MerchantId.generate(), CustomerId.generate()
    r1, r2 = RefundId.generate(), RefundId.generate()
    engine = FinancialStateEngine()
    engine.ingest_many([
        payment_created(pid, oid, mid, cid), payment_captured(pid, mid),
        refund_requested(r1, pid, oid, mid, cid, amount=60_000), refund_created(r1, pid, mid), refund_processed(r1, pid, mid, 60_000),
        refund_requested(r2, pid, oid, mid, cid, amount=40_000), refund_created(r2, pid, mid), refund_processed(r2, pid, mid, 40_000),
    ])
    assert engine.get_payment(pid).cumulative_refunded.amount_paise == 100_000
    assert engine.get_payment(pid).remaining_refundable.amount_paise == 0


def test_one_paise_over_capture_is_quarantined():
    pid, oid, mid, cid = PaymentId.generate(), OrderId.generate(), MerchantId.generate(), CustomerId.generate()
    r1, r2 = RefundId.generate(), RefundId.generate()
    engine = FinancialStateEngine()
    engine.ingest_many([
        payment_created(pid, oid, mid, cid), payment_captured(pid, mid),
        refund_requested(r1, pid, oid, mid, cid, amount=60_000), refund_created(r1, pid, mid), refund_processed(r1, pid, mid, 60_000),
        refund_requested(r2, pid, oid, mid, cid, amount=40_001), refund_created(r2, pid, mid),
    ])
    result = engine.ingest(refund_processed(r2, pid, mid, 40_001))
    assert result.ingestion_outcome is IngestionOutcome.QUARANTINED
    assert engine.get_payment(pid).cumulative_refunded.amount_paise == 60_000
    assert len(engine.get_quarantine()) == 1


def test_reconstruction_is_pure_and_deterministic_for_same_event_set():
    pid, oid, mid, cid = PaymentId.generate(), OrderId.generate(), MerchantId.generate(), CustomerId.generate()
    events = [payment_created(pid, oid, mid, cid), payment_captured(pid, mid)]
    engine = FinancialStateEngine()
    a = engine.reconstruct_from(events)
    b = engine.reconstruct_from(list(reversed(events)))
    assert a.payments[pid].status is PaymentStatus.CAPTURED
    assert b.payments[pid].status is PaymentStatus.CAPTURED
    assert a.payments[pid].authorised_amount == b.payments[pid].authorised_amount
    assert a.event_count == b.event_count == 2


def test_payload_hash_excludes_delivery_envelope():
    pid, oid, mid, cid = PaymentId.generate(), OrderId.generate(), MerchantId.generate(), CustomerId.generate()
    eid = EventId.generate()
    a = payment_created(pid, oid, mid, cid, eid=eid)
    # Same payload and EventId, different received_at is still the same business fact.
    b = PaymentCreatedEvent(
        envelope=EventEnvelope(
            event_id=eid, event_type=EventType.PAYMENT_CREATED,
            occurred_at=ts(10), received_at=ts(10, 5), source=DataSource.SIMULATOR,
        ),
        payload=a.payload,
    )
    assert payload_hash(a) == payload_hash(b)
