from __future__ import annotations

import hashlib
import json

from backend.app.domain.enums import EventType, PaymentStatus, RefundStatus
from backend.app.domain.events import (
    AnyDomainEvent,
    OrderCreatedEvent, OrderDeliveredEvent, OrderShippedEvent,
    PaymentCapturedEvent, PaymentCreatedEvent, PaymentFailedEvent,
    RefundCreatedEvent, RefundFailedEvent, RefundProcessedEvent,
    RefundRequestedEvent,
)
from backend.app.domain.identifiers import EventId, OrderId, PaymentId, RefundId
from backend.app.domain.value_objects import UTCDateTime
from backend.app.finance.aggregates import PaymentState, RefundState, OrderState
from backend.app.finance.processing import ReasonCode
from backend.app.finance.types import (
    IngestionOutcome, IngestionRecord, PendingEvent, PendingReason,
    QuarantinedEvent, ReconstructionAnomaly, ReconstructionSnapshot, RetainedEvent,
)

_PAYMENT_TRANSITIONS = frozenset({
    (PaymentStatus.CREATED, PaymentStatus.CAPTURED),
    (PaymentStatus.CREATED, PaymentStatus.FAILED),
})

_REFUND_TRANSITIONS = frozenset({
    (RefundStatus.REQUESTED, RefundStatus.CREATED),
    (RefundStatus.REQUESTED, RefundStatus.FAILED),
    (RefundStatus.CREATED, RefundStatus.PROCESSED),
    (RefundStatus.CREATED, RefundStatus.FAILED),
})


def payload_hash(event: AnyDomainEvent) -> str:
    """Hash business payload only; delivery envelope is deliberately excluded."""
    payload = event.payload.model_dump(mode="json")
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _sort_key(retained: RetainedEvent):
    envelope = retained.event.envelope
    return (envelope.occurred_at.value, envelope.event_id.value.int)


class FinancialStateEngine:
    """Deterministic event-ingestion and global financial reconstruction engine."""

    def __init__(self) -> None:
        self._canonical_store: list[RetainedEvent] = []
        self._pending_store: list[PendingEvent] = []
        self._quarantine_store: list[QuarantinedEvent] = []
        self._ingestion_registry: dict[EventId, IngestionRecord] = {}
        self._ingestion_log: list[IngestionRecord] = []
        self._current_snapshot = ReconstructionSnapshot({}, {}, {}, 0, 0, ())

    def ingest(self, event: AnyDomainEvent) -> IngestionRecord:
        event_id = event.envelope.event_id
        phash = payload_hash(event)
        existing = self._ingestion_registry.get(event_id)

        if existing is not None:
            outcome = (
                IngestionOutcome.DUPLICATE
                if existing.payload_hash == phash
                else IngestionOutcome.CONFLICT
            )
            record = IngestionRecord(
                event_id=event_id,
                event_type=event.envelope.event_type,
                occurred_at=event.envelope.occurred_at,
                received_at=event.envelope.received_at,
                submission_ordinal=len(self._ingestion_log) + 1,
                payload_hash=phash,
                ingestion_outcome=outcome,
                reason_code=(
                    ReasonCode.DUPLICATE_EVENT_ID
                    if outcome is IngestionOutcome.DUPLICATE
                    else ReasonCode.CONFLICTING_EVENT_ID_PAYLOAD
                ),
                detail=(
                    "Same EventId and payload."
                    if outcome is IngestionOutcome.DUPLICATE
                    else "Same EventId was submitted with a different payload."
                ),
                original_submission_ordinal=existing.submission_ordinal,
            )
            self._ingestion_log.append(record)
            return record

        record = IngestionRecord(
            event_id=event_id,
            event_type=event.envelope.event_type,
            occurred_at=event.envelope.occurred_at,
            received_at=event.envelope.received_at,
            submission_ordinal=len(self._ingestion_log) + 1,
            payload_hash=phash,
            ingestion_outcome=IngestionOutcome.PENDING,
            reason_code=ReasonCode.PENDING_PREREQUISITE,
        )
        self._ingestion_registry[event_id] = record
        self._ingestion_log.append(record)

        decision = self._decide(event)
        outcome, reason, detail = decision

        if outcome is IngestionOutcome.REJECTED:
            record.ingestion_outcome = outcome
            record.reason_code = reason
            record.detail = detail
            return record

        if outcome is IngestionOutcome.QUARANTINED:
            record.ingestion_outcome = outcome
            record.reason_code = reason
            record.detail = detail
            self._quarantine_store.append(
                QuarantinedEvent(event, UTCDateTime.now(), reason, detail)
            )
            return record

        if outcome is IngestionOutcome.PENDING:
            record.ingestion_outcome = outcome
            record.reason_code = reason
            record.detail = detail
            record.pending = True
            self._pending_store.append(
                PendingEvent(event, event.envelope.received_at, PendingReason.PREREQUISITE_NOT_YET_SEEN)
            )
            return record

        self._retain(event, record)
        self._resolve_pending()
        return record

    def ingest_many(self, events: list[AnyDomainEvent]) -> list[IngestionRecord]:
        return [self.ingest(event) for event in events]

    def reconstruct_from(self, events: list[AnyDomainEvent]) -> ReconstructionSnapshot:
        retained = [
            RetainedEvent(event, event.envelope.received_at, index + 1)
            for index, event in enumerate(
                sorted(events, key=lambda e: (e.envelope.occurred_at.value, e.envelope.event_id.value.int))
            )
        ]
        return self._reconstruct(
            retained,
            self._current_snapshot.reconstruction_ordinal + 1,
        )

    def get_payment(self, payment_id: PaymentId) -> PaymentState | None:
        return self._current_snapshot.payments.get(payment_id)

    def get_refund(self, refund_id: RefundId) -> RefundState | None:
        return self._current_snapshot.refunds.get(refund_id)

    def get_order(self, order_id: OrderId) -> OrderState | None:
        return self._current_snapshot.orders.get(order_id)

    def get_snapshot(self) -> ReconstructionSnapshot:
        return self._current_snapshot

    def get_ingestion_log(self) -> list[IngestionRecord]:
        return list(self._ingestion_log)

    def get_ingestion_record(self, event_id: EventId) -> IngestionRecord | None:
        return self._ingestion_registry.get(event_id)

    def get_canonical_store(self) -> list[RetainedEvent]:
        return list(sorted(self._canonical_store, key=_sort_key))

    def get_pending_events(self) -> list[PendingEvent]:
        return list(self._pending_store)

    def get_quarantine(self) -> list[QuarantinedEvent]:
        return list(self._quarantine_store)

    def _retain(self, event: AnyDomainEvent, record: IngestionRecord) -> None:
        retained_at = UTCDateTime.now()
        self._canonical_store.append(
            RetainedEvent(event, retained_at, len(self._canonical_store) + 1)
        )
        self._canonical_store.sort(key=_sort_key)
        record.ingestion_outcome = IngestionOutcome.RETAINED
        record.reason_code = None
        record.detail = None
        record.retained = True
        record.pending = False
        self._reconstruct_current(record)

    def _reconstruct_current(self, trigger_record: IngestionRecord | None = None) -> None:
        self._current_snapshot = self._reconstruct(
            self._canonical_store,
            self._current_snapshot.reconstruction_ordinal + 1,
        )
        if trigger_record is not None:
            trigger_record.triggered_reconstruction = True
            trigger_record.reconstruction_ordinal = self._current_snapshot.reconstruction_ordinal

    def _resolve_pending(self) -> None:
        changed = True
        while changed:
            changed = False
            for pending in list(self._pending_store):
                outcome, reason, detail = self._decide(pending.event)
                if outcome is not IngestionOutcome.RETAINED:
                    continue
                self._pending_store.remove(pending)
                record = self._ingestion_registry[pending.event.envelope.event_id]
                record.ingestion_outcome = IngestionOutcome.RETAINED
                record.reason_code = None
                record.detail = None
                record.pending = False
                record.retained = True
                record.promoted_at = UTCDateTime.now()
                self._canonical_store.append(
                    RetainedEvent(
                        pending.event,
                        record.promoted_at,
                        len(self._canonical_store) + 1,
                    )
                )
                self._canonical_store.sort(key=_sort_key)
                self._reconstruct_current(record)
                changed = True

    def _decide(self, event: AnyDomainEvent):
        p = event.payload
        snapshot = self._current_snapshot

        if isinstance(event, OrderCreatedEvent):
            if p.order_id in snapshot.orders:
                return (IngestionOutcome.REJECTED, ReasonCode.UNKNOWN_ORDER, "Order already exists.")
            return (IngestionOutcome.RETAINED, None, None)

        if isinstance(event, PaymentCreatedEvent):
            if p.payment_id in snapshot.payments:
                return (IngestionOutcome.REJECTED, ReasonCode.UNKNOWN_PAYMENT, "Payment already exists.")
            return (IngestionOutcome.RETAINED, None, None)

        if isinstance(event, PaymentCapturedEvent):
            payment = snapshot.payments.get(p.payment_id)
            if payment is None:
                return (IngestionOutcome.PENDING, ReasonCode.PENDING_PREREQUISITE, "PaymentCreated not yet seen.")
            if p.merchant_id != payment.merchant_id:
                return (IngestionOutcome.REJECTED, ReasonCode.MERCHANT_MISMATCH, "Payment merchant does not match.")
            if (payment.status, PaymentStatus.CAPTURED) not in _PAYMENT_TRANSITIONS:
                return (IngestionOutcome.REJECTED, ReasonCode.ILLEGAL_PAYMENT_TRANSITION, "Payment cannot transition to captured.")
            if p.captured_amount.is_zero():
                return (IngestionOutcome.REJECTED, ReasonCode.CAPTURE_AMOUNT_ZERO_OR_NEGATIVE, "Captured amount must be positive.")
            if p.captured_amount > payment.authorised_amount:
                return (IngestionOutcome.REJECTED, ReasonCode.CAPTURE_AMOUNT_EXCEEDS_AUTHORISED, "Captured amount exceeds authorised amount.")
            return (IngestionOutcome.RETAINED, None, None)

        if isinstance(event, PaymentFailedEvent):
            payment = snapshot.payments.get(p.payment_id)
            if payment is None:
                return (IngestionOutcome.PENDING, ReasonCode.PENDING_PREREQUISITE, "PaymentCreated not yet seen.")
            if p.merchant_id != payment.merchant_id:
                return (IngestionOutcome.REJECTED, ReasonCode.MERCHANT_MISMATCH, "Payment merchant does not match.")
            if (payment.status, PaymentStatus.FAILED) not in _PAYMENT_TRANSITIONS:
                return (IngestionOutcome.REJECTED, ReasonCode.ILLEGAL_PAYMENT_TRANSITION, "Payment cannot transition to failed.")
            return (IngestionOutcome.RETAINED, None, None)

        if isinstance(event, RefundRequestedEvent):
            payment = snapshot.payments.get(p.payment_id)
            if payment is None:
                return (IngestionOutcome.PENDING, ReasonCode.PENDING_PREREQUISITE, "Payment not yet seen.")
            if p.merchant_id != payment.merchant_id:
                return (IngestionOutcome.REJECTED, ReasonCode.MERCHANT_MISMATCH, "Refund merchant does not match payment.")
            if payment.status == PaymentStatus.FAILED:
                return (IngestionOutcome.REJECTED, ReasonCode.REFUND_AGAINST_UNCAPTURED_PAYMENT, "Refund cannot target a failed payment.")
            if payment.status != PaymentStatus.CAPTURED:
                return (IngestionOutcome.PENDING, ReasonCode.PENDING_PREREQUISITE, "Payment is not captured yet.")
            if p.refund_id in snapshot.refunds:
                return (IngestionOutcome.REJECTED, ReasonCode.UNKNOWN_REFUND, "Refund already exists.")
            return (IngestionOutcome.RETAINED, None, None)

        if isinstance(event, (RefundCreatedEvent, RefundProcessedEvent, RefundFailedEvent)):
            refund = snapshot.refunds.get(p.refund_id)
            if refund is None:
                return (IngestionOutcome.PENDING, ReasonCode.PENDING_PREREQUISITE, "RefundRequested not yet seen.")
            if p.merchant_id != refund.merchant_id:
                return (IngestionOutcome.REJECTED, ReasonCode.MERCHANT_MISMATCH, "Refund merchant does not match.")

            if isinstance(event, RefundCreatedEvent):
                if (refund.status, RefundStatus.CREATED) not in _REFUND_TRANSITIONS:
                    return (IngestionOutcome.REJECTED, ReasonCode.ILLEGAL_REFUND_TRANSITION, "Refund cannot transition to created.")
                return (IngestionOutcome.RETAINED, None, None)

            if isinstance(event, RefundFailedEvent):
                if (refund.status, RefundStatus.FAILED) not in _REFUND_TRANSITIONS:
                    return (IngestionOutcome.REJECTED, ReasonCode.ILLEGAL_REFUND_TRANSITION, "Refund cannot transition to failed.")
                return (IngestionOutcome.RETAINED, None, None)

            if (refund.status, RefundStatus.PROCESSED) not in _REFUND_TRANSITIONS:
                return (IngestionOutcome.REJECTED, ReasonCode.ILLEGAL_REFUND_TRANSITION, "Refund cannot transition to processed.")
            if p.processed_amount.is_zero():
                return (IngestionOutcome.REJECTED, ReasonCode.ZERO_REFUND_AMOUNT, "Processed refund amount must be positive.")
            if p.processed_amount > refund.requested_amount:
                return (IngestionOutcome.QUARANTINED, ReasonCode.PROCESSED_AMOUNT_EXCEEDS_REQUESTED, "Processed amount exceeds requested amount.")
            payment = snapshot.payments.get(refund.payment_id)
            if payment is None or payment.captured_amount is None:
                return (IngestionOutcome.PENDING, ReasonCode.PENDING_PREREQUISITE, "Captured payment not yet available.")
            if payment.cumulative_refunded + p.processed_amount > payment.captured_amount:
                return (IngestionOutcome.QUARANTINED, ReasonCode.CUMULATIVE_REFUND_EXCEEDS_CAPTURED, "Cumulative refunds would exceed captured amount.")
            return (IngestionOutcome.RETAINED, None, None)

        if isinstance(event, (OrderShippedEvent, OrderDeliveredEvent)):
            order = snapshot.orders.get(p.order_id)
            if order is None:
                return (IngestionOutcome.PENDING, ReasonCode.PENDING_PREREQUISITE, "OrderCreated not yet seen.")
            if p.merchant_id != order.merchant_id:
                return (IngestionOutcome.REJECTED, ReasonCode.MERCHANT_MISMATCH, "Order merchant does not match.")
            if isinstance(event, OrderShippedEvent) and order.shipped_at is not None:
                return (IngestionOutcome.REJECTED, ReasonCode.ILLEGAL_PAYMENT_TRANSITION, "Order already has a shipped timestamp.")
            if isinstance(event, OrderDeliveredEvent) and order.delivered_at is not None:
                return (IngestionOutcome.REJECTED, ReasonCode.ILLEGAL_PAYMENT_TRANSITION, "Order already has a delivered timestamp.")
            return (IngestionOutcome.RETAINED, None, None)

        raise TypeError(f"Unsupported event type: {type(event).__name__}")

    @staticmethod
    def _reconstruct(
        canonical_store: list[RetainedEvent],
        ordinal: int,
    ) -> ReconstructionSnapshot:
        payments: dict[PaymentId, PaymentState] = {}
        refunds: dict[RefundId, RefundState] = {}
        orders: dict[OrderId, OrderState] = {}
        anomalies: list[ReconstructionAnomaly] = []

        for retained in sorted(canonical_store, key=_sort_key):
            event = retained.event
            p = event.payload
            event_id = event.envelope.event_id

            try:
                if isinstance(event, OrderCreatedEvent):
                    if p.order_id in orders:
                        anomalies.append(ReconstructionAnomaly(event_id, event.envelope.event_type, "Duplicate OrderCreated for the same order_id."))
                        continue
                    orders[p.order_id] = OrderState(
                        p.order_id, p.merchant_id, p.customer_id, p.amount,
                        event.envelope.occurred_at,
                        shipping_address_id=p.shipping_address_id,
                    )

                elif isinstance(event, PaymentCreatedEvent):
                    if p.payment_id in payments:
                        anomalies.append(ReconstructionAnomaly(event_id, event.envelope.event_type, "Duplicate PaymentCreated for the same payment_id."))
                        continue
                    payments[p.payment_id] = PaymentState(
                        p.payment_id, p.merchant_id, p.order_id, p.customer_id,
                        p.amount, PaymentStatus.CREATED, event.envelope.occurred_at,
                    )

                elif isinstance(event, PaymentCapturedEvent):
                    pay = payments.get(p.payment_id)
                    if pay is None:
                        anomalies.append(ReconstructionAnomaly(event_id, event.envelope.event_type, "PaymentCaptured has no PaymentCreated in canonical order."))
                        continue
                    if (pay.status, PaymentStatus.CAPTURED) not in _PAYMENT_TRANSITIONS:
                        anomalies.append(ReconstructionAnomaly(event_id, event.envelope.event_type, "Illegal payment transition during reconstruction."))
                        continue
                    if p.captured_amount.is_zero() or p.captured_amount > pay.authorised_amount:
                        anomalies.append(ReconstructionAnomaly(event_id, event.envelope.event_type, "Invalid captured amount during reconstruction."))
                        continue
                    pay.status = PaymentStatus.CAPTURED
                    pay.captured_amount = p.captured_amount
                    pay.captured_at = p.captured_at
                    pay.applied_event_ids.add(event_id)

                elif isinstance(event, PaymentFailedEvent):
                    pay = payments.get(p.payment_id)
                    if pay is None:
                        anomalies.append(ReconstructionAnomaly(event_id, event.envelope.event_type, "PaymentFailed has no PaymentCreated in canonical order."))
                        continue
                    if (pay.status, PaymentStatus.FAILED) not in _PAYMENT_TRANSITIONS:
                        anomalies.append(ReconstructionAnomaly(event_id, event.envelope.event_type, "Illegal payment transition during reconstruction."))
                        continue
                    pay.status = PaymentStatus.FAILED
                    pay.failed_at = p.failed_at
                    pay.failure_reason = p.failure_reason
                    pay.applied_event_ids.add(event_id)

                elif isinstance(event, RefundRequestedEvent):
                    pay = payments.get(p.payment_id)
                    if pay is None or pay.status != PaymentStatus.CAPTURED:
                        anomalies.append(ReconstructionAnomaly(event_id, event.envelope.event_type, "RefundRequested prerequisite missing during reconstruction."))
                        continue
                    if p.refund_id in refunds:
                        anomalies.append(ReconstructionAnomaly(event_id, event.envelope.event_type, "Duplicate RefundRequested for the same refund_id."))
                        continue
                    refunds[p.refund_id] = RefundState(
                        p.refund_id, p.payment_id, p.merchant_id, p.customer_id,
                        p.order_id, p.amount, RefundStatus.REQUESTED,
                        event.envelope.occurred_at,
                    )

                elif isinstance(event, RefundCreatedEvent):
                    ref = refunds.get(p.refund_id)
                    if ref is None or (ref.status, RefundStatus.CREATED) not in _REFUND_TRANSITIONS:
                        anomalies.append(ReconstructionAnomaly(event_id, event.envelope.event_type, "Illegal refund created transition during reconstruction."))
                        continue
                    ref.status = RefundStatus.CREATED
                    ref.created_at = p.created_at
                    ref.applied_event_ids.add(event_id)

                elif isinstance(event, RefundProcessedEvent):
                    ref = refunds.get(p.refund_id)
                    if ref is None or (ref.status, RefundStatus.PROCESSED) not in _REFUND_TRANSITIONS:
                        anomalies.append(ReconstructionAnomaly(event_id, event.envelope.event_type, "Illegal refund processed transition during reconstruction."))
                        continue
                    pay = payments.get(ref.payment_id)
                    if pay is None or pay.captured_amount is None:
                        anomalies.append(ReconstructionAnomaly(event_id, event.envelope.event_type, "Processed refund has no captured payment."))
                        continue
                    if p.processed_amount > ref.requested_amount:
                        anomalies.append(ReconstructionAnomaly(event_id, event.envelope.event_type, "Processed amount exceeds requested amount."))
                        continue
                    if pay.cumulative_refunded + p.processed_amount > pay.captured_amount:
                        anomalies.append(ReconstructionAnomaly(event_id, event.envelope.event_type, "Cumulative refund exceeds captured amount."))
                        continue
                    ref.status = RefundStatus.PROCESSED
                    ref.processed_at = p.processed_at
                    ref.processed_amount = p.processed_amount
                    pay.cumulative_refunded = pay.cumulative_refunded + p.processed_amount
                    ref.applied_event_ids.add(event_id)

                elif isinstance(event, RefundFailedEvent):
                    ref = refunds.get(p.refund_id)
                    if ref is None or (ref.status, RefundStatus.FAILED) not in _REFUND_TRANSITIONS:
                        anomalies.append(ReconstructionAnomaly(event_id, event.envelope.event_type, "Illegal refund failed transition during reconstruction."))
                        continue
                    ref.status = RefundStatus.FAILED
                    ref.failed_at = p.failed_at
                    ref.failure_reason = p.failure_reason
                    ref.applied_event_ids.add(event_id)

                elif isinstance(event, OrderShippedEvent):
                    order = orders.get(p.order_id)
                    if order is None:
                        anomalies.append(ReconstructionAnomaly(event_id, event.envelope.event_type, "OrderShipped has no OrderCreated."))
                        continue
                    order.shipped_at = p.shipped_at
                    order.applied_event_ids.add(event_id)

                elif isinstance(event, OrderDeliveredEvent):
                    order = orders.get(p.order_id)
                    if order is None:
                        anomalies.append(ReconstructionAnomaly(event_id, event.envelope.event_type, "OrderDelivered has no OrderCreated."))
                        continue
                    order.delivered_at = p.delivered_at
                    order.applied_event_ids.add(event_id)

            except (TypeError, ValueError) as exc:
                anomalies.append(
                    ReconstructionAnomaly(event_id, event.envelope.event_type, str(exc))
                )

        return ReconstructionSnapshot(
            payments, refunds, orders, ordinal, len(canonical_store), tuple(anomalies)
        )
