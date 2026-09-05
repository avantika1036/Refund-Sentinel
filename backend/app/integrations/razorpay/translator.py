"""Translate Razorpay webhook payloads into deterministic domain events."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any

from backend.app.domain.enums import DataSource, EventType
from backend.app.domain.events import (
    AnyDomainEvent,
    EventEnvelope,
    PaymentCapturedEvent,
    PaymentCapturedPayload,
    RefundCreatedEvent,
    RefundCreatedPayload,
    RefundFailedEvent,
    RefundFailedPayload,
    RefundProcessedEvent,
    RefundProcessedPayload,
)
from backend.app.domain.identifiers import EventId, MerchantId, PaymentId, RefundId
from backend.app.domain.value_objects import Money, UTCDateTime


class RazorpayTranslationError(ValueError):
    """Raised when a Razorpay payload cannot be translated safely."""


_EVENT_NAMESPACE = uuid.UUID("a52b6e6f-6f6d-5cf9-9af5-6f0cf9e72f9d")


def _stable_uuid(raw: str, prefix: str) -> uuid.UUID:
    if not raw:
        raise RazorpayTranslationError(f"Missing required {prefix} identifier")
    try:
        return uuid.UUID(raw)
    except ValueError:
        return uuid.uuid5(uuid.NAMESPACE_URL, f"refund-sentinel:{prefix}:{raw}")


def _canonical_event_id(payload: dict[str, Any], *, event_name: str, entity_id: str) -> EventId:
    """Create one stable internal EventId for one logical Razorpay delivery.

    Razorpay retries the same logical event with the same payload/entity data.
    UUID5 makes those retries resolve to the same ledger key, while a materially
    different payload produces a different key and is therefore not silently
    deduplicated.
    """
    provider_event_id = str(payload.get("event_id") or payload.get("id") or "")
    account_id = str(payload.get("account_id") or "")
    canonical_payload = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    payload_hash = hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()
    identity = "|".join(
        ["razorpay", event_name, account_id, provider_event_id, entity_id, payload_hash]
    )
    return EventId(uuid.uuid5(_EVENT_NAMESPACE, identity))


def _entity_for_event(payload: dict[str, Any], event_name: str) -> dict[str, Any]:
    event_payload = payload.get("payload")
    if not isinstance(event_payload, dict):
        raise RazorpayTranslationError("Missing payload object")

    key = "payment" if event_name == "payment.captured" else "refund"
    container = event_payload.get(key)
    if not isinstance(container, dict):
        raise RazorpayTranslationError(f"Missing payload.{key} object")
    entity = container.get("entity")
    if not isinstance(entity, dict):
        raise RazorpayTranslationError(f"Missing payload.{key}.entity object")
    return entity


class RazorpayWebhookTranslator:
    """Translate supported Razorpay webhook events into domain events."""

    @classmethod
    def translate(
        cls,
        payload: dict[str, Any],
        received_at: datetime | None = None,
    ) -> AnyDomainEvent:
        event_name = payload.get("event")
        if not isinstance(event_name, str) or not event_name:
            raise RazorpayTranslationError("Missing 'event' field in webhook payload")

        if event_name not in {
            "payment.captured",
            "refund.created",
            "refund.processed",
            "refund.failed",
        }:
            raise RazorpayTranslationError(f"Unsupported Razorpay webhook event: {event_name}")

        entity = _entity_for_event(payload, event_name)
        entity_id = str(entity.get("id") or "")
        if not entity_id:
            raise RazorpayTranslationError("Webhook entity has no stable id")

        now = received_at or datetime.now(timezone.utc)
        created_at_epoch = entity.get("created_at")
        if created_at_epoch is not None:
            try:
                occurred_at = datetime.fromtimestamp(float(created_at_epoch), tz=timezone.utc)
            except (TypeError, ValueError, OSError) as exc:
                raise RazorpayTranslationError("Invalid entity created_at timestamp") from exc
        else:
            occurred_at = now

        account_id = str(payload.get("account_id") or "default")
        event_id = _canonical_event_id(
            payload,
            event_name=event_name,
            entity_id=entity_id,
        )
        merchant_id = MerchantId(_stable_uuid(account_id, "merchant"))
        envelope = dict(
            event_id=event_id,
            occurred_at=UTCDateTime(value=occurred_at),
            received_at=UTCDateTime(value=now),
            source=DataSource.RAZORPAY_WEBHOOK,
        )

        if event_name == "payment.captured":
            amount_paise = int(entity.get("amount") or 0)
            if amount_paise < 0:
                raise RazorpayTranslationError("Payment amount cannot be negative")
            return PaymentCapturedEvent(
                envelope=EventEnvelope(event_type=EventType.PAYMENT_CAPTURED, **envelope),
                payload=PaymentCapturedPayload(
                    payment_id=PaymentId(_stable_uuid(str(entity_id), "payment")),
                    merchant_id=merchant_id,
                    captured_amount=Money.of_paise(amount_paise),
                    captured_at=UTCDateTime(value=occurred_at),
                ),
            )

        refund_id = RefundId(_stable_uuid(entity_id, "refund"))
        payment_raw = str(entity.get("payment_id") or "")
        payment_id = PaymentId(_stable_uuid(payment_raw, "payment"))

        if event_name == "refund.created":
            return RefundCreatedEvent(
                envelope=EventEnvelope(event_type=EventType.REFUND_CREATED, **envelope),
                payload=RefundCreatedPayload(
                    refund_id=refund_id,
                    payment_id=payment_id,
                    merchant_id=merchant_id,
                    created_at=UTCDateTime(value=occurred_at),
                ),
            )

        if event_name == "refund.processed":
            amount_paise = int(entity.get("amount") or 0)
            if amount_paise < 0:
                raise RazorpayTranslationError("Refund amount cannot be negative")
            return RefundProcessedEvent(
                envelope=EventEnvelope(event_type=EventType.REFUND_PROCESSED, **envelope),
                payload=RefundProcessedPayload(
                    refund_id=refund_id,
                    payment_id=payment_id,
                    merchant_id=merchant_id,
                    processed_at=UTCDateTime(value=occurred_at),
                    processed_amount=Money.of_paise(amount_paise),
                ),
            )

        return RefundFailedEvent(
            envelope=EventEnvelope(event_type=EventType.REFUND_FAILED, **envelope),
            payload=RefundFailedPayload(
                refund_id=refund_id,
                payment_id=payment_id,
                merchant_id=merchant_id,
                failed_at=UTCDateTime(value=occurred_at),
                failure_reason=str(entity.get("error_description") or "Unknown failure"),
            ),
        )
