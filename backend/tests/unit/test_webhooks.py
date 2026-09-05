"""Unit tests for Razorpay webhook translation and verification."""

from __future__ import annotations

import hmac
import hashlib
import json
import pytest

from backend.app.domain.enums import EventType
from backend.app.integrations.razorpay.security import (
    RazorpaySignatureVerificationError,
    verify_razorpay_signature,
)
from backend.app.integrations.razorpay.translator import (
    RazorpayTranslationError,
    RazorpayWebhookTranslator,
)


def test_verify_razorpay_signature():
    """Verify HMAC signature checking works accurately."""
    secret = "test_webhook_secret_key"
    payload = b'{"event":"payment.captured"}'

    valid_signature = hmac.new(
        key=secret.encode("utf-8"),
        msg=payload,
        digestmod=hashlib.sha256,
    ).hexdigest()

    # Valid signature
    assert verify_razorpay_signature(payload, valid_signature, secret) is True

    # Invalid signature
    with pytest.raises(RazorpaySignatureVerificationError):
        verify_razorpay_signature(payload, "invalid_signature_hex", secret)


def test_translate_payment_captured():
    """Verify Razorpay payment.captured translates to PaymentCapturedEvent."""
    payload = {
        "event": "payment.captured",
        "account_id": "acc_test123",
        "payload": {
            "payment": {
                "entity": {
                    "id": "pay_xyz123",
                    "order_id": "order_abc456",
                    "customer_id": "cust_user99",
                    "amount": 50000,
                    "currency": "INR",
                    "created_at": 1700000000,
                }
            }
        }
    }

    event = RazorpayWebhookTranslator.translate(payload)
    assert event.envelope.event_type == EventType.PAYMENT_CAPTURED
    assert event.payload.captured_amount.amount_paise == 50000


def test_translate_refund_created():
    """Verify Razorpay refund.created translates to RefundCreatedEvent."""
    payload = {
        "event": "refund.created",
        "account_id": "acc_test123",
        "payload": {
            "refund": {
                "entity": {
                    "id": "rfnd_test999",
                    "payment_id": "pay_xyz123",
                    "amount": 25000,
                    "created_at": 1700000050,
                }
            }
        }
    }

    event = RazorpayWebhookTranslator.translate(payload)
    assert event.envelope.event_type == EventType.REFUND_CREATED
    assert event.payload.refund_id is not None


def test_translation_is_idempotent_for_same_payload():
    """The same Razorpay retry must map to the same internal event ID."""
    payload = {
        "event": "refund.created",
        "account_id": "acc_test123",
        "payload": {
            "refund": {
                "entity": {
                    "id": "rfnd_retry_1",
                    "payment_id": "pay_retry_1",
                    "amount": 25000,
                    "created_at": 1700000050,
                }
            }
        },
    }

    first = RazorpayWebhookTranslator.translate(payload)
    second = RazorpayWebhookTranslator.translate(payload)

    assert first.envelope.event_id == second.envelope.event_id


def test_translation_rejects_missing_stable_entity_id():
    """Idempotency cannot be guaranteed without a provider entity ID."""
    payload = {
        "event": "refund.created",
        "payload": {
            "refund": {
                "entity": {
                    "payment_id": "pay_retry_1",
                    "created_at": 1700000050,
                }
            }
        },
    }

    with pytest.raises(RazorpayTranslationError):
        RazorpayWebhookTranslator.translate(payload)
