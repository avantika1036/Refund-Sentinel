from __future__ import annotations

from enum import Enum


class ReasonCode(str, Enum):
    DUPLICATE_EVENT_ID = "duplicate_event_id"
    CONFLICTING_EVENT_ID_PAYLOAD = "conflicting_event_id_payload"
    UNKNOWN_PAYMENT = "unknown_payment"
    UNKNOWN_REFUND = "unknown_refund"
    UNKNOWN_ORDER = "unknown_order"
    ILLEGAL_PAYMENT_TRANSITION = "illegal_payment_transition"
    ILLEGAL_REFUND_TRANSITION = "illegal_refund_transition"
    REFUND_AGAINST_UNCAPTURED_PAYMENT = "refund_against_uncaptured_payment"
    MERCHANT_MISMATCH = "merchant_mismatch"
    PAYMENT_ALREADY_CAPTURED = "payment_already_captured"
    CAPTURE_AMOUNT_EXCEEDS_AUTHORISED = "capture_amount_exceeds_authorised"
    CAPTURE_AMOUNT_ZERO_OR_NEGATIVE = "capture_amount_zero_or_negative"
    ZERO_REFUND_AMOUNT = "zero_refund_amount"
    CUMULATIVE_REFUND_EXCEEDS_CAPTURED = "cumulative_refund_exceeds_captured"
    PROCESSED_AMOUNT_EXCEEDS_REQUESTED = "processed_amount_exceeds_requested"
    RECONSTRUCTION_ANOMALY = "reconstruction_anomaly"
    PENDING_PREREQUISITE = "pending_prerequisite"
