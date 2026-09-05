"""Razorpay webhook HMAC signature verification."""

from __future__ import annotations

import hmac
import hashlib


class RazorpaySignatureVerificationError(ValueError):
    """Raised when the webhook signature does not match the secret."""


def verify_razorpay_signature(
    payload_body: bytes,
    signature: str,
    secret: str,
) -> bool:
    """Verify Razorpay HMAC-SHA256 signature.

    Args:
        payload_body: Raw bytes of the request body.
        signature: Value from 'X-Razorpay-Signature' header.
        secret: Webhook secret configured in Razorpay dashboard.

    Returns:
        True if signature is valid.

    Raises:
        RazorpaySignatureVerificationError: If signatures do not match.
    """
    if not secret:
        raise RazorpaySignatureVerificationError("Razorpay webhook secret is not configured")

    expected_signature = hmac.new(
        key=secret.encode("utf-8"),
        msg=payload_body,
        digestmod=hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(expected_signature, signature):
        raise RazorpaySignatureVerificationError("Invalid Razorpay webhook signature")

    return True
