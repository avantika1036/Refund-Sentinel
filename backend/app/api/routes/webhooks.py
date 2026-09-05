"""Webhook ingestion routes for payment gateways."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.integrations.razorpay.security import (
    RazorpaySignatureVerificationError,
    verify_razorpay_signature,
)
from backend.app.integrations.razorpay.translator import (
    RazorpayTranslationError,
    RazorpayWebhookTranslator,
)
from backend.app.persistence.database import get_db
from backend.app.persistence.repositories.events import EventRepository

router = APIRouter(prefix="/api/v1/webhooks", tags=["webhooks"])


@router.post("/razorpay", status_code=status.HTTP_200_OK)
async def handle_razorpay_webhook(
    request: Request,
    x_razorpay_signature: str | None = Header(
        default=None,
        alias="X-Razorpay-Signature",
    ),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    """Receive, authenticate, translate, and idempotently persist a webhook."""
    body_bytes = await request.body()
    webhook_secret = settings.razorpay_webhook_secret.strip()

    if webhook_secret:
        if not x_razorpay_signature:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing Razorpay webhook signature",
            )
        try:
            verify_razorpay_signature(
                payload_body=body_bytes,
                signature=x_razorpay_signature,
                secret=webhook_secret,
            )
        except RazorpaySignatureVerificationError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(exc),
            ) from exc
    elif not settings.allow_insecure_webhook_bypass:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Razorpay webhook secret is not configured. Configure "
                "RAZORPAY_WEBHOOK_SECRET or explicitly enable "
                "ALLOW_INSECURE_WEBHOOK_BYPASS for local development."
            ),
        )

    try:
        payload_json = json.loads(body_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON payload",
        ) from exc

    try:
        domain_event = RazorpayWebhookTranslator.translate(payload_json)
    except RazorpayTranslationError as exc:
        return {"status": "ignored", "reason": str(exc)}

    event_repo = EventRepository(db)
    save_result = event_repo.save(domain_event)
    db.commit()

    return {
        "status": "processed",
        "event_id": str(domain_event.envelope.event_id),
        "outcome": save_result.outcome.value,
    }
