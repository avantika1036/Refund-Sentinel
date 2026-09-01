"""Financial exposure estimation for Refund Sentinel.

Separates three different forms of risk-weighted financial exposure:

- realized_suspicious_amount:
    Money already returned through processed refunds.

- pending_refund_exposure:
    Money requested for refund but not yet processed.

- remaining_refundable_exposure:
    Additional money that could still potentially be refunded in the future.

All monetary values remain integer paise. Risk weighting is rounded to the
nearest paise using Decimal arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Iterable

from backend.app.domain.enums import RefundStatus
from backend.app.domain.identifiers import RefundId
from backend.app.domain.value_objects import Money
from backend.app.finance.types import ReconstructionSnapshot


@dataclass(frozen=True)
class FinancialExposure:
    """Risk-weighted financial exposure for a case or cluster.

    The three values intentionally represent different concepts and must not
    be combined into a single "loss" or "recoverable amount".
    """

    realized_suspicious_amount: Money
    pending_refund_exposure: Money
    remaining_refundable_exposure: Money


def compute_financial_exposure(
    snapshot: ReconstructionSnapshot,
    refund_ids: Iterable[RefundId],
    risk_score: float,
) -> FinancialExposure:
    """Compute risk-weighted financial exposure for selected refunds.

    Args:
        snapshot:
            Reconstructed financial state.

        refund_ids:
            Refunds belonging to the case or cluster being evaluated.

        risk_score:
            Final risk score in the inclusive range [0.0, 1.0].

    Returns:
        FinancialExposure containing three separate risk-weighted amounts.

    Raises:
        ValueError:
            If risk_score is outside [0.0, 1.0] or a refund does not exist.
    """

    if not 0.0 <= risk_score <= 1.0:
        raise ValueError(
            f"risk_score must be between 0.0 and 1.0, got {risk_score}"
        )

    selected_refund_ids = tuple(refund_ids)

    if not selected_refund_ids:
        return FinancialExposure(
            realized_suspicious_amount=Money.of_paise(0),
            pending_refund_exposure=Money.of_paise(0),
            remaining_refundable_exposure=Money.of_paise(0),
        )

    selected_refunds = []

    for refund_id in selected_refund_ids:
        refund_state = snapshot.refunds.get(refund_id)

        if refund_state is None:
            raise ValueError(
                f"Refund {refund_id} not found in snapshot"
            )

        selected_refunds.append(refund_state)

    realized_amount_paise = 0
    pending_amount_paise = 0

    selected_payment_ids = set()

    for refund_state in selected_refunds:
        selected_payment_ids.add(refund_state.payment_id)

        if refund_state.status == RefundStatus.PROCESSED:
            if refund_state.processed_amount is not None:
                realized_amount_paise += (
                    refund_state.processed_amount.amount_paise
                )

        elif refund_state.status in {
            RefundStatus.REQUESTED,
            RefundStatus.CREATED,
        }:
            pending_amount_paise += (
                refund_state.requested_amount.amount_paise
            )

    remaining_amount_paise = 0

    for payment_id in selected_payment_ids:
        payment_state = snapshot.payments.get(payment_id)

        if payment_state is None:
            continue

        remaining_paise = (
            payment_state.remaining_refundable.amount_paise
        )

        # remaining_refundable on PaymentState subtracts processed refunds.
        # For forward-looking exposure, refunds that are already requested but
        # not processed must also be excluded because they belong to pending
        # exposure rather than future, not-yet-requested exposure.
        pending_on_payment_paise = sum(
            refund.requested_amount.amount_paise
            for refund in snapshot.refunds.values()
            if refund.payment_id == payment_id
            and refund.status in {
                RefundStatus.REQUESTED,
                RefundStatus.CREATED,
            }
        )

        remaining_after_pending_paise = max(
            0,
            remaining_paise - pending_on_payment_paise,
        )

        remaining_amount_paise += remaining_after_pending_paise

    return FinancialExposure(
        realized_suspicious_amount=Money.of_paise(
            _apply_risk_weight(
                realized_amount_paise,
                risk_score,
            )
        ),
        pending_refund_exposure=Money.of_paise(
            _apply_risk_weight(
                pending_amount_paise,
                risk_score,
            )
        ),
        remaining_refundable_exposure=Money.of_paise(
            _apply_risk_weight(
                remaining_amount_paise,
                risk_score,
            )
        ),
    )


def _apply_risk_weight(
    amount_paise: int,
    risk_score: float,
) -> int:
    """Apply a risk score to an integer paise amount deterministically."""

    weighted_amount = (
        Decimal(amount_paise)
        * Decimal(str(risk_score))
    )

    return int(
        weighted_amount.quantize(
            Decimal("1"),
            rounding=ROUND_HALF_UP,
        )
    )