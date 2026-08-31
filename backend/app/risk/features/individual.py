"""Individual-level feature extraction.

Computes behavioral features for a single refund/customer based on
reconstructed financial state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from backend.app.domain.enums import RefundReasonCode
from backend.app.domain.identifiers import CustomerId, PaymentId, RefundId
from backend.app.domain.value_objects import Money, UTCDateTime
from backend.app.finance.aggregates import OrderState, PaymentState, RefundState
from backend.app.finance.types import ReconstructionSnapshot

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class IndividualFeatures:
    """Individual-level behavioral features for a refund/customer.

    All features are deterministic and based on observable behavior.
    Missing data is represented as None/null, not imputed.
    """

    # Lifecycle timing features
    capture_to_refund_latency_hrs: float | None
    order_to_refund_latency_hrs: float | None
    delivery_to_refund_latency_hrs: float | None

    # Refund amount features
    refund_amount_fraction: float | None
    is_full_refund: bool | None

    # Customer history features
    customer_refund_rate_90d: float | None
    customer_refund_velocity_7d: int | None
    refund_reason_code: str | None
    reason_rotation_count_90d: int | None
    account_age_at_refund_days: int | None
    prior_successful_orders_no_refund: int | None


class IndividualFeatureExtractor:
    """Extracts individual-level features from reconstructed state."""

    def __init__(self, snapshot: ReconstructionSnapshot) -> None:
        self._snapshot = snapshot

    def extract_for_refund(self, refund_id: RefundId) -> IndividualFeatures:
        """Extract features for a specific refund."""
        refund_state = self._snapshot.refunds.get(refund_id)
        if refund_state is None:
            raise ValueError(f"Refund {refund_id} not found in snapshot")

        payment_state = self._snapshot.payments.get(refund_state.payment_id)
        if payment_state is None:
            raise ValueError(f"Payment {refund_state.payment_id} not found in snapshot")

        order_state = self._snapshot.orders.get(refund_state.order_id)
        if order_state is None:
            raise ValueError(f"Order {refund_state.order_id} not found in snapshot")

        # Lifecycle timing features
        capture_to_refund_latency_hrs = self._compute_capture_to_refund_latency(
            payment_state, refund_state
        )
        order_to_refund_latency_hrs = self._compute_order_to_refund_latency(
            order_state, refund_state
        )
        delivery_to_refund_latency_hrs = self._compute_delivery_to_refund_latency(
            order_state, refund_state
        )

        # Refund amount features
        refund_amount_fraction = self._compute_refund_amount_fraction(
            payment_state, refund_state
        )
        is_full_refund = self._compute_is_full_refund(refund_amount_fraction)

        # Customer history features
        customer_refund_rate_90d = self._compute_customer_refund_rate_90d(
            refund_state.customer_id, refund_state.requested_at
        )
        customer_refund_velocity_7d = self._compute_customer_refund_velocity_7d(
            refund_state.customer_id, refund_state.requested_at
        )
        refund_reason_code = self._encode_refund_reason_code(refund_state)
        reason_rotation_count_90d = self._compute_reason_rotation_count_90d(
            refund_state.customer_id, refund_state.requested_at
        )
        account_age_at_refund_days = self._compute_account_age_at_refund_days(
            refund_state.customer_id, refund_state.requested_at
        )
        prior_successful_orders_no_refund = self._compute_prior_successful_orders_no_refund(
            refund_state.customer_id, refund_state.order_id, refund_state.requested_at
        )

        return IndividualFeatures(
            capture_to_refund_latency_hrs=capture_to_refund_latency_hrs,
            order_to_refund_latency_hrs=order_to_refund_latency_hrs,
            delivery_to_refund_latency_hrs=delivery_to_refund_latency_hrs,
            refund_amount_fraction=refund_amount_fraction,
            is_full_refund=is_full_refund,
            customer_refund_rate_90d=customer_refund_rate_90d,
            customer_refund_velocity_7d=customer_refund_velocity_7d,
            refund_reason_code=refund_reason_code,
            reason_rotation_count_90d=reason_rotation_count_90d,
            account_age_at_refund_days=account_age_at_refund_days,
            prior_successful_orders_no_refund=prior_successful_orders_no_refund,
        )

    def _compute_capture_to_refund_latency(
        self, payment_state: PaymentState, refund_state: RefundState
    ) -> float | None:
        """Compute hours from payment capture to refund request."""
        if payment_state.captured_at is None:
            return None
        delta = refund_state.requested_at.value - payment_state.captured_at.value
        return delta.total_seconds() / 3600.0

    def _compute_order_to_refund_latency(
        self, order_state: OrderState, refund_state: RefundState
    ) -> float | None:
        """Compute hours from order creation to refund request."""
        delta = refund_state.requested_at.value - order_state.created_at.value
        return delta.total_seconds() / 3600.0

    def _compute_delivery_to_refund_latency(
        self, order_state: OrderState, refund_state: RefundState
    ) -> float | None:
        """Compute hours from order delivery to refund request.

        Returns None if delivery data does not exist.
        """
        if order_state.delivered_at is None:
            return None
        delta = refund_state.requested_at.value - order_state.delivered_at.value
        return delta.total_seconds() / 3600.0

    def _compute_refund_amount_fraction(
        self, payment_state: PaymentState, refund_state: RefundState
    ) -> float | None:
        """Compute refund amount as fraction of captured amount."""
        if payment_state.captured_amount is None:
            return None
        if payment_state.captured_amount.amount_paise == 0:
            return None
        return (
            refund_state.requested_amount.amount_paise
            / payment_state.captured_amount.amount_paise
        )

    def _compute_is_full_refund(self, refund_amount_fraction: float | None) -> bool | None:
        """Determine if this is a full refund."""
        if refund_amount_fraction is None:
            return None
        return refund_amount_fraction >= 0.99

    def _compute_customer_refund_rate_90d(
        self, customer_id: CustomerId, refund_requested_at: UTCDateTime
    ) -> float | None:
        """Compute customer refund rate in prior 90 days.

        Returns None if insufficient history (less than 5 orders in prior 90 days).
        """
        cutoff_date = refund_requested_at.value - timedelta(days=90)

        # Count orders in prior 90 days
        order_count = 0
        for order_state in self._snapshot.orders.values():
            if (
                order_state.customer_id == customer_id
                and order_state.created_at.value >= cutoff_date
                and order_state.created_at.value < refund_requested_at.value
            ):
                order_count += 1

        # Count refunds in prior 90 days
        refund_count = 0
        for refund_state in self._snapshot.refunds.values():
            if (
                refund_state.customer_id == customer_id
                and refund_state.requested_at.value >= cutoff_date
                and refund_state.requested_at.value < refund_requested_at.value
            ):
                refund_count += 1

        # Require minimum history
        if order_count < 5:
            return None

        return refund_count / order_count if order_count > 0 else 0.0

    def _compute_customer_refund_velocity_7d(
        self, customer_id: CustomerId, refund_requested_at: UTCDateTime
    ) -> int:
        """Count refunds in prior 7 days."""
        cutoff_date = refund_requested_at.value - timedelta(days=7)

        count = 0
        for refund_state in self._snapshot.refunds.values():
            if (
                refund_state.customer_id == customer_id
                and refund_state.requested_at.value >= cutoff_date
                and refund_state.requested_at.value < refund_requested_at.value
            ):
                count += 1

        return count

    def _encode_refund_reason_code(self, refund_state: RefundState) -> str | None:
        """Encode refund reason code as a stable string.

        Extracts reason code from event history since RefundState doesn't store it directly.
        """
        for _, event in refund_state.event_history:
            if hasattr(event, "payload") and hasattr(event.payload, "reason_code"):
                return event.payload.reason_code.value

        return None

    def _compute_reason_rotation_count_90d(
        self, customer_id: CustomerId, refund_requested_at: UTCDateTime
    ) -> int:
        """Count distinct refund reasons in prior 90 days."""
        cutoff_date = refund_requested_at.value - timedelta(days=90)

        reasons = set()
        for refund_state in self._snapshot.refunds.values():
            if (
                refund_state.customer_id == customer_id
                and refund_state.requested_at.value >= cutoff_date
                and refund_state.requested_at.value < refund_requested_at.value
            ):
                reason = self._encode_refund_reason_code(refund_state)
                if reason is not None:
                    reasons.add(reason)

        return len(reasons)

    def _compute_account_age_at_refund_days(
        self, customer_id: CustomerId, refund_requested_at: UTCDateTime
    ) -> int | None:
        """Compute days from customer registration to refund request.

        Returns None if customer registration date is not available.
        Note: Customer registration date is not in the current reconstructed state.
        This is a known limitation.
        """
        # Customer registration date is not available in PaymentState/OrderState/RefundState
        # This would require access to Customer entity which is not in reconstructed state
        return None

    def _compute_prior_successful_orders_no_refund(
        self, customer_id: CustomerId, current_order_id: object, refund_requested_at: UTCDateTime
    ) -> int:
        """Count prior orders that completed without refund."""
        count = 0

        for order_state in self._snapshot.orders.values():
            if (
                order_state.customer_id == customer_id
                and order_state.order_id != current_order_id
                and order_state.created_at.value < refund_requested_at.value
            ):
                # Check if this order has any refund
                has_refund = False
                for refund_state in self._snapshot.refunds.values():
                    if refund_state.order_id == order_state.order_id:
                        has_refund = True
                        break

                if not has_refund:
                    count += 1

        return count
