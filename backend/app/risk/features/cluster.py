"""Cluster-level feature extraction.

Computes behavioral coordination features for structural clusters.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta, timezone
from typing import TYPE_CHECKING

from backend.app.domain.identifiers import CustomerId, RefundId
from backend.app.domain.value_objects import UTCDateTime
from backend.app.finance.aggregates import PaymentState, RefundState
from backend.app.finance.types import ReconstructionSnapshot
from backend.app.graph.components import ConnectedComponent

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class ClusterFeatures:
    """Cluster-level behavioral features.

    These features measure behavioral coordination of cluster members.
    For singleton clusters, values reflect the lack of coordination evidence.
    """

    cluster_size: int
    cluster_refund_active_fraction: float
    cluster_lifecycle_timing_alignment: float
    cluster_temporal_burst_score: float
    cluster_reason_similarity: float
    cluster_amount_concentration: float


class ClusterFeatureExtractor:
    """Extracts cluster-level features from reconstructed state and graph components."""

    def __init__(self, snapshot: ReconstructionSnapshot) -> None:
        self._snapshot = snapshot

    def extract_for_component(self, component: ConnectedComponent) -> ClusterFeatures:
        """Extract features for a connected component."""
        # Get all customer IDs in this component
        customer_ids = self._get_customer_ids_in_component(component)

        cluster_size = len(customer_ids)
        cluster_refund_active_fraction = self._compute_refund_active_fraction(
            customer_ids
        )
        cluster_lifecycle_timing_alignment = self._compute_lifecycle_timing_alignment(
            customer_ids
        )
        cluster_temporal_burst_score = self._compute_temporal_burst_score(customer_ids)
        cluster_reason_similarity = self._compute_reason_similarity(customer_ids)
        cluster_amount_concentration = self._compute_amount_concentration(customer_ids)

        return ClusterFeatures(
            cluster_size=cluster_size,
            cluster_refund_active_fraction=cluster_refund_active_fraction,
            cluster_lifecycle_timing_alignment=cluster_lifecycle_timing_alignment,
            cluster_temporal_burst_score=cluster_temporal_burst_score,
            cluster_reason_similarity=cluster_reason_similarity,
            cluster_amount_concentration=cluster_amount_concentration,
        )

    def _get_customer_ids_in_component(self, component: ConnectedComponent) -> list[CustomerId]:
        """Extract customer IDs from component nodes."""
        from backend.app.graph.model import NodeType

        customer_ids = []
        for node in component.nodes:
            if node.node_type == NodeType.CUSTOMER:
                customer_ids.append(CustomerId.from_str(node.node_id))
        return customer_ids

    def _compute_refund_active_fraction(self, customer_ids: list[CustomerId]) -> float:
        """Fraction of customers in cluster with active refunds."""
        if not customer_ids:
            return 0.0

        active_customers = 0
        for customer_id in customer_ids:
            has_refund = any(
                refund.customer_id == customer_id
                for refund in self._snapshot.refunds.values()
            )
            if has_refund:
                active_customers += 1

        return active_customers / len(customer_ids)

    def _compute_lifecycle_timing_alignment(self, customer_ids: list[CustomerId]) -> float:
        """Measure alignment of capture-to-refund latencies among active refunding members.

        Formula: 1 - (coefficient of variation of latencies)
        Higher values indicate tighter alignment (lower variance).
        For singletons or insufficient data, returns 0.
        """
        # Collect latencies for all refunds in the cluster
        latencies = []
        for refund_state in self._snapshot.refunds.values():
            if refund_state.customer_id in customer_ids:
                payment_state = self._snapshot.payments.get(refund_state.payment_id)
                if payment_state and payment_state.captured_at:
                    delta = (
                        refund_state.requested_at.value - payment_state.captured_at.value
                    )
                    latencies.append(delta.total_seconds() / 3600.0)

        if len(latencies) < 2:
            return 0.0

        # Compute coefficient of variation
        mean = sum(latencies) / len(latencies)
        variance = sum((x - mean) ** 2 for x in latencies) / len(latencies)
        std = variance ** 0.5

        if mean == 0:
            return 0.0

        cv = std / mean
        # Alignment is inverse of CV, capped at [0, 1]
        alignment = max(0.0, 1.0 - cv)
        return min(1.0, alignment)

    def _compute_temporal_burst_score(self, customer_ids: list[CustomerId]) -> float:
        """Measure concentration of refund requests within a 48-hour window.

        Formula: max_refunds_in_48h / total_refunds
        Higher values indicate burst behavior.
        """
        # Collect all refund timestamps for the cluster
        timestamps = []
        for refund_state in self._snapshot.refunds.values():
            if refund_state.customer_id in customer_ids:
                timestamps.append(refund_state.requested_at.value)

        if not timestamps:
            return 0.0

        if len(timestamps) == 1:
            return 1.0

        # Sort timestamps
        timestamps.sort()

        # Find maximum number of refunds in any 48-hour window
        max_in_window = 0
        for i, ts in enumerate(timestamps):
            window_end = ts + timedelta(hours=48)
            count = 1
            for j in range(i + 1, len(timestamps)):
                if timestamps[j] <= window_end:
                    count += 1
                else:
                    break
            max_in_window = max(max_in_window, count)

        return max_in_window / len(timestamps)

    def _compute_reason_similarity(self, customer_ids: list[CustomerId]) -> float:
        """Measure similarity of primary reason codes across cluster members.

        Formula: (count of most common reason) / (total refunds)
        Higher values indicate reason similarity.
        """
        # Collect reason codes for all refunds in the cluster
        reason_codes = []
        for refund_state in self._snapshot.refunds.values():
            if refund_state.customer_id in customer_ids:
                # Extract reason code from event history
                reason = self._extract_reason_code_from_state(refund_state)
                if reason:
                    reason_codes.append(reason)

        if not reason_codes:
            return 0.0

        # Count frequency of each reason
        from collections import Counter
        counter = Counter(reason_codes)
        most_common_count = counter.most_common(1)[0][1]

        return most_common_count / len(reason_codes)

    def _extract_reason_code_from_state(self, refund_state: RefundState) -> str | None:
        """Extract reason code from refund state event history."""
        for _, event in refund_state.event_history:
            if hasattr(event, 'payload') and hasattr(event.payload, 'reason_code'):
                return event.payload.reason_code.value
        return None

    def _compute_amount_concentration(self, customer_ids: list[CustomerId]) -> float:
        """Measure concentration of refund amounts using coefficient of variation.

        Formula: 1 - (coefficient of variation of refund amounts)
        Higher values indicate amount concentration (low variance).
        """
        # Collect refund amounts for the cluster
        amounts = []
        for refund_state in self._snapshot.refunds.values():
            if refund_state.customer_id in customer_ids:
                amounts.append(refund_state.requested_amount.amount_paise)

        if len(amounts) < 2:
            return 1.0

        # Compute coefficient of variation
        mean = sum(amounts) / len(amounts)
        variance = sum((x - mean) ** 2 for x in amounts) / len(amounts)
        std = variance ** 0.5

        if mean == 0:
            return 0.0

        cv = std / mean
        # Concentration is inverse of CV, capped at [0, 1]
        concentration = max(0.0, 1.0 - cv)
        return min(1.0, concentration)
