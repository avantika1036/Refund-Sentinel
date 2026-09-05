"""Relationship-level feature extraction.

Computes features based on structural graph relationships.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from backend.app.domain.identifiers import CustomerId
from backend.app.domain.value_objects import UTCDateTime
from backend.app.finance.types import ReconstructionSnapshot
from backend.app.graph.components import ConnectedComponent
from backend.app.graph.model import EdgeType, NodeType

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class RelationshipFeatures:
    """Relationship-level features based on graph structure."""

    shared_attribute_type_count: int
    neighborhood_active_refund_count: int


class RelationshipFeatureExtractor:
    """Extracts relationship features from graph components."""

    def __init__(self, snapshot: ReconstructionSnapshot) -> None:
        self._snapshot = snapshot

    def extract_for_component(
        self,
        component: ConnectedComponent,
        as_of: UTCDateTime | None = None,
    ) -> RelationshipFeatures:
        """Extract relationship features for a connected component at a point in time."""
        shared_attribute_type_count = self._compute_shared_attribute_type_count(component)
        neighborhood_active_refund_count = self._compute_neighborhood_active_refund_count(
            component,
            as_of=as_of,
        )

        return RelationshipFeatures(
            shared_attribute_type_count=shared_attribute_type_count,
            neighborhood_active_refund_count=neighborhood_active_refund_count,
        )

    def _compute_shared_attribute_type_count(self, component: ConnectedComponent) -> int:
        """Count distinct shared attribute types in the component.

        Shared attributes are those that connect multiple entities:
        - ADDRESS (shared via order shipping addresses)
        - DEVICE (shared via payment context)
        """
        shared_types = set()

        # Check for shared addresses
        address_nodes = [n for n in component.nodes if n.node_type == NodeType.ADDRESS]
        if address_nodes:
            # Count how many orders share each address
            from collections import defaultdict
            address_order_count = defaultdict(int)
            for edge in component.edges:
                if edge.edge_type == EdgeType.ORDER_SHIPS_TO_ADDRESS:
                    address_order_count[edge.target_node_id] += 1

            # An address is shared if >1 order uses it
            for address_id, count in address_order_count.items():
                if count > 1:
                    shared_types.add("ADDRESS")

        # Check for shared devices.
        device_nodes = [
            node
            for node in component.nodes
            if node.node_type == NodeType.DEVICE
        ]
        if device_nodes:
            from collections import defaultdict

            device_customer_count = defaultdict(set)
            for edge in component.edges:
                if edge.edge_type == EdgeType.CUSTOMER_USES_DEVICE:
                    device_customer_count[edge.target_node_id].add(
                        edge.source_node_id
                    )

            if any(
                len(customer_ids) > 1
                for customer_ids in device_customer_count.values()
            ):
                shared_types.add("DEVICE")

        return len(shared_types)

    def _compute_neighborhood_active_refund_count(
        self,
        component: ConnectedComponent,
        as_of: UTCDateTime | None = None,
    ) -> int:
        """Count unique customers in the component neighborhood with active refunds.

        Per PLAN.MD Section 7: "Count of customers within 1 hop sharing any attribute
        who have active refund events". Returns the count of unique customers (not refund events)
        in this component who have at least one refund.
        """
        # Get all customer IDs in this component
        customer_ids = set()
        for node in component.nodes:
            if node.node_type == NodeType.CUSTOMER:
                customer_ids.add(CustomerId.from_str(node.node_id))

        # Count unique customers with at least one refund
        active_customers = set()
        for refund_state in self._snapshot.refunds.values():
            if (
                refund_state.customer_id in customer_ids
                and (as_of is None or refund_state.requested_at.value <= as_of.value)
            ):
                active_customers.add(refund_state.customer_id)

        return len(active_customers)
