"""Structural graph builder from reconstructed financial state.

Builds a structural graph from reconstructed financial facts (PaymentState,
RefundState, OrderState). The builder is deterministic and does not mutate
the source state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from backend.app.domain.identifiers import (
    AddressId,
    CustomerId,
    DeviceId,
    OrderId,
    PaymentId,
    RefundId,
)
from backend.app.finance.types import ReconstructionSnapshot
from backend.app.graph.model import (
    EdgeType,
    GraphEdge,
    GraphNode,
    NodeType,
    StructuralGraph,
)

if TYPE_CHECKING:
    pass


class StructuralGraphBuilder:
    """Builds a structural graph from reconstructed financial state.

    The builder:
    - Consumes domain/reconstructed state
    - Creates nodes and edges for supported relationships
    - Deduplicates identical edges
    - Avoids duplicate nodes
    - Never mutates the source state
    - Produces deterministic output
    """

    def build(self, snapshot: ReconstructionSnapshot) -> StructuralGraph:
        """Build a structural graph from the reconstructed snapshot.

        Args:
            snapshot: The reconstructed financial state from the state engine.

        Returns:
            A deterministic structural graph representing entity relationships.
        """
        nodes: set[GraphNode] = set()
        edges: set[GraphEdge] = set()

        # Build nodes and edges from payments
        for payment_id, payment_state in snapshot.payments.items():
            # Customer node
            customer_node = GraphNode(
                node_id=str(payment_state.customer_id),
                node_type=NodeType.CUSTOMER,
            )
            nodes.add(customer_node)

            # Payment node
            payment_node = GraphNode(
                node_id=str(payment_state.payment_id),
                node_type=NodeType.PAYMENT,
            )
            nodes.add(payment_node)

            # Order node
            order_node = GraphNode(
                node_id=str(payment_state.order_id),
                node_type=NodeType.ORDER,
            )
            nodes.add(order_node)

            # Customer → Payment edge
            edges.add(
                GraphEdge(
                    source_node_id=str(payment_state.customer_id),
                    target_node_id=str(payment_state.payment_id),
                    edge_type=EdgeType.CUSTOMER_OWNS_PAYMENT,
                )
            )

            # Payment → Order edge
            edges.add(
                GraphEdge(
                    source_node_id=str(payment_state.payment_id),
                    target_node_id=str(payment_state.order_id),
                    edge_type=EdgeType.PAYMENT_BELONGS_TO_ORDER,
                )
            )

            # Customer → Order edge
            edges.add(
                GraphEdge(
                    source_node_id=str(payment_state.customer_id),
                    target_node_id=str(payment_state.order_id),
                    edge_type=EdgeType.CUSTOMER_PLACES_ORDER,
                )
            )

        # Build nodes and edges from refunds
        for refund_id, refund_state in snapshot.refunds.items():
            # Refund node
            refund_node = GraphNode(
                node_id=str(refund_state.refund_id),
                node_type=NodeType.REFUND,
            )
            nodes.add(refund_node)

            # Ensure payment node exists (may already be added)
            payment_node = GraphNode(
                node_id=str(refund_state.payment_id),
                node_type=NodeType.PAYMENT,
            )
            nodes.add(payment_node)

            # Ensure customer node exists (may already be added)
            customer_node = GraphNode(
                node_id=str(refund_state.customer_id),
                node_type=NodeType.CUSTOMER,
            )
            nodes.add(customer_node)

            # Refund → Payment edge
            edges.add(
                GraphEdge(
                    source_node_id=str(refund_state.refund_id),
                    target_node_id=str(refund_state.payment_id),
                    edge_type=EdgeType.REFUND_TARGETS_PAYMENT,
                )
            )

        # Build nodes and edges from orders (for address relationships)
        for order_id, order_state in snapshot.orders.items():
            # Order node (may already exist)
            order_node = GraphNode(
                node_id=str(order_state.order_id),
                node_type=NodeType.ORDER,
            )
            nodes.add(order_node)

            # Customer node (may already exist)
            customer_node = GraphNode(
                node_id=str(order_state.customer_id),
                node_type=NodeType.CUSTOMER,
            )
            nodes.add(customer_node)

            # Address node (if present)
            if order_state.shipping_address_id is not None:
                address_node = GraphNode(
                    node_id=str(order_state.shipping_address_id),
                    node_type=NodeType.ADDRESS,
                )
                nodes.add(address_node)

                # Order → Address edge
                edges.add(
                    GraphEdge(
                        source_node_id=str(order_state.order_id),
                        target_node_id=str(order_state.shipping_address_id),
                        edge_type=EdgeType.ORDER_SHIPS_TO_ADDRESS,
                    )
                )

        # Build device nodes and edges from payment events
        # Note: PaymentState doesn't include device_id, so we need to check
        # the event history. For now, we'll skip device edges since they're
        # not in the reconstructed state aggregates.
        # This is a known limitation - device relationships would require
        # access to the original events, not just the reconstructed state.

        # Convert to frozenset for immutability and deterministic ordering
        return StructuralGraph(
            nodes=frozenset(nodes),
            edges=frozenset(edges),
        )
