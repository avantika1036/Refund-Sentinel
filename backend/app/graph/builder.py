"""Structural graph builder from reconstructed financial state.

Builds a structural graph from reconstructed financial facts (PaymentState,
RefundState, OrderState). The builder is deterministic and does not mutate
the source state.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from backend.app.domain.identifiers import (
    AddressId,
    CustomerId,
    DeviceId,
    OrderId,
    PaymentId,
    RefundId,
)
from backend.app.domain.value_objects import UTCDateTime
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

    def build(
        self,
        snapshot: ReconstructionSnapshot,
        as_of: UTCDateTime | None = None,
    ) -> StructuralGraph:
        """Build a structural graph from facts known at ``as_of``.

        Args:
            snapshot: The reconstructed financial state from the state engine.

        Returns:
            A deterministic structural graph representing entity relationships.
        """
        nodes: set[GraphNode] = set()
        edges: set[GraphEdge] = set()
        observation_time = as_of.value if as_of is not None else datetime.max.replace(tzinfo=timezone.utc)

        # Build nodes and edges from payments
        for payment_id, payment_state in snapshot.payments.items():
            if payment_state.created_at.value > observation_time:
                continue
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
            if refund_state.requested_at.value > observation_time:
                continue
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
            if order_state.created_at.value > observation_time:
                continue
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

        # Build device nodes and edges from retained payment event history.
        # PaymentCreatedEvent carries the device identifier, while the
        # aggregate itself intentionally stores the financial state only.
        for payment_state in snapshot.payments.values():
            if payment_state.created_at.value > observation_time:
                continue
            for _, event in payment_state.event_history:
                if event.envelope.occurred_at.value > observation_time:
                    continue
                payload = getattr(event, "payload", None)
                device_id = getattr(payload, "device_id", None)
                if device_id is None:
                    continue

                device_node = GraphNode(
                    node_id=str(device_id),
                    node_type=NodeType.DEVICE,
                )
                nodes.add(device_node)
                edges.add(
                    GraphEdge(
                        source_node_id=str(payment_state.customer_id),
                        target_node_id=str(device_id),
                        edge_type=EdgeType.CUSTOMER_USES_DEVICE,
                    )
                )
                break

        # Convert to frozenset for immutability and deterministic ordering
        return StructuralGraph(
            nodes=frozenset(nodes),
            edges=frozenset(edges),
        )
