"""Structural graph model for Refund Sentinel.

Defines immutable representations for nodes, edges, and the structural graph.
The graph represents relationships between entities based on reconstructed financial facts.

IMPORTANT: This is a STRUCTURAL graph only. It does not classify fraud, calculate risk,
or make abuse decisions. Structural connectivity alone is not a fraud signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from backend.app.domain.identifiers import (
    AddressId,
    CustomerId,
    DeviceId,
    OrderId,
    PaymentId,
    RefundId,
)

if TYPE_CHECKING:
    pass


class NodeType(str, Enum):
    """Types of nodes in the structural graph."""

    CUSTOMER = "CUSTOMER"
    PAYMENT = "PAYMENT"
    ORDER = "ORDER"
    REFUND = "REFUND"
    DEVICE = "DEVICE"
    ADDRESS = "ADDRESS"


class EdgeType(str, Enum):
    """Types of edges in the structural graph."""

    CUSTOMER_OWNS_PAYMENT = "customer_owns_payment"
    PAYMENT_BELONGS_TO_ORDER = "payment_belongs_to_order"
    REFUND_TARGETS_PAYMENT = "refund_targets_payment"
    CUSTOMER_PLACES_ORDER = "customer_places_order"
    CUSTOMER_USES_DEVICE = "customer_uses_device"
    ORDER_SHIPS_TO_ADDRESS = "order_ships_to_address"


@dataclass(frozen=True)
class GraphNode:
    """A node in the structural graph.

    Nodes are immutable and identified by their entity ID and type.
    """

    node_id: str  # String representation of the entity ID
    node_type: NodeType

    def __post_init__(self) -> None:
        # Normalize node_id to string for consistent hashing
        object.__setattr__(self, "node_id", str(self.node_id))

    def __hash__(self) -> int:
        return hash((self.node_id, self.node_type))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GraphNode):
            return False
        return self.node_id == other.node_id and self.node_type == other.node_type


@dataclass(frozen=True)
class GraphEdge:
    """A directed edge in the structural graph.

    Edges are immutable and represent structural relationships between entities.
    """

    source_node_id: str
    target_node_id: str
    edge_type: EdgeType

    def __post_init__(self) -> None:
        # Normalize node IDs to strings for consistent hashing
        object.__setattr__(self, "source_node_id", str(self.source_node_id))
        object.__setattr__(self, "target_node_id", str(self.target_node_id))

    def __hash__(self) -> int:
        return hash((self.source_node_id, self.target_node_id, self.edge_type))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GraphEdge):
            return False
        return (
            self.source_node_id == other.source_node_id
            and self.target_node_id == other.target_node_id
            and self.edge_type == other.edge_type
        )


@dataclass(frozen=True)
class StructuralGraph:
    """Immutable structural graph of entity relationships.

    The graph is built from reconstructed financial state and represents
    structural relationships between customers, payments, orders, refunds,
    devices, and addresses.

    The graph is deterministic: for the same input state, the same
    nodes, edges, and ordering are produced.
    """

    nodes: frozenset[GraphNode] = field(default_factory=frozenset)
    edges: frozenset[GraphEdge] = field(default_factory=frozenset)

    def get_nodes_by_type(self, node_type: NodeType) -> list[GraphNode]:
        """Return all nodes of a specific type, deterministically ordered."""
        return sorted(
            [node for node in self.nodes if node.node_type == node_type],
            key=lambda n: n.node_id,
        )

    def get_edges_by_type(self, edge_type: EdgeType) -> list[GraphEdge]:
        """Return all edges of a specific type, deterministically ordered."""
        return sorted(
            [edge for edge in self.edges if edge.edge_type == edge_type],
            key=lambda e: (e.source_node_id, e.target_node_id),
        )

    def get_edges_for_node(self, node_id: str) -> list[GraphEdge]:
        """Return all edges connected to a node, deterministically ordered."""
        node_id_str = str(node_id)
        return sorted(
            [
                edge
                for edge in self.edges
                if edge.source_node_id == node_id_str or edge.target_node_id == node_id_str
            ],
            key=lambda e: (e.source_node_id, e.target_node_id),
        )

    def node_count(self) -> int:
        """Return the total number of nodes in the graph."""
        return len(self.nodes)

    def edge_count(self) -> int:
        """Return the total number of edges in the graph."""
        return len(self.edges)

    def node_count_by_type(self) -> dict[NodeType, int]:
        """Return the count of nodes by type."""
        counts: dict[NodeType, int] = {}
        for node in self.nodes:
            counts[node.node_type] = counts.get(node.node_type, 0) + 1
        return counts
