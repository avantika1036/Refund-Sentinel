"""Connected component extraction for structural graphs.

Implements deterministic connected-component extraction using DFS.
Each component has a stable ID and deterministic ordering.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from backend.app.graph.model import (
    GraphEdge,
    GraphNode,
    NodeType,
    StructuralGraph,
)

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class ConnectedComponent:
    """A connected component in the structural graph.

    Components are immutable and have deterministic IDs based on the
    minimum node ID in the component.
    """

    component_id: str
    nodes: frozenset[GraphNode]
    edges: frozenset[GraphEdge]

    def node_count(self) -> int:
        """Return the total number of nodes in this component."""
        return len(self.nodes)

    def edge_count(self) -> int:
        """Return the total number of edges in this component."""
        return len(self.edges)

    def node_count_by_type(self) -> dict[NodeType, int]:
        """Return the count of nodes by type in this component."""
        counts: dict[NodeType, int] = {}
        for node in self.nodes:
            counts[node.node_type] = counts.get(node.node_type, 0) + 1
        return counts

    def get_nodes_by_type(self, node_type: NodeType) -> list[GraphNode]:
        """Return all nodes of a specific type, deterministically ordered."""
        return sorted(
            [node for node in self.nodes if node.node_type == node_type],
            key=lambda n: n.node_id,
        )

    def contains_node(self, node_id: str) -> bool:
        """Check if a node ID is in this component."""
        return any(node.node_id == node_id for node in self.nodes)


class ConnectedComponentExtractor:
    """Extracts connected components from a structural graph.

    Uses DFS for traversal and produces deterministic output:
    - Component IDs are stable (based on minimum node ID)
    - Components are deterministically ordered
    - Node ordering within components is deterministic
    """

    def extract(self, graph: StructuralGraph) -> list[ConnectedComponent]:
        """Extract connected components from the structural graph.

        Args:
            graph: The structural graph to analyze.

        Returns:
            A deterministically ordered list of connected components.
        """
        # Build adjacency list
        adjacency: dict[str, set[str]] = defaultdict(set)
        for edge in graph.edges:
            adjacency[edge.source_node_id].add(edge.target_node_id)
            adjacency[edge.target_node_id].add(edge.source_node_id)

        # Track visited nodes
        visited: set[str] = set()
        components: list[ConnectedComponent] = []

        # Find all node IDs
        node_ids = {node.node_id for node in graph.nodes}

        # DFS to find connected components
        for node_id in sorted(node_ids):  # Sorted for determinism
            if node_id not in visited:
                component_nodes, component_edges = self._dfs_component(
                    node_id, adjacency, graph, visited
                )
                component_id = min(component_nodes)  # Stable ID
                components.append(
                    ConnectedComponent(
                        component_id=component_id,
                        nodes=frozenset(
                            GraphNode(nid, self._get_node_type(nid, graph))
                            for nid in component_nodes
                        ),
                        edges=frozenset(component_edges),
                    )
                )

        # Sort components by ID for deterministic ordering
        components.sort(key=lambda c: c.component_id)

        return components

    def _dfs_component(
        self,
        start_node: str,
        adjacency: dict[str, set[str]],
        graph: StructuralGraph,
        visited: set[str],
    ) -> tuple[set[str], set[GraphEdge]]:
        """Perform DFS to find all nodes and edges in a component."""
        stack = [start_node]
        component_nodes: set[str] = set()
        component_edges: set[GraphEdge] = set()

        while stack:
            node_id = stack.pop()
            if node_id in visited:
                continue

            visited.add(node_id)
            component_nodes.add(node_id)

            # Add edges connected to this node
            for edge in graph.get_edges_for_node(node_id):
                component_edges.add(edge)

            # Add neighbors to stack
            for neighbor in adjacency.get(node_id, set()):
                if neighbor not in visited:
                    stack.append(neighbor)

        return component_nodes, component_edges

    def _get_node_type(self, node_id: str, graph: StructuralGraph) -> NodeType:
        """Get the node type for a given node ID."""
        for node in graph.nodes:
            if node.node_id == node_id:
                return node.node_type
        raise ValueError(f"Node ID {node_id} not found in graph")
