"""Structural graph module for Refund Sentinel.

The structural graph converts persisted/reconstructed financial facts into
a structural relationship graph. It identifies relationships such as:
- customer ↔ payment
- customer ↔ order
- refund ↔ payment
- order ↔ address

Then it derives structural connected components/clusters.

IMPORTANT:
This is a STRUCTURAL graph only. It does NOT:
- classify fraud
- calculate risk scores
- use ground-truth labels
- use refund timing coordination as a fraud decision
- use ML
- decide AS-01 is malicious
- decide LL-01 is legitimate

Structural connectivity alone is not a fraud decision. That is handled
in later phases (behavioral confirmation, risk scoring).
"""

from backend.app.graph.builder import StructuralGraphBuilder
from backend.app.graph.components import (
    ConnectedComponent,
    ConnectedComponentExtractor,
)
from backend.app.graph.model import (
    EdgeType,
    GraphEdge,
    GraphNode,
    NodeType,
    StructuralGraph,
)

__all__ = [
    "StructuralGraphBuilder",
    "ConnectedComponentExtractor",
    "ConnectedComponent",
    "EdgeType",
    "GraphEdge",
    "GraphNode",
    "NodeType",
    "StructuralGraph",
]
