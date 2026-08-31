"""Demo graph analysis script.

Builds the structural graph from the demo dataset and reports statistics.
"""

from backend.app.graph.builder import StructuralGraphBuilder
from backend.app.graph.components import ConnectedComponentExtractor
from backend.app.persistence.database import SessionLocal
from backend.app.persistence.reconstruction import ReconstructionService


def analyze_demo_graph() -> None:
    """Build and analyze the structural graph from demo data."""
    print("Building structural graph from demo dataset...")

    # Reconstruct from PostgreSQL
    with SessionLocal() as session:
        reconstruction_service = ReconstructionService(session)
        snapshot = reconstruction_service.reconstruct()

    print(f"Reconstructed state:")
    print(f"  Payments: {len(snapshot.payments)}")
    print(f"  Refunds: {len(snapshot.refunds)}")
    print(f"  Orders: {len(snapshot.orders)}")

    # Build structural graph
    builder = StructuralGraphBuilder()
    graph = builder.build(snapshot)

    print(f"\nStructural graph:")
    print(f"  Total nodes: {graph.node_count()}")
    print(f"  Total edges: {graph.edge_count()}")
    print(f"  Nodes by type:")
    for node_type, count in graph.node_count_by_type().items():
        print(f"    {node_type.value}: {count}")

    # Extract connected components
    extractor = ConnectedComponentExtractor()
    components = extractor.extract(graph)

    print(f"\nConnected components:")
    print(f"  Total components: {len(components)}")

    # Show component statistics
    for i, component in enumerate(components[:10]):  # Show first 10
        print(f"\n  Component {i + 1} (ID: {component.component_id}):")
        print(f"    Nodes: {component.node_count()}")
        print(f"    Edges: {component.edge_count()}")
        print(f"    Nodes by type:")
        for node_type, count in component.node_count_by_type().items():
            print(f"      {node_type.value}: {count}")

    if len(components) > 10:
        print(f"\n  ... and {len(components) - 10} more components")


if __name__ == "__main__":
    analyze_demo_graph()
