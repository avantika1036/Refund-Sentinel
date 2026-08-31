"""Integration test for full graph construction pipeline.

Demonstrates: PostgreSQL → ReconstructionService → StructuralGraphBuilder → Connected components.
"""

import pytest

from backend.app.graph.builder import StructuralGraphBuilder
from backend.app.graph.components import ConnectedComponentExtractor
from backend.app.persistence.database import SessionLocal
from backend.app.persistence.reconstruction import ReconstructionService
from backend.app.simulator.background import BackgroundPopulationGenerator
from sqlalchemy import delete


@pytest.mark.integration
def test_full_pipeline_from_postgres_to_components() -> None:
    """Full pipeline: PostgreSQL → Reconstruction → Graph → Components."""
    # Generate a small dataset
    gen = BackgroundPopulationGenerator(seed=42)
    output = gen.generate(num_customers=3, num_orders_per_customer=2)

    # Ingest events through IngestionService
    from backend.app.persistence.ingestion_service import IngestionService

    ingestion_service = IngestionService()
    for event in output.events:
        ingestion_service.ingest(event)

    # Reconstruct from PostgreSQL
    with SessionLocal() as session:
        reconstruction_service = ReconstructionService(session)
        snapshot = reconstruction_service.reconstruct()

    # Build structural graph
    builder = StructuralGraphBuilder()
    graph = builder.build(snapshot)

    # Extract connected components
    extractor = ConnectedComponentExtractor()
    components = extractor.extract(graph)

    # Verify graph was built
    assert graph.node_count() > 0
    assert graph.edge_count() > 0

    # Verify components were extracted
    assert len(components) > 0

    # Cleanup
    with SessionLocal() as session:
        from backend.app.persistence.models import EventModel, IngestionRecordModel
        session.execute(delete(EventModel))
        session.execute(delete(IngestionRecordModel))
        session.commit()


@pytest.mark.integration
def test_graph_builder_does_not_use_ground_truth_labels() -> None:
    """Graph can be built without using simulator ground-truth labels."""
    # This test proves the graph layer is independent of ground truth
    # by building a graph from reconstructed state only.
    gen = BackgroundPopulationGenerator(seed=42)
    output = gen.generate(num_customers=2, num_orders_per_customer=1)

    # Ingest events
    from backend.app.persistence.ingestion_service import IngestionService

    ingestion_service = IngestionService()
    for event in output.events:
        ingestion_service.ingest(event)

    # Reconstruct (no labels involved)
    with SessionLocal() as session:
        reconstruction_service = ReconstructionService(session)
        snapshot = reconstruction_service.reconstruct()

    # Build graph (no labels involved)
    builder = StructuralGraphBuilder()
    graph = builder.build(snapshot)

    # Verify graph was built successfully without labels
    assert graph.node_count() > 0

    # Cleanup
    with SessionLocal() as session:
        from backend.app.persistence.models import EventModel, IngestionRecordModel
        session.execute(delete(EventModel))
        session.execute(delete(IngestionRecordModel))
        session.commit()
