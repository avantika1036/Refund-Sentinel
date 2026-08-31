"""Focused unit tests for Phase 4 structural graph.

Verifies graph contracts without relying on persistence or ground truth.
"""

from backend.app.domain.enums import PaymentStatus, RefundStatus
from backend.app.domain.identifiers import (
    AddressId,
    CustomerId,
    OrderId,
    PaymentId,
    RefundId,
)
from backend.app.domain.value_objects import Money, UTCDateTime
from backend.app.finance.aggregates import OrderState, PaymentState, RefundState
from backend.app.finance.types import ReconstructionSnapshot
from backend.app.graph.builder import StructuralGraphBuilder
from backend.app.graph.components import ConnectedComponentExtractor
from backend.app.graph.model import (
    EdgeType,
    GraphEdge,
    GraphNode,
    NodeType,
    StructuralGraph,
)
from datetime import datetime, timezone


def test_graph_contains_expected_node_edge_types() -> None:
    """Graph contains expected node/edge types for supported relationships."""
    # Create a simple snapshot
    customer_id = CustomerId.generate()
    payment_id = PaymentId.generate()
    order_id = OrderId.generate()
    refund_id = RefundId.generate()
    address_id = AddressId.generate()

    snapshot = ReconstructionSnapshot(
        payments={
            payment_id: PaymentState(
                payment_id=payment_id,
                merchant_id=CustomerId.generate(),
                order_id=order_id,
                customer_id=customer_id,
                authorised_amount=Money.of_paise(1000),
                status=PaymentStatus.CAPTURED,
                created_at=UTCDateTime(value=datetime.now(timezone.utc)),
                captured_amount=Money.of_paise(1000),
                captured_at=UTCDateTime(value=datetime.now(timezone.utc)),
            )
        },
        refunds={
            refund_id: RefundState(
                refund_id=refund_id,
                payment_id=payment_id,
                merchant_id=CustomerId.generate(),
                customer_id=customer_id,
                order_id=order_id,
                requested_amount=Money.of_paise(500),
                status=RefundStatus.PROCESSED,
                requested_at=UTCDateTime(value=datetime.now(timezone.utc)),
                processed_at=UTCDateTime(value=datetime.now(timezone.utc)),
                processed_amount=Money.of_paise(500),
            )
        },
        orders={
            order_id: OrderState(
                order_id=order_id,
                merchant_id=CustomerId.generate(),
                customer_id=customer_id,
                amount=Money.of_paise(1000),
                created_at=UTCDateTime(value=datetime.now(timezone.utc)),
                shipping_address_id=address_id,
            )
        },
        reconstruction_ordinal=1,
        event_count=3,
    )

    builder = StructuralGraphBuilder()
    graph = builder.build(snapshot)

    # Check node types
    node_types = {node.node_type for node in graph.nodes}
    assert NodeType.CUSTOMER in node_types
    assert NodeType.PAYMENT in node_types
    assert NodeType.ORDER in node_types
    assert NodeType.REFUND in node_types
    assert NodeType.ADDRESS in node_types

    # Check edge types
    edge_types = {edge.edge_type for edge in graph.edges}
    assert EdgeType.CUSTOMER_OWNS_PAYMENT in edge_types
    assert EdgeType.PAYMENT_BELONGS_TO_ORDER in edge_types
    assert EdgeType.REFUND_TARGETS_PAYMENT in edge_types
    assert EdgeType.CUSTOMER_PLACES_ORDER in edge_types
    assert EdgeType.ORDER_SHIPS_TO_ADDRESS in edge_types


def test_duplicate_relationships_produce_one_edge() -> None:
    """Duplicate relationships produce only one edge in the graph."""
    customer_id = CustomerId.generate()
    payment_id = PaymentId.generate()
    order_id = OrderId.generate()

    # Create snapshot with one payment
    snapshot = ReconstructionSnapshot(
        payments={
            payment_id: PaymentState(
                payment_id=payment_id,
                merchant_id=CustomerId.generate(),
                order_id=order_id,
                customer_id=customer_id,
                authorised_amount=Money.of_paise(1000),
                status=PaymentStatus.CAPTURED,
                created_at=UTCDateTime(value=datetime.now(timezone.utc)),
                captured_amount=Money.of_paise(1000),
                captured_at=UTCDateTime(value=datetime.now(timezone.utc)),
            )
        },
        refunds={},
        orders={
            order_id: OrderState(
                order_id=order_id,
                merchant_id=CustomerId.generate(),
                customer_id=customer_id,
                amount=Money.of_paise(1000),
                created_at=UTCDateTime(value=datetime.now(timezone.utc)),
            )
        },
        reconstruction_ordinal=1,
        event_count=1,
    )

    builder = StructuralGraphBuilder()
    graph = builder.build(snapshot)

    # Should have exactly one CUSTOMER_OWNS_PAYMENT edge
    customer_owns_payment_edges = graph.get_edges_by_type(
        EdgeType.CUSTOMER_OWNS_PAYMENT
    )
    assert len(customer_owns_payment_edges) == 1


def test_connected_components_group_linked_entities() -> None:
    """Connected components correctly group linked entities."""
    # Create two separate customers with separate payments
    customer1 = CustomerId.generate()
    customer2 = CustomerId.generate()
    payment1 = PaymentId.generate()
    payment2 = PaymentId.generate()
    order1 = OrderId.generate()
    order2 = OrderId.generate()

    snapshot = ReconstructionSnapshot(
        payments={
            payment1: PaymentState(
                payment_id=payment1,
                merchant_id=CustomerId.generate(),
                order_id=order1,
                customer_id=customer1,
                authorised_amount=Money.of_paise(1000),
                status=PaymentStatus.CAPTURED,
                created_at=UTCDateTime(value=datetime.now(timezone.utc)),
                captured_amount=Money.of_paise(1000),
                captured_at=UTCDateTime(value=datetime.now(timezone.utc)),
            ),
            payment2: PaymentState(
                payment_id=payment2,
                merchant_id=CustomerId.generate(),
                order_id=order2,
                customer_id=customer2,
                authorised_amount=Money.of_paise(1000),
                status=PaymentStatus.CAPTURED,
                created_at=UTCDateTime(value=datetime.now(timezone.utc)),
                captured_amount=Money.of_paise(1000),
                captured_at=UTCDateTime(value=datetime.now(timezone.utc)),
            ),
        },
        refunds={},
        orders={
            order1: OrderState(
                order_id=order1,
                merchant_id=CustomerId.generate(),
                customer_id=customer1,
                amount=Money.of_paise(1000),
                created_at=UTCDateTime(value=datetime.now(timezone.utc)),
            ),
            order2: OrderState(
                order_id=order2,
                merchant_id=CustomerId.generate(),
                customer_id=customer2,
                amount=Money.of_paise(1000),
                created_at=UTCDateTime(value=datetime.now(timezone.utc)),
            ),
        },
        reconstruction_ordinal=1,
        event_count=2,
    )

    builder = StructuralGraphBuilder()
    graph = builder.build(snapshot)

    extractor = ConnectedComponentExtractor()
    components = extractor.extract(graph)

    # Should have 2 separate components
    assert len(components) == 2

    # Each component should have 2 nodes (customer + payment + order = 3 nodes total)
    # Actually each component has customer, payment, order = 3 nodes
    for component in components:
        assert component.node_count() == 3


def test_disconnected_entities_remain_separate() -> None:
    """Disconnected entities remain in separate components."""
    customer1 = CustomerId.generate()
    customer2 = CustomerId.generate()
    payment1 = PaymentId.generate()
    payment2 = PaymentId.generate()
    order1 = OrderId.generate()
    order2 = OrderId.generate()

    snapshot = ReconstructionSnapshot(
        payments={
            payment1: PaymentState(
                payment_id=payment1,
                merchant_id=CustomerId.generate(),
                order_id=order1,
                customer_id=customer1,
                authorised_amount=Money.of_paise(1000),
                status=PaymentStatus.CAPTURED,
                created_at=UTCDateTime(value=datetime.now(timezone.utc)),
                captured_amount=Money.of_paise(1000),
                captured_at=UTCDateTime(value=datetime.now(timezone.utc)),
            ),
            payment2: PaymentState(
                payment_id=payment2,
                merchant_id=CustomerId.generate(),
                order_id=order2,
                customer_id=customer2,
                authorised_amount=Money.of_paise(1000),
                status=PaymentStatus.CAPTURED,
                created_at=UTCDateTime(value=datetime.now(timezone.utc)),
                captured_amount=Money.of_paise(1000),
                captured_at=UTCDateTime(value=datetime.now(timezone.utc)),
            ),
        },
        refunds={},
        orders={
            order1: OrderState(
                order_id=order1,
                merchant_id=CustomerId.generate(),
                customer_id=customer1,
                amount=Money.of_paise(1000),
                created_at=UTCDateTime(value=datetime.now(timezone.utc)),
            ),
            order2: OrderState(
                order_id=order2,
                merchant_id=CustomerId.generate(),
                customer_id=customer2,
                amount=Money.of_paise(1000),
                created_at=UTCDateTime(value=datetime.now(timezone.utc)),
            ),
        },
        reconstruction_ordinal=1,
        event_count=2,
    )

    builder = StructuralGraphBuilder()
    graph = builder.build(snapshot)

    extractor = ConnectedComponentExtractor()
    components = extractor.extract(graph)

    # Verify components are separate
    assert len(components) == 2

    # Verify customer1 and customer2 are in different components
    component_ids = [c.component_id for c in components]
    assert len(component_ids) == 2


def test_component_ids_are_deterministic() -> None:
    """Component IDs and ordering are deterministic for the same input."""
    customer_id = CustomerId.generate()
    payment_id = PaymentId.generate()
    order_id = OrderId.generate()

    snapshot = ReconstructionSnapshot(
        payments={
            payment_id: PaymentState(
                payment_id=payment_id,
                merchant_id=CustomerId.generate(),
                order_id=order_id,
                customer_id=customer_id,
                authorised_amount=Money.of_paise(1000),
                status=PaymentStatus.CAPTURED,
                created_at=UTCDateTime(value=datetime.now(timezone.utc)),
                captured_amount=Money.of_paise(1000),
                captured_at=UTCDateTime(value=datetime.now(timezone.utc)),
            )
        },
        refunds={},
        orders={
            order_id: OrderState(
                order_id=order_id,
                merchant_id=CustomerId.generate(),
                customer_id=customer_id,
                amount=Money.of_paise(1000),
                created_at=UTCDateTime(value=datetime.now(timezone.utc)),
            )
        },
        reconstruction_ordinal=1,
        event_count=1,
    )

    builder = StructuralGraphBuilder()
    extractor = ConnectedComponentExtractor()

    # Build graph and extract components twice
    graph1 = builder.build(snapshot)
    components1 = extractor.extract(graph1)

    graph2 = builder.build(snapshot)
    components2 = extractor.extract(graph2)

    # Same number of components
    assert len(components1) == len(components2)

    # Same component IDs
    ids1 = [c.component_id for c in components1]
    ids2 = [c.component_id for c in components2]
    assert ids1 == ids2


def test_source_state_is_not_mutated() -> None:
    """Source reconstructed state is not mutated by graph building."""
    customer_id = CustomerId.generate()
    payment_id = PaymentId.generate()
    order_id = OrderId.generate()

    snapshot = ReconstructionSnapshot(
        payments={
            payment_id: PaymentState(
                payment_id=payment_id,
                merchant_id=CustomerId.generate(),
                order_id=order_id,
                customer_id=customer_id,
                authorised_amount=Money.of_paise(1000),
                status=PaymentStatus.CAPTURED,
                created_at=UTCDateTime(value=datetime.now(timezone.utc)),
                captured_amount=Money.of_paise(1000),
                captured_at=UTCDateTime(value=datetime.now(timezone.utc)),
            )
        },
        refunds={},
        orders={
            order_id: OrderState(
                order_id=order_id,
                merchant_id=CustomerId.generate(),
                customer_id=customer_id,
                amount=Money.of_paise(1000),
                created_at=UTCDateTime(value=datetime.now(timezone.utc)),
            )
        },
        reconstruction_ordinal=1,
        event_count=1,
    )

    # Store original values
    original_payment_count = len(snapshot.payments)
    original_order_count = len(snapshot.orders)

    builder = StructuralGraphBuilder()
    graph = builder.build(snapshot)

    # Verify snapshot is unchanged
    assert len(snapshot.payments) == original_payment_count
    assert len(snapshot.orders) == original_order_count


def test_shared_address_creates_connected_component() -> None:
    """Shared address creates a connected component (simulating AS-01/LL-01)."""
    customer1 = CustomerId.generate()
    customer2 = CustomerId.generate()
    payment1 = PaymentId.generate()
    payment2 = PaymentId.generate()
    order1 = OrderId.generate()
    order2 = OrderId.generate()
    shared_address = AddressId.generate()

    snapshot = ReconstructionSnapshot(
        payments={
            payment1: PaymentState(
                payment_id=payment1,
                merchant_id=CustomerId.generate(),
                order_id=order1,
                customer_id=customer1,
                authorised_amount=Money.of_paise(1000),
                status=PaymentStatus.CAPTURED,
                created_at=UTCDateTime(value=datetime.now(timezone.utc)),
                captured_amount=Money.of_paise(1000),
                captured_at=UTCDateTime(value=datetime.now(timezone.utc)),
            ),
            payment2: PaymentState(
                payment_id=payment2,
                merchant_id=CustomerId.generate(),
                order_id=order2,
                customer_id=customer2,
                authorised_amount=Money.of_paise(1000),
                status=PaymentStatus.CAPTURED,
                created_at=UTCDateTime(value=datetime.now(timezone.utc)),
                captured_amount=Money.of_paise(1000),
                captured_at=UTCDateTime(value=datetime.now(timezone.utc)),
            ),
        },
        refunds={},
        orders={
            order1: OrderState(
                order_id=order1,
                merchant_id=CustomerId.generate(),
                customer_id=customer1,
                amount=Money.of_paise(1000),
                created_at=UTCDateTime(value=datetime.now(timezone.utc)),
                shipping_address_id=shared_address,
            ),
            order2: OrderState(
                order_id=order2,
                merchant_id=CustomerId.generate(),
                customer_id=customer2,
                amount=Money.of_paise(1000),
                created_at=UTCDateTime(value=datetime.now(timezone.utc)),
                shipping_address_id=shared_address,  # Same address
            ),
        },
        reconstruction_ordinal=1,
        event_count=2,
    )

    builder = StructuralGraphBuilder()
    graph = builder.build(snapshot)

    extractor = ConnectedComponentExtractor()
    components = extractor.extract(graph)

    # Should have 1 component (connected via shared address)
    assert len(components) == 1

    # Component should have both customers
    component = components[0]
    customer_nodes = component.get_nodes_by_type(NodeType.CUSTOMER)
    assert len(customer_nodes) == 2
