"""Focused unit tests for feature extraction.

Verifies feature calculations without relying on persistence or ground truth.
"""

from datetime import datetime, timedelta, timezone

from backend.app.domain.enums import PaymentStatus, RefundReasonCode, RefundStatus
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
from backend.app.graph.components import ConnectedComponent
from backend.app.graph.model import GraphNode, NodeType
from backend.app.risk.features import ClusterFeatures, IndividualFeatures, RelationshipFeatures


def test_capture_to_refund_latency_uses_occurred_at() -> None:
    """Lifecycle calculations use occurred_at timestamps."""
    customer_id = CustomerId.generate()
    payment_id = PaymentId.generate()
    order_id = OrderId.generate()
    refund_id = RefundId.generate()

    now = datetime.now(timezone.utc)
    captured_at = UTCDateTime(value=now - timedelta(hours=2))
    refund_requested_at = UTCDateTime(value=now)

    snapshot = ReconstructionSnapshot(
        payments={
            payment_id: PaymentState(
                payment_id=payment_id,
                merchant_id=CustomerId.generate(),
                order_id=order_id,
                customer_id=customer_id,
                authorised_amount=Money.of_paise(1000),
                status=PaymentStatus.CAPTURED,
                created_at=UTCDateTime(value=now - timedelta(hours=3)),
                captured_amount=Money.of_paise(1000),
                captured_at=captured_at,
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
                requested_at=refund_requested_at,
                processed_at=UTCDateTime(value=now),
                processed_amount=Money.of_paise(500),
            )
        },
        orders={
            order_id: OrderState(
                order_id=order_id,
                merchant_id=CustomerId.generate(),
                customer_id=customer_id,
                amount=Money.of_paise(1000),
                created_at=UTCDateTime(value=now - timedelta(hours=4)),
            )
        },
        reconstruction_ordinal=1,
        event_count=2,
    )

    from backend.app.risk.features.individual import IndividualFeatureExtractor

    extractor = IndividualFeatureExtractor(snapshot)
    features = extractor.extract_for_refund(refund_id)

    # Should be 2 hours (not using received_at)
    assert features.capture_to_refund_latency_hrs == 2.0


def test_missing_delivery_data_produces_null() -> None:
    """Missing delivery data produces null where required."""
    customer_id = CustomerId.generate()
    payment_id = PaymentId.generate()
    order_id = OrderId.generate()
    refund_id = RefundId.generate()

    now = datetime.now(timezone.utc)

    snapshot = ReconstructionSnapshot(
        payments={
            payment_id: PaymentState(
                payment_id=payment_id,
                merchant_id=CustomerId.generate(),
                order_id=order_id,
                customer_id=customer_id,
                authorised_amount=Money.of_paise(1000),
                status=PaymentStatus.CAPTURED,
                created_at=UTCDateTime(value=now - timedelta(hours=3)),
                captured_amount=Money.of_paise(1000),
                captured_at=UTCDateTime(value=now - timedelta(hours=2)),
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
                requested_at=UTCDateTime(value=now),
                processed_at=UTCDateTime(value=now),
                processed_amount=Money.of_paise(500),
            )
        },
        orders={
            order_id: OrderState(
                order_id=order_id,
                merchant_id=CustomerId.generate(),
                customer_id=customer_id,
                amount=Money.of_paise(1000),
                created_at=UTCDateTime(value=now - timedelta(hours=4)),
                # No delivered_at
            )
        },
        reconstruction_ordinal=1,
        event_count=2,
    )

    from backend.app.risk.features.individual import IndividualFeatureExtractor

    extractor = IndividualFeatureExtractor(snapshot)
    features = extractor.extract_for_refund(refund_id)

    # Should be None when delivery data is missing
    assert features.delivery_to_refund_latency_hrs is None


def test_full_partial_refund_fractions() -> None:
    """Full and partial refund fractions are correct."""
    customer_id = CustomerId.generate()
    payment_id = PaymentId.generate()
    order_id = OrderId.generate()

    now = datetime.now(timezone.utc)

    snapshot = ReconstructionSnapshot(
        payments={
            payment_id: PaymentState(
                payment_id=payment_id,
                merchant_id=CustomerId.generate(),
                order_id=order_id,
                customer_id=customer_id,
                authorised_amount=Money.of_paise(1000),
                status=PaymentStatus.CAPTURED,
                created_at=UTCDateTime(value=now - timedelta(hours=3)),
                captured_amount=Money.of_paise(1000),
                captured_at=UTCDateTime(value=now - timedelta(hours=2)),
            )
        },
        refunds={},
        orders={
            order_id: OrderState(
                order_id=order_id,
                merchant_id=CustomerId.generate(),
                customer_id=customer_id,
                amount=Money.of_paise(1000),
                created_at=UTCDateTime(value=now - timedelta(hours=4)),
            )
        },
        reconstruction_ordinal=1,
        event_count=1,
    )

    from backend.app.risk.features.individual import IndividualFeatureExtractor

    extractor = IndividualFeatureExtractor(snapshot)

    # Full refund
    full_refund_id = RefundId.generate()
    snapshot.refunds[full_refund_id] = RefundState(
        refund_id=full_refund_id,
        payment_id=payment_id,
        merchant_id=CustomerId.generate(),
        customer_id=customer_id,
        order_id=order_id,
        requested_amount=Money.of_paise(1000),
        status=RefundStatus.PROCESSED,
        requested_at=UTCDateTime(value=now),
        processed_at=UTCDateTime(value=now),
        processed_amount=Money.of_paise(1000),
    )
    features = extractor.extract_for_refund(full_refund_id)
    assert features.refund_amount_fraction == 1.0
    assert features.is_full_refund is True

    # Partial refund
    partial_refund_id = RefundId.generate()
    snapshot.refunds[partial_refund_id] = RefundState(
        refund_id=partial_refund_id,
        payment_id=payment_id,
        merchant_id=CustomerId.generate(),
        customer_id=customer_id,
        order_id=order_id,
        requested_amount=Money.of_paise(500),
        status=RefundStatus.PROCESSED,
        requested_at=UTCDateTime(value=now),
        processed_at=UTCDateTime(value=now),
        processed_amount=Money.of_paise(500),
    )
    features = extractor.extract_for_refund(partial_refund_id)
    assert features.refund_amount_fraction == 0.5
    assert features.is_full_refund is False


def test_insufficient_history_produces_null() -> None:
    """Insufficient history produces null where required."""
    customer_id = CustomerId.generate()
    payment_id = PaymentId.generate()
    order_id = OrderId.generate()
    refund_id = RefundId.generate()

    now = datetime.now(timezone.utc)

    snapshot = ReconstructionSnapshot(
        payments={
            payment_id: PaymentState(
                payment_id=payment_id,
                merchant_id=CustomerId.generate(),
                order_id=order_id,
                customer_id=customer_id,
                authorised_amount=Money.of_paise(1000),
                status=PaymentStatus.CAPTURED,
                created_at=UTCDateTime(value=now - timedelta(hours=3)),
                captured_amount=Money.of_paise(1000),
                captured_at=UTCDateTime(value=now - timedelta(hours=2)),
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
                requested_at=UTCDateTime(value=now),
                processed_at=UTCDateTime(value=now),
                processed_amount=Money.of_paise(500),
            )
        },
        orders={
            order_id: OrderState(
                order_id=order_id,
                merchant_id=CustomerId.generate(),
                customer_id=customer_id,
                amount=Money.of_paise(1000),
                created_at=UTCDateTime(value=now - timedelta(hours=4)),
            )
        },
        reconstruction_ordinal=1,
        event_count=2,
    )

    from backend.app.risk.features.individual import IndividualFeatureExtractor

    extractor = IndividualFeatureExtractor(snapshot)
    features = extractor.extract_for_refund(refund_id)

    # Only 1 order, insufficient for 90-day rate calculation
    assert features.customer_refund_rate_90d is None


def test_cluster_features_for_abuse_like_cluster() -> None:
    """Cluster features behave correctly for abuse-like cluster."""
    customer1 = CustomerId.generate()
    customer2 = CustomerId.generate()
    payment1 = PaymentId.generate()
    payment2 = PaymentId.generate()
    order1 = OrderId.generate()
    order2 = OrderId.generate()
    refund1 = RefundId.generate()
    refund2 = RefundId.generate()
    shared_address = AddressId.generate()

    now = datetime.now(timezone.utc)

    snapshot = ReconstructionSnapshot(
        payments={
            payment1: PaymentState(
                payment_id=payment1,
                merchant_id=CustomerId.generate(),
                order_id=order1,
                customer_id=customer1,
                authorised_amount=Money.of_paise(1000),
                status=PaymentStatus.CAPTURED,
                created_at=UTCDateTime(value=now - timedelta(hours=10)),
                captured_amount=Money.of_paise(1000),
                captured_at=UTCDateTime(value=now - timedelta(hours=9)),
            ),
            payment2: PaymentState(
                payment_id=payment2,
                merchant_id=CustomerId.generate(),
                order_id=order2,
                customer_id=customer2,
                authorised_amount=Money.of_paise(1000),
                status=PaymentStatus.CAPTURED,
                created_at=UTCDateTime(value=now - timedelta(hours=10)),
                captured_amount=Money.of_paise(1000),
                captured_at=UTCDateTime(value=now - timedelta(hours=9)),
            ),
        },
        refunds={
            refund1: RefundState(
                refund_id=refund1,
                payment_id=payment1,
                merchant_id=CustomerId.generate(),
                customer_id=customer1,
                order_id=order1,
                requested_amount=Money.of_paise(800),
                status=RefundStatus.PROCESSED,
                requested_at=UTCDateTime(value=now - timedelta(hours=8)),
                processed_at=UTCDateTime(value=now - timedelta(hours=7)),
                processed_amount=Money.of_paise(800),
            ),
            refund2: RefundState(
                refund_id=refund2,
                payment_id=payment2,
                merchant_id=CustomerId.generate(),
                customer_id=customer2,
                order_id=order2,
                requested_amount=Money.of_paise(800),
                status=RefundStatus.PROCESSED,
                requested_at=UTCDateTime(value=now - timedelta(hours=8)),
                processed_at=UTCDateTime(value=now - timedelta(hours=7)),
                processed_amount=Money.of_paise(800),
            ),
        },
        orders={
            order1: OrderState(
                order_id=order1,
                merchant_id=CustomerId.generate(),
                customer_id=customer1,
                amount=Money.of_paise(1000),
                created_at=UTCDateTime(value=now - timedelta(hours=11)),
                shipping_address_id=shared_address,
            ),
            order2: OrderState(
                order_id=order2,
                merchant_id=CustomerId.generate(),
                customer_id=customer2,
                amount=Money.of_paise(1000),
                created_at=UTCDateTime(value=now - timedelta(hours=11)),
                shipping_address_id=shared_address,
            ),
        },
        reconstruction_ordinal=1,
        event_count=4,
    )

    # Create a component with both customers
    component = ConnectedComponent(
        component_id=customer1.value,  # Use UUID value
        nodes=frozenset([
            GraphNode(str(customer1), NodeType.CUSTOMER),
            GraphNode(str(customer2), NodeType.CUSTOMER),
        ]),
        edges=frozenset(),
    )

    from backend.app.risk.features.cluster import ClusterFeatureExtractor

    extractor = ClusterFeatureExtractor(snapshot)
    features = extractor.extract_for_component(component)

    # Both customers have refunds
    assert features.cluster_refund_active_fraction == 1.0
    # Both refunds have same amount
    assert features.cluster_amount_concentration == 1.0
    # Reason similarity is 0 since we don't have event history in test data
    assert features.cluster_reason_similarity == 0.0


def test_cluster_features_for_singleton() -> None:
    """Cluster features for singleton cluster."""
    customer_id = CustomerId.generate()
    payment_id = PaymentId.generate()
    order_id = OrderId.generate()
    refund_id = RefundId.generate()

    now = datetime.now(timezone.utc)

    snapshot = ReconstructionSnapshot(
        payments={
            payment_id: PaymentState(
                payment_id=payment_id,
                merchant_id=CustomerId.generate(),
                order_id=order_id,
                customer_id=customer_id,
                authorised_amount=Money.of_paise(1000),
                status=PaymentStatus.CAPTURED,
                created_at=UTCDateTime(value=now - timedelta(hours=3)),
                captured_amount=Money.of_paise(1000),
                captured_at=UTCDateTime(value=now - timedelta(hours=2)),
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
                requested_at=UTCDateTime(value=now),
                processed_at=UTCDateTime(value=now),
                processed_amount=Money.of_paise(500),
            )
        },
        orders={
            order_id: OrderState(
                order_id=order_id,
                merchant_id=CustomerId.generate(),
                customer_id=customer_id,
                amount=Money.of_paise(1000),
                created_at=UTCDateTime(value=now - timedelta(hours=4)),
            )
        },
        reconstruction_ordinal=1,
        event_count=2,
    )

    component = ConnectedComponent(
        component_id=str(customer_id),
        nodes=frozenset([GraphNode(str(customer_id), NodeType.CUSTOMER)]),
        edges=frozenset(),
    )

    from backend.app.risk.features.cluster import ClusterFeatureExtractor

    extractor = ClusterFeatureExtractor(snapshot)
    features = extractor.extract_for_component(component)

    # Singleton cluster
    assert features.cluster_size == 1
    # One customer with refund
    assert features.cluster_refund_active_fraction == 1.0
    # Single refund, no variance to measure
    assert features.cluster_lifecycle_timing_alignment == 0.0


def test_relationship_features_use_graph_structure() -> None:
    """Relationship features use graph structure."""
    customer1 = CustomerId.generate()
    customer2 = CustomerId.generate()
    order1 = OrderId.generate()
    order2 = OrderId.generate()
    shared_address = AddressId.generate()

    now = datetime.now(timezone.utc)

    snapshot = ReconstructionSnapshot(
        payments={},
        refunds={},
        orders={
            order1: OrderState(
                order_id=order1,
                merchant_id=CustomerId.generate(),
                customer_id=customer1,
                amount=Money.of_paise(1000),
                created_at=UTCDateTime(value=now),
                shipping_address_id=shared_address,
            ),
            order2: OrderState(
                order_id=order2,
                merchant_id=CustomerId.generate(),
                customer_id=customer2,
                amount=Money.of_paise(1000),
                created_at=UTCDateTime(value=now),
                shipping_address_id=shared_address,
            ),
        },
        reconstruction_ordinal=1,
        event_count=2,
    )

    from backend.app.graph.model import GraphEdge, EdgeType

    component = ConnectedComponent(
        component_id=str(customer1),
        nodes=frozenset([
            GraphNode(str(customer1), NodeType.CUSTOMER),
            GraphNode(str(customer2), NodeType.CUSTOMER),
            GraphNode(str(shared_address), NodeType.ADDRESS),
        ]),
        edges=frozenset([
            GraphEdge(str(order1), str(shared_address), EdgeType.ORDER_SHIPS_TO_ADDRESS),
            GraphEdge(str(order2), str(shared_address), EdgeType.ORDER_SHIPS_TO_ADDRESS),
        ]),
    )

    from backend.app.risk.features.relationship import RelationshipFeatureExtractor

    extractor = RelationshipFeatureExtractor(snapshot)
    features = extractor.extract_for_component(component)

    # Shared address detected
    assert features.shared_attribute_type_count == 1


def test_deterministic_repeated_calculation() -> None:
    """Feature calculation is deterministic."""
    customer_id = CustomerId.generate()
    payment_id = PaymentId.generate()
    order_id = OrderId.generate()
    refund_id = RefundId.generate()

    now = datetime.now(timezone.utc)

    snapshot = ReconstructionSnapshot(
        payments={
            payment_id: PaymentState(
                payment_id=payment_id,
                merchant_id=CustomerId.generate(),
                order_id=order_id,
                customer_id=customer_id,
                authorised_amount=Money.of_paise(1000),
                status=PaymentStatus.CAPTURED,
                created_at=UTCDateTime(value=now - timedelta(hours=3)),
                captured_amount=Money.of_paise(1000),
                captured_at=UTCDateTime(value=now - timedelta(hours=2)),
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
                requested_at=UTCDateTime(value=now),
                processed_at=UTCDateTime(value=now),
                processed_amount=Money.of_paise(500),
            )
        },
        orders={
            order_id: OrderState(
                order_id=order_id,
                merchant_id=CustomerId.generate(),
                customer_id=customer_id,
                amount=Money.of_paise(1000),
                created_at=UTCDateTime(value=now - timedelta(hours=4)),
            )
        },
        reconstruction_ordinal=1,
        event_count=2,
    )

    from backend.app.risk.features.individual import IndividualFeatureExtractor

    extractor1 = IndividualFeatureExtractor(snapshot)
    features1 = extractor1.extract_for_refund(refund_id)

    extractor2 = IndividualFeatureExtractor(snapshot)
    features2 = extractor2.extract_for_refund(refund_id)

    assert features1 == features2


def test_no_label_leakage() -> None:
    """Features do not use ground-truth labels."""
    # This test verifies that the feature extraction code
    # does not import or use any label-related modules
    from backend.app.risk.features import individual, cluster, relationship

    # Check that these modules don't import simulator labels
    assert "simulator" not in str(individual.__dict__)
    assert "simulator" not in str(cluster.__dict__)
    assert "simulator" not in str(relationship.__dict__)
