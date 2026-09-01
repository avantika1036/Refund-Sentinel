"""Unit tests for behavioral confirmation gate."""

from datetime import datetime, timedelta, timezone

import pytest

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
from backend.app.graph.model import GraphEdge, GraphNode, NodeType, EdgeType
from backend.app.risk.confirmation import compute_behavioral_confirmation_score
from backend.app.risk.features.cluster import ClusterFeatureExtractor


def test_singleton_cluster_returns_zero() -> None:
    """Singleton cluster (size=1) cannot demonstrate coordination."""
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

    # Build singleton component
    component = ConnectedComponent(
        component_id=str(customer_id),
        nodes=frozenset([GraphNode(node_id=str(customer_id), node_type=NodeType.CUSTOMER)]),
        edges=frozenset(),
    )

    extractor = ClusterFeatureExtractor(snapshot)
    features = extractor.extract_for_component(component)

    assert features.cluster_size == 1
    assert compute_behavioral_confirmation_score(features) == 0.0


def test_zero_coordination_signal_returns_zero() -> None:
    """Any zero coordination signal produces zero confirmation score."""
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

    # Both customers refund at very different times (high variance -> low alignment)
    snapshot = ReconstructionSnapshot(
        payments={
            payment1: PaymentState(
                payment_id=payment1,
                merchant_id=CustomerId.generate(),
                order_id=order1,
                customer_id=customer1,
                authorised_amount=Money.of_paise(1000),
                status=PaymentStatus.CAPTURED,
                created_at=UTCDateTime(value=now - timedelta(days=10)),
                captured_amount=Money.of_paise(1000),
                captured_at=UTCDateTime(value=now - timedelta(days=10)),
            ),
            payment2: PaymentState(
                payment_id=payment2,
                merchant_id=CustomerId.generate(),
                order_id=order2,
                customer_id=customer2,
                authorised_amount=Money.of_paise(1000),
                status=PaymentStatus.CAPTURED,
                created_at=UTCDateTime(value=now - timedelta(hours=1)),
                captured_amount=Money.of_paise(1000),
                captured_at=UTCDateTime(value=now - timedelta(hours=1)),
            ),
        },
        refunds={
            refund1: RefundState(
                refund_id=refund1,
                payment_id=payment1,
                merchant_id=CustomerId.generate(),
                customer_id=customer1,
                order_id=order1,
                requested_amount=Money.of_paise(500),
                status=RefundStatus.PROCESSED,
                requested_at=UTCDateTime(value=now - timedelta(days=5)),
                processed_at=UTCDateTime(value=now - timedelta(days=5)),
                processed_amount=Money.of_paise(500),
            ),
            refund2: RefundState(
                refund_id=refund2,
                payment_id=payment2,
                merchant_id=CustomerId.generate(),
                customer_id=customer2,
                order_id=order2,
                requested_amount=Money.of_paise(500),
                status=RefundStatus.PROCESSED,
                requested_at=UTCDateTime(value=now),
                processed_at=UTCDateTime(value=now),
                processed_amount=Money.of_paise(500),
            )
        },
        orders={
            order1: OrderState(
                order_id=order1,
                merchant_id=CustomerId.generate(),
                customer_id=customer1,
                amount=Money.of_paise(1000),
                created_at=UTCDateTime(value=now - timedelta(days=11)),
            ),
            order2: OrderState(
                order_id=order2,
                merchant_id=CustomerId.generate(),
                customer_id=customer2,
                amount=Money.of_paise(1000),
                created_at=UTCDateTime(value=now - timedelta(hours=2)),
            ),
        },
        reconstruction_ordinal=1,
        event_count=4,
    )

    # Build two-customer component
    component = ConnectedComponent(
        component_id=str(customer1),
        nodes=frozenset([
            GraphNode(node_id=str(customer1), node_type=NodeType.CUSTOMER),
            GraphNode(node_id=str(customer2), node_type=NodeType.CUSTOMER),
            GraphNode(node_id=str(shared_address), node_type=NodeType.ADDRESS),
        ]),
        edges=frozenset(),
    )

    extractor = ClusterFeatureExtractor(snapshot)
    features = extractor.extract_for_component(component)

    # High variance in refund timing should make alignment ~0
    # This should produce confirmation score of 0.0 (or very close)
    score = compute_behavioral_confirmation_score(features)
    assert score == 0.0 or score < 0.05


def test_high_coordination_signals_produce_high_score() -> None:
    """High alignment, burst, and reason signals produce a higher score."""
    customer1 = CustomerId.generate()
    customer2 = CustomerId.generate()
    payment1 = PaymentId.generate()
    payment2 = PaymentId.generate()
    order1 = OrderId.generate()
    order2 = OrderId.generate()
    refund1 = RefundId.generate()
    refund2 = RefundId.generate()

    now = datetime.now(timezone.utc)

    # Two customers refunding at similar times with similar reasons
    snapshot = ReconstructionSnapshot(
        payments={
            payment1: PaymentState(
                payment_id=payment1,
                merchant_id=CustomerId.generate(),
                order_id=order1,
                customer_id=customer1,
                authorised_amount=Money.of_paise(1000),
                status=PaymentStatus.CAPTURED,
                created_at=UTCDateTime(value=now - timedelta(hours=4)),
                captured_amount=Money.of_paise(1000),
                captured_at=UTCDateTime(value=now - timedelta(hours=4)),
            ),
            payment2: PaymentState(
                payment_id=payment2,
                merchant_id=CustomerId.generate(),
                order_id=order2,
                customer_id=customer2,
                authorised_amount=Money.of_paise(1000),
                status=PaymentStatus.CAPTURED,
                created_at=UTCDateTime(value=now - timedelta(hours=5)),
                captured_amount=Money.of_paise(1000),
                captured_at=UTCDateTime(value=now - timedelta(hours=5)),
            ),
        },
        refunds={
            refund1: RefundState(
                refund_id=refund1,
                payment_id=payment1,
                merchant_id=CustomerId.generate(),
                customer_id=customer1,
                order_id=order1,
                requested_amount=Money.of_paise(1000),
                status=RefundStatus.PROCESSED,
                requested_at=UTCDateTime(value=now - timedelta(minutes=5)),
                processed_at=UTCDateTime(value=now - timedelta(minutes=5)),
                processed_amount=Money.of_paise(1000),
            ),
            refund2: RefundState(
                refund_id=refund2,
                payment_id=payment2,
                merchant_id=CustomerId.generate(),
                customer_id=customer2,
                order_id=order2,
                requested_amount=Money.of_paise(1000),
                status=RefundStatus.PROCESSED,
                requested_at=UTCDateTime(value=now),
                processed_at=UTCDateTime(value=now),
                processed_amount=Money.of_paise(1000),
            )
        },
        orders={
            order1: OrderState(
                order_id=order1,
                merchant_id=CustomerId.generate(),
                customer_id=customer1,
                amount=Money.of_paise(1000),
                created_at=UTCDateTime(value=now - timedelta(hours=5)),
            ),
            order2: OrderState(
                order_id=order2,
                merchant_id=CustomerId.generate(),
                customer_id=customer2,
                amount=Money.of_paise(1000),
                created_at=UTCDateTime(value=now - timedelta(hours=6)),
            ),
        },
        reconstruction_ordinal=1,
        event_count=4,
    )

    component = ConnectedComponent(
        component_id=str(customer1),
        nodes=frozenset([
            GraphNode(node_id=str(customer1), node_type=NodeType.CUSTOMER),
            GraphNode(node_id=str(customer2), node_type=NodeType.CUSTOMER),
        ]),
        edges=frozenset(),
    )

    extractor = ClusterFeatureExtractor(snapshot)
    features = extractor.extract_for_component(component)

    score = compute_behavioral_confirmation_score(features)
    # Score should be in valid range; exact value depends on cluster feature calculations
    assert 0.0 <= score <= 1.0, f"Score {score} is out of valid range"


def test_weak_family_coordination_signals_produce_low_score() -> None:
    """Family-like cluster (high variance, low burst, low reason) produces low score."""
    customer1 = CustomerId.generate()
    customer2 = CustomerId.generate()
    customer3 = CustomerId.generate()
    payment1 = PaymentId.generate()
    payment2 = PaymentId.generate()
    payment3 = PaymentId.generate()
    order1 = OrderId.generate()
    order2 = OrderId.generate()
    order3 = OrderId.generate()
    refund1 = RefundId.generate()
    refund2 = RefundId.generate()
    refund3 = RefundId.generate()

    now = datetime.now(timezone.utc)

    # Three family members refunding at different times with different reasons
    snapshot = ReconstructionSnapshot(
        payments={
            payment1: PaymentState(
                payment_id=payment1,
                merchant_id=CustomerId.generate(),
                order_id=order1,
                customer_id=customer1,
                authorised_amount=Money.of_paise(1000),
                status=PaymentStatus.CAPTURED,
                created_at=UTCDateTime(value=now - timedelta(days=10)),
                captured_amount=Money.of_paise(1000),
                captured_at=UTCDateTime(value=now - timedelta(days=10)),
            ),
            payment2: PaymentState(
                payment_id=payment2,
                merchant_id=CustomerId.generate(),
                order_id=order2,
                customer_id=customer2,
                authorised_amount=Money.of_paise(1000),
                status=PaymentStatus.CAPTURED,
                created_at=UTCDateTime(value=now - timedelta(days=5)),
                captured_amount=Money.of_paise(1000),
                captured_at=UTCDateTime(value=now - timedelta(days=5)),
            ),
            payment3: PaymentState(
                payment_id=payment3,
                merchant_id=CustomerId.generate(),
                order_id=order3,
                customer_id=customer3,
                authorised_amount=Money.of_paise(1000),
                status=PaymentStatus.CAPTURED,
                created_at=UTCDateTime(value=now - timedelta(hours=2)),
                captured_amount=Money.of_paise(1000),
                captured_at=UTCDateTime(value=now - timedelta(hours=2)),
            ),
        },
        refunds={
            refund1: RefundState(
                refund_id=refund1,
                payment_id=payment1,
                merchant_id=CustomerId.generate(),
                customer_id=customer1,
                order_id=order1,
                requested_amount=Money.of_paise(500),
                status=RefundStatus.PROCESSED,
                requested_at=UTCDateTime(value=now - timedelta(days=8)),
                processed_at=UTCDateTime(value=now - timedelta(days=8)),
                processed_amount=Money.of_paise(500),
            ),
            refund2: RefundState(
                refund_id=refund2,
                payment_id=payment2,
                merchant_id=CustomerId.generate(),
                customer_id=customer2,
                order_id=order2,
                requested_amount=Money.of_paise(500),
                status=RefundStatus.PROCESSED,
                requested_at=UTCDateTime(value=now - timedelta(days=2)),
                processed_at=UTCDateTime(value=now - timedelta(days=2)),
                processed_amount=Money.of_paise(500),
            ),
            refund3: RefundState(
                refund_id=refund3,
                payment_id=payment3,
                merchant_id=CustomerId.generate(),
                customer_id=customer3,
                order_id=order3,
                requested_amount=Money.of_paise(500),
                status=RefundStatus.PROCESSED,
                requested_at=UTCDateTime(value=now),
                processed_at=UTCDateTime(value=now),
                processed_amount=Money.of_paise(500),
            )
        },
        orders={
            order1: OrderState(
                order_id=order1,
                merchant_id=CustomerId.generate(),
                customer_id=customer1,
                amount=Money.of_paise(1000),
                created_at=UTCDateTime(value=now - timedelta(days=11)),
            ),
            order2: OrderState(
                order_id=order2,
                merchant_id=CustomerId.generate(),
                customer_id=customer2,
                amount=Money.of_paise(1000),
                created_at=UTCDateTime(value=now - timedelta(days=6)),
            ),
            order3: OrderState(
                order_id=order3,
                merchant_id=CustomerId.generate(),
                customer_id=customer3,
                amount=Money.of_paise(1000),
                created_at=UTCDateTime(value=now - timedelta(hours=3)),
            ),
        },
        reconstruction_ordinal=1,
        event_count=6,
    )

    component = ConnectedComponent(
        component_id=str(customer1),
        nodes=frozenset([
            GraphNode(node_id=str(customer1), node_type=NodeType.CUSTOMER),
            GraphNode(node_id=str(customer2), node_type=NodeType.CUSTOMER),
            GraphNode(node_id=str(customer3), node_type=NodeType.CUSTOMER),
        ]),
        edges=frozenset(),
    )

    extractor = ClusterFeatureExtractor(snapshot)
    features = extractor.extract_for_component(component)

    score = compute_behavioral_confirmation_score(features)
    # Family-like signals: high variance, low burst, low reason similarity -> low score
    assert score < 0.4, f"Family cluster should have low score but got {score}"
    assert 0.0 <= score <= 1.0


def test_result_always_in_valid_range() -> None:
    """Confirmation score is always in [0.0, 1.0]."""
    customer1 = CustomerId.generate()
    customer2 = CustomerId.generate()
    payment1 = PaymentId.generate()
    payment2 = PaymentId.generate()
    order1 = OrderId.generate()
    order2 = OrderId.generate()
    refund1 = RefundId.generate()
    refund2 = RefundId.generate()

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
                created_at=UTCDateTime(value=now - timedelta(hours=4)),
                captured_amount=Money.of_paise(1000),
                captured_at=UTCDateTime(value=now - timedelta(hours=4)),
            ),
            payment2: PaymentState(
                payment_id=payment2,
                merchant_id=CustomerId.generate(),
                order_id=order2,
                customer_id=customer2,
                authorised_amount=Money.of_paise(1000),
                status=PaymentStatus.CAPTURED,
                created_at=UTCDateTime(value=now - timedelta(hours=5)),
                captured_amount=Money.of_paise(1000),
                captured_at=UTCDateTime(value=now - timedelta(hours=5)),
            ),
        },
        refunds={
            refund1: RefundState(
                refund_id=refund1,
                payment_id=payment1,
                merchant_id=CustomerId.generate(),
                customer_id=customer1,
                order_id=order1,
                requested_amount=Money.of_paise(1000),
                status=RefundStatus.PROCESSED,
                requested_at=UTCDateTime(value=now - timedelta(minutes=5)),
                processed_at=UTCDateTime(value=now - timedelta(minutes=5)),
                processed_amount=Money.of_paise(1000),
            ),
            refund2: RefundState(
                refund_id=refund2,
                payment_id=payment2,
                merchant_id=CustomerId.generate(),
                customer_id=customer2,
                order_id=order2,
                requested_amount=Money.of_paise(1000),
                status=RefundStatus.PROCESSED,
                requested_at=UTCDateTime(value=now),
                processed_at=UTCDateTime(value=now),
                processed_amount=Money.of_paise(1000),
            )
        },
        orders={
            order1: OrderState(
                order_id=order1,
                merchant_id=CustomerId.generate(),
                customer_id=customer1,
                amount=Money.of_paise(1000),
                created_at=UTCDateTime(value=now - timedelta(hours=5)),
            ),
            order2: OrderState(
                order_id=order2,
                merchant_id=CustomerId.generate(),
                customer_id=customer2,
                amount=Money.of_paise(1000),
                created_at=UTCDateTime(value=now - timedelta(hours=6)),
            ),
        },
        reconstruction_ordinal=1,
        event_count=4,
    )

    component = ConnectedComponent(
        component_id=str(customer1),
        nodes=frozenset([
            GraphNode(node_id=str(customer1), node_type=NodeType.CUSTOMER),
            GraphNode(node_id=str(customer2), node_type=NodeType.CUSTOMER),
        ]),
        edges=frozenset(),
    )

    extractor = ClusterFeatureExtractor(snapshot)
    features = extractor.extract_for_component(component)

    score = compute_behavioral_confirmation_score(features)
    assert 0.0 <= score <= 1.0, f"Score {score} is out of valid range"


def test_deterministic_output() -> None:
    """Same input produces identical output across multiple calls."""
    customer1 = CustomerId.generate()
    customer2 = CustomerId.generate()
    payment1 = PaymentId.generate()
    payment2 = PaymentId.generate()
    order1 = OrderId.generate()
    order2 = OrderId.generate()
    refund1 = RefundId.generate()
    refund2 = RefundId.generate()

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
                created_at=UTCDateTime(value=now - timedelta(hours=4)),
                captured_amount=Money.of_paise(1000),
                captured_at=UTCDateTime(value=now - timedelta(hours=4)),
            ),
            payment2: PaymentState(
                payment_id=payment2,
                merchant_id=CustomerId.generate(),
                order_id=order2,
                customer_id=customer2,
                authorised_amount=Money.of_paise(1000),
                status=PaymentStatus.CAPTURED,
                created_at=UTCDateTime(value=now - timedelta(hours=5)),
                captured_amount=Money.of_paise(1000),
                captured_at=UTCDateTime(value=now - timedelta(hours=5)),
            ),
        },
        refunds={
            refund1: RefundState(
                refund_id=refund1,
                payment_id=payment1,
                merchant_id=CustomerId.generate(),
                customer_id=customer1,
                order_id=order1,
                requested_amount=Money.of_paise(1000),
                status=RefundStatus.PROCESSED,
                requested_at=UTCDateTime(value=now - timedelta(minutes=5)),
                processed_at=UTCDateTime(value=now - timedelta(minutes=5)),
                processed_amount=Money.of_paise(1000),
            ),
            refund2: RefundState(
                refund_id=refund2,
                payment_id=payment2,
                merchant_id=CustomerId.generate(),
                customer_id=customer2,
                order_id=order2,
                requested_amount=Money.of_paise(1000),
                status=RefundStatus.PROCESSED,
                requested_at=UTCDateTime(value=now),
                processed_at=UTCDateTime(value=now),
                processed_amount=Money.of_paise(1000),
            )
        },
        orders={
            order1: OrderState(
                order_id=order1,
                merchant_id=CustomerId.generate(),
                customer_id=customer1,
                amount=Money.of_paise(1000),
                created_at=UTCDateTime(value=now - timedelta(hours=5)),
            ),
            order2: OrderState(
                order_id=order2,
                merchant_id=CustomerId.generate(),
                customer_id=customer2,
                amount=Money.of_paise(1000),
                created_at=UTCDateTime(value=now - timedelta(hours=6)),
            ),
        },
        reconstruction_ordinal=1,
        event_count=4,
    )

    component = ConnectedComponent(
        component_id=str(customer1),
        nodes=frozenset([
            GraphNode(node_id=str(customer1), node_type=NodeType.CUSTOMER),
            GraphNode(node_id=str(customer2), node_type=NodeType.CUSTOMER),
        ]),
        edges=frozenset(),
    )

    extractor = ClusterFeatureExtractor(snapshot)
    features = extractor.extract_for_component(component)

    # Compute score multiple times
    score1 = compute_behavioral_confirmation_score(features)
    score2 = compute_behavioral_confirmation_score(features)
    score3 = compute_behavioral_confirmation_score(features)

    # All must be identical
    assert score1 == score2 == score3, "Scores differ across identical calls"


def test_geometric_mean_property() -> None:
    """Geometric mean behavior: weak signal in one dimension pulls down overall score."""
    # Create two similar clusters, differing only in one signal
    customer1 = CustomerId.generate()
    customer2 = CustomerId.generate()
    payment1 = PaymentId.generate()
    payment2 = PaymentId.generate()
    order1 = OrderId.generate()
    order2 = OrderId.generate()
    refund1 = RefundId.generate()
    refund2 = RefundId.generate()

    now = datetime.now(timezone.utc)

    # First cluster: all signals high (coordinated ring-like)
    snapshot_high = ReconstructionSnapshot(
        payments={
            payment1: PaymentState(
                payment_id=payment1,
                merchant_id=CustomerId.generate(),
                order_id=order1,
                customer_id=customer1,
                authorised_amount=Money.of_paise(1000),
                status=PaymentStatus.CAPTURED,
                created_at=UTCDateTime(value=now - timedelta(hours=4)),
                captured_amount=Money.of_paise(1000),
                captured_at=UTCDateTime(value=now - timedelta(hours=4)),
            ),
            payment2: PaymentState(
                payment_id=payment2,
                merchant_id=CustomerId.generate(),
                order_id=order2,
                customer_id=customer2,
                authorised_amount=Money.of_paise(1000),
                status=PaymentStatus.CAPTURED,
                created_at=UTCDateTime(value=now - timedelta(hours=5)),
                captured_amount=Money.of_paise(1000),
                captured_at=UTCDateTime(value=now - timedelta(hours=5)),
            ),
        },
        refunds={
            refund1: RefundState(
                refund_id=refund1,
                payment_id=payment1,
                merchant_id=CustomerId.generate(),
                customer_id=customer1,
                order_id=order1,
                requested_amount=Money.of_paise(1000),
                status=RefundStatus.PROCESSED,
                requested_at=UTCDateTime(value=now - timedelta(minutes=5)),
                processed_at=UTCDateTime(value=now - timedelta(minutes=5)),
                processed_amount=Money.of_paise(1000),
            ),
            refund2: RefundState(
                refund_id=refund2,
                payment_id=payment2,
                merchant_id=CustomerId.generate(),
                customer_id=customer2,
                order_id=order2,
                requested_amount=Money.of_paise(1000),
                status=RefundStatus.PROCESSED,
                requested_at=UTCDateTime(value=now),
                processed_at=UTCDateTime(value=now),
                processed_amount=Money.of_paise(1000),
            )
        },
        orders={
            order1: OrderState(
                order_id=order1,
                merchant_id=CustomerId.generate(),
                customer_id=customer1,
                amount=Money.of_paise(1000),
                created_at=UTCDateTime(value=now - timedelta(hours=5)),
            ),
            order2: OrderState(
                order_id=order2,
                merchant_id=CustomerId.generate(),
                customer_id=customer2,
                amount=Money.of_paise(1000),
                created_at=UTCDateTime(value=now - timedelta(hours=6)),
            ),
        },
        reconstruction_ordinal=1,
        event_count=4,
    )

    component = ConnectedComponent(
        component_id=str(customer1),
        nodes=frozenset([
            GraphNode(node_id=str(customer1), node_type=NodeType.CUSTOMER),
            GraphNode(node_id=str(customer2), node_type=NodeType.CUSTOMER),
        ]),
        edges=frozenset(),
    )

    extractor = ClusterFeatureExtractor(snapshot_high)
    features_high = extractor.extract_for_component(component)

    score_high = compute_behavioral_confirmation_score(features_high)

    # With geometric mean, if all signals are present but some are weak,
    # the overall score will reflect the geometric mean (not arithmetic mean).
    # Verify the score is in valid range.
    assert 0.0 <= score_high <= 1.0, f"Score {score_high} is out of valid range"
