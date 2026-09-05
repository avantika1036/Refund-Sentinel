"""Tests for medium-priority evaluation methodology improvements."""

from collections import Counter

from backend.app.simulator.labels import LabelClassification
from scripts.generate_datasets import generate_partition


def test_heldout_partition_uses_unseen_abuse_and_legitimate_families() -> None:
    rows, summary = generate_partition(
        num_background_customers=5,
        include_held_out=True,
        seed=999,
    )

    scenario_counts = Counter(row["scenario"] for row in rows)

    assert summary["abuse_count"] > 0
    assert scenario_counts["AS03_SHARED_PAYMENT_DEVICE_RING"] > 0
    assert scenario_counts["AS04_ISOLATED_REFUND_CHURN"] > 0
    assert scenario_counts["LL02_FREQUENT_SHOPPER"] > 0
    assert scenario_counts["LL03_SHARED_HOUSEHOLD"] > 0

    # Training-only families must not leak into the held-out partition.
    assert "AS01_DENSE_COORDINATED_REFUND_RING" not in scenario_counts
    assert "AS02_VELOCITY_REFUND_ABUSE" not in scenario_counts
    assert "LL01_LEGITIMATE_FAMILY" not in scenario_counts


def test_partition_generators_use_distinct_event_id_streams() -> None:
    # The partition should preserve all scenario events rather than allowing
    # deterministic generator streams with the same seed to collide.
    rows, _ = generate_partition(
        num_background_customers=5,
        include_held_out=False,
        seed=42,
    )
    refund_ids = [row["refund_id"] for row in rows]
    assert len(refund_ids) == len(set(refund_ids))
