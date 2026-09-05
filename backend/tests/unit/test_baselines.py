"""Unit tests for ML Baselines A, B, and C."""

from __future__ import annotations

import pytest

from backend.app.ml.baselines import (
    BaselineAIndividualOnly,
    BaselineBGraphHeuristicOnly,
    BaselineCFullSystem,
)
from backend.app.ml.dataset import create_dataset


def test_baseline_models():
    """Verify Baseline A, Baseline B, and Baseline C fit and predict properly."""
    feature_names = [
        "individual_velocity",
        "individual_refund_rate",
        "cluster_cluster_size",
        "relationship_shared_attribute_type_count",
    ]

    # Mock dataset with abuse and legit patterns
    rows = [
        [10.0, 0.9, 5.0, 2.0],
        [8.0, 0.85, 4.0, 1.0],
        [1.0, 0.05, 1.0, 0.0],
        [2.0, 0.1, 1.0, 0.0],
    ]
    labels = [1, 1, 0, 0]

    dataset = create_dataset(
        feature_names=feature_names,
        feature_rows=rows,
        labels=labels,
    )

    # 1. Baseline A
    base_a = BaselineAIndividualOnly()
    base_a.fit(dataset)
    p_a = base_a.predict_proba(rows[0])
    assert 0.0 <= p_a <= 1.0

    # 2. Baseline B
    base_b = BaselineBGraphHeuristicOnly(cluster_threshold=2)
    is_abuse_b = base_b.predict_is_abuse(rows[0], feature_names)
    assert is_abuse_b is True
    is_abuse_b_legit = base_b.predict_is_abuse(rows[2], feature_names)
    assert is_abuse_b_legit is False

    # 3. Baseline C
    base_c = BaselineCFullSystem()
    base_c.fit(dataset)
    p_c = base_c.predict_proba(rows[0])
    assert 0.0 <= p_c <= 1.0
