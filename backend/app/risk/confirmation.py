"""Behavioral confirmation gate for cluster-level scoring.

Evaluates whether behavioral evidence indicates genuine coordination within
a cluster before allowing cluster features to influence the risk score.
"""

from backend.app.risk.features.cluster import ClusterFeatures


def compute_behavioral_confirmation_score(cluster_features: ClusterFeatures) -> float:
    """Compute behavioral confirmation score from coordination signals.

    The behavioral confirmation gate measures whether cluster members exhibit
    genuine behavioral coordination. A high score indicates members are likely
    acting in coordinated fashion (e.g., timing their refunds similarly, using
    the same reasons, refunding in concentrated bursts).

    Args:
        cluster_features: ClusterFeatures containing the three coordination signals.

    Returns:
        float: Score in [0.0, 1.0]. Higher values indicate stronger coordination evidence.
               Returns 0.0 for singletons (no coordination possible) or when
               any coordination signal is zero or missing.

    Coordination signals used:
    - cluster_lifecycle_timing_alignment: [0, 1] - how aligned refund latencies are
    - cluster_temporal_burst_score: [0, 1] - refund concentration in time
    - cluster_reason_similarity: [0, 1] - similarity of refund reason codes

    Formula: Geometric mean of the three signals
    score = (alignment * burst * reason) ** (1 / 3)

    Rationale:
    Geometric mean enforces conjunctive logic: all three signals must be present
    for the score to be high. Weak signals in any dimension produce low overall
    score, preventing compensatory scoring where one strong signal masks weak
    coordination evidence in other dimensions.
    """
    # Singletons cannot demonstrate coordination
    if cluster_features.cluster_size <= 1:
        return 0.0

    alignment = cluster_features.cluster_lifecycle_timing_alignment
    burst = cluster_features.cluster_temporal_burst_score
    reason = cluster_features.cluster_reason_similarity

    # Geometric mean requires all signals to be present
    product = alignment * burst * reason

    # If any signal is zero (or product underflows), coordination is absent
    if product <= 0.0:
        return 0.0

    # Cube root: (x)^(1/3)
    score = product ** (1.0 / 3.0)

    # Ensure result is in valid range [0.0, 1.0]
    # This is guaranteed by the input ranges and geometric mean properties,
    # but we check explicitly for safety.
    return min(max(score, 0.0), 1.0)
