"""Unit tests for risk score composition."""

from backend.app.risk.features.cluster import ClusterFeatures
from backend.app.risk.rules import (
    EvidenceType,
    RuleId,
    RuleOutput,
)
from backend.app.risk.scoring import compute_risk_score


def make_rule_output(
    *,
    rule_id: RuleId,
    triggered: bool,
    weight: float,
) -> RuleOutput:
    """Create a minimal rule output for scoring tests."""
    return RuleOutput(
        rule_id=rule_id,
        triggered=triggered,
        evidence_type=EvidenceType.BOOLEAN,
        evidence_value=triggered,
        evidence_threshold=None,
        base_signal_weight=weight,
        notes="test",
    )


def make_cluster_features(
    *,
    cluster_size: int = 3,
    alignment: float = 0.0,
    burst: float = 0.0,
    reason: float = 0.0,
) -> ClusterFeatures:
    """Create minimal cluster features for scoring tests."""
    return ClusterFeatures(
        cluster_size=cluster_size,
        cluster_refund_active_fraction=1.0,
        cluster_lifecycle_timing_alignment=alignment,
        cluster_temporal_burst_score=burst,
        cluster_reason_similarity=reason,
        cluster_amount_concentration=1.0,
    )


def test_rules_only_score() -> None:
    """Risk score reflects rule evidence without cluster features."""
    outputs = [
        make_rule_output(
            rule_id=RuleId.R01_RAPID_REFUND_AFTER_CAPTURE,
            triggered=True,
            weight=0.5,
        ),
        make_rule_output(
            rule_id=RuleId.R02_REFUND_BEFORE_DELIVERY,
            triggered=False,
            weight=0.5,
        ),
    ]

    result = compute_risk_score(outputs)

    assert result.rule_signal_component == 0.5
    assert result.behavioral_confirmation_score == 0.0
    assert result.cluster_signal_component == 0.0
    assert result.final_score == 0.5


def test_empty_rules_and_no_cluster_produces_zero() -> None:
    """No evidence produces zero risk score."""
    result = compute_risk_score([])

    assert result.final_score == 0.0


def test_coordinated_cluster_contributes_to_score() -> None:
    """Strong coordination contributes through the cluster pathway."""
    cluster_features = make_cluster_features(
        alignment=0.9,
        burst=0.9,
        reason=0.9,
    )

    result = compute_risk_score(
        [],
        cluster_features,
    )

    assert result.rule_signal_component == 0.0
    assert result.behavioral_confirmation_score > 0.8
    assert result.cluster_signal_component > 0.8
    assert result.final_score > 0.8


def test_unconfirmed_cluster_does_not_increase_score() -> None:
    """Weak coordination does not create a strong cluster risk signal."""
    cluster_features = make_cluster_features(
        alignment=0.0,
        burst=0.9,
        reason=0.9,
    )

    result = compute_risk_score(
        [],
        cluster_features,
    )

    assert result.behavioral_confirmation_score == 0.0
    assert result.cluster_signal_component == 0.0
    assert result.final_score == 0.0


def test_stronger_evidence_path_determines_final_score() -> None:
    """Final score preserves the strongest evidence pathway."""
    outputs = [
        make_rule_output(
            rule_id=RuleId.R01_RAPID_REFUND_AFTER_CAPTURE,
            triggered=True,
            weight=0.5,
        ),
        make_rule_output(
            rule_id=RuleId.R02_REFUND_BEFORE_DELIVERY,
            triggered=False,
            weight=0.5,
        ),
    ]

    cluster_features = make_cluster_features(
        alignment=0.9,
        burst=0.9,
        reason=0.9,
    )

    result = compute_risk_score(
        outputs,
        cluster_features,
    )

    assert result.rule_signal_component == 0.5
    assert result.behavioral_confirmation_score > 0.8
    assert result.final_score == result.cluster_signal_component


def test_final_score_is_always_in_valid_range() -> None:
    """Final score remains normalized."""
    outputs = [
        make_rule_output(
            rule_id=RuleId.R01_RAPID_REFUND_AFTER_CAPTURE,
            triggered=True,
            weight=0.3,
        ),
        make_rule_output(
            rule_id=RuleId.R02_REFUND_BEFORE_DELIVERY,
            triggered=True,
            weight=0.7,
        ),
    ]

    cluster_features = make_cluster_features(
        alignment=1.0,
        burst=1.0,
        reason=1.0,
    )

    result = compute_risk_score(
        outputs,
        cluster_features,
    )

    assert 0.0 <= result.final_score <= 1.0