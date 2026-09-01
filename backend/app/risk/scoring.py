"""Risk score composition.

Combines deterministic rule evidence with behavioral coordination evidence.

The resulting score is a risk prioritization signal in the range [0.0, 1.0].
It is not a calibrated fraud probability.
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.app.risk.confirmation import compute_behavioral_confirmation_score
from backend.app.risk.features.cluster import ClusterFeatures
from backend.app.risk.rules import RuleOutput, compute_rule_signal_component


@dataclass(frozen=True)
class RiskScore:
    """Structured result of risk score computation.

    Attributes:
        rule_signal_component:
            Fraction of deterministic rule signal strength that fired.

        behavioral_confirmation_score:
            Strength of behavioral coordination evidence within the cluster.

        cluster_signal_component:
            Cluster evidence allowed to contribute after behavioral
            confirmation is applied.

        final_score:
            Final risk prioritization score in [0.0, 1.0].

    IMPORTANT:
        final_score is not a fraud probability.
    """

    rule_signal_component: float
    behavioral_confirmation_score: float
    cluster_signal_component: float
    final_score: float


def compute_risk_score(
    rule_outputs: list[RuleOutput],
    cluster_features: ClusterFeatures | None = None,
) -> RiskScore:
    """Compute the final risk score.

    The score combines two evidence layers:

    1. Individual deterministic rule evidence.
    2. Cluster behavioral coordination evidence.

    Cluster evidence is gated by behavioral confirmation. This prevents
    structural cluster membership alone from increasing risk when the
    cluster does not exhibit coordinated behavior.

    Formula:

        final_score = max(
            rule_signal_component,
            cluster_signal_component,
        )

    The maximum is used intentionally because these are independent
    evidence pathways rather than additive fraud probabilities.

    Args:
        rule_outputs:
            Outputs from deterministic rule evaluation.

        cluster_features:
            Optional behavioral features for the refund's connected
            component.

    Returns:
        RiskScore containing all intermediate components and the final score.
    """

    rule_signal_component = compute_rule_signal_component(rule_outputs)

    behavioral_confirmation_score = 0.0
    cluster_signal_component = 0.0

    if cluster_features is not None:
        behavioral_confirmation_score = (
            compute_behavioral_confirmation_score(cluster_features)
        )

        # Cluster evidence can only contribute when behavioral
        # coordination is actually present.
        cluster_signal_component = behavioral_confirmation_score

    final_score = max(
        rule_signal_component,
        cluster_signal_component,
    )

    return RiskScore(
        rule_signal_component=rule_signal_component,
        behavioral_confirmation_score=behavioral_confirmation_score,
        cluster_signal_component=cluster_signal_component,
        final_score=final_score,
    )