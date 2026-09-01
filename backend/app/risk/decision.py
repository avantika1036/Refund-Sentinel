"""Risk decision classification.

Transforms a numerical risk assessment into a deterministic, explainable
decision suitable for application APIs and user interfaces.

Risk levels are prioritization categories, not fraud determinations.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from backend.app.domain.identifiers import RefundId
from backend.app.risk.assessment import RiskAssessment
from backend.app.risk.rules import RuleId


class RiskLevel(str, Enum):
    """Human-readable risk prioritization levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class DecisionAction(str, Enum):
    """Recommended operational actions."""

    ALLOW = "allow"
    REVIEW = "review"
    INVESTIGATE = "investigate"


@dataclass(frozen=True)
class RiskDecision:
    """Application-level decision derived from a risk assessment.

    This object intentionally preserves the distinction between:

    - a numerical prioritization score
    - an operational recommendation
    - the underlying evidence

    It does not claim that a refund is fraudulent.
    """

    refund_id: RefundId
    final_score: float
    risk_level: RiskLevel
    action: DecisionAction

    triggered_rule_ids: tuple[RuleId, ...]

    behavioral_confirmation_score: float

    explanation: str


class RiskDecisionEngine:
    """Classify risk assessments into deterministic decisions.

    Thresholds intentionally remain centralized here so that the scoring
    layer continues to compute evidence while this layer owns operational
    interpretation.
    """

    HIGH_RISK_THRESHOLD = 0.75
    MEDIUM_RISK_THRESHOLD = 0.40

    def decide(
        self,
        assessment: RiskAssessment,
    ) -> RiskDecision:
        """Create an explainable decision from a risk assessment."""

        final_score = assessment.risk_score.final_score

        risk_level = self._classify_risk_level(final_score)

        action = self._select_action(risk_level)

        triggered_rule_ids = tuple(
            output.rule_id
            for output in assessment.rule_outputs
            if output.triggered
        )

        explanation = self._build_explanation(
            assessment=assessment,
            risk_level=risk_level,
            action=action,
            triggered_rule_ids=triggered_rule_ids,
        )

        return RiskDecision(
            refund_id=assessment.refund_id,
            final_score=final_score,
            risk_level=risk_level,
            action=action,
            triggered_rule_ids=triggered_rule_ids,
            behavioral_confirmation_score=(
                assessment.behavioral_confirmation_score
            ),
            explanation=explanation,
        )

    def _classify_risk_level(
        self,
        final_score: float,
    ) -> RiskLevel:
        """Classify a numerical score into a risk level."""

        if final_score >= self.HIGH_RISK_THRESHOLD:
            return RiskLevel.HIGH

        if final_score >= self.MEDIUM_RISK_THRESHOLD:
            return RiskLevel.MEDIUM

        return RiskLevel.LOW

    def _select_action(
        self,
        risk_level: RiskLevel,
    ) -> DecisionAction:
        """Map risk levels to recommended actions."""

        if risk_level is RiskLevel.HIGH:
            return DecisionAction.INVESTIGATE

        if risk_level is RiskLevel.MEDIUM:
            return DecisionAction.REVIEW

        return DecisionAction.ALLOW

    def _build_explanation(
        self,
        *,
        assessment: RiskAssessment,
        risk_level: RiskLevel,
        action: DecisionAction,
        triggered_rule_ids: tuple[RuleId, ...],
    ) -> str:
        """Build a concise deterministic explanation."""

        score = assessment.risk_score.final_score

        if triggered_rule_ids:
            rule_summary = ", ".join(
                rule_id.value
                for rule_id in triggered_rule_ids
            )
        else:
            rule_summary = "no deterministic rules triggered"

        return (
            f"Refund {assessment.refund_id} received a "
            f"{risk_level.value} risk classification "
            f"with score {score:.3f}. "
            f"Recommended action: {action.value}. "
            f"Rule evidence: {rule_summary}. "
            f"Behavioral confirmation score: "
            f"{assessment.behavioral_confirmation_score:.3f}."
        )