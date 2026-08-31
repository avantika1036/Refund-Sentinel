"""Risk analysis module for Refund Sentinel.

This module implements deterministic feature extraction and rule-based
evidence signals for fraud detection.

IMPORTANT:
- Features measure observable behavior
- Rules produce deterministic evidence signals
- Neither features nor rules claim that fraud is proven
- This layer does NOT include ML, final scoring, or LLM/evidence bundles
- Ground-truth labels are NOT used in feature/rule computation

"""

from backend.app.risk.features import ClusterFeatures, IndividualFeatures, RelationshipFeatures
from backend.app.risk.rules import (
    EvidenceType,
    RuleEngine,
    RuleId,
    RuleOutput,
    compute_rule_signal_component,
)

__all__ = [
    "ClusterFeatures",
    "IndividualFeatures",
    "RelationshipFeatures",
    "RuleEngine",
    "RuleId",
    "RuleOutput",
    "EvidenceType",
    "compute_rule_signal_component",
]
