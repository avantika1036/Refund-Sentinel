"""Unit tests for the Evidence Bundle and Investigation Explanation service."""

from __future__ import annotations

import pytest

from backend.app.finance.state_engine import FinancialStateEngine
from backend.app.investigator.evidence import EvidenceBundleBuilder
from backend.app.investigator.explanation import InvestigationExplanationService
from backend.app.risk.investigation import InvestigationService
from backend.app.simulator.scenarios import AS01_DENSE_COORDINATED_REFUND_RING


def test_evidence_bundle_builder_and_explanation():
    """Verify EvidenceBundle builds correctly and explanation generates narrative."""
    # Generate scenario
    scenario = AS01_DENSE_COORDINATED_REFUND_RING(seed=42).generate()
    events = scenario.events
    snapshot = FinancialStateEngine().reconstruct_from(events)

    assert len(snapshot.refunds) > 0

    refund_id = next(iter(snapshot.refunds.keys()))
    inv_service = InvestigationService(snapshot)
    investigation = inv_service.investigate(refund_id)

    bundle = EvidenceBundleBuilder.build(
        snapshot=snapshot,
        refund_id=refund_id,
        assessment=investigation.assessment,
        decision=investigation.decision,
        exposure=investigation.exposure,
        component_refund_ids=investigation.component_refund_ids,
    )

    assert bundle.refund_id == str(refund_id)
    assert bundle.customer_profile.customer_id == str(snapshot.refunds[refund_id].customer_id)
    assert bundle.financial_exposure.requested_amount_paise > 0
    assert bundle.final_risk_score >= 0.0
    assert bundle.graph_topology.cluster_size >= 1
    assert len(bundle.feature_contributions) > 0

    # Test explanation service fallback
    explanation_service = InvestigationExplanationService()
    explanation = explanation_service.explain(bundle)

    assert len(explanation.headline) > 0
    assert len(explanation.narrative_summary) > 0
    assert len(explanation.key_risk_drivers) > 0
    assert len(explanation.suggested_action_rationale) > 0
    assert explanation.is_llm_generated is False
