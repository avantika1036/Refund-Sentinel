"""Case-level investigation orchestration.

Combines risk assessment, operational decision, and financial exposure
into a single investigation result for a refund and its connected
customer component.
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.app.domain.identifiers import RefundId
from backend.app.finance.exposure import (
    FinancialExposure,
    compute_financial_exposure,
)
from backend.app.finance.types import ReconstructionSnapshot
from backend.app.graph.builder import StructuralGraphBuilder
from backend.app.graph.components import (
    ConnectedComponent,
    ConnectedComponentExtractor,
)
from backend.app.graph.model import NodeType
from backend.app.risk.assessment import (
    RiskAssessment,
    RiskAssessor,
)
from backend.app.risk.decision import (
    RiskDecision,
    RiskDecisionEngine,
)


@dataclass(frozen=True)
class Investigation:
    """Complete investigation result for a refund.

    Combines:

    - risk assessment and underlying evidence
    - operational risk decision
    - financial exposure across the connected component

    The investigation result is intended to provide a complete,
    explainable view of why a refund should be prioritized and what
    financial exposure may be associated with the related component.
    """

    assessment: RiskAssessment
    decision: RiskDecision
    exposure: FinancialExposure

    component_refund_ids: tuple[RefundId, ...]


class InvestigationService:
    """Build complete investigations from reconstructed system state.

    This service is a composition layer. It intentionally delegates
    feature extraction, risk assessment, decision classification, and
    financial exposure calculation to their existing specialized
    components.
    """

    def __init__(self, snapshot: ReconstructionSnapshot) -> None:
        self._snapshot = snapshot

        graph = StructuralGraphBuilder().build(snapshot)

        self._components = (
            ConnectedComponentExtractor().extract(graph)
        )

        self._risk_assessor = RiskAssessor(snapshot)

        self._decision_engine = RiskDecisionEngine()

    def investigate(
        self,
        refund_id: RefundId,
    ) -> Investigation:
        """Perform a complete investigation for one refund.

        The investigation assesses the target refund, derives an
        operational decision, and calculates risk-weighted financial
        exposure across all refunds belonging to customers in the same
        connected component.

        Args:
            refund_id:
                Identifier of the refund to investigate.

        Returns:
            Complete Investigation result.

        Raises:
            ValueError:
                If the refund does not exist or its customer cannot be
                associated with a connected component.
        """

        if refund_id not in self._snapshot.refunds:
            raise ValueError(
                f"Refund {refund_id} not found in snapshot"
            )

        assessment = self._risk_assessor.assess(refund_id)

        decision = self._decision_engine.decide(assessment)

        component = self._find_component(
            assessment.customer_id
        )

        component_refund_ids = (
            self._find_component_refunds(component)
        )

        exposure = compute_financial_exposure(
            snapshot=self._snapshot,
            refund_ids=component_refund_ids,
            risk_score=assessment.risk_score.final_score,
        )

        return Investigation(
            assessment=assessment,
            decision=decision,
            exposure=exposure,
            component_refund_ids=component_refund_ids,
        )

    def _find_component(
        self,
        customer_id,
    ) -> ConnectedComponent:
        """Find the connected component containing a customer."""

        customer_node_id = str(customer_id)

        for component in self._components:
            if component.contains_node(customer_node_id):
                return component

        raise ValueError(
            f"Customer {customer_id} does not belong "
            f"to any graph component"
        )

    def _find_component_refunds(
        self,
        component: ConnectedComponent,
    ) -> tuple[RefundId, ...]:
        """Find all refunds belonging to customers in a component.

        Refunds are returned in deterministic chronological order.
        """

        customer_ids = {
            node.node_id
            for node in component.nodes
            if node.node_type == NodeType.CUSTOMER
        }

        component_refunds = [
            refund
            for refund in self._snapshot.refunds.values()
            if str(refund.customer_id) in customer_ids
        ]

        component_refunds.sort(
            key=lambda refund: (
                refund.requested_at.value,
                str(refund.refund_id),
            )
        )

        return tuple(
            refund.refund_id
            for refund in component_refunds
        )