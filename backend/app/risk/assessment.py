"""End-to-end risk assessment orchestration.

Connects feature extraction, deterministic rules, behavioral confirmation,
and final risk score composition for a single refund.
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.app.domain.identifiers import CustomerId, RefundId
from backend.app.finance.types import ReconstructionSnapshot
from backend.app.graph.builder import StructuralGraphBuilder
from backend.app.graph.components import (
    ConnectedComponent,
    ConnectedComponentExtractor,
)
from backend.app.graph.model import NodeType
from backend.app.risk.confirmation import compute_behavioral_confirmation_score
from backend.app.risk.features.cluster import (
    ClusterFeatureExtractor,
    ClusterFeatures,
)
from backend.app.risk.features.individual import (
    IndividualFeatureExtractor,
    IndividualFeatures,
)
from backend.app.risk.features.relationship import (
    RelationshipFeatureExtractor,
    RelationshipFeatures,
)
from backend.app.risk.rules import (
    RuleEngine,
    RuleOutput,
)
from backend.app.risk.scoring import RiskScore, compute_risk_score


@dataclass(frozen=True)
class RiskAssessment:
    """Complete risk assessment for a single refund.

    Contains the features, rule evidence, behavioral confirmation,
    and final risk score produced during assessment.
    """

    refund_id: RefundId
    customer_id: CustomerId
    component_id: str

    individual_features: IndividualFeatures
    cluster_features: ClusterFeatures
    relationship_features: RelationshipFeatures

    rule_outputs: tuple[RuleOutput, ...]
    behavioral_confirmation_score: float
    risk_score: RiskScore


class RiskAssessor:
    """Orchestrates end-to-end risk assessment for refunds.

    The assessor is intentionally a composition layer. It does not
    reimplement feature, rule, confirmation, or scoring logic.

    Pipeline:

        ReconstructionSnapshot
                ↓
        Structural Graph
                ↓
        Connected Components
                ↓
        Feature Extraction
                ↓
        R01-R05 Individual Rules
                ↓
        R06 Cluster Rule
                ↓
        Behavioral Confirmation
                ↓
        Final Risk Score
    """

    def __init__(self, snapshot: ReconstructionSnapshot) -> None:
        self._snapshot = snapshot

        graph = StructuralGraphBuilder().build(snapshot)
        self._components = ConnectedComponentExtractor().extract(graph)

        self._individual_feature_extractor = IndividualFeatureExtractor(snapshot)
        self._cluster_feature_extractor = ClusterFeatureExtractor(snapshot)
        self._relationship_feature_extractor = RelationshipFeatureExtractor(
            snapshot
        )
        self._rule_engine = RuleEngine(snapshot)

    def assess(self, refund_id: RefundId) -> RiskAssessment:
        """Perform a complete risk assessment for one refund.

        Args:
            refund_id: Identifier of the refund to assess.

        Returns:
            Complete RiskAssessment.

        Raises:
            ValueError: If the refund does not exist or cannot be associated
                with a graph component.
        """

        refund_state = self._snapshot.refunds.get(refund_id)

        if refund_state is None:
            raise ValueError(
                f"Refund {refund_id} not found in snapshot"
            )

        customer_id = refund_state.customer_id

        component = self._find_component_for_customer(customer_id)

        individual_features = (
            self._individual_feature_extractor.extract_for_refund(refund_id)
        )

        cluster_features = (
            self._cluster_feature_extractor.extract_for_component(component)
        )

        relationship_features = (
            self._relationship_feature_extractor.extract_for_component(
                component
            )
        )

        rule_outputs = self._evaluate_rules_for_refund(
            refund_id=refund_id,
            component=component,
            target_customer_id=customer_id,
            target_individual_features=individual_features,
        )

        behavioral_confirmation_score = (
            compute_behavioral_confirmation_score(cluster_features)
        )

        risk_score = compute_risk_score(
            rule_outputs=list(rule_outputs),
            cluster_features=cluster_features,
        )

        return RiskAssessment(
            refund_id=refund_id,
            customer_id=customer_id,
            component_id=component.component_id,
            individual_features=individual_features,
            cluster_features=cluster_features,
            relationship_features=relationship_features,
            rule_outputs=tuple(rule_outputs),
            behavioral_confirmation_score=behavioral_confirmation_score,
            risk_score=risk_score,
        )

    def _find_component_for_customer(
        self,
        customer_id: CustomerId,
    ) -> ConnectedComponent:
        """Find the connected component containing a customer."""

        customer_node_id = str(customer_id)

        for component in self._components:
            if component.contains_node(customer_node_id):
                return component

        raise ValueError(
            f"Customer {customer_id} does not belong to any graph component"
        )

    def _evaluate_rules_for_refund(
        self,
        *,
        refund_id: RefundId,
        component: ConnectedComponent,
        target_customer_id: CustomerId,
        target_individual_features: IndividualFeatures,
    ) -> list[RuleOutput]:
        """Evaluate R01-R06 for the target refund.

        R01-R05 are evaluated directly for the target refund.

        R06 depends on whether multiple distinct customers in the same
        component have triggered individual rules, so individual rules
        must also be evaluated for the other refunding customers first.
        """

        refund_state = self._snapshot.refunds[refund_id]

        payment_state = self._snapshot.payments.get(
            refund_state.payment_id
        )

        order_state = self._snapshot.orders.get(
            refund_state.order_id
        )

        if payment_state is None:
            raise ValueError(
                f"Payment {refund_state.payment_id} "
                f"for refund {refund_id} not found"
            )

        if order_state is None:
            raise ValueError(
                f"Order {refund_state.order_id} "
                f"for refund {refund_id} not found"
            )

        # Evaluate R01-R05 for the target refund.
        target_outputs = self._evaluate_individual_rules(
            refund_id=refund_id,
            individual_features=target_individual_features,
        )

        # R06 requires rule results for distinct customers in the
        # connected component.
        member_rule_outputs = (
            self._evaluate_component_member_rules(
                component=component,
                target_customer_id=target_customer_id,
                target_refund_id=refund_id,
                target_outputs=target_outputs,
            )
        )

        r06_output = self._rule_engine.evaluate_r06_cluster_flags(
            component,
            member_rule_outputs,
        )

        return [
            *target_outputs,
            r06_output,
        ]

    def _evaluate_individual_rules(
        self,
        *,
        refund_id: RefundId,
        individual_features: IndividualFeatures,
    ) -> list[RuleOutput]:
        """Evaluate R01-R05 for one refund."""

        refund_state = self._snapshot.refunds[refund_id]

        payment_state = self._snapshot.payments.get(
            refund_state.payment_id
        )

        order_state = self._snapshot.orders.get(
            refund_state.order_id
        )

        if payment_state is None:
            raise ValueError(
                f"Payment for refund {refund_id} not found"
            )

        if order_state is None:
            raise ValueError(
                f"Order for refund {refund_id} not found"
            )

        return [
            self._rule_engine.evaluate_r01_rapid_refund(
                individual_features,
                payment_state,
            ),
            self._rule_engine.evaluate_r02_refund_before_delivery(
                individual_features,
                order_state.delivered_at,
            ),
            self._rule_engine.evaluate_r03_full_refund(
                individual_features,
                payment_state,
            ),
            self._rule_engine.evaluate_r04_refund_rate_anomaly(
                individual_features,
            ),
            self._rule_engine.evaluate_r05_refund_velocity_spike(
                individual_features,
            ),
        ]

    def _evaluate_component_member_rules(
        self,
        *,
        component: ConnectedComponent,
        target_customer_id: CustomerId,
        target_refund_id: RefundId,
        target_outputs: list[RuleOutput],
    ) -> dict[CustomerId, list[RuleOutput]]:
        """Evaluate individual rule evidence for component members.

        R06 operates at the customer level, not the refund-event level.
        A customer is considered flagged when at least one of their
        evaluated individual rules has triggered.

        For customers with multiple refunds, the first refund in
        deterministic order is used as the representative assessment.
        """

        customer_ids = [
            CustomerId.from_str(node.node_id)
            for node in component.nodes
            if node.node_type == NodeType.CUSTOMER
        ]

        member_rule_outputs: dict[
            CustomerId,
            list[RuleOutput],
        ] = {}

        for customer_id in customer_ids:
            if customer_id == target_customer_id:
                member_rule_outputs[customer_id] = target_outputs
                continue

            customer_refunds = sorted(
                [
                    refund
                    for refund in self._snapshot.refunds.values()
                    if refund.customer_id == customer_id
                ],
                key=lambda refund: (
                    refund.requested_at.value,
                    str(refund.refund_id),
                ),
            )

            if not customer_refunds:
                member_rule_outputs[customer_id] = []
                continue

            representative_refund = customer_refunds[0]

            member_features = (
                self._individual_feature_extractor.extract_for_refund(
                    representative_refund.refund_id
                )
            )

            member_rule_outputs[customer_id] = (
                self._evaluate_individual_rules(
                    refund_id=representative_refund.refund_id,
                    individual_features=member_features,
                )
            )

        return member_rule_outputs