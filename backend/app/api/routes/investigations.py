"""Complete investigation API routes."""

from __future__ import annotations

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    status,
)
from sqlalchemy.orm import Session

from backend.app.api.schemas import (
    AssessmentResponse,
    CustomerProfileResponse,
    EvidenceBundleResponse,
    FeatureContributionResponse,
    FinancialExposureResponse,
    GraphTopologyResponse,
    InvestigationExplanationResponse,
    InvestigationQueueResponse,
    InvestigationResponse,
    MLPredictionResponse,
    QueueCaseResponse,
    QueueMetricsResponse,
    RiskScoreResponse,
    RuleEvidenceResponse,
)
from backend.app.api.security import require_api_key
from backend.app.domain.identifiers import RefundId
from backend.app.persistence.database import get_db
from backend.app.persistence.reconstruction import ReconstructionService
from backend.app.finance.exposure import compute_financial_exposure
from backend.app.investigator.evidence import EvidenceBundleBuilder
from backend.app.investigator.explanation import InvestigationExplanationService
from backend.app.risk.investigation import InvestigationService


router = APIRouter(
    prefix="/api/v1/investigations",
    tags=["investigations"],
    dependencies=[Depends(require_api_key)],
)


@router.get(
    "",
    response_model=InvestigationQueueResponse,
)
def get_investigation_queue(
    request: Request,
    db: Session = Depends(get_db),
) -> InvestigationQueueResponse:
    """Return every reconstructable refund, sorted for analyst triage.

    The priority order is deliberately transparent and stable:
    deterministic final risk score, optional ML probability, triggered-rule
    count, connected-refund count, then refund ID. Queue exposure is computed
    for each individual refund so aggregate metrics do not double-count
    connected components.
    """

    snapshot = ReconstructionService(db).reconstruct()
    investigation_service = InvestigationService(
        snapshot,
        ml_inference_service=request.app.state.ml_inference_service,
    )
    cases: list[QueueCaseResponse] = []

    for refund_id, refund in snapshot.refunds.items():
        try:
            investigation = investigation_service.investigate(refund_id)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Could not assess refund {refund_id}",
            ) from exc

        assessment = investigation.assessment
        decision = investigation.decision
        case_exposure = compute_financial_exposure(
            snapshot=snapshot,
            refund_ids=[refund_id],
            risk_score=assessment.risk_score.final_score,
        )
        cases.append(
            QueueCaseResponse(
                refund_id=str(refund_id),
                customer_id=str(refund.customer_id),
                component_id=assessment.component_id,
                status=refund.status.value,
                requested_at=refund.requested_at.value.isoformat(),
                requested_amount_paise=refund.requested_amount.amount_paise,
                risk_level=decision.risk_level,
                action=decision.action,
                risk_score=assessment.risk_score.final_score,
                triggered_rule_ids=[
                    rule_id.value for rule_id in decision.triggered_rule_ids
                ],
                component_refund_count=len(
                    investigation.component_refund_ids
                ),
                exposure=FinancialExposureResponse(
                    realized_suspicious_amount_paise=(
                        case_exposure.realized_suspicious_amount.amount_paise
                    ),
                    pending_refund_exposure_paise=(
                        case_exposure.pending_refund_exposure.amount_paise
                    ),
                    remaining_refundable_exposure_paise=(
                        case_exposure.remaining_refundable_exposure.amount_paise
                    ),
                ),
                ml_prediction=(
                    None
                    if investigation.ml_prediction is None
                    else MLPredictionResponse(
                        probability=investigation.ml_prediction.probability,
                        is_high_risk=investigation.ml_prediction.is_high_risk,
                    )
                ),
            )
        )

    cases.sort(
        key=lambda case: (
            -case.risk_score,
            -(
                case.ml_prediction.probability
                if case.ml_prediction is not None
                else -1.0
            ),
            -len(case.triggered_rule_ids),
            -case.component_refund_count,
            case.refund_id,
        )
    )

    metrics = QueueMetricsResponse(
        open_case_count=len(cases),
        high_risk_count=sum(case.risk_level.value == "high" for case in cases),
        medium_risk_count=sum(
            case.risk_level.value == "medium" for case in cases
        ),
        low_risk_count=sum(case.risk_level.value == "low" for case in cases),
        triggered_case_count=sum(bool(case.triggered_rule_ids) for case in cases),
        clustered_case_count=sum(
            case.component_refund_count > 1 for case in cases
        ),
        pending_refund_exposure_paise=sum(
            case.exposure.pending_refund_exposure_paise for case in cases
        ),
        realized_suspicious_amount_paise=sum(
            case.exposure.realized_suspicious_amount_paise for case in cases
        ),
    )
    return InvestigationQueueResponse(cases=cases, metrics=metrics)


@router.get(
    "/{refund_id}",
    response_model=InvestigationResponse,
)
def get_investigation(
    refund_id: str,
    request: Request,
    db: Session = Depends(get_db),
) -> InvestigationResponse:
    """Perform a complete investigation for one refund."""
    print(f"[API] get_investigation called for refund_id: {refund_id}")

    try:
        parsed_refund_id = RefundId.from_str(refund_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid refund ID format",
        ) from exc

    snapshot = ReconstructionService(db).reconstruct()

    if parsed_refund_id not in snapshot.refunds:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Refund {refund_id} was not found",
        )

    ml_inference_service = (
        request.app.state.ml_inference_service
    )

    investigation = InvestigationService(
        snapshot,
        ml_inference_service=ml_inference_service,
    ).investigate(parsed_refund_id)

    assessment = investigation.assessment
    decision = investigation.decision
    exposure = investigation.exposure

    # Build EvidenceBundle & Narrative explanation
    bundle = EvidenceBundleBuilder.build(
        snapshot=snapshot,
        refund_id=parsed_refund_id,
        assessment=assessment,
        decision=decision,
        exposure=exposure,
        component_refund_ids=investigation.component_refund_ids,
        ml_inference_service=ml_inference_service,
    )
    print(f"[API] Building explanation for refund {refund_id}")
    explanation_service = InvestigationExplanationService()
    explanation_result = explanation_service.explain(bundle)
    print(f"[API] Explanation generated - is_llm_generated: {explanation_result.is_llm_generated}")

    evidence_bundle_resp = EvidenceBundleResponse(
        refund_id=bundle.refund_id,
        assessed_at=bundle.assessed_at,
        risk_level=bundle.risk_level,
        action=bundle.action,
        final_risk_score=bundle.final_risk_score,
        behavioral_confirmation_score=bundle.behavioral_confirmation_score,
        customer_profile=CustomerProfileResponse(
            customer_id=bundle.customer_profile.customer_id,
            email=bundle.customer_profile.email,
            phone=bundle.customer_profile.phone,
            created_at=bundle.customer_profile.created_at,
            total_order_count=bundle.customer_profile.total_order_count,
            total_refund_count=bundle.customer_profile.total_refund_count,
            total_paid_paise=bundle.customer_profile.total_paid_paise,
            total_refunded_paise=bundle.customer_profile.total_refunded_paise,
            refund_rate_by_count=bundle.customer_profile.refund_rate_by_count,
            refund_rate_by_amount=bundle.customer_profile.refund_rate_by_amount,
        ),
        financial_exposure=FinancialExposureResponse(
            realized_suspicious_amount_paise=bundle.financial_exposure.realized_suspicious_amount_paise,
            pending_refund_exposure_paise=bundle.financial_exposure.pending_refund_exposure_paise,
            remaining_refundable_exposure_paise=bundle.financial_exposure.remaining_refundable_exposure_paise,
        ),
        graph_topology=GraphTopologyResponse(
            cluster_id=bundle.graph_topology.cluster_id,
            cluster_size=bundle.graph_topology.cluster_size,
            connected_customer_ids=bundle.graph_topology.connected_customer_ids,
            connected_refund_ids=bundle.graph_topology.connected_refund_ids,
            shared_ip_addresses=bundle.graph_topology.shared_ip_addresses,
            shared_shipping_addresses=bundle.graph_topology.shared_shipping_addresses,
            shared_device_fingerprints=bundle.graph_topology.shared_device_fingerprints,
            shared_bank_accounts=bundle.graph_topology.shared_bank_accounts,
            is_multi_entity_cluster=bundle.graph_topology.is_multi_entity_cluster,
        ),
        rule_violations=[
            RuleEvidenceResponse(
                rule_id=rv.rule_id,
                triggered=rv.triggered,
                evidence_type=rv.evidence_type,
                evidence_value=rv.evidence_value,
                evidence_threshold=rv.evidence_threshold,
                base_signal_weight=0.0,
                notes=rv.notes,
            )
            for rv in bundle.rule_violations
        ],
        feature_contributions=[
            FeatureContributionResponse(
                feature_name=fc.feature_name,
                value=fc.value,
                direction=fc.direction,
                description=fc.description,
            )
            for fc in bundle.feature_contributions
        ],
    )

    explanation_resp = InvestigationExplanationResponse(
        headline=explanation_result.headline,
        narrative_summary=explanation_result.narrative_summary,
        key_risk_drivers=explanation_result.key_risk_drivers,
        suggested_action_rationale=explanation_result.suggested_action_rationale,
        is_llm_generated=explanation_result.is_llm_generated,
    )

    return InvestigationResponse(
        ml_prediction=(
            None
            if investigation.ml_prediction is None
            else MLPredictionResponse(
                probability=(
                    investigation.ml_prediction.probability
                ),
                is_high_risk=(
                    investigation.ml_prediction.is_high_risk
                ),
            )
        ),
        assessment=AssessmentResponse(
            refund_id=str(assessment.refund_id),
            customer_id=str(assessment.customer_id),
            component_id=assessment.component_id,
            risk_level=decision.risk_level,
            action=decision.action,
            triggered_rule_ids=[
                rule_id.value
                for rule_id in decision.triggered_rule_ids
            ],
            behavioral_confirmation_score=(
                decision.behavioral_confirmation_score
            ),
            risk_score=RiskScoreResponse(
                rule_signal_component=(
                    assessment.risk_score.rule_signal_component
                ),
                behavioral_confirmation_score=(
                    assessment.risk_score
                    .behavioral_confirmation_score
                ),
                cluster_signal_component=(
                    assessment.risk_score
                    .cluster_signal_component
                ),
                final_score=(
                    assessment.risk_score.final_score
                ),
            ),
            rule_outputs=[
                RuleEvidenceResponse(
                    rule_id=output.rule_id.value,
                    triggered=output.triggered,
                    evidence_type=output.evidence_type.value,
                    evidence_value=output.evidence_value,
                    evidence_threshold=(
                        output.evidence_threshold
                    ),
                    base_signal_weight=(
                        output.base_signal_weight
                    ),
                    notes=output.notes,
                )
                for output in assessment.rule_outputs
            ],
            explanation=decision.explanation,
        ),
        exposure=FinancialExposureResponse(
            realized_suspicious_amount_paise=(
                exposure.realized_suspicious_amount.amount_paise
            ),
            pending_refund_exposure_paise=(
                exposure.pending_refund_exposure.amount_paise
            ),
            remaining_refundable_exposure_paise=(
                exposure.remaining_refundable_exposure.amount_paise
            ),
        ),
        component_refund_ids=[
            str(component_refund_id)
            for component_refund_id
            in investigation.component_refund_ids
        ],
        evidence_bundle=evidence_bundle_resp,
        explanation_summary=explanation_resp,
    )