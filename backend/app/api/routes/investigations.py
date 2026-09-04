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
    FinancialExposureResponse,
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
    )