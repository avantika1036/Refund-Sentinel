"""Complete investigation API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.api.schemas import (
    AssessmentResponse,
    FinancialExposureResponse,
    InvestigationResponse,
    RiskScoreResponse,
    RuleEvidenceResponse,
)
from backend.app.domain.identifiers import RefundId
from backend.app.persistence.database import get_db
from backend.app.persistence.reconstruction import ReconstructionService
from backend.app.risk.investigation import InvestigationService


router = APIRouter(
    prefix="/api/v1/investigations",
    tags=["investigations"],
)


@router.get(
    "/{refund_id}",
    response_model=InvestigationResponse,
)
def get_investigation(
    refund_id: str,
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

    investigation = InvestigationService(
        snapshot
    ).investigate(parsed_refund_id)

    assessment = investigation.assessment
    decision = investigation.decision
    exposure = investigation.exposure

    return InvestigationResponse(
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