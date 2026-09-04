"""Risk assessment API routes."""

from __future__ import annotations

from collections.abc import Generator

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.api.schemas import (
    AssessmentResponse,
    RiskScoreResponse,
    RuleEvidenceResponse,
)
from backend.app.api.security import require_api_key
from backend.app.domain.identifiers import RefundId
from backend.app.persistence.database import get_db
from backend.app.persistence.reconstruction import ReconstructionService
from backend.app.risk.assessment import RiskAssessor
from backend.app.risk.decision import RiskDecisionEngine


router = APIRouter(
    prefix="/api/v1/assessments",
    tags=["assessments"],
    dependencies=[Depends(require_api_key)],
)


@router.get(
    "/{refund_id}",
    response_model=AssessmentResponse,
)
def get_assessment(
    refund_id: str,
    db: Session = Depends(get_db),
) -> AssessmentResponse:
    """Assess one refund using the reconstructed system state.

    The assessment is computed from the durable event ledger at request time.
    No assessment result is persisted or mutated by this endpoint.
    """

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

    assessment = RiskAssessor(snapshot).assess(
        parsed_refund_id
    )

    decision = RiskDecisionEngine().decide(
        assessment
    )

    return AssessmentResponse(
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
    )