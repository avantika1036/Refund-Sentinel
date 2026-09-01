"""Unit tests for ML integration with investigations."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from backend.app.domain.enums import (
    PaymentStatus,
    RefundStatus,
)
from backend.app.domain.identifiers import (
    CustomerId,
    MerchantId,
    OrderId,
    PaymentId,
    RefundId,
)
from backend.app.domain.value_objects import (
    Money,
    UTCDateTime,
)
from backend.app.finance.aggregates import (
    OrderState,
    PaymentState,
    RefundState,
)
from backend.app.finance.types import (
    ReconstructionSnapshot,
)
from backend.app.ml.features import (
    build_feature_vector,
)
from backend.app.ml.inference import (
    MLInferenceService,
)
from backend.app.ml.model import (
    LogisticRiskModel,
)
from backend.app.ml.persistence import (
    PersistedModelBundle,
)
from backend.app.ml.preprocessing import (
    MLPreprocessor,
)
from backend.app.risk.assessment import (
    RiskAssessor,
)
from backend.app.risk.investigation import (
    InvestigationService,
)


def _build_snapshot() -> tuple[
    ReconstructionSnapshot,
    RefundId,
]:
    """Build a simple refund scenario."""

    now = datetime.now(timezone.utc)

    merchant_id = MerchantId.generate()
    customer_id = CustomerId.generate()

    payment_id = PaymentId.generate()
    order_id = OrderId.generate()
    refund_id = RefundId.generate()

    payment = PaymentState(
        payment_id=payment_id,
        merchant_id=merchant_id,
        order_id=order_id,
        customer_id=customer_id,
        authorised_amount=Money.of_paise(10000),
        status=PaymentStatus.CAPTURED,
        created_at=UTCDateTime(value=now),
        captured_amount=Money.of_paise(10000),
        captured_at=UTCDateTime(value=now),
    )

    order = OrderState(
        order_id=order_id,
        merchant_id=merchant_id,
        customer_id=customer_id,
        amount=Money.of_paise(10000),
        created_at=UTCDateTime(value=now),
    )

    refund = RefundState(
        refund_id=refund_id,
        payment_id=payment_id,
        merchant_id=merchant_id,
        customer_id=customer_id,
        order_id=order_id,
        requested_amount=Money.of_paise(5000),
        status=RefundStatus.PROCESSED,
        requested_at=UTCDateTime(value=now),
        processed_at=UTCDateTime(value=now),
        processed_amount=Money.of_paise(5000),
    )

    snapshot = ReconstructionSnapshot(
        payments={
            payment_id: payment,
        },
        refunds={
            refund_id: refund,
        },
        orders={
            order_id: order,
        },
        reconstruction_ordinal=1,
        event_count=1,
    )

    return snapshot, refund_id


def _build_ml_service(
    snapshot: ReconstructionSnapshot,
    refund_id: RefundId,
) -> MLInferenceService:
    """Build an ML inference service matching the assessment schema."""

    assessment = RiskAssessor(
        snapshot
    ).assess(refund_id)

    feature_vector = build_feature_vector(
        assessment
    )

    feature_names = feature_vector.feature_names

    model = LogisticRiskModel(
        feature_names=feature_names,
        coefficients=tuple(
            0.0
            for _ in feature_names
        ),
        intercept=0.0,
    )

    preprocessor = MLPreprocessor(
        feature_names=feature_names,
        replacement_values=tuple(
            0.0
            for _ in feature_names
        ),
    )

    bundle = PersistedModelBundle(
        model=model,
        preprocessor=preprocessor,
    )

    return MLInferenceService(
        bundle
    )


def test_investigation_without_ml_returns_no_prediction() -> None:
    """ML prediction should be absent when inference is not configured."""

    snapshot, refund_id = _build_snapshot()

    investigation = InvestigationService(
        snapshot
    ).investigate(refund_id)

    assert investigation.ml_prediction is None


def test_investigation_with_ml_returns_prediction() -> None:
    """Configured ML inference should produce a prediction."""

    snapshot, refund_id = _build_snapshot()

    ml_service = _build_ml_service(
        snapshot,
        refund_id,
    )

    investigation = InvestigationService(
        snapshot,
        ml_inference_service=ml_service,
    ).investigate(refund_id)

    assert investigation.ml_prediction is not None

    assert (
        investigation.ml_prediction.probability
        == pytest.approx(0.5)
    )

    assert (
        investigation.ml_prediction.is_high_risk
        is True
    )


def test_ml_prediction_does_not_replace_decision() -> None:
    """Deterministic decision should remain independent of ML output."""

    snapshot, refund_id = _build_snapshot()

    ml_service = _build_ml_service(
        snapshot,
        refund_id,
    )

    investigation = InvestigationService(
        snapshot,
        ml_inference_service=ml_service,
    ).investigate(refund_id)

    assert (
        investigation.decision.final_score
        == investigation.assessment.risk_score.final_score
    )

    assert investigation.ml_prediction is not None


def test_investigation_rejects_invalid_ml_service() -> None:
    """Only an MLInferenceService or None should be accepted."""

    snapshot, _ = _build_snapshot()

    with pytest.raises(
        TypeError,
        match="ml_inference_service",
    ):
        InvestigationService(
            snapshot,
            ml_inference_service=object(),
        )


def test_ml_prediction_uses_assessment_generated_by_investigation() -> None:
    """ML inference should operate on the investigation assessment."""

    snapshot, refund_id = _build_snapshot()

    ml_service = _build_ml_service(
        snapshot,
        refund_id,
    )

    investigation = InvestigationService(
        snapshot,
        ml_inference_service=ml_service,
    ).investigate(refund_id)

    expected_prediction = ml_service.predict(
        investigation.assessment
    )

    assert (
        investigation.ml_prediction
        == expected_prediction
    )