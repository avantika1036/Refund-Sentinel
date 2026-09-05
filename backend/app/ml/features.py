"""ML feature vector construction.

Converts the deterministic feature objects produced by the risk pipeline into
stable numeric feature vectors suitable for machine learning.

This module deliberately does not include:

- Refund, customer, component, or cluster identifiers
- Ground-truth labels
- Scenario names or simulator metadata
- Financial exposure values

Those values could cause entity leakage, label leakage, or cause the model to
confuse financial value with behavioral risk.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import Any

from backend.app.domain.enums import RefundReasonCode
from backend.app.config import settings
from backend.app.risk.assessment import RiskAssessment


class FeatureConstructionError(ValueError):
    """Raised when a feature object cannot be converted to numeric values."""


@dataclass(frozen=True)
class FeatureVector:
    """Structured feature vector for ML inference.

    Attributes:
        feature_names: Ordered tuple of feature names.
        values: Ordered list of numeric feature values.
    """

    feature_names: tuple[str, ...]
    values: list[float]


def build_feature_vector(assessment: RiskAssessment) -> FeatureVector:
    """Build a deterministic numeric feature vector from a risk assessment.

    The vector contains only numeric fields from the three feature groups:

    - Individual features
    - Cluster features
    - Relationship features

    Identifiers and other non-numeric metadata are excluded automatically.

    Feature names are prefixed with their feature group to prevent naming
    collisions.

    Args:
        assessment: Completed risk assessment containing extracted features.

    Returns:
        FeatureVector with ordered feature names and values.

    Raises:
        FeatureConstructionError: If a supported feature field contains an
            unsupported value type.
    """

    feature_dict: dict[str, float] = {}

    feature_dict.update(
        _extract_numeric_fields(
            prefix="individual",
            value=assessment.individual_features,
        )
    )

    # Graph/cluster evidence is a supporting signal, not a standalone fraud
    # detector. When behavioral confirmation fails, structural features are
    # zeroed before entering the ML model. This keeps the learned model
    # consistent with the deterministic behavioral-confirmation gate.
    cluster_features = assessment.cluster_features
    relationship_features = assessment.relationship_features
    confirmation_passed = (
        assessment.behavioral_confirmation_score
        >= settings.ml_behavioral_confirmation_threshold
    )

    if confirmation_passed:
        gated_cluster = cluster_features
        gated_relationship = relationship_features
    else:
        gated_cluster = _zero_dataclass_numeric_fields(cluster_features)
        gated_relationship = _zero_dataclass_numeric_fields(relationship_features)

    feature_dict.update(
        _extract_numeric_fields(
            prefix="cluster",
            value=gated_cluster,
        )
    )

    feature_dict.update(
        _extract_numeric_fields(
            prefix="relationship",
            value=gated_relationship,
        )
    )

    feature_dict.update(
        _extract_refund_reason_features(
            assessment.individual_features.refund_reason_code
        )
    )

    # Convert to ordered structure for ML inference
    feature_names = tuple(sorted(feature_dict.keys()))
    values = [feature_dict[name] for name in feature_names]

    return FeatureVector(
        feature_names=feature_names,
        values=values,
    )


def build_feature_matrix(
    assessments: list[RiskAssessment],
) -> tuple[tuple[str, ...], list[list[float]]]:
    """Build a stable feature matrix from multiple risk assessments.

    All assessments must produce exactly the same feature schema.

    Args:
        assessments: Completed risk assessments.

    Returns:
        A tuple containing:

        - Ordered feature names
        - Rows of numeric feature values

    Raises:
        FeatureConstructionError: If assessments produce inconsistent schemas.
    """

    if not assessments:
        return (), []

    first_vector = build_feature_vector(assessments[0])
    feature_names = tuple(first_vector.keys())

    rows: list[list[float]] = [
        [first_vector[name] for name in feature_names]
    ]

    expected_names = set(feature_names)

    for assessment in assessments[1:]:
        vector = build_feature_vector(assessment)

        if set(vector.keys()) != expected_names:
            raise FeatureConstructionError(
                "Assessments produced inconsistent feature schemas"
            )

        rows.append(
            [vector[name] for name in feature_names]
        )

    return feature_names, rows



def _extract_refund_reason_features(reason_code: object) -> dict[str, float]:
    """Encode the categorical refund reason with a fixed one-hot schema.

    The generic numeric extractor intentionally ignores strings.  The previous
    implementation accidentally emitted a reason feature only when the value
    was missing, which made the inference schema depend on runtime data.
    Keeping a fixed feature for every supported reason makes the schema stable
    for both training and production inference.
    """

    normalized = None
    if reason_code is not None:
        normalized = str(
            getattr(reason_code, "value", reason_code)
        ).strip().lower()

    extracted: dict[str, float] = {}
    for reason in RefundReasonCode:
        name = f"individual_refund_reason_{reason.value}"
        extracted[name] = float(normalized == reason.value)

    extracted["individual_refund_reason_is_missing"] = float(
        normalized is None or normalized not in {r.value for r in RefundReasonCode}
    )
    return extracted


def _extract_numeric_fields(
    *,
    prefix: str,
    value: Any,
) -> dict[str, float]:
    """Extract numeric fields and missing-value indicators from a dataclass.

    Numeric values are preserved as floats.

    Missing numeric values are represented by:

    - the original feature value set to NaN
    - a corresponding ``<feature_name>_is_missing`` indicator set to 1.0

    Non-missing values receive a missing indicator of 0.0.

    Identifiers and other non-numeric metadata are intentionally excluded from
    the ML feature vector.
    """

    if not is_dataclass(value):
        raise FeatureConstructionError(
            f"Expected dataclass for feature group '{prefix}'"
        )

    extracted: dict[str, float] = {}

    for field in fields(value):
        field_value = getattr(value, field.name)
        feature_name = f"{prefix}_{field.name}"

        if isinstance(field_value, bool):
            extracted[feature_name] = float(field_value)
            continue

        if isinstance(field_value, (int, float)):
            extracted[feature_name] = float(field_value)

            if field_value != field_value:
                extracted[f"{feature_name}_is_missing"] = 1.0
            else:
                extracted[f"{feature_name}_is_missing"] = 0.0

            continue

        if field_value is None:
            extracted[feature_name] = float("nan")
            extracted[f"{feature_name}_is_missing"] = 1.0
            continue

        # Identifiers and other non-numeric metadata must not become
        # model features.
        continue

    return extracted

def _zero_dataclass_numeric_fields(value: Any) -> Any:
    """Return a lightweight dataclass copy with numeric fields zeroed."""
    if not is_dataclass(value):
        raise FeatureConstructionError("Expected dataclass for gated feature group")

    values: dict[str, Any] = {}
    for field in fields(value):
        original = getattr(value, field.name)
        if isinstance(original, bool):
            values[field.name] = False
        elif isinstance(original, (int, float)):
            values[field.name] = 0
        else:
            values[field.name] = original
    return type(value)(**values)
