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

    feature_dict.update(
        _extract_numeric_fields(
            prefix="cluster",
            value=assessment.cluster_features,
        )
    )

    feature_dict.update(
        _extract_numeric_fields(
            prefix="relationship",
            value=assessment.relationship_features,
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