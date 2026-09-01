"""Unit tests for ML feature construction."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from backend.app.ml.features import (
    FeatureConstructionError,
    _extract_numeric_fields,
)


@dataclass(frozen=True)
class ExampleFeatures:
    """Small test feature object."""

    numeric_feature: float
    integer_feature: int
    boolean_feature: bool
    missing_feature: float | None
    identifier: str


def test_extracts_numeric_fields() -> None:
    """Numeric values are converted to floats."""

    features = ExampleFeatures(
        numeric_feature=0.75,
        integer_feature=4,
        boolean_feature=True,
        missing_feature=None,
        identifier="customer-123",
    )

    result = _extract_numeric_fields(
        prefix="example",
        value=features,
    )

    assert result["example_numeric_feature"] == 0.75
    assert result["example_integer_feature"] == 4.0
    assert result["example_boolean_feature"] == 1.0


def test_non_numeric_metadata_is_excluded() -> None:
    """Identifiers and other non-numeric values are not ML features."""

    features = ExampleFeatures(
        numeric_feature=0.5,
        integer_feature=2,
        boolean_feature=False,
        missing_feature=None,
        identifier="refund-123",
    )

    result = _extract_numeric_fields(
        prefix="example",
        value=features,
    )

    assert "example_identifier" not in result


def test_boolean_false_becomes_zero() -> None:
    """Boolean feature values have deterministic numeric representation."""

    features = ExampleFeatures(
        numeric_feature=0.5,
        integer_feature=2,
        boolean_feature=False,
        missing_feature=None,
        identifier="refund-123",
    )

    result = _extract_numeric_fields(
        prefix="example",
        value=features,
    )

    assert result["example_boolean_feature"] == 0.0


def test_none_creates_nan_and_missing_indicator() -> None:
    """Missing numeric values have an explicit missing indicator."""

    features = ExampleFeatures(
        numeric_feature=0.5,
        integer_feature=2,
        boolean_feature=False,
        missing_feature=None,
        identifier="refund-123",
    )

    result = _extract_numeric_fields(
        prefix="example",
        value=features,
    )

    assert result["example_missing_feature"] != result[
        "example_missing_feature"
    ]

    assert (
        result["example_missing_feature_is_missing"]
        == 1.0
    )


def test_prefix_prevents_feature_name_collisions() -> None:
    """Feature names are namespaced by feature group."""

    features = ExampleFeatures(
        numeric_feature=0.5,
        integer_feature=2,
        boolean_feature=False,
        missing_feature=None,
        identifier="refund-123",
    )

    result = _extract_numeric_fields(
        prefix="individual",
        value=features,
    )

    assert "individual_numeric_feature" in result
    assert "numeric_feature" not in result


def test_rejects_non_dataclass_feature_group() -> None:
    """Feature extraction requires structured dataclass input."""

    with pytest.raises(FeatureConstructionError):
        _extract_numeric_fields(
            prefix="invalid",
            value={"numeric_feature": 0.5},
        )


def test_present_numeric_value_has_zero_missing_indicator() -> None:
    """Present numeric values explicitly indicate that they are available."""

    features = ExampleFeatures(
        numeric_feature=0.5,
        integer_feature=2,
        boolean_feature=False,
        missing_feature=0.25,
        identifier="refund-123",
    )

    result = _extract_numeric_fields(
        prefix="example",
        value=features,
    )

    assert result["example_missing_feature"] == 0.25
    assert (
        result["example_missing_feature_is_missing"]
        == 0.0
    )