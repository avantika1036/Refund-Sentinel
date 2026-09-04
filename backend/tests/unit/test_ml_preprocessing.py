"""Unit tests for ML preprocessing."""

from math import isnan

import pytest

from backend.app.ml.preprocessing import (
    MLPreprocessor,
    PreprocessingError,
    fit_preprocessor,
)


def test_fit_uses_column_medians_for_missing_values() -> None:
    """Missing values should be replaced with training column medians."""

    preprocessor = MLPreprocessor.fit(
        feature_names=("a", "b"),
        feature_rows=[
            [1.0, 10.0],
            [3.0, float("nan")],
            [5.0, 30.0],
        ],
    )

    assert preprocessor.feature_names == ("a", "b")
    assert preprocessor.replacement_values == (3.0, 20.0)


def test_transform_replaces_nan_values() -> None:
    """Transform should replace missing values using fitted statistics."""

    preprocessor = MLPreprocessor.fit(
        feature_names=("a", "b"),
        feature_rows=[
            [1.0, 10.0],
            [3.0, 30.0],
            [5.0, 50.0],
        ],
    )

    transformed = preprocessor.transform(
        feature_names=("a", "b"),
        feature_rows=[
            [float("nan"), 20.0],
            [7.0, float("nan")],
        ],
    )

    assert transformed[0] == pytest.approx(
        [0.0, -0.6123724356957945]
    )
    assert transformed[1] == pytest.approx(
        [2.449489742783178, 0.0]
    )


def test_fit_does_not_modify_original_rows() -> None:
    """Fitting must not mutate caller-owned feature data."""

    rows = [
        [1.0, float("nan")],
        [3.0, 20.0],
    ]

    MLPreprocessor.fit(
        feature_names=("a", "b"),
        feature_rows=rows,
    )

    assert isnan(rows[0][1])


def test_all_missing_column_defaults_to_zero() -> None:
    """A completely missing training column should use 0.0."""

    preprocessor = MLPreprocessor.fit(
        feature_names=("a", "b"),
        feature_rows=[
            [float("nan"), 1.0],
            [float("nan"), 3.0],
        ],
    )

    assert preprocessor.replacement_values == (0.0, 2.0)


def test_transform_requires_matching_feature_schema() -> None:
    """Prediction data must match the fitted training schema."""

    preprocessor = MLPreprocessor.fit(
        feature_names=("a", "b"),
        feature_rows=[
            [1.0, 2.0],
        ],
    )

    with pytest.raises(
        PreprocessingError,
        match="Feature schema does not match",
    ):
        preprocessor.transform(
            feature_names=("b", "a"),
            feature_rows=[
                [2.0, 1.0],
            ],
        )


def test_fit_rejects_inconsistent_row_width() -> None:
    """Training rows must match the feature schema width."""

    with pytest.raises(
        PreprocessingError,
        match="were expected",
    ):
        MLPreprocessor.fit(
            feature_names=("a", "b"),
            feature_rows=[
                [1.0, 2.0],
                [3.0],
            ],
        )


def test_transform_rejects_infinite_values() -> None:
    """Infinite values must not enter the model-ready dataset."""

    preprocessor = MLPreprocessor.fit(
        feature_names=("a",),
        feature_rows=[
            [1.0],
        ],
    )

    with pytest.raises(
        PreprocessingError,
        match="must not be infinite",
    ):
        preprocessor.transform(
            feature_names=("a",),
            feature_rows=[
                [float("inf")],
            ],
        )


def test_fit_preprocessor_helper() -> None:
    """The helper should return a fitted preprocessor."""

    preprocessor = fit_preprocessor(
        feature_names=("score",),
        feature_rows=[
            [1.0],
            [3.0],
            [5.0],
        ],
    )

    assert isinstance(preprocessor, MLPreprocessor)
    assert preprocessor.replacement_values == (3.0,)