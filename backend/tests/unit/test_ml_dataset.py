"""Unit tests for the ML dataset layer."""

import math

import pytest

from backend.app.ml.dataset import MLDataset, create_dataset


def _create_valid_dataset() -> MLDataset:
    """Create a small valid dataset for tests."""

    return create_dataset(
        feature_names=(
            "capture_to_refund_latency_hrs",
            "customer_refund_rate_90d",
            "cluster_size",
        ),
        feature_rows=(
            (2.0, 0.1, 3.0),
            (24.0, 0.0, 1.0),
            (4.0, 0.8, 5.0),
        ),
        labels=(1, 0, 1),
    )


def test_create_valid_dataset() -> None:
    """A valid dataset is created successfully."""

    dataset = _create_valid_dataset()

    assert dataset.feature_names == (
        "capture_to_refund_latency_hrs",
        "customer_refund_rate_90d",
        "cluster_size",
    )

    assert dataset.row_count == 3
    assert dataset.feature_count == 3
    assert dataset.labels == (1, 0, 1)


def test_dataset_positive_and_negative_counts() -> None:
    """Class counts are computed correctly."""

    dataset = _create_valid_dataset()

    assert dataset.positive_count == 2
    assert dataset.negative_count == 1


def test_feature_matrix_returns_expected_values() -> None:
    """Feature matrix preserves values and ordering."""

    dataset = _create_valid_dataset()

    assert dataset.feature_matrix() == [
        [2.0, 0.1, 3.0],
        [24.0, 0.0, 1.0],
        [4.0, 0.8, 5.0],
    ]


def test_label_vector_returns_expected_values() -> None:
    """Label vector preserves ordering."""

    dataset = _create_valid_dataset()

    assert dataset.label_vector() == [1, 0, 1]


def test_empty_feature_names_are_rejected() -> None:
    """Datasets require at least one feature."""

    with pytest.raises(
        ValueError,
        match="at least one feature name",
    ):
        create_dataset(
            feature_names=(),
            feature_rows=((1.0,),),
            labels=(1,),
        )


def test_empty_feature_rows_are_rejected() -> None:
    """Datasets require at least one example."""

    with pytest.raises(
        ValueError,
        match="at least one feature row",
    ):
        create_dataset(
            feature_names=("feature_a",),
            feature_rows=(),
            labels=(),
        )


def test_duplicate_feature_names_are_rejected() -> None:
    """Feature names must be unique."""

    with pytest.raises(
        ValueError,
        match="Feature names must be unique",
    ):
        create_dataset(
            feature_names=(
                "feature_a",
                "feature_a",
            ),
            feature_rows=((1.0, 2.0),),
            labels=(1,),
        )


def test_feature_row_width_mismatch_is_rejected() -> None:
    """Every feature row must match the feature schema width."""

    with pytest.raises(
        ValueError,
        match="were expected",
    ):
        create_dataset(
            feature_names=(
                "feature_a",
                "feature_b",
            ),
            feature_rows=(
                (1.0,),
            ),
            labels=(1,),
        )


def test_label_count_must_match_row_count() -> None:
    """Every feature row requires exactly one label."""

    with pytest.raises(
        ValueError,
        match="Number of labels must match",
    ):
        create_dataset(
            feature_names=("feature_a",),
            feature_rows=(
                (1.0,),
                (2.0,),
            ),
            labels=(1,),
        )


@pytest.mark.parametrize(
    "invalid_label",
    [-1, 2, 0.5, "1", True, False],
)
def test_invalid_labels_are_rejected(
    invalid_label: object,
) -> None:
    """Only integer binary labels are accepted."""

    with pytest.raises(
        ValueError,
        match="Labels must be binary integers",
    ):
        create_dataset(
            feature_names=("feature_a",),
            feature_rows=((1.0,),),
            labels=(invalid_label,),
        )


def test_nan_feature_values_are_allowed_for_preprocessing() -> None:
    """Raw datasets may carry NaN until training-time imputation."""

    dataset = create_dataset(
        feature_names=("feature_a",),
        feature_rows=((math.nan,),),
        labels=(1,),
    )

    assert math.isnan(dataset.feature_rows[0][0])


@pytest.mark.parametrize("invalid_value", [math.inf, -math.inf])
def test_infinite_feature_values_are_rejected(
    invalid_value: float,
) -> None:
    """Infinities are invalid even though NaN represents missing data."""

    with pytest.raises(
        ValueError,
        match="must not be infinite",
    ):
        create_dataset(
            feature_names=("feature_a",),
            feature_rows=((invalid_value,),),
            labels=(1,),
        )


def test_non_numeric_feature_value_is_rejected() -> None:
    """Feature values must be numeric."""

    with pytest.raises(
        TypeError,
        match="Feature values must be numeric",
    ):
        create_dataset(
            feature_names=("feature_a",),
            feature_rows=(("invalid",),),
            labels=(1,),
        )


def test_boolean_feature_value_is_rejected() -> None:
    """Boolean values are not accepted as numeric features."""

    with pytest.raises(
        TypeError,
        match="not boolean",
    ):
        create_dataset(
            feature_names=("feature_a",),
            feature_rows=((True,),),
            labels=(1,),
        )


def test_returned_matrix_is_independent_copy() -> None:
    """Mutating a returned matrix cannot mutate the immutable dataset."""

    dataset = _create_valid_dataset()

    matrix = dataset.feature_matrix()
    matrix[0][0] = 999.0

    assert dataset.feature_rows[0][0] == 2.0


def test_returned_label_vector_is_independent_copy() -> None:
    """Mutating a returned label vector cannot mutate the dataset."""

    dataset = _create_valid_dataset()

    labels = dataset.label_vector()
    labels[0] = 0

    assert dataset.labels[0] == 1