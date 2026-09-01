"""Dataset structures for Refund Sentinel machine learning.

This module provides a validated, immutable representation of a supervised
machine-learning dataset.

It intentionally does not know how features are computed. Feature extraction
belongs to ``backend.app.ml.features``. This module receives already-computed
numeric feature vectors together with their ground-truth labels and validates
that they form a consistent training dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Sequence


@dataclass(frozen=True)
class MLDataset:
    """Validated supervised learning dataset.

    Attributes:
        feature_names:
            Ordered names corresponding to each column in ``feature_rows``.

        feature_rows:
            One numeric feature vector per training example.

        labels:
            Binary ground-truth labels corresponding positionally to
            ``feature_rows``.

    Invariants:
        - The dataset contains at least one row.
        - Feature names are non-empty and unique.
        - Every row has exactly ``len(feature_names)`` values.
        - Every feature value is finite.
        - Labels are binary integers: 0 or 1.
        - The number of labels equals the number of feature rows.
    """

    feature_names: tuple[str, ...]
    feature_rows: tuple[tuple[float, ...], ...]
    labels: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.feature_names:
            raise ValueError(
                "Dataset must contain at least one feature name"
            )

        if not self.feature_rows:
            raise ValueError(
                "Dataset must contain at least one feature row"
            )

        if len(self.feature_names) != len(set(self.feature_names)):
            raise ValueError(
                "Feature names must be unique"
            )

        for feature_name in self.feature_names:
            if not isinstance(feature_name, str):
                raise TypeError(
                    "Feature names must be strings"
                )

            if not feature_name.strip():
                raise ValueError(
                    "Feature names must not be empty"
                )

        expected_width = len(self.feature_names)

        for row_index, row in enumerate(self.feature_rows):
            if len(row) != expected_width:
                raise ValueError(
                    "Feature row "
                    f"{row_index} has {len(row)} values but "
                    f"{expected_width} were expected"
                )

            for feature_index, value in enumerate(row):
                if isinstance(value, bool):
                    raise TypeError(
                        "Feature values must be numeric, not boolean "
                        f"(row={row_index}, column={feature_index})"
                    )

                if not isinstance(value, (int, float)):
                    raise TypeError(
                        "Feature values must be numeric "
                        f"(row={row_index}, column={feature_index})"
                    )

                numeric_value = float(value)

                if not isfinite(numeric_value):
                    raise ValueError(
                        "Feature values must be finite "
                        f"(row={row_index}, column={feature_index})"
                    )

        if len(self.labels) != len(self.feature_rows):
            raise ValueError(
                "Number of labels must match number of feature rows"
            )

        for label_index, label in enumerate(self.labels):
            if isinstance(label, bool) or label not in (0, 1):
                raise ValueError(
                    "Labels must be binary integers 0 or 1 "
                    f"(index={label_index}, value={label!r})"
                )

    @property
    def row_count(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.feature_rows)

    @property
    def feature_count(self) -> int:
        """Return the number of features per example."""
        return len(self.feature_names)

    @property
    def positive_count(self) -> int:
        """Return the number of positive examples."""
        return sum(self.labels)

    @property
    def negative_count(self) -> int:
        """Return the number of negative examples."""
        return self.row_count - self.positive_count

    def feature_matrix(self) -> list[list[float]]:
        """Return a mutable copy of the feature matrix.

        Each row preserves the order defined by ``feature_names``.
        """

        return [
            [float(value) for value in row]
            for row in self.feature_rows
        ]

    def label_vector(self) -> list[int]:
        """Return a mutable copy of the label vector."""
        return list(self.labels)


def create_dataset(
    *,
    feature_names: Sequence[str],
    feature_rows: Sequence[Sequence[float]],
    labels: Sequence[int],
) -> MLDataset:
    """Create a validated immutable ML dataset.

    This function accepts common mutable Python sequence types and converts
    them into the immutable representation used internally by ``MLDataset``.

    Args:
        feature_names:
            Ordered names for the feature columns.

        feature_rows:
            Numeric feature vectors. Every vector must have the same width.

        labels:
            Binary ground-truth labels corresponding positionally to
            ``feature_rows``.

    Returns:
        A validated ``MLDataset``.

    Raises:
        TypeError:
            If unsupported values are supplied.

        ValueError:
            If dataset invariants are violated.
    """

    return MLDataset(
        feature_names=tuple(feature_names),
        feature_rows=tuple(
            tuple(row)
            for row in feature_rows
        ),
        labels=tuple(labels),
    )