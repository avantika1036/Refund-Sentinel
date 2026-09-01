"""ML preprocessing for Refund Sentinel.

Transforms raw numeric feature matrices into finite, model-ready values.

The feature construction layer represents missing numeric values as NaN and
adds explicit ``*_is_missing`` indicator features. This module learns
replacement values from training data and uses those values consistently for
future transformations.

The preprocessor deliberately does not perform scaling. Scaling requirements
depend on the model family and should remain explicit in the training layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isnan
from statistics import median
from typing import Sequence


class PreprocessingError(ValueError):
    """Raised when feature preprocessing cannot be performed safely."""


@dataclass(frozen=True)
class MLPreprocessor:
    """Fitted missing-value preprocessor.

    ``feature_names`` defines the exact schema the preprocessor was fitted on.

    ``replacement_values`` contains one replacement value for each feature
    column. Values are learned from the training data only.
    """

    feature_names: tuple[str, ...]
    replacement_values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.feature_names:
            raise PreprocessingError(
                "Preprocessor must contain at least one feature name"
            )

        if len(self.feature_names) != len(self.replacement_values):
            raise PreprocessingError(
                "Feature names and replacement values must have "
                "the same length"
            )

        if len(self.feature_names) != len(set(self.feature_names)):
            raise PreprocessingError(
                "Feature names must be unique"
            )

        for value in self.replacement_values:
            if isinstance(value, bool) or not isinstance(
                value,
                (int, float),
            ):
                raise TypeError(
                    "Replacement values must be numeric"
                )

            numeric_value = float(value)

            if isnan(numeric_value):
                raise PreprocessingError(
                    "Replacement values must not be NaN"
                )

    @classmethod
    def fit(
        cls,
        *,
        feature_names: Sequence[str],
        feature_rows: Sequence[Sequence[float]],
    ) -> "MLPreprocessor":
        """Fit replacement values from training feature rows.

        Missing values are represented by NaN. For each feature column, the
        median of the non-missing training values is used as the replacement.

        If an entire column is missing, its replacement value defaults to 0.0.

        Args:
            feature_names:
                Ordered feature schema.

            feature_rows:
                Training feature rows containing finite numeric values or NaN.

        Returns:
            A fitted immutable preprocessor.
        """

        normalized_names = tuple(feature_names)

        if not normalized_names:
            raise PreprocessingError(
                "Cannot fit a preprocessor without feature names"
            )

        if not feature_rows:
            raise PreprocessingError(
                "Cannot fit a preprocessor without feature rows"
            )

        expected_width = len(normalized_names)

        columns: list[list[float]] = [
            []
            for _ in range(expected_width)
        ]

        for row_index, row in enumerate(feature_rows):
            if len(row) != expected_width:
                raise PreprocessingError(
                    f"Feature row {row_index} has {len(row)} values "
                    f"but {expected_width} were expected"
                )

            for column_index, value in enumerate(row):
                numeric_value = _validate_feature_value(
                    value=value,
                    row_index=row_index,
                    column_index=column_index,
                )

                if not isnan(numeric_value):
                    columns[column_index].append(numeric_value)

        replacement_values = tuple(
            float(median(column)) if column else 0.0
            for column in columns
        )

        return cls(
            feature_names=normalized_names,
            replacement_values=replacement_values,
        )

    def transform(
        self,
        *,
        feature_names: Sequence[str],
        feature_rows: Sequence[Sequence[float]],
    ) -> list[list[float]]:
        """Transform feature rows into finite model-ready values.

        The incoming feature schema must exactly match the schema used during
        fitting. NaN values are replaced using the corresponding learned
        replacement value.
        """

        normalized_names = tuple(feature_names)

        if normalized_names != self.feature_names:
            raise PreprocessingError(
                "Feature schema does not match the fitted preprocessor"
            )

        expected_width = len(self.feature_names)
        transformed_rows: list[list[float]] = []

        for row_index, row in enumerate(feature_rows):
            if len(row) != expected_width:
                raise PreprocessingError(
                    f"Feature row {row_index} has {len(row)} values "
                    f"but {expected_width} were expected"
                )

            transformed_row: list[float] = []

            for column_index, value in enumerate(row):
                numeric_value = _validate_feature_value(
                    value=value,
                    row_index=row_index,
                    column_index=column_index,
                )

                if isnan(numeric_value):
                    transformed_row.append(
                        self.replacement_values[column_index]
                    )
                else:
                    transformed_row.append(numeric_value)

            transformed_rows.append(transformed_row)

        return transformed_rows


def fit_preprocessor(
    *,
    feature_names: Sequence[str],
    feature_rows: Sequence[Sequence[float]],
) -> MLPreprocessor:
    """Fit and return an ML preprocessor."""

    return MLPreprocessor.fit(
        feature_names=feature_names,
        feature_rows=feature_rows,
    )


def _validate_feature_value(
    *,
    value: object,
    row_index: int,
    column_index: int,
) -> float:
    """Validate and normalize one raw feature value."""

    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise TypeError(
            "Feature values must be numeric "
            f"(row={row_index}, column={column_index})"
        )

    numeric_value = float(value)

    if numeric_value == float("inf") or numeric_value == float("-inf"):
        raise PreprocessingError(
            "Feature values must not be infinite "
            f"(row={row_index}, column={column_index})"
        )

    return numeric_value