"""ML preprocessing for Refund Sentinel.

The feature pipeline can contain missing numeric values and features with very
 different numeric ranges (for example cluster size, hours, and rates).  A
 logistic-regression model is sensitive to those ranges, so this preprocessor
 performs two fitted steps using training data only:

1. Median imputation for missing numeric values.
2. Per-feature standardization with a zero-mean, unit-scale transform.

The fitted statistics are persisted with the model and reused unchanged during
inference.  Constant columns use a scale of 1.0 so they remain finite.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, isnan, sqrt
from statistics import median
from typing import Sequence


class PreprocessingError(ValueError):
    """Raised when feature preprocessing cannot be performed safely."""


@dataclass(frozen=True)
class MLPreprocessor:
    """Fitted imputation and standardization state."""

    feature_names: tuple[str, ...]
    replacement_values: tuple[float, ...]
    means: tuple[float, ...] = ()
    scales: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        width = len(self.feature_names)

        if width == 0:
            raise PreprocessingError(
                "Preprocessor must contain at least one feature name"
            )

        if len(self.replacement_values) != width:
            raise PreprocessingError(
                "Feature names and replacement values must have "
                "the same length"
            )

        if len(self.feature_names) != len(set(self.feature_names)):
            raise PreprocessingError("Feature names must be unique")

        # Empty statistics are accepted only for legacy artifacts and are
        # normalized to an identity scaling transform.
        means = self.means or tuple(0.0 for _ in range(width))
        scales = self.scales or tuple(1.0 for _ in range(width))

        if len(means) != width or len(scales) != width:
            raise PreprocessingError(
                "Feature names, means, and scales must have the same length"
            )

        for value in self.replacement_values:
            _validate_finite_statistic(value, "Replacement values")

        for value in means:
            _validate_finite_statistic(value, "Means")

        for value in scales:
            numeric_value = _validate_finite_statistic(value, "Scales")
            if numeric_value <= 0.0:
                raise PreprocessingError("Scales must be greater than zero")

        object.__setattr__(self, "means", tuple(float(v) for v in means))
        object.__setattr__(self, "scales", tuple(float(v) for v in scales))

    @classmethod
    def fit(
        cls,
        *,
        feature_names: Sequence[str],
        feature_rows: Sequence[Sequence[float]],
    ) -> "MLPreprocessor":
        """Fit imputation and scaling statistics from training rows only."""

        normalized_names = tuple(feature_names)
        if not normalized_names:
            raise PreprocessingError(
                "Cannot fit a preprocessor without feature names"
            )
        if not feature_rows:
            raise PreprocessingError(
                "Cannot fit a preprocessor without feature rows"
            )

        width = len(normalized_names)
        columns: list[list[float]] = [[] for _ in range(width)]

        for row_index, row in enumerate(feature_rows):
            if len(row) != width:
                raise PreprocessingError(
                    f"Feature row {row_index} has {len(row)} values "
                    f"but {width} were expected"
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

        imputed_columns: list[list[float]] = []
        for column_index, column in enumerate(columns):
            replacement = replacement_values[column_index]
            # Missing rows are represented by the learned replacement value.
            # Include those imputed values when fitting the exact transform
            # used by the model.
            values = list(column)
            missing_count = len(feature_rows) - len(column)
            if missing_count:
                values.extend([replacement] * missing_count)
            imputed_columns.append(values)

        means = tuple(
            sum(column) / len(column)
            for column in imputed_columns
        )

        scales: list[float] = []
        for column, mean in zip(imputed_columns, means):
            variance = sum(
                (value - mean) ** 2
                for value in column
            ) / len(column)
            scale = sqrt(variance)
            scales.append(scale if scale > 1e-12 else 1.0)

        return cls(
            feature_names=normalized_names,
            replacement_values=replacement_values,
            means=means,
            scales=tuple(scales),
        )

    def transform(
        self,
        *,
        feature_names: Sequence[str],
        feature_rows: Sequence[Sequence[float]],
    ) -> list[list[float]]:
        """Impute and standardize rows using fitted training statistics."""

        normalized_names = tuple(feature_names)
        if normalized_names != self.feature_names:
            raise PreprocessingError(
                "Feature schema does not match the fitted preprocessor"
            )

        width = len(self.feature_names)
        transformed_rows: list[list[float]] = []

        for row_index, row in enumerate(feature_rows):
            if len(row) != width:
                raise PreprocessingError(
                    f"Feature row {row_index} has {len(row)} values "
                    f"but {width} were expected"
                )

            transformed_row: list[float] = []
            for column_index, value in enumerate(row):
                numeric_value = _validate_feature_value(
                    value=value,
                    row_index=row_index,
                    column_index=column_index,
                )
                if isnan(numeric_value):
                    numeric_value = self.replacement_values[column_index]

                standardized = (
                    (numeric_value - self.means[column_index])
                    / self.scales[column_index]
                )

                if not isfinite(standardized):
                    raise PreprocessingError(
                        "Standardized feature value must be finite "
                        f"(row={row_index}, column={column_index})"
                    )

                transformed_row.append(standardized)

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

    if isinstance(value, bool) or not isinstance(value, (int, float)):
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


def _validate_finite_statistic(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    numeric_value = float(value)
    if not isfinite(numeric_value):
        raise PreprocessingError(f"{label} must be finite")
    return numeric_value
