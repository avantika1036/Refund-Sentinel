"""Persistence for trained Refund Sentinel ML model artifacts.

Stores a trained logistic regression model and its fitted preprocessor as a
single versioned JSON artifact.

The persistence format intentionally uses JSON rather than pickle because the
model state consists entirely of primitive values. This keeps artifacts
inspectable and avoids executing arbitrary code during loading.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.app.ml.model import LogisticRiskModel
from backend.app.ml.preprocessing import MLPreprocessor
from backend.app.ml.training import TrainingPipelineResult


class ModelPersistenceError(ValueError):
    """Raised when a persisted ML model artifact is invalid or unsafe."""


_ARTIFACT_VERSION = 1


@dataclass(frozen=True)
class PersistedModelBundle:
    """Complete deployable ML model bundle.

    A model cannot safely be used for inference without the preprocessor that
    was fitted during training. This bundle therefore persists both objects
    together and validates that their feature schemas match.

    Attributes:
        model:
            Trained logistic regression model.

        preprocessor:
            Fitted preprocessing state required before model prediction.
    """

    model: LogisticRiskModel
    preprocessor: MLPreprocessor

    def __post_init__(self) -> None:
        if not isinstance(self.model, LogisticRiskModel):
            raise TypeError(
                "model must be a LogisticRiskModel"
            )

        if not isinstance(
            self.preprocessor,
            MLPreprocessor,
        ):
            raise TypeError(
                "preprocessor must be an MLPreprocessor"
            )

        if (
            self.model.feature_names
            != self.preprocessor.feature_names
        ):
            raise ModelPersistenceError(
                "Model and preprocessor feature schemas must match"
            )

    @classmethod
    def from_training_result(
        cls,
        training_result: TrainingPipelineResult,
    ) -> "PersistedModelBundle":
        """Create a deployable bundle from a training pipeline result."""

        if not isinstance(
            training_result,
            TrainingPipelineResult,
        ):
            raise TypeError(
                "training_result must be a TrainingPipelineResult"
            )

        return cls(
            model=training_result.model,
            preprocessor=training_result.preprocessor,
        )


def save_model_bundle(
    bundle: PersistedModelBundle,
    path: str | Path,
) -> Path:
    """Save a validated model bundle as a JSON artifact.

    The artifact is written atomically through a temporary file to reduce the
    risk of leaving a partially written model artifact behind.

    Args:
        bundle:
            Validated model bundle to persist.

        path:
            Destination JSON artifact path.

    Returns:
        The resolved artifact path.

    Raises:
        ModelPersistenceError:
            If the path is invalid or the artifact cannot be saved safely.
    """

    if not isinstance(
        bundle,
        PersistedModelBundle,
    ):
        raise TypeError(
            "bundle must be a PersistedModelBundle"
        )

    artifact_path = _normalize_artifact_path(path)

    payload = _bundle_to_payload(bundle)

    serialized_payload = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )

    parent_directory = artifact_path.parent

    try:
        parent_directory.mkdir(
            parents=True,
            exist_ok=True,
        )
    except OSError as error:
        raise ModelPersistenceError(
            f"Could not create artifact directory: "
            f"{parent_directory}"
        ) from error

    temporary_path = artifact_path.with_suffix(
        artifact_path.suffix + ".tmp"
    )

    try:
        temporary_path.write_text(
            serialized_payload,
            encoding="utf-8",
        )

        os.replace(
            temporary_path,
            artifact_path,
        )
    except OSError as error:
        _remove_temporary_file(temporary_path)

        raise ModelPersistenceError(
            f"Could not save model artifact: "
            f"{artifact_path}"
        ) from error

    return artifact_path


def load_model_bundle(
    path: str | Path,
) -> PersistedModelBundle:
    """Load and validate a persisted model bundle.

    Args:
        path:
            JSON artifact path.

    Returns:
        Fully reconstructed and validated model bundle.

    Raises:
        ModelPersistenceError:
            If the artifact does not exist, is malformed, or contains invalid
            model data.
    """

    artifact_path = _normalize_artifact_path(path)

    if not artifact_path.exists():
        raise ModelPersistenceError(
            f"Model artifact does not exist: "
            f"{artifact_path}"
        )

    if not artifact_path.is_file():
        raise ModelPersistenceError(
            f"Model artifact path is not a file: "
            f"{artifact_path}"
        )

    try:
        serialized_payload = artifact_path.read_text(
            encoding="utf-8",
        )
    except OSError as error:
        raise ModelPersistenceError(
            f"Could not read model artifact: "
            f"{artifact_path}"
        ) from error

    try:
        payload = json.loads(
            serialized_payload,
        )
    except json.JSONDecodeError as error:
        raise ModelPersistenceError(
            "Model artifact contains invalid JSON"
        ) from error

    return _payload_to_bundle(payload)


def _normalize_artifact_path(
    path: str | Path,
) -> Path:
    """Validate and normalize an artifact path."""

    if isinstance(path, Path):
        artifact_path = path
    elif isinstance(path, str):
        if not path.strip():
            raise ModelPersistenceError(
                "Artifact path must not be empty"
            )

        artifact_path = Path(path)
    else:
        raise TypeError(
            "path must be a string or pathlib.Path"
        )

    if artifact_path.name in {"", ".", ".."}:
        raise ModelPersistenceError(
            "Artifact path must identify a file"
        )

    return artifact_path


def _bundle_to_payload(
    bundle: PersistedModelBundle,
) -> dict[str, Any]:
    """Convert a validated bundle into JSON-compatible data."""

    return {
        "artifact_version": _ARTIFACT_VERSION,
        "model": {
            "feature_names": list(
                bundle.model.feature_names
            ),
            "coefficients": list(
                bundle.model.coefficients
            ),
            "intercept": bundle.model.intercept,
        },
        "preprocessor": {
            "feature_names": list(
                bundle.preprocessor.feature_names
            ),
            "replacement_values": list(
                bundle.preprocessor.replacement_values
            ),
        },
    }


def _payload_to_bundle(
    payload: object,
) -> PersistedModelBundle:
    """Validate JSON payload structure and reconstruct a model bundle."""

    if not isinstance(payload, dict):
        raise ModelPersistenceError(
            "Model artifact root must be an object"
        )

    _validate_exact_keys(
        payload,
        expected_keys={
            "artifact_version",
            "model",
            "preprocessor",
        },
        context="Model artifact",
    )

    artifact_version = payload["artifact_version"]

    if (
        isinstance(artifact_version, bool)
        or not isinstance(artifact_version, int)
    ):
        raise ModelPersistenceError(
            "artifact_version must be an integer"
        )

    if artifact_version != _ARTIFACT_VERSION:
        raise ModelPersistenceError(
            f"Unsupported artifact version: "
            f"{artifact_version}"
        )

    model_payload = payload["model"]
    preprocessor_payload = payload["preprocessor"]

    model = _payload_to_model(
        model_payload,
    )

    preprocessor = _payload_to_preprocessor(
        preprocessor_payload,
    )

    try:
        return PersistedModelBundle(
            model=model,
            preprocessor=preprocessor,
        )
    except (
        TypeError,
        ValueError,
    ) as error:
        raise ModelPersistenceError(
            "Model artifact contains incompatible "
            "model and preprocessor schemas"
        ) from error


def _payload_to_model(
    payload: object,
) -> LogisticRiskModel:
    """Reconstruct a validated LogisticRiskModel from JSON data."""

    if not isinstance(payload, dict):
        raise ModelPersistenceError(
            "model must be an object"
        )

    _validate_exact_keys(
        payload,
        expected_keys={
            "feature_names",
            "coefficients",
            "intercept",
        },
        context="model",
    )

    feature_names = _parse_string_list(
        payload["feature_names"],
        context="model.feature_names",
    )

    coefficients = _parse_numeric_list(
        payload["coefficients"],
        context="model.coefficients",
    )

    intercept = _parse_numeric_value(
        payload["intercept"],
        context="model.intercept",
    )

    try:
        return LogisticRiskModel(
            feature_names=tuple(feature_names),
            coefficients=tuple(coefficients),
            intercept=intercept,
        )
    except (
        TypeError,
        ValueError,
    ) as error:
        raise ModelPersistenceError(
            "Model artifact contains an invalid model"
        ) from error


def _payload_to_preprocessor(
    payload: object,
) -> MLPreprocessor:
    """Reconstruct a validated MLPreprocessor from JSON data."""

    if not isinstance(payload, dict):
        raise ModelPersistenceError(
            "preprocessor must be an object"
        )

    _validate_exact_keys(
        payload,
        expected_keys={
            "feature_names",
            "replacement_values",
        },
        context="preprocessor",
    )

    feature_names = _parse_string_list(
        payload["feature_names"],
        context="preprocessor.feature_names",
    )

    replacement_values = _parse_numeric_list(
        payload["replacement_values"],
        context="preprocessor.replacement_values",
    )

    try:
        return MLPreprocessor(
            feature_names=tuple(feature_names),
            replacement_values=tuple(
                replacement_values
            ),
        )
    except (
        TypeError,
        ValueError,
    ) as error:
        raise ModelPersistenceError(
            "Model artifact contains an invalid preprocessor"
        ) from error


def _validate_exact_keys(
    payload: dict[str, Any],
    *,
    expected_keys: set[str],
    context: str,
) -> None:
    """Require an artifact object to contain exactly expected keys."""

    actual_keys = set(payload)

    missing_keys = (
        expected_keys
        - actual_keys
    )

    unexpected_keys = (
        actual_keys
        - expected_keys
    )

    if missing_keys:
        missing = ", ".join(
            sorted(missing_keys)
        )

        raise ModelPersistenceError(
            f"{context} is missing required keys: "
            f"{missing}"
        )

    if unexpected_keys:
        unexpected = ", ".join(
            sorted(unexpected_keys)
        )

        raise ModelPersistenceError(
            f"{context} contains unexpected keys: "
            f"{unexpected}"
        )


def _parse_string_list(
    value: object,
    *,
    context: str,
) -> list[str]:
    """Validate and normalize a JSON list of strings."""

    if not isinstance(value, list):
        raise ModelPersistenceError(
            f"{context} must be a list"
        )

    normalized_values: list[str] = []

    for item in value:
        if not isinstance(item, str):
            raise ModelPersistenceError(
                f"{context} must contain only strings"
            )

        normalized_values.append(item)

    return normalized_values


def _parse_numeric_list(
    value: object,
    *,
    context: str,
) -> list[float]:
    """Validate and normalize a JSON list of finite numbers."""

    if not isinstance(value, list):
        raise ModelPersistenceError(
            f"{context} must be a list"
        )

    return [
        _parse_numeric_value(
            item,
            context=context,
        )
        for item in value
    ]


def _parse_numeric_value(
    value: object,
    *,
    context: str,
) -> float:
    """Validate and normalize one finite numeric JSON value."""

    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise ModelPersistenceError(
            f"{context} must be numeric"
        )

    numeric_value = float(value)

    if (
        numeric_value == float("inf")
        or numeric_value == float("-inf")
        or numeric_value != numeric_value
    ):
        raise ModelPersistenceError(
            f"{context} must be finite"
        )

    return numeric_value


def _remove_temporary_file(
    path: Path,
) -> None:
    """Remove a temporary artifact file if it exists."""

    try:
        if path.exists():
            path.unlink()
    except OSError:
        # The original persistence failure is more useful than a cleanup error.
        pass