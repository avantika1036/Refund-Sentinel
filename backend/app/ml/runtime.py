"""Runtime ML service construction.

Loads an optional persisted model artifact from application configuration
and constructs the inference service used by the application.

The deterministic Refund Sentinel pipeline does not depend on ML being
configured. When no model path is configured, this module returns None.
"""

from __future__ import annotations

from backend.app.config import Settings
from backend.app.ml.inference import MLInferenceService
from backend.app.ml.persistence import load_model_bundle


def create_ml_inference_service(
    settings: Settings,
) -> MLInferenceService | None:
    """Create the configured ML inference service.

    Args:
        settings:
            Application configuration.

    Returns:
        A configured MLInferenceService when an ML model path is configured.
        Otherwise, None.

    Raises:
        TypeError:
            If settings is not a Settings instance.

        ModelPersistenceError:
            If a configured model artifact cannot be loaded or validated.
    """

    if not isinstance(settings, Settings):
        raise TypeError(
            "settings must be a Settings instance"
        )

    model_path = settings.ml_model_path.strip()

    if not model_path:
        return None

    bundle = load_model_bundle(
        model_path
    )

    return MLInferenceService(
        bundle,
        classification_threshold=(
            settings.ml_classification_threshold
        ),
    )