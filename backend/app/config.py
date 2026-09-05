from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # SQLite keeps the app runnable for local demos and unit/integration tests.
    # Production can override this with a PostgreSQL URL.
    database_url: str = "sqlite:///./refund_sentinel.db"

    app_env: str = "development"

    app_api_key: str = ""

    razorpay_key_id: str = ""
    razorpay_key_secret: str = ""
    razorpay_webhook_secret: str = ""
    # Webhooks are rejected without a configured secret unless this explicit
    # local-development bypass is enabled.
    allow_insecure_webhook_bypass: bool = False

    # Optional investigation narrative provider. When no key is configured,
    # the application uses the deterministic explanation fallback.
    llm_provider: str = ""
    llm_api_key: str = ""
    # Provider-native key names are supported as aliases for local .env files.
    gemini_api_key: str = ""
    openai_api_key: str = ""
    llm_model: str = "gemini-2.5-flash"
    llm_timeout_seconds: float = 10.0

    demo_seed: int = 42
    train_seed: int = 100
    eval_seed: int = 200

    ml_model_path: str = ""
    evaluation_results_path: str = "data/results.json"

    ml_classification_threshold: float = 0.5
    ml_behavioral_confirmation_threshold: float = 0.5


settings = Settings()