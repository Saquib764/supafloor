"""Application configuration."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from floorplan.constants import GENERATE_MAX_ATTEMPTS, SCORE_LEVEL, SCORE_THRESHOLD

_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    service_name: str = "supafloor"
    service_version: str = "1.0.0"
    cors_origins: list[str] = ["*"]

    openai_api_key: str = ""
    openai_model: str = "gpt-5.2"

    score_level: str = SCORE_LEVEL
    score_threshold: float = SCORE_THRESHOLD
    generate_max_attempts: int = GENERATE_MAX_ATTEMPTS


settings = Settings()
