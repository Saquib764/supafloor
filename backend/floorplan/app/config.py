"""Application configuration."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

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


settings = Settings()
