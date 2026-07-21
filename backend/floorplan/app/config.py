"""Application configuration."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    service_name: str = "supafloor"
    service_version: str = "1.0.0"
    cors_origins: list[str] = field(default_factory=lambda: ["*"])


settings = Settings()
