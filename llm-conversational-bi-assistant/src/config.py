"""
Central application configuration via Pydantic Settings.

Loads environment variables from .env and validates types at startup —
fails fast if anything is misconfigured.

During tests (PYTEST_CURRENT_TEST is set automatically by pytest), disk
path validation is skipped — tests use tmp_path fixtures instead.

Usage:
    from src.config import settings
    print(settings.openrouter_model)
"""

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_TESTING = bool(os.getenv("PYTEST_CURRENT_TEST")) or ("pytest" in sys.modules)


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    openrouter_api_key: str = Field(
        default="test-key",
        validation_alias=AliasChoices("OPENROUTER_API_KEY", "OPENAI_API_KEY"),
        description="OpenRouter/OpenAI-compatible API key",
    )
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        validation_alias=AliasChoices("OPENROUTER_BASE_URL", "OPENAI_BASE_URL"),
        description="OpenAI-compatible base URL",
    )
    openrouter_model: str = Field(
        default="openai/gpt-4o-mini",
        validation_alias=AliasChoices("OPENROUTER_MODEL", "OPENAI_MODEL"),
        description="Model identifier",
    )
    openrouter_app_name: str = Field(
        default="BI Assistant",
        description="App name sent in HTTP-Referer headers",
    )
    openrouter_app_url: str = Field(
        default="http://localhost:8000",
        description="App URL sent in HTTP-Referer headers",
    )

    db_path: Path = Field(
        default=Path("data/staffing.db"),
        validation_alias=AliasChoices("database_path", "db_path"),
        description="Path to the SQLite database",
    )

    user_permissions_path: Path = Field(
        default=Path("data/user_permissions.json"),
        validation_alias=AliasChoices("permissions_path", "user_permissions_path"),
        description="Path to the user permissions JSON file",
    )

    chroma_persist_path: Path = Field(
        default=Path(".chroma"),
        description="ChromaDB persistence directory",
    )
    documents_path: Path = Field(
        default=Path("data/documents"),
        description="Directory containing Markdown files for RAG ingestion",
    )

    cache_similarity_threshold: float = Field(
        default=0.92,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold for a cache hit",
    )
    cache_ttl_seconds: int = Field(
        default=300,
        gt=0,
        description="Cache entry TTL in seconds",
    )
    cache_max_entries_per_user: int = Field(
        default=100,
        gt=0,
        description="Maximum in-memory cache entries per user namespace",
    )

    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_debug: bool = Field(default=False)

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")

    @field_validator("db_path", "user_permissions_path", "documents_path", mode="after")
    @classmethod
    def path_must_exist(cls, v: Path) -> Path:
        """
        Validate that required paths exist on disk.
        Skipped during tests — pytest sets PYTEST_CURRENT_TEST automatically.
        """
        if _TESTING:
            return v
        if not v.exists():
            raise ValueError(
                f"Path not found: '{v}'. "
                "Check your .env and run from the project root."
            )
        return v

    @model_validator(mode="after")
    def ensure_chroma_dir(self) -> "Settings":
        """Create the ChromaDB directory if it does not exist (skipped in tests)."""
        if not _TESTING:
            self.chroma_persist_path.mkdir(parents=True, exist_ok=True)
        return self

    @model_validator(mode="after")
    def ensure_llm_is_configured(self) -> "Settings":
        if _TESTING:
            return self

        key = (self.openrouter_api_key or "").strip()
        if not key or key in {"test-key", "skd_example"} or (key.startswith("<") and key.endswith(">")):
            raise ValueError(
                "LLM is not configured. Set OPENROUTER_API_KEY (or OPENAI_API_KEY) "
                "in your .env or environment variables."
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the singleton Settings instance.

    lru_cache ensures the .env file is read only once — O(1) on subsequent
    calls and safe to import across modules.
    """
    return Settings()


settings = get_settings()
