"""Application configuration for the Hybrid RAG API."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Centralized application configuration.

    Loads configuration values from environment variables and the
    local .env file while keeping secrets out of the source code.
    """

    app_name: str = "Hybrid RAG Enterprise Document Retrieval API"
    app_version: str = "1.0.0"
    debug: bool = False

    gemini_api_key: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()