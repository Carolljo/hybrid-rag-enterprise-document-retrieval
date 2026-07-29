"""Application configuration for the Hybrid RAG API."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings.

    Centralizes configuration values used across the API.
    These settings can later be extended with environment variables,
    database configuration, vector store settings, and model settings.
    """

    app_name: str = "Hybrid RAG Enterprise Document Retrieval"
    app_version: str = "1.0.0"
    debug: bool = False


# Create a reusable settings instance
settings = Settings()