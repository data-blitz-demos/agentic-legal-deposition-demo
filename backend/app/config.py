"""Application configuration loaded from environment variables.

This module centralizes runtime settings so all subsystems (API routes,
workflow engine, LLM selection, storage) read from one typed source.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed environment-backed configuration for the application."""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    model_name: str = Field(default="gpt-5.2", alias="MODEL_NAME")
    openai_models: str = Field(default="gpt-5.2", alias="OPENAI_MODELS")
    default_llm_provider: str = Field(default="openai", alias="DEFAULT_LLM_PROVIDER")
    ollama_url: str = Field(default="http://localhost:11434", alias="OLLAMA_URL")
    ollama_default_model: str = Field(default="llama3.3", alias="OLLAMA_DEFAULT_MODEL")
    ollama_models: str = Field(default="", alias="OLLAMA_MODELS")
    ollama_keep_alive: str = Field(default="10m", alias="OLLAMA_KEEP_ALIVE")
    llm_readiness_ttl_seconds: int = Field(default=120, alias="LLM_READINESS_TTL_SECONDS")
    llm_probe_timeout_seconds: int = Field(default=12, alias="LLM_PROBE_TIMEOUT_SECONDS")
    llm_options_probe_workers: int = Field(default=3, alias="LLM_OPTIONS_PROBE_WORKERS")
    couchdb_url: str = Field(default="http://admin:password@localhost:5984", alias="COUCHDB_URL")
    couchdb_db: str = Field(default="depositions", alias="COUCHDB_DB")
    max_context_depositions: int = Field(default=20, alias="MAX_CONTEXT_DEPOSITIONS")
    deposition_dir: str = Field(default="./sample_depositions", alias="DEPOSITION_DIR")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached ``Settings`` instance for the current process."""

    return Settings()
