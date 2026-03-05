# Copyright (c) 2026 Data-Blitz Inc. All rights reserved.
# License: Proprietary. See NOTICE.md.
# Author: Paul Harvener.

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
    memory_db: str = Field(default="memory", alias="MEMORY_DB")
    thought_stream_db: str = Field(default="thought_stream", alias="THOUGHT_STREAM_DB")
    rag_stream_db: str = Field(default="rag-stream", alias="RAG_STREAM_DB")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_file_path: str = Field(default="./logs/application.log", alias="LOG_FILE_PATH")
    neo4j_uri: str = Field(default="", alias="NEO4J_URI")
    neo4j_user: str = Field(default="", alias="NEO4J_USER")
    neo4j_password: str = Field(default="", alias="NEO4J_PASSWORD")
    neo4j_database: str = Field(default="neo4j", alias="NEO4J_DATABASE")
    neo4j_browser_url: str = Field(default="http://localhost:7474/browser/", alias="NEO4J_BROWSER_URL")
    graph_rag_embedding_provider: str = Field(default="openai", alias="GRAPH_RAG_EMBEDDING_PROVIDER")
    graph_rag_embedding_model: str = Field(default="text-embedding-3-small", alias="GRAPH_RAG_EMBEDDING_MODEL")
    graph_rag_embedding_dimensions: int = Field(default=1536, alias="GRAPH_RAG_EMBEDDING_DIMENSIONS")
    graph_rag_embedding_index: str = Field(default="resource_embeddings", alias="GRAPH_RAG_EMBEDDING_INDEX")
    graph_rag_embedding_node_label: str = Field(default="Resource", alias="GRAPH_RAG_EMBEDDING_NODE_LABEL")
    graph_rag_embedding_property: str = Field(default="embedding", alias="GRAPH_RAG_EMBEDDING_PROPERTY")
    grafana_url: str = Field(default="http://localhost:3000", alias="GRAFANA_URL")
    grafana_admin_user: str = Field(default="admin", alias="GRAFANA_ADMIN_USER")
    grafana_admin_password: str = Field(default="password", alias="GRAFANA_ADMIN_PASSWORD")
    ontology_dir: str = Field(default="./ontology", alias="ONTOLOGY_DIR")
    max_context_depositions: int = Field(default=20, alias="MAX_CONTEXT_DEPOSITIONS")
    deposition_dir: str = Field(default="./depositions", alias="DEPOSITION_DIR")
    deposition_extra_dirs: str = Field(default="", alias="DEPOSITION_EXTRA_DIRS")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached ``Settings`` instance for the current process."""

    return Settings()
