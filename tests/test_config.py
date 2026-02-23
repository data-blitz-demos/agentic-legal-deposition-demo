from __future__ import annotations

from backend.app.config import get_settings


def test_get_settings_reads_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("MODEL_NAME", "gpt-test")
    monkeypatch.setenv("OPENAI_MODELS", "gpt-test,gpt-other")
    monkeypatch.setenv("DEFAULT_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_URL", "http://localhost:11434")
    monkeypatch.setenv("OLLAMA_DEFAULT_MODEL", "llama3.3")
    monkeypatch.setenv("OLLAMA_MODELS", "llama3.3,mistral")
    monkeypatch.setenv("OLLAMA_KEEP_ALIVE", "15m")
    monkeypatch.setenv("LLM_READINESS_TTL_SECONDS", "45")
    monkeypatch.setenv("LLM_PROBE_TIMEOUT_SECONDS", "9")
    monkeypatch.setenv("LLM_OPTIONS_PROBE_WORKERS", "4")
    monkeypatch.setenv("COUCHDB_URL", "http://example.test:5984")
    monkeypatch.setenv("COUCHDB_DB", "db-test")
    monkeypatch.setenv("MAX_CONTEXT_DEPOSITIONS", "9")
    monkeypatch.setenv("DEPOSITION_DIR", "/tmp/deps")

    settings = get_settings()

    assert settings.openai_api_key == "test-key"
    assert settings.model_name == "gpt-test"
    assert settings.openai_models == "gpt-test,gpt-other"
    assert settings.default_llm_provider == "ollama"
    assert settings.ollama_url == "http://localhost:11434"
    assert settings.ollama_default_model == "llama3.3"
    assert settings.ollama_models == "llama3.3,mistral"
    assert settings.ollama_keep_alive == "15m"
    assert settings.llm_readiness_ttl_seconds == 45
    assert settings.llm_probe_timeout_seconds == 9
    assert settings.llm_options_probe_workers == 4
    assert settings.couchdb_url == "http://example.test:5984"
    assert settings.couchdb_db == "db-test"
    assert settings.max_context_depositions == 9
    assert settings.deposition_dir == "/tmp/deps"


def test_get_settings_is_cached(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "first")
    first = get_settings()

    monkeypatch.setenv("MODEL_NAME", "second")
    second = get_settings()

    assert first is second
    assert second.model_name == "first"
