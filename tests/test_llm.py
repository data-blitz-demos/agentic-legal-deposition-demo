from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from backend.app import llm as llm_module


@pytest.fixture(autouse=True)
def clear_readiness_cache():
    llm_module.clear_llm_readiness_cache()
    yield
    llm_module.clear_llm_readiness_cache()


def make_settings(**overrides):
    base = {
        "default_llm_provider": "openai",
        "model_name": "gpt-5.2",
        "openai_models": "gpt-5.2,gpt-5.1-mini",
        "openai_api_key": "key",
        "ollama_url": "http://localhost:11434",
        "ollama_default_model": "llama3.3",
        "ollama_models": "llama3.3,mistral",
        "ollama_keep_alive": "10m",
        "llm_readiness_ttl_seconds": 120,
        "llm_probe_timeout_seconds": 12,
        "llm_options_probe_workers": 3,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_parse_model_list_filters_empty_values():
    assert llm_module.parse_model_list("a, ,b,, c") == ["a", "b", "c"]


def test_dedupe_preserve_order():
    assert llm_module._dedupe_preserve_order(["a", "b", "a", "c"]) == ["a", "b", "c"]


def test_resolve_llm_selection_rejects_invalid_provider():
    settings = make_settings()

    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        llm_module.resolve_llm_selection(settings, "invalid", None)


def test_resolve_llm_selection_uses_explicit_model():
    settings = make_settings()

    provider, model = llm_module.resolve_llm_selection(settings, "ollama", "qwen2.5")

    assert provider == "ollama"
    assert model == "qwen2.5"


def test_resolve_llm_selection_uses_openai_fallback_model_list():
    settings = make_settings(model_name="", openai_models="gpt-4.1")

    provider, model = llm_module.resolve_llm_selection(settings, "openai", None)

    assert provider == "openai"
    assert model == "gpt-4.1"


def test_resolve_llm_selection_uses_ollama_fallback_model_list():
    settings = make_settings(ollama_default_model="", ollama_models="mistral")

    provider, model = llm_module.resolve_llm_selection(settings, "ollama", None)

    assert provider == "ollama"
    assert model == "mistral"


def test_resolve_llm_selection_uses_ollama_default_model():
    settings = make_settings(ollama_default_model="llama3.3")

    provider, model = llm_module.resolve_llm_selection(settings, "ollama", None)

    assert provider == "ollama"
    assert model == "llama3.3"


def test_build_chat_model_openai(monkeypatch):
    captured = {}

    class StubChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(llm_module, "ChatOpenAI", StubChatOpenAI)
    settings = make_settings()

    model = llm_module.build_chat_model(settings, "openai", "gpt-5.2", temperature=0.4)

    assert isinstance(model, StubChatOpenAI)
    assert captured == {"model": "gpt-5.2", "api_key": "key", "temperature": 0.4}


def test_build_chat_model_ollama_without_dependency(monkeypatch):
    monkeypatch.setattr(llm_module, "ChatOllama", None)
    settings = make_settings()

    with pytest.raises(RuntimeError, match="langchain-ollama"):
        llm_module.build_chat_model(settings, "ollama", "llama3.3", temperature=0)


def test_build_chat_model_ollama(monkeypatch):
    captured = {}

    class StubChatOllama:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(llm_module, "ChatOllama", StubChatOllama)
    settings = make_settings()

    model = llm_module.build_chat_model(settings, "ollama", "llama3.3", temperature=0.1)

    assert isinstance(model, StubChatOllama)
    assert captured == {
        "model": "llama3.3",
        "base_url": "http://localhost:11434",
        "temperature": 0.1,
        "keep_alive": "10m",
    }


def test_fetch_ollama_models_success(monkeypatch):
    class StubResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"models": [{"name": "llama3.3"}, {"name": "mistral"}, {"name": "llama3.3"}]}

    monkeypatch.setattr(llm_module.httpx, "get", lambda _url, timeout: StubResponse())

    models = llm_module.fetch_ollama_models(make_settings())

    assert models == ["llama3.3", "mistral"]


def test_fetch_ollama_models_error_returns_empty(monkeypatch):
    def raise_error(_url, timeout):
        raise RuntimeError("offline")

    monkeypatch.setattr(llm_module.httpx, "get", raise_error)

    models = llm_module.fetch_ollama_models(make_settings())

    assert models == []


def test_list_llm_options_uses_live_ollama_and_configured_openai(monkeypatch):
    monkeypatch.setattr(llm_module, "fetch_ollama_models", lambda _settings: ["llama3.3", "qwen2.5"])

    options = llm_module.list_llm_options(make_settings(openai_models="gpt-5.2,gpt-4.1"))

    assert options[0] == {"provider": "openai", "model": "gpt-5.2", "label": "ChatGPT - gpt-5.2"}
    assert {"provider": "ollama", "model": "qwen2.5", "label": "Ollama - qwen2.5"} in options


def test_list_llm_options_uses_ollama_fallback_when_tags_unavailable(monkeypatch):
    monkeypatch.setattr(llm_module, "fetch_ollama_models", lambda _settings: [])

    options = llm_module.list_llm_options(make_settings(ollama_default_model="llama3.3", ollama_models="mistral"))

    assert {"provider": "ollama", "model": "llama3.3", "label": "Ollama - llama3.3"} in options
    assert {"provider": "ollama", "model": "mistral", "label": "Ollama - mistral"} in options


def test_list_llm_options_skips_blank_openai_entries(monkeypatch):
    monkeypatch.setattr(llm_module, "fetch_ollama_models", lambda _settings: [])

    options = llm_module.list_llm_options(make_settings(model_name="", openai_models="gpt-5.2"))

    assert options[0] == {"provider": "openai", "model": "gpt-5.2", "label": "ChatGPT - gpt-5.2"}


def test_possible_fix_from_exception_for_openai_missing_key():
    settings = make_settings(openai_api_key="")

    fix = llm_module._possible_fix_from_exception(settings, "openai", "gpt-5.2", RuntimeError("boom"))

    assert "OPENAI_API_KEY" in fix


def test_possible_fix_from_exception_for_ollama_connection_refused():
    settings = make_settings(ollama_url="http://localhost:11434")

    fix = llm_module._possible_fix_from_exception(
        settings,
        "ollama",
        "llama3.3",
        RuntimeError("Connection refused"),
    )

    assert "host.docker.internal" in fix


def test_possible_fix_from_exception_for_openai_model_not_found():
    settings = make_settings()

    fix = llm_module._possible_fix_from_exception(
        settings,
        "openai",
        "missing-model",
        RuntimeError("404 model not found"),
    )

    assert "valid OpenAI model name" in fix


def test_possible_fix_from_exception_for_openai_generic():
    settings = make_settings()

    fix = llm_module._possible_fix_from_exception(
        settings,
        "openai",
        "gpt-5.2",
        RuntimeError("temporary issue"),
    )

    assert "OpenAI connectivity" in fix


def test_possible_fix_from_exception_for_openai_output_parsing():
    settings = make_settings()

    fix = llm_module._possible_fix_from_exception(
        settings,
        "openai",
        "gpt-5.2",
        RuntimeError("OUTPUT_PARSING_FAILURE"),
    )

    assert "schema" in fix.lower()


def test_possible_fix_from_exception_for_openai_timeout():
    settings = make_settings()

    fix = llm_module._possible_fix_from_exception(
        settings,
        "openai",
        "gpt-5.2",
        RuntimeError("timed out"),
    )

    assert "LLM_PROBE_TIMEOUT_SECONDS" in fix


def test_possible_fix_from_exception_for_ollama_model_not_found():
    settings = make_settings()

    fix = llm_module._possible_fix_from_exception(
        settings,
        "ollama",
        "llama3.3",
        RuntimeError("unknown model"),
    )

    assert "ollama pull" in fix


def test_possible_fix_from_exception_for_ollama_generic():
    settings = make_settings()

    fix = llm_module._possible_fix_from_exception(
        settings,
        "ollama",
        "llama3.3",
        RuntimeError("other"),
    )

    assert "OLLAMA_URL" in fix


def test_possible_fix_from_exception_for_ollama_output_parsing():
    settings = make_settings()

    fix = llm_module._possible_fix_from_exception(
        settings,
        "ollama",
        "llama3.3",
        RuntimeError("failed to parse DepositionSchema"),
    )

    assert "non-conformant structured output" in fix


def test_possible_fix_from_exception_for_ollama_timeout():
    settings = make_settings(llm_probe_timeout_seconds=7)

    fix = llm_module._possible_fix_from_exception(
        settings,
        "ollama",
        "llama3.3",
        RuntimeError("timed out"),
    )

    assert "7s" in fix
    assert "smaller local model" in fix


def test_normalize_ollama_model_names():
    normalized = llm_module._normalize_ollama_model_names(["llama3.3", "mistral:latest"])

    assert "llama3.3" in normalized
    assert "llama3.3:latest" in normalized
    assert "mistral" in normalized
    assert "mistral:latest" in normalized


def test_normalize_ollama_model_names_skips_empty_items():
    normalized = llm_module._normalize_ollama_model_names([" ", "llama3.3"])

    assert normalized == {"llama3.3", "llama3.3:latest"}


def test_assert_ollama_model_loaded_missing_service(monkeypatch):
    monkeypatch.setattr(llm_module, "fetch_ollama_models", lambda _settings: [])

    with pytest.raises(llm_module.LLMOperationalError, match="not reachable"):
        llm_module._assert_ollama_model_loaded(make_settings(), "llama3.3")


def test_assert_ollama_model_loaded_model_not_found(monkeypatch):
    monkeypatch.setattr(llm_module, "fetch_ollama_models", lambda _settings: ["mistral:latest"])

    with pytest.raises(llm_module.LLMOperationalError, match="not loaded"):
        llm_module._assert_ollama_model_loaded(make_settings(), "llama3.3")


def test_assert_ollama_model_loaded_accepts_latest_alias(monkeypatch):
    monkeypatch.setattr(llm_module, "fetch_ollama_models", lambda _settings: ["llama3.3:latest"])

    llm_module._assert_ollama_model_loaded(make_settings(), "llama3.3")


def test_probe_structured_output_empty_ready_raises(monkeypatch):
    class StubParser:
        def invoke(self, _messages):
            return SimpleNamespace(ready="")

    class StubLLM:
        def with_structured_output(self, _schema):
            return StubParser()

    monkeypatch.setattr(llm_module, "build_chat_model", lambda *_args, **_kwargs: StubLLM())

    with pytest.raises(RuntimeError, match="empty value"):
        llm_module._probe_structured_output(make_settings(), "openai", "gpt-5.2")


def test_probe_structured_output_timeout(monkeypatch):
    class StubFuture:
        def result(self, timeout):  # noqa: ARG002 - stub signature mirrors Future
            raise llm_module.FutureTimeoutError()

        def cancel(self):
            return True

    class StubExecutor:
        def __init__(self, max_workers):  # noqa: ARG002 - stub signature mirrors executor
            pass

        def submit(self, _fn):
            return StubFuture()

        def shutdown(self, wait=False, cancel_futures=True):  # noqa: FBT002
            return None

    monkeypatch.setattr(llm_module, "ThreadPoolExecutor", StubExecutor)

    with pytest.raises(RuntimeError, match="timed out"):
        llm_module._probe_structured_output(
            make_settings(llm_probe_timeout_seconds=2),
            "openai",
            "gpt-5.2",
        )


def test_ensure_llm_operational_openai_missing_key():
    with pytest.raises(llm_module.LLMOperationalError, match="OPENAI_API_KEY"):
        llm_module.ensure_llm_operational(make_settings(openai_api_key=""), "openai", "gpt-5.2")


def test_ensure_llm_operational_probe_failure_wrapped(monkeypatch):
    monkeypatch.setattr(llm_module, "_probe_structured_output", Mock(side_effect=RuntimeError("bad probe")))
    monkeypatch.setattr(llm_module, "_assert_ollama_model_loaded", lambda *_args, **_kwargs: None)

    with pytest.raises(llm_module.LLMOperationalError, match="failed readiness probe"):
        llm_module.ensure_llm_operational(make_settings(), "openai", "gpt-5.2")


def test_ensure_llm_operational_reraises_operational_error(monkeypatch):
    op_error = llm_module.LLMOperationalError("probe failed", "fix")
    monkeypatch.setattr(llm_module, "_probe_structured_output", Mock(side_effect=op_error))

    with pytest.raises(llm_module.LLMOperationalError, match="probe failed"):
        llm_module.ensure_llm_operational(make_settings(), "openai", "gpt-5.2")


def test_ensure_llm_operational_success(monkeypatch):
    monkeypatch.setattr(llm_module, "_probe_structured_output", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(llm_module, "_assert_ollama_model_loaded", lambda *_args, **_kwargs: None)

    llm_module.ensure_llm_operational(make_settings(), "openai", "gpt-5.2")
    llm_module.ensure_llm_operational(make_settings(), "ollama", "llama3.3")


def test_ensure_llm_operational_uses_ttl_cache(monkeypatch):
    probe = Mock()
    monkeypatch.setattr(llm_module, "_probe_structured_output", probe)
    monkeypatch.setattr(llm_module, "_assert_ollama_model_loaded", lambda *_args, **_kwargs: None)
    now = {"value": 1000.0}
    monkeypatch.setattr(llm_module.time, "time", lambda: now["value"])
    settings = make_settings(llm_readiness_ttl_seconds=120)

    llm_module.ensure_llm_operational(settings, "ollama", "llama3.3")
    llm_module.ensure_llm_operational(settings, "ollama", "llama3.3")

    assert probe.call_count == 1


def test_ensure_llm_operational_force_probe_bypasses_cache(monkeypatch):
    probe = Mock()
    monkeypatch.setattr(llm_module, "_probe_structured_output", probe)
    monkeypatch.setattr(llm_module, "_assert_ollama_model_loaded", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(llm_module.time, "time", lambda: 1000.0)
    settings = make_settings(llm_readiness_ttl_seconds=120)

    llm_module.ensure_llm_operational(settings, "ollama", "llama3.3")
    llm_module.ensure_llm_operational(settings, "ollama", "llama3.3", force_probe=True)

    assert probe.call_count == 2


def test_ensure_llm_operational_no_cache_when_ttl_disabled(monkeypatch):
    probe = Mock()
    monkeypatch.setattr(llm_module, "_probe_structured_output", probe)
    monkeypatch.setattr(llm_module, "_assert_ollama_model_loaded", lambda *_args, **_kwargs: None)
    settings = make_settings(llm_readiness_ttl_seconds=0)

    llm_module.ensure_llm_operational(settings, "ollama", "llama3.3")
    llm_module.ensure_llm_operational(settings, "ollama", "llama3.3")

    assert probe.call_count == 2


def test_clear_llm_readiness_cache_forces_reprobe(monkeypatch):
    probe = Mock()
    monkeypatch.setattr(llm_module, "_probe_structured_output", probe)
    monkeypatch.setattr(llm_module, "_assert_ollama_model_loaded", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(llm_module.time, "time", lambda: 1000.0)
    settings = make_settings(llm_readiness_ttl_seconds=120)

    llm_module.ensure_llm_operational(settings, "ollama", "llama3.3")
    llm_module.clear_llm_readiness_cache()
    llm_module.ensure_llm_operational(settings, "ollama", "llama3.3")

    assert probe.call_count == 2


def test_quick_llm_option_check_openai_key_required():
    with pytest.raises(llm_module.LLMOperationalError, match="OPENAI_API_KEY"):
        llm_module._quick_llm_option_check(make_settings(openai_api_key=""), "openai", "gpt-5.2")


def test_quick_llm_option_check_openai_with_key_succeeds():
    llm_module._quick_llm_option_check(make_settings(openai_api_key="key"), "openai", "gpt-5.2")


def test_quick_llm_option_check_ollama_uses_model_assertion(monkeypatch):
    assertion = Mock()
    monkeypatch.setattr(llm_module, "_assert_ollama_model_loaded", assertion)

    llm_module._quick_llm_option_check(make_settings(), "ollama", "llama3.3")

    assertion.assert_called_once()


def test_get_llm_option_status_success(monkeypatch):
    quick = Mock()
    ensure = Mock()
    monkeypatch.setattr(llm_module, "_quick_llm_option_check", quick)
    monkeypatch.setattr(llm_module, "ensure_llm_operational", ensure)
    settings = make_settings()

    status = llm_module.get_llm_option_status(settings, "openai", "gpt-5.2")

    assert status == {"operational": True, "error": None, "possible_fix": None}
    quick.assert_called_once_with(settings, "openai", "gpt-5.2")
    ensure.assert_not_called()


def test_get_llm_option_status_force_probe_success(monkeypatch):
    ensure = Mock()
    quick = Mock()
    monkeypatch.setattr(llm_module, "ensure_llm_operational", ensure)
    monkeypatch.setattr(llm_module, "_quick_llm_option_check", quick)
    settings = make_settings()

    status = llm_module.get_llm_option_status(settings, "openai", "gpt-5.2", force_probe=True)

    assert status == {"operational": True, "error": None, "possible_fix": None}
    ensure.assert_called_once_with(settings, "openai", "gpt-5.2", force_probe=True)
    quick.assert_not_called()


def test_get_llm_option_status_failure(monkeypatch):
    quick = Mock(side_effect=llm_module.LLMOperationalError("bad", "fix"))
    ensure = Mock()
    monkeypatch.setattr(
        llm_module,
        "_quick_llm_option_check",
        quick,
    )
    monkeypatch.setattr(llm_module, "ensure_llm_operational", ensure)

    settings = make_settings()
    status = llm_module.get_llm_option_status(settings, "openai", "gpt-5.2")

    assert status["operational"] is False
    assert status["possible_fix"] == "fix"
    quick.assert_called_once_with(settings, "openai", "gpt-5.2")
    ensure.assert_not_called()


def test_get_llm_option_status_force_probe_failure(monkeypatch):
    ensure = Mock(side_effect=llm_module.LLMOperationalError("bad", "fix"))
    quick = Mock()
    monkeypatch.setattr(llm_module, "ensure_llm_operational", ensure)
    monkeypatch.setattr(llm_module, "_quick_llm_option_check", quick)

    settings = make_settings()
    status = llm_module.get_llm_option_status(settings, "openai", "gpt-5.2", force_probe=True)

    assert status["operational"] is False
    assert status["possible_fix"] == "fix"
    ensure.assert_called_once_with(settings, "openai", "gpt-5.2", force_probe=True)
    quick.assert_not_called()


def test_llm_failure_message_includes_fix():
    message = llm_module.llm_failure_message(
        make_settings(),
        "openai",
        "gpt-5.2",
        RuntimeError("unauthorized"),
    )

    assert "Possible fix:" in message
