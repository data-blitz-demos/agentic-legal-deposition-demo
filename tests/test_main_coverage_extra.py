from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import HTTPException

from backend.app import main
from backend.app.models import (
    AdminPersonaPromptSections,
    GraphRagEmbeddingConfigRequest,
    IngestSchemaOption,
    IngestSchemaSaveRequest,
)


def test_default_graph_rag_embedding_config_falls_back_to_openai(monkeypatch):
    monkeypatch.setattr(main, "settings", SimpleNamespace(
        graph_rag_embedding_provider="unsupported",
        graph_rag_embedding_dimensions=0,
        graph_rag_embedding_model="",
        graph_rag_embedding_index="",
        graph_rag_embedding_node_label="",
        graph_rag_embedding_property="",
    ))

    payload = main._default_graph_rag_embedding_config()

    assert payload.provider == "openai"
    assert payload.dimensions == 1536


def test_load_graph_rag_embedding_config_falls_back_when_provider_invalid_or_validation_fails(monkeypatch):
    monkeypatch.setattr(
        main,
        "_default_graph_rag_embedding_config",
        lambda: main.GraphRagEmbeddingConfigResponse(
            enabled=False,
            provider="openai",
            model="text-embedding-3-small",
            dimensions=1536,
            index_name="idx",
            node_label="Resource",
            property_name="embedding",
            source="defaults",
            configured=False,
            last_saved_at=None,
        ),
    )
    monkeypatch.setattr(main.couchdb, "get_doc", lambda _doc_id: {"provider": "bad", "model": "m", "dimensions": 5})
    validate = Mock(side_effect=ValueError("bad"))
    monkeypatch.setattr(main.GraphRagEmbeddingConfigResponse, "model_validate", validate)

    payload = main._load_graph_rag_embedding_config()

    assert payload.provider == "openai"


def test_save_graph_rag_embedding_config_preserves_existing_rev(monkeypatch):
    monkeypatch.setattr(main.couchdb, "get_doc", lambda _doc_id: {"_rev": "1-abc"})
    saved = []
    monkeypatch.setattr(main.couchdb, "update_doc", lambda doc: saved.append(doc) or doc)

    payload = main._save_graph_rag_embedding_config(
        GraphRagEmbeddingConfigRequest(
            enabled=True,
            provider="openai",
            model="text-embedding-3-small",
            dimensions=1536,
            index_name="idx",
            node_label="Resource",
            property_name="embedding",
        )
    )

    assert saved[0]["_rev"] == "1-abc"
    assert payload.configured is True


def test_build_graph_rag_query_embedding_handles_empty_openai_missing_key_and_no_data(monkeypatch):
    config = GraphRagEmbeddingConfigRequest(
        enabled=True,
        provider="openai",
        model="text-embedding-3-small",
        dimensions=5,
        index_name="idx",
        node_label="Resource",
        property_name="embedding",
    )

    embedding, error = main._build_graph_rag_query_embedding("   ", config)
    assert embedding is None
    assert "Question is required" in error

    monkeypatch.setattr(main, "settings", SimpleNamespace(openai_api_key="", ollama_url="http://ollama"))
    embedding, error = main._build_graph_rag_query_embedding("contract", config)
    assert embedding is None
    assert error == "OPENAI_API_KEY is not configured."

    monkeypatch.setattr(main, "settings", SimpleNamespace(openai_api_key="key", ollama_url="http://ollama"))

    class NoDataResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": []}

    monkeypatch.setattr(main.httpx, "post", lambda *args, **kwargs: NoDataResponse())
    embedding, error = main._build_graph_rag_query_embedding("contract", config)
    assert embedding is None
    assert "did not include any vectors" in error


def test_build_graph_rag_query_embedding_uses_ollama_embeddings_fallback_and_handles_invalid_vector(monkeypatch):
    monkeypatch.setattr(main, "settings", SimpleNamespace(openai_api_key="key", ollama_url="http://ollama"))
    config = GraphRagEmbeddingConfigRequest(
        enabled=True,
        provider="ollama",
        model="nomic-embed",
        dimensions=5,
        index_name="idx",
        node_label="Resource",
        property_name="embedding",
    )

    class EmbeddingsResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"embeddings": [[1, 2, 3]]}

    monkeypatch.setattr(main.httpx, "post", lambda *args, **kwargs: EmbeddingsResponse())
    embedding, error = main._build_graph_rag_query_embedding("contract", config)
    assert embedding == [1.0, 2.0, 3.0]
    assert error is None

    class BadEmbeddingResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"embedding": ""}

    monkeypatch.setattr(main.httpx, "post", lambda *args, **kwargs: BadEmbeddingResponse())
    embedding, error = main._build_graph_rag_query_embedding("contract", config)
    assert embedding is None
    assert "usable vector" in error


def test_build_graph_rag_query_embedding_logs_warning_on_exception(monkeypatch):
    monkeypatch.setattr(main, "settings", SimpleNamespace(openai_api_key="key", ollama_url="http://ollama"))
    warning = Mock()
    monkeypatch.setattr(main.logger, "warning", warning)
    monkeypatch.setattr(main.httpx, "post", Mock(side_effect=RuntimeError("down")))
    config = GraphRagEmbeddingConfigRequest(
        enabled=True,
        provider="ollama",
        model="nomic-embed",
        dimensions=5,
        index_name="idx",
        node_label="Resource",
        property_name="embedding",
    )

    embedding, error = main._build_graph_rag_query_embedding("contract", config)

    assert embedding is None
    assert error == "down"
    warning.assert_called_once()


def test_resolve_deposition_browser_directory_strips_leading_dot_parts(monkeypatch, tmp_path):
    real_path = Path
    base = tmp_path / "depositions"
    nested = base / "default"
    nested.mkdir(parents=True)
    monkeypatch.setattr(main, "_deposition_browser_base_directory", lambda: base.resolve())

    class DotCandidate:
        parts = (".", "default")

        def expanduser(self):
            return self

        def is_absolute(self):
            return False

        def __fspath__(self):
            return "default"

    class PathProxy:
        def __call__(self, *args):
            if args == ("./default",):
                return DotCandidate()
            return real_path(*args)

        @staticmethod
        def cwd():
            return real_path.cwd()

    monkeypatch.setattr(main, "Path", PathProxy())

    _, current = main._resolve_deposition_browser_directory("./default")

    assert current == nested.resolve()


def test_toolathlon_helper_branches():
    assert main._is_toolathlon_tool_like_event(None) is False
    assert main._is_toolathlon_tool_like_event({"llm_provider": "openai"}) is True
    assert main._is_toolathlon_tool_like_event({"input_preview": "prompt"}) is True
    assert main._is_toolathlon_tool_like_event({"phase": "graph_rag_answer"}) is True


def test_compute_toolathlon_metrics_covers_dual_persona_and_error_recovery():
    events = [
        {"persona": "Persona:Legal Clerk", "phase": "graph_rag_context_ready"},
        {"persona": "Persona:Attorney", "phase": "chat_response", "llm_provider": "openai"},
    ] * 6
    error_events = [
        {"persona": "Persona:Legal Clerk", "phase": "ingest_error"},
        {"persona": "Persona:Attorney", "phase": "chat_error"},
    ]
    metrics = main._compute_toolathlon_metrics(
        [
            {"status": "completed", "trace": {"legal_clerk": events[:6], "attorney": events[6:]}},
            {"status": "completed", "trace": {"legal_clerk": error_events, "attorney": []}},
        ]
    )

    by_key = {item.key: item for item in metrics}
    assert by_key["toolathlon_multi_persona_completion_rate_pct"].value is not None
    assert by_key["toolathlon_error_recovery_rate_pct"].value is not None


def test_legacy_and_normalized_persona_prompt_sections_cover_marker_paths():
    parsed = main._legacy_persona_prompts_to_sections("System: intro\nAssistant: answer\nContext: facts")
    assert parsed.system == "intro"
    assert parsed.assistant == "answer"
    assert parsed.context == "facts"

    normalized = main._normalize_persona_prompt_sections(None, "Assistant: answer", None)
    assert normalized.assistant == "answer"

    context_only = main._normalize_persona_prompt_sections(None, "legacy context", "chat_user_context")
    assert context_only.context == "legacy context"


def test_legacy_persona_prompts_unknown_marker_falls_back_to_system(monkeypatch):
    real_compile = main.re.compile

    class FakeMatch:
        def group(self, index):
            return "Other" if index == 1 else "surprise"

    class FakeMarker:
        def match(self, line):
            return FakeMatch() if line == "Other: surprise" else None

    monkeypatch.setattr(main.re, "compile", lambda pattern, flags=0: FakeMarker() if "system|assistant|context" in pattern else real_compile(pattern, flags))

    parsed = main._legacy_persona_prompts_to_sections("Other: surprise")

    assert parsed.system == "surprise"


def test_list_admin_personas_skips_docs_without_prompt_content(monkeypatch):
    monkeypatch.setattr(
        main.couchdb,
        "find",
        lambda *_args, **_kwargs: [
            {
                "_id": "persona-1",
                "name": "Empty",
                "llm_provider": "openai",
                "llm_model": "gpt-5.2",
                "prompt_sections": {"system": "", "assistant": "", "context": ""},
                "created_at": "2026-01-01T00:00:00Z",
            }
        ],
    )

    assert main._list_admin_personas() == []


@pytest.mark.parametrize(
    ("persona_id", "question", "answer", "detail"),
    [
        ("", "Q", "A", "Persona id is required"),
        ("persona-1", "", "A", "Graph question is required"),
        ("persona-1", "Q", "", "Graph answer is required"),
    ],
)
def test_save_admin_persona_graph_session_validates_required_fields(persona_id, question, answer, detail):
    with pytest.raises(HTTPException, match=detail):
        main._save_admin_persona_graph_session(persona_id, question, answer)


def test_save_admin_persona_graph_session_requires_persona_doc(monkeypatch):
    monkeypatch.setattr(main.couchdb, "get_doc", lambda _doc_id: {"type": "not-persona"})

    with pytest.raises(HTTPException, match="Persona not found"):
        main._save_admin_persona_graph_session("persona-1", "Q", "A")


def test_grafana_access_info_falls_back_when_url_blank(monkeypatch):
    monkeypatch.setattr(main, "settings", SimpleNamespace(
        grafana_url="   ",
        grafana_admin_user="admin",
        grafana_admin_password="pw",
    ))

    payload = main.grafana_access_info()

    assert payload.url == "http://localhost:3000"


def test_lifespan_logs_langfuse_enabled_message(monkeypatch):
    info = Mock()
    monkeypatch.setattr(main.logger, "info", info)
    monkeypatch.setattr(main, "initialize_langfuse", lambda _settings: True)
    monkeypatch.setattr(main, "_ensure_startup_llm_connectivity", lambda: None)
    monkeypatch.setattr(main, "couchdb", SimpleNamespace(ensure_db=lambda: None, ensure_deposition_views=lambda: None, close=lambda: None))
    monkeypatch.setattr(main, "memory_couchdb", SimpleNamespace(ensure_db=lambda: None, close=lambda: None))
    monkeypatch.setattr(main, "trace_couchdb", SimpleNamespace(ensure_db=lambda: None, close=lambda: None))
    monkeypatch.setattr(main, "rag_couchdb", SimpleNamespace(ensure_db=lambda: None, close=lambda: None))
    monkeypatch.setattr(main, "neo4j_graph", SimpleNamespace(close=lambda: None))

    async def run_lifespan() -> None:
        async with main.lifespan(main.app):
            pass

    asyncio.run(run_lifespan())

    assert any(call.args and call.args[0] == "langfuse tracing enabled base_url=%s" for call in info.call_args_list)


def test_compute_correctness_drift_metrics_skips_non_dict_sessions():
    metrics = main._compute_correctness_drift_metrics(
        [None, {"status": "completed", "trace": {"legal_clerk": [], "attorney": []}}],
        [],
    )

    assert any(item.key == "model_mix_drift_jsd" for item in metrics)


def test_refresh_thought_stream_inventory_metrics_skips_non_dict_events(monkeypatch):
    monkeypatch.setattr(
        main,
        "_collect_runtime_trace_sessions",
        lambda: (
            [{"status": "completed", "trace": {"legal_clerk": ["bad"], "attorney": [{"persona": "Persona:Attorney", "phase": "chat_response"}]}}],
            True,
        ),
    )
    sync = Mock()
    monkeypatch.setattr(main, "sync_thought_stream_inventory", sync)

    main._refresh_thought_stream_inventory_metrics()

    event_counts, session_counts = sync.call_args.args
    assert event_counts[("Persona:Attorney", "chat_response")] == 1
    assert session_counts["completed"] == 1


@pytest.mark.parametrize(
    ("kwargs", "detail"),
    [
        ({"model": "   "}, "Embedding model is required."),
        ({"index_name": "   "}, "Embedding index name is required."),
        ({"node_label": "   "}, "Embedding node label is required."),
        ({"property_name": "   "}, "Embedding property name is required."),
    ],
)
def test_save_graph_rag_embedding_config_validates_fields(kwargs, detail):
    payload = {
        "enabled": True,
        "provider": "openai",
        "model": "text-embedding-3-small",
        "dimensions": 1536,
        "index_name": "idx",
        "node_label": "Resource",
        "property_name": "embedding",
    }
    payload.update(kwargs)

    with pytest.raises(HTTPException, match=detail):
        main.save_graph_rag_embedding_config(GraphRagEmbeddingConfigRequest(**payload))


def test_graph_rag_query_passes_langchain_config_when_present(monkeypatch):
    llm = Mock()
    llm.invoke.return_value = SimpleNamespace(content="Short answer: Answer.")
    monkeypatch.setattr(main, "_resolve_request_llm", lambda *_args, **_kwargs: ("openai", "gpt-5.2"))
    monkeypatch.setattr(main, "_ensure_request_llm_operational", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main, "_load_graph_rag_embedding_config", lambda: main._default_graph_rag_embedding_config())
    monkeypatch.setattr(main, "_build_graph_rag_query_embedding", lambda _question, _config: (None, None))
    monkeypatch.setattr(
        main,
        "neo4j_graph",
        SimpleNamespace(retrieve_context=lambda *args, **kwargs: {"resource_count": 0, "resources": [], "terms": [], "retrieval_mode": "keyword", "query_embedding_used": False, "vector_index_name": "idx", "vector_error": None, "context_text": "ctx"}),
    )
    monkeypatch.setattr(main, "_append_trace_events", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main, "_append_rag_stream_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main, "build_chat_model", lambda *_args, **_kwargs: llm)
    monkeypatch.setattr(main, "build_langchain_config", lambda *_args, **_kwargs: {"callbacks": ["cb"]})

    response = main.query_graph_rag(
        main.GraphRagQueryRequest(question="What is the rule?", top_k=3, use_rag=True, stream_rag=False)
    )

    assert response.answer.startswith("Short answer:")
    assert llm.invoke.call_args.kwargs["config"] == {"callbacks": ["cb"]}


def test_normalize_ingest_schema_key_and_custom_schema_helpers(monkeypatch):
    with pytest.raises(HTTPException, match="Schema key is required"):
        main._normalize_ingest_schema_key("!!!")

    monkeypatch.setattr(main.couchdb, "find", Mock(side_effect=RuntimeError("down")))
    assert main._list_custom_ingest_schema_docs() == []

    monkeypatch.setattr(main.couchdb, "find", lambda *_args, **_kwargs: iter([{"key": "custom", "schema": {}}]))
    assert main._list_custom_ingest_schema_docs() == [{"key": "custom", "schema": {}}]


def test_build_ingest_schema_options_skips_duplicates_and_invalid_custom(monkeypatch):
    monkeypatch.setattr(
        main,
        "list_schema_options",
        lambda: [
            {"key": "builtin", "file_name": "builtin.json", "mode": "native"},
            {"key": "", "file_name": "blank.json", "mode": "native"},
            {"key": "builtin", "file_name": "dup.json", "mode": "native"},
            {"key": "broken", "file_name": "broken.json", "mode": "native"},
        ],
    )
    monkeypatch.setattr(main, "load_schema", Mock(side_effect=[{"title": "Builtin", "type": "object"}, ValueError("bad")]))
    monkeypatch.setattr(
        main,
        "_list_custom_ingest_schema_docs",
        lambda: [{"key": "builtin", "schema": {}}, {"key": "custom", "schema": "bad"}],
    )

    options = main._build_ingest_schema_options()

    assert [item.key for item in options] == ["builtin"]


def test_resolve_ingest_schema_selection_and_schema_save_delete_branches(monkeypatch):
    monkeypatch.setattr(main, "_build_ingest_schema_options", lambda: [])
    with pytest.raises(HTTPException, match="schema was not found"):
        main._resolve_ingest_schema_selection("missing")

    monkeypatch.setattr(
        main,
        "_build_ingest_schema_options",
        lambda: [IngestSchemaOption(key="builtin", file_name="x", mode="native", builtin=True, removable=False, schema_payload={})],
    )
    with pytest.raises(HTTPException, match="Schema payload must be a JSON object"):
        main.save_ingest_schema(IngestSchemaSaveRequest.model_construct(key="custom", schema_payload="bad"))

    monkeypatch.setattr(main.couchdb, "get_doc", lambda _doc_id: {"_rev": "1-abc", "created_at": "2026-01-01T00:00:00Z"})
    saved_docs: list[dict] = []
    monkeypatch.setattr(main.couchdb, "save_doc", lambda doc: saved_docs.append(doc) or {"_id": doc["_id"]})
    main.save_ingest_schema(IngestSchemaSaveRequest(key="custom", schema_payload={"type": "object"}))
    assert saved_docs[0]["_rev"] == "1-abc"

    monkeypatch.setattr(main.couchdb, "get_doc", Mock(side_effect=RuntimeError("missing")))
    with pytest.raises(HTTPException, match="Ingest schema not found"):
        main.delete_ingest_schema("custom")

    monkeypatch.setattr(main.couchdb, "get_doc", lambda _doc_id: {"type": "other"})
    with pytest.raises(HTTPException, match="Ingest schema not found"):
        main.delete_ingest_schema("custom")
