from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import HTTPException

from backend.app import main
from backend.app.models import (
    ChatRequest,
    ContradictionFinding,
    ContradictionReasonRequest,
    IngestCaseRequest,
)


@pytest.fixture(autouse=True)
def stub_llm_readiness(monkeypatch):
    monkeypatch.setattr(main, "ensure_llm_operational", lambda *_args, **_kwargs: None)


def test_path_has_glob_detection():
    assert main._path_has_glob(Path("/tmp/*.txt")) is True
    assert main._path_has_glob(Path("/tmp/depositions")) is False


def test_collect_txt_files_from_directory_file_and_glob(tmp_path):
    dep_dir = tmp_path / "deps"
    dep_dir.mkdir()
    a = dep_dir / "a.txt"
    b = dep_dir / "b.TXT"
    c = dep_dir / "c.pdf"
    a.write_text("a", encoding="utf-8")
    b.write_text("b", encoding="utf-8")
    c.write_text("c", encoding="utf-8")

    directory_files = main._collect_txt_files(dep_dir)
    single_file = main._collect_txt_files(a)
    glob_files = main._collect_txt_files(dep_dir / "*.txt")
    non_txt_file = main._collect_txt_files(c)

    assert directory_files == [a, b]
    assert single_file == [a]
    assert glob_files == [a]
    assert non_txt_file == []


def test_build_ingest_candidates_handles_container_mapping(monkeypatch):
    monkeypatch.setattr(main, "app_root", Path("/workspace/app"))
    monkeypatch.setattr(main, "settings", SimpleNamespace(deposition_dir="/host/deps"))

    candidates = main._build_ingest_candidates("/data/depositions/folder")

    assert Path("/data/depositions/folder") in candidates
    assert Path("/host/deps/folder") in candidates
    assert Path("/workspace/app/sample_depositions/folder") in candidates


def test_build_ingest_candidates_relative_input_and_relative_config(monkeypatch):
    monkeypatch.setattr(main, "app_root", Path("/workspace/app"))
    monkeypatch.setattr(main, "settings", SimpleNamespace(deposition_dir="relative_deps"))

    candidates = main._build_ingest_candidates("my-folder")

    assert Path("my-folder") in candidates
    assert Path("/workspace/app/my-folder") in candidates

    mapped = main._build_ingest_candidates("/data/depositions/sub")
    assert Path("/workspace/app/relative_deps/sub") in mapped
    assert Path.cwd() / "relative_deps/sub" in mapped


def test_build_ingest_candidates_deduplicates(monkeypatch):
    monkeypatch.setattr(main, "app_root", Path("/workspace/app"))
    monkeypatch.setattr(main, "settings", SimpleNamespace(deposition_dir="sample_depositions"))

    candidates = main._build_ingest_candidates("/data/depositions")

    assert candidates.count(Path("/workspace/app/sample_depositions")) == 1


def test_build_ingest_candidates_non_container_path(monkeypatch):
    monkeypatch.setattr(main, "app_root", Path("/workspace/app"))
    monkeypatch.setattr(main, "settings", SimpleNamespace(deposition_dir="/host/deps"))

    candidates = main._build_ingest_candidates("/outside/path")

    assert Path("/host/deps/path") not in candidates
    assert Path("/outside/path") in candidates


def test_lifespan_ensures_and_closes_couchdb(monkeypatch):
    couchdb = Mock()
    startup = Mock()
    monkeypatch.setattr(main, "couchdb", couchdb)
    monkeypatch.setattr(main, "_ensure_startup_llm_connectivity", startup)

    async def run_lifespan() -> None:
        async with main.lifespan(main.app):
            pass

    asyncio.run(run_lifespan())

    startup.assert_called_once_with()
    couchdb.ensure_db.assert_called_once_with()
    couchdb.close.assert_called_once_with()


def test_lifespan_stops_when_startup_llm_check_fails(monkeypatch):
    couchdb = Mock()
    monkeypatch.setattr(main, "couchdb", couchdb)
    monkeypatch.setattr(
        main,
        "_ensure_startup_llm_connectivity",
        Mock(side_effect=RuntimeError("startup failed")),
    )

    async def run_lifespan() -> None:
        async with main.lifespan(main.app):
            pass

    with pytest.raises(RuntimeError, match="startup failed"):
        asyncio.run(run_lifespan())

    couchdb.ensure_db.assert_not_called()
    couchdb.close.assert_not_called()


def test_resolve_ingest_txt_files_deduplicates(monkeypatch, tmp_path):
    dep_dir = tmp_path / "deps"
    dep_dir.mkdir()
    txt = dep_dir / "a.txt"
    txt.write_text("a", encoding="utf-8")

    monkeypatch.setattr(main, "_build_ingest_candidates", lambda _path: [dep_dir, txt])

    files = main._resolve_ingest_txt_files("ignored")

    assert files == [txt]


def test_resolve_ingest_txt_files_raises_when_no_matches(monkeypatch):
    empty = Path("/tmp/does-not-exist")
    monkeypatch.setattr(main, "_build_ingest_candidates", lambda _path: [empty])

    with pytest.raises(HTTPException) as exc:
        main._resolve_ingest_txt_files("missing")

    assert exc.value.status_code == 400
    assert "No .txt deposition files found" in exc.value.detail


def test_root_returns_index_file_response():
    response = main.root()
    assert str(response.path).endswith("frontend/index.html")


def test_get_llm_options_returns_payload(monkeypatch):
    monkeypatch.setattr(main, "resolve_llm_selection", lambda *_args: ("openai", "gpt-5.2"))
    monkeypatch.setattr(
        main,
        "list_llm_options",
        lambda _settings: [{"provider": "openai", "model": "gpt-5.2", "label": "ChatGPT - gpt-5.2"}],
    )
    monkeypatch.setattr(
        main,
        "get_llm_option_status",
        lambda _settings, _provider, _model, *, force_probe=False: {
            "operational": True,
            "error": None,
            "possible_fix": None,
        },
    )

    payload = main.get_llm_options()

    assert payload.selected_provider == "openai"
    assert payload.selected_model == "gpt-5.2"
    assert payload.options[0].label == "ChatGPT - gpt-5.2"


def test_get_llm_options_force_probe_forwards_flag(monkeypatch):
    monkeypatch.setattr(main, "resolve_llm_selection", lambda *_args: ("openai", "gpt-5.2"))
    monkeypatch.setattr(
        main,
        "list_llm_options",
        lambda _settings: [{"provider": "openai", "model": "gpt-5.2", "label": "ChatGPT - gpt-5.2"}],
    )
    status = Mock(
        return_value={
            "operational": True,
            "error": None,
            "possible_fix": None,
        }
    )
    monkeypatch.setattr(main, "get_llm_option_status", status)

    payload = main.get_llm_options(force_probe=True)

    assert payload.options[0].label == "ChatGPT - gpt-5.2"
    status.assert_called_once_with(main.settings, "openai", "gpt-5.2", force_probe=True)


def test_get_llm_options_force_probe_parallel_branch(monkeypatch):
    monkeypatch.setattr(main, "settings", SimpleNamespace(llm_options_probe_workers=2))
    monkeypatch.setattr(main, "resolve_llm_selection", lambda *_args: ("openai", "gpt-5.2"))
    monkeypatch.setattr(
        main,
        "list_llm_options",
        lambda _settings: [
            {"provider": "openai", "model": "gpt-5.2", "label": "ChatGPT - gpt-5.2"},
            {"provider": "ollama", "model": "llama3.3", "label": "Ollama - llama3.3"},
        ],
    )
    monkeypatch.setattr(
        main,
        "get_llm_option_status",
        lambda _settings, _provider, _model, *, force_probe=False: {
            "operational": True,
            "error": None,
            "possible_fix": None,
        },
    )

    payload = main.get_llm_options(force_probe=True)

    assert len(payload.options) == 2
    assert payload.options[1].label == "Ollama - llama3.3"


def test_resolve_request_llm_wraps_validation_errors(monkeypatch):
    monkeypatch.setattr(main, "resolve_llm_selection", Mock(side_effect=ValueError("Unsupported")))

    with pytest.raises(HTTPException) as exc:
        main._resolve_request_llm("bad", "x")

    assert exc.value.status_code == 400
    assert "Unsupported" in exc.value.detail


def test_ensure_request_llm_operational_wraps_errors(monkeypatch):
    monkeypatch.setattr(main, "ensure_llm_operational", Mock(side_effect=main.LLMOperationalError("bad", "fix it")))

    with pytest.raises(HTTPException) as exc:
        main._ensure_request_llm_operational("openai", "gpt-5.2")

    assert exc.value.status_code == 503
    assert "Possible fix: fix it" in exc.value.detail


def test_ensure_startup_llm_connectivity_success(monkeypatch):
    monkeypatch.setattr(main, "_resolve_request_llm", lambda *_args: ("openai", "gpt-5.2"))
    ready = Mock()
    monkeypatch.setattr(main, "ensure_llm_operational", ready)

    main._ensure_startup_llm_connectivity()

    ready.assert_called_once_with(main.settings, "openai", "gpt-5.2")


def test_ensure_startup_llm_connectivity_failure(monkeypatch):
    monkeypatch.setattr(main, "_resolve_request_llm", lambda *_args: ("openai", "gpt-5.2"))
    monkeypatch.setattr(
        main,
        "ensure_llm_operational",
        Mock(side_effect=main.LLMOperationalError("bad", "set OPENAI_API_KEY")),
    )

    with pytest.raises(RuntimeError) as exc:
        main._ensure_startup_llm_connectivity()

    assert "Startup failed" in str(exc.value)
    assert "Possible fix: set OPENAI_API_KEY" in str(exc.value)


def test_ingest_case_success(monkeypatch):
    file_a = Path("/tmp/a.txt")
    file_b = Path("/tmp/b.txt")
    monkeypatch.setattr(main, "_resolve_ingest_txt_files", lambda _path: [file_a, file_b])

    workflow = Mock()
    workflow.run.side_effect = [
        {"deposition_doc": {"_id": "dep:1"}},
        {"deposition_doc": {"_id": "dep:2"}},
    ]
    workflow.reassess_case.return_value = [
        {
            "_id": "dep:1",
            "file_name": "a.txt",
            "witness_name": "Jane",
            "contradiction_score": 10,
            "flagged": False,
        },
        {
            "_id": "dep:2",
            "file_name": "b.txt",
            "witness_name": "Alan",
            "contradiction_score": 20,
            "flagged": True,
        },
    ]
    couchdb = Mock()

    monkeypatch.setattr(main, "workflow", workflow)
    monkeypatch.setattr(main, "couchdb", couchdb)

    response = main.ingest_case(
        IngestCaseRequest(
            case_id="case-1",
            directory="/tmp",
            llm_provider="ollama",
            llm_model="llama3.3",
        )
    )

    assert response.case_id == "case-1"
    assert len(response.ingested) == 2
    assert workflow.run.call_count == 2
    workflow.run.assert_any_call(
        case_id="case-1",
        file_path="/tmp/a.txt",
        llm_provider="ollama",
        llm_model="llama3.3",
    )
    workflow.reassess_case.assert_called_once_with(
        "case-1",
        llm_provider="ollama",
        llm_model="llama3.3",
    )


def test_ingest_case_wraps_processing_errors(monkeypatch):
    file_a = Path("/tmp/a.txt")
    monkeypatch.setattr(main, "_resolve_ingest_txt_files", lambda _path: [file_a])
    workflow = Mock()
    workflow.run.side_effect = RuntimeError("parse failed")
    monkeypatch.setattr(main, "workflow", workflow)

    with pytest.raises(HTTPException) as exc:
        main.ingest_case(IngestCaseRequest(case_id="case-1", directory="/tmp"))

    assert exc.value.status_code == 502
    assert "a.txt" in exc.value.detail


def test_ingest_case_skip_reassess_uses_per_file_docs(monkeypatch):
    file_a = Path("/tmp/a.txt")
    file_b = Path("/tmp/b.txt")
    monkeypatch.setattr(main, "_resolve_ingest_txt_files", lambda _path: [file_a, file_b])

    workflow = Mock()
    workflow.run.side_effect = [
        {"deposition_doc": {"_id": "dep:1"}},
        {"deposition_doc": {"_id": "dep:2"}},
    ]
    couchdb = Mock()
    couchdb.get_doc.side_effect = [
        {
            "_id": "dep:1",
            "file_name": "a.txt",
            "witness_name": "Jane",
            "contradiction_score": 5,
            "flagged": False,
        },
        {
            "_id": "dep:2",
            "file_name": "b.txt",
            "witness_name": "Alan",
            "contradiction_score": 9,
            "flagged": True,
        },
    ]

    monkeypatch.setattr(main, "workflow", workflow)
    monkeypatch.setattr(main, "couchdb", couchdb)

    response = main.ingest_case(
        IngestCaseRequest(
            case_id="case-1",
            directory="/tmp",
            llm_provider="ollama",
            llm_model="llama3.3",
            skip_reassess=True,
        )
    )

    assert response.case_id == "case-1"
    assert [item.deposition_id for item in response.ingested] == ["dep:1", "dep:2"]
    workflow.reassess_case.assert_not_called()
    assert couchdb.get_doc.call_count == 2


def test_list_depositions_sorts_descending(monkeypatch):
    couchdb = Mock()
    couchdb.list_depositions.return_value = [
        {"_id": "dep:1", "contradiction_score": 10},
        {"_id": "dep:2", "contradiction_score": 80},
    ]
    monkeypatch.setattr(main, "couchdb", couchdb)

    result = main.list_depositions("case-1")

    assert [doc["_id"] for doc in result] == ["dep:2", "dep:1"]


def test_list_depositions_dashboard_handles_missing_and_none_scores(monkeypatch):
    couchdb = Mock()
    couchdb.list_depositions.return_value = [
        {"_id": "dep:1", "contradiction_score": None},
        {"_id": "dep:2", "contradiction_score": 25},
        {"_id": "dep:3"},
        {"_id": "dep:4", "contradiction_score": "high"},
    ]
    monkeypatch.setattr(main, "couchdb", couchdb)

    result = main.list_depositions("case-1")

    assert [doc["_id"] for doc in result] == ["dep:2", "dep:1", "dep:3", "dep:4"]


def test_get_deposition_success_and_not_found(monkeypatch):
    couchdb = Mock()
    couchdb.get_doc.return_value = {"_id": "dep:1"}
    monkeypatch.setattr(main, "couchdb", couchdb)

    assert main.get_deposition("dep:1") == {"_id": "dep:1"}

    couchdb.get_doc.side_effect = RuntimeError("missing")
    with pytest.raises(HTTPException) as exc:
        main.get_deposition("dep:2")
    assert exc.value.status_code == 404


def test_chat_success(monkeypatch):
    couchdb = Mock()
    couchdb.get_doc.return_value = {"_id": "dep:1", "case_id": "case-1"}
    couchdb.list_depositions.return_value = [{"_id": "dep:1"}, {"_id": "dep:2"}]
    chat_service = Mock()
    chat_service.respond.return_value = "Short answer: ok\nDetails:\n- a\n- b"

    monkeypatch.setattr(main, "couchdb", couchdb)
    monkeypatch.setattr(main, "chat_service", chat_service)

    response = main.chat(
        ChatRequest(
            case_id="case-1",
            deposition_id="dep:1",
            message="hello",
            llm_provider="ollama",
            llm_model="llama3.3",
        )
    )

    assert response.response.startswith("Short answer:")
    chat_service.respond.assert_called_once()
    assert chat_service.respond.call_args.kwargs["llm_provider"] == "ollama"
    assert chat_service.respond.call_args.kwargs["llm_model"] == "llama3.3"


def test_chat_wraps_service_errors(monkeypatch):
    couchdb = Mock()
    couchdb.get_doc.return_value = {"_id": "dep:1", "case_id": "case-1"}
    couchdb.list_depositions.return_value = [{"_id": "dep:1"}, {"_id": "dep:2"}]
    chat_service = Mock()
    chat_service.respond.side_effect = RuntimeError("model failed")

    monkeypatch.setattr(main, "couchdb", couchdb)
    monkeypatch.setattr(main, "chat_service", chat_service)

    with pytest.raises(HTTPException) as exc:
        main.chat(ChatRequest(case_id="case-1", deposition_id="dep:1", message="hello"))

    assert exc.value.status_code == 502
    assert "model failed" in exc.value.detail


def test_chat_errors(monkeypatch):
    couchdb = Mock()
    couchdb.get_doc.side_effect = RuntimeError("missing")
    monkeypatch.setattr(main, "couchdb", couchdb)

    with pytest.raises(HTTPException) as missing:
        main.chat(ChatRequest(case_id="case-1", deposition_id="dep:1", message="hello"))
    assert missing.value.status_code == 404

    couchdb.get_doc.side_effect = None
    couchdb.get_doc.return_value = {"_id": "dep:1", "case_id": "other-case"}
    with pytest.raises(HTTPException) as mismatch:
        main.chat(ChatRequest(case_id="case-1", deposition_id="dep:1", message="hello"))
    assert mismatch.value.status_code == 400


def test_reason_contradiction_success(monkeypatch):
    couchdb = Mock()
    couchdb.get_doc.return_value = {"_id": "dep:1", "case_id": "case-1"}
    couchdb.list_depositions.return_value = [{"_id": "dep:1"}, {"_id": "dep:2"}]
    chat_service = Mock()
    chat_service.reason_about_contradiction.return_value = "Short answer: ok\nDetails:\n- a\n- b\n- c"

    monkeypatch.setattr(main, "couchdb", couchdb)
    monkeypatch.setattr(main, "chat_service", chat_service)

    request = ContradictionReasonRequest(
        case_id="case-1",
        deposition_id="dep:1",
        llm_provider="ollama",
        llm_model="llama3.3",
        contradiction=ContradictionFinding(
            other_deposition_id="dep:2",
            other_witness_name="Peer",
            topic="Timeline",
            rationale="Mismatch",
            severity=60,
        ),
    )

    response = main.reason_contradiction(request)

    assert response.response.startswith("Short answer:")
    chat_service.reason_about_contradiction.assert_called_once()
    assert chat_service.reason_about_contradiction.call_args.kwargs["llm_provider"] == "ollama"
    assert chat_service.reason_about_contradiction.call_args.kwargs["llm_model"] == "llama3.3"


def test_reason_contradiction_wraps_service_errors(monkeypatch):
    couchdb = Mock()
    couchdb.get_doc.return_value = {"_id": "dep:1", "case_id": "case-1"}
    couchdb.list_depositions.return_value = [{"_id": "dep:1"}, {"_id": "dep:2"}]
    chat_service = Mock()
    chat_service.reason_about_contradiction.side_effect = RuntimeError("model failed")

    monkeypatch.setattr(main, "couchdb", couchdb)
    monkeypatch.setattr(main, "chat_service", chat_service)

    request = ContradictionReasonRequest(
        case_id="case-1",
        deposition_id="dep:1",
        contradiction=ContradictionFinding(
            other_deposition_id="dep:2",
            other_witness_name="Peer",
            topic="Timeline",
            rationale="Mismatch",
            severity=60,
        ),
    )

    with pytest.raises(HTTPException) as exc:
        main.reason_contradiction(request)

    assert exc.value.status_code == 502
    assert "model failed" in exc.value.detail


def test_reason_contradiction_errors(monkeypatch):
    couchdb = Mock()
    couchdb.get_doc.side_effect = RuntimeError("missing")
    monkeypatch.setattr(main, "couchdb", couchdb)

    request = ContradictionReasonRequest(
        case_id="case-1",
        deposition_id="dep:1",
        contradiction=ContradictionFinding(
            other_deposition_id="dep:2",
            other_witness_name="Peer",
            topic="Timeline",
            rationale="Mismatch",
            severity=60,
        ),
    )

    with pytest.raises(HTTPException) as missing:
        main.reason_contradiction(request)
    assert missing.value.status_code == 404

    couchdb.get_doc.side_effect = None
    couchdb.get_doc.return_value = {"_id": "dep:1", "case_id": "other-case"}
    with pytest.raises(HTTPException) as mismatch:
        main.reason_contradiction(request)
    assert mismatch.value.status_code == 400
