from __future__ import annotations

from copy import deepcopy
from unittest.mock import Mock

from fastapi.testclient import TestClient
import pytest

from backend.app import main


class _InMemoryDocStore:
    """Minimal in-memory document store for endpoint integration tests."""

    def __init__(self, docs: list[dict] | None = None):
        self._docs: dict[str, dict] = {}
        self._next_id = 1
        for doc in docs or []:
            self.save_doc(doc)

    def ensure_db(self) -> None:
        """Mirror CouchDB client setup API."""

    def close(self) -> None:
        """Mirror CouchDB client teardown API."""

    def get_doc(self, doc_id: str) -> dict:
        if doc_id not in self._docs:
            raise KeyError(doc_id)
        return deepcopy(self._docs[doc_id])

    def find(self, selector: dict, limit: int = 10000) -> list[dict]:
        results: list[dict] = []
        for doc in self._docs.values():
            if all(doc.get(key) == value for key, value in selector.items()):
                results.append(deepcopy(doc))
            if len(results) >= limit:
                break
        return results

    def save_doc(self, doc: dict) -> dict:
        stored = deepcopy(doc)
        doc_id = str(stored.get("_id") or f"doc:{self._next_id}")
        self._next_id += 1
        current = self._docs.get(doc_id, {})
        stored["_id"] = doc_id
        stored["_rev"] = str(int(current.get("_rev", "0") or 0) + 1)
        self._docs[doc_id] = stored
        return deepcopy(stored)

    def update_doc(self, doc: dict) -> dict:
        if not doc.get("_id"):
            raise KeyError("Document _id is required")
        return self.save_doc(doc)

    def list_depositions(self, case_id: str) -> list[dict]:
        return self.find({"type": "deposition", "case_id": case_id}, limit=10000)


@pytest.fixture
def api_client(monkeypatch):
    """Build a FastAPI test client with startup dependencies stubbed."""

    monkeypatch.setattr(main, "_ensure_startup_llm_connectivity", lambda: None)
    monkeypatch.setattr(main, "_ensure_request_llm_operational", lambda *_args, **_kwargs: None)

    couchdb = Mock()
    couchdb.ensure_db = Mock()
    couchdb.close = Mock()
    couchdb.list_depositions.return_value = []
    monkeypatch.setattr(main, "couchdb", couchdb)

    memory_couchdb = Mock()
    memory_couchdb.ensure_db = Mock()
    memory_couchdb.close = Mock()
    monkeypatch.setattr(main, "memory_couchdb", memory_couchdb)

    trace_couchdb = Mock()
    trace_couchdb.ensure_db = Mock()
    trace_couchdb.close = Mock()
    monkeypatch.setattr(main, "trace_couchdb", trace_couchdb)

    rag_couchdb = Mock()
    rag_couchdb.ensure_db = Mock()
    rag_couchdb.close = Mock()
    monkeypatch.setattr(main, "rag_couchdb", rag_couchdb)

    neo4j_graph = Mock()
    neo4j_graph.close = Mock()
    monkeypatch.setattr(main, "neo4j_graph", neo4j_graph)

    with TestClient(main.app) as client:
        yield client


def test_root_serves_frontend_shell(api_client):
    response = api_client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Data-Blitz Demo logo" in response.text


def test_admin_test_report_serves_html_artifact(api_client, monkeypatch, tmp_path):
    report_file = tmp_path / "tests.html"
    report_file.write_text(
        "<!doctype html><html><head><title>tests</title></head><body><h1>377 Passed</h1></body></html>",
        encoding="utf-8",
    )
    monkeypatch.setattr(main, "tests_report_file", report_file)

    response = api_client.get("/admin/test-report")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert 'id="admin-report-theme"' in response.text
    assert "377 Passed" in response.text


def test_admin_test_report_returns_fallback_when_artifact_missing(api_client, monkeypatch, tmp_path):
    monkeypatch.setattr(main, "tests_report_file", tmp_path / "missing-tests.html")

    response = api_client.get("/admin/test-report")

    assert response.status_code == 404
    assert "text/html" in response.headers["content-type"]
    assert "tests.html is not available" in response.text


def test_admin_test_log_extracts_explicit_logs(api_client, monkeypatch, tmp_path):
    report_file = tmp_path / "tests.html"
    report_file.write_text(
        (
            '<!doctype html><html><head><title>tests</title></head><body>'
            '<div id="data-container" data-jsonblob="{&#34;tests&#34;:{&#34;tests/test_sample.py::test_alpha&#34;:'
            '[{&#34;result&#34;:&#34;Failed&#34;,&#34;duration&#34;:&#34;3 ms&#34;,&#34;log&#34;:&#34;Assertion failed&#34;}],'
            '&#34;tests/test_sample.py::test_beta&#34;:[{&#34;result&#34;:&#34;Passed&#34;,&#34;log&#34;:&#34;No log output captured.&#34;}]}}"></div>'
            '</body></html>'
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(main, "tests_report_file", report_file)

    response = api_client.get("/api/admin/test-log")

    assert response.status_code == 200
    payload = response.json()
    assert "Collected explicit log output for 1 of 2" in payload["summary"]
    assert "[Failed] tests/test_sample.py::test_alpha (3 ms)" in payload["log_output"]
    assert "Assertion failed" in payload["log_output"]


def test_admin_user_round_trip(api_client, monkeypatch):
    couchdb = _InMemoryDocStore()
    monkeypatch.setattr(main, "couchdb", couchdb)

    create_response = api_client.post("/api/admin/users", json={"name": "Paul Harvener"})

    assert create_response.status_code == 200
    payload = create_response.json()
    assert payload["name"] == "Paul Harvener"
    assert payload["user_id"]
    assert payload["created_at"]

    list_response = api_client.get("/api/admin/users")

    assert list_response.status_code == 200
    assert list_response.json()["users"][0]["name"] == "Paul Harvener"


def test_deposition_upload_endpoint_round_trip(api_client, monkeypatch, tmp_path):
    target = tmp_path / "deps"
    target.mkdir()
    monkeypatch.setattr(main, "_resolve_upload_directory", lambda _directory: target)

    response = api_client.post(
        "/api/depositions/upload",
        data={"directory": "/data/depositions/default"},
        files=[("files", ("new_witness.txt", b"Date: April 1, 2025\nWitness: New Witness", "text/plain"))],
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["file_count"] == 1
    saved_path = target / "new_witness.txt"
    assert payload["saved_files"] == [str(saved_path)]
    assert saved_path.read_text(encoding="utf-8") == "Date: April 1, 2025\nWitness: New Witness"


def test_deposition_upload_endpoint_rejects_non_txt(api_client, monkeypatch, tmp_path):
    target = tmp_path / "deps"
    target.mkdir()
    monkeypatch.setattr(main, "_resolve_upload_directory", lambda _directory: target)

    response = api_client.post(
        "/api/depositions/upload",
        data={"directory": "/data/depositions/default"},
        files=[("files", ("bad.pdf", b"%PDF-1.4", "application/pdf"))],
    )

    assert response.status_code == 400
    assert ".txt file" in response.json()["detail"]


def test_list_cases_endpoint_returns_serialized_payload(api_client, monkeypatch):
    monkeypatch.setattr(
        main,
        "_list_case_summaries",
        lambda: [main.CaseSummary(case_id="CASE-INT-1", deposition_count=3, memory_entries=2)],
    )

    response = api_client.get("/api/cases")

    assert response.status_code == 200
    assert response.json() == {
        "cases": [
            {
                "case_id": "CASE-INT-1",
                "deposition_count": 3,
                "memory_entries": 2,
                "updated_at": None,
                "last_action": None,
                "last_directory": None,
                "last_llm_provider": None,
                "last_llm_model": None,
            }
        ]
    }


def test_summarize_focused_reasoning_endpoint_round_trip(api_client, monkeypatch):
    main.couchdb.get_doc.return_value = {"_id": "dep:1", "case_id": "case-1"}
    monkeypatch.setattr(
        main.chat_service,
        "summarize_focused_reasoning",
        lambda **_kwargs: "Short answer: Condensed integration summary.",
    )
    monkeypatch.setattr(main, "_save_case_memory", Mock())
    monkeypatch.setattr(main, "_upsert_case_doc", Mock())
    monkeypatch.setattr(main, "_safe_case_depositions", lambda _case_id: [{"_id": "dep:1"}])

    response = api_client.post(
        "/api/summarize-focused-reasoning",
        json={
            "case_id": "case-1",
            "deposition_id": "dep:1",
            "reasoning_text": "Short answer: Full focused reasoning text.",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"summary": "Short answer: Condensed integration summary."}


def test_summarize_focused_reasoning_endpoint_validates_payload(api_client):
    response = api_client.post(
        "/api/summarize-focused-reasoning",
        json={
            "case_id": "case-1",
            "deposition_id": "dep:1",
            "reasoning_text": "",
        },
    )

    assert response.status_code == 422


def test_deposition_sentiment_endpoint_round_trip(api_client, monkeypatch):
    main.couchdb.get_doc.return_value = {
        "_id": "dep:1",
        "case_id": "case-1",
        "raw_text": "The witness remained calm and credible but described one conflict and one risk.",
    }
    monkeypatch.setattr(main, "_save_case_memory", Mock())
    monkeypatch.setattr(main, "_upsert_case_doc", Mock())
    monkeypatch.setattr(main, "_safe_case_depositions", lambda _case_id: [{"_id": "dep:1"}])

    response = api_client.post(
        "/api/deposition-sentiment",
        json={
            "case_id": "case-1",
            "deposition_id": "dep:1",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["label"] in {"positive", "neutral", "negative"}
    assert payload["word_count"] > 0
    assert "Overall deposition sentiment is" in payload["summary"]


def test_deposition_sentiment_endpoint_handles_case_mismatch(api_client):
    main.couchdb.get_doc.return_value = {"_id": "dep:1", "case_id": "other-case", "raw_text": "text"}

    response = api_client.post(
        "/api/deposition-sentiment",
        json={
            "case_id": "case-1",
            "deposition_id": "dep:1",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Deposition does not belong to requested case"


def test_save_case_then_get_case_detail_round_trip(api_client, monkeypatch):
    couchdb = _InMemoryDocStore(
        [
            {
                "_id": "dep:case-roundtrip:1",
                "type": "deposition",
                "case_id": "CASE-ROUNDTRIP",
                "file_name": "witness_1.txt",
            }
        ]
    )
    monkeypatch.setattr(main, "couchdb", couchdb)

    save_response = api_client.post(
        "/api/cases",
        json={
            "case_id": "CASE-ROUNDTRIP",
            "directory": "/tmp/case-roundtrip",
            "llm_provider": "openai",
            "llm_model": "gpt-4o-mini",
            "snapshot": {
                "active_tab": "deposition",
                "chat": {"draft_input": "Preserve this note."},
            },
        },
    )

    assert save_response.status_code == 200
    assert save_response.json()["case_id"] == "CASE-ROUNDTRIP"
    assert save_response.json()["deposition_count"] == 1

    detail_response = api_client.get("/api/cases/CASE-ROUNDTRIP")

    assert detail_response.status_code == 200
    assert detail_response.json() == {
        "case_id": "CASE-ROUNDTRIP",
        "deposition_count": 1,
        "memory_entries": 0,
        "updated_at": detail_response.json()["updated_at"],
        "last_action": "save",
        "last_directory": "/tmp/case-roundtrip",
        "last_llm_provider": "openai",
        "last_llm_model": "gpt-4o-mini",
        "snapshot": {
            "active_tab": "deposition",
            "chat": {"draft_input": "Preserve this note."},
        },
    }
    assert detail_response.json()["updated_at"]


def test_save_case_version_then_list_versions_clones_source_docs(api_client, monkeypatch):
    couchdb = _InMemoryDocStore(
        [
            {
                "_id": "dep:source-case:1",
                "type": "deposition",
                "case_id": "SOURCE-CASE",
                "file_name": "source_witness.txt",
            }
        ]
    )
    memory_couchdb = _InMemoryDocStore(
        [
            {
                "_id": "mem:source-case:1",
                "type": "case_memory",
                "case_id": "SOURCE-CASE",
                "channel": "chat",
                "created_at": "2026-02-28T00:00:00+00:00",
                "payload": {"response": "Saved memory."},
            }
        ]
    )
    monkeypatch.setattr(main, "couchdb", couchdb)
    monkeypatch.setattr(main, "memory_couchdb", memory_couchdb)

    save_response = api_client.post(
        "/api/cases/version",
        json={
            "case_id": "CLONED-CASE",
            "source_case_id": "SOURCE-CASE",
            "directory": "/tmp/cloned-case",
            "llm_provider": "openai",
            "llm_model": "gpt-4o-mini",
            "snapshot": {"active_tab": "case"},
        },
    )

    assert save_response.status_code == 200
    assert save_response.json()["case_id"] == "CLONED-CASE"
    assert save_response.json()["version"] == 1

    versions_response = api_client.get("/api/cases/CLONED-CASE/versions")

    assert versions_response.status_code == 200
    assert versions_response.json() == {
        "case_id": "CLONED-CASE",
        "versions": [
            {
                "case_id": "CLONED-CASE",
                "version": 1,
                "created_at": versions_response.json()["versions"][0]["created_at"],
                "directory": "/tmp/cloned-case",
                "llm_provider": "openai",
                "llm_model": "gpt-4o-mini",
                "snapshot": {"active_tab": "case"},
            }
        ],
    }
    assert couchdb.list_depositions("CLONED-CASE")
    assert memory_couchdb.find({"type": "case_memory", "case_id": "CLONED-CASE"}, limit=10)


def test_chat_then_reason_updates_case_memory_and_case_detail(api_client, monkeypatch):
    couchdb = _InMemoryDocStore(
        [
            {
                "_id": "dep:chat-case:1",
                "type": "deposition",
                "case_id": "CHAT-CASE",
                "file_name": "target.txt",
                "witness_name": "Target Witness",
            },
            {
                "_id": "dep:chat-case:2",
                "type": "deposition",
                "case_id": "CHAT-CASE",
                "file_name": "peer.txt",
                "witness_name": "Peer Witness",
            },
        ]
    )
    memory_couchdb = _InMemoryDocStore()
    monkeypatch.setattr(main, "couchdb", couchdb)
    monkeypatch.setattr(main, "memory_couchdb", memory_couchdb)
    monkeypatch.setattr(
        main,
        "_resolve_request_llm",
        lambda llm_provider, llm_model: (llm_provider or "openai", llm_model or "gpt-4o-mini"),
    )
    monkeypatch.setattr(
        main.chat_service,
        "respond_with_trace",
        lambda *_args, **_kwargs: ("Short answer: Attorney response.", []),
    )
    monkeypatch.setattr(
        main.chat_service,
        "reason_about_contradiction",
        lambda **_kwargs: "Short answer: Focused contradiction review.",
    )

    chat_response = api_client.post(
        "/api/chat",
        json={
            "case_id": "CHAT-CASE",
            "deposition_id": "dep:chat-case:1",
            "message": "What is the key issue?",
            "history": [{"role": "user", "content": "Start with breach."}],
            "llm_provider": "openai",
            "llm_model": "gpt-4o-mini",
        },
    )

    assert chat_response.status_code == 200
    assert chat_response.json()["response"] == "Short answer: Attorney response."

    reason_response = api_client.post(
        "/api/reason-contradiction",
        json={
            "case_id": "CHAT-CASE",
            "deposition_id": "dep:chat-case:1",
            "contradiction": {
                "other_deposition_id": "dep:chat-case:2",
                "other_witness_name": "Peer Witness",
                "topic": "Material breach timing",
                "rationale": "The two witnesses give different timelines.",
                "severity": 42,
            },
            "llm_provider": "openai",
            "llm_model": "gpt-4o-mini",
        },
    )

    assert reason_response.status_code == 200
    assert reason_response.json()["response"] == "Short answer: Focused contradiction review."

    detail_response = api_client.get("/api/cases/CHAT-CASE")

    assert detail_response.status_code == 200
    assert detail_response.json()["memory_entries"] == 2
    assert detail_response.json()["last_action"] == "reason"
    assert detail_response.json()["last_llm_provider"] == "openai"
    assert detail_response.json()["last_llm_model"] == "gpt-4o-mini"

    memory_docs = memory_couchdb.find({"type": "case_memory", "case_id": "CHAT-CASE"}, limit=10)
    assert len(memory_docs) == 2
    assert {doc["channel"] for doc in memory_docs} == {"chat", "reason"}
