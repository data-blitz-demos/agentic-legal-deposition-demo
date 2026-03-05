# Copyright (c) 2026 Data-Blitz Inc. All rights reserved.
# License: Proprietary. See NOTICE.md.
# Author: Paul Harvener.

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
from types import SimpleNamespace
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

    def ensure_deposition_views(self) -> None:
        """Mirror CouchDB design-doc setup API."""

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

    def delete_doc(self, doc_id: str, rev: str | None = None) -> None:
        if doc_id not in self._docs:
            raise KeyError(doc_id)
        if rev is not None and str(self._docs[doc_id].get("_rev") or "") != str(rev):
            raise KeyError(f"Revision mismatch for {doc_id}")
        del self._docs[doc_id]

    def list_depositions(self, case_id: str) -> list[dict]:
        return self.find({"type": "deposition", "case_id": case_id}, limit=10000)

    def list_deposition_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for doc in self.find({"type": "deposition"}, limit=10000):
            case_id = str(doc.get("case_id") or "").strip()
            if not case_id:
                continue
            counts[case_id] = counts.get(case_id, 0) + 1
        return counts


@pytest.fixture
def api_client(monkeypatch):
    """Build a FastAPI test client with startup dependencies stubbed."""

    monkeypatch.setattr(main, "_ensure_startup_llm_connectivity", lambda: None)
    monkeypatch.setattr(main, "_ensure_request_llm_operational", lambda *_args, **_kwargs: None)

    couchdb = Mock()
    couchdb.ensure_db = Mock()
    couchdb.ensure_deposition_views = Mock()
    couchdb.close = Mock()
    couchdb.find.return_value = []
    couchdb.list_depositions.return_value = []
    couchdb.list_deposition_counts.return_value = {}
    monkeypatch.setattr(main, "couchdb", couchdb)

    memory_couchdb = Mock()
    memory_couchdb.ensure_db = Mock()
    memory_couchdb.close = Mock()
    monkeypatch.setattr(main, "memory_couchdb", memory_couchdb)

    trace_couchdb = Mock()
    trace_couchdb.ensure_db = Mock()
    trace_couchdb.close = Mock()
    trace_couchdb.find.return_value = []
    monkeypatch.setattr(main, "trace_couchdb", trace_couchdb)

    rag_couchdb = Mock()
    rag_couchdb.ensure_db = Mock()
    rag_couchdb.close = Mock()
    rag_couchdb.find.return_value = []
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
    assert 'id="timelineScale"' in response.text
    assert 'id="adminPersonaSystemChoosePromptBtn"' in response.text
    assert 'id="adminPersonaAssistantChoosePromptBtn"' in response.text
    assert 'id="adminPersonaContextChoosePromptBtn"' in response.text
    assert 'id="adminPersonaSystemObservableBtn"' in response.text
    assert 'id="adminPersonaAssistantObservableBtn"' in response.text
    assert 'id="adminPersonaContextObservableBtn"' in response.text
    assert re.search(r'id="adminPersonaSystemObservableBtn"[^>]*\bdisabled\b', response.text)
    assert re.search(r'id="adminPersonaAssistantObservableBtn"[^>]*\bdisabled\b', response.text)
    assert re.search(r'id="adminPersonaContextObservableBtn"[^>]*\bdisabled\b', response.text)
    assert 'id="adminPersonaSystemPromptTemplateSelect"' in response.text
    assert 'id="adminPersonaAssistantPromptTemplateSelect"' in response.text
    assert 'id="adminPersonaContextPromptTemplateSelect"' in response.text
    assert re.search(r'id="adminPersonaSystemPromptTemplateSelect"[^>]*\bhidden\b', response.text)
    assert re.search(r'id="adminPersonaAssistantPromptTemplateSelect"[^>]*\bhidden\b', response.text)
    assert re.search(r'id="adminPersonaContextPromptTemplateSelect"[^>]*\bhidden\b', response.text)
    assert 'id="adminPersonaSystemSavePromptBtn"' in response.text
    assert 'id="adminPersonaAssistantSavePromptBtn"' in response.text
    assert 'id="adminPersonaContextSavePromptBtn"' in response.text
    assert 'id="adminPersonaSmokeTestBtn"' in response.text
    assert 'id="adminPersonaFormPromptSentimentBtn"' in response.text
    assert 'id="adminPersonaFormPromptSentiment"' in response.text
    assert 'id="adminPersonaOpenPromptModalBtn"' in response.text
    assert 'id="adminPersonaToggleRagBtn"' in response.text
    assert 'id="adminPersonaTogglePromptObservablesBtn"' in response.text
    assert 'id="adminPersonaPromptPanel"' in response.text
    assert 'id="adminPersonaRagPanel"' in response.text
    assert 'id="adminPersonaPromptObservablesPanel"' in response.text
    assert re.search(r'id="adminPersonaPromptPanel"[^>]*\bhidden\b', response.text)
    assert re.search(r'id="adminPersonaRagPanel"[^>]*\bhidden\b', response.text)
    assert re.search(r'id="adminPersonaPromptObservablesPanel"[^>]*\bhidden\b', response.text)
    assert 'id="adminPersonaSystemPrompt"' in response.text
    assert 'id="adminPersonaAssistantPrompt"' in response.text
    assert 'id="adminPersonaContextPrompt"' in response.text
    assert 'id="adminPersonaPromptModal"' in response.text
    assert 'id="adminPersonaPromptModalSystem"' in response.text
    assert 'id="adminPersonaPromptModalAssistant"' in response.text
    assert 'id="adminPersonaPromptModalContext"' in response.text
    assert 'id="adminPersonaPromptObservableModal"' in response.text
    assert 'id="adminPersonaPromptObservableModalTitle"' in response.text
    assert 'id="adminPersonaPromptObservableModalMeta"' in response.text
    assert 'id="adminPersonaPromptObservableModalBody"' in response.text
    assert 'id="adminPersonaPromptObservableModalCloseBtn"' in response.text
    assert re.search(r'id="adminPersonaPromptObservableModal"[^>]*\bhidden\b', response.text)
    assert 'id="adminPersonaPromptSentimentBtn"' in response.text
    assert 'id="adminPersonaPromptResaveBtn"' in response.text
    assert 'id="adminPersonaToggleToolsBtn"' in response.text
    assert 'id="adminPersonaToolsPanel"' in response.text
    assert 'id="adminPersonaToolSelect"' in response.text
    assert 'id="adminPersonaLoadToolsBtn"' in response.text
    assert 'id="adminPersonaToolAddBtn"' in response.text
    assert 'id="adminPersonaToolList"' in response.text
    assert 'id="adminPersonaRefreshPromptObservablesBtn"' in response.text
    assert 'id="adminPersonaPromptObservablesList"' in response.text
    assert 'id="adminPersonaPromptObservablesDetail"' in response.text
    assert 'id="adminPersonaGraphProgress"' in response.text
    assert 'id="adminPersonaGraphClock"' in response.text
    rag_panel_match = re.search(
        r'<details id="adminPersonaRagPanel"[\s\S]*?</details>',
        response.text,
        flags=re.IGNORECASE,
    )
    assert rag_panel_match is not None
    rag_panel_html = rag_panel_match.group(0)
    assert 'Ask Graph Questionnaire' in rag_panel_html
    assert 'id="adminPersonaGraphQuestion"' in rag_panel_html
    assert 'id="adminPersonaGraphAskBtn"' in rag_panel_html
    assert 'id="adminPersonaGraphClearBtn"' in rag_panel_html
    assert 'id="adminPersonaGraphAnswer"' in rag_panel_html
    assert re.search(r'id="adminPersonaRagSelect"[^>]*\bdisabled\b', rag_panel_html)
    assert re.search(r'id="adminPersonaRagAddBtn"[^>]*\bdisabled\b', rag_panel_html)
    assert re.search(r'id="adminPersonaLoadRagsBtn"[^>]*\bdisabled\b', rag_panel_html)
    assert re.search(r'id="adminPersonaGraphQuestion"[^>]*\bdisabled\b', rag_panel_html)
    assert re.search(r'id="adminPersonaGraphAskBtn"[^>]*\bdisabled\b', rag_panel_html)
    assert re.search(r'id="adminPersonaGraphClearBtn"[^>]*\bdisabled\b', rag_panel_html)
    tools_panel_match = re.search(
        r'<details id="adminPersonaToolsPanel"[\s\S]*?</details>',
        response.text,
        flags=re.IGNORECASE,
    )
    assert tools_panel_match is not None
    tools_panel_html = tools_panel_match.group(0)
    assert 'Persona MCP Tools' in tools_panel_html
    assert 'id="adminPersonaToolSelect"' in tools_panel_html
    assert 'id="adminPersonaLoadToolsBtn"' in tools_panel_html
    assert 'id="adminPersonaToolAddBtn"' in tools_panel_html
    assert 'id="adminPersonaToolList"' in tools_panel_html
    assert re.search(r'id="adminPersonaToolSelect"[^>]*\bdisabled\b', tools_panel_html)
    assert re.search(r'id="adminPersonaLoadToolsBtn"[^>]*\bdisabled\b', tools_panel_html)
    assert re.search(r'id="adminPersonaToolAddBtn"[^>]*\bdisabled\b', tools_panel_html)
    prompt_observables_panel_match = re.search(
        r'<details id="adminPersonaPromptObservablesPanel"[\s\S]*?</details>',
        response.text,
        flags=re.IGNORECASE,
    )
    assert prompt_observables_panel_match is not None
    prompt_observables_panel_html = prompt_observables_panel_match.group(0)
    assert 'All Prompts Observables' in prompt_observables_panel_html
    assert 'id="adminPersonaRefreshPromptObservablesBtn"' in prompt_observables_panel_html
    assert 'id="adminPersonaPromptObservablesList"' in prompt_observables_panel_html
    assert 'id="adminPersonaPromptObservablesDetail"' in prompt_observables_panel_html
    assert re.search(r'id="adminPersonaRefreshPromptObservablesBtn"[^>]*\bdisabled\b', prompt_observables_panel_html)
    assert 'id="graphRagEmbeddingEnabled"' in response.text
    assert 'id="saveGraphRagEmbeddingBtn"' in response.text


def test_prometheus_metrics_endpoint_exposes_application_metrics(api_client):
    root_response = api_client.get("/")
    assert root_response.status_code == 200

    response = api_client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    body = response.text
    assert "deposition_http_requests_total" in body
    assert 'path="/"' in body
    assert "deposition_http_request_duration_seconds" in body
    assert "deposition_app_log_events_total" in body
    assert "deposition_admin_test_runs_total" in body


def test_graph_rag_embedding_config_endpoints_round_trip(api_client, monkeypatch):
    store = _InMemoryDocStore()
    monkeypatch.setattr(main, "couchdb", store)

    get_response = api_client.get("/api/graph-rag/embedding-config")
    assert get_response.status_code == 200
    assert get_response.json()["enabled"] is False

    post_response = api_client.post(
        "/api/graph-rag/embedding-config",
        json={
            "enabled": True,
            "provider": "openai",
            "model": "text-embedding-3-small",
            "dimensions": 768,
            "index_name": "resource_embeddings",
            "node_label": "Resource",
            "property_name": "embedding",
        },
    )
    assert post_response.status_code == 200
    payload = post_response.json()
    assert payload["enabled"] is True
    assert payload["configured"] is True
    assert payload["source"] == "saved"

    saved_doc = store.get_doc("graph_rag_embedding_config")
    assert saved_doc["type"] == "graph_rag_embedding_config"
    assert saved_doc["dimensions"] == 768

    refreshed = api_client.get("/api/graph-rag/embedding-config")
    assert refreshed.status_code == 200
    assert refreshed.json()["enabled"] is True


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
    assert 'id="admin-report-helper"' in response.text
    assert "toggleDetailRowFromCell" in response.text
    assert "collapseDetailRowFromExpander" in response.text
    assert "377 Passed" in response.text


def test_admin_test_report_returns_fallback_when_artifact_missing(api_client, monkeypatch, tmp_path):
    monkeypatch.setattr(main, "tests_report_file", tmp_path / "missing-tests.html")

    response = api_client.get("/admin/test-report")

    assert response.status_code == 404
    assert "text/html" in response.headers["content-type"]
    assert "tests.html is not available" in response.text


def test_admin_test_report_does_not_duplicate_existing_theme_or_helper(api_client, monkeypatch, tmp_path):
    report_file = tmp_path / "tests.html"
    report_file.write_text(
        (
            "<!doctype html><html><head><style id=\"admin-report-theme\"></style></head>"
            "<body><h1>377 Passed</h1><script id=\"admin-report-helper\"></script></body></html>"
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(main, "tests_report_file", report_file)

    response = api_client.get("/admin/test-report")

    assert response.status_code == 200
    assert response.text.count('id="admin-report-theme"') == 1
    assert response.text.count('id="admin-report-helper"') == 1


def test_admin_test_log_extracts_explicit_logs(api_client, monkeypatch, tmp_path):
    report_file = tmp_path / "tests.html"
    output_file = tmp_path / "last-test-run-output.txt"
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
    output_file.write_text("404 passed", encoding="utf-8")
    monkeypatch.setattr(main, "tests_report_file", report_file)
    monkeypatch.setattr(main, "last_test_run_output_file", output_file)

    response = api_client.get("/api/admin/test-log")

    assert response.status_code == 200
    payload = response.json()
    assert "Collected explicit log output for 1 of 2" in payload["summary"]
    assert "Included the latest pytest console output." in payload["summary"]
    assert "=== Latest pytest console output ===" in payload["log_output"]
    assert "404 passed" in payload["log_output"]
    assert "[Failed] tests/test_sample.py::test_alpha (3 ms)" in payload["log_output"]
    assert "Assertion failed" in payload["log_output"]


def test_admin_test_log_falls_back_to_latest_console_output(api_client, monkeypatch, tmp_path):
    output_file = tmp_path / "last-test-run-output.txt"
    output_file.write_text("403 passed\n1 skipped", encoding="utf-8")
    monkeypatch.setattr(main, "tests_report_file", tmp_path / "missing-tests.html")
    monkeypatch.setattr(main, "last_test_run_output_file", output_file)

    response = api_client.get("/api/admin/test-log")

    assert response.status_code == 200
    payload = response.json()
    assert "Showing captured console output from the most recent pytest run." in payload["summary"]
    assert "403 passed" in payload["log_output"]
    assert "1 skipped" in payload["log_output"]


def test_admin_run_tests_endpoint_returns_command_result(api_client, monkeypatch, tmp_path):
    output_file = tmp_path / "last-test-run-output.txt"

    monotonic_values = iter([10.0, 100.0, 102.4, 102.5])
    expected_cmd = [
        main.sys.executable,
        "-m",
        "pytest",
        "-vv",
        "-rA",
        "--tb=short",
        "--capture=tee-sys",
        "-o",
        "log_cli=true",
        "-o",
        "log_cli_level=INFO",
    ]

    def _run(cmd, cwd, capture_output, text, timeout, check):
        assert cmd == expected_cmd
        assert cwd == str(main.app_root)
        assert capture_output is True
        assert text is True
        assert timeout == 900
        assert check is False
        return type(
            "_Completed",
            (),
            {
                "returncode": 0,
                "stdout": "404 passed",
                "stderr": "",
            },
        )()

    monkeypatch.setattr(main.subprocess, "run", _run)
    monkeypatch.setattr(main, "last_test_run_output_file", output_file)
    monkeypatch.setattr(main, "monotonic", lambda: next(monotonic_values))

    response = api_client.post("/api/admin/run-tests")

    assert response.status_code == 200
    payload = response.json()
    assert payload["succeeded"] is True
    assert payload["exit_code"] == 0
    assert payload["summary"] == "All tests passed."
    assert "404 passed" in payload["output"]
    assert payload["duration_seconds"] == 2.4
    assert output_file.read_text(encoding="utf-8") == "404 passed"


def test_admin_run_tests_endpoint_preserves_subsecond_duration(api_client, monkeypatch, tmp_path):
    output_file = tmp_path / "last-test-run-output.txt"
    monotonic_values = iter([1.0, 10.0, 10.34, 10.35])
    expected_cmd = [
        main.sys.executable,
        "-m",
        "pytest",
        "-vv",
        "-rA",
        "--tb=short",
        "--capture=tee-sys",
        "-o",
        "log_cli=true",
        "-o",
        "log_cli_level=INFO",
    ]

    def _run(cmd, cwd, capture_output, text, timeout, check):
        assert cmd == expected_cmd
        assert cwd == str(main.app_root)
        assert capture_output is True
        assert text is True
        assert timeout == 900
        assert check is False
        return type(
            "_Completed",
            (),
            {
                "returncode": 0,
                "stdout": "ok",
                "stderr": "",
            },
        )()

    monkeypatch.setattr(main.subprocess, "run", _run)
    monkeypatch.setattr(main, "last_test_run_output_file", output_file)
    monkeypatch.setattr(main, "monotonic", lambda: next(monotonic_values))

    response = api_client.post("/api/admin/run-tests")

    assert response.status_code == 200
    payload = response.json()
    assert payload["duration_seconds"] == 0.34
    assert payload["duration_seconds"] > 0


def test_admin_user_round_trip(api_client, monkeypatch):
    couchdb = _InMemoryDocStore()
    monkeypatch.setattr(main, "couchdb", couchdb)

    create_response = api_client.post(
        "/api/admin/users",
        json={
            "first_name": "Paul",
            "last_name": "Harvener",
            "authorization_level": "open",
        },
    )

    assert create_response.status_code == 200
    payload = create_response.json()
    assert payload["name"] == "Paul Harvener"
    assert payload["first_name"] == "Paul"
    assert payload["last_name"] == "Harvener"
    assert payload["authorization_level"] == "open"
    assert payload["user_id"]
    assert payload["created_at"]
    stored_doc = couchdb.get_doc(payload["user_id"])
    assert stored_doc["type"] == "user"
    assert stored_doc["name"] == "Paul Harvener"
    assert stored_doc["first_name"] == "Paul"
    assert stored_doc["last_name"] == "Harvener"
    assert stored_doc["authorization_level"] == "open"

    list_response = api_client.get("/api/admin/users")

    assert list_response.status_code == 200
    assert list_response.json()["users"][0]["name"] == "Paul Harvener"
    assert list_response.json()["users"][0]["first_name"] == "Paul"
    assert list_response.json()["users"][0]["last_name"] == "Harvener"
    assert list_response.json()["users"][0]["authorization_level"] == "open"

    update_response = api_client.post(
        "/api/admin/users",
        json={
            "user_id": payload["user_id"],
            "first_name": "Paula",
            "last_name": "Harvener",
            "authorization_level": "admin",
        },
    )

    assert update_response.status_code == 200
    updated_payload = update_response.json()
    assert updated_payload["user_id"] == payload["user_id"]
    assert updated_payload["name"] == "Paula Harvener"
    assert updated_payload["authorization_level"] == "admin"
    stored_updated_doc = couchdb.get_doc(payload["user_id"])
    assert stored_updated_doc["name"] == "Paula Harvener"
    assert stored_updated_doc["first_name"] == "Paula"
    assert stored_updated_doc["authorization_level"] == "admin"

    delete_response = api_client.delete(f"/api/admin/users/{payload['user_id']}")

    assert delete_response.status_code == 200
    delete_payload = delete_response.json()
    assert delete_payload["user_id"] == payload["user_id"]
    assert delete_payload["deleted"] is True
    with pytest.raises(KeyError):
        couchdb.get_doc(payload["user_id"])

    post_delete_list = api_client.get("/api/admin/users")

    assert post_delete_list.status_code == 200
    assert post_delete_list.json()["users"] == []


def test_admin_persona_round_trip(api_client, monkeypatch):
    couchdb = _InMemoryDocStore()
    monkeypatch.setattr(main, "couchdb", couchdb)

    create_response = api_client.post(
        "/api/admin/personas",
        json={
            "name": "Cross Examiner",
            "llm_provider": "openai",
            "llm_model": "gpt-5.2",
            "prompt_template_key": "chat_system",
            "prompts": "",
            "prompt_sections": {
                "system": "Challenge weak assumptions.",
                "assistant": "Use tight legal phrasing.",
                "context": "Only use evidence in the selected case.",
            },
            "rag_sequence": [{"key": "graph_rag_neo4j", "enabled": False}],
            "tool_sequence": [{"key": "mcp_couchdb_deposition_access", "enabled": True}],
        },
    )

    assert create_response.status_code == 200
    payload = create_response.json()
    assert payload["name"] == "Cross Examiner"
    assert payload["llm_provider"] == "openai"
    assert payload["llm_model"] == "gpt-5.2"
    assert payload["prompt_template_key"] == "chat_system"
    assert payload["prompt_sections"] == {
        "system": "Challenge weak assumptions.",
        "assistant": "Use tight legal phrasing.",
        "context": "Only use evidence in the selected case.",
    }
    assert "System:\nChallenge weak assumptions." in payload["prompts"]
    assert "Assistant:\nUse tight legal phrasing." in payload["prompts"]
    assert "Context:\nOnly use evidence in the selected case." in payload["prompts"]
    assert payload["rag_sequence"] == [{"key": "graph_rag_neo4j", "enabled": False}]
    assert payload["tool_sequence"] == [{"key": "mcp_couchdb_deposition_access", "enabled": True}]
    assert payload["persona_id"]
    stored_doc = couchdb.get_doc(payload["persona_id"])
    assert stored_doc["type"] == "persona"
    assert stored_doc["prompt_template_key"] == "chat_system"
    assert stored_doc["prompt_sections"] == {
        "system": "Challenge weak assumptions.",
        "assistant": "Use tight legal phrasing.",
        "context": "Only use evidence in the selected case.",
    }
    assert "System:\nChallenge weak assumptions." in stored_doc["prompts"]
    assert stored_doc["rag_sequence"] == [{"key": "graph_rag_neo4j", "enabled": False}]
    assert stored_doc["tool_sequence"] == [{"key": "mcp_couchdb_deposition_access", "enabled": True}]

    list_response = api_client.get("/api/admin/personas")

    assert list_response.status_code == 200
    assert list_response.json()["personas"][0]["name"] == "Cross Examiner"

    update_response = api_client.post(
        "/api/admin/personas",
        json={
            "persona_id": payload["persona_id"],
            "name": "Lead Attorney",
            "llm_provider": "ollama",
            "llm_model": "law_model",
            "prompt_template_key": "graph_rag_system",
            "prompt_sections": {
                "system": "Stay direct.",
                "assistant": "Keep answers concise.",
                "context": "",
            },
            "prompts": "",
            "rag_sequence": [{"key": "graph_rag_neo4j", "enabled": True}, "unknown_rag"],
            "tool_sequence": [{"key": "mcp_couchdb_thought_stream_access", "enabled": False}, "unknown_tool"],
        },
    )

    assert update_response.status_code == 200
    updated_payload = update_response.json()
    assert updated_payload["persona_id"] == payload["persona_id"]
    assert updated_payload["name"] == "Lead Attorney"
    assert updated_payload["llm_provider"] == "ollama"
    assert updated_payload["llm_model"] == "law_model"
    assert updated_payload["prompt_template_key"] == "graph_rag_system"
    assert updated_payload["prompt_sections"] == {
        "system": "Stay direct.",
        "assistant": "Keep answers concise.",
        "context": "",
    }
    assert updated_payload["rag_sequence"] == [{"key": "graph_rag_neo4j", "enabled": True}]
    assert updated_payload["tool_sequence"] == [{"key": "mcp_couchdb_thought_stream_access", "enabled": False}]
    stored_updated_doc = couchdb.get_doc(payload["persona_id"])
    assert stored_updated_doc["name"] == "Lead Attorney"
    assert stored_updated_doc["prompt_template_key"] == "graph_rag_system"
    assert stored_updated_doc["prompt_sections"] == {
        "system": "Stay direct.",
        "assistant": "Keep answers concise.",
        "context": "",
    }
    assert "System:\nStay direct." in stored_updated_doc["prompts"]
    assert stored_updated_doc["rag_sequence"] == [{"key": "graph_rag_neo4j", "enabled": True}]
    assert stored_updated_doc["tool_sequence"] == [{"key": "mcp_couchdb_thought_stream_access", "enabled": False}]


def test_admin_persona_prompt_templates_endpoint(api_client):
    response = api_client.get("/api/admin/personas/prompts")

    assert response.status_code == 200
    payload = response.json()
    assert payload["prompts"]
    assert payload["prompts"][0]["key"] == "map_deposition_system"
    assert payload["prompts"][0]["file_name"] == "map_deposition_system.txt"
    assert payload["prompts"][0]["content"]


def test_admin_persona_rag_options_endpoint(api_client):
    response = api_client.get("/api/admin/personas/rags")

    assert response.status_code == 200
    assert response.json() == {
        "rags": [
            {
                "key": "graph_rag_neo4j",
                "label": "Graph RAG (Neo4j)",
                "description": (
                    "Runs the current Neo4j-backed Graph RAG retrieval flow. "
                    "If multiple RAGs are configured later, they execute in list order."
                ),
                "available": True,
            }
        ]
    }


def test_admin_persona_tool_options_endpoint(api_client):
    response = api_client.get("/api/admin/personas/tools")

    assert response.status_code == 200
    assert response.json() == {
        "tools": [
            {
                "key": "mcp_couchdb_deposition_access",
                "label": "MCP: CouchDB Deposition Access",
                "description": (
                    "Expose the deposition CouchDB MCP server tools to the persona pipeline "
                    "for case/deposition retrieval."
                ),
                "available": True,
            },
            {
                "key": "mcp_couchdb_thought_stream_access",
                "label": "MCP: Thought Stream Access",
                "description": (
                    "Expose the thought-stream MCP server tools to the persona pipeline "
                    "for trace read/write operations."
                ),
                "available": True,
            },
        ]
    }


def test_admin_persona_graph_session_round_trip(api_client, monkeypatch):
    couchdb = _InMemoryDocStore()
    monkeypatch.setattr(main, "couchdb", couchdb)

    create_response = api_client.post(
        "/api/admin/personas",
        json={
            "name": "Graph Tester",
            "llm_provider": "openai",
            "llm_model": "gpt-5.2",
            "prompt_template_key": "graph_rag_system",
            "prompt_sections": {
                "system": "Use the graph.",
                "assistant": "",
                "context": "Use Neo4j context first.",
            },
            "prompts": "",
            "rag_sequence": [{"key": "graph_rag_neo4j", "enabled": True}],
            "tool_sequence": [{"key": "mcp_couchdb_deposition_access", "enabled": True}],
        },
    )

    assert create_response.status_code == 200
    persona_id = create_response.json()["persona_id"]

    session_response = api_client.post(
        f"/api/admin/personas/{persona_id}/graph-session",
        json={
            "question": "What relationships exist?",
            "answer": "Graph answer output.",
        },
    )

    assert session_response.status_code == 200
    payload = session_response.json()
    assert payload["prompt_template_key"] == "graph_rag_system"
    assert payload["prompt_sections"]["system"] == "Use the graph."
    assert payload["prompt_sections"]["context"] == "Use Neo4j context first."
    assert payload["tool_sequence"] == [{"key": "mcp_couchdb_deposition_access", "enabled": True}]
    assert payload["last_graph_question"] == "What relationships exist?"
    assert payload["last_graph_answer"] == "Graph answer output."
    assert payload["last_graph_asked_at"]

    stored_doc = couchdb.get_doc(persona_id)
    assert stored_doc["last_graph_question"] == "What relationships exist?"
    assert stored_doc["last_graph_answer"] == "Graph answer output."


def test_admin_persona_graph_session_missing_persona_returns_404(api_client, monkeypatch):
    couchdb = _InMemoryDocStore()
    monkeypatch.setattr(main, "couchdb", couchdb)

    response = api_client.post(
        "/api/admin/personas/persona:missing/graph-session",
        json={
            "question": "What is missing?",
            "answer": "No persona exists.",
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Persona not found"


def test_admin_persona_create_rejects_empty_prompt_sections(api_client, monkeypatch):
    couchdb = _InMemoryDocStore()
    monkeypatch.setattr(main, "couchdb", couchdb)

    response = api_client.post(
        "/api/admin/personas",
        json={
            "name": "Empty Prompt Persona",
            "llm_provider": "openai",
            "llm_model": "gpt-5.2",
            "prompt_template_key": "chat_system",
            "prompts": "",
            "prompt_sections": {
                "system": "",
                "assistant": "",
                "context": "",
            },
            "rag_sequence": [],
            "tool_sequence": [],
        },
    )

    assert response.status_code == 400
    assert "persona prompts are required" in response.json()["detail"].lower()


def test_admin_persona_complete_functionality_chain(api_client, monkeypatch):
    """Exercise a full persona workflow chain across multiple admin endpoints."""
    couchdb = _InMemoryDocStore()
    monkeypatch.setattr(main, "couchdb", couchdb)

    templates_response = api_client.get("/api/admin/personas/prompts")
    assert templates_response.status_code == 200
    templates = templates_response.json()["prompts"]
    assert templates
    system_template = next((item for item in templates if "system" in str(item.get("key", "")).lower()), None)
    assistant_template = next((item for item in templates if "assistant" in str(item.get("key", "")).lower()), None)
    context_template = next((item for item in templates if "context" in str(item.get("key", "")).lower()), None)
    assert system_template is not None
    assert context_template is not None
    assistant_content = (
        str(assistant_template["content"]).strip()
        if assistant_template is not None
        else "Respond in concise legal prose and cite only grounded facts."
    )

    rag_options_response = api_client.get("/api/admin/personas/rags")
    assert rag_options_response.status_code == 200
    rag_options = rag_options_response.json()["rags"]
    assert rag_options
    rag_key = rag_options[0]["key"]
    tool_options_response = api_client.get("/api/admin/personas/tools")
    assert tool_options_response.status_code == 200
    tool_options = tool_options_response.json()["tools"]
    assert tool_options
    tool_key = tool_options[0]["key"]

    create_response = api_client.post(
        "/api/admin/personas",
        json={
            "name": "Chain Persona",
            "llm_provider": "openai",
            "llm_model": "gpt-5.2",
            "prompt_template_key": system_template["key"],
            "prompt_sections": {
                "system": system_template["content"],
                "assistant": assistant_content,
                "context": context_template["content"],
            },
            "prompts": "",
            "rag_sequence": [{"key": rag_key, "enabled": True}],
            "tool_sequence": [{"key": tool_key, "enabled": True}],
        },
    )
    assert create_response.status_code == 200
    created = create_response.json()
    persona_id = created["persona_id"]
    assert created["prompt_template_key"] == system_template["key"]
    assert created["rag_sequence"] == [{"key": rag_key, "enabled": True}]
    assert created["tool_sequence"] == [{"key": tool_key, "enabled": True}]
    assert created["prompt_sections"]["system"] == system_template["content"].strip()
    assert created["prompt_sections"]["assistant"] == assistant_content
    assert created["prompt_sections"]["context"] == context_template["content"].strip()

    graph_response = api_client.post(
        f"/api/admin/personas/{persona_id}/graph-session",
        json={
            "question": "List the key ontology relationships.",
            "answer": "Graph chain answer for integration test.",
        },
    )
    assert graph_response.status_code == 200
    graph_payload = graph_response.json()
    assert graph_payload["persona_id"] == persona_id
    assert graph_payload["last_graph_question"] == "List the key ontology relationships."
    assert graph_payload["last_graph_answer"] == "Graph chain answer for integration test."
    assert graph_payload["last_graph_asked_at"]

    update_response = api_client.post(
        "/api/admin/personas",
        json={
            "persona_id": persona_id,
            "name": "Chain Persona v2",
            "llm_provider": "openai",
            "llm_model": "gpt-5.2",
            "prompt_template_key": context_template["key"],
            "prompt_sections": {
                "system": system_template["content"],
                "assistant": assistant_content,
                "context": f"{context_template['content']}\n\nKeep legal context grounded in case facts.",
            },
            "prompts": "",
            "rag_sequence": [{"key": rag_key, "enabled": True}],
            "tool_sequence": [{"key": tool_key, "enabled": False}],
        },
    )
    assert update_response.status_code == 200
    updated = update_response.json()
    assert updated["persona_id"] == persona_id
    assert updated["name"] == "Chain Persona v2"
    assert updated["prompt_template_key"] == context_template["key"]
    assert "Keep legal context grounded in case facts." in updated["prompt_sections"]["context"]
    assert "System:\n" in updated["prompts"]
    assert "Assistant:\n" in updated["prompts"]
    assert "Context:\n" in updated["prompts"]

    sentiment_response = api_client.post(
        "/api/text-sentiment",
        json={"text": updated["prompts"]},
    )
    assert sentiment_response.status_code == 200
    sentiment_payload = sentiment_response.json()
    assert sentiment_payload["word_count"] > 0
    assert sentiment_payload["summary"].startswith("Overall deposition sentiment is")

    list_response = api_client.get("/api/admin/personas")
    assert list_response.status_code == 200
    listed = list_response.json()["personas"]
    match = next((item for item in listed if item["persona_id"] == persona_id), None)
    assert match is not None
    assert match["name"] == "Chain Persona v2"
    assert match["prompt_template_key"] == context_template["key"]
    assert match["rag_sequence"] == [{"key": rag_key, "enabled": True}]
    assert match["tool_sequence"] == [{"key": tool_key, "enabled": False}]
    assert match["last_graph_question"] == "List the key ontology relationships."

    stored_doc = couchdb.get_doc(persona_id)
    assert stored_doc["name"] == "Chain Persona v2"
    assert stored_doc["prompt_template_key"] == context_template["key"]
    assert stored_doc["last_graph_answer"] == "Graph chain answer for integration test."
    assert stored_doc["rag_sequence"] == [{"key": rag_key, "enabled": True}]
    assert stored_doc["tool_sequence"] == [{"key": tool_key, "enabled": False}]


def test_deposition_upload_endpoint_round_trip(api_client, monkeypatch, tmp_path):
    root = tmp_path / "deps"
    target = root / "default"
    target.mkdir(parents=True)
    monkeypatch.setattr(main, "_resolve_upload_directory", lambda _directory: target)
    monkeypatch.setattr(main, "_resolve_upload_root_directory", lambda _target: root)

    response = api_client.post(
        "/api/depositions/upload",
        data={"directory": "/data/depositions/default"},
        files=[("files", ("new_witness.txt", b"Date: April 1, 2025\nWitness: New Witness", "text/plain"))],
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["file_count"] == 1
    saved_path = target / "new_witness.txt"
    root_copy_path = root / "new_witness.txt"
    assert payload["saved_files"] == [str(saved_path)]
    assert payload["root_directory"] == str(root)
    assert payload["copied_to_root_files"] == [str(root_copy_path)]
    assert saved_path.read_text(encoding="utf-8") == "Date: April 1, 2025\nWitness: New Witness"
    assert root_copy_path.read_text(encoding="utf-8") == "Date: April 1, 2025\nWitness: New Witness"


def test_deposition_browser_endpoint_returns_serialized_payload(api_client, monkeypatch):
    base = Path("/data/depositions")
    current = base / "default"
    directories = [
        main.DepositionBrowserEntry(
            path="/data/depositions/default/sub",
            name="sub",
            kind="directory",
        )
    ]
    files = [
        main.DepositionBrowserEntry(
            path="/data/depositions/default/sample.txt",
            name="sample.txt",
            kind="file",
        )
    ]
    monkeypatch.setattr(main, "_resolve_deposition_browser_directory", lambda _path: (base, current))
    monkeypatch.setattr(main, "_list_deposition_browser_entries", lambda _dir: (directories, files))

    response = api_client.get("/api/deposition-browser", params={"path": "/tmp/example.txt"})

    assert response.status_code == 200
    assert response.json() == {
        "base_directory": "/data/depositions",
        "current_directory": "/data/depositions/default",
        "parent_directory": "/data/depositions",
        "wildcard_path": "/data/depositions/default/*.txt",
        "directories": [
            {
                "path": "/data/depositions/default/sub",
                "name": "sub",
                "kind": "directory",
            }
        ],
        "files": [
            {
                "path": "/data/depositions/default/sample.txt",
                "name": "sample.txt",
                "kind": "file",
            }
        ],
    }


def test_grafana_access_endpoint_returns_serialized_payload(api_client, monkeypatch):
    monkeypatch.setattr(
        main,
        "settings",
        SimpleNamespace(
            grafana_url="http://localhost:3000/",
            grafana_admin_user="admin",
            grafana_admin_password="password",
        ),
    )

    response = api_client.get("/api/observability/grafana")

    assert response.status_code == 200
    assert response.json() == {
        "url": "http://localhost:3000",
        "login_url": "http://localhost:3000/login",
        "dashboard_url": "http://localhost:3000/d/attorneyos-observability/attorneyos-observability?orgId=1",
        "username": "admin",
        "password": "password",
    }


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


def test_all_current_observables_have_history(api_client):
    metrics_response = api_client.get("/api/agent-metrics?lookback_hours=24")

    assert metrics_response.status_code == 200
    payload = metrics_response.json()
    all_keys = [item["key"] for item in payload["metrics"]] + [
        item["key"] for item in payload["correctness_metrics"]
    ]
    assert all_keys
    assert {
        "llm_calls_sampled",
        "avg_prompt_context_bytes_per_llm_call",
        "avg_estimated_prompt_tokens_per_llm_call",
        "avg_estimated_output_tokens_per_llm_call",
    }.issubset(set(all_keys))

    for metric_key in all_keys:
        history_response = api_client.get(
            f"/api/agent-metrics/history?metric_key={metric_key}&lookback_hours=24&bucket_hours=2"
        )

        assert history_response.status_code == 200, metric_key
        history_payload = history_response.json()
        assert history_payload["key"] == metric_key
        assert isinstance(history_payload["points"], list)


def test_agent_metrics_endpoint_persists_observable_snapshot(api_client, monkeypatch):
    trace_couchdb = _InMemoryDocStore()
    monkeypatch.setattr(main, "trace_couchdb", trace_couchdb)

    response = api_client.get("/api/agent-metrics?lookback_hours=24")

    assert response.status_code == 200
    snapshots = trace_couchdb.find({"type": "observable_snapshot"}, limit=10)
    assert len(snapshots) == 1
    snapshot = snapshots[0]
    assert snapshot["generated_at"] == response.json()["generated_at"]
    assert len(snapshot["metrics"]) == len(response.json()["metrics"])
    assert len(snapshot["correctness_metrics"]) == len(response.json()["correctness_metrics"])


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


def test_text_sentiment_endpoint_round_trip(api_client):
    response = api_client.post(
        "/api/text-sentiment",
        json={
            "text": "This prompt is strong, clear, and consistent even with one small risk.",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["label"] in {"positive", "neutral", "negative"}
    assert payload["word_count"] > 0
    assert payload["summary"].startswith("Overall deposition sentiment is")


def test_text_sentiment_endpoint_validates_empty_payload(api_client):
    response = api_client.post(
        "/api/text-sentiment",
        json={"text": ""},
    )

    assert response.status_code == 422


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


def test_thought_stream_lifecycle_round_trip(api_client, monkeypatch):
    couchdb = _InMemoryDocStore(
        [
            {
                "_id": "dep:trace-case:1",
                "type": "deposition",
                "case_id": "TRACE-CASE",
                "file_name": "trace.txt",
            }
        ]
    )
    memory_couchdb = _InMemoryDocStore()
    trace_couchdb = _InMemoryDocStore()
    monkeypatch.setattr(main, "couchdb", couchdb)
    monkeypatch.setattr(main, "memory_couchdb", memory_couchdb)
    monkeypatch.setattr(main, "trace_couchdb", trace_couchdb)

    trace_id = "trace-integration-lifecycle"
    main._append_trace_events(
        trace_id,
        status="running",
        legal_clerk=[
            {
                "persona": "Persona:Legal Clerk",
                "phase": "seed",
                "notes": "Seeded for integration lifecycle.",
            }
        ],
    )

    get_response = api_client.get(f"/api/thought-streams/{trace_id}")

    assert get_response.status_code == 200
    assert get_response.json()["thought_stream"]["legal_clerk"][0]["phase"] == "seed"

    save_response = api_client.post(
        f"/api/thought-streams/{trace_id}/save",
        json={"case_id": "TRACE-CASE", "channel": "chat"},
    )

    assert save_response.status_code == 200
    assert save_response.json()["saved"] is True

    case_response = api_client.get("/api/cases/TRACE-CASE")

    assert case_response.status_code == 200
    assert case_response.json()["memory_entries"] == 1
    assert case_response.json()["last_action"] == "thought_stream_save"

    memory_docs = memory_couchdb.find({"type": "case_memory", "case_id": "TRACE-CASE"}, limit=10)
    assert len(memory_docs) == 1
    assert memory_docs[0]["channel"] == "thought_stream"

    delete_response = api_client.delete(f"/api/thought-streams/{trace_id}")

    assert delete_response.status_code == 200
    assert delete_response.json()["deleted"] is True

    missing_response = api_client.get(f"/api/thought-streams/{trace_id}")

    assert missing_response.status_code == 404


def test_rename_case_round_trip_moves_case_docs_memory_and_depositions(api_client, monkeypatch):
    couchdb = _InMemoryDocStore(
        [
            {
                "_id": "case:RENAME-SOURCE",
                "type": "case",
                "case_id": "RENAME-SOURCE",
                "memory_entries": 1,
                "last_action": "save",
            },
            {
                "_id": "dep:rename-source:1",
                "type": "deposition",
                "case_id": "RENAME-SOURCE",
                "file_name": "rename.txt",
                "witness_name": "Rename Witness",
                "contradiction_score": 12,
                "flagged": False,
            },
        ]
    )
    memory_couchdb = _InMemoryDocStore(
        [
            {
                "_id": "mem:rename-source:1",
                "type": "case_memory",
                "case_id": "RENAME-SOURCE",
                "channel": "chat",
                "payload": {"response": "hello"},
            }
        ]
    )
    monkeypatch.setattr(main, "couchdb", couchdb)
    monkeypatch.setattr(main, "memory_couchdb", memory_couchdb)

    rename_response = api_client.put(
        "/api/cases/RENAME-SOURCE/rename",
        json={"new_case_id": "RENAME-TARGET"},
    )

    assert rename_response.status_code == 200
    assert rename_response.json()["moved_docs"] == 3

    old_case_response = api_client.get("/api/cases/RENAME-SOURCE")
    assert old_case_response.status_code == 404

    new_case_response = api_client.get("/api/cases/RENAME-TARGET")

    assert new_case_response.status_code == 200
    assert new_case_response.json()["case_id"] == "RENAME-TARGET"
    assert new_case_response.json()["deposition_count"] == 1
    assert new_case_response.json()["last_action"] == "rename"

    depositions_response = api_client.get("/api/depositions/RENAME-TARGET")

    assert depositions_response.status_code == 200
    assert len(depositions_response.json()) == 1
    assert depositions_response.json()[0]["case_id"] == "RENAME-TARGET"

    moved_memory = memory_couchdb.find({"type": "case_memory", "case_id": "RENAME-TARGET"}, limit=10)
    assert len(moved_memory) == 1


def test_clear_case_depositions_preserves_case_metadata(api_client, monkeypatch):
    couchdb = _InMemoryDocStore(
        [
            {
                "_id": "case:CLEAR-ME",
                "type": "case",
                "case_id": "CLEAR-ME",
                "memory_entries": 3,
                "last_action": "save",
            },
            {
                "_id": "dep:clear-me:1",
                "type": "deposition",
                "case_id": "CLEAR-ME",
                "file_name": "one.txt",
                "witness_name": "One",
            },
            {
                "_id": "dep:clear-me:2",
                "type": "deposition",
                "case_id": "CLEAR-ME",
                "file_name": "two.txt",
                "witness_name": "Two",
            },
        ]
    )
    monkeypatch.setattr(main, "couchdb", couchdb)

    clear_response = api_client.delete("/api/cases/CLEAR-ME/depositions")

    assert clear_response.status_code == 200
    assert clear_response.json()["deleted_depositions"] == 2

    detail_response = api_client.get("/api/cases/CLEAR-ME")

    assert detail_response.status_code == 200
    assert detail_response.json()["deposition_count"] == 0
    assert detail_response.json()["last_action"] == "refresh"

    depositions_response = api_client.get("/api/depositions/CLEAR-ME")

    assert depositions_response.status_code == 200
    assert depositions_response.json() == []


def test_ingest_case_round_trip_populates_case_and_depositions(api_client, monkeypatch, tmp_path):
    txt_file = tmp_path / "ingest_target.txt"
    txt_file.write_text("Date: April 1, 2025\nWitness: Integration Witness", encoding="utf-8")

    couchdb = _InMemoryDocStore()
    memory_couchdb = _InMemoryDocStore()
    trace_couchdb = _InMemoryDocStore()
    monkeypatch.setattr(main, "couchdb", couchdb)
    monkeypatch.setattr(main, "memory_couchdb", memory_couchdb)
    monkeypatch.setattr(main, "trace_couchdb", trace_couchdb)
    monkeypatch.setattr(main, "_resolve_ingest_txt_files", lambda _directory: [txt_file])
    monkeypatch.setattr(
        main,
        "_resolve_request_llm",
        lambda llm_provider, llm_model: (llm_provider or "openai", llm_model or "gpt-4o-mini"),
    )

    def _run_workflow(
        *,
        case_id,
        file_path,
        llm_provider,
        llm_model,
        schema_name,
        selected_schema,
        selected_schema_mode,
    ):
        assert case_id == "INGEST-CASE"
        assert file_path == str(txt_file)
        assert llm_provider == "openai"
        assert llm_model == "gpt-4o-mini"
        assert schema_name == "deposition_schema_g1"
        assert selected_schema_mode == "raw_capture"
        assert isinstance(selected_schema, dict)
        stored = couchdb.save_doc(
            {
                "_id": "dep:ingest-case:1",
                "type": "deposition",
                "case_id": case_id,
                "file_name": Path(file_path).name,
                "witness_name": "Integration Witness",
                "contradiction_score": 9,
                "flagged": False,
            }
        )
        return {
            "deposition_doc": stored,
            "legal_clerk_trace": [],
            "attorney_trace": [],
        }

    monkeypatch.setattr(main.workflow, "run", _run_workflow)
    monkeypatch.setattr(main.workflow, "reassess_case", lambda *_args, **_kwargs: [])

    ingest_response = api_client.post(
        "/api/ingest-case",
        json={
            "case_id": "INGEST-CASE",
            "directory": str(tmp_path),
            "llm_provider": "openai",
            "llm_model": "gpt-4o-mini",
            "schema_name": "deposition_schema_g1",
        },
    )

    assert ingest_response.status_code == 200
    payload = ingest_response.json()
    assert payload["case_id"] == "INGEST-CASE"
    assert len(payload["ingested"]) == 1
    assert payload["ingested"][0]["file_name"] == "ingest_target.txt"

    list_response = api_client.get("/api/depositions/INGEST-CASE")

    assert list_response.status_code == 200
    assert len(list_response.json()) == 1
    assert list_response.json()[0]["witness_name"] == "Integration Witness"

    detail_response = api_client.get("/api/cases/INGEST-CASE")

    assert detail_response.status_code == 200
    assert detail_response.json()["deposition_count"] == 1
    assert detail_response.json()["memory_entries"] == 1
    assert detail_response.json()["last_action"] == "ingest"


def test_ingest_schema_options_endpoint_returns_discovered_schemas(api_client):
    response = api_client.get("/api/ingest-schemas")

    assert response.status_code == 200
    payload = response.json()
    keys = {item["key"] for item in payload}
    assert "deposition_schema" in keys
    assert "deposition_schema_g1" in keys
    assert "model_schema" not in keys


def test_ingest_schema_crud_round_trip(api_client, monkeypatch):
    couchdb = _InMemoryDocStore()
    monkeypatch.setattr(main, "couchdb", couchdb)

    create_response = api_client.post(
        "/api/ingest-schemas",
        json={
            "key": "custom_legal_schema",
            "schema": {
                "title": "Custom Legal Schema",
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                },
            },
        },
    )

    assert create_response.status_code == 200
    assert create_response.json()["key"] == "custom_legal_schema"
    assert create_response.json()["removable"] is True

    list_response = api_client.get("/api/ingest-schemas")

    assert list_response.status_code == 200
    by_key = {item["key"]: item for item in list_response.json()}
    assert "custom_legal_schema" in by_key
    assert by_key["custom_legal_schema"]["schema"]["title"] == "Custom Legal Schema"

    delete_response = api_client.delete("/api/ingest-schemas/custom_legal_schema")

    assert delete_response.status_code == 200
    assert delete_response.json() == {"deleted": True, "key": "custom_legal_schema"}

    final_list_response = api_client.get("/api/ingest-schemas")
    final_keys = {item["key"] for item in final_list_response.json()}
    assert "custom_legal_schema" not in final_keys


def test_graph_rag_query_round_trip_streams_trace_and_rag_records(api_client, monkeypatch):
    rag_couchdb = _InMemoryDocStore()
    trace_couchdb = _InMemoryDocStore()
    monkeypatch.setattr(main, "rag_couchdb", rag_couchdb)
    monkeypatch.setattr(main, "trace_couchdb", trace_couchdb)
    monkeypatch.setattr(
        main.neo4j_graph,
        "retrieve_context",
        lambda question, node_limit, **_kwargs: {
            "resource_count": 1,
            "terms": ["material breach"],
            "context_text": f"Graph context for: {question}",
            "resources": [
                {
                    "iri": "urn:test:1",
                    "label": "Material Breach",
                    "relations": [
                        {
                            "predicate": "relatedTo",
                            "object_label": "Contract",
                            "object_iri": "urn:test:contract",
                        }
                    ],
                    "literals": [
                        {
                            "predicate": "definition",
                            "value": "An uncured failure of a core obligation.",
                            "datatype": "xsd:string",
                            "lang": "en",
                        }
                    ],
                }
            ],
        },
    )

    class _FakeLlm:
        def invoke(self, _messages):
            return type("_Result", (), {"content": "Short answer: Graph-backed answer."})()

    monkeypatch.setattr(main, "build_chat_model", lambda *_args, **_kwargs: _FakeLlm())

    response = api_client.post(
        "/api/graph-rag/query",
        json={
            "question": "What is a material breach?",
            "use_rag": True,
            "stream_rag": True,
            "top_k": 3,
            "thought_stream_id": "graph-rag-integration",
            "llm_provider": "openai",
            "llm_model": "gpt-4o-mini",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "Short answer: Graph-backed answer."
    assert payload["context_rows"] == 1
    assert payload["sources"] == [{"iri": "urn:test:1", "label": "Material Breach"}]
    assert payload["monitor"]["rag_enabled"] is True
    assert payload["monitor"]["rag_stream_enabled"] is True
    assert payload["monitor"]["retrieval_terms"] == ["material breach"]

    trace_response = api_client.get("/api/thought-streams/graph-rag-integration")

    assert trace_response.status_code == 200
    assert trace_response.json()["status"] == "completed"
    assert trace_response.json()["thought_stream"]["attorney"][-1]["phase"] == "graph_rag_answer"

    rag_docs = rag_couchdb.find({"type": "rag_stream"}, limit=10)
    assert len(rag_docs) == 1
    assert rag_docs[0]["status"] == "completed"
    assert rag_docs[0]["phase"] == "answer"
    assert rag_docs[0]["context_rows"] == 1


def test_graph_rag_query_with_stream_disabled_skips_rag_stream_persistence(api_client, monkeypatch):
    rag_couchdb = _InMemoryDocStore()
    trace_couchdb = _InMemoryDocStore()
    monkeypatch.setattr(main, "rag_couchdb", rag_couchdb)
    monkeypatch.setattr(main, "trace_couchdb", trace_couchdb)
    monkeypatch.setattr(
        main.neo4j_graph,
        "retrieve_context",
        lambda question, node_limit, **_kwargs: {
            "resource_count": 1,
            "terms": ["duty"],
            "context_text": f"Graph context for: {question}",
            "resources": [
                {
                    "iri": "urn:test:duty",
                    "label": "Duty",
                    "relations": [],
                    "literals": [],
                }
            ],
        },
    )

    class _FakeLlm:
        def invoke(self, _messages):
            return type("_Result", (), {"content": "Short answer: Stream-disabled graph answer."})()

    monkeypatch.setattr(main, "build_chat_model", lambda *_args, **_kwargs: _FakeLlm())

    response = api_client.post(
        "/api/graph-rag/query",
        json={
            "question": "Define legal duty.",
            "use_rag": True,
            "stream_rag": False,
            "top_k": 3,
            "thought_stream_id": "graph-rag-stream-off",
            "llm_provider": "openai",
            "llm_model": "gpt-4o-mini",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "Short answer: Stream-disabled graph answer."
    assert payload["context_rows"] == 1
    assert payload["monitor"]["rag_enabled"] is True
    assert payload["monitor"]["rag_stream_enabled"] is False

    trace_response = api_client.get("/api/thought-streams/graph-rag-stream-off")
    assert trace_response.status_code == 200
    assert trace_response.json()["status"] == "completed"
    assert trace_response.json()["thought_stream"]["attorney"][-1]["phase"] == "graph_rag_answer"

    rag_docs = rag_couchdb.find({"type": "rag_stream"}, limit=10)
    assert rag_docs == []


def test_graph_rag_query_without_rag_returns_answer_and_marks_disabled_phase(api_client, monkeypatch):
    rag_couchdb = _InMemoryDocStore()
    trace_couchdb = _InMemoryDocStore()
    monkeypatch.setattr(main, "rag_couchdb", rag_couchdb)
    monkeypatch.setattr(main, "trace_couchdb", trace_couchdb)
    monkeypatch.setattr(
        main.neo4j_graph,
        "retrieve_context",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("retrieve_context should not be called")),
    )

    class _FakeLlm:
        def invoke(self, _messages):
            return type("_Result", (), {"content": "Short answer: Non-RAG response."})()

    monkeypatch.setattr(main, "build_chat_model", lambda *_args, **_kwargs: _FakeLlm())

    response = api_client.post(
        "/api/graph-rag/query",
        json={
            "question": "Answer without retrieval.",
            "use_rag": False,
            "stream_rag": False,
            "top_k": 5,
            "thought_stream_id": "graph-rag-disabled",
            "llm_provider": "openai",
            "llm_model": "gpt-4o-mini",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "Short answer: Non-RAG response."
    assert payload["context_rows"] == 0
    assert payload["monitor"]["rag_enabled"] is False
    assert payload["monitor"]["rag_stream_enabled"] is False
    assert payload["monitor"]["retrieval_terms"] == []
    assert payload["monitor"]["context_preview"] == "RAG processing was disabled for this request."

    trace_response = api_client.get("/api/thought-streams/graph-rag-disabled")
    assert trace_response.status_code == 200
    assert trace_response.json()["status"] == "completed"
    legal_clerk_phases = [item["phase"] for item in trace_response.json()["thought_stream"]["legal_clerk"]]
    assert "graph_rag_disabled" in legal_clerk_phases

    rag_docs = rag_couchdb.find({"type": "rag_stream"}, limit=10)
    assert rag_docs == []


def test_add_deposition_root_then_list_directories_includes_uploaded_root(api_client, monkeypatch, tmp_path):
    couchdb = _InMemoryDocStore()
    monkeypatch.setattr(main, "couchdb", couchdb)

    extra_root = tmp_path / "import-root"
    nested = extra_root / "nested"
    nested.mkdir(parents=True)
    (extra_root / "root_doc.txt").write_text("Date: April 2, 2025\nWitness: Root", encoding="utf-8")
    (nested / "nested_doc.txt").write_text("Date: April 3, 2025\nWitness: Nested", encoding="utf-8")

    add_response = api_client.post(
        "/api/deposition-roots",
        json={"path": str(extra_root)},
    )

    assert add_response.status_code == 200
    assert add_response.json()["path"] == str(extra_root)

    directories_response = api_client.get("/api/deposition-directories")

    assert directories_response.status_code == 200
    option_paths = {item["path"] for item in directories_response.json()["options"]}
    assert str(extra_root) in option_paths
    assert str(nested) in option_paths
