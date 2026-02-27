from __future__ import annotations

from copy import deepcopy

import pytest

from mcp_servers import thought_stream_server as ts


class FakeCouchDBClient:
    def __init__(self) -> None:
        self.docs: dict[str, dict] = {}
        self.rev_counter: dict[str, int] = {}

    def ensure_db(self) -> None:
        return None

    def close(self) -> None:
        return None

    def get_doc(self, doc_id: str) -> dict:
        if doc_id not in self.docs:
            raise RuntimeError("missing")
        return deepcopy(self.docs[doc_id])

    def update_doc(self, doc: dict) -> dict:
        doc_id = str(doc["_id"])
        rev = self.rev_counter.get(doc_id, 0) + 1
        self.rev_counter[doc_id] = rev
        saved = deepcopy(doc)
        saved["_rev"] = f"{rev}-fake"
        self.docs[doc_id] = deepcopy(saved)
        return saved

    def find(self, selector: dict, limit: int = 200) -> list[dict]:
        matches: list[dict] = []
        for item in self.docs.values():
            if all(item.get(key) == value for key, value in selector.items()):
                matches.append(deepcopy(item))
        return matches[:limit]

    def delete_doc(self, doc_id: str, rev: str | None = None) -> None:
        self.docs.pop(doc_id, None)


@pytest.fixture
def fake_client(monkeypatch) -> FakeCouchDBClient:
    client = FakeCouchDBClient()
    monkeypatch.setattr(ts, "_with_client", lambda fn: fn(client))
    return client


def test_settings_defaults_and_overrides(monkeypatch):
    monkeypatch.delenv("COUCHDB_URL", raising=False)
    monkeypatch.delenv("THOUGHT_STREAM_DB", raising=False)
    assert ts._settings() == ("http://admin:password@localhost:5984", "thought_stream")

    monkeypatch.setenv("COUCHDB_URL", "http://user:pass@host:5984")
    monkeypatch.setenv("THOUGHT_STREAM_DB", "ts_db")
    assert ts._settings() == ("http://user:pass@host:5984", "ts_db")


def test_trace_doc_id_and_status_validation():
    assert ts._trace_doc_id("abc") == ts._trace_doc_id("abc")
    assert ts._normalize_status("running") == "running"
    assert ts._normalize_status(" COMPLETED ") == "completed"
    with pytest.raises(ValueError, match="status must be one of"):
        ts._normalize_status("pending")


def test_coerce_events_validation():
    assert ts._coerce_events(None) == []
    assert ts._coerce_events([{"phase": "x"}]) == [{"phase": "x"}]
    with pytest.raises(ValueError, match="event must be an object"):
        ts._coerce_events(["not-a-dict"])  # type: ignore[list-item]


def test_thought_stream_health(fake_client):
    payload = ts.thought_stream_health()
    assert payload["connected"] is True
    assert "database" in payload


def test_append_get_list_delete_flow(fake_client):
    trace_id = "trace-1"

    append_payload = ts.append_thought_stream_events(
        trace_id=trace_id,
        case_id="CASE-1",
        status="running",
        legal_clerk=[{"persona": "Persona:Legal Clerk", "phase": "ingest_start"}],
        attorney=[{"persona": "Persona:Attorney", "phase": "assessment_start"}],
    )

    assert append_payload["trace_id"] == trace_id
    assert append_payload["appended"] == {"legal_clerk": 1, "attorney": 1}

    doc = ts.get_thought_stream(trace_id)
    assert doc["trace_id"] == trace_id
    assert doc["case_id"] == "CASE-1"
    assert len(doc["trace"]["legal_clerk"]) == 1
    assert len(doc["trace"]["attorney"]) == 1

    listed = ts.list_thought_streams(case_id="CASE-1", limit=10)
    assert listed["count"] == 1
    assert listed["thought_streams"][0]["trace_id"] == trace_id

    deleted = ts.delete_thought_stream(trace_id)
    assert deleted["deleted"] is True

    deleted_again = ts.delete_thought_stream(trace_id)
    assert deleted_again["deleted"] is False


def test_append_assigns_sequence_across_calls(fake_client):
    trace_id = "trace-seq"

    ts.append_thought_stream_events(
        trace_id=trace_id,
        legal_clerk=[{"persona": "Persona:Legal Clerk", "phase": "step1"}],
    )
    ts.append_thought_stream_events(
        trace_id=trace_id,
        attorney=[{"persona": "Persona:Attorney", "phase": "step2"}],
        status="completed",
    )

    doc = ts.get_thought_stream(trace_id)
    seq_1 = doc["trace"]["legal_clerk"][0]["sequence"]
    seq_2 = doc["trace"]["attorney"][0]["sequence"]

    assert seq_1 == 1
    assert seq_2 == 2
    assert doc["status"] == "completed"


def test_append_requires_trace_id_and_valid_status(fake_client):
    with pytest.raises(ValueError, match="trace_id is required"):
        ts.append_thought_stream_events(trace_id="   ")

    with pytest.raises(ValueError, match="status must be one of"):
        ts.append_thought_stream_events(trace_id="trace-1", status="unknown")
