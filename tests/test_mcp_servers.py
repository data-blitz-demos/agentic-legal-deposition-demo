# Copyright (c) 2026 Data-Blitz Inc. All rights reserved.
# License: Proprietary. See NOTICE.md.
# Author: Paul Harvener.

from __future__ import annotations

import importlib
import runpy
import sys
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeThoughtStreamClient:
    def __init__(self) -> None:
        self.docs: dict[str, dict] = {}
        self.closed = False
        self.ensured = False
        self.deleted: list[tuple[str, str | None]] = []

    def ensure_db(self) -> None:
        self.ensured = True

    def close(self) -> None:
        self.closed = True

    def get_doc(self, doc_id: str) -> dict:
        if doc_id not in self.docs:
            raise RuntimeError("missing")
        return deepcopy(self.docs[doc_id])

    def update_doc(self, doc: dict) -> dict:
        saved = deepcopy(doc)
        saved["_rev"] = "1-fake"
        self.docs[str(saved["_id"])] = deepcopy(saved)
        return saved

    def find(self, selector: dict, limit: int = 200) -> list[dict]:
        matches = []
        for item in self.docs.values():
            if all(item.get(key) == value for key, value in selector.items()):
                matches.append(deepcopy(item))
        return matches[:limit]

    def delete_doc(self, doc_id: str, rev: str | None = None) -> None:
        self.deleted.append((doc_id, rev))
        self.docs.pop(doc_id, None)


class FakeDepositionClient:
    def __init__(self, docs: list[dict] | None = None) -> None:
        self.docs = {str(doc["_id"]): deepcopy(doc) for doc in (docs or [])}
        self.closed = False
        self.ensured = False

    def ensure_db(self) -> None:
        self.ensured = True

    def close(self) -> None:
        self.closed = True

    def list_depositions(self, case_id: str) -> list[dict]:
        return [deepcopy(doc) for doc in self.docs.values() if doc.get("case_id") == case_id]

    def get_doc(self, deposition_id: str) -> dict:
        return deepcopy(self.docs[deposition_id])


class FakeOntologyGraph:
    def __init__(self) -> None:
        self.closed = False
        self.calls: list[tuple[str, tuple, dict]] = []

    def close(self) -> None:
        self.closed = True

    def health(self) -> dict:
        self.calls.append(("health", (), {}))
        return {"ok": True}

    def retrieve_context(self, question: str, node_limit: int = 8) -> dict:
        self.calls.append(("retrieve_context", (question,), {"node_limit": node_limit}))
        return {"resources": [{"iri": "iri:contract"}], "count": 1}

    def list_resources(self, search: str = "", limit: int = 50) -> list[dict]:
        self.calls.append(("list_resources", (), {"search": search, "limit": limit}))
        return [{"iri": "iri:contract", "label": "Contract"}]

    def get_resource(self, iri: str, neighbor_limit: int = 20, literal_limit: int = 20) -> dict | None:
        self.calls.append(
            (
                "get_resource",
                (iri,),
                {"neighbor_limit": neighbor_limit, "literal_limit": literal_limit},
            )
        )
        if iri == "missing":
            return None
        return {"iri": iri, "neighbors": [], "literals": []}


def _import_fresh(module_name: str):
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def test_thought_stream_with_client_and_error_branches(monkeypatch):
    ts = _import_fresh("mcp_servers.thought_stream_server")
    fake_client = FakeThoughtStreamClient()
    logger = Mock()

    monkeypatch.setattr(ts, "_settings", lambda: ("http://couchdb", "thought_stream"))
    monkeypatch.setattr(ts, "CouchDBClient", lambda url, db: fake_client)
    monkeypatch.setattr(ts, "logger", logger)

    result = ts._with_client(lambda client: {"same_client": client is fake_client})
    assert result == {"same_client": True}
    assert fake_client.ensured is True
    assert fake_client.closed is True

    fake_client.closed = False
    with pytest.raises(RuntimeError, match="boom"):
        ts._with_client(lambda _client: (_ for _ in ()).throw(RuntimeError("boom")))
    assert fake_client.closed is True
    assert any(call.args and call.args[0] == "langfuse mcp_client_failure server=couchdb_thought_stream_access db=%s" for call in logger.exception.call_args_list)

    assert ts._max_sequence({"legal_clerk": [{"sequence": "bad"}], "attorney": [{"sequence": 4}]}) == 4

    with pytest.raises(ValueError, match="trace_id is required"):
        ts.get_thought_stream("   ")

    with pytest.raises(ValueError, match="trace_id is required"):
        ts.delete_thought_stream("   ")

    class NonDictClient:
        def get_doc(self, _doc_id: str):
            return "not-a-dict"

    monkeypatch.setattr(ts, "_with_client", lambda fn: fn(NonDictClient()))
    with pytest.raises(ValueError, match="was not found"):
        ts.get_thought_stream("trace-404")


@pytest.mark.parametrize(
    ("module_name", "path_name"),
    [
        ("mcp_servers.thought_stream_server", "thought_stream_server.py"),
        ("mcp_servers.couchdb_server", "couchdb_server.py"),
        ("mcp_servers.neo4j_ontology_server", "neo4j_ontology_server.py"),
    ],
)
def test_mcp_server_main_runs_stdio(monkeypatch, module_name: str, path_name: str):
    project_root = str(REPO_ROOT)
    original_path = list(sys.path)
    sys.modules.pop(module_name, None)
    monkeypatch.setattr(sys, "path", [entry for entry in original_path if entry != project_root])

    import mcp.server.fastmcp as fastmcp_module

    calls: list[str | None] = []
    monkeypatch.setattr(
        fastmcp_module.FastMCP,
        "run",
        lambda self, transport=None: calls.append(transport),
    )

    runpy.run_path(str(REPO_ROOT / "mcp_servers" / path_name), run_name="__main__")

    assert calls == ["stdio"]
    assert project_root in sys.path


def test_couchdb_server_end_to_end(monkeypatch):
    cs = _import_fresh("mcp_servers.couchdb_server")
    docs = [
        {
            "_id": "dep-1",
            "case_id": "CASE-1",
            "witness_name": "Alice",
            "contradiction_score": 2,
            "flagged": False,
            "claims": [{"topic": "Timeline", "statement": "Met on Monday", "source_quote": "Monday"}],
        },
        {
            "_id": "dep-2",
            "case_id": "CASE-1",
            "witness_name": "Bob",
            "contradiction_score": 7,
            "flagged": True,
            "claims": [{"topic": "Contract", "statement": "Signed later", "source_quote": "contract"}],
        },
    ]
    fake_client = FakeDepositionClient(docs)
    logger = Mock()

    monkeypatch.setattr(cs, "_settings", lambda: ("http://couchdb", "depositions"))
    monkeypatch.setattr(cs, "CouchDBClient", lambda url, db: fake_client)
    monkeypatch.setattr(cs, "logger", logger)

    assert cs._settings() == ("http://couchdb", "depositions")

    listed = cs.list_case_depositions("CASE-1", limit=0)
    assert listed["count"] == 1
    assert listed["depositions"][0]["_id"] == "dep-2"

    flagged = cs.list_flagged_depositions("CASE-1", min_score=999)
    assert flagged["min_score"] == 100
    assert flagged["count"] == 1
    assert flagged["depositions"][0]["_id"] == "dep-2"

    search = cs.search_claims("CASE-1", "contract", limit=1)
    assert search["count"] == 1
    assert search["matches"][0]["deposition_id"] == "dep-2"

    no_match = cs.search_claims("CASE-1", "   ", limit=5)
    assert no_match["count"] == 0

    assert cs.get_deposition("dep-1")["_id"] == "dep-1"
    assert fake_client.ensured is True
    assert fake_client.closed is True
    messages = [call.args[0] for call in logger.info.call_args_list if call.args]
    assert "langfuse mcp_request tool=list_case_depositions case_id=%s requested_limit=%s bounded_limit=%s" in messages
    assert "langfuse mcp_result tool=list_case_depositions case_id=%s count=%s" in messages
    assert "langfuse mcp_request tool=get_deposition deposition_id=%s" in messages
    assert "langfuse mcp_result tool=search_claims case_id=%s count=%s" in messages


def test_couchdb_server_settings_and_with_client(monkeypatch):
    cs = _import_fresh("mcp_servers.couchdb_server")
    monkeypatch.delenv("COUCHDB_URL", raising=False)
    monkeypatch.delenv("COUCHDB_DB", raising=False)
    assert cs._settings() == ("http://admin:password@localhost:5984", "depositions")

    monkeypatch.setenv("COUCHDB_URL", "http://user:pass@host:5984")
    monkeypatch.setenv("COUCHDB_DB", "legal_docs")
    assert cs._settings() == ("http://user:pass@host:5984", "legal_docs")

    fake_client = FakeDepositionClient()
    logger = Mock()
    monkeypatch.setattr(cs, "CouchDBClient", lambda url, db: fake_client)
    monkeypatch.setattr(cs, "logger", logger)

    assert cs._with_client(lambda client: client is fake_client) is True
    assert fake_client.ensured is True
    assert fake_client.closed is True
    assert any(call.args and call.args[0] == "langfuse mcp_client_open server=couchdb_deposition_access db=%s" for call in logger.info.call_args_list)

    fake_client.closed = False
    with pytest.raises(RuntimeError, match="boom"):
        cs._with_client(lambda _client: (_ for _ in ()).throw(RuntimeError("boom")))
    assert fake_client.closed is True
    assert any(call.args and call.args[0] == "langfuse mcp_client_failure server=couchdb_deposition_access db=%s" for call in logger.exception.call_args_list)


def test_neo4j_server_end_to_end(monkeypatch):
    ns = _import_fresh("mcp_servers.neo4j_ontology_server")
    fake_graph = FakeOntologyGraph()
    logger = Mock()

    monkeypatch.setattr(
        ns,
        "_settings",
        lambda: {
            "uri": "bolt://localhost:7687",
            "user": "neo4j",
            "password": "password",
            "database": "neo4j",
            "browser_url": "http://localhost:7474/browser/",
        },
    )
    monkeypatch.setattr(ns, "Neo4jOntologyGraph", lambda **kwargs: fake_graph)
    monkeypatch.setattr(ns, "logger", logger)

    assert ns.ontology_health() == {"ok": True}

    context = ns.search_ontology_context("  What is Contract?  ", node_limit=999)
    assert context["question"] == "What is Contract?"
    assert context["node_limit"] == 50

    listing = ns.list_ontology_resources("  contract  ", limit=0)
    assert listing["search"] == "contract"
    assert listing["limit"] == 50
    assert listing["count"] == 1

    resource = ns.get_ontology_resource("  iri:contract  ", neighbor_limit=999, literal_limit=-1)
    assert resource["iri"] == "iri:contract"

    with pytest.raises(ValueError, match="question is required"):
        ns.search_ontology_context("   ")

    with pytest.raises(ValueError, match="iri is required"):
        ns.get_ontology_resource("   ")

    with pytest.raises(ValueError, match="was not found"):
        ns.get_ontology_resource("missing")

    assert fake_graph.closed is True
    messages = [call.args[0] for call in logger.info.call_args_list if call.args]
    assert "langfuse mcp_request tool=ontology_health" in messages
    assert "langfuse mcp_result tool=search_ontology_context resource_count=%s retrieval_mode=%s" in messages
    assert "langfuse mcp_request tool=get_ontology_resource iri=%s neighbor_limit=%s literal_limit=%s" in messages


def test_neo4j_server_settings_and_with_graph(monkeypatch):
    ns = _import_fresh("mcp_servers.neo4j_ontology_server")
    monkeypatch.delenv("NEO4J_URI", raising=False)
    monkeypatch.delenv("NEO4J_USER", raising=False)
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
    monkeypatch.delenv("NEO4J_DATABASE", raising=False)
    monkeypatch.delenv("NEO4J_BROWSER_URL", raising=False)
    assert ns._settings() == {
        "uri": "bolt://localhost:7687",
        "user": "neo4j",
        "password": "password",
        "database": "neo4j",
        "browser_url": "http://localhost:7474/browser/",
    }

    monkeypatch.setenv("NEO4J_URI", "bolt://example:7687")
    monkeypatch.setenv("NEO4J_USER", "graph-user")
    monkeypatch.setenv("NEO4J_PASSWORD", "graph-pass")
    monkeypatch.setenv("NEO4J_DATABASE", "contracts")
    monkeypatch.setenv("NEO4J_BROWSER_URL", "http://example/browser/")
    assert ns._settings() == {
        "uri": "bolt://example:7687",
        "user": "graph-user",
        "password": "graph-pass",
        "database": "contracts",
        "browser_url": "http://example/browser/",
    }

    fake_graph = FakeOntologyGraph()
    logger = Mock()
    monkeypatch.setattr(ns, "Neo4jOntologyGraph", lambda **kwargs: fake_graph)
    monkeypatch.setattr(ns, "logger", logger)
    assert ns._with_graph(lambda graph: graph is fake_graph) is True
    assert fake_graph.closed is True
    assert any(call.args and call.args[0] == "langfuse mcp_graph_open server=neo4j_legal_ontology_access database=%s" for call in logger.info.call_args_list)
