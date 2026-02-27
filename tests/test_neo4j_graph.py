from __future__ import annotations

from pathlib import Path

import pytest
from rdflib import BNode, URIRef

from backend.app import neo4j_graph as neo_module


class _FakeResult:
    def single(self):
        return {"ok": 1}


class _FakeSession:
    def __init__(self, should_fail: bool = False):
        self.calls: list[tuple[str, dict]] = []
        self.should_fail = should_fail

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def run(self, query, **params):
        if self.should_fail:
            raise RuntimeError("neo4j down")
        self.calls.append((str(query).strip(), params))
        return _FakeResult()


class _FakeDriver:
    def __init__(self, should_fail: bool = False):
        self.session_obj = _FakeSession(should_fail=should_fail)
        self.closed = False
        self.last_database = None

    def session(self, database=None):
        self.last_database = database
        return self.session_obj

    def close(self):
        self.closed = True


class _FakeGraphDatabase:
    @staticmethod
    def driver(_uri, auth=None):
        assert auth is not None
        return _FakeDriver()


def _sample_owl_text() -> str:
    return """<?xml version=\"1.0\"?>
<rdf:RDF
  xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\"
  xmlns:rdfs=\"http://www.w3.org/2000/01/rdf-schema#\"
  xmlns:ex=\"http://example.org/\">
  <rdf:Description rdf:about=\"http://example.org/A\">
    <rdf:type rdf:resource=\"http://example.org/LegalConcept\"/>
    <rdfs:label>Alpha Concept</rdfs:label>
    <ex:relatedTo rdf:resource=\"http://example.org/B\"/>
  </rdf:Description>
</rdf:RDF>
"""


def test_split_iri_and_resource_helpers():
    namespace, local = neo_module.split_iri("http://example.org/legal#Class")
    assert namespace == "http://example.org/legal#"
    assert local == "Class"

    ns2, local2 = neo_module.split_iri("plain")
    assert ns2 == ""
    assert local2 == "plain"
    ns3, local3 = neo_module.split_iri("http://example.org/")
    assert ns3 == ""
    assert local3 == "http://example.org/"

    assert neo_module.node_to_iri(URIRef("http://example.org/X")) == "http://example.org/X"
    bnode_iri = neo_module.node_to_iri(BNode("abc"))
    assert bnode_iri.startswith("_bnode:")

    row = neo_module.resource_row("http://example.org/X")
    assert row["kind"] == "uri"
    assert row["local_name"] == "X"

    bnode_row = neo_module.resource_row("_bnode:123")
    assert bnode_row["kind"] == "bnode"

    assert neo_module.literal_key("A", "", "") == neo_module.literal_key("A", "", "")


def test_get_driver_requires_configuration():
    graph = neo_module.Neo4jOntologyGraph(
        uri="",
        user="",
        password="",
        database="neo4j",
        browser_url="http://localhost:7474/browser/",
    )

    with pytest.raises(RuntimeError, match="Neo4j is not configured"):
        graph._get_driver()


def test_get_driver_requires_neo4j_package(monkeypatch):
    graph = neo_module.Neo4jOntologyGraph(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password",
        database="neo4j",
        browser_url="http://localhost:7474/browser/",
    )
    monkeypatch.setattr(neo_module, "GraphDatabase", None)

    with pytest.raises(RuntimeError, match="neo4j package is not installed"):
        graph._get_driver()


def test_health_when_not_configured():
    graph = neo_module.Neo4jOntologyGraph(
        uri="",
        user="",
        password="",
        database="neo4j",
        browser_url="http://localhost:7474/browser/",
    )

    payload = graph.health()

    assert payload["configured"] is False
    assert payload["connected"] is False
    assert "not configured" in payload["error"].lower()


def test_health_when_connected(monkeypatch):
    graph = neo_module.Neo4jOntologyGraph(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password",
        database="neo4j",
        browser_url="http://localhost:7474/browser/",
    )
    monkeypatch.setattr(neo_module, "GraphDatabase", _FakeGraphDatabase)

    payload = graph.health()

    assert payload["configured"] is True
    assert payload["connected"] is True
    assert payload["error"] is None


def test_health_when_driver_fails(monkeypatch):
    graph = neo_module.Neo4jOntologyGraph(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password",
        database="neo4j",
        browser_url="http://localhost:7474/browser/",
    )

    class _BadGraphDatabase:
        @staticmethod
        def driver(_uri, auth=None):
            assert auth is not None
            return _FakeDriver(should_fail=True)

    monkeypatch.setattr(neo_module, "GraphDatabase", _BadGraphDatabase)

    payload = graph.health()

    assert payload["configured"] is True
    assert payload["connected"] is False
    assert "neo4j down" in str(payload["error"])


def test_close_closes_driver(monkeypatch):
    graph = neo_module.Neo4jOntologyGraph(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password",
        database="neo4j",
        browser_url="http://localhost:7474/browser/",
    )
    fake_driver = _FakeDriver()

    monkeypatch.setattr(graph, "_get_driver", lambda: fake_driver)
    assert graph._get_driver() is fake_driver
    graph._driver = fake_driver

    graph.close()

    assert fake_driver.closed is True
    assert graph._driver is None


def test_parse_graph_file_retries_with_xml(monkeypatch, tmp_path):
    graph = neo_module.Neo4jOntologyGraph(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password",
        database="neo4j",
        browser_url="http://localhost:7474/browser/",
    )
    owl = tmp_path / "legal.owl"
    owl.write_text(_sample_owl_text(), encoding="utf-8")

    parse_calls: list[str] = []
    original_parse = neo_module.Graph.parse

    def fake_parse(self, source, format=None):
        parse_calls.append(f"{Path(str(source)).name}|{format or 'auto'}")
        if format is None:
            raise RuntimeError("initial parse failed")
        return original_parse(self, source, format=format)

    monkeypatch.setattr(neo_module.Graph, "parse", fake_parse)

    parsed = graph._parse_graph_file(owl)

    assert len(parsed) == 3
    assert parse_calls[0].endswith("|auto")
    assert parse_calls[1].endswith("|xml")


def test_load_owl_files_validates_empty_input():
    graph = neo_module.Neo4jOntologyGraph(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password",
        database="neo4j",
        browser_url="http://localhost:7474/browser/",
    )

    with pytest.raises(ValueError, match="No ontology files"):
        graph.load_owl_files([], clear_existing=False, batch_size=500)


def test_load_owl_files_imports_triples(monkeypatch, tmp_path):
    graph = neo_module.Neo4jOntologyGraph(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password",
        database="neo4j",
        browser_url="http://localhost:7474/browser/",
    )
    owl = tmp_path / "legal.owl"
    owl.write_text(_sample_owl_text(), encoding="utf-8")

    fake_driver = _FakeDriver()
    monkeypatch.setattr(graph, "_get_driver", lambda: fake_driver)

    payload = graph.load_owl_files([owl], clear_existing=True, batch_size=1)

    assert payload["loaded_files"] == 1
    assert payload["triples"] == 3
    assert payload["resource_relationships"] == 2
    assert payload["literal_relationships"] == 1
    assert payload["cleared"] is True
    assert payload["matched_files"] == [str(owl)]

    queries = [call[0] for call in fake_driver.session_obj.calls]
    assert any("CREATE CONSTRAINT resource_iri_unique" in query for query in queries)
    assert any("CREATE CONSTRAINT literal_key_unique" in query for query in queries)
    assert any("MATCH (n) DETACH DELETE n" in query for query in queries)
    assert any("MERGE (s)-[r:RELATES_TO" in query for query in queries)
    assert any("MERGE (s)-[r:HAS_LITERAL" in query for query in queries)


def test_load_owl_files_skips_invalid_triples_and_uses_flush_guard(monkeypatch, tmp_path):
    graph = neo_module.Neo4jOntologyGraph(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password",
        database="neo4j",
        browser_url="http://localhost:7474/browser/",
    )
    owl = tmp_path / "invalid.owl"
    owl.write_text("x", encoding="utf-8")

    invalid_predicate = BNode("predicate")
    valid_subject = URIRef("http://example.org/S")
    valid_predicate = URIRef("http://example.org/p")
    valid_object = URIRef("http://example.org/O")
    triples = [
        (valid_subject, invalid_predicate, valid_object),  # skipped: predicate is not URIRef
        ("not-a-subject", valid_predicate, valid_object),  # skipped: subject is invalid type
        (valid_subject, valid_predicate, valid_object),  # imported
    ]
    monkeypatch.setattr(graph, "_parse_graph_file", lambda _path: triples)

    fake_driver = _FakeDriver()
    monkeypatch.setattr(graph, "_get_driver", lambda: fake_driver)

    payload = graph.load_owl_files([owl], clear_existing=False, batch_size=500)

    assert payload["triples"] == 1
    assert payload["resource_relationships"] == 1
    assert payload["literal_relationships"] == 0


def test_clear_graph_executes_delete(monkeypatch):
    graph = neo_module.Neo4jOntologyGraph(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password",
        database="neo4j",
        browser_url="http://localhost:7474/browser/",
    )
    fake_driver = _FakeDriver()
    monkeypatch.setattr(graph, "_get_driver", lambda: fake_driver)

    graph.clear_graph()

    assert fake_driver.session_obj.calls[0][0] == "MATCH (n) DETACH DELETE n"


def test_search_terms_removes_stopwords_and_deduplicates():
    graph = neo_module.Neo4jOntologyGraph(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password",
        database="neo4j",
        browser_url="http://localhost:7474/browser/",
    )

    terms = graph._search_terms("What does the legal concept and legal concept show about contract breach?")

    assert "what" not in terms
    assert "and" not in terms
    assert terms.count("legal") == 1
    assert "contract" in terms
    assert "breach" in terms


def test_search_terms_caps_result_count_at_eight():
    graph = neo_module.Neo4jOntologyGraph(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password",
        database="neo4j",
        browser_url="http://localhost:7474/browser/",
    )

    terms = graph._search_terms("one two three four five six seven eight nine ten eleven")

    assert len(terms) == 8


def test_search_terms_fallbacks_for_short_input():
    graph = neo_module.Neo4jOntologyGraph(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password",
        database="neo4j",
        browser_url="http://localhost:7474/browser/",
    )

    assert graph._search_terms("  ") == []
    assert graph._search_terms("x") == ["x"]


def test_retrieve_context_validates_question():
    graph = neo_module.Neo4jOntologyGraph(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password",
        database="neo4j",
        browser_url="http://localhost:7474/browser/",
    )

    with pytest.raises(ValueError, match="Question is required"):
        graph.retrieve_context("   ")


def test_retrieve_context_raises_when_search_terms_are_empty(monkeypatch):
    graph = neo_module.Neo4jOntologyGraph(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password",
        database="neo4j",
        browser_url="http://localhost:7474/browser/",
    )
    monkeypatch.setattr(graph, "_search_terms", lambda _question: [])

    with pytest.raises(ValueError, match="did not produce retrieval terms"):
        graph.retrieve_context("contract")


def test_retrieve_context_builds_context_text(monkeypatch):
    graph = neo_module.Neo4jOntologyGraph(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password",
        database="neo4j",
        browser_url="http://localhost:7474/browser/",
    )

    class _RetrievalSession:
        def __init__(self):
            self.params = {}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def run(self, _query, **params):
            self.params = params
            return [
                {
                    "iri": "http://example.org/Contract",
                    "label": "Contract",
                    "relations": [
                        {
                            "predicate": "relatedTo",
                            "object_label": "Breach",
                            "object_iri": "http://example.org/Breach",
                        }
                    ],
                    "literals": [
                        {
                            "predicate": "label",
                            "value": "Contract",
                            "datatype": "",
                            "lang": "en",
                        }
                    ],
                },
                {
                    "iri": "",
                    "label": "invalid",
                    "relations": [],
                    "literals": [],
                },
            ]

    class _RetrievalDriver:
        def __init__(self):
            self.session_obj = _RetrievalSession()

        def session(self, database=None):
            assert database == "neo4j"
            return self.session_obj

    driver = _RetrievalDriver()
    monkeypatch.setattr(graph, "_get_driver", lambda: driver)

    payload = graph.retrieve_context("contract breach", node_limit=3)

    assert payload["resource_count"] == 1
    assert payload["resources"][0]["label"] == "Contract"
    assert "RELATES_TO relatedTo -> Breach" in payload["context_text"]
    assert "label: Contract" in payload["context_text"]
    assert driver.session_obj.params["node_limit"] == 3
    assert driver.session_obj.params["terms"] == ["contract", "breach"]


def test_retrieve_context_handles_no_rows(monkeypatch):
    graph = neo_module.Neo4jOntologyGraph(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password",
        database="neo4j",
        browser_url="http://localhost:7474/browser/",
    )

    class _EmptySession:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def run(self, _query, **_params):
            return []

    class _EmptyDriver:
        def session(self, database=None):
            assert database == "neo4j"
            return _EmptySession()

    monkeypatch.setattr(graph, "_get_driver", lambda: _EmptyDriver())

    payload = graph.retrieve_context("contract")

    assert payload["resource_count"] == 0
    assert payload["context_text"] == "No matching ontology context found."
