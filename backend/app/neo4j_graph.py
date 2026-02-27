from __future__ import annotations

"""Neo4j graph loader for OWL/RDF legal ontology files.

This module provides:
- lightweight Neo4j connectivity checks
- batched ontology triple import from OWL files
- stable resource/literal graph projection for Graph RAG use
"""

from collections.abc import Iterable
import hashlib
from pathlib import Path
import re
from typing import Any

from rdflib import BNode, Graph, Literal, URIRef

try:
    from neo4j import GraphDatabase
except Exception:  # pragma: no cover - exercised via runtime misconfiguration
    GraphDatabase = None


_RESOURCE_CONSTRAINT = """
CREATE CONSTRAINT resource_iri_unique IF NOT EXISTS
FOR (n:Resource)
REQUIRE n.iri IS UNIQUE
""".strip()

_LITERAL_CONSTRAINT = """
CREATE CONSTRAINT literal_key_unique IF NOT EXISTS
FOR (n:Literal)
REQUIRE n.literal_key IS UNIQUE
""".strip()

_MERGE_RESOURCES = """
UNWIND $rows AS row
MERGE (n:Resource {iri: row.iri})
ON CREATE SET
  n.local_name = row.local_name,
  n.namespace = row.namespace,
  n.kind = row.kind
ON MATCH SET
  n.local_name = coalesce(n.local_name, row.local_name),
  n.namespace = coalesce(n.namespace, row.namespace),
  n.kind = coalesce(n.kind, row.kind)
""".strip()

_MERGE_RESOURCE_RELATIONSHIPS = """
UNWIND $rows AS row
MERGE (s:Resource {iri: row.subject_iri})
MERGE (o:Resource {iri: row.object_iri})
MERGE (s)-[r:RELATES_TO {
  predicate_iri: row.predicate_iri,
  object_iri: row.object_iri,
  source_file: row.source_file
}]->(o)
ON CREATE SET
  r.predicate_local_name = row.predicate_local_name,
  r.predicate_namespace = row.predicate_namespace
ON MATCH SET
  r.predicate_local_name = coalesce(r.predicate_local_name, row.predicate_local_name),
  r.predicate_namespace = coalesce(r.predicate_namespace, row.predicate_namespace)
""".strip()

_MERGE_LITERAL_RELATIONSHIPS = """
UNWIND $rows AS row
MERGE (s:Resource {iri: row.subject_iri})
MERGE (l:Literal {literal_key: row.literal_key})
ON CREATE SET
  l.value = row.value,
  l.datatype = row.datatype,
  l.lang = row.lang
ON MATCH SET
  l.value = coalesce(l.value, row.value),
  l.datatype = coalesce(l.datatype, row.datatype),
  l.lang = coalesce(l.lang, row.lang)
MERGE (s)-[r:HAS_LITERAL {
  predicate_iri: row.predicate_iri,
  literal_key: row.literal_key,
  source_file: row.source_file
}]->(l)
ON CREATE SET
  r.predicate_local_name = row.predicate_local_name,
  r.predicate_namespace = row.predicate_namespace
ON MATCH SET
  r.predicate_local_name = coalesce(r.predicate_local_name, row.predicate_local_name),
  r.predicate_namespace = coalesce(r.predicate_namespace, row.predicate_namespace)
""".strip()

_GRAPH_RAG_RETRIEVAL = """
CALL () {
  WITH $terms AS terms
  MATCH (n:Resource)
  WHERE any(term IN terms WHERE toLower(coalesce(n.local_name, '')) CONTAINS term OR toLower(n.iri) CONTAINS term)
  RETURN DISTINCT n
  UNION
  WITH $terms AS terms
  MATCH (s:Resource)-[:HAS_LITERAL]->(l:Literal)
  WHERE any(term IN terms WHERE toLower(coalesce(l.value, '')) CONTAINS term)
  RETURN DISTINCT s AS n
}
WITH DISTINCT n LIMIT $node_limit
OPTIONAL MATCH (n)-[rr:RELATES_TO]->(o:Resource)
WITH n, collect(DISTINCT {
  predicate: coalesce(rr.predicate_local_name, rr.predicate_iri),
  object_label: coalesce(o.local_name, o.iri),
  object_iri: o.iri
})[..$neighbor_limit] AS relations
OPTIONAL MATCH (n)-[hl:HAS_LITERAL]->(l:Literal)
RETURN
  n.iri AS iri,
  coalesce(n.local_name, n.iri) AS label,
  relations,
  collect(DISTINCT {
    predicate: coalesce(hl.predicate_local_name, hl.predicate_iri),
    value: l.value,
    datatype: coalesce(l.datatype, ''),
    lang: coalesce(l.lang, '')
  })[..$literal_limit] AS literals
ORDER BY label
""".strip()


def split_iri(value: str) -> tuple[str, str]:
    """Split an IRI into namespace and local-name fragments."""

    text = str(value or "")
    for token in ("#", "/"):
        if token not in text:
            continue
        idx = text.rfind(token)
        if idx <= 0 or idx >= len(text) - 1:
            continue
        return text[: idx + 1], text[idx + 1 :]
    return "", text


def node_to_iri(node: URIRef | BNode) -> str:
    """Normalize RDF URI/BNode into one stable graph identifier."""

    if isinstance(node, URIRef):
        return str(node)
    return f"_bnode:{str(node)}"


def resource_row(iri: str) -> dict[str, str]:
    """Build one :Resource upsert row from an IRI string."""

    namespace, local_name = split_iri(iri)
    return {
        "iri": iri,
        "namespace": namespace,
        "local_name": local_name,
        "kind": "bnode" if iri.startswith("_bnode:") else "uri",
    }


def literal_key(value: str, datatype: str, lang: str) -> str:
    """Create one deterministic key for literal node de-duplication."""

    payload = f"{value}|{datatype}|{lang}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


class Neo4jOntologyGraph:
    """Neo4j graph adapter focused on OWL ontology ingestion."""

    def __init__(
        self,
        *,
        uri: str,
        user: str,
        password: str,
        database: str,
        browser_url: str,
    ) -> None:
        """Initialize one graph adapter with connection settings."""

        self.uri = str(uri or "").strip()
        self.user = str(user or "").strip()
        self.password = str(password or "")
        self.database = str(database or "neo4j").strip() or "neo4j"
        self.browser_url = str(browser_url or "http://localhost:7474/browser/").strip()
        self._driver = None

    @property
    def configured(self) -> bool:
        """Return whether essential Neo4j connection settings are present."""

        return bool(self.uri and self.user and self.password)

    def _get_driver(self):
        """Return a lazily-created Neo4j driver instance."""

        if not self.configured:
            raise RuntimeError("Neo4j is not configured. Set NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD.")
        if GraphDatabase is None:
            raise RuntimeError("neo4j package is not installed.")
        if self._driver is None:
            self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        return self._driver

    def close(self) -> None:
        """Close active Neo4j driver connection when present."""

        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def health(self) -> dict[str, Any]:
        """Check basic Neo4j connectivity and return status payload."""

        if not self.configured:
            return {
                "configured": False,
                "connected": False,
                "error": "Neo4j settings are not configured.",
                "bolt_url": self.uri,
                "database": self.database,
                "browser_url": self.browser_url,
            }

        try:
            driver = self._get_driver()
            with driver.session(database=self.database) as session:
                session.run("RETURN 1").single()
            return {
                "configured": True,
                "connected": True,
                "error": None,
                "bolt_url": self.uri,
                "database": self.database,
                "browser_url": self.browser_url,
            }
        except Exception as exc:
            return {
                "configured": True,
                "connected": False,
                "error": str(exc),
                "bolt_url": self.uri,
                "database": self.database,
                "browser_url": self.browser_url,
            }

    def ensure_constraints(self) -> None:
        """Ensure uniqueness constraints used by ontology graph upserts."""

        driver = self._get_driver()
        with driver.session(database=self.database) as session:
            session.run(_RESOURCE_CONSTRAINT)
            session.run(_LITERAL_CONSTRAINT)

    def clear_graph(self) -> None:
        """Delete all graph nodes/relationships from the selected database."""

        driver = self._get_driver()
        with driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")

    def _parse_graph_file(self, file_path: Path) -> Graph:
        """Parse one ontology file as RDF graph with OWL-friendly fallbacks."""

        graph = Graph()
        try:
            graph.parse(file_path.as_posix())
            return graph
        except Exception:
            graph.parse(file_path.as_posix(), format="xml")
            return graph

    def _flush_batch(
        self,
        session,
        *,
        resource_rows: dict[str, dict[str, str]],
        resource_rels: list[dict[str, str]],
        literal_rels: list[dict[str, str]],
    ) -> None:
        """Flush one prepared batch into Neo4j and clear mutable buffers."""

        if resource_rows:
            session.run(_MERGE_RESOURCES, rows=list(resource_rows.values()))
            resource_rows.clear()

        if resource_rels:
            session.run(_MERGE_RESOURCE_RELATIONSHIPS, rows=resource_rels)
            resource_rels.clear()

        if literal_rels:
            session.run(_MERGE_LITERAL_RELATIONSHIPS, rows=literal_rels)
            literal_rels.clear()

    def load_owl_files(
        self,
        file_paths: Iterable[Path],
        *,
        clear_existing: bool,
        batch_size: int,
    ) -> dict[str, Any]:
        """Load OWL file triples into Neo4j with batched MERGE upserts."""

        paths = [Path(item) for item in file_paths]
        if not paths:
            raise ValueError("No ontology files were provided for import.")

        self.ensure_constraints()
        if clear_existing:
            self.clear_graph()

        triples = 0
        resource_relationships = 0
        literal_relationships = 0
        loaded_files = 0
        matched_files = [str(path) for path in paths]

        driver = self._get_driver()
        with driver.session(database=self.database) as session:
            resource_rows: dict[str, dict[str, str]] = {}
            resource_rels: list[dict[str, str]] = []
            literal_rels: list[dict[str, str]] = []

            def maybe_flush(force: bool = False) -> None:
                row_count = len(resource_rels) + len(literal_rels)
                if not force and row_count < batch_size:
                    return
                self._flush_batch(
                    session,
                    resource_rows=resource_rows,
                    resource_rels=resource_rels,
                    literal_rels=literal_rels,
                )

            for path in paths:
                graph = self._parse_graph_file(path)
                loaded_files += 1
                source_file = path.name

                for subject, predicate, obj in graph:
                    if not isinstance(predicate, URIRef):
                        continue
                    if not isinstance(subject, (URIRef, BNode)):
                        continue

                    triples += 1
                    subject_iri = node_to_iri(subject)
                    resource_rows.setdefault(subject_iri, resource_row(subject_iri))

                    predicate_iri = str(predicate)
                    predicate_namespace, predicate_local_name = split_iri(predicate_iri)

                    if isinstance(obj, (URIRef, BNode)):
                        object_iri = node_to_iri(obj)
                        resource_rows.setdefault(object_iri, resource_row(object_iri))
                        resource_rels.append(
                            {
                                "subject_iri": subject_iri,
                                "object_iri": object_iri,
                                "predicate_iri": predicate_iri,
                                "predicate_namespace": predicate_namespace,
                                "predicate_local_name": predicate_local_name,
                                "source_file": source_file,
                            }
                        )
                        resource_relationships += 1
                    elif isinstance(obj, Literal):
                        literal_value = str(obj)
                        datatype = str(obj.datatype or "")
                        lang = str(obj.language or "")
                        literal_rels.append(
                            {
                                "subject_iri": subject_iri,
                                "predicate_iri": predicate_iri,
                                "predicate_namespace": predicate_namespace,
                                "predicate_local_name": predicate_local_name,
                                "literal_key": literal_key(literal_value, datatype, lang),
                                "value": literal_value,
                                "datatype": datatype,
                                "lang": lang,
                                "source_file": source_file,
                            }
                        )
                        literal_relationships += 1
                    maybe_flush()

            maybe_flush(force=True)

        return {
            "matched_files": matched_files,
            "loaded_files": loaded_files,
            "triples": triples,
            "resource_relationships": resource_relationships,
            "literal_relationships": literal_relationships,
            "cleared": clear_existing,
            "database": self.database,
            "browser_url": self.browser_url,
        }

    def _search_terms(self, question: str) -> list[str]:
        """Extract stable lowercase retrieval terms from a Graph RAG question."""

        raw_terms = re.findall(r"[a-zA-Z0-9_:-]{3,}", str(question or "").lower())
        stopwords = {
            "the",
            "and",
            "for",
            "with",
            "that",
            "this",
            "from",
            "what",
            "when",
            "where",
            "which",
            "about",
            "into",
            "onto",
            "over",
            "under",
            "their",
            "there",
            "have",
            "does",
            "tell",
            "show",
        }
        terms: list[str] = []
        seen: set[str] = set()
        for item in raw_terms:
            if item in stopwords:
                continue
            if item in seen:
                continue
            seen.add(item)
            terms.append(item)
            if len(terms) >= 8:
                break
        if terms:
            return terms

        fallback = str(question or "").strip().lower()
        if fallback:
            return [fallback[:80]]
        return []

    def retrieve_context(self, question: str, *, node_limit: int = 8) -> dict[str, Any]:
        """Retrieve ontology context rows relevant to one natural-language question."""

        text = str(question or "").strip()
        if not text:
            raise ValueError("Question is required for graph retrieval.")

        bounded_limit = max(1, min(int(node_limit or 8), 50))
        terms = self._search_terms(text)
        if not terms:
            raise ValueError("Question did not produce retrieval terms.")

        driver = self._get_driver()
        with driver.session(database=self.database) as session:
            records = session.run(
                _GRAPH_RAG_RETRIEVAL,
                terms=terms,
                node_limit=bounded_limit,
                neighbor_limit=3,
                literal_limit=4,
            )

            resources: list[dict[str, Any]] = []
            for record in records:
                iri = str(record.get("iri") or "").strip()
                label = str(record.get("label") or iri).strip()
                if not iri:
                    continue
                relations = [
                    item
                    for item in (record.get("relations") or [])
                    if isinstance(item, dict) and str(item.get("object_iri") or "").strip()
                ]
                literals = [
                    item
                    for item in (record.get("literals") or [])
                    if isinstance(item, dict) and str(item.get("value") or "").strip()
                ]
                resources.append(
                    {
                        "iri": iri,
                        "label": label,
                        "relations": relations,
                        "literals": literals,
                    }
                )

        lines: list[str] = []
        for item in resources:
            lines.append(f"Resource: {item['label']} ({item['iri']})")
            for rel in item["relations"][:3]:
                predicate = str(rel.get("predicate") or "related_to")
                target_label = str(rel.get("object_label") or rel.get("object_iri") or "")
                target_iri = str(rel.get("object_iri") or "")
                lines.append(f"  - RELATES_TO {predicate} -> {target_label} ({target_iri})")
            for literal in item["literals"][:4]:
                predicate = str(literal.get("predicate") or "value")
                value = str(literal.get("value") or "")
                lines.append(f"  - {predicate}: {value}")

        return {
            "terms": terms,
            "resource_count": len(resources),
            "resources": resources,
            "context_text": "\n".join(lines) if lines else "No matching ontology context found.",
        }
