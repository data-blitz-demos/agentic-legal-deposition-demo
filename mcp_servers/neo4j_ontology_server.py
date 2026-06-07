# Copyright (c) 2026 Data-Blitz Inc. All rights reserved.
# License: Proprietary. See NOTICE.md.
# Author: Paul Harvener.

from __future__ import annotations

"""MCP server exposing read tools for Neo4j legal ontology access."""

import os
import sys
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

# Ensure project root is importable when server is executed as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.neo4j_graph import Neo4jOntologyGraph

mcp = FastMCP("neo4j-legal-ontology-access")


def _settings() -> dict[str, str]:
    """Read Neo4j graph settings from environment."""

    return {
        "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        "user": os.getenv("NEO4J_USER", "neo4j"),
        "password": os.getenv("NEO4J_PASSWORD", "password"),
        "database": os.getenv("NEO4J_DATABASE", "neo4j"),
        "browser_url": os.getenv("NEO4J_BROWSER_URL", "http://localhost:7474/browser/"),
    }


def _with_graph(fn):
    """Execute a function with a managed ``Neo4jOntologyGraph`` lifecycle."""

    graph = Neo4jOntologyGraph(**_settings())
    try:
        return fn(graph)
    finally:
        graph.close()


@mcp.tool()
def ontology_health() -> dict[str, Any]:
    """Check Neo4j connectivity and ontology graph availability."""

    return _with_graph(lambda graph: graph.health())


@mcp.tool()
def search_ontology_context(question: str, node_limit: int = 8) -> dict[str, Any]:
    """Retrieve ontology-backed context rows for one natural-language question."""

    normalized_question = str(question or "").strip()
    if not normalized_question:
        raise ValueError("question is required")
    bounded_limit = max(1, min(int(node_limit or 8), 50))

    def _run(graph: Neo4jOntologyGraph) -> dict[str, Any]:
        retrieval = graph.retrieve_context(normalized_question, node_limit=bounded_limit)
        return {
            "question": normalized_question,
            "node_limit": bounded_limit,
            **retrieval,
        }

    return _with_graph(_run)


@mcp.tool()
def list_ontology_resources(search: str = "", limit: int = 50) -> dict[str, Any]:
    """List ontology resources, optionally filtered by a label/IRI substring."""

    normalized_search = str(search or "").strip()
    bounded_limit = max(1, min(int(limit or 50), 500))

    def _run(graph: Neo4jOntologyGraph) -> dict[str, Any]:
        resources = graph.list_resources(search=normalized_search, limit=bounded_limit)
        return {
            "search": normalized_search,
            "limit": bounded_limit,
            "count": len(resources),
            "resources": resources,
        }

    return _with_graph(_run)


@mcp.tool()
def get_ontology_resource(iri: str, neighbor_limit: int = 20, literal_limit: int = 20) -> dict[str, Any]:
    """Get one ontology resource by IRI with outgoing relations and literals."""

    normalized_iri = str(iri or "").strip()
    if not normalized_iri:
        raise ValueError("iri is required")

    bounded_neighbor_limit = max(0, min(int(neighbor_limit or 20), 100))
    bounded_literal_limit = max(0, min(int(literal_limit or 20), 100))

    def _run(graph: Neo4jOntologyGraph) -> dict[str, Any]:
        resource = graph.get_resource(
            normalized_iri,
            neighbor_limit=bounded_neighbor_limit,
            literal_limit=bounded_literal_limit,
        )
        if resource is None:
            raise ValueError(f"Ontology resource '{normalized_iri}' was not found.")
        return resource

    return _with_graph(_run)


if __name__ == "__main__":
    """Run the MCP server over stdio transport."""

    mcp.run(transport="stdio")
