# Copyright (c) 2026 Data-Blitz Inc. All rights reserved.
# License: Proprietary. See NOTICE.md.
# Author: Paul Harvener.

from __future__ import annotations

"""MCP server exposing deposition-focused CouchDB tools.

The tools are intentionally read-oriented and optimized for analyst workflows.
"""

import os
import sys
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

# Ensure project root is importable when server is executed as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.couchdb import CouchDBClient
from backend.app.logging_config import configure_application_logging, get_logger

mcp = FastMCP("couchdb-deposition-access")
configure_application_logging()
logger = get_logger("mcp.couchdb_server")


def _settings() -> tuple[str, str]:
    """Read CouchDB connection settings from environment."""

    couchdb_url = os.getenv("COUCHDB_URL", "http://admin:password@localhost:5984")
    couchdb_db = os.getenv("COUCHDB_DB", "depositions")
    return couchdb_url, couchdb_db


def _with_client(fn):
    """Execute a function with a managed ``CouchDBClient`` lifecycle."""

    couchdb_url, couchdb_db = _settings()
    client = CouchDBClient(couchdb_url, couchdb_db)
    logger.info("langfuse mcp_client_open server=couchdb_deposition_access db=%s", couchdb_db)
    try:
        client.ensure_db()
        result = fn(client)
        logger.info("langfuse mcp_client_success server=couchdb_deposition_access db=%s", couchdb_db)
        return result
    except Exception:
        logger.exception("langfuse mcp_client_failure server=couchdb_deposition_access db=%s", couchdb_db)
        raise
    finally:
        client.close()
        logger.info("langfuse mcp_client_closed server=couchdb_deposition_access db=%s", couchdb_db)


def _doc_preview(doc: dict[str, Any]) -> dict[str, Any]:
    """Project full deposition docs into lightweight listing payloads."""

    return {
        "_id": doc.get("_id"),
        "case_id": doc.get("case_id"),
        "file_name": doc.get("file_name"),
        "witness_name": doc.get("witness_name"),
        "witness_role": doc.get("witness_role"),
        "deposition_date": doc.get("deposition_date"),
        "contradiction_score": doc.get("contradiction_score", 0),
        "flagged": doc.get("flagged", False),
    }


@mcp.tool()
def list_case_depositions(case_id: str, limit: int = 200) -> dict[str, Any]:
    """List deposition documents in a case, ranked by contradiction score."""

    bounded_limit = max(1, min(limit, 500))
    logger.info(
        "langfuse mcp_request tool=list_case_depositions case_id=%s requested_limit=%s bounded_limit=%s",
        case_id,
        limit,
        bounded_limit,
    )

    def _run(client: CouchDBClient) -> dict[str, Any]:
        docs = client.list_depositions(case_id)
        docs.sort(key=lambda item: item.get("contradiction_score", 0), reverse=True)
        items = [_doc_preview(doc) for doc in docs[:bounded_limit]]
        payload = {"case_id": case_id, "count": len(items), "depositions": items}
        logger.info(
            "langfuse mcp_result tool=list_case_depositions case_id=%s count=%s",
            case_id,
            len(items),
        )
        return payload

    return _with_client(_run)


@mcp.tool()
def get_deposition(deposition_id: str) -> dict[str, Any]:
    """Get full deposition document by id."""

    logger.info("langfuse mcp_request tool=get_deposition deposition_id=%s", deposition_id)

    def _run(client: CouchDBClient) -> dict[str, Any]:
        doc = client.get_doc(deposition_id)
        logger.info(
            "langfuse mcp_result tool=get_deposition deposition_id=%s claim_count=%s flagged=%s",
            deposition_id,
            len(doc.get("claims", []) or []),
            bool(doc.get("flagged")),
        )
        return doc

    return _with_client(_run)


@mcp.tool()
def list_flagged_depositions(case_id: str, min_score: int = 1) -> dict[str, Any]:
    """List flagged/contradictory depositions for a case."""

    bounded_score = max(0, min(min_score, 100))
    logger.info(
        "langfuse mcp_request tool=list_flagged_depositions case_id=%s requested_min_score=%s bounded_min_score=%s",
        case_id,
        min_score,
        bounded_score,
    )

    def _run(client: CouchDBClient) -> dict[str, Any]:
        docs = client.list_depositions(case_id)
        filtered = [
            doc
            for doc in docs
            if bool(doc.get("flagged")) or int(doc.get("contradiction_score", 0)) >= bounded_score
        ]
        filtered.sort(key=lambda item: item.get("contradiction_score", 0), reverse=True)
        payload = {
            "case_id": case_id,
            "count": len(filtered),
            "min_score": bounded_score,
            "depositions": [_doc_preview(doc) for doc in filtered],
        }
        logger.info(
            "langfuse mcp_result tool=list_flagged_depositions case_id=%s count=%s min_score=%s",
            case_id,
            len(filtered),
            bounded_score,
        )
        return payload

    return _with_client(_run)


@mcp.tool()
def search_claims(case_id: str, query: str, limit: int = 20) -> dict[str, Any]:
    """Search extracted claims by text match across witness statements in a case."""

    normalized = query.strip().lower()
    bounded_limit = max(1, min(limit, 100))
    logger.info(
        "langfuse mcp_request tool=search_claims case_id=%s query_chars=%s has_query=%s requested_limit=%s bounded_limit=%s",
        case_id,
        len(query or ""),
        bool(normalized),
        limit,
        bounded_limit,
    )

    def _run(client: CouchDBClient) -> dict[str, Any]:
        docs = client.list_depositions(case_id)
        matches: list[dict[str, Any]] = []
        for doc in docs:
            for claim in doc.get("claims", []):
                haystack = " ".join(
                    [
                        str(claim.get("topic", "")),
                        str(claim.get("statement", "")),
                        str(claim.get("source_quote", "")),
                    ]
                ).lower()
                if normalized and normalized in haystack:
                    matches.append(
                        {
                            "deposition_id": doc.get("_id"),
                            "witness_name": doc.get("witness_name"),
                            "topic": claim.get("topic"),
                            "statement": claim.get("statement"),
                            "source_quote": claim.get("source_quote"),
                        }
                    )
                if len(matches) >= bounded_limit:
                    break
            if len(matches) >= bounded_limit:
                break

        payload = {"case_id": case_id, "query": query, "count": len(matches), "matches": matches}
        logger.info(
            "langfuse mcp_result tool=search_claims case_id=%s count=%s",
            case_id,
            len(matches),
        )
        return payload

    return _with_client(_run)


if __name__ == "__main__":
    """Run the MCP server over stdio transport."""

    mcp.run(transport="stdio")
