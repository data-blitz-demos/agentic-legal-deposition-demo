# Copyright (c) 2026 Data-Blitz Inc. All rights reserved.
# License: Proprietary. See NOTICE.md.
# Author: Paul Harvener.

from __future__ import annotations

"""MCP server for writing/reading thought-stream sessions in CouchDB."""

import hashlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP

# Ensure project root is importable when server is executed as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.couchdb import CouchDBClient
from backend.app.logging_config import configure_application_logging, get_logger

mcp = FastMCP("couchdb-thought-stream-access")
configure_application_logging()
logger = get_logger("mcp.thought_stream_server")


def _settings() -> tuple[str, str]:
    """Read CouchDB URL and thought-stream DB from environment."""

    couchdb_url = os.getenv("COUCHDB_URL", "http://admin:password@localhost:5984")
    thought_stream_db = os.getenv("THOUGHT_STREAM_DB", "thought_stream")
    return couchdb_url, thought_stream_db


def _with_client(fn):
    """Execute a function with a managed ``CouchDBClient`` lifecycle."""

    couchdb_url, thought_stream_db = _settings()
    client = CouchDBClient(couchdb_url, thought_stream_db)
    logger.info("langfuse mcp_client_open server=couchdb_thought_stream_access db=%s", thought_stream_db)
    try:
        client.ensure_db()
        result = fn(client)
        logger.info("langfuse mcp_client_success server=couchdb_thought_stream_access db=%s", thought_stream_db)
        return result
    except Exception:
        logger.exception("langfuse mcp_client_failure server=couchdb_thought_stream_access db=%s", thought_stream_db)
        raise
    finally:
        client.close()
        logger.info("langfuse mcp_client_closed server=couchdb_thought_stream_access db=%s", thought_stream_db)


def _utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _trace_doc_id(trace_id: str) -> str:
    """Build deterministic thought-stream document id from trace id."""

    digest = hashlib.sha1(trace_id.encode("utf-8")).hexdigest()[:24]
    return f"thought_stream:{digest}"


def _normalize_status(status: str) -> Literal["running", "completed", "failed"]:
    """Validate and normalize thought-stream status."""

    normalized = status.strip().lower()
    if normalized not in {"running", "completed", "failed"}:
        raise ValueError("status must be one of: running, completed, failed")
    return normalized  # type: ignore[return-value]


def _max_sequence(trace_payload: dict[str, Any]) -> int:
    """Return max assigned sequence across both trace channels."""

    max_sequence = 0
    for stream_name in ("legal_clerk", "attorney"):
        for item in trace_payload.get(stream_name, []):
            try:
                max_sequence = max(max_sequence, int(item.get("sequence") or 0))
            except Exception:
                continue
    return max_sequence


def _coerce_events(events: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Defensively normalize event list values into dict items."""

    if not events:
        return []
    normalized: list[dict[str, Any]] = []
    for item in events:
        if not isinstance(item, dict):
            raise ValueError("each event must be an object/dict")
        normalized.append(dict(item))
    return normalized


def _load_stream_doc(client: CouchDBClient, trace_id: str) -> dict[str, Any]:
    """Load one thought-stream document or create an empty base payload."""

    doc_id = _trace_doc_id(trace_id)
    try:
        existing = client.get_doc(doc_id)
    except Exception:
        existing = None

    now = _utc_now_iso()
    if not isinstance(existing, dict):
        return {
            "_id": doc_id,
            "type": "thought_stream",
            "trace_id": trace_id,
            "status": "running",
            "created_at": now,
            "updated_at": now,
            "trace": {
                "legal_clerk": [],
                "attorney": [],
            },
        }

    trace_payload = existing.get("trace") if isinstance(existing.get("trace"), dict) else {}
    trace_payload.setdefault("legal_clerk", [])
    trace_payload.setdefault("attorney", [])

    return {
        "_id": existing.get("_id", doc_id),
        "_rev": existing.get("_rev"),
        "type": "thought_stream",
        "trace_id": str(existing.get("trace_id") or trace_id),
        "status": str(existing.get("status") or "running"),
        "created_at": str(existing.get("created_at") or now),
        "updated_at": str(existing.get("updated_at") or now),
        "case_id": existing.get("case_id"),
        "trace": trace_payload,
    }


@mcp.tool()
def thought_stream_health() -> dict[str, Any]:
    """Check thought-stream CouchDB connectivity and report active database."""

    couchdb_url, thought_stream_db = _settings()
    logger.info("langfuse mcp_request tool=thought_stream_health db=%s", thought_stream_db)

    def _run(_client: CouchDBClient) -> dict[str, Any]:
        payload = {
            "connected": True,
            "database": thought_stream_db,
            "couchdb_url": couchdb_url,
        }
        logger.info("langfuse mcp_result tool=thought_stream_health db=%s connected=true", thought_stream_db)
        return payload

    return _with_client(_run)


@mcp.tool()
def append_thought_stream_events(
    trace_id: str,
    legal_clerk: list[dict[str, Any]] | None = None,
    attorney: list[dict[str, Any]] | None = None,
    status: str = "running",
    case_id: str | None = None,
) -> dict[str, Any]:
    """Append legal-clerk/attorney events to one thought-stream document."""

    normalized_trace_id = trace_id.strip()
    if not normalized_trace_id:
        raise ValueError("trace_id is required")

    normalized_status = _normalize_status(status)
    legal_items = _coerce_events(legal_clerk)
    attorney_items = _coerce_events(attorney)
    logger.info(
        "langfuse mcp_request tool=append_thought_stream_events trace_id=%s status=%s case_id_present=%s legal_clerk_events=%s attorney_events=%s",
        normalized_trace_id,
        normalized_status,
        bool(case_id and case_id.strip()),
        len(legal_items),
        len(attorney_items),
    )

    def _run(client: CouchDBClient) -> dict[str, Any]:
        doc = _load_stream_doc(client, normalized_trace_id)
        if case_id is not None and case_id.strip():
            doc["case_id"] = case_id.strip()

        now = _utc_now_iso()
        next_sequence = _max_sequence(doc["trace"]) + 1

        for event in legal_items:
            event["sequence"] = next_sequence
            next_sequence += 1
            event.setdefault("at", now)
            doc["trace"]["legal_clerk"].append(event)

        for event in attorney_items:
            event["sequence"] = next_sequence
            next_sequence += 1
            event.setdefault("at", now)
            doc["trace"]["attorney"].append(event)

        doc["status"] = normalized_status
        doc["updated_at"] = now

        saved = client.update_doc(doc)
        payload = {
            "trace_id": normalized_trace_id,
            "doc_id": saved.get("_id"),
            "status": normalized_status,
            "case_id": saved.get("case_id"),
            "appended": {
                "legal_clerk": len(legal_items),
                "attorney": len(attorney_items),
            },
            "total_events": {
                "legal_clerk": len(doc["trace"]["legal_clerk"]),
                "attorney": len(doc["trace"]["attorney"]),
            },
            "updated_at": saved.get("updated_at", now),
        }
        logger.info(
            "langfuse mcp_result tool=append_thought_stream_events trace_id=%s status=%s total_legal_clerk=%s total_attorney=%s",
            normalized_trace_id,
            normalized_status,
            payload["total_events"]["legal_clerk"],
            payload["total_events"]["attorney"],
        )
        return payload

    return _with_client(_run)


@mcp.tool()
def get_thought_stream(trace_id: str) -> dict[str, Any]:
    """Get one thought-stream document by trace id."""

    normalized_trace_id = trace_id.strip()
    if not normalized_trace_id:
        raise ValueError("trace_id is required")
    logger.info("langfuse mcp_request tool=get_thought_stream trace_id=%s", normalized_trace_id)

    def _run(client: CouchDBClient) -> dict[str, Any]:
        doc = client.get_doc(_trace_doc_id(normalized_trace_id))
        if not isinstance(doc, dict):
            raise ValueError(f"Thought stream '{normalized_trace_id}' was not found")
        trace_payload = doc.get("trace") if isinstance(doc.get("trace"), dict) else {}
        trace_payload.setdefault("legal_clerk", [])
        trace_payload.setdefault("attorney", [])
        payload = {
            "trace_id": str(doc.get("trace_id") or normalized_trace_id),
            "doc_id": str(doc.get("_id") or _trace_doc_id(normalized_trace_id)),
            "status": str(doc.get("status") or "running"),
            "case_id": doc.get("case_id"),
            "created_at": doc.get("created_at"),
            "updated_at": doc.get("updated_at"),
            "trace": trace_payload,
        }
        logger.info(
            "langfuse mcp_result tool=get_thought_stream trace_id=%s status=%s legal_clerk_events=%s attorney_events=%s",
            normalized_trace_id,
            payload["status"],
            len(trace_payload["legal_clerk"]),
            len(trace_payload["attorney"]),
        )
        return payload

    return _with_client(_run)


@mcp.tool()
def list_thought_streams(case_id: str | None = None, limit: int = 100) -> dict[str, Any]:
    """List thought-stream documents, optionally filtered by case id."""

    bounded_limit = max(1, min(limit, 500))
    logger.info(
        "langfuse mcp_request tool=list_thought_streams case_id_present=%s requested_limit=%s bounded_limit=%s",
        bool(case_id and case_id.strip()),
        limit,
        bounded_limit,
    )

    def _run(client: CouchDBClient) -> dict[str, Any]:
        selector: dict[str, Any] = {"type": "thought_stream"}
        if case_id is not None and case_id.strip():
            selector["case_id"] = case_id.strip()

        docs = client.find(selector, limit=bounded_limit)
        docs.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
        items = [
            {
                "trace_id": str(item.get("trace_id") or ""),
                "doc_id": item.get("_id"),
                "status": str(item.get("status") or "running"),
                "case_id": item.get("case_id"),
                "updated_at": item.get("updated_at"),
                "legal_clerk_events": len((item.get("trace") or {}).get("legal_clerk", [])),
                "attorney_events": len((item.get("trace") or {}).get("attorney", [])),
            }
            for item in docs
        ]
        payload = {
            "count": len(items),
            "case_id": case_id,
            "thought_streams": items,
        }
        logger.info(
            "langfuse mcp_result tool=list_thought_streams count=%s case_id_present=%s",
            len(items),
            bool(case_id and case_id.strip()),
        )
        return payload

    return _with_client(_run)


@mcp.tool()
def delete_thought_stream(trace_id: str) -> dict[str, Any]:
    """Delete one thought-stream document by trace id."""

    normalized_trace_id = trace_id.strip()
    if not normalized_trace_id:
        raise ValueError("trace_id is required")
    logger.info("langfuse mcp_request tool=delete_thought_stream trace_id=%s", normalized_trace_id)

    def _run(client: CouchDBClient) -> dict[str, Any]:
        doc_id = _trace_doc_id(normalized_trace_id)
        try:
            existing = client.get_doc(doc_id)
        except Exception:
            payload = {
                "trace_id": normalized_trace_id,
                "doc_id": doc_id,
                "deleted": False,
            }
            logger.info("langfuse mcp_result tool=delete_thought_stream trace_id=%s deleted=false", normalized_trace_id)
            return payload

        client.delete_doc(doc_id, rev=existing.get("_rev") if isinstance(existing, dict) else None)
        payload = {
            "trace_id": normalized_trace_id,
            "doc_id": doc_id,
            "deleted": True,
        }
        logger.info("langfuse mcp_result tool=delete_thought_stream trace_id=%s deleted=true", normalized_trace_id)
        return payload

    return _with_client(_run)


if __name__ == "__main__":
    """Run the MCP server over stdio transport."""

    mcp.run(transport="stdio")
