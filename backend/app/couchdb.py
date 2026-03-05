# Copyright (c) 2026 Data-Blitz Inc. All rights reserved.
# License: Proprietary. See NOTICE.md.
# Author: Paul Harvener.

from __future__ import annotations

"""Minimal CouchDB data-access client used by API and workflow components."""

import time
import json
from typing import Any

import httpx


class CouchDBClient:
    """Thin wrapper around CouchDB HTTP APIs for deposition documents."""

    _DEPOSITION_DESIGN_DOC_ID = "_design/depositions"
    _DEPOSITION_DESIGN_DOC = {
        "_id": _DEPOSITION_DESIGN_DOC_ID,
        "language": "javascript",
        "views": {
            "by_case": {
                "map": (
                    "function (doc) { "
                    "if (doc.type === 'deposition' && doc.case_id) { "
                    "emit(doc.case_id, null); "
                    "} "
                    "}"
                ),
                "reduce": "_count",
            }
        },
    }

    def __init__(self, url: str, db_name: str) -> None:
        """Initialize a persistent HTTP client and DB base URL."""

        self.url = url.rstrip("/")
        self.db_name = db_name
        self.base_db_url = f"{self.url}/{self.db_name}"
        self._client = httpx.Client(timeout=30.0)

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""

        self._client.close()

    def ensure_db(self, retries: int = 30, delay_seconds: float = 1.0) -> None:
        """Create the target DB if needed, retrying during startup races."""

        last_error: Exception | None = None
        for _ in range(retries):
            try:
                response = self._client.put(self.base_db_url)
                if response.status_code in (201, 202, 412):
                    return
                response.raise_for_status()
            except Exception as exc:  # pragma: no cover - startup resilience
                last_error = exc
                time.sleep(delay_seconds)
                continue
        if last_error is not None:
            raise last_error

    def ensure_deposition_views(self) -> None:
        """Create or refresh the deposition design doc used for case-id indexing."""

        target_url = f"{self.base_db_url}/{self._DEPOSITION_DESIGN_DOC_ID}"
        response = self._client.get(target_url)
        if response.status_code == 404:
            doc = dict(self._DEPOSITION_DESIGN_DOC)
        else:
            response.raise_for_status()
            existing = response.json()
            if (
                existing.get("language") == self._DEPOSITION_DESIGN_DOC["language"]
                and existing.get("views") == self._DEPOSITION_DESIGN_DOC["views"]
            ):
                return
            doc = dict(self._DEPOSITION_DESIGN_DOC)
            if existing.get("_rev"):
                doc["_rev"] = existing["_rev"]

        put_response = self._client.put(target_url, json=doc)
        put_response.raise_for_status()

    def save_doc(self, doc: dict[str, Any]) -> dict[str, Any]:
        """Insert a new document, or update when ``_id`` already exists."""

        if "_id" in doc:
            return self.update_doc(doc)

        response = self._client.post(self.base_db_url, json=doc)
        response.raise_for_status()
        payload = response.json()
        doc["_id"] = payload["id"]
        doc["_rev"] = payload["rev"]
        return doc

    def update_doc(self, doc: dict[str, Any]) -> dict[str, Any]:
        """Upsert a document by ``_id``, hydrating ``_rev`` when needed."""

        doc_id = doc.get("_id")
        if not doc_id:
            raise ValueError("Document requires _id for updates")

        if "_rev" not in doc:
            response = self._client.get(f"{self.base_db_url}/{doc_id}")
            if response.status_code == 200:
                doc["_rev"] = response.json().get("_rev")
            elif response.status_code != 404:
                response.raise_for_status()

        response = self._client.put(f"{self.base_db_url}/{doc_id}", json=doc)
        response.raise_for_status()
        payload = response.json()
        doc["_rev"] = payload["rev"]
        return doc

    def get_doc(self, doc_id: str) -> dict[str, Any]:
        """Fetch a single document by id."""

        response = self._client.get(f"{self.base_db_url}/{doc_id}")
        response.raise_for_status()
        return response.json()

    def delete_doc(self, doc_id: str, rev: str | None = None) -> None:
        """Delete a document by id/rev (fetch rev when omitted)."""

        if not doc_id:
            raise ValueError("Document id is required for delete")

        resolved_rev = rev
        if not resolved_rev:
            response = self._client.get(f"{self.base_db_url}/{doc_id}")
            if response.status_code == 404:
                return
            response.raise_for_status()
            resolved_rev = response.json().get("_rev")
            if not resolved_rev:
                raise ValueError(f"Document {doc_id} is missing _rev and cannot be deleted")

        response = self._client.delete(
            f"{self.base_db_url}/{doc_id}",
            params={"rev": resolved_rev},
        )
        if response.status_code == 404:
            return
        response.raise_for_status()

    def find(self, selector: dict[str, Any], limit: int = 200) -> list[dict[str, Any]]:
        """Execute a Mango ``_find`` query and return matched docs."""

        response = self._client.post(
            f"{self.base_db_url}/_find",
            json={"selector": selector, "limit": limit},
        )
        response.raise_for_status()
        return response.json().get("docs", [])

    def view(
        self,
        design_doc: str,
        view_name: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a CouchDB view and return the raw payload."""

        normalized_params: dict[str, Any] = {}
        for key, value in (params or {}).items():
            if isinstance(value, bool):
                normalized_params[key] = "true" if value else "false"
            elif value is None:
                continue
            else:
                normalized_params[key] = value

        response = self._client.get(
            f"{self.base_db_url}/_design/{design_doc}/_view/{view_name}",
            params=normalized_params,
        )
        response.raise_for_status()
        return response.json()

    def list_depositions(self, case_id: str) -> list[dict[str, Any]]:
        """List deposition documents for a case, using the indexed view when available."""

        try:
            payload = self.view(
                "depositions",
                "by_case",
                params={
                    "key": json.dumps(case_id),
                    "include_docs": True,
                    "reduce": False,
                },
            )
            rows = payload.get("rows", [])
            return [row.get("doc") for row in rows if isinstance(row.get("doc"), dict)]
        except Exception:
            return self.find({"type": "deposition", "case_id": case_id}, limit=500)

    def list_deposition_counts(self) -> dict[str, int]:
        """Return deposition counts grouped by case id from the reduce view."""

        try:
            payload = self.view("depositions", "by_case", params={"group": True})
            counts: dict[str, int] = {}
            for row in payload.get("rows", []):
                case_id = str(row.get("key") or "").strip()
                if not case_id:
                    continue
                counts[case_id] = int(row.get("value", 0) or 0)
            return counts
        except Exception:
            counts: dict[str, int] = {}
            for doc in self.find({"type": "deposition"}, limit=5000):
                case_id = str(doc.get("case_id") or "").strip()
                if not case_id:
                    continue
                counts[case_id] = counts.get(case_id, 0) + 1
            return counts
