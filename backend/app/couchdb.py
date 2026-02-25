from __future__ import annotations

"""Minimal CouchDB data-access client used by API and workflow components."""

import time
from typing import Any

import httpx


class CouchDBClient:
    """Thin wrapper around CouchDB HTTP APIs for deposition documents."""

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

    def list_depositions(self, case_id: str) -> list[dict[str, Any]]:
        """List deposition documents for a case."""

        return self.find({"type": "deposition", "case_id": case_id}, limit=500)
