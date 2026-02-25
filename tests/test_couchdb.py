from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from backend.app import couchdb as couchdb_module
from backend.app.couchdb import CouchDBClient


class StubResponse:
    def __init__(self, status_code: int = 200, payload: dict | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")


@pytest.fixture
def client(monkeypatch) -> tuple[CouchDBClient, Mock]:
    http_client = Mock()
    monkeypatch.setattr(couchdb_module.httpx, "Client", lambda timeout=30.0: http_client)
    client = CouchDBClient("http://host:5984", "db")
    return client, http_client


def test_init_sets_urls(client):
    couch, _ = client
    assert couch.url == "http://host:5984"
    assert couch.base_db_url == "http://host:5984/db"


def test_close_calls_http_client_close(client):
    couch, http_client = client
    couch.close()
    http_client.close.assert_called_once_with()


def test_ensure_db_accepts_created_and_existing(client, monkeypatch):
    couch, http_client = client
    monkeypatch.setattr(couchdb_module.time, "sleep", lambda *_: None)
    http_client.put.side_effect = [RuntimeError("booting"), StubResponse(412)]

    couch.ensure_db(retries=2, delay_seconds=0)

    assert http_client.put.call_count == 2


def test_ensure_db_raises_last_error(client, monkeypatch):
    couch, http_client = client
    monkeypatch.setattr(couchdb_module.time, "sleep", lambda *_: None)
    http_client.put.side_effect = [RuntimeError("first"), RuntimeError("last")]

    with pytest.raises(RuntimeError, match="last"):
        couch.ensure_db(retries=2, delay_seconds=0)


def test_ensure_db_raises_on_http_error_response(client, monkeypatch):
    couch, http_client = client
    monkeypatch.setattr(couchdb_module.time, "sleep", lambda *_: None)
    http_client.put.return_value = StubResponse(500, {})

    with pytest.raises(RuntimeError, match="http 500"):
        couch.ensure_db(retries=1, delay_seconds=0)


def test_save_doc_creates_new_doc(client):
    couch, http_client = client
    http_client.post.return_value = StubResponse(201, {"id": "dep:1", "rev": "2-b"})
    doc = {"type": "deposition"}

    saved = couch.save_doc(doc)

    assert saved["_id"] == "dep:1"
    assert saved["_rev"] == "2-b"
    http_client.post.assert_called_once()


def test_save_doc_with_id_delegates_to_update(client, monkeypatch):
    couch, _ = client
    expected = {"_id": "dep:1", "_rev": "3-c"}
    update = Mock(return_value=expected)
    monkeypatch.setattr(couch, "update_doc", update)

    result = couch.save_doc({"_id": "dep:1", "x": 1})

    assert result == expected
    update.assert_called_once()


def test_update_doc_requires_id(client):
    couch, _ = client
    with pytest.raises(ValueError, match="requires _id"):
        couch.update_doc({"type": "deposition"})


def test_update_doc_fetches_rev_when_missing(client):
    couch, http_client = client
    http_client.get.return_value = StubResponse(200, {"_rev": "1-old"})
    http_client.put.return_value = StubResponse(201, {"rev": "2-new"})

    updated = couch.update_doc({"_id": "dep:1", "type": "deposition"})

    assert updated["_rev"] == "2-new"
    http_client.get.assert_called_once_with("http://host:5984/db/dep:1")
    http_client.put.assert_called_once()


def test_update_doc_allows_missing_existing_doc(client):
    couch, http_client = client
    http_client.get.return_value = StubResponse(404, {})
    http_client.put.return_value = StubResponse(201, {"rev": "1-a"})

    updated = couch.update_doc({"_id": "dep:new", "type": "deposition"})

    assert updated["_rev"] == "1-a"


def test_update_doc_raises_when_lookup_fails_with_non_404(client):
    couch, http_client = client
    http_client.get.return_value = StubResponse(500, {})

    with pytest.raises(RuntimeError, match="http 500"):
        couch.update_doc({"_id": "dep:new", "type": "deposition"})


def test_get_doc_returns_payload(client):
    couch, http_client = client
    http_client.get.return_value = StubResponse(200, {"_id": "dep:1"})

    assert couch.get_doc("dep:1") == {"_id": "dep:1"}


def test_delete_doc_uses_given_rev(client):
    couch, http_client = client
    http_client.delete.return_value = StubResponse(200, {"ok": True})

    couch.delete_doc("dep:1", rev="2-abc")

    http_client.delete.assert_called_once_with(
        "http://host:5984/db/dep:1",
        params={"rev": "2-abc"},
    )


def test_delete_doc_fetches_rev_when_missing(client):
    couch, http_client = client
    http_client.get.return_value = StubResponse(200, {"_rev": "3-def"})
    http_client.delete.return_value = StubResponse(200, {"ok": True})

    couch.delete_doc("dep:1")

    http_client.get.assert_called_once_with("http://host:5984/db/dep:1")
    http_client.delete.assert_called_once_with(
        "http://host:5984/db/dep:1",
        params={"rev": "3-def"},
    )


def test_delete_doc_returns_when_doc_missing(client):
    couch, http_client = client
    http_client.get.return_value = StubResponse(404, {})

    couch.delete_doc("dep:missing")

    http_client.delete.assert_not_called()


def test_delete_doc_raises_when_lookup_missing_rev(client):
    couch, http_client = client
    http_client.get.return_value = StubResponse(200, {})

    with pytest.raises(ValueError, match="missing _rev"):
        couch.delete_doc("dep:1")


def test_delete_doc_returns_when_delete_reports_not_found(client):
    couch, http_client = client
    http_client.delete.return_value = StubResponse(404, {})

    couch.delete_doc("dep:1", rev="2-abc")

    http_client.delete.assert_called_once_with(
        "http://host:5984/db/dep:1",
        params={"rev": "2-abc"},
    )


def test_delete_doc_requires_id(client):
    couch, _ = client
    with pytest.raises(ValueError, match="required for delete"):
        couch.delete_doc("")


def test_find_uses_selector(client):
    couch, http_client = client
    http_client.post.return_value = StubResponse(200, {"docs": [{"_id": "dep:1"}]})

    docs = couch.find({"type": "deposition"}, limit=10)

    assert docs == [{"_id": "dep:1"}]
    http_client.post.assert_called_once_with(
        "http://host:5984/db/_find",
        json={"selector": {"type": "deposition"}, "limit": 10},
    )


def test_list_depositions_calls_find(client, monkeypatch):
    couch, _ = client
    find = Mock(return_value=[{"_id": "dep:1"}])
    monkeypatch.setattr(couch, "find", find)

    result = couch.list_depositions("case-1")

    assert result == [{"_id": "dep:1"}]
    find.assert_called_once_with({"type": "deposition", "case_id": "case-1"}, limit=500)
