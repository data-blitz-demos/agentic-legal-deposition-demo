# Copyright (c) 2026 Data-Blitz Inc. All rights reserved.
# License: Proprietary. See NOTICE.md.
# Author: Paul Harvener.

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from backend.app import main
from backend.app.logging_config import (
    APP_LOGGER_NAME,
    _PrometheusLogHandler,
    _resolve_log_level,
    configure_application_logging,
    get_logger,
)


@pytest.fixture
def logging_client(monkeypatch):
    """Provide a lightweight test client with startup dependencies stubbed."""

    monkeypatch.setattr(main, "_ensure_startup_llm_connectivity", lambda: None)
    monkeypatch.setattr(main, "_ensure_request_llm_operational", lambda *_args, **_kwargs: None)

    couchdb = Mock()
    couchdb.ensure_db = Mock()
    couchdb.ensure_deposition_views = Mock()
    couchdb.close = Mock()
    monkeypatch.setattr(main, "couchdb", couchdb)

    for attr in ("memory_couchdb", "trace_couchdb", "rag_couchdb"):
        stub = Mock()
        stub.ensure_db = Mock()
        stub.close = Mock()
        monkeypatch.setattr(main, attr, stub)

    neo4j_graph = Mock()
    neo4j_graph.close = Mock()
    monkeypatch.setattr(main, "neo4j_graph", neo4j_graph)

    with TestClient(main.app) as client:
        yield client


def test_configure_application_logging_is_idempotent():
    configure_application_logging("DEBUG")
    logger = configure_application_logging("INFO")

    root_logger = logging.getLogger()
    tagged_handlers = [
        item for item in root_logger.handlers if getattr(item, "_legal_deposition_demo_handler", False)
    ]
    prometheus_handlers = [
        item
        for item in root_logger.handlers
        if getattr(item, "_legal_deposition_demo_prometheus_handler", False)
    ]
    file_handlers = [
        item for item in root_logger.handlers if getattr(item, "_legal_deposition_demo_file_handler", False)
    ]

    assert logger.name == APP_LOGGER_NAME
    assert len(tagged_handlers) == 1
    assert len(prometheus_handlers) == 1
    assert len(file_handlers) == 1
    assert tagged_handlers[0].level == logging.INFO


def test_configure_application_logging_writes_to_requested_file(tmp_path):
    root_logger = logging.getLogger()
    existing = [
        item for item in root_logger.handlers if getattr(item, "_legal_deposition_demo_file_handler", False)
    ]
    for item in existing:
        root_logger.removeHandler(item)
        item.close()

    log_path = tmp_path / "application.log"
    logger = configure_application_logging("INFO", str(log_path))
    logger.info("file-backed test message")

    for item in root_logger.handlers:
        if getattr(item, "_legal_deposition_demo_file_handler", False):
            item.flush()

    assert log_path.exists()
    assert "file-backed test message" in log_path.read_text(encoding="utf-8")


def test_get_logger_returns_application_scoped_logger():
    logger = get_logger("backend.app.main")

    assert logger.name == f"{APP_LOGGER_NAME}.backend.app.main"


def test_log_level_resolution_and_logger_variants():
    assert _resolve_log_level(15) == 15
    assert _resolve_log_level("") == logging.INFO
    assert _resolve_log_level("not-a-real-level") == logging.INFO
    assert get_logger().name == APP_LOGGER_NAME
    assert get_logger(f"{APP_LOGGER_NAME}.child").name == f"{APP_LOGGER_NAME}.child"


def test_prometheus_log_handler_swallows_metric_recording_errors(monkeypatch):
    handler = _PrometheusLogHandler()
    monkeypatch.setattr(
        "backend.app.logging_config.record_log_event",
        lambda _level: (_ for _ in ()).throw(RuntimeError("metric down")),
    )

    handler.emit(logging.LogRecord("test", logging.INFO, __file__, 1, "hello", (), None))


def test_request_logging_middleware_emits_completion_log(logging_client, caplog):
    with caplog.at_level(logging.INFO):
        response = logging_client.get("/")

    assert response.status_code == 200
    assert any(
        "request complete method=GET path=/ status=200" in record.getMessage()
        for record in caplog.records
    )


def test_request_logging_middleware_emits_failure_log(caplog):
    request = main.Request(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/api/failure",
            "raw_path": b"/api/failure",
            "query_string": b"",
            "headers": [],
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
        }
    )

    async def _raise(_request):
        raise RuntimeError("boom")

    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError, match="boom"):
            asyncio.run(main.log_http_requests(request, _raise))

    assert any(
        "request failed method=GET path=/api/failure" in record.getMessage()
        for record in caplog.records
    )


def test_run_admin_tests_logs_start_and_completion(monkeypatch, caplog):
    monkeypatch.setattr(
        main.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="438 passed", stderr="", returncode=0),
    )
    monkeypatch.setattr(main, "_persist_last_test_run_output", lambda _output: None)

    with caplog.at_level(logging.INFO):
        response = main.run_admin_tests()

    assert response.succeeded is True
    messages = [record.getMessage() for record in caplog.records]
    assert any("admin test run started command=" in item for item in messages)
    assert any("admin test run completed succeeded=True exit_code=0" in item for item in messages)


def test_save_admin_user_logs_create(monkeypatch, caplog):
    couchdb = Mock()
    couchdb.save_doc.return_value = {"_id": "user:1"}
    monkeypatch.setattr(main, "couchdb", couchdb)

    with caplog.at_level(logging.INFO):
        response = main._save_admin_user(None, "Test", "User", None, "admin")

    assert response.user_id == "user:1"
    assert any(
        "admin user created user_id=user:1 authorization_level=admin" in record.getMessage()
        for record in caplog.records
    )
