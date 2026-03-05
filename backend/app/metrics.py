# Copyright (c) 2026 Data-Blitz Inc. All rights reserved.
# License: Proprietary. See NOTICE.md.
# Author: Paul Harvener.

from __future__ import annotations

"""Prometheus metrics for backend runtime observability.

Prometheus is used here for numeric telemetry, not raw log storage. The
application keeps regular Python logs via ``logging`` and also exports a
metrics surface that Prometheus can scrape and Grafana can visualize.
"""

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest


REGISTRY = CollectorRegistry(auto_describe=True)

HTTP_REQUESTS_TOTAL = Counter(
    "deposition_http_requests_total",
    "Total HTTP requests handled by the API.",
    ["method", "path", "status"],
    registry=REGISTRY,
)
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "deposition_http_request_duration_seconds",
    "HTTP request latency in seconds.",
    ["method", "path"],
    registry=REGISTRY,
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
HTTP_INFLIGHT_REQUESTS = Gauge(
    "deposition_http_inflight_requests",
    "Current number of in-flight HTTP requests.",
    ["method", "path"],
    registry=REGISTRY,
)
APP_LOG_EVENTS_TOTAL = Counter(
    "deposition_app_log_events_total",
    "Application log events emitted through the Python logging pipeline.",
    ["level"],
    registry=REGISTRY,
)
ADMIN_TEST_RUNS_TOTAL = Counter(
    "deposition_admin_test_runs_total",
    "Admin-triggered pytest runs.",
    ["result"],
    registry=REGISTRY,
)
CHAT_OPERATIONS_TOTAL = Counter(
    "deposition_llm_operations_total",
    "LLM-backed operations handled by the API.",
    ["operation", "result", "provider"],
    registry=REGISTRY,
)
ADMIN_USER_MUTATIONS_TOTAL = Counter(
    "deposition_admin_user_mutations_total",
    "Admin user create, update, and delete operations.",
    ["action"],
    registry=REGISTRY,
)
ADMIN_PERSONA_MUTATIONS_TOTAL = Counter(
    "deposition_admin_persona_mutations_total",
    "Admin persona create and update operations.",
    ["action"],
    registry=REGISTRY,
)
CASE_MEMORY_WRITES_TOTAL = Counter(
    "deposition_case_memory_writes_total",
    "Case memory events written to CouchDB.",
    ["channel"],
    registry=REGISTRY,
)
CASE_DOC_UPSERTS_TOTAL = Counter(
    "deposition_case_doc_upserts_total",
    "Case metadata upserts written to CouchDB.",
    registry=REGISTRY,
)
THOUGHT_STREAM_EVENTS_TOTAL = Counter(
    "deposition_thought_stream_events_total",
    "Thought stream events emitted by persona and phase.",
    ["persona", "phase"],
    registry=REGISTRY,
)
THOUGHT_STREAM_SESSIONS_TOTAL = Counter(
    "deposition_thought_stream_sessions_total",
    "Thought stream session status transitions.",
    ["status"],
    registry=REGISTRY,
)
THOUGHT_STREAM_PERSISTED_EVENTS = Gauge(
    "deposition_thought_stream_persisted_events",
    "Current count of persisted thought stream events grouped by persona and phase.",
    ["persona", "phase"],
    registry=REGISTRY,
)
THOUGHT_STREAM_PERSISTED_SESSIONS = Gauge(
    "deposition_thought_stream_persisted_sessions",
    "Current count of persisted thought stream sessions grouped by status.",
    ["status"],
    registry=REGISTRY,
)


def record_http_request(method: str, path: str, status: int | str, duration_seconds: float) -> None:
    """Record one completed HTTP request."""

    method_label = str(method or "UNKNOWN").upper()
    path_label = str(path or "unknown")
    status_label = str(status or "500")
    HTTP_REQUESTS_TOTAL.labels(method=method_label, path=path_label, status=status_label).inc()
    HTTP_REQUEST_DURATION_SECONDS.labels(method=method_label, path=path_label).observe(max(duration_seconds, 0.0))


def track_inflight_request(method: str, path: str):
    """Return the labeled in-flight gauge for one route."""

    method_label = str(method or "UNKNOWN").upper()
    path_label = str(path or "unknown")
    return HTTP_INFLIGHT_REQUESTS.labels(method=method_label, path=path_label)


def record_log_event(level: str) -> None:
    """Increment the log counter for one emitted log record."""

    APP_LOG_EVENTS_TOTAL.labels(level=str(level or "INFO").upper()).inc()


def record_admin_test_run(success: bool) -> None:
    """Increment the admin test-run counter."""

    ADMIN_TEST_RUNS_TOTAL.labels(result="success" if success else "failure").inc()


def record_llm_operation(operation: str, success: bool, provider: str | None) -> None:
    """Increment one LLM operation counter."""

    CHAT_OPERATIONS_TOTAL.labels(
        operation=str(operation or "unknown"),
        result="success" if success else "failure",
        provider=str(provider or "unknown"),
    ).inc()


def record_admin_user_mutation(action: str) -> None:
    """Increment one admin user mutation counter."""

    ADMIN_USER_MUTATIONS_TOTAL.labels(action=str(action or "unknown")).inc()


def record_admin_persona_mutation(action: str) -> None:
    """Increment one admin persona mutation counter."""

    ADMIN_PERSONA_MUTATIONS_TOTAL.labels(action=str(action or "unknown")).inc()


def record_case_memory_write(channel: str) -> None:
    """Increment one case-memory write counter."""

    CASE_MEMORY_WRITES_TOTAL.labels(channel=str(channel or "unknown")).inc()


def record_case_doc_upsert() -> None:
    """Increment one case-doc upsert counter."""

    CASE_DOC_UPSERTS_TOTAL.inc()


def record_thought_stream_event(persona: str, phase: str) -> None:
    """Increment one thought-stream event counter."""

    THOUGHT_STREAM_EVENTS_TOTAL.labels(
        persona=str(persona or "unknown"),
        phase=str(phase or "unknown"),
    ).inc()


def record_thought_stream_session(status: str) -> None:
    """Increment one thought-stream session status counter."""

    THOUGHT_STREAM_SESSIONS_TOTAL.labels(status=str(status or "unknown")).inc()


def sync_thought_stream_inventory(
    event_counts: dict[tuple[str, str], int],
    session_counts: dict[str, int],
) -> None:
    """Replace the persisted thought-stream inventory gauges from current storage state."""

    THOUGHT_STREAM_PERSISTED_EVENTS.clear()
    THOUGHT_STREAM_PERSISTED_SESSIONS.clear()

    for (persona, phase), count in event_counts.items():
        THOUGHT_STREAM_PERSISTED_EVENTS.labels(
            persona=str(persona or "unknown"),
            phase=str(phase or "unknown"),
        ).set(max(0, int(count)))

    for status, count in session_counts.items():
        THOUGHT_STREAM_PERSISTED_SESSIONS.labels(status=str(status or "unknown")).set(max(0, int(count)))


def render_metrics_payload() -> bytes:
    """Render the Prometheus exposition payload."""

    return generate_latest(REGISTRY)


__all__ = [
    "CONTENT_TYPE_LATEST",
    "REGISTRY",
    "record_admin_persona_mutation",
    "record_admin_test_run",
    "record_admin_user_mutation",
    "record_case_doc_upsert",
    "record_case_memory_write",
    "record_http_request",
    "record_llm_operation",
    "record_log_event",
    "record_thought_stream_event",
    "record_thought_stream_session",
    "render_metrics_payload",
    "sync_thought_stream_inventory",
    "track_inflight_request",
]
