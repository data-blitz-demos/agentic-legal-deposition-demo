# Copyright (c) 2026 Data-Blitz Inc. All rights reserved.
# License: Proprietary. See NOTICE.md.
# Author: Paul Harvener.

from __future__ import annotations

from backend.app import metrics


def test_metrics_helpers_render_prometheus_payload():
    inflight = metrics.track_inflight_request("get", "/api/test")
    inflight.inc()
    metrics.record_http_request("get", "/api/test", 200, 0.125)
    metrics.record_log_event("info")
    metrics.record_admin_test_run(True)
    metrics.record_admin_test_run(False)
    metrics.record_llm_operation("chat", True, "openai")
    metrics.record_admin_user_mutation("created")
    metrics.record_admin_persona_mutation("updated")
    metrics.record_case_memory_write("chat")
    metrics.record_case_doc_upsert()
    metrics.record_thought_stream_event("Persona:Legal Clerk", "ingest_start")
    metrics.record_thought_stream_session("completed")
    metrics.sync_thought_stream_inventory(
        {("Persona:Legal Clerk", "ingest_start"): 2},
        {"completed": 1},
    )
    inflight.dec()

    payload = metrics.render_metrics_payload().decode("utf-8")

    assert "deposition_http_requests_total" in payload
    assert 'path="/api/test"' in payload
    assert "deposition_http_request_duration_seconds" in payload
    assert "deposition_http_inflight_requests" in payload
    assert "deposition_app_log_events_total" in payload
    assert "deposition_admin_test_runs_total" in payload
    assert "deposition_llm_operations_total" in payload
    assert "deposition_admin_user_mutations_total" in payload
    assert "deposition_admin_persona_mutations_total" in payload
    assert "deposition_case_memory_writes_total" in payload
    assert "deposition_case_doc_upserts_total" in payload
    assert "deposition_thought_stream_events_total" in payload
    assert 'persona="Persona:Legal Clerk"' in payload
    assert "deposition_thought_stream_sessions_total" in payload
    assert "deposition_thought_stream_persisted_events" in payload
    assert "deposition_thought_stream_persisted_sessions" in payload
