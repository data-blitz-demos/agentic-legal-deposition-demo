from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.app.models import (
    AdminTestLogResponse,
    AdminUserListResponse,
    AdminUserRequest,
    AdminUserResponse,
    AgentRuntimeMetric,
    AgentRuntimeMetricsResponse,
    AgentTraceEvent,
    AgentTracePayload,
    CaseListResponse,
    CaseDetailResponse,
    CaseVersionListResponse,
    CaseVersionSummary,
    RenameCaseRequest,
    RenameCaseResponse,
    CaseSummary,
    ChatRequest,
    Claim,
    ContradictionAssessment,
    ContradictionFinding,
    ContradictionReasonRequest,
    FocusedReasoningSummaryRequest,
    FocusedReasoningSummaryResponse,
    DepositionDocument,
    DeleteCaseResponse,
    DepositionSentimentRequest,
    DepositionSentimentResponse,
    DepositionUploadResponse,
    DepositionSchema,
    GraphBrowserResponse,
    GraphHealthResponse,
    GraphOntologyBrowserEntry,
    GraphOntologyBrowserResponse,
    GraphOntologyLoadRequest,
    GraphOntologyLoadResponse,
    GraphRagMonitor,
    GraphOntologyOption,
    GraphOntologyOptionsResponse,
    GraphRagQueryRequest,
    GraphRagQueryResponse,
    GraphRagSource,
    IngestCaseRequest,
    IngestCaseResponse,
    LLMOption,
    LLMOptionsResponse,
    SaveCaseRequest,
    SaveCaseVersionRequest,
)


def test_claim_confidence_validation():
    with pytest.raises(ValidationError):
        Claim(topic="T", statement="S", confidence=1.2, source_quote="Q")


def test_claim_confidence_normalizes_integer_percent_scale():
    claim = Claim(topic="T", statement="S", confidence=80, source_quote="Q")
    assert claim.confidence == 0.8


def test_claim_confidence_normalizes_percent_string():
    claim = Claim(topic="T", statement="S", confidence="75%", source_quote="Q")
    assert claim.confidence == 0.75


def test_claim_confidence_empty_string_is_invalid():
    with pytest.raises(ValidationError):
        Claim(topic="T", statement="S", confidence="", source_quote="Q")


def test_claim_confidence_non_numeric_string_is_invalid():
    with pytest.raises(ValidationError):
        Claim(topic="T", statement="S", confidence="high", source_quote="Q")


def test_claim_confidence_plain_numeric_string_normalized_when_integer_scale():
    claim = Claim(topic="T", statement="S", confidence="60", source_quote="Q")
    assert claim.confidence == 0.6


def test_claim_confidence_bool_is_invalid():
    with pytest.raises(ValidationError):
        Claim(topic="T", statement="S", confidence=True, source_quote="Q")


def test_claim_confidence_none_is_invalid():
    with pytest.raises(ValidationError):
        Claim(topic="T", statement="S", confidence=None, source_quote="Q")


def test_deposition_schema_defaults_claims():
    schema = DepositionSchema(
        case_id="case-1",
        file_name="file.txt",
        witness_name="Jane",
        witness_role="Manager",
        summary="Summary",
    )

    assert schema.claims == []
    assert schema.deposition_date is None


def test_contradiction_assessment_bounds():
    with pytest.raises(ValidationError):
        ContradictionAssessment(
            contradiction_score=101,
            flagged=True,
            explanation="Too high",
            contradictions=[],
        )


def test_deposition_document_alias_fields():
    doc = DepositionDocument(
        _id="dep:case:file",
        _rev="1-a",
        type="deposition",
        case_id="case",
        file_name="file.txt",
        witness_name="Jane",
        witness_role="Manager",
        summary="Summary",
        claims=[],
        raw_text="raw",
        contradiction_score=0,
        flagged=False,
    )

    dumped = doc.model_dump(by_alias=True)
    assert dumped["_id"] == "dep:case:file"
    assert dumped["_rev"] == "1-a"


def test_ingest_case_response_default_empty():
    payload = IngestCaseResponse(case_id="case-1")
    assert payload.ingested == []
    assert payload.thought_stream is None


def test_agent_trace_payload_and_event_model():
    event = AgentTraceEvent(
        persona="Persona:Attorney",
        phase="chat_response",
        llm_provider="openai",
        llm_model="gpt-5.2",
        notes="ok",
    )
    payload = AgentTracePayload(attorney=[event])
    assert payload.legal_clerk == []
    assert payload.attorney[0].phase == "chat_response"


def test_request_models_roundtrip():
    contradiction = ContradictionFinding(
        other_deposition_id="dep:1",
        other_witness_name="Alan",
        topic="Timeline",
        rationale="Mismatch",
        severity=50,
    )

    chat = ChatRequest(
        case_id="case",
        deposition_id="dep:1",
        message="Hello",
        llm_provider="ollama",
        llm_model="llama3.3",
    )
    ingest = IngestCaseRequest(
        case_id="case",
        directory="/tmp",
        llm_provider="openai",
        llm_model="gpt-5.2",
        skip_reassess=True,
    )
    reason = ContradictionReasonRequest(
        case_id="case",
        deposition_id="dep:1",
        llm_provider="openai",
        llm_model="gpt-5.2",
        contradiction=contradiction,
    )
    focused_summary = FocusedReasoningSummaryRequest(
        case_id="case",
        deposition_id="dep:1",
        reasoning_text="Short answer: Full focused analysis.",
        llm_provider="openai",
        llm_model="gpt-5.2",
    )
    focused_summary_response = FocusedReasoningSummaryResponse(
        summary="Short answer: Condensed focused analysis."
    )
    sentiment_request = DepositionSentimentRequest(case_id="case", deposition_id="dep:1")
    sentiment_response = DepositionSentimentResponse(
        score=-0.5,
        label="negative",
        summary="Overall deposition sentiment is negative.",
        positive_matches=1,
        negative_matches=3,
        word_count=120,
    )

    assert chat.history == []
    assert chat.llm_provider == "ollama"
    assert chat.llm_model == "llama3.3"
    assert chat.thought_stream_id is None
    assert chat.trace_id is None
    assert ingest.directory == "/tmp"
    assert ingest.llm_provider == "openai"
    assert ingest.skip_reassess is True
    assert ingest.thought_stream_id is None
    assert ingest.trace_id is None
    assert reason.contradiction.topic == "Timeline"
    assert reason.llm_model == "gpt-5.2"
    assert focused_summary.reasoning_text == "Short answer: Full focused analysis."
    assert focused_summary_response.summary.startswith("Short answer:")
    assert sentiment_request.deposition_id == "dep:1"
    assert sentiment_response.label == "negative"


def test_llm_options_response_defaults():
    payload = LLMOptionsResponse(
        selected_provider="openai",
        selected_model="gpt-5.2",
        options=[LLMOption(provider="openai", model="gpt-5.2", label="ChatGPT - gpt-5.2")],
    )

    assert payload.options[0].provider == "openai"


def test_case_models_defaults_and_bounds():
    summary = CaseSummary(case_id="CASE-001")
    assert summary.deposition_count == 0
    assert summary.memory_entries == 0

    payload = CaseListResponse(cases=[summary])
    assert payload.cases[0].case_id == "CASE-001"

    detail = CaseDetailResponse(case_id="CASE-001", snapshot={"chat": {"message_count": 2}})
    assert detail.snapshot["chat"]["message_count"] == 2

    deleted = DeleteCaseResponse(case_id="CASE-001", deleted_docs=3)
    assert deleted.deleted_docs == 3

    saved = SaveCaseRequest(
        case_id="CASE-002",
        directory="/data/depositions/default",
        snapshot={"selected_deposition_id": "dep:1"},
    )
    assert saved.case_id == "CASE-002"
    assert saved.snapshot["selected_deposition_id"] == "dep:1"

    rename_request = RenameCaseRequest(new_case_id="CASE-003")
    assert rename_request.new_case_id == "CASE-003"

    rename_response = RenameCaseResponse(
        old_case_id="CASE-001",
        new_case_id="CASE-003",
        moved_docs=5,
    )
    assert rename_response.moved_docs == 5

    save_version = SaveCaseVersionRequest(
        case_id="CASE-001",
        source_case_id="CASE-BASE",
        directory="/data/depositions/default",
        llm_provider="openai",
        llm_model="gpt-5.2",
        snapshot={"status": "ready"},
    )
    assert save_version.snapshot["status"] == "ready"
    assert save_version.source_case_id == "CASE-BASE"

    version_summary = CaseVersionSummary(
        case_id="CASE-001",
        version=1,
        created_at="2026-02-25T00:00:00+00:00",
        directory="/data/depositions/default",
        llm_provider="openai",
        llm_model="gpt-5.2",
        snapshot={"status": "ready"},
    )
    assert version_summary.version == 1

    version_list = CaseVersionListResponse(case_id="CASE-001", versions=[version_summary])
    assert version_list.versions[0].case_id == "CASE-001"

    admin_user_request = AdminUserRequest(name="Paul Harvener")
    assert admin_user_request.name == "Paul Harvener"

    admin_user = AdminUserResponse(
        user_id="admin-user-1",
        name="Paul Harvener",
        created_at="2026-02-28T00:00:00+00:00",
    )
    assert admin_user.user_id == "admin-user-1"

    admin_users = AdminUserListResponse(users=[admin_user])
    assert admin_users.users[0].name == "Paul Harvener"

    admin_test_log = AdminTestLogResponse(
        summary="Parsed 1 recorded test runs from tests.html.",
        log_output="No explicit per-test log output was captured.",
    )
    assert "Parsed 1 recorded test runs" in admin_test_log.summary

    upload_response = DepositionUploadResponse(
        directory="/data/depositions/default",
        saved_files=["/data/depositions/default/new_dep.txt"],
        file_count=1,
    )
    assert upload_response.file_count == 1


def test_agent_runtime_metrics_models():
    metric = AgentRuntimeMetric(
        key="task_success_rate_pct",
        label="Task Success Rate",
        value=97.2,
        display="97.2%",
        unit="%",
        status="good",
        target=">= 95%",
        description="Completed runs divided by all finished runs.",
    )
    payload = AgentRuntimeMetricsResponse(
        generated_at="2026-02-26T00:00:00+00:00",
        lookback_hours=24,
        sampled_runs=18,
        running_runs=2,
        finished_runs=16,
        storage_connected=True,
        rag_storage_connected=True,
        rag_sampled_queries=6,
        rag_paired_comparisons=2,
        metrics=[metric],
        correctness_metrics=[metric],
    )
    assert payload.lookback_hours == 24
    assert payload.metrics[0].status == "good"
    assert payload.correctness_metrics[0].key == "task_success_rate_pct"
    assert payload.rag_sampled_queries == 6
    assert payload.rag_paired_comparisons == 2


def test_graph_rag_models():
    request = GraphOntologyLoadRequest(path="/data/ontology/*.owl", clear_existing=True, batch_size=750)
    assert request.clear_existing is True
    assert request.batch_size == 750

    response = GraphOntologyLoadResponse(
        path="/data/ontology/*.owl",
        matched_files=["/data/ontology/legal.owl"],
        loaded_files=1,
        triples=120,
        resource_relationships=80,
        literal_relationships=40,
        cleared=True,
        database="neo4j",
        browser_url="http://localhost:7474/browser/",
    )
    assert response.loaded_files == 1
    assert response.cleared is True

    browser = GraphBrowserResponse(
        browser_url="http://localhost:7474/browser/",
        bolt_url="bolt://localhost:7687",
        database="neo4j",
        launch_url="http://localhost:7474/browser/?cmd=edit&arg=MATCH",
    )
    assert browser.database == "neo4j"
    assert "cmd=edit" in browser.launch_url

    health = GraphHealthResponse(
        configured=True,
        connected=False,
        bolt_url="bolt://localhost:7687",
        database="neo4j",
        browser_url="http://localhost:7474/browser/",
        error="connection refused",
    )
    assert health.connected is False

    option = GraphOntologyOption(
        path="/data/ontology/legal.owl",
        label="legal.owl",
    )
    options_payload = GraphOntologyOptionsResponse(
        base_directory="/data/ontology",
        suggested="/data/ontology/*.owl",
        options=[option],
    )
    assert options_payload.options[0].path == "/data/ontology/legal.owl"

    browser_entry = GraphOntologyBrowserEntry(
        path="/data/ontology/contracts",
        name="contracts",
        kind="directory",
    )
    browser_payload = GraphOntologyBrowserResponse(
        base_directory="/data/ontology",
        current_directory="/data/ontology",
        parent_directory=None,
        wildcard_path="/data/ontology/*.owl",
        directories=[browser_entry],
        files=[],
    )
    assert browser_payload.directories[0].kind == "directory"

    query_request = GraphRagQueryRequest(
        question="What is contract breach?",
        top_k=12,
        use_rag=False,
        stream_rag=False,
        llm_provider="openai",
        llm_model="gpt-5.2",
    )
    assert query_request.top_k == 12
    assert query_request.use_rag is False
    assert query_request.stream_rag is False

    source = GraphRagSource(
        iri="http://example.org/Contract",
        label="Contract",
    )
    query_response = GraphRagQueryResponse(
        question="What is contract breach?",
        answer="Short answer: ...",
        context_rows=1,
        sources=[source],
        llm_provider="openai",
        llm_model="gpt-5.2",
        monitor=GraphRagMonitor(
            rag_enabled=True,
            rag_stream_enabled=False,
            retrieval_terms=["contract", "breach"],
            retrieved_resources=[
                {
                    "iri": "http://example.org/Contract",
                    "label": "Contract",
                    "relations": [
                        {
                            "predicate": "relatedTo",
                            "object_label": "Offer",
                            "object_iri": "http://example.org/Offer",
                        }
                    ],
                    "literals": [
                        {
                            "predicate": "definition",
                            "value": "Contract is agreement",
                            "datatype": "",
                            "lang": "en",
                        }
                    ],
                }
            ],
            context_preview="Resource: Contract",
            llm_system_prompt="system",
            llm_user_prompt="user",
        ),
    )
    assert query_response.sources[0].label == "Contract"
    assert query_response.monitor is not None
    assert query_response.monitor.rag_enabled is True
    assert query_response.monitor.rag_stream_enabled is False
    assert query_response.monitor.retrieved_resources[0].relations[0].predicate == "relatedTo"
