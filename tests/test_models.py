from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.app.models import (
    CaseListResponse,
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
    DepositionDocument,
    DeleteCaseResponse,
    DepositionSchema,
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

    assert chat.history == []
    assert chat.llm_provider == "ollama"
    assert chat.llm_model == "llama3.3"
    assert ingest.directory == "/tmp"
    assert ingest.llm_provider == "openai"
    assert ingest.skip_reassess is True
    assert reason.contradiction.topic == "Timeline"
    assert reason.llm_model == "gpt-5.2"


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

    deleted = DeleteCaseResponse(case_id="CASE-001", deleted_docs=3)
    assert deleted.deleted_docs == 3

    saved = SaveCaseRequest(case_id="CASE-002", directory="/data/depositions/default")
    assert saved.case_id == "CASE-002"

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
