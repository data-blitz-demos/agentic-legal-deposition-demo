from __future__ import annotations

"""Pydantic domain and API contract models.

These models define:
- deposition extraction structures
- contradiction analysis output
- request/response payloads for FastAPI endpoints
- LLM option metadata returned to the frontend
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Claim(BaseModel):
    """Single factual claim extracted from testimony."""

    topic: str = Field(description="Short topic label for a factual claim")
    statement: str = Field(description="Detailed claim from testimony")
    confidence: float = Field(ge=0.0, le=1.0, description="Model confidence in extraction")
    source_quote: str = Field(description="Direct quote supporting this claim")

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence_scale(cls, value):
        """Accept percent-style confidence values and normalize to 0..1.

        Some local models emit confidence as whole-number percentages
        (for example ``80`` or ``"80%"``). Normalize those values to
        decimal form while keeping strict validation for other out-of-range
        decimal values (for example ``1.2`` remains invalid).
        """

        def _normalize_numeric(number: float):
            if number.is_integer() and 1 < number <= 100:
                return number / 100.0
            return number

        if isinstance(value, str):
            text = value.strip()
            if not text:
                return value
            percent = text.endswith("%")
            candidate = text[:-1].strip() if percent else text
            try:
                numeric = float(candidate)
            except Exception:
                return value
            if percent and 0 <= numeric <= 100:
                return numeric / 100.0
            return _normalize_numeric(numeric)

        if isinstance(value, bool):
            raise ValueError("confidence must be a numeric value between 0 and 1")

        if isinstance(value, (int, float)):
            return _normalize_numeric(float(value))

        return value


class DepositionSchema(BaseModel):
    """Runtime validator for deposition extraction payloads.

    The extraction contract itself is versioned in
    ``backend/schemas/deposition_schema.json`` and passed directly to the LLM.
    """

    case_id: str
    file_name: str
    witness_name: str
    witness_role: str
    deposition_date: str | None = None
    summary: str
    claims: list[Claim] = Field(default_factory=list)


class ContradictionFinding(BaseModel):
    """One contradiction item comparing target vs peer deposition."""

    other_deposition_id: str
    other_witness_name: str
    topic: str
    rationale: str
    severity: int = Field(ge=0, le=100)


class ContradictionAssessment(BaseModel):
    """Aggregate contradiction result for one deposition."""

    contradiction_score: int = Field(ge=0, le=100)
    flagged: bool
    explanation: str
    contradictions: list[ContradictionFinding] = Field(default_factory=list)


class DepositionDocument(BaseModel):
    """Stored CouchDB representation of a deposition record."""

    id: str = Field(alias="_id")
    rev: str | None = Field(default=None, alias="_rev")
    type: Literal["deposition"]
    case_id: str
    file_name: str
    witness_name: str
    witness_role: str
    deposition_date: str | None = None
    summary: str
    claims: list[Claim] = Field(default_factory=list)
    raw_text: str
    contradiction_score: int = Field(ge=0, le=100)
    flagged: bool
    contradiction_explanation: str = ""
    contradictions: list[ContradictionFinding] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class IngestCaseRequest(BaseModel):
    """Request payload for bulk deposition ingestion."""

    case_id: str
    directory: str
    llm_provider: Literal["openai", "ollama"] | None = None
    llm_model: str | None = None
    skip_reassess: bool = False


class IngestedDepositionResult(BaseModel):
    """Result item for a single ingested deposition."""

    deposition_id: str
    file_name: str
    witness_name: str
    contradiction_score: int
    flagged: bool


class IngestCaseResponse(BaseModel):
    """Response payload for the ingest endpoint."""

    case_id: str
    ingested: list[IngestedDepositionResult] = Field(default_factory=list)


class ChatRequest(BaseModel):
    """Request payload for attorney chat interactions."""

    case_id: str
    deposition_id: str
    message: str
    history: list[dict[str, str]] = Field(default_factory=list)
    llm_provider: Literal["openai", "ollama"] | None = None
    llm_model: str | None = None


class ChatResponse(BaseModel):
    """Response payload for attorney chat endpoint."""

    response: str


class ContradictionReasonRequest(BaseModel):
    """Request payload for focused contradiction re-analysis."""

    case_id: str
    deposition_id: str
    contradiction: ContradictionFinding
    llm_provider: Literal["openai", "ollama"] | None = None
    llm_model: str | None = None


class ContradictionReasonResponse(BaseModel):
    """Response payload for focused contradiction reasoning endpoint."""

    response: str


class LLMOption(BaseModel):
    """Selectable LLM option with runtime readiness metadata."""

    provider: Literal["openai", "ollama"]
    model: str
    label: str
    operational: bool = True
    error: str | None = None
    possible_fix: str | None = None


class LLMOptionsResponse(BaseModel):
    """Response payload for listing LLM options to the UI."""

    selected_provider: Literal["openai", "ollama"]
    selected_model: str
    options: list[LLMOption] = Field(default_factory=list)


class DepositionDirectoryOption(BaseModel):
    """One ingestable directory option exposed to the UI."""

    path: str
    label: str
    file_count: int = Field(ge=0)
    source: Literal["mounted", "configured", "repo"]


class DepositionDirectoriesResponse(BaseModel):
    """Response payload for directory-navigation options."""

    base_directory: str | None = None
    suggested: str | None = None
    options: list[DepositionDirectoryOption] = Field(default_factory=list)


class DepositionRootRequest(BaseModel):
    """Request payload for caching an additional deposition root path."""

    path: str


class DepositionRootResponse(BaseModel):
    """Response payload for one cached deposition root path."""

    path: str


class CaseSummary(BaseModel):
    """Case index item shown in the UI case browser."""

    case_id: str
    deposition_count: int = Field(ge=0, default=0)
    memory_entries: int = Field(ge=0, default=0)
    updated_at: str | None = None
    last_action: str | None = None
    last_directory: str | None = None
    last_llm_provider: Literal["openai", "ollama"] | None = None
    last_llm_model: str | None = None


class CaseListResponse(BaseModel):
    """Response payload listing all known cases."""

    cases: list[CaseSummary] = Field(default_factory=list)


class DeleteCaseResponse(BaseModel):
    """Response payload after deleting a case and related docs."""

    case_id: str
    deleted_docs: int = Field(ge=0)


class ClearCaseDepositionsResponse(BaseModel):
    """Response payload after clearing all deposition docs for a case."""

    case_id: str
    deleted_depositions: int = Field(ge=0)


class SaveCaseRequest(BaseModel):
    """Request payload for creating/updating a saved case entry."""

    case_id: str
    directory: str | None = None
    llm_provider: Literal["openai", "ollama"] | None = None
    llm_model: str | None = None


class SaveCaseVersionRequest(BaseModel):
    """Request payload for saving a versioned case snapshot."""

    case_id: str
    source_case_id: str | None = None
    directory: str
    llm_provider: Literal["openai", "ollama"]
    llm_model: str
    snapshot: dict = Field(default_factory=dict)


class CaseVersionSummary(BaseModel):
    """One saved version entry for a case."""

    case_id: str
    version: int = Field(ge=1)
    created_at: str
    directory: str
    llm_provider: Literal["openai", "ollama"]
    llm_model: str
    snapshot: dict = Field(default_factory=dict)


class CaseVersionListResponse(BaseModel):
    """Response payload listing all saved versions for a case."""

    case_id: str
    versions: list[CaseVersionSummary] = Field(default_factory=list)


class RenameCaseRequest(BaseModel):
    """Request payload for renaming a case id across saved records."""

    new_case_id: str


class RenameCaseResponse(BaseModel):
    """Response payload after renaming a case id."""

    old_case_id: str
    new_case_id: str
    moved_docs: int = Field(ge=0)
