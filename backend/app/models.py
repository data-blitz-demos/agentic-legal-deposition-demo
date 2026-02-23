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
    """Structured deposition schema produced from raw text."""

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
