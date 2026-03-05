# Copyright (c) 2026 Data-Blitz Inc. All rights reserved.
# License: Proprietary. See NOTICE.md.
# Author: Paul Harvener.

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

AuthorizationLevel = Literal["open", "admin", "expert_user", "user", "read_only"]


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
    ingest_schema_name: str = "deposition_schema"
    ingest_schema_mode: str = "native"
    ingest_schema_payload: dict | None = None

    model_config = {"populate_by_name": True}


class IngestCaseRequest(BaseModel):
    """Request payload for bulk deposition ingestion."""

    case_id: str
    directory: str
    llm_provider: Literal["openai", "ollama"] | None = None
    llm_model: str | None = None
    schema_name: str | None = None
    skip_reassess: bool = False
    thought_stream_id: str | None = None
    trace_id: str | None = None


class IngestedDepositionResult(BaseModel):
    """Result item for a single ingested deposition."""

    deposition_id: str
    file_name: str
    witness_name: str
    contradiction_score: int
    flagged: bool


class IngestSchemaOption(BaseModel):
    """One selectable ingest schema exposed to the UI."""

    key: str
    file_name: str
    mode: str
    builtin: bool = True
    removable: bool = False
    schema_payload: dict | None = Field(default=None, alias="schema")

    model_config = {"populate_by_name": True}


class IngestSchemaSaveRequest(BaseModel):
    """Request payload for creating or updating a custom ingest schema."""

    key: str
    schema_payload: dict = Field(alias="schema")

    model_config = {"populate_by_name": True}


class IngestSchemaDeleteResponse(BaseModel):
    """Response payload for deleting a persisted custom ingest schema."""

    deleted: bool
    key: str


class AgentTraceEvent(BaseModel):
    """One trace event describing visible agent processing steps."""

    persona: Literal["Persona:Legal Clerk", "Persona:Attorney"]
    phase: str
    sequence: int | None = None
    at: str | None = None
    file_name: str | None = None
    llm_provider: Literal["openai", "ollama"] | None = None
    llm_model: str | None = None
    input_preview: str | None = None
    system_prompt: str | None = None
    user_prompt: str | None = None
    output_preview: str | None = None
    notes: str | None = None


class AgentTracePayload(BaseModel):
    """Grouped agent trace payload returned to UI."""

    legal_clerk: list[AgentTraceEvent] = Field(default_factory=list)
    attorney: list[AgentTraceEvent] = Field(default_factory=list)


class IngestCaseResponse(BaseModel):
    """Response payload for the ingest endpoint."""

    case_id: str
    ingested: list[IngestedDepositionResult] = Field(default_factory=list)
    thought_stream: AgentTracePayload | None = None


class ChatRequest(BaseModel):
    """Request payload for attorney chat interactions."""

    case_id: str
    deposition_id: str
    message: str
    history: list[dict[str, str]] = Field(default_factory=list)
    llm_provider: Literal["openai", "ollama"] | None = None
    llm_model: str | None = None
    thought_stream_id: str | None = None
    trace_id: str | None = None


class ChatResponse(BaseModel):
    """Response payload for attorney chat endpoint."""

    response: str
    thought_stream: AgentTracePayload | None = None


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


class FocusedReasoningSummaryRequest(BaseModel):
    """Request payload for summarizing focused contradiction reasoning text."""

    case_id: str
    deposition_id: str
    reasoning_text: str = Field(min_length=1)
    llm_provider: Literal["openai", "ollama"] | None = None
    llm_model: str | None = None


class FocusedReasoningSummaryResponse(BaseModel):
    """Response payload for focused contradiction reasoning summary endpoint."""

    summary: str


class DepositionSentimentRequest(BaseModel):
    """Request payload for computing sentiment across one deposition's full text."""

    case_id: str
    deposition_id: str


class TextSentimentRequest(BaseModel):
    """Request payload for scoring sentiment for arbitrary freeform text."""

    text: str = Field(min_length=1)


class DepositionSentimentResponse(BaseModel):
    """Response payload for deposition-wide sentiment analysis."""

    score: float = Field(ge=-1.0, le=1.0)
    label: Literal["positive", "neutral", "negative"]
    summary: str
    positive_matches: int = Field(ge=0)
    negative_matches: int = Field(ge=0)
    word_count: int = Field(ge=0)


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


class GraphOntologyOption(BaseModel):
    """One selectable OWL ontology path for Graph RAG loading."""

    path: str
    label: str


class GraphOntologyOptionsResponse(BaseModel):
    """Response payload for ontology path dropdown options."""

    base_directory: str | None = None
    suggested: str | None = None
    options: list[GraphOntologyOption] = Field(default_factory=list)


class GraphOntologyBrowserEntry(BaseModel):
    """One directory or file row returned by ontology browser API."""

    path: str
    name: str
    kind: Literal["directory", "file"]


class GraphOntologyBrowserResponse(BaseModel):
    """Response payload for server-side ontology file browser."""

    base_directory: str
    current_directory: str
    parent_directory: str | None = None
    wildcard_path: str
    directories: list[GraphOntologyBrowserEntry] = Field(default_factory=list)
    files: list[GraphOntologyBrowserEntry] = Field(default_factory=list)


class DepositionBrowserEntry(BaseModel):
    """One directory or deposition text file row returned by deposition browser API."""

    path: str
    name: str
    kind: Literal["directory", "file"]


class DepositionBrowserResponse(BaseModel):
    """Response payload for the server-side deposition file browser."""

    base_directory: str
    current_directory: str
    parent_directory: str | None = None
    wildcard_path: str
    directories: list[DepositionBrowserEntry] = Field(default_factory=list)
    files: list[DepositionBrowserEntry] = Field(default_factory=list)


class GraphOntologyLoadRequest(BaseModel):
    """Request payload for loading OWL ontology files into Neo4j."""

    path: str
    clear_existing: bool = False
    batch_size: int = Field(default=500, ge=100, le=5000)


class GraphOntologyLoadResponse(BaseModel):
    """Response payload after loading ontology triples into Neo4j."""

    path: str
    matched_files: list[str] = Field(default_factory=list)
    loaded_files: int = Field(ge=0, default=0)
    triples: int = Field(ge=0, default=0)
    resource_relationships: int = Field(ge=0, default=0)
    literal_relationships: int = Field(ge=0, default=0)
    cleared: bool = False
    database: str
    browser_url: str


class GraphBrowserResponse(BaseModel):
    """Response payload exposing graph browser connection details."""

    browser_url: str
    bolt_url: str
    database: str
    launch_url: str = ""


class GrafanaAccessResponse(BaseModel):
    """Response payload exposing Grafana access details for the local UI."""

    url: str
    login_url: str
    dashboard_url: str
    username: str
    password: str


class GraphHealthResponse(BaseModel):
    """Response payload reporting Neo4j graph availability."""

    configured: bool
    connected: bool
    bolt_url: str
    database: str
    browser_url: str
    error: str | None = None


class GraphRagEmbeddingConfigRequest(BaseModel):
    """Request payload for configuring Graph RAG embedding-backed retrieval."""

    enabled: bool = False
    provider: Literal["openai", "ollama"] = "openai"
    model: str = "text-embedding-3-small"
    dimensions: int | None = Field(default=1536, ge=1, le=8192)
    index_name: str = "resource_embeddings"
    node_label: str = "Resource"
    property_name: str = "embedding"


class GraphRagEmbeddingConfigResponse(GraphRagEmbeddingConfigRequest):
    """Response payload describing the active Graph RAG embedding configuration."""

    source: Literal["defaults", "saved"] = "defaults"
    configured: bool = False
    last_saved_at: str | None = None


class GraphRagQueryRequest(BaseModel):
    """Request payload for asking Graph RAG questions over Neo4j ontology data."""

    question: str
    top_k: int = Field(default=8, ge=1, le=50)
    use_rag: bool = True
    stream_rag: bool = True
    embedding_config: GraphRagEmbeddingConfigRequest | None = None
    llm_provider: Literal["openai", "ollama"] | None = None
    llm_model: str | None = None
    thought_stream_id: str | None = None
    trace_id: str | None = None


class GraphRagSource(BaseModel):
    """One source resource node included in Graph RAG answer context."""

    iri: str
    label: str


class GraphRagRelation(BaseModel):
    """One graph relationship attached to a retrieved resource node."""

    predicate: str
    object_label: str
    object_iri: str


class GraphRagLiteral(BaseModel):
    """One literal attribute attached to a retrieved resource node."""

    predicate: str
    value: str
    datatype: str = ""
    lang: str = ""


class GraphRagRetrievedResource(BaseModel):
    """Detailed retrieved resource row used as LLM grounding context."""

    iri: str
    label: str
    relations: list[GraphRagRelation] = Field(default_factory=list)
    literals: list[GraphRagLiteral] = Field(default_factory=list)


class GraphRagMonitor(BaseModel):
    """Telemetry payload showing retrieval-to-LLM handoff per inference cycle."""

    rag_enabled: bool = True
    rag_stream_enabled: bool = True
    retrieval_mode: Literal["keyword", "vector", "keyword_fallback"] = "keyword"
    retrieval_terms: list[str] = Field(default_factory=list)
    query_embedding_used: bool = False
    embedding_enabled: bool = False
    embedding_provider: Literal["openai", "ollama"] | None = None
    embedding_model: str | None = None
    embedding_index_name: str | None = None
    embedding_error: str | None = None
    retrieved_resources: list[GraphRagRetrievedResource] = Field(default_factory=list)
    context_preview: str
    llm_system_prompt: str
    llm_user_prompt: str


class GraphRagQueryResponse(BaseModel):
    """Response payload for Graph RAG ontology question answering."""

    question: str
    answer: str
    context_rows: int = Field(ge=0, default=0)
    sources: list[GraphRagSource] = Field(default_factory=list)
    llm_provider: Literal["openai", "ollama"]
    llm_model: str
    monitor: GraphRagMonitor | None = None


class SaveTraceRequest(BaseModel):
    """Request payload for persisting one in-memory trace session."""

    case_id: str
    channel: Literal["ingest", "chat"] = "ingest"


class SaveTraceResponse(BaseModel):
    """Response payload after saving a thought-stream session."""

    thought_stream_id: str
    case_id: str
    channel: Literal["ingest", "chat"]
    saved: bool = True


class DeleteTraceResponse(BaseModel):
    """Response payload after discarding a thought-stream session."""

    thought_stream_id: str
    deleted: bool = True


class TraceSessionResponse(BaseModel):
    """Response payload for one live thought-stream session."""

    thought_stream_id: str
    status: Literal["running", "completed", "failed"]
    updated_at: str
    thought_stream: AgentTracePayload


class AgentRuntimeMetric(BaseModel):
    """One computed runtime KPI row for dashboard rendering."""

    key: str
    label: str
    value: float
    display: str
    unit: str | None = None
    status: Literal["good", "warn", "bad", "info"] = "info"
    target: str
    description: str
    formula: str | None = None
    detail: str | None = None
    tracking: str | None = None


class AgentRuntimeMetricsResponse(BaseModel):
    """Response payload for the runtime metrics dashboard."""

    generated_at: str
    lookback_hours: int = Field(ge=1, le=168)
    sampled_runs: int = Field(ge=0, default=0)
    running_runs: int = Field(ge=0, default=0)
    finished_runs: int = Field(ge=0, default=0)
    storage_connected: bool = True
    rag_storage_connected: bool = True
    rag_sampled_queries: int = Field(ge=0, default=0)
    rag_paired_comparisons: int = Field(ge=0, default=0)
    metrics: list[AgentRuntimeMetric] = Field(default_factory=list)
    correctness_metrics: list[AgentRuntimeMetric] = Field(default_factory=list)


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


class CaseDetailResponse(CaseSummary):
    """One saved case record including persisted snapshot state."""

    snapshot: dict = Field(default_factory=dict)


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
    snapshot: dict = Field(default_factory=dict)


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


class AdminUserRequest(BaseModel):
    """Request payload for adding one user record with an authorization level."""

    user_id: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    name: str | None = None
    authorization_level: AuthorizationLevel = "user"


class AdminUserResponse(BaseModel):
    """One saved user record."""

    user_id: str
    name: str
    first_name: str
    last_name: str
    authorization_level: AuthorizationLevel
    created_at: str


class AdminUserListResponse(BaseModel):
    """Response payload listing lightweight admin user records."""

    users: list[AdminUserResponse] = Field(default_factory=list)


class AdminUserDeleteResponse(BaseModel):
    """Response payload after permanently deleting one user record."""

    user_id: str
    deleted: bool = True


class AdminPersonaRagBinding(BaseModel):
    """One ordered RAG step attached to a persona, with explicit enabled state."""

    key: str
    enabled: bool = True


class AdminPersonaToolBinding(BaseModel):
    """One ordered MCP tool step attached to a persona, with explicit enabled state."""

    key: str
    enabled: bool = True


class AdminPersonaPromptSections(BaseModel):
    """Structured persona prompts grouped by their runtime role."""

    system: str = ""
    assistant: str = ""
    context: str = ""


class AdminPersonaRequest(BaseModel):
    """Request payload for creating or updating one persona definition."""

    persona_id: str | None = None
    name: str
    llm_provider: Literal["openai", "ollama"]
    llm_model: str
    prompt_template_key: str | None = None
    prompts: str | None = None
    prompt_sections: AdminPersonaPromptSections | None = None
    rag_sequence: list[str | AdminPersonaRagBinding] = Field(default_factory=list)
    tool_sequence: list[str | AdminPersonaToolBinding] = Field(default_factory=list)


class AdminPersonaResponse(BaseModel):
    """One saved persona definition."""

    persona_id: str
    name: str
    llm_provider: Literal["openai", "ollama"]
    llm_model: str
    prompt_template_key: str | None = None
    prompts: str
    prompt_sections: AdminPersonaPromptSections = Field(default_factory=AdminPersonaPromptSections)
    rag_sequence: list[AdminPersonaRagBinding] = Field(default_factory=list)
    tool_sequence: list[AdminPersonaToolBinding] = Field(default_factory=list)
    last_graph_question: str = ""
    last_graph_answer: str = ""
    last_graph_asked_at: str | None = None
    created_at: str


class AdminPersonaGraphSessionRequest(BaseModel):
    """Request payload for persisting the last graph-only question/answer for one persona."""

    question: str
    answer: str


class AdminPersonaListResponse(BaseModel):
    """Response payload listing saved persona definitions."""

    personas: list[AdminPersonaResponse] = Field(default_factory=list)


class AdminPersonaRagOption(BaseModel):
    """One selectable RAG chain step that can be attached to a persona."""

    key: str
    label: str
    description: str
    available: bool = True


class AdminPersonaRagOptionsResponse(BaseModel):
    """Response payload listing current RAG chain steps available to personas."""

    rags: list[AdminPersonaRagOption] = Field(default_factory=list)


class AdminPersonaToolOption(BaseModel):
    """One selectable MCP tool step that can be attached to a persona."""

    key: str
    label: str
    description: str
    available: bool = True


class AdminPersonaToolOptionsResponse(BaseModel):
    """Response payload listing current MCP tool steps available to personas."""

    tools: list[AdminPersonaToolOption] = Field(default_factory=list)


class AdminPersonaPromptTemplate(BaseModel):
    """One built-in runtime prompt template that can seed a persona definition."""

    key: str
    file_name: str
    content: str


class AdminPersonaPromptTemplatesResponse(BaseModel):
    """Response payload listing the current built-in runtime prompt templates."""

    prompts: list[AdminPersonaPromptTemplate] = Field(default_factory=list)


class AdminTestLogResponse(BaseModel):
    """Response payload for summarized test-log output shown in Admin/Test."""

    summary: str
    log_output: str


class AdminTestRunResponse(BaseModel):
    """Response payload after invoking the full pytest suite from the Admin/Test tab."""

    summary: str
    succeeded: bool
    exit_code: int
    output: str
    duration_seconds: float = Field(ge=0)


class DepositionUploadResponse(BaseModel):
    """Response payload after uploading deposition text files into one folder."""

    directory: str
    saved_files: list[str] = Field(default_factory=list)
    root_directory: str | None = None
    copied_to_root_files: list[str] = Field(default_factory=list)
    file_count: int = Field(ge=0)
