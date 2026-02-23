from __future__ import annotations

"""FastAPI entrypoint and HTTP route handlers for the deposition demo.

This module wires together:
- configuration and service singletons
- startup/shutdown lifecycle hooks
- ingestion, listing, detail, chat, and contradiction reasoning APIs
- strict LLM readiness validation for startup and per-request execution
"""

from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .chat import AttorneyChatService
from .config import get_settings
from .couchdb import CouchDBClient
from .graph import DepositionWorkflow
from .llm import (
    LLMOperationalError,
    ensure_llm_operational,
    get_llm_option_status,
    list_llm_options,
    resolve_llm_selection,
)
from .models import (
    ChatRequest,
    ChatResponse,
    ContradictionReasonRequest,
    ContradictionReasonResponse,
    IngestCaseRequest,
    IngestCaseResponse,
    IngestedDepositionResult,
    LLMOption,
    LLMOptionsResponse,
)

settings = get_settings()
couchdb = CouchDBClient(settings.couchdb_url, settings.couchdb_db)
workflow = DepositionWorkflow(settings, couchdb)
chat_service = AttorneyChatService(settings)
app_root = Path(__file__).resolve().parents[2]
container_deposition_root = Path("/data/depositions")


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Application lifecycle hook.

    Startup:
    - validates default LLM connectivity (fail-fast)
    - ensures CouchDB database exists
    Shutdown:
    - closes CouchDB HTTP client
    """

    _ensure_startup_llm_connectivity()
    couchdb.ensure_db()
    yield
    couchdb.close()


app = FastAPI(title="Legal Deposition Analysis Demo", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_dir = Path(__file__).resolve().parents[2] / "frontend"
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


def _build_ingest_candidates(requested_path: str) -> list[Path]:
    """Build candidate paths for ingestion, including container/host mappings."""

    requested = Path(requested_path).expanduser()
    candidates: list[Path] = []
    seen: set[str] = set()

    def add_candidate(path: Path) -> None:
        candidate = path.expanduser()
        key = str(candidate)
        if key in seen:
            return
        seen.add(key)
        candidates.append(candidate)

    add_candidate(requested)
    if not requested.is_absolute():
        add_candidate(app_root / requested)

    try:
        suffix = requested.relative_to(container_deposition_root)
    except ValueError:
        suffix = None

    if suffix is not None:
        configured = Path(settings.deposition_dir).expanduser()
        if configured.is_absolute():
            add_candidate(configured / suffix)
        else:
            add_candidate(app_root / configured / suffix)
            add_candidate(Path.cwd() / configured / suffix)
        add_candidate(app_root / "sample_depositions" / suffix)

    return candidates


def _path_has_glob(path: Path) -> bool:
    """Return ``True`` when a path string contains basic glob tokens."""

    text = str(path)
    return any(token in text for token in ("*", "?", "[", "]"))


def _collect_txt_files(candidate: Path) -> list[Path]:
    """Collect ``.txt`` files from a file, directory, or glob candidate path."""

    if candidate.exists():
        if candidate.is_dir():
            return sorted(
                [
                    path
                    for path in candidate.iterdir()
                    if path.is_file() and path.suffix.lower() == ".txt"
                ]
            )
        if candidate.is_file() and candidate.suffix.lower() == ".txt":
            return [candidate]
        return []

    if _path_has_glob(candidate):
        matched = [Path(item) for item in glob(str(candidate), recursive=False)]
        return sorted([path for path in matched if path.is_file() and path.suffix.lower() == ".txt"])

    return []


def _resolve_ingest_txt_files(requested_path: str) -> list[Path]:
    """Resolve requested input into unique, sorted deposition text files."""

    candidates = _build_ingest_candidates(requested_path)
    all_matches: list[Path] = []
    seen: set[str] = set()

    for candidate in candidates:
        for file_path in _collect_txt_files(candidate):
            key = str(file_path.resolve()) if file_path.exists() else str(file_path)
            if key in seen:
                continue
            seen.add(key)
            all_matches.append(file_path)

    if all_matches:
        return sorted(all_matches)

    attempted = ", ".join(str(item) for item in candidates)
    raise HTTPException(
        status_code=400,
        detail=(
            f"No .txt deposition files found for input: {requested_path}. "
            f"Tried: {attempted}"
        ),
    )


def _dashboard_score(item: dict) -> float:
    """Normalize contradiction score for stable dashboard sorting."""

    value = item.get("contradiction_score", 0)
    return value if isinstance(value, (int, float)) else 0


def _resolve_request_llm(llm_provider: str | None, llm_model: str | None) -> tuple[str, str]:
    """Resolve and validate requested LLM provider/model for an API call."""

    try:
        return resolve_llm_selection(settings, llm_provider, llm_model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _ensure_request_llm_operational(llm_provider: str, llm_model: str) -> None:
    """Enforce runtime readiness for the selected provider/model."""

    try:
        ensure_llm_operational(settings, llm_provider, llm_model)
    except LLMOperationalError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"{exc} Possible fix: {exc.possible_fix}",
        ) from exc


def _ensure_startup_llm_connectivity() -> None:
    """Fail startup when default configured LLM is not operational."""

    llm_provider, llm_model = _resolve_request_llm(None, None)
    try:
        ensure_llm_operational(settings, llm_provider, llm_model)
    except LLMOperationalError as exc:
        raise RuntimeError(
            "Startup failed: no operational default LLM is available "
            f"for {llm_provider}:{llm_model}. Possible fix: {exc.possible_fix}"
        ) from exc


@app.get("/")
def root() -> FileResponse:
    """Serve the frontend entry HTML."""

    return FileResponse(frontend_dir / "index.html")


@app.get("/api/llm-options", response_model=LLMOptionsResponse)
def get_llm_options(force_probe: bool = False) -> LLMOptionsResponse:
    """Return model dropdown options with per-model readiness metadata."""

    selected_provider, selected_model = _resolve_request_llm(None, None)
    items = list_llm_options(settings)

    def _build_option(item: dict[str, str]) -> LLMOption:
        return LLMOption(
            **item,
            **get_llm_option_status(
                settings,
                item["provider"],
                item["model"],
                force_probe=force_probe,
            ),
        )

    if force_probe and len(items) > 1:
        workers = max(1, min(int(settings.llm_options_probe_workers), len(items)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            options = list(executor.map(_build_option, items))
    else:
        options = [_build_option(item) for item in items]

    return LLMOptionsResponse(
        selected_provider=selected_provider,
        selected_model=selected_model,
        options=options,
    )


@app.post("/api/ingest-case", response_model=IngestCaseResponse)
def ingest_case(request: IngestCaseRequest) -> IngestCaseResponse:
    """Ingest all deposition text files for a case and return updated scores."""

    llm_provider, llm_model = _resolve_request_llm(request.llm_provider, request.llm_model)
    _ensure_request_llm_operational(llm_provider, llm_model)
    txt_files = _resolve_ingest_txt_files(request.directory)

    results: list[IngestedDepositionResult] = []
    ingested_ids: list[str] = []
    for file_path in txt_files:
        try:
            state = workflow.run(
                case_id=request.case_id,
                file_path=str(file_path),
                llm_provider=llm_provider,
                llm_model=llm_model,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to process deposition file '{file_path.name}': {exc}",
            ) from exc
        doc = state["deposition_doc"]
        ingested_ids.append(doc["_id"])

    if request.skip_reassess:
        by_id = {}
    else:
        reassessed = workflow.reassess_case(
            request.case_id,
            llm_provider=llm_provider,
            llm_model=llm_model,
        )
        by_id = {doc["_id"]: doc for doc in reassessed}
    for deposition_id in ingested_ids:
        doc = by_id.get(deposition_id) or couchdb.get_doc(deposition_id)
        results.append(
            IngestedDepositionResult(
                deposition_id=doc["_id"],
                file_name=doc["file_name"],
                witness_name=doc["witness_name"],
                contradiction_score=doc["contradiction_score"],
                flagged=doc["flagged"],
            )
        )

    return IngestCaseResponse(case_id=request.case_id, ingested=results)


@app.get("/api/depositions/{case_id}")
def list_depositions(case_id: str) -> list[dict]:
    """List case depositions sorted by descending contradiction score."""

    docs = couchdb.list_depositions(case_id)
    docs.sort(key=_dashboard_score, reverse=True)
    return docs


@app.get("/api/deposition/{deposition_id}")
def get_deposition(deposition_id: str) -> dict:
    """Return full deposition document by id."""

    try:
        return couchdb.get_doc(deposition_id)
    except Exception as exc:  # pragma: no cover - passthrough to API
        raise HTTPException(status_code=404, detail="Deposition not found") from exc


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Generate attorney chat response for a selected deposition context."""

    llm_provider, llm_model = _resolve_request_llm(request.llm_provider, request.llm_model)
    _ensure_request_llm_operational(llm_provider, llm_model)
    try:
        deposition = couchdb.get_doc(request.deposition_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail="Deposition not found") from exc
    if deposition.get("case_id") != request.case_id:
        raise HTTPException(status_code=400, detail="Deposition does not belong to requested case")

    peers = [
        doc
        for doc in couchdb.list_depositions(request.case_id)
        if doc.get("_id") != request.deposition_id
    ]
    try:
        response = chat_service.respond(
            deposition,
            peers,
            request.message,
            request.history,
            llm_provider=llm_provider,
            llm_model=llm_model,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return ChatResponse(response=response)


@app.post("/api/reason-contradiction", response_model=ContradictionReasonResponse)
def reason_contradiction(request: ContradictionReasonRequest) -> ContradictionReasonResponse:
    """Re-analyze a single contradiction item with strict model validation."""

    llm_provider, llm_model = _resolve_request_llm(request.llm_provider, request.llm_model)
    _ensure_request_llm_operational(llm_provider, llm_model)
    try:
        deposition = couchdb.get_doc(request.deposition_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail="Deposition not found") from exc
    if deposition.get("case_id") != request.case_id:
        raise HTTPException(status_code=400, detail="Deposition does not belong to requested case")

    peers = [
        doc
        for doc in couchdb.list_depositions(request.case_id)
        if doc.get("_id") != request.deposition_id
    ]
    try:
        response = chat_service.reason_about_contradiction(
            deposition=deposition,
            peers=peers,
            contradiction=request.contradiction.model_dump(),
            llm_provider=llm_provider,
            llm_model=llm_model,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return ContradictionReasonResponse(response=response)
