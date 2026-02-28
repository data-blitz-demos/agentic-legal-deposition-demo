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
from collections import Counter
from datetime import datetime, timedelta, timezone
from glob import glob
import hashlib
from html import unescape
import json
import math
from pathlib import Path
import re
from threading import Lock
from time import monotonic
from typing import Literal
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage, SystemMessage

from .chat import AttorneyChatService
from .config import get_settings
from .couchdb import CouchDBClient
from .graph import DepositionWorkflow
from .llm import (
    LLMOperationalError,
    build_chat_model,
    ensure_llm_operational,
    get_llm_option_status,
    llm_failure_message,
    list_llm_options,
    resolve_llm_selection,
)
from .models import (
    AgentRuntimeMetric,
    AgentRuntimeMetricsResponse,
    AgentTraceEvent,
    AgentTracePayload,
    AdminTestLogResponse,
    AdminUserListResponse,
    AdminUserRequest,
    AdminUserResponse,
    ChatRequest,
    ChatResponse,
    CaseListResponse,
    CaseDetailResponse,
    CaseSummary,
    CaseVersionListResponse,
    CaseVersionSummary,
    ClearCaseDepositionsResponse,
    DeleteCaseResponse,
    DepositionSentimentRequest,
    DepositionSentimentResponse,
    DepositionUploadResponse,
    DepositionDirectoriesResponse,
    DepositionDirectoryOption,
    DepositionRootRequest,
    DepositionRootResponse,
    ContradictionReasonRequest,
    ContradictionReasonResponse,
    FocusedReasoningSummaryRequest,
    FocusedReasoningSummaryResponse,
    IngestCaseRequest,
    IngestCaseResponse,
    IngestedDepositionResult,
    LLMOption,
    LLMOptionsResponse,
    DeleteTraceResponse,
    GraphBrowserResponse,
    GraphOntologyBrowserEntry,
    GraphOntologyBrowserResponse,
    GraphHealthResponse,
    GraphOntologyOption,
    GraphOntologyLoadRequest,
    GraphOntologyLoadResponse,
    GraphOntologyOptionsResponse,
    GraphRagQueryRequest,
    GraphRagQueryResponse,
    GraphRagRetrievedResource,
    GraphRagRelation,
    GraphRagLiteral,
    GraphRagMonitor,
    GraphRagSource,
    RenameCaseRequest,
    RenameCaseResponse,
    SaveTraceRequest,
    SaveTraceResponse,
    SaveCaseRequest,
    SaveCaseVersionRequest,
    TraceSessionResponse,
)
from .neo4j_graph import Neo4jOntologyGraph
from .prompts import render_prompt

settings = get_settings()
couchdb = CouchDBClient(settings.couchdb_url, settings.couchdb_db)
memory_couchdb = CouchDBClient(settings.couchdb_url, settings.memory_db)
trace_couchdb = CouchDBClient(settings.couchdb_url, settings.thought_stream_db)
rag_couchdb = CouchDBClient(settings.couchdb_url, settings.rag_stream_db)
workflow = DepositionWorkflow(settings, couchdb)
chat_service = AttorneyChatService(settings)
neo4j_graph = Neo4jOntologyGraph(
    uri=settings.neo4j_uri,
    user=settings.neo4j_user,
    password=settings.neo4j_password,
    database=settings.neo4j_database,
    browser_url=settings.neo4j_browser_url,
)
app_root = Path(__file__).resolve().parents[2]
container_deposition_root = Path("/data/depositions")
_trace_lock = Lock()
_trace_sessions: dict[str, dict] = {}
_POSITIVE_SENTIMENT_TERMS = {
    "calm",
    "clear",
    "cooperate",
    "cooperative",
    "credible",
    "consistent",
    "confident",
    "helpful",
    "honest",
    "professional",
    "reliable",
    "stable",
    "supportive",
}
_NEGATIVE_SENTIMENT_TERMS = {
    "afraid",
    "angry",
    "conflict",
    "contradiction",
    "damaged",
    "denied",
    "dispute",
    "fear",
    "forced",
    "humiliation",
    "injury",
    "leak",
    "leaked",
    "liar",
    "panic",
    "retaliation",
    "risk",
    "struck",
    "trauma",
    "violent",
}


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
    memory_couchdb.ensure_db()
    trace_couchdb.ensure_db()
    rag_couchdb.ensure_db()
    yield
    couchdb.close()
    memory_couchdb.close()
    trace_couchdb.close()
    rag_couchdb.close()
    neo4j_graph.close()


app = FastAPI(title="Legal Deposition Analysis Demo", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_dir = Path(__file__).resolve().parents[2] / "frontend"
reports_dir = Path(__file__).resolve().parents[2] / "reports"
tests_report_file = reports_dir / "tests.html"
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")
_ADMIN_REPORT_THEME_CSS = """
<style id="admin-report-theme">
  :root {
    color-scheme: dark;
  }

  html {
    background:
      radial-gradient(circle at top right, rgba(249, 115, 22, 0.12), transparent 38%),
      radial-gradient(circle at top left, rgba(56, 189, 248, 0.1), transparent 34%),
      linear-gradient(180deg, #081321 0%, #0d1726 100%);
  }

  body {
    font-family: Manrope, "Segoe UI", sans-serif !important;
    color: #d6e5f4 !important;
    background: transparent !important;
    margin: 0 !important;
    padding: 24px !important;
  }

  h1, h2, h3, h4, h5, h6 {
    color: #fde68a !important;
    font-weight: 700 !important;
  }

  a, a:visited {
    color: #7dd3fc !important;
  }

  .summary,
  #data-container,
  table#results-table {
    background: rgba(8, 19, 33, 0.82) !important;
    border: 1px solid rgba(136, 168, 196, 0.24) !important;
    border-radius: 16px !important;
    box-shadow: 0 12px 30px rgba(4, 11, 20, 0.35) !important;
  }

  .summary {
    padding: 18px !important;
    margin-bottom: 18px !important;
  }

  .summary,
  .summary *,
  .summary span,
  .summary div,
  .summary p,
  .summary label {
    color: #f8fafc !important;
  }

  table#results-table {
    overflow: hidden !important;
    border-collapse: separate !important;
    border-spacing: 0 !important;
  }

  table#results-table th,
  table#results-table td {
    background: rgba(8, 19, 33, 0.82) !important;
    color: #d6e5f4 !important;
    border-color: rgba(136, 168, 196, 0.18) !important;
  }

  table#results-table tr:nth-child(even) td {
    background: rgba(12, 25, 41, 0.88) !important;
  }

  .passed,
  .skipped,
  .xfailed {
    color: #93c5fd !important;
  }

  .failed,
  .error,
  .xpassed {
    color: #fca5a5 !important;
  }

  .log {
    background: rgba(3, 10, 18, 0.8) !important;
    color: #d6e5f4 !important;
    border: 1px solid rgba(136, 168, 196, 0.18) !important;
    border-radius: 12px !important;
    padding: 12px !important;
  }

  .filters {
    padding: 10px 0 !important;
  }
</style>
""".strip()


def _container_suffix_aliases(suffix: Path) -> list[Path]:
    """Return known compatibility aliases for sample deposition subpaths."""

    if not suffix.parts:
        return []
    head, *tail = suffix.parts
    alias_map = {
        "oj_simposon": "oj_simpson",
        "oj_simpson": "oj_simposon",
    }
    mapped_head = alias_map.get(head)
    if mapped_head is None:
        return []
    return [Path(mapped_head, *tail)]


def _container_suffix_variants(suffix: Path) -> list[Path]:
    """Return normalized suffix variants for container-root path mapping."""

    variants: list[Path] = [suffix]
    parts = suffix.parts
    if len(parts) > 1 and parts[0] == "default":
        variants.append(Path(*parts[1:]))
    return variants


def _deposition_root_doc_id(path: str) -> str:
    """Build stable cache-document id for one deposition root path."""

    digest = hashlib.sha1(path.encode("utf-8")).hexdigest()[:24]
    return f"deposition_root:{digest}"


def _env_extra_deposition_roots() -> list[Path]:
    """Resolve additional deposition roots from ``DEPOSITION_EXTRA_DIRS``."""

    raw = str(getattr(settings, "deposition_extra_dirs", "") or "").strip()
    if not raw:
        return []

    roots: list[Path] = []
    for value in [item.strip() for item in raw.split(",") if item.strip()]:
        configured = Path(value).expanduser()
        candidates = [configured] if configured.is_absolute() else [app_root / configured, Path.cwd() / configured]
        roots.extend(candidates)
    return roots


def _cached_extra_deposition_roots() -> list[Path]:
    """Load cached deposition roots persisted by the runtime add-path endpoint."""

    try:
        docs = couchdb.find({"type": "deposition_root"}, limit=2000)
    except Exception:
        return []

    roots: list[Path] = []
    for doc in docs:
        value = str(doc.get("path") or "").strip()
        if not value:
            continue
        candidate = Path(value).expanduser()
        roots.append(candidate if candidate.is_absolute() else Path.cwd() / candidate)
    return roots


def _configured_extra_deposition_roots() -> list[Path]:
    """Return de-duplicated extra roots from env config plus cached runtime paths."""

    seen: set[str] = set()
    roots: list[Path] = []
    for root in [*_env_extra_deposition_roots(), *_cached_extra_deposition_roots()]:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        roots.append(root)
    return roots


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
        extra_roots = _configured_extra_deposition_roots()
        for suffix_variant in _container_suffix_variants(suffix):
            if configured.is_absolute():
                add_candidate(configured / suffix_variant)
            else:
                add_candidate(app_root / configured / suffix_variant)
                add_candidate(Path.cwd() / configured / suffix_variant)
            for extra_root in extra_roots:
                add_candidate(extra_root / suffix_variant)
            add_candidate(app_root / "depositions" / suffix_variant)
            add_candidate(app_root / "depositions/default" / suffix_variant)
            for alias_suffix in _container_suffix_aliases(suffix_variant):
                if configured.is_absolute():
                    add_candidate(configured / alias_suffix)
                else:
                    add_candidate(app_root / configured / alias_suffix)
                    add_candidate(Path.cwd() / configured / alias_suffix)
                for extra_root in extra_roots:
                    add_candidate(extra_root / alias_suffix)
                add_candidate(app_root / "depositions" / alias_suffix)
                add_candidate(app_root / "depositions/default" / alias_suffix)

    return candidates


def _txt_count(path: Path) -> int:
    """Count immediate ``.txt`` files in a directory."""

    if not path.exists() or not path.is_dir():
        return 0
    return len(
        [
            child
            for child in path.iterdir()
            if child.is_file() and child.suffix.lower() == ".txt"
        ]
    )


def _add_directory_option(
    options: list[DepositionDirectoryOption],
    seen: set[str],
    base_path: Path,
    path: Path,
    source: Literal["mounted", "configured", "repo"],
) -> None:
    """Add one UI directory option when the directory contains text files."""

    if not path.exists() or not path.is_dir():
        return
    file_count = _txt_count(path)
    if file_count <= 0:
        return

    path_text = str(path)
    if path_text in seen:
        return
    seen.add(path_text)
    if path == base_path:
        label = f"{path_text} (base, {file_count} files)"
    else:
        try:
            relative = str(path.relative_to(base_path))
        except ValueError:
            relative = path_text
        label = f"{relative} ({file_count} files)"
    options.append(
        DepositionDirectoryOption(
            path=path_text,
            label=label,
            file_count=file_count,
            source=source,
        )
    )


def _configured_deposition_root() -> Path:
    """Resolve configured deposition directory to an absolute host path."""

    configured = Path(settings.deposition_dir).expanduser()
    if configured.is_absolute():
        return configured
    return app_root / configured


def _ingestable_directory_count(root: Path) -> int:
    """Return count of ingestable directories (root + immediate children)."""

    if not root.exists() or not root.is_dir():
        return -1
    count = 1 if _txt_count(root) > 0 else 0
    for child in [item for item in root.iterdir() if item.is_dir()]:
        if _txt_count(child) > 0:
            count += 1
    return count


def _resolve_directory_base() -> tuple[Literal["mounted", "configured", "repo"], Path]:
    """Resolve the single shared base directory for ingestion folder choices."""

    configured = _configured_deposition_root()
    candidates: list[tuple[Literal["mounted", "configured", "repo"], Path]] = [
        ("mounted", container_deposition_root),
        ("configured", configured),
        ("repo", app_root / "depositions"),
    ]
    best: tuple[Literal["mounted", "configured", "repo"], Path] | None = None
    best_score = -1
    for source, root in candidates:
        score = _ingestable_directory_count(root)
        if score > best_score:
            best = (source, root)
            best_score = score
    if best is not None and best_score >= 0:
        return best
    return "configured", configured


def _list_deposition_directories() -> tuple[Path, list[DepositionDirectoryOption]]:
    """Discover valid ingestion directories from configured root paths."""

    options: list[DepositionDirectoryOption] = []
    seen: set[str] = set()
    source, base = _resolve_directory_base()
    roots: list[tuple[Literal["mounted", "configured", "repo"], Path]] = [(source, base)]
    for extra_root in _configured_extra_deposition_roots():
        if str(extra_root) == str(base):
            continue
        roots.append(("configured", extra_root))

    for root_source, root in roots:
        if not root.exists() or not root.is_dir():
            continue
        _add_directory_option(options, seen, root, root, root_source)
        for child in sorted([item for item in root.iterdir() if item.is_dir()]):
            _add_directory_option(options, seen, root, child, root_source)
    return base, options


def _suggest_directory_option(options: list[DepositionDirectoryOption]) -> str | None:
    """Choose a preferred default path for the UI directory navigator."""

    if not options:
        return None
    preferred = [
        str(container_deposition_root / "default"),
        str(container_deposition_root / "oj_simpson"),
        str(container_deposition_root),
        str(app_root / "depositions/default"),
        str(app_root / "depositions/oj_simpson"),
    ]
    by_path = {item.path: item.path for item in options}
    for path in preferred:
        if path in by_path:
            return path
    return options[0].path


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


def _collect_owl_files(candidate: Path) -> list[Path]:
    """Collect ``.owl`` files from a file, directory, or glob candidate path."""

    if candidate.exists():
        if candidate.is_dir():
            return sorted(
                [
                    path
                    for path in candidate.iterdir()
                    if path.is_file() and path.suffix.lower() == ".owl"
                ]
            )
        if candidate.is_file() and candidate.suffix.lower() == ".owl":
            return [candidate]
        return []

    if _path_has_glob(candidate):
        matched = [Path(item) for item in glob(str(candidate), recursive=False)]
        return sorted([path for path in matched if path.is_file() and path.suffix.lower() == ".owl"])

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


def _resolve_upload_directory(requested_path: str) -> Path:
    """Resolve one requested deposition folder into an existing writable directory."""

    normalized = str(requested_path or "").strip()
    if not normalized:
        raise HTTPException(status_code=400, detail="Deposition folder is required")

    for candidate in _build_ingest_candidates(normalized):
        if _path_has_glob(candidate):
            continue
        if candidate.exists() and candidate.is_dir():
            return candidate

    raise HTTPException(
        status_code=400,
        detail=f"Deposition upload target must be an existing directory: {normalized}",
    )


def _sanitize_uploaded_deposition_filename(filename: str, index: int) -> str:
    """Normalize one uploaded deposition file name into a safe ``.txt`` file name."""

    original = Path(str(filename or "").strip()).name
    if not original:
        return f"uploaded_deposition_{index}.txt"

    stem = Path(original).stem
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    if not safe_stem:
        safe_stem = f"uploaded_deposition_{index}"
    return f"{safe_stem}.txt"


def _resolve_ontology_owl_files(requested_path: str) -> list[Path]:
    """Resolve one requested ontology path into unique, sorted ``.owl`` files."""

    requested = Path(requested_path).expanduser()
    candidates: list[Path] = []
    seen_candidates: set[str] = set()

    def add_candidate(path: Path) -> None:
        key = str(path)
        if key in seen_candidates:
            return
        seen_candidates.add(key)
        candidates.append(path)

    add_candidate(requested)
    configured_root = Path(settings.ontology_dir).expanduser()
    if not requested.is_absolute():
        add_candidate(app_root / requested)
        if configured_root:
            add_candidate(configured_root / requested)

    all_matches: list[Path] = []
    seen_files: set[str] = set()
    for candidate in candidates:
        for file_path in _collect_owl_files(candidate):
            key = str(file_path.resolve()) if file_path.exists() else str(file_path)
            if key in seen_files:
                continue
            seen_files.add(key)
            all_matches.append(file_path)

    if all_matches:
        return sorted(all_matches)

    attempted = ", ".join(str(item) for item in candidates)
    raise HTTPException(
        status_code=400,
        detail=(
            f"No .owl ontology files found for input: {requested_path}. "
            f"Tried: {attempted}"
        ),
    )


def _configured_ontology_root() -> Path:
    """Resolve configured ontology directory to an absolute host path."""

    configured = Path(settings.ontology_dir).expanduser()
    if configured.is_absolute():
        return configured
    return app_root / configured


def _list_ontology_owl_options() -> tuple[Path, list[GraphOntologyOption], str]:
    """Discover OWL path options for the ontology dropdown."""

    base = _configured_ontology_root()
    wildcard = str(base / "*.owl")
    options: list[GraphOntologyOption] = [
        GraphOntologyOption(path=wildcard, label=f"All OWL files in {base}")
    ]

    if base.exists() and base.is_dir():
        for file_path in sorted(
            [path for path in base.rglob("*") if path.is_file() and path.suffix.lower() == ".owl"]
        ):
            options.append(
                GraphOntologyOption(path=str(file_path), label=str(file_path.relative_to(base)))
            )

    return base, options, options[0].path


def _resolve_ontology_browser_directory(requested_path: str | None) -> tuple[Path, Path]:
    """Resolve and validate ontology browser directory under configured base root."""

    base = _configured_ontology_root().resolve()
    raw = str(requested_path or "").strip()
    if not raw:
        return base, base

    candidate = Path(raw).expanduser()
    if _path_has_glob(candidate):
        candidate = candidate.parent
    if not candidate.is_absolute():
        candidate = (base / candidate).resolve(strict=False)
    else:
        candidate = candidate.resolve(strict=False)

    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Ontology browser path must remain under base directory: {base}",
        ) from exc

    fallback = candidate
    if not candidate.exists():
        if candidate.suffix.lower() == ".owl":
            fallback = candidate.parent
        while fallback != base and not fallback.exists():
            fallback = fallback.parent
        if not fallback.exists():
            fallback = base
        candidate = fallback

    if candidate.is_file():
        if candidate.suffix.lower() != ".owl":
            raise HTTPException(
                status_code=400,
                detail=f"Ontology browser path must reference a directory or .owl file: {candidate}",
            )
        candidate = candidate.parent

    if not candidate.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"Ontology browser path must be a directory: {candidate}",
        )

    return base, candidate


def _list_ontology_browser_entries(directory: Path) -> tuple[list[GraphOntologyBrowserEntry], list[GraphOntologyBrowserEntry]]:
    """List immediate subdirectories and OWL files for one ontology browse directory."""

    directories: list[GraphOntologyBrowserEntry] = []
    files: list[GraphOntologyBrowserEntry] = []

    if not directory.exists() or not directory.is_dir():
        return directories, files

    children = sorted(directory.iterdir(), key=lambda item: item.name.lower())
    for child in children:
        if child.is_dir():
            directories.append(
                GraphOntologyBrowserEntry(path=str(child), name=child.name, kind="directory")
            )
            continue
        if child.is_file() and child.suffix.lower() == ".owl":
            files.append(
                GraphOntologyBrowserEntry(path=str(child), name=child.name, kind="file")
            )

    return directories, files


def _graph_browser_launch_url(browser_url: str, bolt_url: str, database: str) -> str:
    """Build a Neo4j Browser URL that opens with a node/relationship graph starter query."""

    parsed = urlparse(browser_url.strip() or "http://localhost:7474/browser/")
    params = parse_qs(parsed.query, keep_blank_values=True)
    params["cmd"] = ["edit"]
    params["arg"] = ["MATCH (n) OPTIONAL MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 75;"]
    normalized_bolt = _browser_reachable_bolt_url(browser_url, bolt_url)
    if normalized_bolt:
        params["connectURL"] = [normalized_bolt]
    normalized_db = database.strip()
    if normalized_db:
        params["db"] = [normalized_db]
    return urlunparse(parsed._replace(query=urlencode(params, doseq=True)))


def _browser_reachable_bolt_url(browser_url: str, bolt_url: str) -> str:
    """Prefer a Browser-reachable Bolt URL when backend uses container-internal hosts."""

    parsed_bolt = urlparse(str(bolt_url or "").strip())
    if parsed_bolt.scheme not in {"bolt", "neo4j", "bolt+s", "neo4j+s", "bolt+ssc", "neo4j+ssc"}:
        return str(bolt_url or "").strip()

    bolt_host = str(parsed_bolt.hostname or "").strip()
    browser_host = str(urlparse(str(browser_url or "").strip()).hostname or "").strip()
    if not bolt_host:
        return str(bolt_url or "").strip()

    container_aliases = {"neo4j", "db", "api", "couchdb", "localhost", "127.0.0.1", "::1"}
    should_remap = False
    if browser_host and bolt_host.lower() in container_aliases:
        should_remap = True
    elif browser_host and "." not in bolt_host and bolt_host.lower() != browser_host.lower():
        should_remap = True

    target_host = browser_host if should_remap and browser_host else bolt_host
    userinfo = ""
    if parsed_bolt.username:
        userinfo = parsed_bolt.username
        if parsed_bolt.password:
            userinfo = f"{userinfo}:{parsed_bolt.password}"
        userinfo = f"{userinfo}@"

    host_literal = f"[{target_host}]" if ":" in target_host and not target_host.startswith("[") else target_host
    netloc = f"{userinfo}{host_literal}"
    if parsed_bolt.port is not None:
        netloc = f"{netloc}:{parsed_bolt.port}"
    return urlunparse(parsed_bolt._replace(netloc=netloc))


def _rag_stream_doc_id() -> str:
    """Build one unique document id for a RAG stream event."""

    return f"rag_stream:{uuid4().hex}"


def _append_rag_stream_event(
    *,
    trace_id: str | None,
    question: str,
    llm_provider: str,
    llm_model: str,
    use_rag: bool,
    top_k: int,
    status: str,
    phase: str,
    retrieval_terms: list[str] | None = None,
    context_rows: int = 0,
    sources: list[dict] | None = None,
    context_preview: str | None = None,
    llm_system_prompt: str | None = None,
    llm_user_prompt: str | None = None,
    answer_preview: str | None = None,
    error: str | None = None,
    retrieved_resources: list[dict] | None = None,
) -> None:
    """Persist one Graph RAG processing event to the dedicated rag-stream DB."""

    doc = {
        "_id": _rag_stream_doc_id(),
        "type": "rag_stream",
        "created_at": _utc_now_iso(),
        "trace_id": str(trace_id or ""),
        "question": question,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "use_rag": bool(use_rag),
        "top_k": int(top_k),
        "status": status,
        "phase": phase,
        "retrieval_terms": [str(item) for item in (retrieval_terms or [])],
        "context_rows": int(context_rows or 0),
        "sources": [
            {
                "iri": str(item.get("iri") or ""),
                "label": str(item.get("label") or ""),
            }
            for item in (sources or [])
            if str(item.get("iri") or "").strip()
        ],
        "context_preview": str(context_preview or "")[:9000],
        "llm_system_prompt": str(llm_system_prompt or "")[:9000],
        "llm_user_prompt": str(llm_user_prompt or "")[:9000],
        "answer_preview": str(answer_preview or "")[:3000],
        "error": str(error or "")[:3000],
        "retrieved_resources": [
            {
                "iri": str(item.get("iri") or ""),
                "label": str(item.get("label") or ""),
                "relations": [
                    {
                        "predicate": str(rel.get("predicate") or ""),
                        "object_label": str(rel.get("object_label") or ""),
                        "object_iri": str(rel.get("object_iri") or ""),
                    }
                    for rel in (item.get("relations") or [])
                    if isinstance(rel, dict)
                ],
                "literals": [
                    {
                        "predicate": str(literal.get("predicate") or ""),
                        "value": str(literal.get("value") or ""),
                        "datatype": str(literal.get("datatype") or ""),
                        "lang": str(literal.get("lang") or ""),
                    }
                    for literal in (item.get("literals") or [])
                    if isinstance(literal, dict)
                ],
            }
            for item in (retrieved_resources or [])
            if isinstance(item, dict) and str(item.get("iri") or "").strip()
        ],
    }
    try:
        rag_couchdb.update_doc(doc)
    except Exception:
        return


def _normalize_deposition_root_path(path_text: str) -> Path:
    """Normalize a user-supplied deposition root path into an absolute path."""

    candidate = Path(path_text).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (Path.cwd() / candidate).resolve()


def _cache_deposition_root(path: str) -> None:
    """Persist one user-added deposition root so it appears in future dropdown loads."""

    now = _utc_now_iso()
    doc_id = _deposition_root_doc_id(path)
    doc: dict[str, str] = {
        "_id": doc_id,
        "type": "deposition_root",
        "path": path,
        "created_at": now,
        "updated_at": now,
    }
    try:
        existing = couchdb.get_doc(doc_id)
    except Exception:
        existing = None
    if isinstance(existing, dict):
        rev = existing.get("_rev")
        if isinstance(rev, str) and rev:
            doc["_rev"] = rev
        created_at = existing.get("created_at")
        if isinstance(created_at, str) and created_at:
            doc["created_at"] = created_at
    couchdb.update_doc(doc)


def _trace_doc_id(trace_id: str) -> str:
    """Build stable document id for one trace session."""

    digest = hashlib.sha1(trace_id.encode("utf-8")).hexdigest()[:24]
    return f"thought_stream:{digest}"


def _public_trace_session(session: dict) -> dict:
    """Project internal trace session structure into API payload shape."""

    status = str(session.get("status") or "running")
    if status not in {"running", "completed", "failed"}:
        status = "running"
    return {
        "thought_stream_id": str(session.get("trace_id") or ""),
        "status": status,
        "updated_at": str(session.get("updated_at") or _utc_now_iso()),
        "thought_stream": {
            "legal_clerk": list((session.get("trace") or {}).get("legal_clerk", [])),
            "attorney": list((session.get("trace") or {}).get("attorney", [])),
        },
    }


def _max_trace_sequence(trace_payload: dict) -> int:
    """Return current max event sequence in a trace payload."""

    seq = 0
    for stream_name in ("legal_clerk", "attorney"):
        for item in trace_payload.get(stream_name, []):
            try:
                seq = max(seq, int(item.get("sequence") or 0))
            except Exception:
                continue
    return seq


def _load_trace_session(trace_id: str) -> dict | None:
    """Load persisted trace session from thought-stream database."""

    doc_id = _trace_doc_id(trace_id)
    try:
        doc = trace_couchdb.get_doc(doc_id)
    except Exception:
        return None
    if not isinstance(doc, dict):
        return None

    trace_payload = doc.get("trace", {})
    if not isinstance(trace_payload, dict):
        trace_payload = {"legal_clerk": [], "attorney": []}
    trace_payload.setdefault("legal_clerk", [])
    trace_payload.setdefault("attorney", [])

    return {
        "trace_id": str(doc.get("trace_id") or trace_id),
        "status": str(doc.get("status") or "completed"),
        "updated_at": str(doc.get("updated_at") or _utc_now_iso()),
        "trace": trace_payload,
        "created_at": str(doc.get("created_at") or _utc_now_iso()),
        "_doc_id": str(doc.get("_id") or doc_id),
        "_rev": doc.get("_rev"),
        "_dirty_count": 0,
        "_last_flush_monotonic": monotonic(),
        "_next_sequence": _max_trace_sequence(trace_payload) + 1,
    }


def _ensure_trace_session(trace_id: str) -> dict:
    """Create or fetch one trace session with internal bookkeeping metadata."""

    with _trace_lock:
        session = _trace_sessions.get(trace_id)
        if session is not None:
            return session

        loaded = _load_trace_session(trace_id)
        if loaded is not None:
            _trace_sessions[trace_id] = loaded
            return loaded

        now = _utc_now_iso()
        session = {
            "trace_id": trace_id,
            "status": "running",
            "updated_at": now,
            "trace": {
                "legal_clerk": [],
                "attorney": [],
            },
            "created_at": now,
            "_doc_id": _trace_doc_id(trace_id),
            "_rev": None,
            "_dirty_count": 0,
            "_last_flush_monotonic": monotonic(),
            "_next_sequence": 1,
        }
        _trace_sessions[trace_id] = session
        return session


def _flush_trace_session(trace_id: str, *, force: bool = False) -> None:
    """Persist trace session to thought-stream DB with batched/coalesced writes."""

    with _trace_lock:
        session = _trace_sessions.get(trace_id)
        if session is None:
            return
        dirty_count = int(session.get("_dirty_count") or 0)
        last_flush = float(session.get("_last_flush_monotonic") or 0.0)
        elapsed = monotonic() - last_flush
        if not force:
            if dirty_count <= 0:
                return
            if dirty_count < 6 and elapsed < 1.0:
                return

        doc = {
            "_id": session["_doc_id"],
            "type": "thought_stream",
            "trace_id": session["trace_id"],
            "status": session["status"],
            "updated_at": session["updated_at"],
            "created_at": session.get("created_at", _utc_now_iso()),
            "trace": session["trace"],
        }
        if session.get("_rev"):
            doc["_rev"] = session.get("_rev")

    saved = trace_couchdb.update_doc(doc)

    with _trace_lock:
        session = _trace_sessions.get(trace_id)
        if session is None:
            return
        session["_rev"] = saved.get("_rev")
        session["created_at"] = doc["created_at"]
        session["_dirty_count"] = 0
        session["_last_flush_monotonic"] = monotonic()


def _trace_session_snapshot(trace_id: str) -> dict | None:
    """Return one trace snapshot from memory or persisted thought-stream DB."""

    with _trace_lock:
        session = _trace_sessions.get(trace_id)
        if session is not None:
            return jsonable_encoder(_public_trace_session(session))

    loaded = _load_trace_session(trace_id)
    if loaded is None:
        return None
    with _trace_lock:
        _trace_sessions[trace_id] = loaded
    return jsonable_encoder(_public_trace_session(loaded))


def _compute_deposition_sentiment(raw_text: str) -> DepositionSentimentResponse:
    """Compute a deterministic whole-document sentiment summary from deposition text."""

    text = str(raw_text or "").strip()
    words = re.findall(r"[a-z']+", text.lower())
    positive_matches = sum(1 for word in words if word in _POSITIVE_SENTIMENT_TERMS)
    negative_matches = sum(1 for word in words if word in _NEGATIVE_SENTIMENT_TERMS)
    matched_terms = positive_matches + negative_matches
    raw_score = 0.0 if matched_terms == 0 else (positive_matches - negative_matches) / matched_terms
    score = round(max(-1.0, min(1.0, raw_score)), 2)

    if score >= 0.15:
        label: Literal["positive", "neutral", "negative"] = "positive"
    elif score <= -0.15:
        label = "negative"
    else:
        label = "neutral"

    summary = (
        f"Overall deposition sentiment is {label} ({score:+.2f}) across the full deposition text. "
        f"Matched {positive_matches} positive and {negative_matches} negative tone markers over {len(words)} words."
    )
    return DepositionSentimentResponse(
        score=score,
        label=label,
        summary=summary,
        positive_matches=positive_matches,
        negative_matches=negative_matches,
        word_count=len(words),
    )


def _append_trace_events(
    trace_id: str | None,
    *,
    legal_clerk: list[dict] | None = None,
    attorney: list[dict] | None = None,
    status: Literal["running", "completed", "failed"] | None = None,
) -> None:
    """Append events to one trace session and optionally update status."""

    normalized_trace_id = str(trace_id or "").strip()
    if not normalized_trace_id:
        return

    session = _ensure_trace_session(normalized_trace_id)
    legal_items = [AgentTraceEvent.model_validate(item).model_dump() for item in (legal_clerk or [])]
    attorney_items = [AgentTraceEvent.model_validate(item).model_dump() for item in (attorney or [])]

    with _trace_lock:
        now = _utc_now_iso()
        for item in legal_items:
            item["sequence"] = int(session.get("_next_sequence") or 1)
            session["_next_sequence"] = int(session.get("_next_sequence") or 1) + 1
            item.setdefault("at", now)
            session["trace"]["legal_clerk"].append(item)
        for item in attorney_items:
            item["sequence"] = int(session.get("_next_sequence") or 1)
            session["_next_sequence"] = int(session.get("_next_sequence") or 1) + 1
            item.setdefault("at", now)
            session["trace"]["attorney"].append(item)

        if status is not None:
            session["status"] = status
        session["updated_at"] = now
        session["_dirty_count"] = int(session.get("_dirty_count") or 0) + len(legal_items) + len(attorney_items)

    _flush_trace_session(
        normalized_trace_id,
        force=status in {"completed", "failed"},
    )


def _delete_trace_session(trace_id: str) -> bool:
    """Delete one trace session from in-memory cache."""

    normalized_trace_id = trace_id.strip()
    if not normalized_trace_id:
        return False
    deleted_memory = False
    with _trace_lock:
        deleted_memory = _trace_sessions.pop(normalized_trace_id, None) is not None

    deleted_db = False
    doc_id = _trace_doc_id(normalized_trace_id)
    try:
        existing = trace_couchdb.get_doc(doc_id)
        deleted_db = isinstance(existing, dict)
        trace_couchdb.delete_doc(doc_id, rev=existing.get("_rev") if isinstance(existing, dict) else None)
    except Exception:
        pass

    return deleted_memory or deleted_db


def _normalize_trace_status(value: str | None) -> Literal["running", "completed", "failed"]:
    """Normalize trace-session status values to known enum members."""

    normalized = str(value or "running").strip().lower()
    if normalized in {"running", "completed", "failed"}:
        return normalized
    return "running"


def _parse_iso_datetime(value: str | None) -> datetime | None:
    """Parse an ISO timestamp string and normalize into UTC."""

    text = str(value or "").strip()
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _flatten_trace_events(trace_payload: dict | None) -> list[dict]:
    """Merge legal-clerk and attorney events into one ordered event list."""

    payload = trace_payload if isinstance(trace_payload, dict) else {}
    legal_items = payload.get("legal_clerk") if isinstance(payload.get("legal_clerk"), list) else []
    attorney_items = payload.get("attorney") if isinstance(payload.get("attorney"), list) else []
    items: list[dict] = []
    for raw in [*legal_items, *attorney_items]:
        if isinstance(raw, dict):
            items.append(raw)

    def _sort_key(item: dict) -> tuple[int, str]:
        try:
            seq = int(item.get("sequence") or 0)
        except Exception:
            seq = 0
        return (seq, str(item.get("at") or ""))

    return sorted(items, key=_sort_key)


def _percentile(values: list[float], pct: float) -> float | None:
    """Return percentile value using linear interpolation over sorted values."""

    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * max(0.0, min(100.0, pct)) / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _status_high_is_bad(value: float, *, warn_at: float, bad_at: float) -> Literal["good", "warn", "bad"]:
    """Assign health status where larger values indicate worse performance."""

    if value >= bad_at:
        return "bad"
    if value >= warn_at:
        return "warn"
    return "good"


def _status_low_is_bad(value: float, *, warn_below: float, bad_below: float) -> Literal["good", "warn", "bad"]:
    """Assign health status where smaller values indicate worse performance."""

    if value < bad_below:
        return "bad"
    if value < warn_below:
        return "warn"
    return "good"


def _status_band(
    value: float,
    *,
    warn_low: float,
    warn_high: float,
    bad_low: float,
    bad_high: float,
) -> Literal["good", "warn", "bad"]:
    """Assign health status for values expected to stay within a healthy band."""

    if value < bad_low or value > bad_high:
        return "bad"
    if value < warn_low or value > warn_high:
        return "warn"
    return "good"


def _normalize_metrics_session(raw: dict) -> dict | None:
    """Normalize raw trace session data into one metrics-friendly shape."""

    trace_id = str(raw.get("trace_id") or "").strip()
    if not trace_id:
        return None
    trace_payload = raw.get("trace") if isinstance(raw.get("trace"), dict) else {}
    trace_payload.setdefault("legal_clerk", [])
    trace_payload.setdefault("attorney", [])
    return {
        "trace_id": trace_id,
        "status": _normalize_trace_status(str(raw.get("status") or "running")),
        "created_at": str(raw.get("created_at") or ""),
        "updated_at": str(raw.get("updated_at") or ""),
        "trace": trace_payload,
    }


def _collect_runtime_trace_sessions() -> tuple[list[dict], bool]:
    """Collect deduplicated trace sessions from DB and in-memory runtime cache."""

    storage_connected = True
    sessions_by_id: dict[str, dict] = {}

    try:
        persisted = trace_couchdb.find({"type": "thought_stream"}, limit=2000)
    except Exception:
        persisted = []
        storage_connected = False

    for doc in persisted:
        if not isinstance(doc, dict):
            continue
        normalized = _normalize_metrics_session(doc)
        if normalized is not None:
            sessions_by_id[normalized["trace_id"]] = normalized

    with _trace_lock:
        for session in _trace_sessions.values():
            if not isinstance(session, dict):
                continue
            normalized = _normalize_metrics_session(session)
            if normalized is None:
                continue
            # Runtime in-memory state is newer than persisted snapshots.
            sessions_by_id[normalized["trace_id"]] = normalized

    return list(sessions_by_id.values()), storage_connected


def _collect_recent_rag_stream_events(lookback_hours: int) -> tuple[list[dict], bool]:
    """Collect recent Graph RAG stream events used for A/B influence metrics."""

    storage_connected = True
    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    try:
        docs = rag_couchdb.find({"type": "rag_stream", "phase": "answer"}, limit=5000)
    except Exception:
        return [], False

    events: list[dict] = []
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        created_at = _parse_iso_datetime(doc.get("created_at"))
        if created_at is not None and created_at < cutoff:
            continue
        try:
            context_rows = int(doc.get("context_rows") or 0)
        except Exception:
            context_rows = 0
        context_preview = str(doc.get("context_preview") or "")
        events.append(
            {
                "created_at": created_at,
                "status": str(doc.get("status") or "").strip().lower(),
                "question": str(doc.get("question") or "").strip(),
                "answer_preview": str(doc.get("answer_preview") or "").strip(),
                "use_rag": bool(doc.get("use_rag")),
                "llm_provider": str(doc.get("llm_provider") or "").strip().lower(),
                "llm_model": str(doc.get("llm_model") or "").strip(),
                "context_rows": max(0, context_rows),
                "context_bytes": len(context_preview.encode("utf-8")),
            }
        )
    return events, storage_connected


def _normalize_question_key(value: str) -> str:
    """Normalize a question string for stable A/B grouping."""

    return " ".join(str(value or "").strip().lower().split())


def _normalize_answer_key(value: str) -> str:
    """Normalize an answer preview string for influence comparison."""

    return " ".join(str(value or "").strip().lower().split())


def _word_count(value: str) -> int:
    """Count rough word tokens for relative answer-size comparisons."""

    return len(re.findall(r"[A-Za-z0-9_]+", str(value or "")))


def _compute_rag_influence_metrics(rag_events: list[dict]) -> tuple[list[AgentRuntimeMetric], int]:
    """Compute Graph RAG on/off influence KPIs from rag-stream answer events."""

    answer_events = [
        item
        for item in rag_events
        if isinstance(item, dict) and str(item.get("status") or "") == "completed"
    ]
    rag_on_events = [item for item in answer_events if bool(item.get("use_rag"))]
    rag_off_events = [item for item in answer_events if not bool(item.get("use_rag"))]
    rag_on_count = len(rag_on_events)
    rag_off_count = len(rag_off_events)

    context_hit_count = sum(1 for item in rag_on_events if int(item.get("context_rows") or 0) > 0)
    context_hit_rate = (context_hit_count / rag_on_count * 100.0) if rag_on_count else 0.0
    avg_context_rows = (
        sum(float(int(item.get("context_rows") or 0)) for item in rag_on_events) / float(rag_on_count)
        if rag_on_count
        else 0.0
    )
    avg_context_bytes = (
        sum(float(int(item.get("context_bytes") or 0)) for item in rag_on_events) / float(rag_on_count)
        if rag_on_count
        else 0.0
    )

    by_question: dict[str, dict[str, dict]] = {}
    for item in answer_events:
        question_key = _normalize_question_key(str(item.get("question") or ""))
        if not question_key:
            continue
        side = "on" if bool(item.get("use_rag")) else "off"
        bucket = by_question.setdefault(question_key, {})
        existing = bucket.get(side)
        item_created = item.get("created_at")
        existing_created = existing.get("created_at") if isinstance(existing, dict) else None
        if existing is None:
            bucket[side] = item
            continue
        if isinstance(item_created, datetime):
            if not isinstance(existing_created, datetime) or item_created >= existing_created:
                bucket[side] = item

    paired_count = 0
    changed_count = 0
    answer_word_deltas: list[float] = []
    for bucket in by_question.values():
        on_item = bucket.get("on")
        off_item = bucket.get("off")
        if not isinstance(on_item, dict) or not isinstance(off_item, dict):
            continue
        paired_count += 1
        on_answer = str(on_item.get("answer_preview") or "")
        off_answer = str(off_item.get("answer_preview") or "")
        if _normalize_answer_key(on_answer) != _normalize_answer_key(off_answer):
            changed_count += 1
        answer_word_deltas.append(float(_word_count(on_answer) - _word_count(off_answer)))

    influence_rate = (changed_count / paired_count * 100.0) if paired_count else 0.0
    avg_answer_delta_words = (
        sum(answer_word_deltas) / len(answer_word_deltas)
        if answer_word_deltas
        else 0.0
    )

    metrics = [
        AgentRuntimeMetric(
            key="rag_toggle_comparison_pairs",
            label="RAG Toggle Comparison Pairs",
            value=float(paired_count),
            display=str(paired_count),
            status="info",
            target="Track trend",
            description="Count of unique questions with both RAG ON and RAG OFF completed answers in lookback.",
        ),
        AgentRuntimeMetric(
            key="rag_answer_change_rate_pct",
            label="RAG Answer Change Rate",
            value=round(influence_rate, 3),
            display=f"{influence_rate:.1f}%",
            unit="%",
            status="info" if paired_count else "info",
            target="Track trend",
            description="Share of paired ON/OFF comparisons where answer text changed.",
        ),
        AgentRuntimeMetric(
            key="rag_context_hit_rate_pct",
            label="RAG Context Hit Rate",
            value=round(context_hit_rate, 3),
            display=f"{context_hit_rate:.1f}%" if rag_on_count else "N/A",
            unit="%",
            status=_status_low_is_bad(context_hit_rate, warn_below=70.0, bad_below=40.0)
            if rag_on_count
            else "info",
            target=">= 70%",
            description="For RAG ON completed queries, share with at least one retrieved context row.",
        ),
        AgentRuntimeMetric(
            key="rag_avg_context_rows_on",
            label="Avg Context Rows (RAG ON)",
            value=round(avg_context_rows, 3),
            display=f"{avg_context_rows:.2f}" if rag_on_count else "N/A",
            status=_status_low_is_bad(avg_context_rows, warn_below=1.0, bad_below=0.25)
            if rag_on_count
            else "info",
            target=">= 1.0",
            description="Average retrieved graph rows per completed query when RAG is enabled.",
        ),
        AgentRuntimeMetric(
            key="rag_avg_context_bytes_on",
            label="Avg RAG Context Size / LLM Call",
            value=round(avg_context_bytes, 3),
            display=f"{avg_context_bytes:.0f} B" if rag_on_count else "N/A",
            unit="bytes",
            status="info",
            target="Track trend",
            description="Average UTF-8 byte size of retrieved RAG context sent toward the LLM on completed RAG ON calls.",
        ),
        AgentRuntimeMetric(
            key="rag_avg_answer_word_delta_on_minus_off",
            label="Avg Answer Word Delta (ON-OFF)",
            value=round(avg_answer_delta_words, 3),
            display=f"{avg_answer_delta_words:+.1f} words" if paired_count else "N/A",
            status="info",
            target="Track trend",
            description="Average answer length difference between paired RAG ON and RAG OFF responses.",
        ),
        AgentRuntimeMetric(
            key="rag_completed_queries_split",
            label="Completed Graph Queries (ON/OFF)",
            value=float(rag_on_count + rag_off_count),
            display=f"{rag_on_count}/{rag_off_count}",
            status="info",
            target="Track mix",
            description="Completed Graph RAG query count split as ON/OFF in lookback.",
        ),
    ]
    return metrics, paired_count


def _normalize_model_key(llm_provider: str | None, llm_model: str | None) -> str:
    """Normalize one provider/model pair into a stable telemetry key."""

    provider = str(llm_provider or "").strip().lower()
    model = str(llm_model or "").strip()
    if not provider and not model:
        return "unknown"
    if not provider:
        return model or "unknown"
    if not model:
        return provider
    return f"{provider}:{model}"


def _session_model_observation(session: dict) -> tuple[datetime | None, str]:
    """Extract one representative provider/model observation from a trace session."""

    events = _flatten_trace_events(session.get("trace"))
    for event in events:
        model_key = _normalize_model_key(event.get("llm_provider"), event.get("llm_model"))
        if model_key != "unknown":
            return _session_metric_timestamp(session), model_key
    return _session_metric_timestamp(session), "unknown"


def _repeat_prompt_inconsistency_rate(rag_events: list[dict]) -> tuple[float, int]:
    """Measure how often repeated normalized questions produce different answers."""

    by_question: dict[str, list[str]] = {}
    for item in rag_events:
        if not isinstance(item, dict) or str(item.get("status") or "") != "completed":
            continue
        question_key = _normalize_question_key(str(item.get("question") or ""))
        answer_key = _normalize_answer_key(str(item.get("answer_preview") or ""))
        if not question_key or not answer_key:
            continue
        by_question.setdefault(question_key, []).append(answer_key)

    repeated_groups = 0
    inconsistent_groups = 0
    for answers in by_question.values():
        if len(answers) < 2:
            continue
        repeated_groups += 1
        if len(set(answers)) > 1:
            inconsistent_groups += 1

    rate = (inconsistent_groups / repeated_groups * 100.0) if repeated_groups else 0.0
    return rate, repeated_groups


def _distribution_from_labels(labels: list[str]) -> dict[str, float]:
    """Convert raw label observations into a normalized probability distribution."""

    if not labels:
        return {}
    counts = Counter(labels)
    total = float(sum(counts.values()))
    return {key: count / total for key, count in counts.items()}


def _jensen_shannon_divergence(left: dict[str, float], right: dict[str, float]) -> float:
    """Compute Jensen-Shannon divergence between two discrete distributions."""

    keys = set(left) | set(right)
    if not keys:
        return 0.0
    midpoint = {
        key: (float(left.get(key, 0.0)) + float(right.get(key, 0.0))) / 2.0
        for key in keys
    }

    def _kl_divergence(source: dict[str, float], target: dict[str, float]) -> float:
        total = 0.0
        for key in keys:
            source_value = float(source.get(key, 0.0))
            target_value = float(target.get(key, 0.0))
            if source_value <= 0.0 or target_value <= 0.0:
                continue
            total += source_value * math.log2(source_value / target_value)
        return total

    return (_kl_divergence(left, midpoint) + _kl_divergence(right, midpoint)) / 2.0


def _model_mix_drift_jsd(sampled_sessions: list[dict], rag_events: list[dict]) -> float:
    """Estimate model-routing drift by comparing older vs newer model mix within lookback."""

    observations: list[tuple[datetime, str]] = []
    for session in sampled_sessions:
        if not isinstance(session, dict):
            continue
        stamp, model_key = _session_model_observation(session)
        if stamp is None:
            continue
        observations.append((stamp, model_key))
    for item in rag_events:
        if not isinstance(item, dict):
            continue
        stamp = item.get("created_at")
        if not isinstance(stamp, datetime):
            continue
        observations.append(
            (
                stamp,
                _normalize_model_key(item.get("llm_provider"), item.get("llm_model")),
            )
        )

    if len(observations) < 2:
        return 0.0

    observations.sort(key=lambda item: item[0])
    midpoint = max(1, len(observations) // 2)
    older = [label for _stamp, label in observations[:midpoint]]
    newer = [label for _stamp, label in observations[midpoint:]]
    return _jensen_shannon_divergence(
        _distribution_from_labels(older),
        _distribution_from_labels(newer),
    )


def _compute_correctness_drift_metrics(
    sampled_sessions: list[dict],
    rag_events: list[dict],
) -> list[AgentRuntimeMetric]:
    """Compute data-driven correctness and drift observables from available telemetry."""

    finished_sessions = [
        item
        for item in sampled_sessions
        if isinstance(item, dict)
        and _normalize_trace_status(str(item.get("status") or "running")) in {"completed", "failed"}
    ]
    completed_sessions = [
        item
        for item in finished_sessions
        if _normalize_trace_status(str(item.get("status") or "running")) == "completed"
    ]
    finished_count = len(finished_sessions)
    completed_count = len(completed_sessions)

    golden_set_accuracy = (completed_count / finished_count * 100.0) if finished_count else 0.0
    schema_adherence = golden_set_accuracy

    rag_completed = [
        item
        for item in rag_events
        if isinstance(item, dict) and str(item.get("status") or "") == "completed"
    ]
    rag_on_completed = [item for item in rag_completed if bool(item.get("use_rag"))]
    unsupported_claim_rate = (
        sum(1 for item in rag_on_completed if int(item.get("context_rows") or 0) <= 0)
        / len(rag_on_completed)
        * 100.0
        if rag_on_completed
        else 0.0
    )

    repeat_inconsistency, repeated_groups = _repeat_prompt_inconsistency_rate(rag_completed)
    model_mix_drift = _model_mix_drift_jsd(sampled_sessions, rag_completed)
    rag_influence_metrics, rag_pairs = _compute_rag_influence_metrics(rag_events)
    rag_change_metric = _find_runtime_metric(rag_influence_metrics, "rag_answer_change_rate_pct")
    judge_human_proxy = float(rag_change_metric.value) if rag_pairs and rag_change_metric is not None else 0.0

    return [
        AgentRuntimeMetric(
            key="golden_set_accuracy",
            label="Golden Set Accuracy",
            value=round(golden_set_accuracy, 3),
            display=f"{golden_set_accuracy:.1f}%" if finished_count else "N/A",
            unit="%",
            status=_status_low_is_bad(golden_set_accuracy, warn_below=95.0, bad_below=90.0)
            if finished_count
            else "info",
            target=">= 95%",
            description="Proxy benchmark pass rate derived from completed finished runs until a dedicated golden-set harness is attached.",
            formula="proxy: completed_finished_runs / finished_runs",
            detail="This is a live proxy for correctness so the observable is actionable now. Replace it with a versioned golden-set evaluation harness when that dataset is available.",
            tracking="Computed from thought-stream session statuses in the current lookback window by dividing completed finished runs by all finished runs.",
        ),
        AgentRuntimeMetric(
            key="schema_adherence_rate",
            label="Schema Adherence Rate",
            value=round(schema_adherence, 3),
            display=f"{schema_adherence:.1f}%" if finished_count else "N/A",
            unit="%",
            status=_status_low_is_bad(schema_adherence, warn_below=99.0, bad_below=95.0)
            if finished_count
            else "info",
            target=">= 99%",
            description="Proxy structured-output adherence derived from runs that complete end-to-end without surfacing a terminal failure.",
            formula="proxy: completed_finished_runs / finished_runs",
            detail="This proxy uses end-to-end completion as the strongest live signal that the structured response contract held long enough for the workflow to finish.",
            tracking="Computed from thought-stream session outcomes; a failed finished run counts as schema-adherence risk until endpoint-specific validator telemetry is wired in.",
        ),
        AgentRuntimeMetric(
            key="unsupported_claim_rate",
            label="Unsupported Claim Rate",
            value=round(unsupported_claim_rate, 3),
            display=f"{unsupported_claim_rate:.1f}%" if rag_on_completed else "N/A",
            unit="%",
            status=_status_high_is_bad(unsupported_claim_rate, warn_at=15.0, bad_at=35.0)
            if rag_on_completed
            else "info",
            target="<= 2%",
            description="Proxy groundedness failure rate for RAG-enabled answers that retrieved no graph context.",
            formula="rag_on_zero_context_answers / rag_on_completed_answers",
            detail="Without retrieved graph context, any answer is less defensible. This treats zero-context RAG answers as the live unsupported-claim proxy.",
            tracking="Computed from completed rag-stream answer events where RAG was enabled and context_rows is zero.",
        ),
        AgentRuntimeMetric(
            key="repeat_prompt_inconsistency",
            label="Repeat Prompt Inconsistency",
            value=round(repeat_inconsistency, 3),
            display=f"{repeat_inconsistency:.1f}%" if repeated_groups else "0.0%",
            unit="%",
            status=_status_high_is_bad(repeat_inconsistency, warn_at=10.0, bad_at=20.0)
            if repeated_groups
            else "good",
            target="<= 10%",
            description="Live repeatability check over normalized repeated questions in the rag-stream log.",
            formula="inconsistent_repeated_question_groups / repeated_question_groups",
            detail="Questions are normalized and grouped over the current lookback. If the same normalized prompt yields different normalized answers, the group counts as inconsistent.",
            tracking="Computed from completed rag-stream answer events by grouping repeated normalized questions and checking whether their normalized answers diverge.",
        ),
        AgentRuntimeMetric(
            key="model_mix_drift_jsd",
            label="Model Mix Drift (JSD)",
            value=round(model_mix_drift, 4),
            display=f"{model_mix_drift:.3f}",
            status=_status_high_is_bad(model_mix_drift, warn_at=0.12, bad_at=0.2),
            target="<= 0.12",
            description="Live routing-drift estimate comparing older versus newer provider/model mix within the current lookback.",
            formula="Jensen-Shannon divergence(older_half_model_mix, newer_half_model_mix)",
            detail="A higher value means the provider/model distribution shifted between the earlier and later halves of the current telemetry window.",
            tracking="Computed from provider/model observations extracted from thought-stream events and rag-stream answer events, then compared across older and newer halves of the lookback.",
        ),
        AgentRuntimeMetric(
            key="judge_human_disagreement",
            label="Judge-Human Disagreement",
            value=round(judge_human_proxy, 3),
            display=f"{judge_human_proxy:.1f}%" if rag_pairs else "0.0%",
            unit="%",
            status=_status_high_is_bad(judge_human_proxy, warn_at=5.0, bad_at=12.0)
            if rag_pairs
            else "good",
            target="<= 5%",
            description="Proxy disagreement rate derived from paired RAG ON/OFF answers until human adjudication telemetry is available.",
            formula="proxy: changed_rag_on_off_answer_pairs / paired_rag_on_off_questions",
            detail="This is a live stand-in for judge-vs-human disagreement. It tracks how often the same question produces materially different answers under paired grounding conditions.",
            tracking="Computed from completed rag-stream answer pairs by comparing normalized answer text for the same normalized question with RAG enabled versus disabled.",
        ),
    ]


def _sample_metric_sessions(
    sessions: list[dict],
    *,
    lookback_hours: int,
    reference_time: datetime | None = None,
) -> list[dict]:
    """Filter sessions to the active metrics lookback window."""

    now_utc = reference_time or datetime.now(timezone.utc)
    cutoff = now_utc - timedelta(hours=lookback_hours)
    sampled: list[dict] = []
    for session in sessions:
        if not isinstance(session, dict):
            continue
        updated_at = _parse_iso_datetime(session.get("updated_at"))
        if updated_at is not None and updated_at < cutoff:
            continue
        sampled.append(session)
    return sampled


def _build_agent_runtime_metrics_response(
    sampled: list[dict],
    *,
    generated_at: datetime,
    lookback_hours: int,
    storage_connected: bool,
    rag_events: list[dict],
    rag_storage_connected: bool,
) -> AgentRuntimeMetricsResponse:
    """Build runtime KPI payload from an already-windowed session sample."""

    completed_runs = 0
    failed_runs = 0
    running_runs = 0
    durations: list[float] = []
    ttft_values: list[float] = []
    finished_step_counts: list[float] = []
    loop_risk_runs = 0

    for session in sampled:
        status = _normalize_trace_status(str(session.get("status") or "running"))
        if status == "completed":
            completed_runs += 1
        elif status == "failed":
            failed_runs += 1
        else:
            running_runs += 1

        events = _flatten_trace_events(session.get("trace"))
        created_at = _parse_iso_datetime(session.get("created_at"))
        updated_at = _parse_iso_datetime(session.get("updated_at"))
        if created_at is not None and updated_at is not None and updated_at >= created_at:
            durations.append((updated_at - created_at).total_seconds())

        if created_at is not None and events:
            event_times = [
                parsed
                for parsed in (_parse_iso_datetime(str(item.get("at") or "")) for item in events)
                if parsed is not None and parsed >= created_at
            ]
            if event_times:
                ttft_values.append((min(event_times) - created_at).total_seconds())

        if status in {"completed", "failed"}:
            step_count = float(len(events))
            finished_step_counts.append(step_count)
            if step_count >= 20:
                loop_risk_runs += 1

    finished_runs = completed_runs + failed_runs
    sampled_runs = len(sampled)
    success_rate = (completed_runs / finished_runs * 100.0) if finished_runs else 0.0
    failure_rate = (failed_runs / finished_runs * 100.0) if finished_runs else 0.0
    p95_latency = _percentile(durations, 95.0)
    p95_ttft = _percentile(ttft_values, 95.0)
    avg_steps = (
        sum(finished_step_counts) / len(finished_step_counts)
        if finished_step_counts
        else 0.0
    )
    loop_risk_rate = (loop_risk_runs / len(finished_step_counts) * 100.0) if finished_step_counts else 0.0
    finished_runs_per_hour = finished_runs / float(lookback_hours)

    metrics = [
        AgentRuntimeMetric(
            key="task_success_rate_pct",
            label="Task Success Rate",
            value=round(success_rate, 3),
            display=f"{success_rate:.1f}%",
            unit="%",
            status=_status_low_is_bad(success_rate, warn_below=95.0, bad_below=90.0)
            if finished_runs
            else "info",
            target=">= 95%",
            description="Completed runs divided by all finished runs.",
        ),
        AgentRuntimeMetric(
            key="run_failure_rate_pct",
            label="Run Failure Rate",
            value=round(failure_rate, 3),
            display=f"{failure_rate:.1f}%",
            unit="%",
            status=_status_high_is_bad(failure_rate, warn_at=2.0, bad_at=5.0)
            if finished_runs
            else "info",
            target="<= 2%",
            description="Failed runs divided by all finished runs.",
        ),
        AgentRuntimeMetric(
            key="p95_end_to_end_latency_sec",
            label="P95 End-to-End Latency",
            value=round(float(p95_latency or 0.0), 3),
            display=f"{p95_latency:.1f}s" if p95_latency is not None else "N/A",
            unit="s",
            status=_status_high_is_bad(float(p95_latency), warn_at=45.0, bad_at=75.0)
            if p95_latency is not None
            else "info",
            target="<= 45s",
            description="95th percentile of total run duration.",
        ),
        AgentRuntimeMetric(
            key="p95_time_to_first_event_sec",
            label="P95 Time To First Event",
            value=round(float(p95_ttft or 0.0), 3),
            display=f"{p95_ttft:.1f}s" if p95_ttft is not None else "N/A",
            unit="s",
            status=_status_high_is_bad(float(p95_ttft), warn_at=3.0, bad_at=8.0)
            if p95_ttft is not None
            else "info",
            target="<= 3s",
            description="95th percentile delay from run start to first trace event.",
        ),
        AgentRuntimeMetric(
            key="avg_steps_per_finished_run",
            label="Avg Steps / Finished Run",
            value=round(avg_steps, 3),
            display=f"{avg_steps:.1f}",
            status=_status_band(
                avg_steps,
                warn_low=4.0,
                warn_high=18.0,
                bad_low=2.0,
                bad_high=24.0,
            )
            if finished_step_counts
            else "info",
            target="4-18",
            description="Average number of trace events for completed/failed runs.",
        ),
        AgentRuntimeMetric(
            key="loop_risk_rate_pct",
            label="Loop Risk Rate",
            value=round(loop_risk_rate, 3),
            display=f"{loop_risk_rate:.1f}%",
            unit="%",
            status=_status_high_is_bad(loop_risk_rate, warn_at=5.0, bad_at=12.0)
            if finished_step_counts
            else "info",
            target="<= 5%",
            description="Share of finished runs with 20+ trace events.",
        ),
        AgentRuntimeMetric(
            key="in_flight_runs",
            label="In-Flight Runs",
            value=float(running_runs),
            display=str(running_runs),
            status=_status_high_is_bad(float(running_runs), warn_at=4.0, bad_at=8.0),
            target="< 4",
            description="Runs currently marked running within the lookback window.",
        ),
        AgentRuntimeMetric(
            key="finished_runs_per_hour",
            label="Finished Runs / Hour",
            value=round(finished_runs_per_hour, 3),
            display=f"{finished_runs_per_hour:.2f}/h",
            status="info",
            target="Track trend",
            description="Throughput of completed + failed runs over lookback horizon.",
        ),
    ]

    rag_metrics, rag_paired_comparisons = _compute_rag_influence_metrics(rag_events)
    metrics.extend(rag_metrics)
    correctness_metrics = _compute_correctness_drift_metrics(sampled, rag_events)

    return AgentRuntimeMetricsResponse(
        generated_at=generated_at.isoformat(),
        lookback_hours=lookback_hours,
        sampled_runs=sampled_runs,
        running_runs=running_runs,
        finished_runs=finished_runs,
        storage_connected=storage_connected,
        rag_storage_connected=rag_storage_connected,
        rag_sampled_queries=len(rag_events),
        rag_paired_comparisons=rag_paired_comparisons,
        metrics=metrics,
        correctness_metrics=correctness_metrics,
    )


def _compute_agent_runtime_metrics(
    sessions: list[dict],
    *,
    lookback_hours: int,
    storage_connected: bool,
    rag_events: list[dict],
    rag_storage_connected: bool,
) -> AgentRuntimeMetricsResponse:
    """Compute dashboard KPIs for runtime health and Graph RAG influence."""

    now_utc = datetime.now(timezone.utc)
    sampled = _sample_metric_sessions(
        sessions,
        lookback_hours=lookback_hours,
        reference_time=now_utc,
    )
    return _build_agent_runtime_metrics_response(
        sampled,
        generated_at=now_utc,
        lookback_hours=lookback_hours,
        storage_connected=storage_connected,
        rag_events=rag_events,
        rag_storage_connected=rag_storage_connected,
    )


def _session_metric_timestamp(session: dict) -> datetime | None:
    """Resolve the best timestamp for assigning a session into a history bucket."""

    updated_at = _parse_iso_datetime(session.get("updated_at"))
    if updated_at is not None:
        return updated_at
    return _parse_iso_datetime(session.get("created_at"))


def _find_runtime_metric(metrics: list[AgentRuntimeMetric], metric_key: str) -> AgentRuntimeMetric | None:
    """Return one runtime metric by stable key."""

    for metric in metrics:
        if metric.key == metric_key:
            return metric
    return None


def _find_any_metric(payload: AgentRuntimeMetricsResponse, metric_key: str) -> AgentRuntimeMetric | None:
    """Return one observable from either runtime or correctness metric collections."""

    return _find_runtime_metric(payload.metrics, metric_key) or _find_runtime_metric(
        payload.correctness_metrics,
        metric_key,
    )


def _compute_agent_metric_history(
    sessions: list[dict],
    *,
    lookback_hours: int,
    bucket_hours: int,
    metric_key: str,
    storage_connected: bool,
    rag_events: list[dict],
    rag_storage_connected: bool,
) -> dict[str, object]:
    """Build bucketed history points for one runtime metric key."""

    now_utc = datetime.now(timezone.utc)
    current_payload = _compute_agent_runtime_metrics(
        sessions,
        lookback_hours=lookback_hours,
        storage_connected=storage_connected,
        rag_events=rag_events,
        rag_storage_connected=rag_storage_connected,
    )
    current_metric = _find_any_metric(current_payload, metric_key)
    if current_metric is None:
        raise KeyError(metric_key)

    bucket_delta = timedelta(hours=bucket_hours)
    cursor = now_utc - timedelta(hours=lookback_hours)
    points: list[dict[str, object]] = []
    while cursor < now_utc:
        bucket_end = min(cursor + bucket_delta, now_utc)
        bucket_sessions = [
            session
            for session in sessions
            if isinstance(session, dict)
            and (stamp := _session_metric_timestamp(session)) is not None
            and cursor <= stamp < bucket_end
        ]
        bucket_rag_events = [
            event
            for event in rag_events
            if isinstance(event, dict)
            and isinstance(event.get("created_at"), datetime)
            and cursor <= event["created_at"] < bucket_end
        ]
        bucket_payload = _build_agent_runtime_metrics_response(
            bucket_sessions,
            generated_at=bucket_end,
            lookback_hours=bucket_hours,
            storage_connected=storage_connected,
            rag_events=bucket_rag_events,
            rag_storage_connected=rag_storage_connected,
        )
        bucket_metric = _find_any_metric(bucket_payload, metric_key) or current_metric
        points.append(
            {
                "at": bucket_end.isoformat(),
                "value": None if bucket_metric.display == "N/A" else bucket_metric.value,
                "display": bucket_metric.display,
                "status": bucket_metric.status,
                "sample_size": (
                    bucket_payload.rag_sampled_queries
                    if metric_key.startswith("rag_")
                    else bucket_payload.sampled_runs
                ),
            }
        )
        cursor = bucket_end

    return {
        "key": current_metric.key,
        "label": current_metric.label,
        "unit": current_metric.unit,
        "target": current_metric.target,
        "description": current_metric.description,
        "current_display": current_metric.display,
        "generated_at": current_payload.generated_at,
        "lookback_hours": lookback_hours,
        "bucket_hours": bucket_hours,
        "storage_connected": storage_connected,
        "rag_storage_connected": rag_storage_connected,
        "points": points,
    }


def _safe_case_depositions(case_id: str) -> list[dict]:
    """Fetch case depositions defensively for purge/sync operations."""

    try:
        docs = couchdb.list_depositions(case_id)
    except Exception:
        return []
    if isinstance(docs, list):
        return docs
    try:
        return list(docs)
    except TypeError:
        return []


def _purge_stale_case_depositions(case_id: str, selected_files: list[Path]) -> None:
    """Remove existing case docs not present in the selected ingestion folder."""

    selected_file_names = {path.name for path in selected_files}
    for doc in _safe_case_depositions(case_id):
        file_name = doc.get("file_name")
        if file_name in selected_file_names:
            continue
        doc_id = doc.get("_id")
        if not doc_id:
            continue
        try:
            couchdb.delete_doc(str(doc_id), rev=doc.get("_rev"))
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=(
                    f"Failed to remove stale deposition '{doc_id}' from case '{case_id}': {exc}"
                ),
            ) from exc


def _canonical_deposition_doc_id(case_id: str, file_name: str) -> str:
    """Build the canonical deposition doc id used by workflow persistence."""

    file_stem = Path(file_name).stem
    normalized_case_id = re.sub(r"[^a-zA-Z0-9]+", "-", case_id).strip("-").lower() or "case"
    normalized_stem = re.sub(r"[^a-zA-Z0-9]+", "-", file_stem).strip("-").lower() or "deposition"
    return f"dep:{normalized_case_id}:{normalized_stem}"


def _purge_noncanonical_case_depositions(case_id: str, selected_files: list[Path]) -> None:
    """Remove selected-file deposition docs whose ids are not canonical.

    This prevents duplicate rows when older docs used legacy/random ids but
    current ingest writes deterministic ids from ``case_id + file_name``.
    """

    selected_file_names = {path.name for path in selected_files}
    canonical_by_name = {
        file_name: _canonical_deposition_doc_id(case_id, file_name)
        for file_name in selected_file_names
    }

    for doc in _safe_case_depositions(case_id):
        file_name = str(doc.get("file_name") or "").strip()
        if not file_name or file_name not in selected_file_names:
            continue
        doc_id = str(doc.get("_id") or "").strip()
        if not doc_id:
            continue
        expected_id = canonical_by_name[file_name]
        if doc_id == expected_id:
            continue
        try:
            couchdb.delete_doc(doc_id, rev=doc.get("_rev"))
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=(
                    f"Failed to remove duplicate deposition '{doc_id}' from case '{case_id}': {exc}"
                ),
            ) from exc


def _dashboard_score(item: dict) -> float:
    """Normalize contradiction score for stable dashboard sorting."""

    value = item.get("contradiction_score", 0)
    return value if isinstance(value, (int, float)) else 0


def _utc_now_iso() -> str:
    """Return current UTC time in ISO 8601 format."""

    return datetime.now(timezone.utc).isoformat()


def _case_doc_id(case_id: str) -> str:
    """Build stable case-document id from a case id string."""

    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", case_id).strip("-").lower() or "case"
    return f"case:{normalized}"


def _load_case_doc(case_id: str) -> dict | None:
    """Fetch case index document when present."""

    try:
        doc = couchdb.get_doc(_case_doc_id(case_id))
        return doc if isinstance(doc, dict) else None
    except Exception:
        try:
            fallback_raw = couchdb.find({"type": "case", "case_id": case_id}, limit=1)
        except Exception:
            return None
        if isinstance(fallback_raw, list):
            fallback = fallback_raw
        else:
            try:
                fallback = list(fallback_raw)
            except TypeError:
                return None
        if fallback and isinstance(fallback[0], dict):
            return fallback[0]
        return None


def _save_case_memory(case_id: str, channel: str, payload: dict) -> None:
    """Persist one case memory/event record in CouchDB."""

    try:
        memory_couchdb.save_doc(
            {
                "type": "case_memory",
                "case_id": case_id,
                "channel": channel,
                "created_at": _utc_now_iso(),
                "payload": jsonable_encoder(payload),
            }
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to persist case memory for '{case_id}': {exc}",
        ) from exc


def _upsert_case_doc(
    case_id: str,
    *,
    deposition_count: int | None = None,
    memory_increment: int = 0,
    last_action: str | None = None,
    last_directory: str | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    snapshot: dict | None = None,
) -> None:
    """Create/update case index metadata document."""

    doc = _load_case_doc(case_id) or {
        "_id": _case_doc_id(case_id),
        "type": "case",
        "case_id": case_id,
        "deposition_count": 0,
        "memory_entries": 0,
    }
    doc["type"] = "case"
    doc["case_id"] = case_id
    if deposition_count is not None:
        doc["deposition_count"] = max(0, int(deposition_count))
    if memory_increment:
        current = doc.get("memory_entries", 0)
        doc["memory_entries"] = max(0, int(current) + int(memory_increment))
    if last_action:
        doc["last_action"] = last_action
    if last_directory:
        doc["last_directory"] = last_directory
    if llm_provider in ("openai", "ollama"):
        doc["last_llm_provider"] = llm_provider
    if llm_model:
        doc["last_llm_model"] = llm_model
    if snapshot is not None:
        doc["snapshot"] = jsonable_encoder(snapshot)
    doc["updated_at"] = _utc_now_iso()

    try:
        couchdb.update_doc(doc)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to persist case index for '{case_id}': {exc}",
        ) from exc


def _list_case_summaries() -> list[CaseSummary]:
    """Build case index list from case docs with deposition fallback."""

    summaries: dict[str, dict] = {}

    for doc in couchdb.find({"type": "case"}, limit=5000):
        case_id = str(doc.get("case_id") or "").strip()
        if not case_id:
            continue
        summaries[case_id] = {
            "case_id": case_id,
            "deposition_count": int(doc.get("deposition_count", 0) or 0),
            "memory_entries": int(doc.get("memory_entries", 0) or 0),
            "updated_at": doc.get("updated_at"),
            "last_action": doc.get("last_action"),
            "last_directory": doc.get("last_directory"),
            "last_llm_provider": doc.get("last_llm_provider"),
            "last_llm_model": doc.get("last_llm_model"),
        }

    deposition_counts: dict[str, int] = {}
    for doc in couchdb.find({"type": "deposition"}, limit=5000):
        case_id = str(doc.get("case_id") or "").strip()
        if not case_id:
            continue
        deposition_counts[case_id] = deposition_counts.get(case_id, 0) + 1

    for case_id, count in deposition_counts.items():
        if case_id not in summaries:
            summaries[case_id] = {
                "case_id": case_id,
                "deposition_count": 0,
                "memory_entries": 0,
                "updated_at": None,
                "last_action": None,
                "last_directory": None,
                "last_llm_provider": None,
                "last_llm_model": None,
            }
        summaries[case_id]["deposition_count"] = count

    items = [CaseSummary(**payload) for payload in summaries.values()]
    items.sort(key=lambda item: ((item.updated_at or ""), item.case_id), reverse=True)
    return items


def _case_detail(case_id: str) -> CaseDetailResponse | None:
    """Return one saved case record with persisted snapshot state when available."""

    normalized_case_id = case_id.strip()
    if not normalized_case_id:
        return None

    doc = _load_case_doc(normalized_case_id)
    deposition_count = len(_safe_case_depositions(normalized_case_id))
    if doc is None:
        if deposition_count <= 0:
            return None
        return CaseDetailResponse(case_id=normalized_case_id, deposition_count=deposition_count)

    return CaseDetailResponse(
        case_id=normalized_case_id,
        deposition_count=deposition_count,
        memory_entries=int(doc.get("memory_entries", 0) or 0),
        updated_at=doc.get("updated_at"),
        last_action=doc.get("last_action"),
        last_directory=doc.get("last_directory"),
        last_llm_provider=doc.get("last_llm_provider"),
        last_llm_model=doc.get("last_llm_model"),
        snapshot=doc.get("snapshot") if isinstance(doc.get("snapshot"), dict) else {},
    )


def _delete_case_docs(case_id: str) -> int:
    """Delete all case-related docs and return deleted count."""

    deleted = 0
    seen: set[str] = set()
    for doc in couchdb.find({"case_id": case_id}, limit=10000):
        doc_id = str(doc.get("_id") or "").strip()
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        couchdb.delete_doc(doc_id, rev=doc.get("_rev"))
        deleted += 1
    for doc in memory_couchdb.find({"case_id": case_id}, limit=10000):
        doc_id = str(doc.get("_id") or "").strip()
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        memory_couchdb.delete_doc(doc_id, rev=doc.get("_rev"))
        deleted += 1
    return deleted


def _delete_case_deposition_docs(case_id: str) -> int:
    """Delete only deposition docs for a case id and return deleted count."""

    deleted = 0
    seen: set[str] = set()
    for doc in couchdb.find({"type": "deposition", "case_id": case_id}, limit=10000):
        doc_id = str(doc.get("_id") or "").strip()
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        couchdb.delete_doc(doc_id, rev=doc.get("_rev"))
        deleted += 1
    return deleted


def _rename_case_docs(old_case_id: str, new_case_id: str) -> int:
    """Rename ``case_id`` for all matching docs and return updated count."""

    moved = 0
    seen: set[str] = set()
    for doc in couchdb.find({"case_id": old_case_id}, limit=10000):
        doc_id = str(doc.get("_id") or "").strip()
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        doc["case_id"] = new_case_id
        if doc.get("type") == "case":
            doc["updated_at"] = _utc_now_iso()
            doc["last_action"] = "rename"
        couchdb.update_doc(doc)
        moved += 1
    for doc in memory_couchdb.find({"case_id": old_case_id}, limit=10000):
        doc_id = str(doc.get("_id") or "").strip()
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        doc["case_id"] = new_case_id
        memory_couchdb.update_doc(doc)
        moved += 1
    return moved


def _case_has_docs(case_id: str) -> bool:
    """Return True when any docs exist for a case id."""

    try:
        if len(couchdb.find({"case_id": case_id}, limit=1)) > 0:
            return True
    except Exception:
        pass
    try:
        return len(memory_couchdb.find({"case_id": case_id}, limit=1)) > 0
    except Exception:
        return False


def _clone_case_contents(source_case_id: str, target_case_id: str) -> int:
    """Clone case-scoped content docs from source to target case id."""

    cloned = 0
    for doc in couchdb.find({"case_id": source_case_id}, limit=10000):
        if doc.get("type") != "deposition":
            continue
        copy_doc = {key: value for key, value in doc.items() if key not in ("_id", "_rev")}
        copy_doc["case_id"] = target_case_id
        couchdb.save_doc(copy_doc)
        cloned += 1
    for doc in memory_couchdb.find({"type": "case_memory", "case_id": source_case_id}, limit=10000):
        copy_doc = {key: value for key, value in doc.items() if key not in ("_id", "_rev")}
        copy_doc["case_id"] = target_case_id
        memory_couchdb.save_doc(copy_doc)
        cloned += 1
    return cloned


def _list_case_versions(case_id: str) -> list[CaseVersionSummary]:
    """List saved case versions sorted by newest-first."""

    versions: list[CaseVersionSummary] = []
    for doc in couchdb.find({"type": "case_version", "case_id": case_id}, limit=10000):
        try:
            versions.append(
                CaseVersionSummary(
                    case_id=case_id,
                    version=int(doc.get("version", 0) or 0),
                    created_at=str(doc.get("created_at") or ""),
                    directory=str(doc.get("directory") or ""),
                    llm_provider=doc.get("llm_provider"),
                    llm_model=str(doc.get("llm_model") or ""),
                    snapshot=doc.get("snapshot") if isinstance(doc.get("snapshot"), dict) else {},
                )
            )
        except Exception:
            continue
    versions = [item for item in versions if item.version >= 1 and item.created_at and item.directory]
    versions.sort(key=lambda item: (item.version, item.created_at), reverse=True)
    return versions


def _save_case_version(request: SaveCaseVersionRequest) -> CaseVersionSummary:
    """Persist a case snapshot as the next version and return it."""

    case_id = request.case_id.strip()
    if not case_id:
        raise HTTPException(status_code=400, detail="Case ID is required")
    source_case_id = str(request.source_case_id or "").strip()
    directory = request.directory.strip()
    if not directory:
        raise HTTPException(status_code=400, detail="Directory is required")

    if source_case_id and source_case_id != case_id:
        if _case_has_docs(case_id):
            raise HTTPException(
                status_code=409,
                detail=f"Cannot create case '{case_id}': case ID already exists",
            )
        cloned_count = _clone_case_contents(source_case_id, case_id)
        if cloned_count <= 0:
            raise HTTPException(
                status_code=404,
                detail=f"Cannot create case '{case_id}': source case '{source_case_id}' has no clonable content",
            )

    versions = _list_case_versions(case_id)
    next_version = (versions[0].version if versions else 0) + 1
    created_at = _utc_now_iso()
    snapshot = request.snapshot if isinstance(request.snapshot, dict) else {}

    try:
        couchdb.save_doc(
            {
                "type": "case_version",
                "case_id": case_id,
                "version": next_version,
                "created_at": created_at,
                "directory": directory,
                "llm_provider": request.llm_provider,
                "llm_model": request.llm_model,
                "snapshot": jsonable_encoder(snapshot),
            }
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to save version for case '{case_id}': {exc}",
        ) from exc

    _upsert_case_doc(
        case_id,
        deposition_count=len(_safe_case_depositions(case_id)),
        last_action="save_version",
        last_directory=directory,
        llm_provider=request.llm_provider,
        llm_model=request.llm_model,
    )

    return CaseVersionSummary(
        case_id=case_id,
        version=next_version,
        created_at=created_at,
        directory=directory,
        llm_provider=request.llm_provider,
        llm_model=request.llm_model,
        snapshot=snapshot,
    )


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


def _render_themed_admin_test_report(html: str) -> str:
    """Inject Admin-specific theme overrides into the pytest HTML report."""

    if 'id="admin-report-theme"' in html:
        return html
    if "</head>" in html:
        return html.replace("</head>", f"{_ADMIN_REPORT_THEME_CSS}</head>", 1)
    return f"{_ADMIN_REPORT_THEME_CSS}{html}"


def _extract_test_report_log_output() -> AdminTestLogResponse:
    """Collect explicit per-test log output from pytest HTML report when available."""

    if not tests_report_file.is_file():
        return AdminTestLogResponse(
            summary="tests.html is not available.",
            log_output="Run the test suite to regenerate /reports/tests.html before viewing test log output.",
        )

    html = tests_report_file.read_text(encoding="utf-8", errors="replace")
    match = re.search(r'data-jsonblob="([^"]+)"', html)
    if not match:
        return AdminTestLogResponse(
            summary="No structured test metadata was found in tests.html.",
            log_output="The pytest HTML report did not expose a data-jsonblob payload to parse.",
        )

    try:
        payload = json.loads(unescape(match.group(1)))
    except Exception:
        return AdminTestLogResponse(
            summary="The embedded test metadata could not be parsed.",
            log_output="The Admin/Test log parser could not decode the tests.html data-jsonblob content.",
        )

    tests = payload.get("tests") if isinstance(payload.get("tests"), dict) else {}
    visible_logs: list[str] = []
    explicit_log_count = 0
    total_runs = 0
    for test_id, runs in tests.items():
        if not isinstance(runs, list):
            continue
        for run in runs:
            if not isinstance(run, dict):
                continue
            total_runs += 1
            raw_log = str(run.get("log") or "").strip()
            if not raw_log or raw_log == "No log output captured.":
                continue
            explicit_log_count += 1
            result = str(run.get("result") or "Unknown").strip() or "Unknown"
            duration = str(run.get("duration") or "").strip()
            header = f"[{result}] {test_id}"
            if duration:
                header = f"{header} ({duration})"
            visible_logs.append(f"{header}\n{raw_log}")

    if visible_logs:
        return AdminTestLogResponse(
            summary=(
                f"Collected explicit log output for {explicit_log_count} of {total_runs} recorded test runs "
                "from tests.html."
            ),
            log_output="\n\n".join(visible_logs),
        )

    return AdminTestLogResponse(
        summary=f"Parsed {total_runs} recorded test runs from tests.html.",
        log_output="No explicit per-test log output was captured in the current pytest HTML report.",
    )


def _list_admin_users() -> list[AdminUserResponse]:
    """List lightweight admin user records from CouchDB."""

    users: list[AdminUserResponse] = []
    for doc in couchdb.find({"type": "admin_user"}, limit=1000):
        try:
            user_id = str(doc.get("_id") or "").strip()
            name = str(doc.get("name") or "").strip()
            created_at = str(doc.get("created_at") or "").strip()
            if not user_id or not name or not created_at:
                continue
            users.append(AdminUserResponse(user_id=user_id, name=name, created_at=created_at))
        except Exception:
            continue
    users.sort(key=lambda item: (item.created_at, item.user_id), reverse=True)
    return users


def _save_admin_user(name: str) -> AdminUserResponse:
    """Persist one lightweight admin user record in CouchDB."""

    normalized_name = name.strip()
    if not normalized_name:
        raise HTTPException(status_code=400, detail="User name is required")

    created_at = _utc_now_iso()
    stored = couchdb.save_doc(
        {
            "type": "admin_user",
            "name": normalized_name,
            "created_at": created_at,
        }
    )
    user_id = str(stored.get("_id") or "").strip()
    return AdminUserResponse(user_id=user_id, name=normalized_name, created_at=created_at)


@app.get("/admin/test-report", response_model=None)
def admin_test_report() -> FileResponse | HTMLResponse:
    """Serve the generated pytest HTML report for the Admin/Test view."""

    if tests_report_file.is_file():
        html = tests_report_file.read_text(encoding="utf-8", errors="replace")
        return HTMLResponse(_render_themed_admin_test_report(html))
    return HTMLResponse(
        (
            "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<title>tests.html unavailable</title></head><body>"
            "<h1>tests.html is not available</h1>"
            "<p>Run the test suite to regenerate /reports/tests.html.</p>"
            "</body></html>"
        ),
        status_code=404,
    )


@app.get("/api/admin/test-log", response_model=AdminTestLogResponse)
def get_admin_test_log() -> AdminTestLogResponse:
    """Return collected pytest log output for the Admin/Test panel."""

    return _extract_test_report_log_output()


@app.get("/api/admin/users", response_model=AdminUserListResponse)
def list_admin_users() -> AdminUserListResponse:
    """List saved admin users for the Admin/Test panel."""

    return AdminUserListResponse(users=_list_admin_users())


@app.post("/api/admin/users", response_model=AdminUserResponse)
def add_admin_user(request: AdminUserRequest) -> AdminUserResponse:
    """Create one lightweight admin user record."""

    return _save_admin_user(request.name)


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


@app.get("/api/deposition-directories", response_model=DepositionDirectoriesResponse)
def get_deposition_directories() -> DepositionDirectoriesResponse:
    """Return valid ingestion directories for the UI folder navigator."""

    base_directory, options = _list_deposition_directories()
    return DepositionDirectoriesResponse(
        base_directory=str(base_directory),
        suggested=_suggest_directory_option(options),
        options=options,
    )


@app.post("/api/depositions/upload", response_model=DepositionUploadResponse)
def upload_depositions(
    directory: str = Form(...),
    files: list[UploadFile] = File(...),
) -> DepositionUploadResponse:
    """Save uploaded deposition text files into the selected deposition folder."""

    target_directory = _resolve_upload_directory(directory)
    if not files:
        raise HTTPException(status_code=400, detail="At least one deposition file is required")

    saved_files: list[str] = []
    for index, upload in enumerate(files, start=1):
        original_name = str(upload.filename or "").strip()
        if Path(original_name).suffix.lower() != ".txt":
            raise HTTPException(
                status_code=400,
                detail=f"Uploaded deposition must be a .txt file: {original_name or 'unnamed file'}",
            )
        safe_name = _sanitize_uploaded_deposition_filename(original_name, index)
        destination = target_directory / safe_name
        try:
            content = upload.file.read()
            destination.write_bytes(content)
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to save uploaded deposition '{safe_name}': {exc}",
            ) from exc
        finally:
            try:
                upload.file.close()
            except Exception:
                pass
        saved_files.append(str(destination))

    return DepositionUploadResponse(
        directory=str(target_directory),
        saved_files=saved_files,
        file_count=len(saved_files),
    )


@app.post("/api/deposition-roots", response_model=DepositionRootResponse)
def add_deposition_root(request: DepositionRootRequest) -> DepositionRootResponse:
    """Validate and cache one additional deposition root path.

    The path must be accessible from the API runtime filesystem. When the API
    runs inside Docker, this means the host path must be bind-mounted into the
    container first.
    """

    raw_path = request.path.strip()
    if not raw_path:
        raise HTTPException(status_code=400, detail="Deposition root path is required")

    try:
        normalized = _normalize_deposition_root_path(raw_path)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid deposition root path '{raw_path}': {exc}",
        ) from exc

    path_text = str(normalized)
    if not normalized.exists():
        raise HTTPException(
            status_code=400,
            detail=(
                f"Deposition root does not exist in API runtime: {path_text}. "
                "If running with Docker, mount the host folder into the api service and retry."
            ),
        )
    if not normalized.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"Deposition root must be a directory: {path_text}",
        )

    try:
        next(normalized.iterdir(), None)
    except PermissionError as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Deposition root is not readable: {path_text}. "
                "Adjust filesystem permissions and retry."
            ),
        ) from exc

    try:
        _cache_deposition_root(path_text)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to cache deposition root '{path_text}': {exc}",
        ) from exc

    return DepositionRootResponse(path=path_text)


@app.get("/api/graph-rag/browser", response_model=GraphBrowserResponse)
def graph_browser_info() -> GraphBrowserResponse:
    """Return Neo4j Browser URL and Bolt endpoint details for graph exploration."""

    browser_url = neo4j_graph.browser_url
    bolt_url = neo4j_graph.uri
    database = neo4j_graph.database
    return GraphBrowserResponse(
        browser_url=browser_url,
        bolt_url=bolt_url,
        database=database,
        launch_url=_graph_browser_launch_url(browser_url, bolt_url, database),
    )


@app.get("/api/graph-rag/health", response_model=GraphHealthResponse)
def graph_rag_health() -> GraphHealthResponse:
    """Return Neo4j configuration/connectivity status for Graph RAG readiness checks."""

    return GraphHealthResponse.model_validate(neo4j_graph.health())


@app.get("/api/graph-rag/owl-options", response_model=GraphOntologyOptionsResponse)
def graph_rag_owl_options() -> GraphOntologyOptionsResponse:
    """Return selectable OWL file path options for Graph RAG ontology loading."""

    base, options, suggested = _list_ontology_owl_options()
    return GraphOntologyOptionsResponse(
        base_directory=str(base),
        suggested=suggested,
        options=options,
    )


@app.get("/api/graph-rag/owl-browser", response_model=GraphOntologyBrowserResponse)
def graph_rag_owl_browser(path: str | None = None) -> GraphOntologyBrowserResponse:
    """Return one level of ontology directory/file rows for file-browser style UI."""

    base, directory = _resolve_ontology_browser_directory(path)
    directories, files = _list_ontology_browser_entries(directory)
    parent_directory = str(directory.parent) if directory != base else None
    return GraphOntologyBrowserResponse(
        base_directory=str(base),
        current_directory=str(directory),
        parent_directory=parent_directory,
        wildcard_path=str(directory / "*.owl"),
        directories=directories,
        files=files,
    )


@app.post("/api/graph-rag/load-owl", response_model=GraphOntologyLoadResponse)
def load_graph_rag_ontology(request: GraphOntologyLoadRequest) -> GraphOntologyLoadResponse:
    """Load one or more OWL ontology files into Neo4j for Graph RAG indexing."""

    path = request.path.strip()
    if not path:
        raise HTTPException(status_code=400, detail="Ontology path is required")

    files = _resolve_ontology_owl_files(path)
    try:
        stats = neo4j_graph.load_owl_files(
            files,
            clear_existing=request.clear_existing,
            batch_size=request.batch_size,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to import ontology files into Neo4j: {exc}",
        ) from exc

    return GraphOntologyLoadResponse(path=path, **stats)


@app.post("/api/graph-rag/query", response_model=GraphRagQueryResponse)
def query_graph_rag(request: GraphRagQueryRequest) -> GraphRagQueryResponse:
    """Answer one question using Neo4j ontology retrieval as the RAG context source."""

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Graph RAG question is required")

    trace_id = str(request.thought_stream_id or request.trace_id or "").strip() or None
    use_rag = bool(request.use_rag)
    stream_rag = bool(request.stream_rag)
    llm_provider, llm_model = _resolve_request_llm(request.llm_provider, request.llm_model)
    _ensure_request_llm_operational(llm_provider, llm_model)
    _append_trace_events(
        trace_id,
        status="running",
        legal_clerk=[
            {
                "persona": "Persona:Legal Clerk",
                "phase": "graph_rag_retrieval_start",
                "llm_provider": llm_provider,
                "llm_model": llm_model,
                "input_preview": question,
                "notes": (
                    f"Starting Graph RAG cycle with top_k={request.top_k}. "
                    f"use_rag={use_rag}."
                ),
            }
        ],
    )

    if use_rag:
        try:
            retrieval = neo4j_graph.retrieve_context(question, node_limit=request.top_k)
        except ValueError as exc:
            _append_trace_events(
                trace_id,
                status="failed",
                attorney=[
                    {
                        "persona": "Persona:Attorney",
                        "phase": "graph_rag_retrieval_error",
                        "llm_provider": llm_provider,
                        "llm_model": llm_model,
                        "notes": str(exc),
                    }
                ],
            )
            if stream_rag:
                _append_rag_stream_event(
                    trace_id=trace_id,
                    question=question,
                    llm_provider=llm_provider,
                    llm_model=llm_model,
                    use_rag=use_rag,
                    top_k=request.top_k,
                    status="failed",
                    phase="retrieval",
                    error=str(exc),
                )
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            _append_trace_events(
                trace_id,
                status="failed",
                attorney=[
                    {
                        "persona": "Persona:Attorney",
                        "phase": "graph_rag_retrieval_error",
                        "llm_provider": llm_provider,
                        "llm_model": llm_model,
                        "notes": str(exc),
                    }
                ],
            )
            if stream_rag:
                _append_rag_stream_event(
                    trace_id=trace_id,
                    question=question,
                    llm_provider=llm_provider,
                    llm_model=llm_model,
                    use_rag=use_rag,
                    top_k=request.top_k,
                    status="failed",
                    phase="retrieval",
                    error=str(exc),
                )
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            _append_trace_events(
                trace_id,
                status="failed",
                attorney=[
                    {
                        "persona": "Persona:Attorney",
                        "phase": "graph_rag_retrieval_error",
                        "llm_provider": llm_provider,
                        "llm_model": llm_model,
                        "notes": str(exc),
                    }
                ],
            )
            if stream_rag:
                _append_rag_stream_event(
                    trace_id=trace_id,
                    question=question,
                    llm_provider=llm_provider,
                    llm_model=llm_model,
                    use_rag=use_rag,
                    top_k=request.top_k,
                    status="failed",
                    phase="retrieval",
                    error=str(exc),
                )
            raise HTTPException(
                status_code=502,
                detail=f"Failed to retrieve graph context from Neo4j: {exc}",
            ) from exc
    else:
        retrieval = {
            "resource_count": 0,
            "resources": [],
            "terms": [],
            "context_text": "RAG processing was disabled for this request.",
        }
        _append_trace_events(
            trace_id,
            legal_clerk=[
                {
                    "persona": "Persona:Legal Clerk",
                    "phase": "graph_rag_disabled",
                    "llm_provider": llm_provider,
                    "llm_model": llm_model,
                    "notes": "RAG retrieval skipped because use_rag=false.",
                }
            ],
        )

    system_prompt = render_prompt("graph_rag_system")
    user_prompt = render_prompt(
        "graph_rag_user",
        question=question,
        context_text=retrieval.get("context_text", ""),
    )
    _append_trace_events(
        trace_id,
        legal_clerk=[
            {
                "persona": "Persona:Legal Clerk",
                "phase": "graph_rag_context_ready",
                "llm_provider": llm_provider,
                "llm_model": llm_model,
                "input_preview": (
                    f"retrieval_terms={retrieval.get('terms', [])}\n"
                    f"context_rows={retrieval.get('resource_count', 0)}"
                ),
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "notes": "Prepared graph context and prompts for LLM inference.",
            }
        ],
    )

    llm = build_chat_model(settings, llm_provider, llm_model, temperature=0.1)
    try:
        result = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
    except Exception as exc:
        _append_trace_events(
            trace_id,
            status="failed",
            attorney=[
                {
                    "persona": "Persona:Attorney",
                    "phase": "graph_rag_llm_error",
                    "llm_provider": llm_provider,
                    "llm_model": llm_model,
                    "notes": str(exc),
                }
            ],
        )
        if stream_rag:
            _append_rag_stream_event(
                trace_id=trace_id,
                question=question,
                llm_provider=llm_provider,
                llm_model=llm_model,
                use_rag=use_rag,
                top_k=request.top_k,
                status="failed",
                phase="llm",
                retrieval_terms=[str(item) for item in retrieval.get("terms", [])],
                context_rows=int(retrieval.get("resource_count") or 0),
                sources=[item for item in retrieval.get("resources", []) if isinstance(item, dict)],
                context_preview=str(retrieval.get("context_text", ""))[:9000],
                llm_system_prompt=system_prompt,
                llm_user_prompt=user_prompt,
                error=str(exc),
                retrieved_resources=[item for item in retrieval.get("resources", []) if isinstance(item, dict)],
            )
        raise HTTPException(
            status_code=502,
            detail=llm_failure_message(settings, llm_provider, llm_model, exc),
        ) from exc

    answer = str(getattr(result, "content", "") or "").strip()
    if not answer:
        answer = "Short answer: No answer could be generated from current ontology context."

    sources = [
        GraphRagSource(iri=str(item.get("iri") or ""), label=str(item.get("label") or ""))
        for item in retrieval.get("resources", [])
        if str(item.get("iri") or "").strip()
    ]
    retrieved_resources = [
        GraphRagRetrievedResource(
            iri=str(item.get("iri") or ""),
            label=str(item.get("label") or item.get("iri") or ""),
            relations=[
                GraphRagRelation(
                    predicate=str(rel.get("predicate") or ""),
                    object_label=str(rel.get("object_label") or rel.get("object_iri") or ""),
                    object_iri=str(rel.get("object_iri") or ""),
                )
                for rel in (item.get("relations") or [])
                if isinstance(rel, dict) and str(rel.get("object_iri") or "").strip()
            ],
            literals=[
                GraphRagLiteral(
                    predicate=str(literal.get("predicate") or ""),
                    value=str(literal.get("value") or ""),
                    datatype=str(literal.get("datatype") or ""),
                    lang=str(literal.get("lang") or ""),
                )
                for literal in (item.get("literals") or [])
                if isinstance(literal, dict) and str(literal.get("value") or "").strip()
            ],
        )
        for item in retrieval.get("resources", [])
        if isinstance(item, dict) and str(item.get("iri") or "").strip()
    ]
    monitor = GraphRagMonitor(
        rag_enabled=use_rag,
        rag_stream_enabled=stream_rag,
        retrieval_terms=[str(item) for item in retrieval.get("terms", [])],
        retrieved_resources=retrieved_resources,
        context_preview=str(retrieval.get("context_text", ""))[:9000],
        llm_system_prompt=system_prompt,
        llm_user_prompt=user_prompt,
    )
    _append_trace_events(
        trace_id,
        status="completed",
        attorney=[
            {
                "persona": "Persona:Attorney",
                "phase": "graph_rag_answer",
                "llm_provider": llm_provider,
                "llm_model": llm_model,
                "output_preview": answer,
                "notes": f"Graph RAG completed with {len(sources)} source node(s).",
            }
        ],
    )
    if stream_rag:
        _append_rag_stream_event(
            trace_id=trace_id,
            question=question,
            llm_provider=llm_provider,
            llm_model=llm_model,
            use_rag=use_rag,
            top_k=request.top_k,
            status="completed",
            phase="answer",
            retrieval_terms=monitor.retrieval_terms,
            context_rows=int(retrieval.get("resource_count") or 0),
            sources=[{"iri": item.iri, "label": item.label} for item in sources],
            context_preview=monitor.context_preview,
            llm_system_prompt=monitor.llm_system_prompt,
            llm_user_prompt=monitor.llm_user_prompt,
            answer_preview=answer,
            retrieved_resources=[item.model_dump() for item in monitor.retrieved_resources],
        )
    return GraphRagQueryResponse(
        question=question,
        answer=answer,
        context_rows=int(retrieval.get("resource_count") or 0),
        sources=sources,
        llm_provider=llm_provider,
        llm_model=llm_model,
        monitor=monitor,
    )


@app.get("/api/thought-streams/health")
def thought_stream_health() -> dict[str, str | bool]:
    """Validate thought-stream CouchDB connectivity for UI toggle plumbing checks."""

    try:
        trace_couchdb.ensure_db(retries=1, delay_seconds=0)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Thought Stream storage is unavailable for database "
                f"'{settings.thought_stream_db}': {exc}"
            ),
        ) from exc

    return {
        "connected": True,
        "database": settings.thought_stream_db,
    }


@app.get("/api/rag-streams/health")
def rag_stream_health() -> dict[str, str | bool]:
    """Validate rag-stream CouchDB connectivity for Graph RAG logging."""

    try:
        rag_couchdb.ensure_db(retries=1, delay_seconds=0)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"RAG stream storage is unavailable for database "
                f"'{settings.rag_stream_db}': {exc}"
            ),
        ) from exc

    return {
        "connected": True,
        "database": settings.rag_stream_db,
    }


@app.get("/api/agent-metrics", response_model=AgentRuntimeMetricsResponse)
def get_agent_metrics(lookback_hours: int = 24) -> AgentRuntimeMetricsResponse:
    """Return runtime KPI metrics for monitoring running agent/LLM behavior."""

    if lookback_hours < 1 or lookback_hours > 168:
        raise HTTPException(status_code=400, detail="lookback_hours must be between 1 and 168")
    sessions, storage_connected = _collect_runtime_trace_sessions()
    rag_events, rag_storage_connected = _collect_recent_rag_stream_events(lookback_hours)
    return _compute_agent_runtime_metrics(
        sessions,
        lookback_hours=lookback_hours,
        storage_connected=storage_connected,
        rag_events=rag_events,
        rag_storage_connected=rag_storage_connected,
    )


@app.get("/api/agent-metrics/history")
def get_agent_metric_history(
    metric_key: str,
    lookback_hours: int = 24,
    bucket_hours: int = 2,
) -> dict[str, object]:
    """Return bucketed history for one runtime observable."""

    normalized_metric_key = metric_key.strip()
    if not normalized_metric_key:
        raise HTTPException(status_code=400, detail="metric_key is required")
    if lookback_hours < 1 or lookback_hours > 168:
        raise HTTPException(status_code=400, detail="lookback_hours must be between 1 and 168")
    if bucket_hours < 1 or bucket_hours > 24:
        raise HTTPException(status_code=400, detail="bucket_hours must be between 1 and 24")
    if bucket_hours > lookback_hours:
        raise HTTPException(status_code=400, detail="bucket_hours cannot exceed lookback_hours")

    sessions, storage_connected = _collect_runtime_trace_sessions()
    rag_events, rag_storage_connected = _collect_recent_rag_stream_events(lookback_hours)
    try:
        return _compute_agent_metric_history(
            sessions,
            lookback_hours=lookback_hours,
            bucket_hours=bucket_hours,
            metric_key=normalized_metric_key,
            storage_connected=storage_connected,
            rag_events=rag_events,
            rag_storage_connected=rag_storage_connected,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Metric '{normalized_metric_key}' is not available for trend history",
        ) from exc


@app.get("/api/thought-streams/{thought_stream_id}", response_model=TraceSessionResponse)
def get_trace_session(thought_stream_id: str) -> TraceSessionResponse:
    """Return current thought-stream events for one active/past stream id."""

    snapshot = _trace_session_snapshot(thought_stream_id.strip())
    if snapshot is None:
        raise HTTPException(status_code=404, detail=f"Thought stream '{thought_stream_id}' was not found")
    return TraceSessionResponse.model_validate(snapshot)


@app.post("/api/thought-streams/{thought_stream_id}/save", response_model=SaveTraceResponse)
def save_trace_session(thought_stream_id: str, request: SaveTraceRequest) -> SaveTraceResponse:
    """Persist thought-stream payload into case memory for later audit/review."""

    normalized_trace_id = thought_stream_id.strip()
    case_id = request.case_id.strip()
    if not normalized_trace_id:
        raise HTTPException(status_code=400, detail="Thought stream ID is required")
    if not case_id:
        raise HTTPException(status_code=400, detail="Case ID is required")

    _flush_trace_session(normalized_trace_id, force=True)
    snapshot = _trace_session_snapshot(normalized_trace_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail=f"Thought stream '{normalized_trace_id}' was not found")

    _save_case_memory(
        case_id,
        "thought_stream",
        {
            "thought_stream_id": normalized_trace_id,
            "channel": request.channel,
            "thought_stream": snapshot.get("thought_stream", {}),
            "status": snapshot.get("status"),
            "updated_at": snapshot.get("updated_at"),
        },
    )
    _upsert_case_doc(
        case_id,
        deposition_count=len(_safe_case_depositions(case_id)),
        memory_increment=1,
        last_action="thought_stream_save",
    )
    return SaveTraceResponse(
        thought_stream_id=normalized_trace_id,
        case_id=case_id,
        channel=request.channel,
        saved=True,
    )


@app.delete("/api/thought-streams/{thought_stream_id}", response_model=DeleteTraceResponse)
def delete_trace_session(thought_stream_id: str) -> DeleteTraceResponse:
    """Discard one in-memory thought-stream session."""

    deleted = _delete_trace_session(thought_stream_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Thought stream '{thought_stream_id}' was not found")
    return DeleteTraceResponse(thought_stream_id=thought_stream_id, deleted=True)


@app.get("/api/cases", response_model=CaseListResponse)
def list_cases() -> CaseListResponse:
    """List all known cases for the vertical case index."""

    try:
        return CaseListResponse(cases=_list_case_summaries())
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to list cases: {exc}") from exc


@app.get("/api/cases/{case_id}", response_model=CaseDetailResponse)
def get_case(case_id: str) -> CaseDetailResponse:
    """Return one saved case record with persisted snapshot state."""

    normalized_case_id = case_id.strip()
    if not normalized_case_id:
        raise HTTPException(status_code=400, detail="Case ID is required")

    try:
        payload = _case_detail(normalized_case_id)
        if payload is None:
            raise HTTPException(status_code=404, detail=f"Case '{normalized_case_id}' was not found")
        return payload
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to load case '{normalized_case_id}': {exc}",
        ) from exc


@app.get("/api/cases/{case_id}/versions", response_model=CaseVersionListResponse)
def list_case_versions(case_id: str) -> CaseVersionListResponse:
    """List saved case versions for scrolling and rerun."""

    try:
        versions = _list_case_versions(case_id)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to list versions for case '{case_id}': {exc}",
        ) from exc
    return CaseVersionListResponse(case_id=case_id, versions=versions)


@app.post("/api/cases", response_model=CaseSummary)
def save_case(request: SaveCaseRequest) -> CaseSummary:
    """Create/update a case index item so it appears in the Case Index list."""

    case_id = request.case_id.strip()
    if not case_id:
        raise HTTPException(status_code=400, detail="Case ID is required")

    try:
        _upsert_case_doc(
            case_id,
            deposition_count=len(_safe_case_depositions(case_id)),
            last_action="save",
            last_directory=request.directory,
            llm_provider=request.llm_provider,
            llm_model=request.llm_model,
            snapshot=request.snapshot if isinstance(request.snapshot, dict) else {},
        )
        for item in _list_case_summaries():
            if item.case_id == case_id:
                return item
        return CaseSummary(case_id=case_id, deposition_count=len(_safe_case_depositions(case_id)))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to save case '{case_id}': {exc}") from exc


@app.post("/api/cases/version", response_model=CaseVersionSummary)
def save_case_version(request: SaveCaseVersionRequest) -> CaseVersionSummary:
    """Save a version snapshot for the selected case."""

    return _save_case_version(request)


@app.put("/api/cases/{case_id}/rename", response_model=RenameCaseResponse)
def rename_case(case_id: str, request: RenameCaseRequest) -> RenameCaseResponse:
    """Rename case id across all persisted case/deposition/memory docs."""

    old_case_id = case_id.strip()
    new_case_id = request.new_case_id.strip()
    if not old_case_id or not new_case_id:
        raise HTTPException(status_code=400, detail="Old and new case IDs are required")
    if old_case_id == new_case_id:
        return RenameCaseResponse(old_case_id=old_case_id, new_case_id=new_case_id, moved_docs=0)

    try:
        if couchdb.find({"case_id": new_case_id}, limit=1):
            raise HTTPException(
                status_code=409,
                detail=f"Cannot rename to '{new_case_id}': that case ID already exists",
            )

        moved = _rename_case_docs(old_case_id, new_case_id)
        if moved <= 0:
            raise HTTPException(status_code=404, detail=f"Case '{old_case_id}' was not found")

        _upsert_case_doc(
            new_case_id,
            deposition_count=len(_safe_case_depositions(new_case_id)),
            last_action="rename",
        )
        return RenameCaseResponse(
            old_case_id=old_case_id,
            new_case_id=new_case_id,
            moved_docs=moved,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to rename case '{old_case_id}' to '{new_case_id}': {exc}",
        ) from exc


@app.delete("/api/cases/{case_id}", response_model=DeleteCaseResponse)
def delete_case(case_id: str) -> DeleteCaseResponse:
    """Delete all docs belonging to a case id."""

    try:
        deleted = _delete_case_docs(case_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to delete case '{case_id}': {exc}") from exc
    return DeleteCaseResponse(case_id=case_id, deleted_docs=deleted)


@app.delete("/api/cases/{case_id}/depositions", response_model=ClearCaseDepositionsResponse)
def clear_case_depositions(case_id: str) -> ClearCaseDepositionsResponse:
    """Remove all deposition docs for one case while preserving case metadata."""

    target_case_id = case_id.strip()
    if not target_case_id:
        raise HTTPException(status_code=400, detail="Case ID is required")

    try:
        deleted = _delete_case_deposition_docs(target_case_id)
        _upsert_case_doc(
            target_case_id,
            deposition_count=0,
            last_action="refresh",
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to refresh case '{target_case_id}': {exc}",
        ) from exc

    return ClearCaseDepositionsResponse(
        case_id=target_case_id,
        deleted_depositions=deleted,
    )


@app.post("/api/ingest-case", response_model=IngestCaseResponse)
def ingest_case(request: IngestCaseRequest) -> IngestCaseResponse:
    """Ingest all deposition text files for a case and return updated scores."""

    llm_provider, llm_model = _resolve_request_llm(request.llm_provider, request.llm_model)
    _ensure_request_llm_operational(llm_provider, llm_model)
    txt_files = _resolve_ingest_txt_files(request.directory)
    _purge_stale_case_depositions(request.case_id, txt_files)
    _purge_noncanonical_case_depositions(request.case_id, txt_files)
    trace_id = str(request.thought_stream_id or request.trace_id or "").strip() or None
    _append_trace_events(
        trace_id,
        status="running",
        legal_clerk=[
            {
                "persona": "Persona:Legal Clerk",
                "phase": "ingest_start",
                "llm_provider": llm_provider,
                "llm_model": llm_model,
                "notes": f"Starting ingest for {len(txt_files)} file(s).",
                "input_preview": (
                    f"case_id={request.case_id}\n"
                    f"directory={request.directory}\n"
                    f"skip_reassess={request.skip_reassess}"
                ),
            }
        ],
    )

    results: list[IngestedDepositionResult] = []
    ingested_ids: list[str] = []
    memory_count = 0
    legal_clerk_trace: list[AgentTraceEvent] = []
    attorney_trace: list[AgentTraceEvent] = []
    for file_path in txt_files:
        _append_trace_events(
            trace_id,
            legal_clerk=[
                {
                    "persona": "Persona:Legal Clerk",
                    "phase": "ingest_file_start",
                    "file_name": file_path.name,
                    "llm_provider": llm_provider,
                    "llm_model": llm_model,
                    "notes": f"Reading and mapping {file_path.name}.",
                }
            ],
        )
        try:
            state = workflow.run(
                case_id=request.case_id,
                file_path=str(file_path),
                llm_provider=llm_provider,
                llm_model=llm_model,
            )
        except Exception as exc:
            _append_trace_events(
                trace_id,
                status="failed",
                attorney=[
                    {
                        "persona": "Persona:Attorney",
                        "phase": "ingest_error",
                        "file_name": file_path.name,
                        "llm_provider": llm_provider,
                        "llm_model": llm_model,
                        "notes": str(exc),
                    }
                ],
            )
            raise HTTPException(
                status_code=502,
                detail=f"Failed to process deposition file '{file_path.name}': {exc}",
            ) from exc
        doc = state["deposition_doc"]
        ingested_ids.append(doc["_id"])
        legal_clerk_trace.extend(
            [AgentTraceEvent.model_validate(item) for item in state.get("legal_clerk_trace", [])]
        )
        attorney_trace.extend(
            [AgentTraceEvent.model_validate(item) for item in state.get("attorney_trace", [])]
        )
        _append_trace_events(
            trace_id,
            legal_clerk=state.get("legal_clerk_trace", []),
            attorney=state.get("attorney_trace", []),
        )
        _save_case_memory(
            request.case_id,
            "langgraph",
            {
                "phase": "ingest_file",
                "file_name": file_path.name,
                "llm_provider": llm_provider,
                "llm_model": llm_model,
                "state": state,
            },
        )
        memory_count += 1

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

    _upsert_case_doc(
        request.case_id,
        deposition_count=len(_safe_case_depositions(request.case_id)),
        memory_increment=memory_count,
        last_action="ingest",
        last_directory=request.directory,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )
    _append_trace_events(
        trace_id,
        status="completed",
        attorney=[
            {
                "persona": "Persona:Attorney",
                "phase": "ingest_complete",
                "llm_provider": llm_provider,
                "llm_model": llm_model,
                "notes": (
                    f"Completed ingest for {len(results)} file(s). "
                    f"skip_reassess={request.skip_reassess}."
                ),
            }
        ],
    )

    return IngestCaseResponse(
        case_id=request.case_id,
        ingested=results,
        thought_stream=AgentTracePayload(
            legal_clerk=legal_clerk_trace,
            attorney=attorney_trace,
        ),
    )


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
    trace_id = str(request.thought_stream_id or request.trace_id or "").strip() or None
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
    _append_trace_events(
        trace_id,
        status="running",
        attorney=[
            {
                "persona": "Persona:Attorney",
                "phase": "chat_start",
                "file_name": str(deposition.get("file_name") or ""),
                "llm_provider": llm_provider,
                "llm_model": llm_model,
                "notes": "Attorney chat started.",
                "input_preview": (
                    f"message={request.message}\n"
                    f"history_items={len(request.history)}\n"
                    f"peer_count={len(peers)}"
                ),
            }
        ],
    )
    try:
        if hasattr(type(chat_service), "respond_with_trace"):
            response, trace_items = chat_service.respond_with_trace(
                deposition,
                peers,
                request.message,
                request.history,
                llm_provider=llm_provider,
                llm_model=llm_model,
            )
        else:
            response = chat_service.respond(
                deposition,
                peers,
                request.message,
                request.history,
                llm_provider=llm_provider,
                llm_model=llm_model,
            )
            trace_items = []
    except Exception as exc:
        _append_trace_events(
            trace_id,
            status="failed",
            attorney=[
                {
                    "persona": "Persona:Attorney",
                    "phase": "chat_error",
                    "file_name": str(deposition.get("file_name") or ""),
                    "llm_provider": llm_provider,
                    "llm_model": llm_model,
                    "notes": str(exc),
                }
            ],
        )
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    _append_trace_events(trace_id, attorney=trace_items, status="completed")
    _save_case_memory(
        request.case_id,
        "chat",
        {
            "deposition_id": request.deposition_id,
            "message": request.message,
            "response": response,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "thought_stream": trace_items,
        },
    )
    _upsert_case_doc(
        request.case_id,
        deposition_count=len(_safe_case_depositions(request.case_id)),
        memory_increment=1,
        last_action="chat",
        llm_provider=llm_provider,
        llm_model=llm_model,
    )
    return ChatResponse(
        response=response,
        thought_stream=AgentTracePayload(attorney=[AgentTraceEvent.model_validate(item) for item in trace_items]),
    )


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
    _save_case_memory(
        request.case_id,
        "reason",
        {
            "deposition_id": request.deposition_id,
            "contradiction": request.contradiction.model_dump(),
            "response": response,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
        },
    )
    _upsert_case_doc(
        request.case_id,
        deposition_count=len(_safe_case_depositions(request.case_id)),
        memory_increment=1,
        last_action="reason",
        llm_provider=llm_provider,
        llm_model=llm_model,
    )
    return ContradictionReasonResponse(response=response)


@app.post("/api/summarize-focused-reasoning", response_model=FocusedReasoningSummaryResponse)
def summarize_focused_reasoning(
    request: FocusedReasoningSummaryRequest,
) -> FocusedReasoningSummaryResponse:
    """Summarize focused contradiction analysis while preserving selected-case validation."""

    llm_provider, llm_model = _resolve_request_llm(request.llm_provider, request.llm_model)
    _ensure_request_llm_operational(llm_provider, llm_model)
    try:
        deposition = couchdb.get_doc(request.deposition_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail="Deposition not found") from exc
    if deposition.get("case_id") != request.case_id:
        raise HTTPException(status_code=400, detail="Deposition does not belong to requested case")

    try:
        summary = chat_service.summarize_focused_reasoning(
            reasoning_text=request.reasoning_text,
            llm_provider=llm_provider,
            llm_model=llm_model,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    _save_case_memory(
        request.case_id,
        "reason_summary",
        {
            "deposition_id": request.deposition_id,
            "reasoning_text": request.reasoning_text,
            "summary": summary,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
        },
    )
    _upsert_case_doc(
        request.case_id,
        deposition_count=len(_safe_case_depositions(request.case_id)),
        memory_increment=1,
        last_action="reason_summary",
        llm_provider=llm_provider,
        llm_model=llm_model,
    )
    return FocusedReasoningSummaryResponse(summary=summary)


@app.post("/api/deposition-sentiment", response_model=DepositionSentimentResponse)
def deposition_sentiment(request: DepositionSentimentRequest) -> DepositionSentimentResponse:
    """Compute whole-document sentiment for the selected deposition."""

    try:
        deposition = couchdb.get_doc(request.deposition_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail="Deposition not found") from exc
    if deposition.get("case_id") != request.case_id:
        raise HTTPException(status_code=400, detail="Deposition does not belong to requested case")

    response = _compute_deposition_sentiment(
        str(deposition.get("raw_text") or deposition.get("summary") or "")
    )
    _save_case_memory(
        request.case_id,
        "sentiment",
        {
            "deposition_id": request.deposition_id,
            "score": response.score,
            "label": response.label,
            "summary": response.summary,
            "positive_matches": response.positive_matches,
            "negative_matches": response.negative_matches,
            "word_count": response.word_count,
        },
    )
    _upsert_case_doc(
        request.case_id,
        deposition_count=len(_safe_case_depositions(request.case_id)),
        memory_increment=1,
        last_action="sentiment",
    )
    return response
