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
from datetime import datetime, timezone
from glob import glob
import hashlib
from pathlib import Path
import re
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
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
    CaseListResponse,
    CaseSummary,
    CaseVersionListResponse,
    CaseVersionSummary,
    ClearCaseDepositionsResponse,
    DeleteCaseResponse,
    DepositionDirectoriesResponse,
    DepositionDirectoryOption,
    DepositionRootRequest,
    DepositionRootResponse,
    ContradictionReasonRequest,
    ContradictionReasonResponse,
    IngestCaseRequest,
    IngestCaseResponse,
    IngestedDepositionResult,
    LLMOption,
    LLMOptionsResponse,
    RenameCaseRequest,
    RenameCaseResponse,
    SaveCaseRequest,
    SaveCaseVersionRequest,
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
        couchdb.save_doc(
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
    return moved


def _case_has_docs(case_id: str) -> bool:
    """Return True when any docs exist for a case id."""

    try:
        return len(couchdb.find({"case_id": case_id}, limit=1)) > 0
    except Exception:
        return False


def _clone_case_contents(source_case_id: str, target_case_id: str) -> int:
    """Clone case-scoped content docs from source to target case id."""

    cloned = 0
    for doc in couchdb.find({"case_id": source_case_id}, limit=10000):
        if doc.get("type") not in ("deposition", "case_memory"):
            continue
        copy_doc = {key: value for key, value in doc.items() if key not in ("_id", "_rev")}
        copy_doc["case_id"] = target_case_id
        couchdb.save_doc(copy_doc)
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


@app.get("/api/cases", response_model=CaseListResponse)
def list_cases() -> CaseListResponse:
    """List all known cases for the vertical case index."""

    try:
        return CaseListResponse(cases=_list_case_summaries())
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to list cases: {exc}") from exc


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

    results: list[IngestedDepositionResult] = []
    ingested_ids: list[str] = []
    memory_count = 0
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
    _save_case_memory(
        request.case_id,
        "chat",
        {
            "deposition_id": request.deposition_id,
            "message": request.message,
            "response": response,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
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
