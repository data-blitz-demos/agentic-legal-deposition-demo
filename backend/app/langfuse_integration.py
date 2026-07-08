"""Centralized Langfuse helpers for optional runtime tracing."""

from __future__ import annotations

from contextlib import contextmanager
from contextlib import nullcontext
import json
import logging
import re
from typing import Any

import httpx

from .config import Settings

try:  # pragma: no cover - exercised indirectly when dependency is installed
    from langfuse import Langfuse, get_client, propagate_attributes
    from langfuse.langchain import CallbackHandler
except Exception:  # pragma: no cover - graceful fallback when dependency is absent
    Langfuse = None
    CallbackHandler = None
    get_client = None
    propagate_attributes = None

logger = logging.getLogger(__name__)
_LANGFUSE_INITIALIZED = False
_COMMON_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}
_TRACE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{8,128}$")


def langfuse_sdk_installed() -> bool:
    """Return whether the Langfuse SDK is importable in this runtime."""

    return bool(Langfuse and CallbackHandler and get_client and propagate_attributes)


def langfuse_enabled(settings: Settings) -> bool:
    """Return whether Langfuse tracing is configured and available."""

    if not bool(getattr(settings, "langfuse_enabled", False)):
        return False
    if not langfuse_sdk_installed():
        return False
    return bool(
        str(getattr(settings, "langfuse_public_key", "") or "").strip()
        and str(getattr(settings, "langfuse_secret_key", "") or "").strip()
        and str(getattr(settings, "langfuse_base_url", "") or "").strip()
    )


def initialize_langfuse(settings: Settings) -> bool:
    """Initialize the Langfuse singleton once for the current process."""

    global _LANGFUSE_INITIALIZED
    if _LANGFUSE_INITIALIZED:
        return True
    if not langfuse_enabled(settings):
        return False
    try:
        Langfuse(
            public_key=str(getattr(settings, "langfuse_public_key", "")).strip(),
            secret_key=str(getattr(settings, "langfuse_secret_key", "")).strip(),
            base_url=str(getattr(settings, "langfuse_base_url", "")).strip(),
        )
        _LANGFUSE_INITIALIZED = True
        return True
    except Exception:
        logger.exception("langfuse initialization failed")
        return False


def shutdown_langfuse(settings: Settings) -> None:
    """Flush and shut down Langfuse background workers if initialized."""

    if not _LANGFUSE_INITIALIZED or not langfuse_enabled(settings):
        return
    try:
        get_client().shutdown()
    except Exception:
        logger.exception("langfuse shutdown failed")


def flush_langfuse(settings: Settings) -> bool:
    """Flush pending Langfuse events to storage when the SDK is active."""

    if not initialize_langfuse(settings):
        return False
    client = get_client()
    flush = getattr(client, "flush", None)
    if not callable(flush):
        return False
    try:
        flush()
        return True
    except Exception:
        logger.exception("langfuse flush failed")
        return False


def _trim_string(value: str | None, limit: int = 200) -> str | None:
    """Return a stripped string truncated to Langfuse-safe length."""

    normalized = str(value or "").strip()
    if not normalized:
        return None
    return normalized[:limit]


def _normalize_tags(tags: list[str] | tuple[str, ...] | set[str] | None) -> list[str] | None:
    """Return de-duplicated, non-empty tags suitable for Langfuse metadata."""

    if not tags:
        return None
    seen: set[str] = set()
    normalized: list[str] = []
    for item in tags:
        tag = _trim_string(str(item or ""), limit=64)
        if not tag or tag in seen:
            continue
        seen.add(tag)
        normalized.append(tag)
    return normalized or None


def _normalize_metadata(metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    """Drop empty metadata values while preserving structured payloads."""

    if not isinstance(metadata, dict):
        return None
    normalized: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, str):
            trimmed = value.strip()
            if not trimmed:
                continue
            normalized[str(key)] = trimmed
            continue
        normalized[str(key)] = value
    return normalized or None


def _normalize_propagated_metadata(metadata: dict[str, Any] | None) -> dict[str, str] | None:
    """Return Langfuse propagation metadata with string-only values."""

    normalized = _normalize_metadata(metadata)
    if not normalized:
        return None
    propagated: dict[str, str] = {}
    for key, value in normalized.items():
        if isinstance(value, str):
            propagated[str(key)] = value
        else:
            propagated[str(key)] = _trim_string(json.dumps(value, sort_keys=True), limit=200) or str(value)
    return propagated or None


def _normalize_trace_id(trace_id: str | None) -> str | None:
    """Return one safe Langfuse trace id suitable for SDK and query usage."""

    normalized = _trim_string(trace_id, limit=128)
    if not normalized or not _TRACE_ID_PATTERN.fullmatch(normalized):
        return None
    return normalized


def _sql_string_literal(value: str | None) -> str | None:
    """Return one ClickHouse-safe SQL string literal for a normalized value."""

    normalized = _trim_string(value, limit=512)
    if normalized is None:
        return None
    escaped = normalized.replace("'", "''")
    return "'" + escaped + "'"


def get_current_trace_id(settings: Settings) -> str | None:
    """Return the active Langfuse trace id when one is currently bound."""

    if not initialize_langfuse(settings):
        return None
    client = get_client()
    getter = getattr(client, "get_current_trace_id", None)
    if not callable(getter):
        return None
    try:
        return _normalize_trace_id(getter())
    except Exception:
        logger.exception("langfuse get_current_trace_id failed")
        return None


def resolve_trace_id_from_observation_id(settings: Settings, observation_id: str | None) -> str | None:
    """Resolve the persisted Langfuse trace id for one stored observation id."""

    normalized_observation_id = _normalize_trace_id(observation_id)
    if not normalized_observation_id:
        return None
    observation_id_literal = _sql_string_literal(normalized_observation_id)
    if not observation_id_literal:
        return None
    rows = _clickhouse_json_query(
        settings,
        (
            "SELECT trace_id FROM default.observations "
            f"WHERE id = {observation_id_literal} AND is_deleted = 0 "
            "ORDER BY start_time DESC LIMIT 1 FORMAT JSON"
        ),
    )
    if not rows:
        return None
    return _normalize_trace_id(rows[0].get("trace_id"))


def _clickhouse_json_query(settings: Settings, query: str) -> list[dict[str, Any]]:
    """Run one ClickHouse JSON query against Langfuse storage."""

    query_text = str(query or "").strip()
    if not query_text:
        return []
    base_url = str(getattr(settings, "langfuse_clickhouse_url", "") or "").strip()
    if not base_url:
        return []
    user = str(getattr(settings, "langfuse_clickhouse_user", "") or "").strip()
    password = str(getattr(settings, "langfuse_clickhouse_password", "") or "")
    auth = (user, password) if user else None
    response = httpx.post(
        base_url,
        params={"query": query_text},
        auth=auth,
        timeout=15.0,
    )
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("data")
    return rows if isinstance(rows, list) else []


def fetch_trace_graph(settings: Settings, trace_id: str) -> dict[str, Any] | None:
    """Fetch one Langfuse trace plus its observations and scores from ClickHouse."""

    normalized_trace_id = _normalize_trace_id(trace_id)
    if not normalized_trace_id:
        return None
    trace_id_literal = _sql_string_literal(normalized_trace_id)
    if not trace_id_literal:
        return None
    trace_rows = _clickhouse_json_query(
        settings,
        (
            "SELECT * FROM default.traces "
            f"WHERE id = {trace_id_literal} AND is_deleted = 0 "
            "ORDER BY timestamp DESC LIMIT 1 FORMAT JSON"
        ),
    )
    if not trace_rows:
        return None
    observations = _clickhouse_json_query(
        settings,
        (
            "SELECT * FROM default.observations "
            f"WHERE trace_id = {trace_id_literal} AND is_deleted = 0 "
            "ORDER BY start_time ASC FORMAT JSON"
        ),
    )
    scores = _clickhouse_json_query(
        settings,
        (
            "SELECT * FROM default.scores "
            f"WHERE trace_id = {trace_id_literal} AND is_deleted = 0 "
            "ORDER BY timestamp ASC FORMAT JSON"
        ),
    )
    return {
        "trace": trace_rows[0],
        "observations": observations,
        "scores": scores,
    }


@contextmanager
def observe_operation(
    settings: Settings,
    name: str,
    *,
    as_type: str = "span",
    session_id: str | None = None,
    user_id: str | None = None,
    tags: list[str] | tuple[str, ...] | set[str] | None = None,
    metadata: dict[str, Any] | None = None,
):
    """Create one optional Langfuse observation around an application operation."""

    if not initialize_langfuse(settings):
        yield None
        return

    propagate_kwargs: dict[str, Any] = {}
    normalized_session_id = _trim_string(session_id)
    normalized_user_id = _trim_string(user_id)
    normalized_tags = _normalize_tags(tags)
    normalized_metadata = _normalize_propagated_metadata(metadata)
    if normalized_session_id:
        propagate_kwargs["session_id"] = normalized_session_id
    if normalized_user_id:
        propagate_kwargs["user_id"] = normalized_user_id
    if normalized_tags:
        propagate_kwargs["tags"] = normalized_tags
    if normalized_metadata:
        propagate_kwargs["metadata"] = normalized_metadata

    try:
        propagation_context = propagate_attributes(**propagate_kwargs) if propagate_kwargs else nullcontext()
        observation_context = get_client().start_as_current_observation(as_type=as_type, name=name)
    except Exception:
        logger.exception("langfuse observation failed name=%s", name)
        yield None
        return

    with propagation_context:
        with observation_context as observation:
            yield observation


@contextmanager
def propagate_trace_attributes(
    settings: Settings,
    *,
    session_id: str | None = None,
    user_id: str | None = None,
    tags: list[str] | tuple[str, ...] | set[str] | None = None,
    metadata: dict[str, Any] | None = None,
    trace_name: str | None = None,
):
    """Propagate Langfuse trace attributes onto the current trace/span context."""

    if not initialize_langfuse(settings):
        yield
        return

    propagate_kwargs: dict[str, Any] = {}
    normalized_session_id = _trim_string(session_id)
    normalized_user_id = _trim_string(user_id)
    normalized_tags = _normalize_tags(tags)
    normalized_metadata = _normalize_propagated_metadata(metadata)
    normalized_trace_name = _trim_string(trace_name)
    if normalized_session_id:
        propagate_kwargs["session_id"] = normalized_session_id
    if normalized_user_id:
        propagate_kwargs["user_id"] = normalized_user_id
    if normalized_tags:
        propagate_kwargs["tags"] = normalized_tags
    if normalized_metadata:
        propagate_kwargs["metadata"] = normalized_metadata
    if normalized_trace_name:
        propagate_kwargs["trace_name"] = normalized_trace_name

    if not propagate_kwargs:
        yield
        return

    try:
        propagation_context = propagate_attributes(**propagate_kwargs)
    except Exception:
        logger.exception("langfuse propagate_trace_attributes failed")
        yield
        return

    with propagation_context:
        yield


def build_langchain_config(
    settings: Settings,
    *,
    operation: str,
    session_id: str | None = None,
    user_id: str | None = None,
    tags: list[str] | tuple[str, ...] | set[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return LangChain invoke config with Langfuse callback wiring when enabled."""

    if not initialize_langfuse(settings):
        return {}

    config_metadata = _normalize_metadata(metadata) or {}
    normalized_session_id = _trim_string(session_id)
    normalized_user_id = _trim_string(user_id)
    normalized_tags = _normalize_tags(tags)

    if normalized_session_id:
        config_metadata["langfuse_session_id"] = normalized_session_id
    if normalized_user_id:
        config_metadata["langfuse_user_id"] = normalized_user_id
    if normalized_tags:
        config_metadata["langfuse_tags"] = normalized_tags
    config_metadata.setdefault("operation", operation)

    return {
        "callbacks": [CallbackHandler()],
        "metadata": config_metadata,
        "tags": normalized_tags or [],
    }


def _word_count(value: str | None) -> int:
    """Return a rough token-free word count for lightweight scoring heuristics."""

    return len(re.findall(r"\b[\w'-]+\b", str(value or "")))


def _contains_placeholder_text(value: str | None) -> bool:
    """Detect template-like placeholders that indicate low-quality output."""

    text = str(value or "")
    lowered = text.lower()
    if any(marker in lowered for marker in ("<1 sentence>", "<bullet>", "short answer: <")):
        return True
    return re.search(r"<[^>\n]{3,}>", text) is not None


def _response_has_structure(value: str | None) -> bool:
    """Return whether a response includes recognizable app-level structure markers."""

    lowered = str(value or "").lower()
    return any(
        marker in lowered
        for marker in (
            "short answer:",
            "details:",
            "key points:",
            "recommended action:",
            "why this matters:",
            "\n- ",
        )
    )


def _content_terms(value: str | None) -> set[str]:
    """Extract a small normalized set of content-bearing words."""

    terms = {
        term.lower()
        for term in re.findall(r"\b[a-zA-Z][a-zA-Z0-9_-]{2,}\b", str(value or ""))
        if term.lower() not in _COMMON_STOPWORDS
    }
    return terms


def _clamp01(value: float) -> float:
    """Clamp a float into the closed 0..1 interval."""

    return max(0.0, min(float(value), 1.0))


def _to_rubric_5(value: float) -> float:
    """Map a normalized 0..1 score into a 1..5 rubric scale."""

    return round(1.0 + (_clamp01(value) * 4.0), 2)


def _overall_quality_label(value: float) -> str:
    """Map a normalized quality score into a mainstream categorical band."""

    normalized = _clamp01(value)
    if normalized < 0.25:
        return "poor"
    if normalized < 0.5:
        return "fair"
    if normalized < 0.8:
        return "good"
    return "excellent"


def _prompt_quality_heuristic(prompt_text: str | None) -> float:
    """Return a simple 0..1 heuristic for prompt specificity and clarity."""

    text = str(prompt_text or "").strip()
    if not text:
        return 0.0

    words = _word_count(text)
    chars = len(text)
    lowered = text.lower()
    intent_markers = (
        "who",
        "what",
        "when",
        "where",
        "why",
        "how",
        "compare",
        "identify",
        "explain",
        "summarize",
        "analyze",
        "reason",
    )
    score = 0.0
    score += min(words / 18.0, 1.0) * 0.45
    score += min(chars / 180.0, 1.0) * 0.2
    score += 0.2 if "?" in text or any(marker in lowered for marker in intent_markers) else 0.0
    score += 0.15 if any(marker in lowered for marker in ("witness", "timeline", "contract", "contradiction")) else 0.0
    return round(min(score, 1.0), 4)


def _prompt_clarity_heuristic(prompt_text: str | None) -> float:
    """Approximate prompt clarity using question framing and concise structure."""

    text = str(prompt_text or "").strip()
    if not text:
        return 0.0
    words = _word_count(text)
    lowered = text.lower()
    score = 0.2
    score += 0.2 if "?" in text else 0.0
    score += 0.2 if any(marker in lowered for marker in ("summarize", "compare", "explain", "identify", "analyze", "reason")) else 0.0
    score += 0.2 if 4 <= words <= 60 else 0.1 if words <= 90 else 0.0
    score += 0.2 if len(_content_terms(text)) >= 3 else 0.0
    return round(_clamp01(score), 4)


def _prompt_specificity_heuristic(prompt_text: str | None) -> float:
    """Approximate prompt specificity using lexical richness and domain signals."""

    text = str(prompt_text or "").strip()
    if not text:
        return 0.0
    content_terms = _content_terms(text)
    lowered = text.lower()
    score = 0.0
    score += min(len(content_terms) / 10.0, 1.0) * 0.45
    score += 0.2 if re.search(r"\b\d+\b", text) else 0.0
    score += 0.2 if any(marker in lowered for marker in ("witness", "deposition", "timeline", "contract", "contradiction", "exhibit")) else 0.0
    score += 0.15 if any(char in text for char in (":", "\"", "'")) else 0.0
    return round(_clamp01(score), 4)


def _response_quality_heuristic(response_text: str | None) -> float:
    """Return a simple 0..1 heuristic for response completeness and formatting."""

    text = str(response_text or "").strip()
    if not text:
        return 0.0

    words = _word_count(text)
    lowered = text.lower()
    score = 0.35
    score += 0.2 if not _contains_placeholder_text(text) else 0.0
    score += 0.2 if _response_has_structure(text) else 0.0
    score += min(words / 90.0, 1.0) * 0.15
    score += 0.1 if any(marker in lowered for marker in ("next step", "recommended action", "why this matters")) else 0.0
    return round(min(score, 1.0), 4)


def _response_relevance_heuristic(prompt_text: str | None, response_text: str | None) -> float:
    """Approximate prompt-response topical relevance via content-term overlap."""

    prompt_terms = _content_terms(prompt_text)
    response_terms = _content_terms(response_text)
    if not prompt_terms or not response_terms:
        return 0.0
    overlap = len(prompt_terms & response_terms)
    score = overlap / max(1, min(len(prompt_terms), len(response_terms)))
    return round(_clamp01(score), 4)


def _response_completeness_heuristic(response_text: str | None) -> float:
    """Approximate response completeness using structure, length, and placeholders."""

    text = str(response_text or "").strip()
    if not text:
        return 0.0
    words = _word_count(text)
    score = 0.15
    score += 0.25 if _response_has_structure(text) else 0.0
    score += 0.2 if not _contains_placeholder_text(text) else 0.0
    score += min(words / 120.0, 1.0) * 0.25
    score += 0.15 if any(marker in text.lower() for marker in ("details:", "key points:", "recommended action:", "why this matters:")) else 0.0
    return round(_clamp01(score), 4)


def _response_helpfulness_heuristic(prompt_text: str | None, response_text: str | None) -> float:
    """Approximate helpfulness from relevance, completeness, and actionability."""

    relevance = _response_relevance_heuristic(prompt_text, response_text)
    completeness = _response_completeness_heuristic(response_text)
    actionability = 1.0 if _response_actionable(response_text) else 0.0
    score = (relevance * 0.35) + (completeness * 0.4) + (actionability * 0.25)
    return round(_clamp01(score), 4)


def _response_actionable(response_text: str | None) -> bool:
    """Return whether the response appears to suggest a next step or action."""

    lowered = str(response_text or "").lower()
    return any(
        marker in lowered
        for marker in (
            "next step",
            "recommended action",
            "should",
            "compare",
            "review",
            "verify",
            "use the full",
        )
    )


def score_current_trace(
    settings: Settings,
    *,
    name: str,
    value: float | str,
    data_type: str | None = None,
    comment: str | None = None,
) -> bool:
    """Attach one score to the active Langfuse trace and span when available.

    Langfuse stores trace-level scores correctly, but the UI often surfaces the
    active span/observation first. Writing to both contexts keeps the score tree
    visible in the trace overview and inside the current request observation.
    """

    if not initialize_langfuse(settings):
        return False

    client = get_client()
    trace_scorer = getattr(client, "score_current_trace", None)
    span_scorer = getattr(client, "score_current_span", None)
    if not callable(trace_scorer) and not callable(span_scorer):
        return False

    payload: dict[str, Any] = {
        "name": name,
        "value": value,
    }
    if data_type:
        payload["data_type"] = data_type
    if comment:
        payload["comment"] = comment

    succeeded = False

    if callable(trace_scorer):
        try:
            trace_scorer(**payload)
            succeeded = True
        except Exception:
            logger.exception("langfuse score_current_trace failed name=%s", name)

    if callable(span_scorer):
        try:
            span_scorer(**payload)
            succeeded = True
        except Exception:
            logger.exception("langfuse score_current_span failed name=%s", name)

    return succeeded


def _normalize_mcp_tool_name(tool_name: str | None) -> str:
    """Normalize one MCP tool key into a stable Langfuse namespace segment."""

    normalized = re.sub(r"[^a-z0-9_]+", "_", str(tool_name or "").strip().lower()).strip("_") or "tool"
    if normalized.startswith("mcp_"):
        normalized = normalized[4:] or "tool"
    return normalized


def _normalize_metric_name(metric_name: str | None) -> str:
    """Normalize one score key into a safe Langfuse metric suffix."""

    return re.sub(r"[^a-z0-9_]+", "_", str(metric_name or "").strip().lower()).strip("_")


def score_tagged_event(
    settings: Settings,
    *,
    category: str,
    event_name: str,
    metrics: dict[str, Any],
) -> int:
    """Attach generic tagged-event metrics to the active Langfuse trace/span.

    Metric names are emitted as ``<category>.<event_name>.<metric>`` so prompt,
    memory, and skill scores can share the same normalization behavior as MCP
    metrics while still living in separate namespaces.
    """

    category_segment = _normalize_metric_name(category) or "event"
    event_segment = _normalize_metric_name(event_name) or "operation"
    emitted = 0

    for raw_key, raw_value in dict(metrics or {}).items():
        metric_segment = _normalize_metric_name(raw_key)
        if not metric_segment or raw_value is None:
            continue

        score_name = f"{category_segment}.{event_segment}.{metric_segment}"
        comment = (
            f"Tagged event metric for category={category_segment} "
            f"event={event_segment} metric={metric_segment}."
        )

        if isinstance(raw_value, bool):
            emitted += int(
                score_current_trace(
                    settings,
                    name=score_name,
                    value=1.0 if raw_value else 0.0,
                    data_type="BOOLEAN",
                    comment=comment,
                )
            )
        elif isinstance(raw_value, (int, float)):
            emitted += int(
                score_current_trace(
                    settings,
                    name=score_name,
                    value=float(raw_value),
                    data_type="NUMERIC",
                    comment=comment,
                )
            )
        else:
            text_value = _trim_string(str(raw_value or ""), limit=200)
            if not text_value:
                continue
            emitted += int(
                score_current_trace(
                    settings,
                    name=score_name,
                    value=text_value,
                    data_type="CATEGORICAL",
                    comment=comment,
                )
            )

    return emitted


def score_mcp_operation(
    settings: Settings,
    *,
    tool_name: str,
    operation: str,
    metrics: dict[str, Any],
) -> int:
    """Attach MCP tool metrics to the active Langfuse trace/span.

    Metric names are emitted as ``mcp.<tool>.<operation>.<metric>`` so Langfuse
    can group the same tool surface across health checks, reads, writes, and
    retrieval requests.
    """

    tool_segment = _normalize_mcp_tool_name(tool_name)
    operation_segment = _normalize_metric_name(operation) or "operation"
    emitted = 0

    for raw_key, raw_value in dict(metrics or {}).items():
        metric_segment = _normalize_metric_name(raw_key)
        if not metric_segment or raw_value is None:
            continue

        score_name = f"mcp.{tool_segment}.{operation_segment}.{metric_segment}"
        comment = (
            f"MCP metric for tool={tool_segment} operation={operation_segment} "
            f"metric={metric_segment}."
        )

        if isinstance(raw_value, bool):
            emitted += int(
                score_current_trace(
                    settings,
                    name=score_name,
                    value=1.0 if raw_value else 0.0,
                    data_type="BOOLEAN",
                    comment=comment,
                )
            )
        elif isinstance(raw_value, (int, float)):
            emitted += int(
                score_current_trace(
                    settings,
                    name=score_name,
                    value=float(raw_value),
                    data_type="NUMERIC",
                    comment=comment,
                )
            )
        else:
            text_value = _trim_string(str(raw_value or ""), limit=200)
            if not text_value:
                continue
            emitted += int(
                score_current_trace(
                    settings,
                    name=score_name,
                    value=text_value,
                    data_type="CATEGORICAL",
                    comment=comment,
                )
            )

    return emitted


def score_user_prompt_and_response(
    settings: Settings,
    *,
    operation: str,
    prompt_text: str | None,
    response_text: str | None,
) -> dict[str, float]:
    """Score one user-facing prompt/response pair on the active Langfuse trace."""

    prompt_words = float(_word_count(prompt_text))
    response_words = float(_word_count(response_text))
    prompt_quality = _prompt_quality_heuristic(prompt_text)
    prompt_clarity = _prompt_clarity_heuristic(prompt_text)
    prompt_specificity = _prompt_specificity_heuristic(prompt_text)
    response_quality = _response_quality_heuristic(response_text)
    response_relevance = _response_relevance_heuristic(prompt_text, response_text)
    response_completeness = _response_completeness_heuristic(response_text)
    response_helpfulness = _response_helpfulness_heuristic(prompt_text, response_text)
    response_structured = 1.0 if _response_has_structure(response_text) else 0.0
    response_placeholder_free = 0.0 if _contains_placeholder_text(response_text) else 1.0
    response_actionable = 1.0 if _response_actionable(response_text) else 0.0
    overall_response_quality = round(
        _clamp01(
            (response_quality * 0.3)
            + (response_relevance * 0.25)
            + (response_completeness * 0.25)
            + (response_helpfulness * 0.2),
        ),
        4,
    )
    overall_response_quality_label = _overall_quality_label(overall_response_quality)
    overall_quality = round(
        _clamp01(
            (prompt_quality * 0.15)
            + (prompt_clarity * 0.1)
            + (prompt_specificity * 0.1)
            + (response_quality * 0.2)
            + (response_relevance * 0.15)
            + (response_completeness * 0.15)
            + (response_helpfulness * 0.15),
        ),
        4,
    )
    overall_quality_label = _overall_quality_label(overall_quality)

    comment_prefix = f"Heuristic score for operation={operation}."
    score_current_trace(
        settings,
        name="user_prompt_word_count",
        value=prompt_words,
        data_type="NUMERIC",
        comment=f"{comment_prefix} Word count of the user-provided prompt text.",
    )
    score_current_trace(
        settings,
        name="user_prompt_quality_heuristic",
        value=prompt_quality,
        data_type="NUMERIC",
        comment=f"{comment_prefix} 0..1 heuristic based on prompt length and intent markers.",
    )
    score_current_trace(
        settings,
        name="prompt_clarity_rubric",
        value=_to_rubric_5(prompt_clarity),
        data_type="NUMERIC",
        comment=f"{comment_prefix} 1..5 rubric approximation of prompt clarity.",
    )
    score_current_trace(
        settings,
        name="prompt_specificity_rubric",
        value=_to_rubric_5(prompt_specificity),
        data_type="NUMERIC",
        comment=f"{comment_prefix} 1..5 rubric approximation of prompt specificity.",
    )
    score_current_trace(
        settings,
        name="response_word_count",
        value=response_words,
        data_type="NUMERIC",
        comment=f"{comment_prefix} Word count of the final response returned to the user.",
    )
    score_current_trace(
        settings,
        name="response_quality_heuristic",
        value=response_quality,
        data_type="NUMERIC",
        comment=f"{comment_prefix} 0..1 heuristic based on completeness, structure, and placeholder checks.",
    )
    score_current_trace(
        settings,
        name="response_relevance_rubric",
        value=_to_rubric_5(response_relevance),
        data_type="NUMERIC",
        comment=f"{comment_prefix} 1..5 rubric approximation of response relevance to the prompt.",
    )
    score_current_trace(
        settings,
        name="response_helpfulness_rubric",
        value=_to_rubric_5(response_helpfulness),
        data_type="NUMERIC",
        comment=f"{comment_prefix} 1..5 rubric approximation of response helpfulness.",
    )
    score_current_trace(
        settings,
        name="response_completeness_rubric",
        value=_to_rubric_5(response_completeness),
        data_type="NUMERIC",
        comment=f"{comment_prefix} 1..5 rubric approximation of response completeness.",
    )
    score_current_trace(
        settings,
        name="response_has_structure",
        value=response_structured,
        data_type="BOOLEAN",
        comment=f"{comment_prefix} Whether the response includes expected structured sections or bullets.",
    )
    score_current_trace(
        settings,
        name="response_placeholder_free",
        value=response_placeholder_free,
        data_type="BOOLEAN",
        comment=f"{comment_prefix} Whether the response avoided template placeholders.",
    )
    score_current_trace(
        settings,
        name="response_actionable",
        value=response_actionable,
        data_type="BOOLEAN",
        comment=f"{comment_prefix} Whether the response appears to include a concrete next step or action.",
    )
    score_current_trace(
        settings,
        name="overall_response_quality",
        value=_to_rubric_5(overall_response_quality),
        data_type="NUMERIC",
        comment=f"{comment_prefix} 1..5 aggregate rubric across response quality, relevance, completeness, and helpfulness.",
    )
    score_current_trace(
        settings,
        name="overall_response_quality_label",
        value=overall_response_quality_label,
        data_type="CATEGORICAL",
        comment=f"{comment_prefix} Aggregate categorical band for response quality only.",
    )
    score_current_trace(
        settings,
        name="overall_quality_rubric",
        value=_to_rubric_5(overall_quality),
        data_type="NUMERIC",
        comment=f"{comment_prefix} 1..5 aggregate rubric across prompt and response quality dimensions.",
    )
    score_current_trace(
        settings,
        name="overall_quality_label",
        value=overall_quality_label,
        data_type="CATEGORICAL",
        comment=f"{comment_prefix} Aggregate categorical band for the overall trace quality.",
    )

    return {
        "user_prompt_word_count": prompt_words,
        "user_prompt_quality_heuristic": prompt_quality,
        "prompt_clarity_rubric": _to_rubric_5(prompt_clarity),
        "prompt_specificity_rubric": _to_rubric_5(prompt_specificity),
        "response_word_count": response_words,
        "response_quality_heuristic": response_quality,
        "response_relevance_rubric": _to_rubric_5(response_relevance),
        "response_helpfulness_rubric": _to_rubric_5(response_helpfulness),
        "response_completeness_rubric": _to_rubric_5(response_completeness),
        "response_has_structure": response_structured,
        "response_placeholder_free": response_placeholder_free,
        "response_actionable": response_actionable,
        "overall_response_quality": _to_rubric_5(overall_response_quality),
        "overall_quality_rubric": _to_rubric_5(overall_quality),
    }


def score_prompt_operation(
    settings: Settings,
    *,
    operation: str,
    prompt_text: str | None,
    response_text: str | None,
    system_prompt: str | None = None,
    user_prompt: str | None = None,
    prompt_template_keys: list[str] | tuple[str, ...] | set[str] | None = None,
    extra_metrics: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Attach prompt-specific Langfuse scores for one user-facing prompt event."""

    metrics = score_user_prompt_and_response(
        settings,
        operation=operation,
        prompt_text=prompt_text,
        response_text=response_text,
    )
    tagged_metrics: dict[str, Any] = {
        "system_prompt_chars": _word_count(system_prompt) * 0 + len(str(system_prompt or "")),
        "user_prompt_chars": _word_count(user_prompt) * 0 + len(str(user_prompt or "")),
        "input_prompt_chars": len(str(prompt_text or "")),
        "response_chars": len(str(response_text or "")),
        "template_count": len([item for item in (prompt_template_keys or []) if str(item).strip()]),
        "template_keys": ",".join(
            sorted({str(item).strip() for item in (prompt_template_keys or []) if str(item).strip()})
        ),
        "has_system_prompt": bool(str(system_prompt or "").strip()),
        "has_user_prompt": bool(str(user_prompt or "").strip()),
    }
    if isinstance(extra_metrics, dict):
        tagged_metrics.update(extra_metrics)
    score_tagged_event(
        settings,
        category="prompt",
        event_name=operation,
        metrics=tagged_metrics,
    )
    return metrics


def score_memory_operation(
    settings: Settings,
    *,
    channel: str,
    case_id: str | None,
    payload: dict[str, Any] | None,
    saved: bool,
) -> int:
    """Attach case-memory write metrics to the current Langfuse trace/span."""

    normalized_payload = dict(payload or {})
    thought_stream = normalized_payload.get("thought_stream")
    return score_tagged_event(
        settings,
        category="memory",
        event_name=channel,
        metrics={
            "saved": saved,
            "case_id_present": bool(str(case_id or "").strip()),
            "payload_field_count": len(normalized_payload),
            "payload_bytes": len(str(normalized_payload).encode("utf-8")),
            "has_response": bool(str(normalized_payload.get("response") or normalized_payload.get("summary") or "").strip()),
            "has_thought_stream": isinstance(thought_stream, list) and bool(thought_stream),
            "thought_stream_events": len(thought_stream) if isinstance(thought_stream, list) else 0,
            "channel": channel,
        },
    )


def score_skill_operation(
    settings: Settings,
    *,
    skill_name: str,
    metrics: dict[str, Any],
) -> int:
    """Attach skill/workflow-level metrics to the current Langfuse trace/span."""

    return score_tagged_event(
        settings,
        category="skill",
        event_name=skill_name,
        metrics=metrics,
    )


def _metric_field(metric: Any, field: str, default: Any = None) -> Any:
    """Read one field from either dict-style or attribute-style metric rows."""

    if isinstance(metric, dict):
        return metric.get(field, default)
    return getattr(metric, field, default)


def _normalize_observable_group_name(group_name: str | None) -> str:
    """Normalize observable group labels into a clean Langfuse namespace."""

    normalized = re.sub(r"[^a-z0-9_]+", "_", str(group_name or "").strip().lower()) or "group"
    aliases = {
        "mcp_tool": "mcp",
        "mcp_tools": "mcp",
    }
    return aliases.get(normalized, normalized)


def score_observable_dashboard(
    settings: Settings,
    *,
    operation: str,
    metric_groups: dict[str, list[Any]],
    summary: dict[str, Any],
) -> int:
    """Attach observables-dashboard metrics to the current Langfuse trace.

    Numeric observables are emitted under ``observables.<group>.<key>`` and each
    observable also emits a categorical status score so unavailable metrics still
    appear in Langfuse.
    """

    emitted = 0
    comment_prefix = f"Heuristic observables snapshot for operation={operation}."

    for key, value in summary.items():
        name = f"observables.summary.{str(key).strip()}"
        if not name.endswith("."):
            if isinstance(value, bool):
                emitted += int(
                    score_current_trace(
                        settings,
                        name=name,
                        value=1.0 if value else 0.0,
                        data_type="BOOLEAN",
                        comment=f"{comment_prefix} Summary boolean for {key}.",
                    )
                )
            elif isinstance(value, (int, float)):
                emitted += int(
                    score_current_trace(
                        settings,
                        name=name,
                        value=float(value),
                        data_type="NUMERIC",
                        comment=f"{comment_prefix} Summary numeric for {key}.",
                    )
                )
            elif value is not None:
                emitted += int(
                    score_current_trace(
                        settings,
                        name=name,
                        value=str(value),
                        data_type="CATEGORICAL",
                        comment=f"{comment_prefix} Summary label for {key}.",
                    )
                )

    for group_name, metrics in metric_groups.items():
        normalized_group = _normalize_observable_group_name(group_name)
        for metric in metrics:
            metric_key = str(_metric_field(metric, "key", "") or "").strip()
            if not metric_key:
                continue
            metric_display = str(_metric_field(metric, "display", "N/A") or "N/A")
            metric_status = str(_metric_field(metric, "status", "info") or "info")
            metric_value = _metric_field(metric, "value", None)
            metric_base = f"observables.{normalized_group}.{metric_key}"

            if isinstance(metric_value, bool):
                emitted += int(
                    score_current_trace(
                        settings,
                        name=metric_base,
                        value=1.0 if metric_value else 0.0,
                        data_type="BOOLEAN",
                        comment=f"{comment_prefix} Boolean observable value for {metric_key}.",
                    )
                )
            elif isinstance(metric_value, (int, float)):
                emitted += int(
                    score_current_trace(
                        settings,
                        name=metric_base,
                        value=float(metric_value),
                        data_type="NUMERIC",
                        comment=f"{comment_prefix} Numeric observable value for {metric_key}.",
                    )
                )

            emitted += int(
                score_current_trace(
                    settings,
                    name=f"{metric_base}_status",
                    value=metric_status,
                    data_type="CATEGORICAL",
                    comment=f"{comment_prefix} Current status band for {metric_key}.",
                )
            )
            emitted += int(
                score_current_trace(
                    settings,
                    name=f"{metric_base}_display",
                    value=metric_display,
                    data_type="CATEGORICAL",
                    comment=f"{comment_prefix} Current rendered display value for {metric_key}.",
                )
            )

    return emitted
