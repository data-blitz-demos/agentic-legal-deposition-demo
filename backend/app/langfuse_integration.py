"""Centralized Langfuse helpers for optional runtime tracing."""

from __future__ import annotations

from contextlib import contextmanager
from contextlib import nullcontext
import logging
from typing import Any

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
    normalized_metadata = _normalize_metadata(metadata)
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
    }
