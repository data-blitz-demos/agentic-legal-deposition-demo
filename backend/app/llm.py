from __future__ import annotations

"""LLM provider selection, readiness validation, and error guidance utilities.

This module is the single place where the app:
- chooses OpenAI vs Ollama models
- validates model availability/operability
- generates user-facing remediation hints when failures occur
"""

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import time

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from .config import Settings

try:
    from langchain_ollama import ChatOllama
except Exception:  # pragma: no cover - handled at runtime via provider selection
    ChatOllama = None

_READINESS_CACHE: dict[tuple[str, str], float] = {}


class LLMOperationalError(RuntimeError):
    """Raised when an LLM is not operational for the requested use."""

    def __init__(self, message: str, possible_fix: str) -> None:
        """Attach an actionable fix alongside the failure message."""

        super().__init__(message)
        self.possible_fix = possible_fix


class _StructuredProbe(BaseModel):
    """Structured readiness probe schema expected from model responses."""

    ready: str


def parse_model_list(raw: str) -> list[str]:
    """Parse comma-separated model names into a trimmed list."""

    values = [item.strip() for item in raw.split(",")]
    return [item for item in values if item]


def _dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    """Return unique items while preserving the first-seen order."""

    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def resolve_llm_selection(
    settings: Settings,
    llm_provider: str | None,
    llm_model: str | None,
) -> tuple[str, str]:
    """Resolve final provider/model using request values and settings defaults."""

    provider = (llm_provider or settings.default_llm_provider or "openai").strip().lower()
    if provider not in {"openai", "ollama"}:
        raise ValueError("Unsupported LLM provider. Use 'openai' or 'ollama'.")

    model = (llm_model or "").strip()
    if model:
        return provider, model

    if provider == "openai":
        configured = (settings.model_name or "").strip()
        if configured:
            return provider, configured
        openai_models = parse_model_list(settings.openai_models)
        return provider, openai_models[0] if openai_models else "gpt-5.2"

    ollama_default = (settings.ollama_default_model or "").strip()
    if ollama_default:
        return provider, ollama_default

    ollama_models = parse_model_list(settings.ollama_models)
    return provider, ollama_models[0] if ollama_models else "llama3.3"


def build_chat_model(
    settings: Settings,
    llm_provider: str | None,
    llm_model: str | None,
    *,
    temperature: float,
):
    """Create a LangChain chat model instance for the selected provider/model."""

    provider, model = resolve_llm_selection(settings, llm_provider, llm_model)
    if provider == "openai":
        return ChatOpenAI(
            model=model,
            api_key=settings.openai_api_key,
            temperature=temperature,
        )

    if ChatOllama is None:
        raise RuntimeError("Ollama provider requires langchain-ollama")

    return ChatOllama(
        model=model,
        base_url=settings.ollama_url,
        temperature=temperature,
        keep_alive=settings.ollama_keep_alive,
    )


def _possible_fix_from_exception(
    settings: Settings,
    provider: str,
    model: str,
    exc: Exception,
) -> str:
    """Map low-level provider errors to concrete user remediation guidance."""

    text = str(exc).lower()

    if provider == "openai":
        if not (settings.openai_api_key or "").strip():
            return "Set OPENAI_API_KEY in .env and restart the API service."
        if "timed out" in text:
            return (
                "OpenAI readiness probe timed out. Check network/API latency or increase "
                "LLM_PROBE_TIMEOUT_SECONDS."
            )
        if "401" in text or "unauthorized" in text or "invalid api key" in text:
            return "Verify OPENAI_API_KEY is valid and has access to the selected model."
        if "404" in text or "model" in text and "not" in text and "found" in text:
            return (
                "Use a valid OpenAI model name (for example gpt-5.2), update OPENAI_MODELS, "
                "and refresh models in the UI."
            )
        if "output_parsing_failure" in text or "failed to parse" in text or "validation error" in text:
            return (
                "Model response schema did not match expected structured output. "
                "Retry the request or switch to a more reliable model variant."
            )
        return "Check OpenAI connectivity and model access, then retry."

    if "timed out" in text:
        return (
            f"Ollama readiness probe timed out after about {settings.llm_probe_timeout_seconds}s. "
            "Model may still be cold-loading. Try Refresh Models again, increase "
            "LLM_PROBE_TIMEOUT_SECONDS, or choose a smaller local model."
        )
    if "connection refused" in text or "failed to establish" in text:
        return (
            f"Ensure Ollama is running and reachable at {settings.ollama_url}. "
            "If API runs in Docker, use OLLAMA_URL=http://host.docker.internal:11434."
        )
    if "not found" in text or "unknown model" in text:
        return f"Run `ollama pull {model}` and click Refresh Models in the UI."
    if "output_parsing_failure" in text or "failed to parse" in text or "validation error" in text:
        return (
            "Model returned non-conformant structured output. "
            "Retry, refresh models, or choose another Ollama model/version."
        )
    return "Verify Ollama is running, the model is pulled, and OLLAMA_URL is correct."


def _normalize_ollama_model_names(values: Iterable[str]) -> set[str]:
    """Normalize Ollama model aliases (with/without ``:latest``)."""

    normalized: set[str] = set()
    for value in values:
        item = value.strip()
        if not item:
            continue
        normalized.add(item)
        if item.endswith(":latest"):
            normalized.add(item[:-7])
        elif ":" not in item:
            normalized.add(f"{item}:latest")
    return normalized


def fetch_ollama_models(settings: Settings) -> list[str]:
    """Fetch installed Ollama model tags from ``/api/tags``."""

    url = f"{settings.ollama_url.rstrip('/')}/api/tags"
    try:
        response = httpx.get(url, timeout=3)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return []

    models: list[str] = []
    for item in payload.get("models", []):
        name = str(item.get("name", "")).strip()
        if name:
            models.append(name)
    return _dedupe_preserve_order(models)


def _assert_ollama_model_loaded(settings: Settings, model: str) -> None:
    """Ensure Ollama is reachable and the requested model exists locally."""

    models = fetch_ollama_models(settings)
    if not models:
        raise LLMOperationalError(
            f"Ollama is not reachable at {settings.ollama_url}.",
            (
                "Start Ollama and verify OLLAMA_URL. "
                "If the API runs in Docker, set OLLAMA_URL=http://host.docker.internal:11434."
            ),
        )

    loaded = _normalize_ollama_model_names(models)
    requested = _normalize_ollama_model_names([model])
    if loaded.intersection(requested):
        return

    raise LLMOperationalError(
        f"Ollama model '{model}' is not loaded. Available models: {', '.join(models)}.",
        f"Run `ollama pull {model}` and click Refresh Models.",
    )


def _probe_structured_output(settings: Settings, provider: str, model: str) -> None:
    """Run a lightweight structured-output probe against the selected model."""

    timeout_seconds = max(1, int(settings.llm_probe_timeout_seconds))

    def _invoke():
        llm = build_chat_model(settings, provider, model, temperature=0)
        parser = llm.with_structured_output(_StructuredProbe)
        return parser.invoke(
            [
                SystemMessage(content="Return strict JSON with key 'ready' and value 'ok'."),
                HumanMessage(content="Ready check"),
            ]
        )

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_invoke)
    try:
        result = future.result(timeout=timeout_seconds)
    except FutureTimeoutError as exc:
        future.cancel()
        raise RuntimeError(f"Readiness probe timed out after {timeout_seconds}s.") from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    ready = str(getattr(result, "ready", "")).strip().lower()
    if not ready:
        raise RuntimeError("Structured readiness probe returned an empty value.")


def clear_llm_readiness_cache() -> None:
    """Clear in-process readiness cache used to reduce repetitive probes."""

    _READINESS_CACHE.clear()


def _readiness_cache_key(provider: str, model: str) -> tuple[str, str]:
    """Build normalized cache key for a provider/model pair."""

    return (provider.strip().lower(), model.strip())


def ensure_llm_operational(
    settings: Settings,
    provider: str,
    model: str,
    *,
    force_probe: bool = False,
) -> None:
    """Validate provider prerequisites and perform readiness probe."""

    if provider == "openai" and not (settings.openai_api_key or "").strip():
        raise LLMOperationalError(
            "OPENAI_API_KEY is not configured.",
            "Set OPENAI_API_KEY in .env and restart the API service.",
        )

    cache_key = _readiness_cache_key(provider, model)
    ttl_seconds = max(0, int(settings.llm_readiness_ttl_seconds))
    if not force_probe and ttl_seconds > 0:
        expires_at = _READINESS_CACHE.get(cache_key, 0.0)
        if expires_at > time.time():
            return

    if provider == "ollama":
        _assert_ollama_model_loaded(settings, model)

    try:
        _probe_structured_output(settings, provider, model)
    except LLMOperationalError:
        _READINESS_CACHE.pop(cache_key, None)
        raise
    except Exception as exc:
        _READINESS_CACHE.pop(cache_key, None)
        raise LLMOperationalError(
            f"Model '{model}' failed readiness probe: {exc}",
            _possible_fix_from_exception(settings, provider, model, exc),
        ) from exc
    if ttl_seconds > 0:
        _READINESS_CACHE[cache_key] = time.time() + ttl_seconds
    else:
        _READINESS_CACHE.pop(cache_key, None)


def _quick_llm_option_check(settings: Settings, provider: str, model: str) -> None:
    """Run fast pre-checks used for non-refresh model dropdown loads."""

    if provider == "openai":
        if not (settings.openai_api_key or "").strip():
            raise LLMOperationalError(
                "OPENAI_API_KEY is not configured.",
                "Set OPENAI_API_KEY in .env and restart the API service.",
            )
        return

    _assert_ollama_model_loaded(settings, model)


def get_llm_option_status(
    settings: Settings,
    provider: str,
    model: str,
    *,
    force_probe: bool = False,
) -> dict[str, str | bool | None]:
    """Return operational status payload used by ``/api/llm-options``."""

    try:
        if force_probe:
            ensure_llm_operational(settings, provider, model, force_probe=True)
        else:
            _quick_llm_option_check(settings, provider, model)
        return {"operational": True, "error": None, "possible_fix": None}
    except LLMOperationalError as exc:
        return {
            "operational": False,
            "error": str(exc),
            "possible_fix": exc.possible_fix,
        }


def llm_failure_message(
    settings: Settings,
    llm_provider: str | None,
    llm_model: str | None,
    exc: Exception,
) -> str:
    """Compose standardized runtime failure text with suggested fix."""

    provider, model = resolve_llm_selection(settings, llm_provider, llm_model)
    fix = _possible_fix_from_exception(settings, provider, model, exc)
    return f"LLM operation failed for {provider}:{model}. Error: {exc}. Possible fix: {fix}"


def list_llm_options(settings: Settings) -> list[dict[str, str]]:
    """List selectable OpenAI/Ollama model options for frontend dropdowns."""

    options: list[dict[str, str]] = []

    openai_models = _dedupe_preserve_order(
        [(settings.model_name or "").strip(), *parse_model_list(settings.openai_models)]
    )
    for model in openai_models:
        if not model:
            continue
        options.append(
            {
                "provider": "openai",
                "model": model,
                "label": f"ChatGPT - {model}",
            }
        )

    ollama_models = fetch_ollama_models(settings)
    if not ollama_models:
        fallback = _dedupe_preserve_order(
            [(settings.ollama_default_model or "").strip(), *parse_model_list(settings.ollama_models)]
        )
        ollama_models = [item for item in fallback if item]

    for model in ollama_models:
        options.append(
            {
                "provider": "ollama",
                "model": model,
                "label": f"Ollama - {model}",
            }
        )

    return options
