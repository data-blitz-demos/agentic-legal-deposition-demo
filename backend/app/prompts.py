from __future__ import annotations

"""Prompt template loading and rendering helpers.

Prompt files are stored under ``backend/prompts`` and referenced by stable keys.
This keeps prompt content versioned independently from runtime code.
"""

from functools import lru_cache
from pathlib import Path

PROMPT_FILES: dict[str, str] = {
    "map_deposition_system": "map_deposition_system.txt",
    "map_deposition_user": "map_deposition_user.txt",
    "assess_contradictions_system": "assess_contradictions_system.txt",
    "assess_contradictions_user": "assess_contradictions_user.txt",
    "chat_system": "chat_system.txt",
    "chat_user_context": "chat_user_context.txt",
    "graph_rag_system": "graph_rag_system.txt",
    "graph_rag_user": "graph_rag_user.txt",
    "reason_contradiction_system": "reason_contradiction_system.txt",
    "reason_contradiction_user": "reason_contradiction_user.txt",
}


def _prompt_dir() -> Path:
    """Return the on-disk prompt template directory."""

    return Path(__file__).resolve().parents[1] / "prompts"


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    """Load a named prompt template from disk (cached per process)."""

    try:
        file_name = PROMPT_FILES[name]
    except KeyError as exc:  # pragma: no cover - guarded by tests
        raise ValueError(f"Unknown prompt name: {name}") from exc

    path = _prompt_dir() / file_name
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def render_prompt(name: str, **kwargs) -> str:
    """Render a prompt template using ``str.format`` variables."""

    template = load_prompt(name)
    try:
        return template.format(**kwargs)
    except KeyError as exc:
        missing = exc.args[0]
        raise ValueError(f"Missing prompt variable '{missing}' for {name}") from exc
