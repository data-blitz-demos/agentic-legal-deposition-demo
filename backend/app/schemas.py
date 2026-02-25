from __future__ import annotations

"""JSON schema loading utilities for extraction/validation contracts."""

import json
from functools import lru_cache
from pathlib import Path

SCHEMA_FILES: dict[str, str] = {
    "deposition_schema": "deposition_schema.json",
}


def _schema_dir() -> Path:
    """Return on-disk directory containing JSON schema files."""

    return Path(__file__).resolve().parents[1] / "schemas"


@lru_cache(maxsize=None)
def load_schema(name: str) -> dict:
    """Load a named schema JSON file from disk."""

    try:
        file_name = SCHEMA_FILES[name]
    except KeyError as exc:  # pragma: no cover - guarded by tests
        raise ValueError(f"Unknown schema name: {name}") from exc

    path = _schema_dir() / file_name
    if not path.exists():
        raise FileNotFoundError(f"Schema file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Schema '{name}' must be a JSON object")
    return payload
