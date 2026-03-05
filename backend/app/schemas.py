# Copyright (c) 2026 Data-Blitz Inc. All rights reserved.
# License: Proprietary. See NOTICE.md.
# Author: Paul Harvener.

from __future__ import annotations

"""JSON schema loading utilities for extraction/validation contracts."""

import json
from functools import lru_cache
from pathlib import Path
import re


def _schema_key_from_filename(file_name: str) -> str:
    """Derive a stable schema key from a schema file name."""

    stem = Path(file_name).stem.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return normalized or "schema"


def _schema_dir() -> Path:
    """Return on-disk directory containing JSON schema files."""

    return Path(__file__).resolve().parents[1] / "schemas"


def _discover_schema_files() -> dict[str, str]:
    """Discover candidate JSON schema files from disk.

    A file is considered a schema candidate when its name contains the token
    ``schema`` and ends in ``.json``. Invalid JSON files are filtered later by
    ``list_schema_options`` so the registry can include draft candidates without
    crashing import time.
    """

    schema_dir = _schema_dir()
    if not schema_dir.exists():
        return {"deposition_schema": "deposition_schema.json"}

    discovered: dict[str, str] = {}
    for path in sorted(schema_dir.glob("*schema*.json")):
        discovered[_schema_key_from_filename(path.name)] = path.name

    if "deposition_schema" not in discovered:
        discovered["deposition_schema"] = "deposition_schema.json"
    return discovered


SCHEMA_FILES: dict[str, str] = _discover_schema_files()


def schema_mode(name: str) -> str:
    """Return compatibility mode for a schema key.

    ``native`` means the structured payload is expected to map directly into the
    application's ``DepositionSchema`` model. Other schemas are treated as
    capture-only schemas and are normalized into the runtime model separately.
    """

    return "native" if name == "deposition_schema" else "raw_capture"


def list_schema_options() -> list[dict[str, str]]:
    """Return valid, loadable schema options for UI and API consumers."""

    options: list[dict[str, str]] = []
    load_schema.cache_clear()
    for name, file_name in sorted(SCHEMA_FILES.items()):
        try:
            payload = load_schema(name)
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            continue
        options.append(
            {
                "key": name,
                "file_name": file_name,
                "mode": schema_mode(name),
            }
        )
    if not options:
        options.append(
            {
                "key": "deposition_schema",
                "file_name": "deposition_schema.json",
                "mode": "native",
            }
        )
    return options


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
