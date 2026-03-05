# Copyright (c) 2026 Data-Blitz Inc. All rights reserved.
# License: Proprietary. See NOTICE.md.
# Author: Paul Harvener.

from __future__ import annotations

from pathlib import Path

import pytest

from backend.app import schemas as schemas_module


def test_load_schema_success_reads_json_object():
    schemas_module.load_schema.cache_clear()

    schema = schemas_module.load_schema("deposition_schema")

    assert schema["title"]
    assert schema["type"] == "object"
    assert isinstance(schema["properties"], dict)
    assert schema["properties"]


def test_load_schema_unknown_name_raises_value_error():
    schemas_module.load_schema.cache_clear()

    with pytest.raises(ValueError, match="Unknown schema name"):
        schemas_module.load_schema("does_not_exist")


def test_load_schema_missing_file_raises(monkeypatch):
    schemas_module.load_schema.cache_clear()
    monkeypatch.setitem(schemas_module.SCHEMA_FILES, "fake_schema", "missing_schema.json")

    with pytest.raises(FileNotFoundError, match="Schema file not found"):
        schemas_module.load_schema("fake_schema")


def test_load_schema_non_object_raises(monkeypatch, tmp_path):
    schemas_module.load_schema.cache_clear()
    schema_dir = tmp_path / "schemas"
    schema_dir.mkdir()
    (schema_dir / "bad.json").write_text("[]", encoding="utf-8")
    monkeypatch.setitem(schemas_module.SCHEMA_FILES, "bad_schema", "bad.json")
    monkeypatch.setattr(schemas_module, "_schema_dir", lambda: schema_dir)

    with pytest.raises(ValueError, match="must be a JSON object"):
        schemas_module.load_schema("bad_schema")


def test_schema_dir_points_to_backend_schemas_folder():
    schema_dir = schemas_module._schema_dir()

    assert schema_dir.name == "schemas"
    assert schema_dir == Path(__file__).resolve().parents[1] / "backend/schemas"


def test_schema_key_from_filename_normalizes_punctuation():
    assert schemas_module._schema_key_from_filename("deposition_schema-g1.json") == "deposition_schema_g1"


def test_list_schema_options_returns_valid_known_schemas():
    schemas_module.load_schema.cache_clear()

    options = schemas_module.list_schema_options()
    keys = {item["key"] for item in options}

    assert "deposition_schema" in keys
    assert "deposition_schema_g1" in keys
    assert "model_schema" not in keys


def test_list_schema_options_skips_invalid_json(monkeypatch, tmp_path):
    schemas_module.load_schema.cache_clear()
    schema_dir = tmp_path / "schemas"
    schema_dir.mkdir()
    (schema_dir / "good_schema.json").write_text('{"title":"Good","type":"object"}', encoding="utf-8")
    (schema_dir / "bad_schema.json").write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(schemas_module, "_schema_dir", lambda: schema_dir)
    monkeypatch.setattr(
        schemas_module,
        "SCHEMA_FILES",
        {
            "good_schema": "good_schema.json",
            "bad_schema": "bad_schema.json",
        },
    )

    options = schemas_module.list_schema_options()

    assert options == [
        {
            "key": "good_schema",
            "file_name": "good_schema.json",
            "mode": "raw_capture",
        }
    ]
