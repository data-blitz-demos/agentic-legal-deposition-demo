from __future__ import annotations

from pathlib import Path

import pytest

from backend.app import schemas as schemas_module


def test_load_schema_success_reads_json_object():
    schemas_module.load_schema.cache_clear()

    schema = schemas_module.load_schema("deposition_schema")

    assert schema["title"] == "DepositionSchema"
    assert schema["type"] == "object"
    assert "claims" in schema["properties"]


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
