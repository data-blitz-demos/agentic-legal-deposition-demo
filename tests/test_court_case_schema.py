# Copyright (c) 2026 Data-Blitz Inc. All rights reserved.
# License: Proprietary. See NOTICE.md.
# Author: Paul Harvener.

from __future__ import annotations

import json
from pathlib import Path

from backend.app import schemas as schemas_module


def _load_example_payload() -> dict:
    example_path = (
        Path(__file__).resolve().parents[1]
        / "backend/schemas/examples/court_case_moure_cabrera_mark_rigler_example.json"
    )
    return json.loads(example_path.read_text(encoding="utf-8"))


def test_court_case_schema_is_discoverable_as_raw_capture():
    schemas_module.load_schema.cache_clear()

    schema = schemas_module.load_schema("court_case_schema")
    options = schemas_module.list_schema_options()

    assert schema["title"] == "CourtCaseSchema"
    assert any(item["key"] == "court_case_schema" and item["mode"] == "raw_capture" for item in options)


def test_court_case_schema_captures_parties_timing_and_locations():
    schemas_module.load_schema.cache_clear()

    schema = schemas_module.load_schema("court_case_schema")
    properties = schema.get("properties", {})

    assert {"case", "proceeding", "deponent", "appearances", "exhibit_index", "quality"} <= set(properties)

    case_properties = properties["case"]["properties"]
    assert {"case_number", "caption", "court", "parties", "judges"} <= set(case_properties)

    party_properties = case_properties["parties"]["properties"]
    assert {"plaintiffs", "defendants", "other_parties"} <= set(party_properties)

    proceeding_properties = properties["proceeding"]["properties"]
    assert {"timing", "location", "reporting", "attendance_notes"} <= set(proceeding_properties)

    timing_properties = schema["$defs"]["timing"]["properties"]
    assert {"date", "start_time", "end_time", "certification_date", "review_deadline_days"} <= set(
        timing_properties
    )

    location_properties = schema["$defs"]["location"]["properties"]
    assert {"venue_name", "address_line_1", "city", "county", "state", "postal_code", "country"} <= set(
        location_properties
    )


def test_court_case_example_uses_mark_rigler_transcript_metadata():
    schemas_module.load_schema.cache_clear()

    schema = schemas_module.load_schema("court_case_schema")
    payload = _load_example_payload()

    assert set(schema["required"]) <= set(payload)
    assert payload["document_type"] == "court_case_capture"
    assert (
        payload["source_file"]
        == "depositions/exampleOnlyaFew/2020.01.23 Moure-Cabrera - Depo of Mark Rigler.txt"
    )
    assert payload["case"]["case_number"] == "19-000727"
    assert payload["case"]["parties"]["plaintiffs"][0]["name"] == "Blanca Moure-Cabrera"
    assert [item["name"] for item in payload["case"]["parties"]["defendants"]] == [
        "Johnson & Johnson",
        "Johnson & Johnson Consumer, Inc.",
    ]
    assert payload["case"]["judges"] == []
    assert payload["proceeding"]["timing"]["date"] == "2020-01-23"
    assert payload["proceeding"]["timing"]["start_time"] == "10:40:00"
    assert payload["proceeding"]["timing"]["certification_date"] == "2020-01-24"
    assert payload["proceeding"]["location"]["city"] == "Snellville"
    assert payload["proceeding"]["location"]["state"] == "Georgia"
    assert payload["appearances"]["for_plaintiff"][0]["name"] == "Marc P. Kunen"
    assert payload["appearances"]["for_defendants"][0]["attendance_mode"] == "telephone"
    assert len(payload["exhibit_index"]) == 8
    assert payload["quality"]["warnings"]
