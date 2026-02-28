from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import HTTPException
from fastapi import UploadFile

from backend.app import main
from backend.app.models import (
    ChatRequest,
    ContradictionFinding,
    ContradictionReasonRequest,
    DepositionSentimentRequest,
    DepositionRootRequest,
    FocusedReasoningSummaryRequest,
    GraphOntologyLoadRequest,
    GraphRagQueryRequest,
    IngestCaseRequest,
    RenameCaseRequest,
    SaveCaseRequest,
    SaveTraceRequest,
    SaveCaseVersionRequest,
)


@pytest.fixture(autouse=True)
def stub_llm_readiness(monkeypatch):
    monkeypatch.setattr(main, "ensure_llm_operational", lambda *_args, **_kwargs: None)
    memory_db = Mock()
    memory_db.find.return_value = []
    monkeypatch.setattr(main, "memory_couchdb", memory_db)


def test_path_has_glob_detection():
    assert main._path_has_glob(Path("/tmp/*.txt")) is True
    assert main._path_has_glob(Path("/tmp/depositions")) is False


def test_collect_txt_files_from_directory_file_and_glob(tmp_path):
    dep_dir = tmp_path / "deps"
    dep_dir.mkdir()
    a = dep_dir / "a.txt"
    b = dep_dir / "b.TXT"
    c = dep_dir / "c.pdf"
    a.write_text("a", encoding="utf-8")
    b.write_text("b", encoding="utf-8")
    c.write_text("c", encoding="utf-8")

    directory_files = main._collect_txt_files(dep_dir)
    single_file = main._collect_txt_files(a)
    glob_files = main._collect_txt_files(dep_dir / "*.txt")
    non_txt_file = main._collect_txt_files(c)

    assert directory_files == [a, b]
    assert single_file == [a]
    assert glob_files == [a]
    assert non_txt_file == []


def test_collect_owl_files_from_directory_file_and_glob(tmp_path):
    ontology_dir = tmp_path / "ontology"
    ontology_dir.mkdir()
    a = ontology_dir / "legal.owl"
    b = ontology_dir / "other.OWL"
    c = ontology_dir / "notes.txt"
    a.write_text("a", encoding="utf-8")
    b.write_text("b", encoding="utf-8")
    c.write_text("c", encoding="utf-8")

    directory_files = main._collect_owl_files(ontology_dir)
    single_file = main._collect_owl_files(a)
    glob_files = main._collect_owl_files(ontology_dir / "*.owl")
    non_owl_file = main._collect_owl_files(c)

    assert directory_files == [a, b]
    assert single_file == [a]
    assert glob_files == [a]
    assert non_owl_file == []


def test_resolve_upload_directory_and_sanitize_filename(tmp_path, monkeypatch):
    target = tmp_path / "deps"
    target.mkdir()
    monkeypatch.setattr(main, "_build_ingest_candidates", lambda _path: [Path("/tmp/*.txt"), target])

    resolved = main._resolve_upload_directory("/data/depositions/default")

    assert resolved == target
    assert main._sanitize_uploaded_deposition_filename("witness lisa!.TXT", 1) == "witness_lisa.txt"
    assert main._sanitize_uploaded_deposition_filename("", 2) == "uploaded_deposition_2.txt"
    assert main._sanitize_uploaded_deposition_filename("***.txt", 3) == "uploaded_deposition_3.txt"


def test_resolve_upload_directory_rejects_missing_target(monkeypatch):
    monkeypatch.setattr(main, "_build_ingest_candidates", lambda _path: [Path("/missing/deps")])

    with pytest.raises(HTTPException) as excinfo:
        main._resolve_upload_directory("/missing/deps")

    assert excinfo.value.status_code == 400
    assert "existing directory" in excinfo.value.detail

    with pytest.raises(HTTPException) as empty_exc:
        main._resolve_upload_directory("")

    assert empty_exc.value.status_code == 400
    assert "Deposition folder is required" in empty_exc.value.detail


def test_upload_depositions_rejects_empty_list_and_non_txt(tmp_path, monkeypatch):
    target = tmp_path / "deps"
    target.mkdir()
    monkeypatch.setattr(main, "_resolve_upload_directory", lambda _path: target)

    with pytest.raises(HTTPException) as empty_exc:
        main.upload_depositions(directory=str(target), files=[])

    assert empty_exc.value.status_code == 400

    bad_upload = UploadFile(filename="evidence.pdf", file=SimpleNamespace(read=lambda: b"pdf", close=lambda: None))
    with pytest.raises(HTTPException) as bad_exc:
        main.upload_depositions(directory=str(target), files=[bad_upload])

    assert bad_exc.value.status_code == 400
    assert ".txt file" in bad_exc.value.detail


def test_upload_depositions_wraps_write_error_and_ignores_close_error(tmp_path, monkeypatch):
    target = tmp_path / "deps"
    target.mkdir()
    monkeypatch.setattr(main, "_resolve_upload_directory", lambda _path: target)

    class _BrokenFile:
        def read(self):
            raise OSError("disk full")

        def close(self):
            raise OSError("close failed")

    broken_upload = SimpleNamespace(filename="broken.txt", file=_BrokenFile())

    with pytest.raises(HTTPException) as excinfo:
        main.upload_depositions(directory=str(target), files=[broken_upload])

    assert excinfo.value.status_code == 502
    assert "Failed to save uploaded deposition 'broken.txt'" in excinfo.value.detail


def test_render_themed_admin_test_report_returns_input_when_already_themed():
    html = '<html><head><style id="admin-report-theme"></style></head><body>ready</body></html>'

    themed = main._render_themed_admin_test_report(html)

    assert themed == html


def test_render_themed_admin_test_report_prefixes_style_without_head():
    html = "<div>report body only</div>"

    themed = main._render_themed_admin_test_report(html)

    assert themed.startswith(main._ADMIN_REPORT_THEME_CSS)
    assert themed.endswith(html)


def test_extract_test_report_log_output_handles_missing_file(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "tests_report_file", tmp_path / "missing-tests.html")

    payload = main._extract_test_report_log_output()

    assert payload.summary == "tests.html is not available."
    assert "Run the test suite" in payload.log_output


def test_extract_test_report_log_output_handles_missing_data_blob(monkeypatch, tmp_path):
    report_file = tmp_path / "tests.html"
    report_file.write_text("<html><body><h1>No blob</h1></body></html>", encoding="utf-8")
    monkeypatch.setattr(main, "tests_report_file", report_file)

    payload = main._extract_test_report_log_output()

    assert payload.summary == "No structured test metadata was found in tests.html."
    assert "data-jsonblob" in payload.log_output


def test_extract_test_report_log_output_handles_bad_payload_and_no_explicit_logs(monkeypatch, tmp_path):
    bad_file = tmp_path / "bad-tests.html"
    bad_file.write_text('<div data-jsonblob="{not-json}"></div>', encoding="utf-8")
    monkeypatch.setattr(main, "tests_report_file", bad_file)

    bad_payload = main._extract_test_report_log_output()

    assert bad_payload.summary == "The embedded test metadata could not be parsed."
    assert "could not decode" in bad_payload.log_output

    quiet_file = tmp_path / "quiet-tests.html"
    quiet_file.write_text(
        (
            '<div id="data-container" data-jsonblob="{&#34;tests&#34;:{&#34;tests/test_quiet.py::test_only&#34;:'
            '[{&#34;result&#34;:&#34;Passed&#34;,&#34;log&#34;:&#34;No log output captured.&#34;}]}}"></div>'
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(main, "tests_report_file", quiet_file)

    quiet_payload = main._extract_test_report_log_output()

    assert "Parsed 1 recorded test runs" in quiet_payload.summary
    assert "No explicit per-test log output was captured" in quiet_payload.log_output


def test_extract_test_report_log_output_skips_non_list_and_non_dict_runs(monkeypatch, tmp_path):
    report_file = tmp_path / "tests.html"
    report_file.write_text(
        (
            '<div id="data-container" data-jsonblob="{&#34;tests&#34;:{'
            '&#34;tests/test_skip.py::test_non_list&#34;:&#34;bad&#34;,'
            '&#34;tests/test_skip.py::test_non_dict&#34;:[&#34;bad-run&#34;]}}"></div>'
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(main, "tests_report_file", report_file)

    payload = main._extract_test_report_log_output()

    assert payload.summary == "Parsed 0 recorded test runs from tests.html."
    assert "No explicit per-test log output was captured" in payload.log_output


def test_list_admin_users_skips_invalid_docs_and_save_admin_user_validates(monkeypatch):
    class _BrokenDoc:
        def get(self, _key, _default=None):
            raise RuntimeError("bad doc")

    couchdb = Mock()
    couchdb.find.return_value = [
        {"_id": "", "name": "Missing Id", "created_at": "2026-02-28T00:00:00+00:00"},
        _BrokenDoc(),
        {"_id": "admin:1", "name": "Paul Harvener", "created_at": "2026-02-28T00:00:00+00:00"},
    ]
    monkeypatch.setattr(main, "couchdb", couchdb)

    users = main._list_admin_users()

    assert len(users) == 1
    assert users[0].name == "Paul Harvener"

    with pytest.raises(HTTPException) as excinfo:
        main._save_admin_user("   ")

    assert excinfo.value.status_code == 400


def test_build_ingest_candidates_handles_container_mapping(monkeypatch):
    monkeypatch.setattr(main, "app_root", Path("/workspace/app"))
    monkeypatch.setattr(main, "settings", SimpleNamespace(deposition_dir="/host/deps", deposition_extra_dirs=""))

    candidates = main._build_ingest_candidates("/data/depositions/folder")

    assert Path("/data/depositions/folder") in candidates
    assert Path("/host/deps/folder") in candidates
    assert Path("/workspace/app/depositions/folder") in candidates
    assert Path("/workspace/app/depositions/default/folder") in candidates


def test_build_ingest_candidates_relative_input_and_relative_config(monkeypatch):
    monkeypatch.setattr(main, "app_root", Path("/workspace/app"))
    monkeypatch.setattr(main, "settings", SimpleNamespace(deposition_dir="relative_deps", deposition_extra_dirs=""))

    candidates = main._build_ingest_candidates("my-folder")

    assert Path("my-folder") in candidates
    assert Path("/workspace/app/my-folder") in candidates

    mapped = main._build_ingest_candidates("/data/depositions/sub")
    assert Path("/workspace/app/relative_deps/sub") in mapped
    assert Path.cwd() / "relative_deps/sub" in mapped


def test_build_ingest_candidates_deduplicates(monkeypatch):
    monkeypatch.setattr(main, "app_root", Path("/workspace/app"))
    monkeypatch.setattr(
        main, "settings", SimpleNamespace(deposition_dir="depositions/default", deposition_extra_dirs="")
    )

    candidates = main._build_ingest_candidates("/data/depositions")

    assert candidates.count(Path("/workspace/app/depositions/default")) == 1


def test_build_ingest_candidates_handles_legacy_oj_folder_alias(monkeypatch):
    monkeypatch.setattr(main, "app_root", Path("/workspace/app"))
    monkeypatch.setattr(main, "settings", SimpleNamespace(deposition_dir="/host/deps", deposition_extra_dirs=""))

    candidates = main._build_ingest_candidates("/data/depositions/oj_simpson")

    assert Path("/workspace/app/depositions/oj_simpson") in candidates
    assert Path("/workspace/app/depositions/oj_simposon") in candidates


def test_build_ingest_candidates_oj_alias_with_relative_config(monkeypatch):
    monkeypatch.setattr(main, "app_root", Path("/workspace/app"))
    monkeypatch.setattr(main, "settings", SimpleNamespace(deposition_dir="relative_deps", deposition_extra_dirs=""))

    candidates = main._build_ingest_candidates("/data/depositions/oj_simpson")

    assert Path("/workspace/app/relative_deps/oj_simpson") in candidates
    assert Path("/workspace/app/relative_deps/oj_simposon") in candidates
    assert Path.cwd() / "relative_deps/oj_simpson" in candidates
    assert Path.cwd() / "relative_deps/oj_simposon" in candidates


def test_build_ingest_candidates_handles_default_prefixed_oj_path(monkeypatch):
    monkeypatch.setattr(main, "app_root", Path("/workspace/app"))
    monkeypatch.setattr(
        main, "settings", SimpleNamespace(deposition_dir="depositions/default", deposition_extra_dirs="")
    )

    candidates = main._build_ingest_candidates("/data/depositions/default/oj_simpson")

    assert Path("/workspace/app/depositions/default/oj_simpson") in candidates
    assert Path("/workspace/app/depositions/oj_simpson") in candidates
    assert Path("/workspace/app/depositions/oj_simposon") in candidates


def test_build_ingest_candidates_non_container_path(monkeypatch):
    monkeypatch.setattr(main, "app_root", Path("/workspace/app"))
    monkeypatch.setattr(main, "settings", SimpleNamespace(deposition_dir="/host/deps", deposition_extra_dirs=""))

    candidates = main._build_ingest_candidates("/outside/path")

    assert Path("/host/deps/path") not in candidates
    assert Path("/outside/path") in candidates


def test_lifespan_ensures_and_closes_couchdb(monkeypatch):
    couchdb = Mock()
    memory_couchdb = Mock()
    trace_couchdb = Mock()
    rag_couchdb = Mock()
    startup = Mock()
    monkeypatch.setattr(main, "couchdb", couchdb)
    monkeypatch.setattr(main, "memory_couchdb", memory_couchdb)
    monkeypatch.setattr(main, "trace_couchdb", trace_couchdb)
    monkeypatch.setattr(main, "rag_couchdb", rag_couchdb)
    monkeypatch.setattr(main, "_ensure_startup_llm_connectivity", startup)

    async def run_lifespan() -> None:
        async with main.lifespan(main.app):
            pass

    asyncio.run(run_lifespan())

    startup.assert_called_once_with()
    couchdb.ensure_db.assert_called_once_with()
    couchdb.close.assert_called_once_with()
    memory_couchdb.ensure_db.assert_called_once_with()
    memory_couchdb.close.assert_called_once_with()
    trace_couchdb.ensure_db.assert_called_once_with()
    trace_couchdb.close.assert_called_once_with()
    rag_couchdb.ensure_db.assert_called_once_with()
    rag_couchdb.close.assert_called_once_with()


def test_lifespan_stops_when_startup_llm_check_fails(monkeypatch):
    couchdb = Mock()
    memory_couchdb = Mock()
    trace_couchdb = Mock()
    rag_couchdb = Mock()
    monkeypatch.setattr(main, "couchdb", couchdb)
    monkeypatch.setattr(main, "memory_couchdb", memory_couchdb)
    monkeypatch.setattr(main, "trace_couchdb", trace_couchdb)
    monkeypatch.setattr(main, "rag_couchdb", rag_couchdb)
    monkeypatch.setattr(
        main,
        "_ensure_startup_llm_connectivity",
        Mock(side_effect=RuntimeError("startup failed")),
    )

    async def run_lifespan() -> None:
        async with main.lifespan(main.app):
            pass

    with pytest.raises(RuntimeError, match="startup failed"):
        asyncio.run(run_lifespan())

    couchdb.ensure_db.assert_not_called()
    couchdb.close.assert_not_called()
    memory_couchdb.ensure_db.assert_not_called()
    memory_couchdb.close.assert_not_called()
    trace_couchdb.ensure_db.assert_not_called()
    trace_couchdb.close.assert_not_called()
    rag_couchdb.ensure_db.assert_not_called()
    rag_couchdb.close.assert_not_called()


def test_resolve_ingest_txt_files_deduplicates(monkeypatch, tmp_path):
    dep_dir = tmp_path / "deps"
    dep_dir.mkdir()
    txt = dep_dir / "a.txt"
    txt.write_text("a", encoding="utf-8")

    monkeypatch.setattr(main, "_build_ingest_candidates", lambda _path: [dep_dir, txt])

    files = main._resolve_ingest_txt_files("ignored")

    assert files == [txt]


def test_resolve_ingest_txt_files_raises_when_no_matches(monkeypatch):
    empty = Path("/tmp/does-not-exist")
    monkeypatch.setattr(main, "_build_ingest_candidates", lambda _path: [empty])

    with pytest.raises(HTTPException) as exc:
        main._resolve_ingest_txt_files("missing")

    assert exc.value.status_code == 400
    assert "No .txt deposition files found" in exc.value.detail


def test_resolve_ontology_owl_files_from_relative_path(monkeypatch, tmp_path):
    app_root = tmp_path / "app"
    app_root.mkdir()
    ontology_root = tmp_path / "ontology"
    ontology_root.mkdir()
    owl = ontology_root / "legal.owl"
    owl.write_text("x", encoding="utf-8")

    monkeypatch.setattr(main, "app_root", app_root)
    monkeypatch.setattr(main, "settings", SimpleNamespace(ontology_dir=str(ontology_root)))

    files = main._resolve_ontology_owl_files("legal.owl")

    assert files == [owl]


def test_resolve_ontology_owl_files_raises_when_no_matches(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "app_root", tmp_path)
    monkeypatch.setattr(main, "settings", SimpleNamespace(ontology_dir=str(tmp_path / "ontology")))

    with pytest.raises(HTTPException) as exc:
        main._resolve_ontology_owl_files("/missing/*.owl")

    assert exc.value.status_code == 400
    assert "No .owl ontology files found" in exc.value.detail


def test_resolve_ontology_owl_files_deduplicates_candidates_and_files(monkeypatch, tmp_path):
    shared_root = tmp_path / "shared"
    shared_root.mkdir()
    owl = shared_root / "legal.owl"
    owl.write_text("x", encoding="utf-8")

    monkeypatch.setattr(main, "app_root", shared_root)
    monkeypatch.setattr(main, "settings", SimpleNamespace(ontology_dir=str(shared_root)))
    monkeypatch.setattr(main, "_collect_owl_files", lambda _candidate: [owl])

    files = main._resolve_ontology_owl_files("legal.owl")

    assert files == [owl]


def test_configured_ontology_root_resolves_relative_path(monkeypatch, tmp_path):
    app_root = tmp_path / "app"
    app_root.mkdir()
    monkeypatch.setattr(main, "app_root", app_root)
    monkeypatch.setattr(main, "settings", SimpleNamespace(ontology_dir="ontology"))

    resolved = main._configured_ontology_root()

    assert resolved == app_root / "ontology"


def test_list_ontology_owl_options_includes_wildcard_and_nested_files(monkeypatch, tmp_path):
    ontology_root = tmp_path / "ontology"
    nested = ontology_root / "sasl"
    nested.mkdir(parents=True)
    primary = ontology_root / "legal.owl"
    nested_file = nested / "contracts.OWL"
    (ontology_root / "notes.txt").write_text("ignore", encoding="utf-8")
    primary.write_text("p", encoding="utf-8")
    nested_file.write_text("n", encoding="utf-8")

    monkeypatch.setattr(main, "settings", SimpleNamespace(ontology_dir=str(ontology_root)))
    monkeypatch.setattr(main, "app_root", tmp_path)

    base, options, suggested = main._list_ontology_owl_options()

    assert base == ontology_root
    assert suggested == str(ontology_root / "*.owl")
    option_paths = [item.path for item in options]
    assert option_paths[0] == str(ontology_root / "*.owl")
    assert str(primary) in option_paths
    assert str(nested_file) in option_paths


def test_resolve_ontology_browser_directory_resolves_relative_and_file_paths(monkeypatch, tmp_path):
    ontology_root = tmp_path / "ontology"
    nested = ontology_root / "sub"
    nested.mkdir(parents=True)
    owl = nested / "legal.owl"
    owl.write_text("x", encoding="utf-8")

    monkeypatch.setattr(main, "settings", SimpleNamespace(ontology_dir=str(ontology_root)))
    monkeypatch.setattr(main, "app_root", tmp_path)

    base, current = main._resolve_ontology_browser_directory("sub")
    assert base == ontology_root.resolve()
    assert current == nested.resolve()

    _, current_from_file = main._resolve_ontology_browser_directory(str(owl))
    assert current_from_file == nested.resolve()


def test_resolve_ontology_browser_directory_defaults_to_base_when_path_missing(monkeypatch, tmp_path):
    ontology_root = tmp_path / "ontology"
    ontology_root.mkdir()
    monkeypatch.setattr(main, "settings", SimpleNamespace(ontology_dir=str(ontology_root)))
    monkeypatch.setattr(main, "app_root", tmp_path)

    base, current = main._resolve_ontology_browser_directory(None)

    assert base == ontology_root.resolve()
    assert current == ontology_root.resolve()


def test_resolve_ontology_browser_directory_falls_back_to_nearest_existing_parent(monkeypatch, tmp_path):
    ontology_root = tmp_path / "ontology"
    nested = ontology_root / "contracts"
    nested.mkdir(parents=True)
    missing = nested / "missing" / "deeper"

    monkeypatch.setattr(main, "settings", SimpleNamespace(ontology_dir=str(ontology_root)))
    monkeypatch.setattr(main, "app_root", tmp_path)

    base, current = main._resolve_ontology_browser_directory(str(missing))
    assert base == ontology_root.resolve()
    assert current == nested.resolve()


def test_resolve_ontology_browser_directory_handles_wildcard_and_missing_owl_file(monkeypatch, tmp_path):
    ontology_root = tmp_path / "ontology"
    nested = ontology_root / "contracts"
    nested.mkdir(parents=True)

    monkeypatch.setattr(main, "settings", SimpleNamespace(ontology_dir=str(ontology_root)))
    monkeypatch.setattr(main, "app_root", tmp_path)

    _, wildcard_current = main._resolve_ontology_browser_directory(str(nested / "*.owl"))
    assert wildcard_current == nested.resolve()

    _, missing_file_current = main._resolve_ontology_browser_directory(str(nested / "future.owl"))
    assert missing_file_current == nested.resolve()


def test_resolve_ontology_browser_directory_falls_back_to_base_when_root_missing(monkeypatch, tmp_path):
    ontology_root = tmp_path / "ontology-missing"

    monkeypatch.setattr(main, "settings", SimpleNamespace(ontology_dir=str(ontology_root)))
    monkeypatch.setattr(main, "app_root", tmp_path)

    with pytest.raises(HTTPException) as exc:
        main._resolve_ontology_browser_directory(str(ontology_root / "contracts" / "future.owl"))

    assert exc.value.status_code == 400
    assert "must be a directory" in exc.value.detail


def test_resolve_ontology_browser_directory_rejects_outside_base(monkeypatch, tmp_path):
    ontology_root = tmp_path / "ontology"
    ontology_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    monkeypatch.setattr(main, "settings", SimpleNamespace(ontology_dir=str(ontology_root)))
    monkeypatch.setattr(main, "app_root", tmp_path)

    with pytest.raises(HTTPException) as exc:
        main._resolve_ontology_browser_directory(str(outside))

    assert exc.value.status_code == 400
    assert "must remain under base directory" in exc.value.detail


def test_resolve_ontology_browser_directory_rejects_non_owl_file(monkeypatch, tmp_path):
    ontology_root = tmp_path / "ontology"
    ontology_root.mkdir()
    not_owl = ontology_root / "notes.txt"
    not_owl.write_text("x", encoding="utf-8")

    monkeypatch.setattr(main, "settings", SimpleNamespace(ontology_dir=str(ontology_root)))
    monkeypatch.setattr(main, "app_root", tmp_path)

    with pytest.raises(HTTPException) as exc:
        main._resolve_ontology_browser_directory(str(not_owl))

    assert exc.value.status_code == 400
    assert "directory or .owl file" in exc.value.detail


def test_resolve_ontology_browser_directory_rejects_non_directory_target(monkeypatch, tmp_path):
    ontology_root = tmp_path / "ontology"
    ontology_root.mkdir()
    fifo_path = ontology_root / "stream.fifo"
    import os
    os.mkfifo(fifo_path)

    monkeypatch.setattr(main, "settings", SimpleNamespace(ontology_dir=str(ontology_root)))
    monkeypatch.setattr(main, "app_root", tmp_path)

    with pytest.raises(HTTPException) as exc:
        main._resolve_ontology_browser_directory(str(fifo_path))

    assert exc.value.status_code == 400
    assert "must be a directory" in exc.value.detail


def test_list_ontology_browser_entries_lists_directories_and_owl_files_only(tmp_path):
    root = tmp_path / "ontology"
    child = root / "child"
    root.mkdir()
    child.mkdir()
    (root / "legal.owl").write_text("owl", encoding="utf-8")
    (root / "notes.txt").write_text("ignore", encoding="utf-8")

    directories, files = main._list_ontology_browser_entries(root)

    assert [item.kind for item in directories] == ["directory"]
    assert directories[0].path == str(child)
    assert [item.kind for item in files] == ["file"]
    assert files[0].path == str(root / "legal.owl")


def test_list_ontology_browser_entries_returns_empty_for_missing_path(tmp_path):
    directories, files = main._list_ontology_browser_entries(tmp_path / "missing")
    assert directories == []
    assert files == []


def test_root_returns_index_file_response():
    response = main.root()
    assert str(response.path).endswith("frontend/index.html")


def test_get_deposition_directories_returns_discovered_options(monkeypatch, tmp_path):
    mounted = tmp_path / "mounted"
    mounted.mkdir()
    (mounted / "m1.txt").write_text("m1", encoding="utf-8")
    mounted_default = mounted / "default"
    mounted_oj = mounted / "oj_simpson"
    mounted_default.mkdir()
    mounted_oj.mkdir()
    (mounted_default / "d1.txt").write_text("d1", encoding="utf-8")
    (mounted_oj / "o1.txt").write_text("o1", encoding="utf-8")

    configured = tmp_path / "configured"
    configured.mkdir()
    (configured / "c1.txt").write_text("c1", encoding="utf-8")

    repo_root = tmp_path / "repo"
    repo_deps = repo_root / "depositions"
    repo_default = repo_deps / "default"
    repo_oj = repo_deps / "oj_simpson"
    repo_default.mkdir(parents=True)
    repo_oj.mkdir(parents=True)
    (repo_default / "d1.txt").write_text("d1", encoding="utf-8")
    (repo_oj / "o1.txt").write_text("o1", encoding="utf-8")

    monkeypatch.setattr(main, "container_deposition_root", mounted)
    monkeypatch.setattr(main, "app_root", repo_root)
    monkeypatch.setattr(main, "settings", SimpleNamespace(deposition_dir=str(configured), deposition_extra_dirs=""))

    payload = main.get_deposition_directories()

    paths = [item.path for item in payload.options]
    assert payload.base_directory == str(mounted)
    assert str(mounted) in paths
    assert str(mounted_default) in paths
    assert str(mounted_oj) in paths
    assert str(configured) not in paths
    assert str(repo_default) not in paths
    assert str(repo_oj) not in paths
    assert payload.suggested is not None
    assert payload.suggested in paths


def test_txt_count_returns_zero_for_missing_path(tmp_path):
    assert main._txt_count(tmp_path / "missing") == 0


def test_add_directory_option_skips_missing_and_duplicates(tmp_path):
    options = []
    seen = set()
    missing = tmp_path / "missing"
    base = tmp_path / "base"
    base.mkdir()

    main._add_directory_option(options, seen, base, missing, "mounted")
    assert options == []

    dep_dir = tmp_path / "deps"
    dep_dir.mkdir()
    (dep_dir / "a.txt").write_text("a", encoding="utf-8")

    main._add_directory_option(options, seen, base, dep_dir, "mounted")
    main._add_directory_option(options, seen, base, dep_dir, "mounted")
    assert len(options) == 1


def test_add_directory_option_skips_directory_without_txt(tmp_path):
    options = []
    seen = set()
    base = tmp_path / "base"
    base.mkdir()
    no_txt = base / "empty_child"
    no_txt.mkdir()
    (no_txt / "note.md").write_text("x", encoding="utf-8")

    main._add_directory_option(options, seen, base, no_txt, "mounted")

    assert options == []


def test_configured_deposition_root_uses_relative_app_root(monkeypatch):
    monkeypatch.setattr(main, "app_root", Path("/workspace/app"))
    monkeypatch.setattr(main, "settings", SimpleNamespace(deposition_dir="depositions", deposition_extra_dirs=""))

    assert main._configured_deposition_root() == Path("/workspace/app/depositions")


def test_ingestable_directory_count(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "a.txt").write_text("a", encoding="utf-8")
    child_with_txt = root / "child"
    child_with_txt.mkdir()
    (child_with_txt / "b.txt").write_text("b", encoding="utf-8")
    child_without_txt = root / "empty"
    child_without_txt.mkdir()
    (child_without_txt / "c.md").write_text("c", encoding="utf-8")

    assert main._ingestable_directory_count(tmp_path / "missing") == -1
    assert main._ingestable_directory_count(root) == 2


def test_list_deposition_directories_skips_missing_roots(monkeypatch, tmp_path):
    mounted = tmp_path / "mounted"
    mounted.mkdir()
    (mounted / "m1.txt").write_text("m1", encoding="utf-8")

    missing_repo_root = tmp_path / "missing_repo"
    configured = tmp_path / "missing_configured"

    monkeypatch.setattr(main, "container_deposition_root", mounted)
    monkeypatch.setattr(main, "app_root", missing_repo_root)
    monkeypatch.setattr(main, "settings", SimpleNamespace(deposition_dir=str(configured), deposition_extra_dirs=""))

    base, options = main._list_deposition_directories()
    assert base == mounted
    assert len(options) == 1
    assert options[0].path == str(mounted)


def test_resolve_directory_base_prefers_configured_when_mounted_missing(monkeypatch, tmp_path):
    configured = tmp_path / "configured"
    configured.mkdir()
    repo_root = tmp_path / "repo"
    repo_deps = repo_root / "depositions"
    repo_deps.mkdir(parents=True)

    monkeypatch.setattr(main, "container_deposition_root", tmp_path / "missing_mounted")
    monkeypatch.setattr(main, "app_root", repo_root)
    monkeypatch.setattr(main, "settings", SimpleNamespace(deposition_dir=str(configured), deposition_extra_dirs=""))

    source, base = main._resolve_directory_base()

    assert source == "configured"
    assert base == configured


def test_resolve_directory_base_prefers_root_with_more_options(monkeypatch, tmp_path):
    mounted = tmp_path / "mounted"
    mounted.mkdir()
    (mounted / "m1.txt").write_text("m1", encoding="utf-8")

    configured = tmp_path / "configured"
    configured.mkdir()
    (configured / "c1.txt").write_text("c1", encoding="utf-8")

    repo_root = tmp_path / "repo"
    repo_deps = repo_root / "depositions"
    repo_deps.mkdir(parents=True)
    repo_default = repo_deps / "default"
    repo_oj = repo_deps / "oj_simpson"
    repo_default.mkdir()
    repo_oj.mkdir()
    (repo_default / "d1.txt").write_text("d1", encoding="utf-8")
    (repo_oj / "o1.txt").write_text("o1", encoding="utf-8")

    monkeypatch.setattr(main, "container_deposition_root", mounted)
    monkeypatch.setattr(main, "app_root", repo_root)
    monkeypatch.setattr(main, "settings", SimpleNamespace(deposition_dir=str(configured), deposition_extra_dirs=""))

    source, base = main._resolve_directory_base()

    assert source == "repo"
    assert base == repo_deps


def test_resolve_directory_base_fallback_when_no_roots_exist(monkeypatch, tmp_path):
    configured = tmp_path / "missing_configured"
    repo_root = tmp_path / "missing_repo"

    monkeypatch.setattr(main, "container_deposition_root", tmp_path / "missing_mounted")
    monkeypatch.setattr(main, "app_root", repo_root)
    monkeypatch.setattr(main, "settings", SimpleNamespace(deposition_dir=str(configured), deposition_extra_dirs=""))

    source, base = main._resolve_directory_base()

    assert source == "configured"
    assert base == configured


def test_list_deposition_directories_returns_empty_when_base_missing(monkeypatch, tmp_path):
    missing_base = tmp_path / "missing_base"
    monkeypatch.setattr(main, "_resolve_directory_base", lambda: ("configured", missing_base))

    base, options = main._list_deposition_directories()

    assert base == missing_base
    assert options == []


def test_configured_extra_deposition_roots_resolve_absolute_and_relative(monkeypatch):
    monkeypatch.setattr(main, "app_root", Path("/workspace/app"))
    monkeypatch.setattr(
        main,
        "settings",
        SimpleNamespace(
            deposition_dir="/ignored",
            deposition_extra_dirs="/opt/deps_a, ./extra_rel , ./extra_rel",
        ),
    )

    roots = main._configured_extra_deposition_roots()

    assert Path("/opt/deps_a") in roots
    assert Path("/workspace/app/extra_rel") in roots
    assert Path.cwd() / "extra_rel" in roots
    assert len([root for root in roots if str(root).endswith("extra_rel")]) == 2


def test_build_ingest_candidates_includes_extra_roots(monkeypatch):
    monkeypatch.setattr(main, "app_root", Path("/workspace/app"))
    monkeypatch.setattr(
        main,
        "settings",
        SimpleNamespace(
            deposition_dir="/host/deps",
            deposition_extra_dirs="/extra/deps_a, /extra/deps_b",
        ),
    )

    candidates = main._build_ingest_candidates("/data/depositions/folder")

    assert Path("/extra/deps_a/folder") in candidates
    assert Path("/extra/deps_b/folder") in candidates


def test_list_deposition_directories_includes_extra_roots(monkeypatch, tmp_path):
    primary = tmp_path / "primary"
    primary.mkdir()
    (primary / "a.txt").write_text("a", encoding="utf-8")

    extra = tmp_path / "extra"
    extra.mkdir()
    (extra / "e.txt").write_text("e", encoding="utf-8")
    extra_child = extra / "child"
    extra_child.mkdir()
    (extra_child / "c.txt").write_text("c", encoding="utf-8")

    monkeypatch.setattr(main, "_resolve_directory_base", lambda: ("configured", primary))
    monkeypatch.setattr(
        main,
        "settings",
        SimpleNamespace(deposition_dir=str(primary), deposition_extra_dirs=str(extra)),
    )

    base, options = main._list_deposition_directories()

    assert base == primary
    option_paths = {item.path for item in options}
    assert str(primary) in option_paths
    assert str(extra) in option_paths
    assert str(extra_child) in option_paths


def test_list_deposition_directories_skips_duplicate_extra_root(monkeypatch, tmp_path):
    root = tmp_path / "deps"
    root.mkdir()
    (root / "a.txt").write_text("a", encoding="utf-8")
    (root / "child").mkdir()
    (root / "child" / "b.txt").write_text("b", encoding="utf-8")

    monkeypatch.setattr(main, "_resolve_directory_base", lambda: ("configured", root))
    monkeypatch.setattr(main, "_configured_extra_deposition_roots", lambda: [root])

    base, options = main._list_deposition_directories()
    assert base == root
    assert len([item for item in options if item.path == str(root)]) == 1


def test_cached_extra_deposition_roots_loaded_from_couch(monkeypatch):
    couchdb = Mock()
    couchdb.find.return_value = [
        {"type": "deposition_root", "path": "/opt/depositions_a"},
        {"type": "deposition_root", "path": "relative_deps"},
        {"type": "deposition_root", "path": ""},
    ]
    monkeypatch.setattr(main, "couchdb", couchdb)

    roots = main._cached_extra_deposition_roots()

    assert Path("/opt/depositions_a") in roots
    assert Path.cwd() / "relative_deps" in roots


def test_cached_extra_deposition_roots_handles_find_errors(monkeypatch):
    couchdb = Mock()
    couchdb.find.side_effect = RuntimeError("db down")
    monkeypatch.setattr(main, "couchdb", couchdb)

    assert main._cached_extra_deposition_roots() == []


def test_deposition_root_doc_id_is_stable():
    value = main._deposition_root_doc_id("/tmp/depositions")
    assert value.startswith("deposition_root:")
    assert value == main._deposition_root_doc_id("/tmp/depositions")


def test_build_ingest_candidates_includes_alias_for_extra_root(monkeypatch):
    monkeypatch.setattr(main, "app_root", Path("/workspace/app"))
    monkeypatch.setattr(
        main,
        "settings",
        SimpleNamespace(
            deposition_dir="/host/deps",
            deposition_extra_dirs="/extra/deps",
        ),
    )

    candidates = main._build_ingest_candidates("/data/depositions/oj_simpson")

    assert Path("/extra/deps/oj_simpson") in candidates
    assert Path("/extra/deps/oj_simposon") in candidates


def test_add_deposition_root_endpoint_success(monkeypatch, tmp_path):
    normalized = (tmp_path / "deps").resolve()
    normalized.mkdir()
    request = DepositionRootRequest(path=str(normalized))
    cache = Mock()

    monkeypatch.setattr(main, "_normalize_deposition_root_path", lambda _value: normalized)
    monkeypatch.setattr(main, "_cache_deposition_root", cache)

    payload = main.add_deposition_root(request)

    assert payload.path == str(normalized)
    cache.assert_called_once_with(str(normalized))


def test_add_deposition_root_endpoint_validation_errors(monkeypatch):
    with pytest.raises(HTTPException) as blank_exc:
        main.add_deposition_root(DepositionRootRequest(path=" "))
    assert blank_exc.value.status_code == 400

    file_path = Path(__file__).resolve()
    monkeypatch.setattr(main, "_normalize_deposition_root_path", lambda _value: file_path)
    with pytest.raises(HTTPException) as file_exc:
        main.add_deposition_root(DepositionRootRequest(path=str(file_path)))
    assert file_exc.value.status_code == 400
    assert "must be a directory" in file_exc.value.detail

    missing_dir = Path("/definitely/missing/deposition_root")
    monkeypatch.setattr(main, "_normalize_deposition_root_path", lambda _value: missing_dir)
    with pytest.raises(HTTPException) as missing_exc:
        main.add_deposition_root(DepositionRootRequest(path=str(missing_dir)))
    assert missing_exc.value.status_code == 400
    assert "does not exist in API runtime" in missing_exc.value.detail


def test_add_deposition_root_endpoint_normalize_permission_and_cache_failures(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "_normalize_deposition_root_path", Mock(side_effect=ValueError("bad path")))
    with pytest.raises(HTTPException) as normalize_exc:
        main.add_deposition_root(DepositionRootRequest(path="/tmp/invalid"))
    assert normalize_exc.value.status_code == 400
    assert "Invalid deposition root path" in normalize_exc.value.detail

    class UnreadableDir:
        def __str__(self):
            return "/tmp/unreadable"

        def exists(self):
            return True

        def is_dir(self):
            return True

        def iterdir(self):
            raise PermissionError("denied")

    monkeypatch.setattr(main, "_normalize_deposition_root_path", lambda _value: UnreadableDir())
    with pytest.raises(HTTPException) as permission_exc:
        main.add_deposition_root(DepositionRootRequest(path="/tmp/unreadable"))
    assert permission_exc.value.status_code == 400
    assert "not readable" in permission_exc.value.detail

    readable = tmp_path / "readable"
    readable.mkdir()
    monkeypatch.setattr(main, "_normalize_deposition_root_path", lambda _value: readable)
    monkeypatch.setattr(main, "_cache_deposition_root", Mock(side_effect=RuntimeError("save failed")))
    with pytest.raises(HTTPException) as cache_exc:
        main.add_deposition_root(DepositionRootRequest(path=str(readable)))
    assert cache_exc.value.status_code == 502
    assert "Failed to cache deposition root" in cache_exc.value.detail


def test_normalize_deposition_root_path_resolves_absolute_and_relative(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    relative = main._normalize_deposition_root_path("relative/path")
    absolute = main._normalize_deposition_root_path(str(tmp_path))

    assert relative == (tmp_path / "relative/path").resolve()
    assert absolute == tmp_path.resolve()


def test_cache_deposition_root_creates_and_updates_existing(monkeypatch):
    couchdb = Mock()
    monkeypatch.setattr(main, "couchdb", couchdb)
    monkeypatch.setattr(main, "_utc_now_iso", lambda: "2026-02-26T00:00:00+00:00")

    couchdb.get_doc.side_effect = RuntimeError("missing")
    main._cache_deposition_root("/tmp/deps")
    created_doc = couchdb.update_doc.call_args.args[0]
    assert created_doc["path"] == "/tmp/deps"
    assert "_rev" not in created_doc

    couchdb.reset_mock()
    couchdb.get_doc.side_effect = None
    couchdb.get_doc.return_value = {"_rev": "2-a", "created_at": "2026-01-01T00:00:00+00:00"}
    main._cache_deposition_root("/tmp/deps")
    updated_doc = couchdb.update_doc.call_args.args[0]
    assert updated_doc["_rev"] == "2-a"
    assert updated_doc["created_at"] == "2026-01-01T00:00:00+00:00"


def test_trace_session_endpoints_save_and_delete(monkeypatch):
    main._trace_sessions.clear()
    trace_db = Mock()
    trace_db.update_doc.side_effect = lambda doc: {**doc, "_rev": "1-a"}
    trace_db.get_doc.side_effect = RuntimeError("missing")
    monkeypatch.setattr(main, "trace_couchdb", trace_db)
    main._append_trace_events(
        "trace-1",
        status="running",
        legal_clerk=[
            {
                "persona": "Persona:Legal Clerk",
                "phase": "ingest_start",
                "notes": "start",
            }
        ],
    )

    snapshot = main.get_trace_session("trace-1")
    assert snapshot.thought_stream_id == "trace-1"
    assert snapshot.status == "running"
    assert len(snapshot.thought_stream.legal_clerk) == 1

    save_memory = Mock()
    upsert = Mock()
    monkeypatch.setattr(main, "_save_case_memory", save_memory)
    monkeypatch.setattr(main, "_upsert_case_doc", upsert)
    monkeypatch.setattr(main, "_safe_case_depositions", lambda _case_id: [])

    save_response = main.save_trace_session(
        "trace-1",
        SaveTraceRequest(case_id="CASE-1", channel="ingest"),
    )
    assert save_response.saved is True
    assert save_response.thought_stream_id == "trace-1"
    save_memory.assert_called_once()
    upsert.assert_called_once()

    trace_db.get_doc.side_effect = None
    trace_db.get_doc.return_value = {"_id": main._trace_doc_id("trace-1"), "_rev": "1-a"}
    delete_response = main.delete_trace_session("trace-1")
    assert delete_response.deleted is True
    assert delete_response.thought_stream_id == "trace-1"

    trace_db.get_doc.side_effect = RuntimeError("missing")
    with pytest.raises(HTTPException) as missing_trace:
        main.get_trace_session("trace-1")
    assert missing_trace.value.status_code == 404


def test_public_trace_session_and_trace_sequence_helpers():
    payload = main._public_trace_session(
        {
            "trace_id": "trace-1",
            "status": "invalid",
            "trace": {"legal_clerk": [], "attorney": []},
            "updated_at": "2026-02-26T00:00:00+00:00",
        }
    )
    assert payload["status"] == "running"

    seq = main._max_trace_sequence(
        {
            "legal_clerk": [{"sequence": "bad"}],
            "attorney": [{"sequence": "5"}],
        }
    )
    assert seq == 5


def test_load_and_ensure_trace_session_branches(monkeypatch):
    trace_db = Mock()
    monkeypatch.setattr(main, "trace_couchdb", trace_db)
    main._trace_sessions.clear()

    trace_db.get_doc.return_value = "not-dict"
    assert main._load_trace_session("trace-1") is None

    trace_db.get_doc.return_value = {
        "_id": "doc-1",
        "_rev": "1-a",
        "trace_id": "trace-1",
        "status": "completed",
        "trace": "not-dict",
    }
    loaded = main._load_trace_session("trace-1")
    assert loaded is not None
    assert loaded["trace"]["legal_clerk"] == []
    assert loaded["trace"]["attorney"] == []

    main._trace_sessions["trace-memory"] = {"trace_id": "trace-memory"}
    assert main._ensure_trace_session("trace-memory") == {"trace_id": "trace-memory"}

    monkeypatch.setattr(main, "_load_trace_session", lambda _trace_id: {"trace_id": "trace-loaded"})
    ensured = main._ensure_trace_session("trace-loaded")
    assert ensured == {"trace_id": "trace-loaded"}
    main._trace_sessions.clear()


def test_flush_trace_session_and_snapshot_branches(monkeypatch):
    main._trace_sessions.clear()
    trace_db = Mock()
    monkeypatch.setattr(main, "trace_couchdb", trace_db)

    # Missing session early return.
    main._flush_trace_session("missing")

    # dirty_count <= 0 early return.
    main._trace_sessions["trace-early"] = {
        "_dirty_count": 0,
        "_last_flush_monotonic": 0.0,
    }
    main._flush_trace_session("trace-early", force=False)

    # dirty_count < 6 and elapsed < 1.0 early return.
    monkeypatch.setattr(main, "monotonic", lambda: 10.5)
    main._trace_sessions["trace-small"] = {
        "_dirty_count": 1,
        "_last_flush_monotonic": 10.0,
    }
    main._flush_trace_session("trace-small", force=False)

    # _rev branch and session removed after write (line 608 race-safe return).
    monkeypatch.setattr(main, "monotonic", lambda: 20.0)
    main._trace_sessions["trace-race"] = {
        "_doc_id": "thought_stream:abc",
        "_rev": "3-c",
        "trace_id": "trace-race",
        "status": "running",
        "updated_at": "2026-02-26T00:00:00+00:00",
        "created_at": "2026-02-26T00:00:00+00:00",
        "trace": {"legal_clerk": [], "attorney": []},
        "_dirty_count": 6,
        "_last_flush_monotonic": 0.0,
    }

    def update_doc_side_effect(doc):
        assert doc["_rev"] == "3-c"
        main._trace_sessions.pop("trace-race", None)
        return {"_rev": "4-d"}

    trace_db.update_doc.side_effect = update_doc_side_effect
    main._flush_trace_session("trace-race", force=False)

    # Snapshot loads from storage and caches it.
    monkeypatch.setattr(
        main,
        "_load_trace_session",
        lambda _trace_id: {
            "trace_id": "trace-snap",
            "status": "completed",
            "updated_at": "2026-02-26T00:00:00+00:00",
            "trace": {"legal_clerk": [], "attorney": []},
        },
    )
    snapshot = main._trace_session_snapshot("trace-snap")
    assert snapshot is not None
    assert "trace-snap" in main._trace_sessions
    main._trace_sessions.clear()


def test_append_and_delete_trace_event_branches(monkeypatch):
    main._trace_sessions.clear()
    monkeypatch.setattr(main, "_flush_trace_session", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main, "_utc_now_iso", lambda: "2026-02-26T00:00:00+00:00")

    main._append_trace_events(
        "trace-2",
        attorney=[
            {
                "persona": "Persona:Attorney",
                "phase": "chat_response",
                "notes": "ok",
            }
        ],
    )
    assert len(main._trace_sessions["trace-2"]["trace"]["attorney"]) == 1

    assert main._delete_trace_session(" ") is False

    trace_db = Mock()
    trace_db.get_doc.side_effect = RuntimeError("db down")
    monkeypatch.setattr(main, "trace_couchdb", trace_db)
    assert main._delete_trace_session("trace-unknown") is False
    main._trace_sessions.clear()


def test_rag_stream_doc_id_prefix_and_uniqueness():
    first = main._rag_stream_doc_id()
    second = main._rag_stream_doc_id()
    assert first.startswith("rag_stream:")
    assert second.startswith("rag_stream:")
    assert first != second


def test_append_rag_stream_event_persists_doc(monkeypatch):
    rag_db = Mock()
    monkeypatch.setattr(main, "rag_couchdb", rag_db)
    monkeypatch.setattr(main, "_utc_now_iso", lambda: "2026-02-26T00:00:00+00:00")
    monkeypatch.setattr(main, "_rag_stream_doc_id", lambda: "rag_stream:test")

    main._append_rag_stream_event(
        trace_id="trace-1",
        question="What is breach?",
        llm_provider="openai",
        llm_model="gpt-5.2",
        use_rag=True,
        top_k=8,
        status="completed",
        phase="answer",
        retrieval_terms=["breach"],
        context_rows=2,
        sources=[{"iri": "http://example.org/Contract", "label": "Contract"}],
        context_preview="context",
        llm_system_prompt="system",
        llm_user_prompt="user",
        answer_preview="Short answer",
    )

    saved = rag_db.update_doc.call_args.args[0]
    assert saved["_id"] == "rag_stream:test"
    assert saved["type"] == "rag_stream"
    assert saved["trace_id"] == "trace-1"
    assert saved["use_rag"] is True
    assert saved["context_rows"] == 2
    assert saved["sources"][0]["label"] == "Contract"


def test_append_rag_stream_event_ignores_storage_failures(monkeypatch):
    rag_db = Mock()
    rag_db.update_doc.side_effect = RuntimeError("db down")
    monkeypatch.setattr(main, "rag_couchdb", rag_db)

    main._append_rag_stream_event(
        trace_id=None,
        question="q",
        llm_provider="openai",
        llm_model="gpt-5.2",
        use_rag=False,
        top_k=4,
        status="failed",
        phase="llm",
        error="boom",
    )


def test_thought_stream_health_success(monkeypatch):
    trace_db = Mock()
    monkeypatch.setattr(main, "trace_couchdb", trace_db)
    monkeypatch.setattr(main, "settings", SimpleNamespace(thought_stream_db="thought_stream"))

    payload = main.thought_stream_health()

    assert payload == {"connected": True, "database": "thought_stream"}
    trace_db.ensure_db.assert_called_once_with(retries=1, delay_seconds=0)


def test_thought_stream_health_failure(monkeypatch):
    trace_db = Mock()
    trace_db.ensure_db.side_effect = RuntimeError("connection refused")
    monkeypatch.setattr(main, "trace_couchdb", trace_db)
    monkeypatch.setattr(main, "settings", SimpleNamespace(thought_stream_db="thought_stream"))

    with pytest.raises(HTTPException) as exc:
        main.thought_stream_health()

    assert exc.value.status_code == 503
    assert "Thought Stream storage is unavailable" in exc.value.detail


def test_rag_stream_health_success(monkeypatch):
    rag_db = Mock()
    monkeypatch.setattr(main, "rag_couchdb", rag_db)
    monkeypatch.setattr(main, "settings", SimpleNamespace(rag_stream_db="rag-stream"))

    payload = main.rag_stream_health()

    assert payload == {"connected": True, "database": "rag-stream"}
    rag_db.ensure_db.assert_called_once_with(retries=1, delay_seconds=0)


def test_rag_stream_health_failure(monkeypatch):
    rag_db = Mock()
    rag_db.ensure_db.side_effect = RuntimeError("connection refused")
    monkeypatch.setattr(main, "rag_couchdb", rag_db)
    monkeypatch.setattr(main, "settings", SimpleNamespace(rag_stream_db="rag-stream"))

    with pytest.raises(HTTPException) as exc:
        main.rag_stream_health()

    assert exc.value.status_code == 503
    assert "RAG stream storage is unavailable" in exc.value.detail


def test_thought_stream_health_route_registered_before_dynamic_trace_route():
    api_paths = [getattr(route, "path", "") for route in main.app.routes]
    health_index = api_paths.index("/api/thought-streams/health")
    dynamic_index = api_paths.index("/api/thought-streams/{thought_stream_id}")
    assert health_index < dynamic_index


def test_graph_browser_info_returns_settings(monkeypatch):
    graph_client = SimpleNamespace(
        browser_url="http://localhost:7474/browser/",
        uri="bolt://localhost:7687",
        database="neo4j",
    )
    monkeypatch.setattr(main, "neo4j_graph", graph_client)

    payload = main.graph_browser_info()

    assert payload.browser_url == "http://localhost:7474/browser/"
    assert payload.bolt_url == "bolt://localhost:7687"
    assert payload.database == "neo4j"
    assert payload.launch_url.startswith("http://localhost:7474/browser/?")
    assert "cmd=edit" in payload.launch_url
    assert (
        "MATCH+%28n%29+OPTIONAL+MATCH+%28n%29-%5Br%5D-%3E%28m%29+RETURN+n%2C+r%2C+m+LIMIT+75%3B"
        in payload.launch_url
    )
    assert "connectURL=bolt%3A%2F%2Flocalhost%3A7687" in payload.launch_url
    assert "db=neo4j" in payload.launch_url


def test_graph_browser_launch_url_overrides_existing_query_params():
    launch_url = main._graph_browser_launch_url(
        "http://localhost:7474/browser/?cmd=play&arg=old",
        "bolt://neo4j:7687",
        "neo4j",
    )

    assert launch_url.startswith("http://localhost:7474/browser/?")
    assert "cmd=edit" in launch_url
    assert "cmd=play" not in launch_url
    assert "arg=old" not in launch_url
    assert "connectURL=bolt%3A%2F%2Flocalhost%3A7687" in launch_url
    assert "db=neo4j" in launch_url


def test_browser_reachable_bolt_url_maps_container_alias_to_browser_host():
    mapped = main._browser_reachable_bolt_url(
        "https://graph.example.com/browser/",
        "bolt://neo4j:7687",
    )
    assert mapped == "bolt://graph.example.com:7687"


def test_browser_reachable_bolt_url_keeps_public_host_unchanged():
    mapped = main._browser_reachable_bolt_url(
        "https://graph.example.com/browser/",
        "bolt://graph-bolt.example.com:7687",
    )
    assert mapped == "bolt://graph-bolt.example.com:7687"


def test_browser_reachable_bolt_url_non_bolt_scheme_and_missing_host():
    direct = main._browser_reachable_bolt_url(
        "https://graph.example.com/browser/",
        "http://graph.example.com:7687",
    )
    assert direct == "http://graph.example.com:7687"

    missing_host = main._browser_reachable_bolt_url(
        "https://graph.example.com/browser/",
        "bolt:///just-a-path",
    )
    assert missing_host == "bolt:///just-a-path"


def test_browser_reachable_bolt_url_single_label_and_userinfo_branches():
    single_label = main._browser_reachable_bolt_url(
        "https://graph.example.com/browser/",
        "bolt://cache:7687",
    )
    assert single_label == "bolt://graph.example.com:7687"

    with_userinfo = main._browser_reachable_bolt_url(
        "https://graph.example.com/browser/",
        "bolt://neo4j:secret@neo4j:7687",
    )
    assert with_userinfo == "bolt://neo4j:secret@graph.example.com:7687"


def test_graph_rag_health_uses_client_payload(monkeypatch):
    graph_client = SimpleNamespace(
        health=lambda: {
            "configured": True,
            "connected": True,
            "bolt_url": "bolt://localhost:7687",
            "database": "neo4j",
            "browser_url": "http://localhost:7474/browser/",
            "error": None,
        }
    )
    monkeypatch.setattr(main, "neo4j_graph", graph_client)

    payload = main.graph_rag_health()

    assert payload.connected is True
    assert payload.database == "neo4j"


def test_graph_rag_owl_options_returns_dropdown_payload(monkeypatch):
    base = Path("/data/ontology")
    options = [
        main.GraphOntologyOption(path="/data/ontology/*.owl", label="All OWL files"),
        main.GraphOntologyOption(path="/data/ontology/legal.owl", label="legal.owl"),
    ]
    monkeypatch.setattr(main, "_list_ontology_owl_options", lambda: (base, options, options[0].path))

    payload = main.graph_rag_owl_options()

    assert payload.base_directory == "/data/ontology"
    assert payload.suggested == "/data/ontology/*.owl"
    assert [item.path for item in payload.options] == ["/data/ontology/*.owl", "/data/ontology/legal.owl"]


def test_graph_rag_owl_browser_returns_file_browser_payload(monkeypatch):
    base = Path("/data/ontology")
    current = Path("/data/ontology/contracts")
    directories = [
        main.GraphOntologyBrowserEntry(
            path="/data/ontology/contracts/sub",
            name="sub",
            kind="directory",
        )
    ]
    files = [
        main.GraphOntologyBrowserEntry(
            path="/data/ontology/contracts/legal.owl",
            name="legal.owl",
            kind="file",
        )
    ]
    monkeypatch.setattr(main, "_resolve_ontology_browser_directory", lambda _path: (base, current))
    monkeypatch.setattr(main, "_list_ontology_browser_entries", lambda _dir: (directories, files))

    payload = main.graph_rag_owl_browser("/data/ontology/contracts")

    assert payload.base_directory == "/data/ontology"
    assert payload.current_directory == "/data/ontology/contracts"
    assert payload.parent_directory == "/data/ontology"
    assert payload.wildcard_path == "/data/ontology/contracts/*.owl"
    assert [item.path for item in payload.directories] == ["/data/ontology/contracts/sub"]
    assert [item.path for item in payload.files] == ["/data/ontology/contracts/legal.owl"]


def test_load_graph_rag_ontology_success(monkeypatch, tmp_path):
    owl = tmp_path / "legal.owl"
    owl.write_text("x", encoding="utf-8")
    monkeypatch.setattr(main, "_resolve_ontology_owl_files", lambda _path: [owl])

    graph_client = SimpleNamespace(
        load_owl_files=lambda files, clear_existing, batch_size: {
            "matched_files": [str(path) for path in files],
            "loaded_files": len(files),
            "triples": 12,
            "resource_relationships": 7,
            "literal_relationships": 5,
            "cleared": clear_existing,
            "database": "neo4j",
            "browser_url": "http://localhost:7474/browser/",
        }
    )
    monkeypatch.setattr(main, "neo4j_graph", graph_client)

    payload = main.load_graph_rag_ontology(
        GraphOntologyLoadRequest(path="/data/ontology/*.owl", clear_existing=True, batch_size=700)
    )

    assert payload.path == "/data/ontology/*.owl"
    assert payload.loaded_files == 1
    assert payload.triples == 12
    assert payload.cleared is True


def test_load_graph_rag_ontology_validation_and_error_paths(monkeypatch):
    with pytest.raises(HTTPException) as empty_exc:
        main.load_graph_rag_ontology(GraphOntologyLoadRequest(path="   "))
    assert empty_exc.value.status_code == 400

    monkeypatch.setattr(main, "_resolve_ontology_owl_files", lambda _path: [Path("/tmp/legal.owl")])

    runtime_graph = SimpleNamespace(
        load_owl_files=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("Neo4j unavailable"))
    )
    monkeypatch.setattr(main, "neo4j_graph", runtime_graph)
    with pytest.raises(HTTPException) as runtime_exc:
        main.load_graph_rag_ontology(GraphOntologyLoadRequest(path="/data/ontology/*.owl"))
    assert runtime_exc.value.status_code == 503

    generic_graph = SimpleNamespace(
        load_owl_files=lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("parse failed"))
    )
    monkeypatch.setattr(main, "neo4j_graph", generic_graph)
    with pytest.raises(HTTPException) as generic_exc:
        main.load_graph_rag_ontology(GraphOntologyLoadRequest(path="/data/ontology/*.owl"))
    assert generic_exc.value.status_code == 502
    assert "Failed to import ontology files into Neo4j" in generic_exc.value.detail


def test_query_graph_rag_success(monkeypatch):
    trace_events: list[dict] = []
    rag_events: list[dict] = []
    retrieval = Mock(
        return_value={
            "resource_count": 2,
            "resources": [
                {
                    "iri": "http://example.org/Contract",
                    "label": "Contract",
                    "relations": [
                        {
                            "predicate": "relatedTo",
                            "object_label": "Offer",
                            "object_iri": "http://example.org/Offer",
                        }
                    ],
                    "literals": [
                        {
                            "predicate": "definition",
                            "value": "Contract is agreement",
                            "datatype": "",
                            "lang": "en",
                        }
                    ],
                },
                {"iri": "http://example.org/Breach", "label": "Breach", "relations": [], "literals": []},
            ],
            "terms": ["breach", "contract"],
            "context_text": "Resource: Contract",
        }
    )
    monkeypatch.setattr(main, "neo4j_graph", SimpleNamespace(retrieve_context=retrieval))
    monkeypatch.setattr(main, "_resolve_request_llm", lambda _provider, _model: ("openai", "gpt-5.2"))
    monkeypatch.setattr(main, "_ensure_request_llm_operational", lambda _provider, _model: None)
    monkeypatch.setattr(
        main,
        "_append_trace_events",
        lambda trace_id, **kwargs: trace_events.append({"trace_id": trace_id, **kwargs}),
    )
    monkeypatch.setattr(main, "_append_rag_stream_event", lambda **kwargs: rag_events.append(kwargs))
    monkeypatch.setattr(main, "render_prompt", lambda _name, **kwargs: str(kwargs))
    monkeypatch.setattr(
        main,
        "build_chat_model",
        lambda *_args, **_kwargs: SimpleNamespace(
            invoke=lambda _messages: SimpleNamespace(content="Short answer: Ontology response")
        ),
    )

    payload = main.query_graph_rag(
        GraphRagQueryRequest(
            question="What is breach of contract?",
            top_k=6,
            llm_provider="openai",
            llm_model="gpt-5.2",
            thought_stream_id="trace-graph-1",
        )
    )

    retrieval.assert_called_once_with("What is breach of contract?", node_limit=6)
    assert payload.context_rows == 2
    assert payload.llm_provider == "openai"
    assert payload.llm_model == "gpt-5.2"
    assert payload.sources[0].iri == "http://example.org/Contract"
    assert payload.answer.startswith("Short answer:")
    assert payload.monitor is not None
    assert payload.monitor.rag_enabled is True
    assert payload.monitor.rag_stream_enabled is True
    assert payload.monitor.retrieval_terms == ["breach", "contract"]
    assert len(payload.monitor.retrieved_resources) == 2
    assert payload.monitor.retrieved_resources[0].iri == "http://example.org/Contract"
    assert payload.monitor.retrieved_resources[0].relations[0].predicate == "relatedTo"
    assert payload.monitor.context_preview == "Resource: Contract"
    assert "question" in payload.monitor.llm_user_prompt
    assert payload.monitor.llm_system_prompt == "{}"

    assert [item["trace_id"] for item in trace_events] == ["trace-graph-1", "trace-graph-1", "trace-graph-1"]
    assert trace_events[0]["status"] == "running"
    assert trace_events[2]["status"] == "completed"
    assert trace_events[0]["legal_clerk"][0]["phase"] == "graph_rag_retrieval_start"
    assert trace_events[1]["legal_clerk"][0]["phase"] == "graph_rag_context_ready"
    assert trace_events[2]["attorney"][0]["phase"] == "graph_rag_answer"
    assert rag_events[-1]["status"] == "completed"
    assert rag_events[-1]["use_rag"] is True
    assert rag_events[-1]["context_rows"] == 2


def test_query_graph_rag_with_rag_disabled(monkeypatch):
    trace_events: list[dict] = []
    rag_events: list[dict] = []
    retrieval = Mock()
    monkeypatch.setattr(main, "neo4j_graph", SimpleNamespace(retrieve_context=retrieval))
    monkeypatch.setattr(main, "_resolve_request_llm", lambda _provider, _model: ("openai", "gpt-5.2"))
    monkeypatch.setattr(main, "_ensure_request_llm_operational", lambda _provider, _model: None)
    monkeypatch.setattr(
        main,
        "_append_trace_events",
        lambda trace_id, **kwargs: trace_events.append({"trace_id": trace_id, **kwargs}),
    )
    monkeypatch.setattr(main, "_append_rag_stream_event", lambda **kwargs: rag_events.append(kwargs))
    monkeypatch.setattr(main, "render_prompt", lambda _name, **kwargs: str(kwargs))
    monkeypatch.setattr(
        main,
        "build_chat_model",
        lambda *_args, **_kwargs: SimpleNamespace(
            invoke=lambda _messages: SimpleNamespace(content="Short answer: No retrieval.")
        ),
    )

    payload = main.query_graph_rag(
        GraphRagQueryRequest(
            question="What is breach of contract?",
            top_k=6,
            use_rag=False,
            llm_provider="openai",
            llm_model="gpt-5.2",
        )
    )

    retrieval.assert_not_called()
    assert payload.context_rows == 0
    assert payload.monitor is not None
    assert payload.monitor.rag_enabled is False
    assert payload.monitor.rag_stream_enabled is True
    assert payload.monitor.retrieved_resources == []
    assert "RAG processing was disabled" in payload.monitor.context_preview
    assert rag_events[-1]["status"] == "completed"
    assert rag_events[-1]["use_rag"] is False
    assert trace_events[1]["legal_clerk"][0]["phase"] == "graph_rag_disabled"


def test_query_graph_rag_with_stream_logging_disabled(monkeypatch):
    trace_events: list[dict] = []
    rag_events: list[dict] = []
    retrieval = Mock(
        return_value={
            "resource_count": 1,
            "resources": [{"iri": "http://example.org/Contract", "label": "Contract"}],
            "terms": ["contract"],
            "context_text": "Resource: Contract",
        }
    )
    monkeypatch.setattr(main, "neo4j_graph", SimpleNamespace(retrieve_context=retrieval))
    monkeypatch.setattr(main, "_resolve_request_llm", lambda _provider, _model: ("openai", "gpt-5.2"))
    monkeypatch.setattr(main, "_ensure_request_llm_operational", lambda _provider, _model: None)
    monkeypatch.setattr(
        main,
        "_append_trace_events",
        lambda trace_id, **kwargs: trace_events.append({"trace_id": trace_id, **kwargs}),
    )
    monkeypatch.setattr(main, "_append_rag_stream_event", lambda **kwargs: rag_events.append(kwargs))
    monkeypatch.setattr(main, "render_prompt", lambda _name, **kwargs: str(kwargs))
    monkeypatch.setattr(
        main,
        "build_chat_model",
        lambda *_args, **_kwargs: SimpleNamespace(
            invoke=lambda _messages: SimpleNamespace(content="Short answer: No stream logging.")
        ),
    )

    payload = main.query_graph_rag(
        GraphRagQueryRequest(
            question="What is contract breach?",
            top_k=6,
            use_rag=True,
            stream_rag=False,
            llm_provider="openai",
            llm_model="gpt-5.2",
        )
    )

    assert payload.monitor is not None
    assert payload.monitor.rag_enabled is True
    assert payload.monitor.rag_stream_enabled is False
    assert len(rag_events) == 0
    assert trace_events[2]["status"] == "completed"


def test_query_graph_rag_toggle_influences_answer(monkeypatch):
    retrieval = Mock(
        return_value={
            "resource_count": 1,
            "resources": [
                {
                    "iri": "http://example.org/MaterialBreach",
                    "label": "Material Breach",
                    "relations": [],
                    "literals": [],
                }
            ],
            "terms": ["material", "breach"],
            "context_text": "Fact: A material breach is an uncured failure of a core contract obligation.",
        }
    )
    monkeypatch.setattr(main, "neo4j_graph", SimpleNamespace(retrieve_context=retrieval))
    monkeypatch.setattr(main, "_resolve_request_llm", lambda _provider, _model: ("openai", "gpt-5.2"))
    monkeypatch.setattr(main, "_ensure_request_llm_operational", lambda _provider, _model: None)
    monkeypatch.setattr(main, "_append_trace_events", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main, "_append_rag_stream_event", lambda **_kwargs: None)
    monkeypatch.setattr(main, "render_prompt", lambda _name, **kwargs: str(kwargs))

    class _ContextAwareModel:
        def invoke(self, messages):
            prompt = str(messages[-1].content)
            if "RAG processing was disabled for this request." in prompt:
                return SimpleNamespace(content="Short answer: Graph context was disabled; answer confidence is low.")
            if "material breach is an uncured failure" in prompt:
                return SimpleNamespace(content="Short answer: A material breach is an uncured failure of a core obligation.")
            return SimpleNamespace(content="Short answer: No context.")

    monkeypatch.setattr(main, "build_chat_model", lambda *_args, **_kwargs: _ContextAwareModel())

    rag_on = main.query_graph_rag(
        GraphRagQueryRequest(
            question="What is a material breach?",
            top_k=5,
            use_rag=True,
            stream_rag=False,
            llm_provider="openai",
            llm_model="gpt-5.2",
        )
    )
    rag_off = main.query_graph_rag(
        GraphRagQueryRequest(
            question="What is a material breach?",
            top_k=5,
            use_rag=False,
            stream_rag=False,
            llm_provider="openai",
            llm_model="gpt-5.2",
        )
    )

    assert retrieval.call_count == 1
    assert rag_on.answer != rag_off.answer
    assert "uncured failure" in rag_on.answer
    assert "disabled" in rag_off.answer.lower()
    assert rag_on.context_rows == 1
    assert rag_off.context_rows == 0
    assert rag_on.monitor is not None
    assert rag_off.monitor is not None
    assert "material breach is an uncured failure" in rag_on.monitor.context_preview.lower()
    assert "RAG processing was disabled" in rag_off.monitor.context_preview


def test_query_graph_rag_validation_and_error_paths(monkeypatch):
    with pytest.raises(HTTPException) as empty_exc:
        main.query_graph_rag(GraphRagQueryRequest(question="  "))
    assert empty_exc.value.status_code == 400

    monkeypatch.setattr(main, "_resolve_request_llm", lambda _provider, _model: ("openai", "gpt-5.2"))
    monkeypatch.setattr(main, "_ensure_request_llm_operational", lambda _provider, _model: None)
    monkeypatch.setattr(main, "_append_rag_stream_event", lambda **_kwargs: None)
    monkeypatch.setattr(main, "render_prompt", lambda _name, **kwargs: str(kwargs))

    bad_request_graph = SimpleNamespace(
        retrieve_context=lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad query"))
    )
    monkeypatch.setattr(main, "neo4j_graph", bad_request_graph)
    with pytest.raises(HTTPException) as bad_request_exc:
        main.query_graph_rag(GraphRagQueryRequest(question="contract?"))
    assert bad_request_exc.value.status_code == 400
    assert "bad query" in bad_request_exc.value.detail

    runtime_graph = SimpleNamespace(
        retrieve_context=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("Neo4j unavailable"))
    )
    monkeypatch.setattr(main, "neo4j_graph", runtime_graph)
    with pytest.raises(HTTPException) as runtime_exc:
        main.query_graph_rag(GraphRagQueryRequest(question="contract?"))
    assert runtime_exc.value.status_code == 503

    generic_graph = SimpleNamespace(
        retrieve_context=lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyError("boom"))
    )
    monkeypatch.setattr(main, "neo4j_graph", generic_graph)
    with pytest.raises(HTTPException) as generic_exc:
        main.query_graph_rag(GraphRagQueryRequest(question="contract?"))
    assert generic_exc.value.status_code == 502

    monkeypatch.setattr(
        main,
        "neo4j_graph",
        SimpleNamespace(
            retrieve_context=lambda *_args, **_kwargs: {
                "resource_count": 0,
                "resources": [],
                "context_text": "No matching ontology context found.",
            }
        ),
    )
    monkeypatch.setattr(
        main,
        "build_chat_model",
        lambda *_args, **_kwargs: SimpleNamespace(
            invoke=lambda _messages: (_ for _ in ()).throw(RuntimeError("llm down"))
        ),
    )
    monkeypatch.setattr(main, "llm_failure_message", lambda *_args, **_kwargs: "LLM failed")
    with pytest.raises(HTTPException) as llm_exc:
        main.query_graph_rag(GraphRagQueryRequest(question="contract?"))
    assert llm_exc.value.status_code == 502
    assert llm_exc.value.detail == "LLM failed"


def test_query_graph_rag_fallback_when_llm_content_is_empty(monkeypatch):
    monkeypatch.setattr(main, "_resolve_request_llm", lambda _provider, _model: ("openai", "gpt-5.2"))
    monkeypatch.setattr(main, "_ensure_request_llm_operational", lambda _provider, _model: None)
    monkeypatch.setattr(main, "_append_rag_stream_event", lambda **_kwargs: None)
    monkeypatch.setattr(
        main,
        "neo4j_graph",
        SimpleNamespace(
            retrieve_context=lambda *_args, **_kwargs: {
                "resource_count": 0,
                "resources": [],
                "context_text": "No matching ontology context found.",
            }
        ),
    )
    monkeypatch.setattr(main, "render_prompt", lambda _name, **kwargs: str(kwargs))
    monkeypatch.setattr(
        main,
        "build_chat_model",
        lambda *_args, **_kwargs: SimpleNamespace(invoke=lambda _messages: SimpleNamespace(content="  ")),
    )

    payload = main.query_graph_rag(GraphRagQueryRequest(question="contract?"))

    assert payload.answer == "Short answer: No answer could be generated from current ontology context."
    assert payload.context_rows == 0


def test_save_and_delete_trace_session_endpoint_validation_and_not_found(monkeypatch):
    with pytest.raises(HTTPException) as trace_exc:
        main.save_trace_session(" ", SaveTraceRequest(case_id="CASE-1", channel="ingest"))
    assert trace_exc.value.status_code == 400
    assert "Thought stream ID is required" in trace_exc.value.detail

    with pytest.raises(HTTPException) as case_exc:
        main.save_trace_session("trace-1", SaveTraceRequest(case_id=" ", channel="ingest"))
    assert case_exc.value.status_code == 400
    assert "Case ID is required" in case_exc.value.detail

    monkeypatch.setattr(main, "_flush_trace_session", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main, "_trace_session_snapshot", lambda _trace_id: None)
    with pytest.raises(HTTPException) as missing_exc:
        main.save_trace_session("trace-1", SaveTraceRequest(case_id="CASE-1", channel="ingest"))
    assert missing_exc.value.status_code == 404

    monkeypatch.setattr(main, "_delete_trace_session", lambda _trace_id: False)
    with pytest.raises(HTTPException) as delete_exc:
        main.delete_trace_session("trace-1")
    assert delete_exc.value.status_code == 404


def test_collect_runtime_trace_sessions_prefers_memory_and_handles_db_failure(monkeypatch):
    main._trace_sessions.clear()
    trace_db = Mock()
    trace_db.find.side_effect = RuntimeError("db unavailable")
    monkeypatch.setattr(main, "trace_couchdb", trace_db)

    main._trace_sessions["trace-live"] = {
        "trace_id": "trace-live",
        "status": "running",
        "created_at": "2026-02-26T00:00:00+00:00",
        "updated_at": "2026-02-26T00:00:03+00:00",
        "trace": {"legal_clerk": [], "attorney": []},
    }

    sessions, storage_connected = main._collect_runtime_trace_sessions()

    assert storage_connected is False
    assert len(sessions) == 1
    assert sessions[0]["trace_id"] == "trace-live"
    main._trace_sessions.clear()


def test_metrics_helper_primitives_cover_edge_paths():
    assert main._normalize_trace_status(" completed ") == "completed"
    assert main._normalize_trace_status("unknown") == "running"

    assert main._parse_iso_datetime("") is None
    assert main._parse_iso_datetime("not-a-date") is None
    naive = main._parse_iso_datetime("2026-02-26T10:00:00")
    assert naive is not None
    assert naive.tzinfo is not None

    flattened = main._flatten_trace_events(
        {
            "legal_clerk": [{"phase": "a", "sequence": "bad-seq", "at": "2026-02-26T00:00:02+00:00"}],
            "attorney": [{"phase": "b", "sequence": 1, "at": "2026-02-26T00:00:01+00:00"}],
        }
    )
    assert [item["phase"] for item in flattened] == ["a", "b"]

    assert main._percentile([], 95) is None
    assert main._percentile([3.5], 95) == 3.5
    assert main._status_low_is_bad(92, warn_below=95, bad_below=90) == "warn"
    assert main._status_low_is_bad(96, warn_below=95, bad_below=90) == "good"
    assert main._status_band(25, warn_low=4, warn_high=18, bad_low=2, bad_high=24) == "bad"
    assert main._status_band(3, warn_low=4, warn_high=18, bad_low=2, bad_high=24) == "warn"
    assert main._status_band(10, warn_low=4, warn_high=18, bad_low=2, bad_high=24) == "good"


def test_normalize_metrics_session_defaults_and_missing_trace_id():
    assert main._normalize_metrics_session({"trace_id": " "}) is None

    payload = main._normalize_metrics_session({"trace_id": "trace-1", "trace": "invalid"})
    assert payload is not None
    assert payload["trace_id"] == "trace-1"
    assert payload["trace"]["legal_clerk"] == []
    assert payload["trace"]["attorney"] == []


def test_collect_runtime_trace_sessions_merges_persisted_with_memory(monkeypatch):
    main._trace_sessions.clear()
    trace_db = Mock()
    trace_db.find.return_value = [
        "skip",
        {
            "trace_id": "trace-1",
            "status": "completed",
            "created_at": "2026-02-26T00:00:00+00:00",
            "updated_at": "2026-02-26T00:00:03+00:00",
            "trace": {"legal_clerk": [], "attorney": []},
        },
        {"trace_id": " ", "trace": {"legal_clerk": [], "attorney": []}},
    ]
    monkeypatch.setattr(main, "trace_couchdb", trace_db)

    main._trace_sessions["trace-1"] = {
        "trace_id": "trace-1",
        "status": "running",
        "created_at": "2026-02-26T00:00:00+00:00",
        "updated_at": "2026-02-26T00:00:04+00:00",
        "trace": {"legal_clerk": [], "attorney": []},
    }
    main._trace_sessions["bad-session"] = {"trace_id": " "}
    main._trace_sessions["nondict"] = 42

    sessions, storage_connected = main._collect_runtime_trace_sessions()

    assert storage_connected is True
    by_id = {item["trace_id"]: item for item in sessions}
    assert by_id["trace-1"]["status"] == "running"
    main._trace_sessions.clear()


def test_compute_agent_runtime_metrics_calculates_thresholded_kpis():
    now = datetime.now(timezone.utc)
    created = now - timedelta(minutes=10)
    created_iso = created.isoformat()
    updated_complete_iso = (created + timedelta(seconds=10)).isoformat()
    updated_failed_iso = (created + timedelta(seconds=90)).isoformat()
    first_event_iso = (created + timedelta(seconds=2)).isoformat()
    repeated_event_iso = (created + timedelta(seconds=5)).isoformat()
    sessions = [
        {
            "trace_id": "trace-complete",
            "status": "completed",
            "created_at": created_iso,
            "updated_at": updated_complete_iso,
            "trace": {
                "legal_clerk": [
                    {
                        "persona": "Persona:Legal Clerk",
                        "phase": "start",
                        "sequence": 1,
                        "at": first_event_iso,
                    }
                ],
                "attorney": [],
            },
        },
        {
            "trace_id": "trace-failed",
            "status": "failed",
            "created_at": created_iso,
            "updated_at": updated_failed_iso,
            "trace": {
                "legal_clerk": [],
                "attorney": [
                    {
                        "persona": "Persona:Attorney",
                        "phase": "step",
                        "sequence": idx + 1,
                        "at": repeated_event_iso,
                    }
                    for idx in range(22)
                ],
            },
        },
        {
            "trace_id": "trace-running",
            "status": "running",
            "created_at": created_iso,
            "updated_at": repeated_event_iso,
            "trace": {"legal_clerk": [], "attorney": []},
        },
    ]

    payload = main._compute_agent_runtime_metrics(
        sessions,
        lookback_hours=24,
        storage_connected=True,
        rag_events=[],
        rag_storage_connected=True,
    )

    assert payload.sampled_runs == 3
    assert payload.finished_runs == 2
    assert payload.running_runs == 1

    by_key = {item.key: item for item in payload.metrics}
    correctness_by_key = {item.key: item for item in payload.correctness_metrics}
    assert by_key["task_success_rate_pct"].display == "50.0%"
    assert by_key["task_success_rate_pct"].status == "bad"
    assert by_key["run_failure_rate_pct"].display == "50.0%"
    assert by_key["run_failure_rate_pct"].status == "bad"
    assert by_key["loop_risk_rate_pct"].status == "bad"
    assert by_key["in_flight_runs"].display == "1"
    assert by_key["rag_toggle_comparison_pairs"].display == "0"
    assert correctness_by_key["golden_set_accuracy"].display == "50.0%"
    assert correctness_by_key["schema_adherence_rate"].display == "50.0%"
    assert correctness_by_key["repeat_prompt_inconsistency"].display == "0.0%"
    assert payload.rag_sampled_queries == 0
    assert payload.rag_paired_comparisons == 0
    assert payload.rag_storage_connected is True


def test_compute_agent_runtime_metrics_handles_empty_finished_runs():
    now = datetime.now(timezone.utc)
    old = now - timedelta(days=7)
    recent = now - timedelta(minutes=20)
    sessions = [
        "skip",
        {
            "trace_id": "trace-old",
            "status": "completed",
            "created_at": old.isoformat(),
            "updated_at": (old + timedelta(seconds=1)).isoformat(),
            "trace": {"legal_clerk": [], "attorney": []},
        },
        {
            "trace_id": "trace-running",
            "status": "running",
            "created_at": recent.isoformat(),
            "updated_at": None,
            "trace": {"legal_clerk": [], "attorney": []},
        },
    ]

    payload = main._compute_agent_runtime_metrics(
        sessions,
        lookback_hours=1,
        storage_connected=False,
        rag_events=[],
        rag_storage_connected=False,
    )

    assert payload.storage_connected is False
    assert payload.rag_storage_connected is False
    assert payload.finished_runs == 0
    assert payload.running_runs == 1
    by_key = {item.key: item for item in payload.metrics}
    correctness_by_key = {item.key: item for item in payload.correctness_metrics}
    assert by_key["task_success_rate_pct"].status == "info"
    assert by_key["run_failure_rate_pct"].status == "info"
    assert by_key["p95_end_to_end_latency_sec"].status == "info"
    assert by_key["p95_time_to_first_event_sec"].status == "info"
    assert by_key["avg_steps_per_finished_run"].status == "info"
    assert by_key["loop_risk_rate_pct"].status == "info"
    assert correctness_by_key["golden_set_accuracy"].display == "N/A"
    assert correctness_by_key["schema_adherence_rate"].display == "N/A"
    assert correctness_by_key["unsupported_claim_rate"].display == "N/A"


def test_collect_recent_rag_stream_events_and_compute_influence_metrics(monkeypatch):
    now = datetime.now(timezone.utc)
    old = now - timedelta(hours=30)
    rag_db = Mock()
    rag_db.find.return_value = [
        "skip",
        {
            "type": "rag_stream",
            "phase": "answer",
            "status": "completed",
            "created_at": now.isoformat(),
            "question": "What is material breach?",
            "answer_preview": "A material breach is uncured failure.",
            "use_rag": True,
            "context_rows": 3,
            "context_preview": "abc",
        },
        {
            "type": "rag_stream",
            "phase": "answer",
            "status": "completed",
            "created_at": now.isoformat(),
            "question": "What is material breach?",
            "answer_preview": "Unable to ground answer with graph context disabled.",
            "use_rag": False,
            "context_rows": 0,
            "context_preview": "",
        },
        {
            "type": "rag_stream",
            "phase": "answer",
            "status": "completed",
            "created_at": old.isoformat(),
            "question": "Old event",
            "answer_preview": "stale",
            "use_rag": True,
            "context_rows": 1,
            "context_preview": "stale",
        },
    ]
    monkeypatch.setattr(main, "rag_couchdb", rag_db)

    events, connected = main._collect_recent_rag_stream_events(24)

    assert connected is True
    assert len(events) == 2
    metrics, paired = main._compute_rag_influence_metrics(events)
    assert paired == 1
    by_key = {item.key: item for item in metrics}
    assert by_key["rag_toggle_comparison_pairs"].display == "1"
    assert by_key["rag_answer_change_rate_pct"].display == "100.0%"
    assert by_key["rag_context_hit_rate_pct"].display == "100.0%"
    assert by_key["rag_avg_context_rows_on"].display == "3.00"
    assert by_key["rag_avg_context_bytes_on"].display == "3 B"
    assert by_key["rag_completed_queries_split"].display == "1/1"


def test_collect_recent_rag_stream_events_and_influence_branch_edges(monkeypatch):
    now = datetime.now(timezone.utc)
    rag_db = Mock()
    rag_db.find.return_value = [
        {
            "type": "rag_stream",
            "phase": "answer",
            "status": "completed",
            "created_at": now.isoformat(),
            "question": "   ",
            "answer_preview": "blank question should be ignored for pairing",
            "use_rag": True,
            "context_rows": "not-an-int",
            "context_preview": "size-check",
        },
        {
            "type": "rag_stream",
            "phase": "answer",
            "status": "completed",
            "created_at": None,
            "question": "Define estoppel",
            "answer_preview": "ON older answer",
            "use_rag": True,
            "context_rows": 1,
            "context_preview": "old",
        },
        {
            "type": "rag_stream",
            "phase": "answer",
            "status": "completed",
            "created_at": now.isoformat(),
            "question": "Define estoppel",
            "answer_preview": "ON newer answer",
            "use_rag": True,
            "context_rows": 2,
            "context_preview": "newer context",
        },
        {
            "type": "rag_stream",
            "phase": "answer",
            "status": "completed",
            "created_at": now.isoformat(),
            "question": "Define estoppel",
            "answer_preview": "OFF baseline answer",
            "use_rag": False,
            "context_rows": 0,
            "context_preview": "",
        },
        {
            "type": "rag_stream",
            "phase": "answer",
            "status": "completed",
            "created_at": now.isoformat(),
            "question": "Only ON question",
            "answer_preview": "no OFF pair exists",
            "use_rag": True,
            "context_rows": 1,
            "context_preview": "solo",
        },
    ]
    monkeypatch.setattr(main, "rag_couchdb", rag_db)

    events, connected = main._collect_recent_rag_stream_events(24)

    assert connected is True
    assert events[0]["context_rows"] == 0
    assert events[0]["context_bytes"] == len("size-check".encode("utf-8"))
    metrics, paired = main._compute_rag_influence_metrics(events)
    assert paired == 1
    by_key = {item.key: item for item in metrics}
    assert by_key["rag_toggle_comparison_pairs"].display == "1"
    assert by_key["rag_answer_change_rate_pct"].display == "100.0%"


def test_compute_correctness_drift_metrics_returns_live_values():
    now = datetime.now(timezone.utc)
    sessions = [
        {
            "trace_id": "trace-complete",
            "status": "completed",
            "created_at": (now - timedelta(minutes=40)).isoformat(),
            "updated_at": (now - timedelta(minutes=39)).isoformat(),
            "trace": {
                "legal_clerk": [
                    {
                        "persona": "Persona:Legal Clerk",
                        "phase": "map_deposition",
                        "sequence": 1,
                        "at": (now - timedelta(minutes=39, seconds=30)).isoformat(),
                        "llm_provider": "openai",
                        "llm_model": "gpt-5.2",
                        "output_preview": "mapped",
                    }
                ],
                "attorney": [],
            },
        },
        {
            "trace_id": "trace-failed",
            "status": "failed",
            "created_at": (now - timedelta(minutes=20)).isoformat(),
            "updated_at": (now - timedelta(minutes=19)).isoformat(),
            "trace": {
                "legal_clerk": [],
                "attorney": [
                    {
                        "persona": "Persona:Attorney",
                        "phase": "chat_error",
                        "sequence": 1,
                        "at": (now - timedelta(minutes=19, seconds=45)).isoformat(),
                        "llm_provider": "ollama",
                        "llm_model": "law-model",
                        "notes": "provider error",
                    }
                ],
            },
        },
    ]
    rag_events = [
        {
            "created_at": now - timedelta(minutes=18),
            "status": "completed",
            "question": "What is consideration?",
            "answer_preview": "Consideration requires bargained-for exchange.",
            "use_rag": True,
            "llm_provider": "openai",
            "llm_model": "gpt-5.2",
            "context_rows": 2,
            "context_bytes": 64,
        },
        {
            "created_at": now - timedelta(minutes=16),
            "status": "completed",
            "question": "What is consideration?",
            "answer_preview": "Consideration is something of value.",
            "use_rag": True,
            "llm_provider": "openai",
            "llm_model": "gpt-5.2",
            "context_rows": 0,
            "context_bytes": 0,
        },
        {
            "created_at": now - timedelta(minutes=14),
            "status": "completed",
            "question": "What is consideration?",
            "answer_preview": "Graph context disabled; answer confidence is low.",
            "use_rag": False,
            "llm_provider": "ollama",
            "llm_model": "law-model",
            "context_rows": 0,
            "context_bytes": 0,
        },
    ]

    metrics = main._compute_correctness_drift_metrics(sessions, rag_events)
    by_key = {item.key: item for item in metrics}

    assert by_key["golden_set_accuracy"].display == "50.0%"
    assert by_key["schema_adherence_rate"].display == "50.0%"
    assert by_key["unsupported_claim_rate"].display == "50.0%"
    assert by_key["repeat_prompt_inconsistency"].display == "100.0%"
    assert by_key["model_mix_drift_jsd"].display != "N/A"
    assert by_key["judge_human_disagreement"].display == "100.0%"
    assert "proxy" in by_key["golden_set_accuracy"].description.lower()
    assert "grouping repeated normalized questions" in by_key["repeat_prompt_inconsistency"].tracking.lower()


def test_correctness_helper_branches_and_edge_cases():
    assert main._normalize_model_key(None, None) == "unknown"
    assert main._normalize_model_key(None, "gpt-5.2") == "gpt-5.2"
    assert main._normalize_model_key("openai", None) == "openai"

    session_time = datetime.now(timezone.utc).isoformat()
    unknown_session = {
        "created_at": session_time,
        "updated_at": "",
        "trace": {
            "legal_clerk": ["skip", {"persona": "Persona:Legal Clerk", "phase": "map_deposition"}],
            "attorney": [],
        },
    }
    stamp, model_key = main._session_model_observation(unknown_session)
    assert stamp is not None
    assert model_key == "unknown"

    rate, repeated_groups = main._repeat_prompt_inconsistency_rate(
        [
            "skip",
            {"status": "running", "question": "Q", "answer_preview": "A"},
            {"status": "completed", "question": " ", "answer_preview": "A"},
            {"status": "completed", "question": "Q", "answer_preview": ""},
        ]
    )
    assert rate == 0.0
    assert repeated_groups == 0

    assert main._distribution_from_labels([]) == {}
    assert main._jensen_shannon_divergence({}, {}) == 0.0
    assert main._jensen_shannon_divergence({"a": 1.0}, {"a": 1.0}) == 0.0
    assert main._jensen_shannon_divergence({"a": 1.0}, {"b": 1.0}) > 0.0

    assert main._model_mix_drift_jsd(["skip"], ["skip"]) == 0.0
    assert (
        main._model_mix_drift_jsd(
            [
                {
                    "created_at": "",
                    "updated_at": "",
                    "trace": {"legal_clerk": [], "attorney": []},
                }
            ],
            [{"created_at": "bad-stamp", "llm_provider": "openai", "llm_model": "gpt-5.2"}],
        )
        == 0.0
    )


def test_collect_recent_rag_stream_events_handles_db_failure(monkeypatch):
    rag_db = Mock()
    rag_db.find.side_effect = RuntimeError("db unavailable")
    monkeypatch.setattr(main, "rag_couchdb", rag_db)

    events, connected = main._collect_recent_rag_stream_events(24)

    assert connected is False
    assert events == []


def test_get_agent_metrics_endpoint_validates_lookback(monkeypatch):
    monkeypatch.setattr(main, "_collect_runtime_trace_sessions", lambda: ([], True))
    monkeypatch.setattr(main, "_collect_recent_rag_stream_events", lambda _hours: ([], True))

    payload = main.get_agent_metrics(lookback_hours=24)
    assert payload.lookback_hours == 24
    assert payload.sampled_runs == 0
    assert payload.rag_sampled_queries == 0

    with pytest.raises(HTTPException) as exc:
        main.get_agent_metrics(lookback_hours=0)
    assert exc.value.status_code == 400
    assert "lookback_hours must be between 1 and 168" in exc.value.detail


def test_compute_agent_metric_history_builds_runtime_and_rag_series():
    now = datetime.now(timezone.utc)
    older_created = now - timedelta(hours=3, minutes=1)
    older_updated = now - timedelta(hours=3)
    recent_created = now - timedelta(hours=1, minutes=1)
    recent_updated = now - timedelta(hours=1)
    older_event = (older_created + timedelta(seconds=20)).isoformat()
    recent_event = (recent_created + timedelta(seconds=20)).isoformat()
    sessions = [
        {
            "trace_id": "trace-complete",
            "status": "completed",
            "created_at": older_created.isoformat(),
            "updated_at": older_updated.isoformat(),
            "trace": {
                "legal_clerk": [
                    {
                        "persona": "Persona:Legal Clerk",
                        "phase": "start",
                        "sequence": 1,
                        "at": older_event,
                    }
                ],
                "attorney": [],
            },
        },
        {
            "trace_id": "trace-failed",
            "status": "failed",
            "created_at": recent_created.isoformat(),
            "updated_at": recent_updated.isoformat(),
            "trace": {
                "legal_clerk": [],
                "attorney": [
                    {
                        "persona": "Persona:Attorney",
                        "phase": "step",
                        "sequence": 1,
                        "at": recent_event,
                    }
                ],
            },
        },
        {
            "trace_id": "trace-fallback-created",
            "status": "running",
            "created_at": (now - timedelta(minutes=25)).isoformat(),
            "updated_at": "",
            "trace": {"legal_clerk": [], "attorney": []},
        },
    ]
    rag_events = [
        {
            "created_at": older_updated,
            "status": "completed",
            "question": "What is breach?",
            "answer_preview": "Grounded answer",
            "use_rag": True,
            "context_rows": 2,
            "context_bytes": 24,
        },
        {
            "created_at": recent_updated,
            "status": "completed",
            "question": "What is breach?",
            "answer_preview": "Ungrounded answer",
            "use_rag": False,
            "context_rows": 0,
            "context_bytes": 0,
        },
    ]

    runtime_history = main._compute_agent_metric_history(
        sessions,
        lookback_hours=4,
        bucket_hours=2,
        metric_key="task_success_rate_pct",
        storage_connected=True,
        rag_events=rag_events,
        rag_storage_connected=True,
    )
    assert runtime_history["label"] == "Task Success Rate"
    assert len(runtime_history["points"]) == 2
    assert runtime_history["points"][0]["display"] == "100.0%"
    assert runtime_history["points"][1]["display"] == "0.0%"
    assert runtime_history["points"][1]["sample_size"] == 2

    rag_history = main._compute_agent_metric_history(
        sessions,
        lookback_hours=4,
        bucket_hours=2,
        metric_key="rag_completed_queries_split",
        storage_connected=True,
        rag_events=rag_events,
        rag_storage_connected=True,
    )
    assert rag_history["label"] == "Completed Graph Queries (ON/OFF)"
    assert rag_history["points"][0]["display"] == "1/0"
    assert rag_history["points"][1]["display"] == "0/1"
    assert rag_history["points"][0]["sample_size"] == 1

    correctness_history = main._compute_agent_metric_history(
        sessions,
        lookback_hours=4,
        bucket_hours=2,
        metric_key="golden_set_accuracy",
        storage_connected=True,
        rag_events=rag_events,
        rag_storage_connected=True,
    )
    assert correctness_history["label"] == "Golden Set Accuracy"
    assert correctness_history["points"][0]["display"] == "100.0%"
    assert correctness_history["points"][1]["display"] == "0.0%"


def test_get_agent_metric_history_endpoint_validates_and_handles_missing_metric(monkeypatch):
    monkeypatch.setattr(main, "_collect_runtime_trace_sessions", lambda: ([], True))
    monkeypatch.setattr(main, "_collect_recent_rag_stream_events", lambda _hours: ([], True))

    payload = main.get_agent_metric_history(
        metric_key=" task_success_rate_pct ",
        lookback_hours=24,
        bucket_hours=2,
    )
    assert payload["key"] == "task_success_rate_pct"
    assert len(payload["points"]) == 12

    correctness_payload = main.get_agent_metric_history(
        metric_key="golden_set_accuracy",
        lookback_hours=24,
        bucket_hours=2,
    )
    assert correctness_payload["key"] == "golden_set_accuracy"
    assert len(correctness_payload["points"]) == 12

    with pytest.raises(HTTPException) as blank_exc:
        main.get_agent_metric_history(metric_key=" ", lookback_hours=24, bucket_hours=2)
    assert blank_exc.value.status_code == 400
    assert "metric_key is required" in blank_exc.value.detail

    with pytest.raises(HTTPException) as bucket_exc:
        main.get_agent_metric_history(metric_key="task_success_rate_pct", lookback_hours=2, bucket_hours=3)
    assert bucket_exc.value.status_code == 400
    assert "bucket_hours cannot exceed lookback_hours" in bucket_exc.value.detail

    with pytest.raises(HTTPException) as lookback_exc:
        main.get_agent_metric_history(metric_key="task_success_rate_pct", lookback_hours=0, bucket_hours=1)
    assert lookback_exc.value.status_code == 400
    assert "lookback_hours must be between 1 and 168" in lookback_exc.value.detail

    with pytest.raises(HTTPException) as bucket_range_exc:
        main.get_agent_metric_history(metric_key="task_success_rate_pct", lookback_hours=24, bucket_hours=0)
    assert bucket_range_exc.value.status_code == 400
    assert "bucket_hours must be between 1 and 24" in bucket_range_exc.value.detail

    with pytest.raises(HTTPException) as missing_exc:
        main.get_agent_metric_history(metric_key="not_real_metric", lookback_hours=24, bucket_hours=2)
    assert missing_exc.value.status_code == 404
    assert "not available for trend history" in missing_exc.value.detail


def test_suggest_directory_option_empty_and_fallback(monkeypatch):
    monkeypatch.setattr(main, "app_root", Path("/workspace/app"))
    monkeypatch.setattr(main, "container_deposition_root", Path("/data/depositions"))

    assert main._suggest_directory_option([]) is None

    fallback = main.DepositionDirectoryOption(
        path="/tmp/custom",
        label="/tmp/custom (2 files)",
        file_count=2,
        source="configured",
    )
    assert main._suggest_directory_option([fallback]) == "/tmp/custom"


def test_case_doc_id_and_load_case_doc(monkeypatch):
    assert main._case_doc_id("CASE 001 / Alpha") == "case:case-001-alpha"

    couchdb = Mock()
    couchdb.get_doc.side_effect = RuntimeError("missing")
    couchdb.find.return_value = []
    monkeypatch.setattr(main, "couchdb", couchdb)
    assert main._load_case_doc("case-1") is None

    couchdb = Mock()
    couchdb.get_doc.side_effect = RuntimeError("missing")
    couchdb.find.side_effect = RuntimeError("find failed")
    monkeypatch.setattr(main, "couchdb", couchdb)
    assert main._load_case_doc("case-1") is None

    couchdb = Mock()
    couchdb.get_doc.side_effect = RuntimeError("missing")
    couchdb.find.return_value = [{"_id": "case:legacy", "case_id": "case-1"}]
    monkeypatch.setattr(main, "couchdb", couchdb)
    assert main._load_case_doc("case-1") == {"_id": "case:legacy", "case_id": "case-1"}

    couchdb = Mock()
    couchdb.get_doc.return_value = "not-a-dict"
    monkeypatch.setattr(main, "couchdb", couchdb)
    assert main._load_case_doc("case-1") is None

    couchdb = Mock()
    couchdb.get_doc.return_value = {"_id": "case:case-1", "case_id": "case-1"}
    monkeypatch.setattr(main, "couchdb", couchdb)
    assert main._load_case_doc("case-1") == {"_id": "case:case-1", "case_id": "case-1"}


def test_save_case_memory_and_upsert_case_doc(monkeypatch):
    couchdb = Mock()
    memory_couchdb = Mock()
    monkeypatch.setattr(main, "couchdb", couchdb)
    monkeypatch.setattr(main, "memory_couchdb", memory_couchdb)
    monkeypatch.setattr(main, "_utc_now_iso", lambda: "2026-01-01T00:00:00+00:00")

    main._save_case_memory("case-1", "langgraph", {"x": 1})

    saved = memory_couchdb.save_doc.call_args.args[0]
    assert saved["type"] == "case_memory"
    assert saved["case_id"] == "case-1"
    assert saved["channel"] == "langgraph"
    assert saved["payload"] == {"x": 1}

    couchdb.get_doc.return_value = {
        "_id": "case:case-1",
        "_rev": "3-a",
        "type": "case",
        "case_id": "case-1",
        "memory_entries": 2,
    }

    main._upsert_case_doc(
        "case-1",
        deposition_count=7,
        memory_increment=2,
        last_action="ingest",
        last_directory="/data/depositions/oj_simpson",
        llm_provider="ollama",
        llm_model="llama3.3",
        snapshot={
            "selected_deposition_id": "dep:1",
            "chat": {
                "history": [{"role": "user", "content": "What happened next?"}],
                "visible_messages": [
                    {"role": "user", "content": "What happened next?"},
                    {"role": "assistant", "content": "The witness placed the call immediately."},
                ],
                "message_count": 2,
                "draft_input": "Ask about the next contradiction",
            },
        },
    )

    updated = couchdb.update_doc.call_args.args[0]
    assert updated["deposition_count"] == 7
    assert updated["memory_entries"] == 4
    assert updated["last_action"] == "ingest"
    assert updated["last_directory"] == "/data/depositions/oj_simpson"
    assert updated["last_llm_provider"] == "ollama"
    assert updated["last_llm_model"] == "llama3.3"
    assert updated["snapshot"]["selected_deposition_id"] == "dep:1"
    assert updated["snapshot"]["chat"]["message_count"] == 2
    assert updated["snapshot"]["chat"]["draft_input"] == "Ask about the next contradiction"
    assert updated["snapshot"]["chat"]["visible_messages"][1]["content"] == "The witness placed the call immediately."


def test_save_case_memory_and_upsert_case_doc_wrap_errors(monkeypatch):
    memory_couchdb = Mock()
    memory_couchdb.save_doc.side_effect = RuntimeError("save failed")
    monkeypatch.setattr(main, "memory_couchdb", memory_couchdb)

    with pytest.raises(HTTPException) as memory_exc:
        main._save_case_memory("case-1", "chat", {"message": "x"})
    assert memory_exc.value.status_code == 502

    couchdb = Mock()
    couchdb.get_doc.side_effect = RuntimeError("missing")
    couchdb.update_doc.side_effect = RuntimeError("update failed")
    monkeypatch.setattr(main, "couchdb", couchdb)

    with pytest.raises(HTTPException) as case_exc:
        main._upsert_case_doc("case-1", deposition_count=1)
    assert case_exc.value.status_code == 502


def test_list_case_summaries_and_case_endpoints(monkeypatch):
    couchdb = Mock()
    couchdb.find.side_effect = [
        [
            {
                "_id": "case:case-1",
                "type": "case",
                "case_id": "CASE-001",
                "deposition_count": 2,
                "memory_entries": 3,
                "updated_at": "2026-01-02T00:00:00+00:00",
                "last_action": "chat",
                "last_directory": "/data/depositions/oj_simpson",
                "last_llm_provider": "openai",
                "last_llm_model": "gpt-5.2",
            },
            {"_id": "case:bad", "type": "case"},
        ],
        [
            {"_id": "dep:1", "type": "deposition", "case_id": "CASE-001"},
            {"_id": "dep:2", "type": "deposition", "case_id": "CASE-XYZ"},
            {"_id": "dep:3", "type": "deposition"},
        ],
    ]
    monkeypatch.setattr(main, "couchdb", couchdb)

    summaries = main._list_case_summaries()
    assert [item.case_id for item in summaries] == ["CASE-001", "CASE-XYZ"]
    assert summaries[0].deposition_count == 1
    assert summaries[0].memory_entries == 3
    assert summaries[1].deposition_count == 1

    couchdb.find.side_effect = [
        [
            {
                "_id": "case:case-1",
                "type": "case",
                "case_id": "CASE-001",
                "deposition_count": 2,
                "memory_entries": 3,
                "updated_at": "2026-01-02T00:00:00+00:00",
                "last_action": "chat",
                "last_directory": "/data/depositions/oj_simpson",
                "last_llm_provider": "openai",
                "last_llm_model": "gpt-5.2",
            }
        ],
        [
            {"_id": "dep:1", "type": "deposition", "case_id": "CASE-001"},
            {"_id": "dep:2", "type": "deposition", "case_id": "CASE-XYZ"},
            {"_id": "dep:3", "type": "deposition"},
        ],
    ]
    payload = main.list_cases()
    assert len(payload.cases) == 2

    monkeypatch.setattr(
        main,
        "_load_case_doc",
        lambda _case_id: {
            "_id": "case:case-1",
            "case_id": "CASE-001",
            "memory_entries": 3,
            "updated_at": "2026-01-02T00:00:00+00:00",
            "last_action": "save",
            "last_directory": "/data/depositions/oj_simpson",
            "last_llm_provider": "openai",
            "last_llm_model": "gpt-5.2",
            "snapshot": {"chat": {"message_count": 2}},
        },
    )
    monkeypatch.setattr(main, "_safe_case_depositions", lambda _case_id: [{"_id": "dep:1"}])
    detail = main._case_detail("CASE-001")
    assert detail is not None
    assert detail.snapshot["chat"]["message_count"] == 2
    assert detail.deposition_count == 1

    fetched = main.get_case(" CASE-001 ")
    assert fetched.case_id == "CASE-001"
    assert fetched.snapshot["chat"]["message_count"] == 2

    monkeypatch.setattr(main, "_load_case_doc", lambda _case_id: None)
    monkeypatch.setattr(main, "_safe_case_depositions", lambda _case_id: [])
    assert main._case_detail(" ") is None
    assert main._case_detail("CASE-404") is None
    with pytest.raises(HTTPException) as missing_case_exc:
        main.get_case("CASE-404")
    assert missing_case_exc.value.status_code == 404

    monkeypatch.setattr(main, "_safe_case_depositions", lambda _case_id: [{"_id": "dep:1"}, {"_id": "dep:2"}])
    fallback_detail = main._case_detail("CASE-FALLBACK")
    assert fallback_detail is not None
    assert fallback_detail.deposition_count == 2
    assert fallback_detail.snapshot == {}

    with pytest.raises(HTTPException) as blank_case_exc:
        main.get_case(" ")
    assert blank_case_exc.value.status_code == 400

    monkeypatch.setattr(main, "_case_detail", Mock(side_effect=RuntimeError("detail failed")))
    with pytest.raises(HTTPException) as detail_fail_exc:
        main.get_case("CASE-ERR")
    assert detail_fail_exc.value.status_code == 502

    couchdb = Mock()
    couchdb.find.return_value = [
        {"_id": "dep:1", "_rev": "1-a", "case_id": "CASE-001"},
        {"_id": "dep:1", "_rev": "1-a", "case_id": "CASE-001"},
        {"_id": "", "_rev": "1-a", "case_id": "CASE-001"},
    ]
    monkeypatch.setattr(main, "couchdb", couchdb)
    deleted = main._delete_case_docs("CASE-001")
    assert deleted == 1

    deleted_payload = main.delete_case("CASE-001")
    assert deleted_payload.deleted_docs == 1


def test_case_endpoints_wrap_errors(monkeypatch):
    couchdb = Mock()
    couchdb.find.side_effect = RuntimeError("find failed")
    monkeypatch.setattr(main, "couchdb", couchdb)

    with pytest.raises(HTTPException) as list_exc:
        main.list_cases()
    assert list_exc.value.status_code == 502

    with pytest.raises(HTTPException) as delete_exc:
        main.delete_case("CASE-001")
    assert delete_exc.value.status_code == 502


def test_delete_case_docs_includes_memory_database(monkeypatch):
    couchdb = Mock()
    memory_couchdb = Mock()
    couchdb.find.return_value = [{"_id": "dep:1", "_rev": "1-a", "case_id": "CASE-001"}]
    memory_couchdb.find.return_value = [{"_id": "mem:1", "_rev": "1-b", "case_id": "CASE-001"}]
    monkeypatch.setattr(main, "couchdb", couchdb)
    monkeypatch.setattr(main, "memory_couchdb", memory_couchdb)

    deleted = main._delete_case_docs("CASE-001")

    assert deleted == 2
    couchdb.delete_doc.assert_called_once_with("dep:1", rev="1-a")
    memory_couchdb.delete_doc.assert_called_once_with("mem:1", rev="1-b")


def test_delete_case_docs_skips_duplicate_or_blank_memory_ids(monkeypatch):
    couchdb = Mock()
    memory_couchdb = Mock()
    couchdb.find.return_value = [{"_id": "dep:1", "_rev": "1-a", "case_id": "CASE-001"}]
    memory_couchdb.find.return_value = [
        {"_id": "dep:1", "_rev": "2-a", "case_id": "CASE-001"},
        {"_id": "", "_rev": "2-b", "case_id": "CASE-001"},
    ]
    monkeypatch.setattr(main, "couchdb", couchdb)
    monkeypatch.setattr(main, "memory_couchdb", memory_couchdb)

    deleted = main._delete_case_docs("CASE-001")

    assert deleted == 1
    couchdb.delete_doc.assert_called_once_with("dep:1", rev="1-a")
    memory_couchdb.delete_doc.assert_not_called()


def test_delete_case_deposition_docs_and_clear_case_depositions_endpoint(monkeypatch):
    couchdb = Mock()
    couchdb.find.return_value = [
        {"_id": "dep:1", "_rev": "1-a", "type": "deposition", "case_id": "CASE-001"},
        {"_id": "dep:1", "_rev": "1-a", "type": "deposition", "case_id": "CASE-001"},
        {"_id": "", "_rev": "1-b", "type": "deposition", "case_id": "CASE-001"},
    ]
    monkeypatch.setattr(main, "couchdb", couchdb)

    deleted = main._delete_case_deposition_docs("CASE-001")
    assert deleted == 1
    couchdb.find.assert_called_with({"type": "deposition", "case_id": "CASE-001"}, limit=10000)
    couchdb.delete_doc.assert_called_once_with("dep:1", rev="1-a")

    upsert = Mock()
    monkeypatch.setattr(main, "_delete_case_deposition_docs", lambda _case_id: 3)
    monkeypatch.setattr(main, "_upsert_case_doc", upsert)
    payload = main.clear_case_depositions(" CASE-001 ")
    assert payload.case_id == "CASE-001"
    assert payload.deleted_depositions == 3
    upsert.assert_called_once_with(
        "CASE-001",
        deposition_count=0,
        last_action="refresh",
    )


def test_clear_case_depositions_endpoint_errors(monkeypatch):
    with pytest.raises(HTTPException) as blank_exc:
        main.clear_case_depositions(" ")
    assert blank_exc.value.status_code == 400

    monkeypatch.setattr(main, "_delete_case_deposition_docs", Mock(side_effect=RuntimeError("db failed")))
    monkeypatch.setattr(main, "_upsert_case_doc", Mock())
    with pytest.raises(HTTPException) as refresh_exc:
        main.clear_case_depositions("CASE-001")
    assert refresh_exc.value.status_code == 502
    assert "Failed to refresh case 'CASE-001'" in refresh_exc.value.detail

    monkeypatch.setattr(
        main,
        "_delete_case_deposition_docs",
        Mock(side_effect=HTTPException(status_code=503, detail="upstream")),
    )
    monkeypatch.setattr(main, "_upsert_case_doc", Mock())
    with pytest.raises(HTTPException) as passthrough_exc:
        main.clear_case_depositions("CASE-001")
    assert passthrough_exc.value.status_code == 503


def test_save_case_endpoint_and_rename_case_docs(monkeypatch):
    upsert = Mock()
    monkeypatch.setattr(main, "_upsert_case_doc", upsert)
    monkeypatch.setattr(main, "_safe_case_depositions", lambda _case_id: [{"_id": "dep:1"}, {"_id": "dep:2"}])
    monkeypatch.setattr(main, "_list_case_summaries", lambda: [main.CaseSummary(case_id="CASE-NEW", deposition_count=2)])

    payload = main.save_case(
        SaveCaseRequest(
            case_id=" CASE-NEW ",
            directory="/data/depositions/default",
            llm_provider="openai",
            llm_model="gpt-5.2",
            snapshot={
                "dropdowns": {"llm_selected": "openai::gpt-5.2"},
                "chat": {
                    "history": [{"role": "user", "content": "Summarize the timeline."}],
                    "visible_messages": [
                        {"role": "user", "content": "Summarize the timeline."},
                        {"role": "assistant", "content": "The witness says the meeting happened after lunch."},
                    ],
                    "message_count": 2,
                    "draft_input": "Ask what contradicts this",
                },
            },
        )
    )
    assert payload.case_id == "CASE-NEW"
    assert payload.deposition_count == 2
    upsert.assert_called_once_with(
        "CASE-NEW",
        deposition_count=2,
        last_action="save",
        last_directory="/data/depositions/default",
        llm_provider="openai",
        llm_model="gpt-5.2",
        snapshot={
            "dropdowns": {"llm_selected": "openai::gpt-5.2"},
            "chat": {
                "history": [{"role": "user", "content": "Summarize the timeline."}],
                "visible_messages": [
                    {"role": "user", "content": "Summarize the timeline."},
                    {"role": "assistant", "content": "The witness says the meeting happened after lunch."},
                ],
                "message_count": 2,
                "draft_input": "Ask what contradicts this",
            },
        },
    )

    with pytest.raises(HTTPException) as save_exc:
        main.save_case(SaveCaseRequest(case_id="   "))
    assert save_exc.value.status_code == 400

    monkeypatch.setattr(main, "_list_case_summaries", lambda: [])
    fallback_payload = main.save_case(
        SaveCaseRequest(case_id="CASE-FALLBACK", snapshot={"ui": {"active_tab": "intelligence"}})
    )
    assert fallback_payload.case_id == "CASE-FALLBACK"
    assert fallback_payload.deposition_count == 2

    monkeypatch.setattr(main, "_upsert_case_doc", Mock(side_effect=RuntimeError("upsert failed")))
    with pytest.raises(HTTPException) as save_fail_exc:
        main.save_case(SaveCaseRequest(case_id="CASE-ERR"))
    assert save_fail_exc.value.status_code == 502

    monkeypatch.setattr(main, "_upsert_case_doc", Mock(side_effect=HTTPException(status_code=418, detail="teapot")))
    with pytest.raises(HTTPException) as save_passthrough_exc:
        main.save_case(SaveCaseRequest(case_id="CASE-TEAPOT"))
    assert save_passthrough_exc.value.status_code == 418

    couchdb = Mock()
    couchdb.find.return_value = [
        {"_id": "dep:1", "_rev": "1-a", "case_id": "CASE-OLD"},
        {"_id": "case:case-old", "_rev": "1-b", "type": "case", "case_id": "CASE-OLD"},
        {"_id": "", "_rev": "1-c", "case_id": "CASE-OLD"},
    ]
    monkeypatch.setattr(main, "couchdb", couchdb)
    monkeypatch.setattr(main, "_utc_now_iso", lambda: "2026-01-03T00:00:00+00:00")

    moved = main._rename_case_docs("CASE-OLD", "CASE-NEW")
    assert moved == 2
    assert couchdb.update_doc.call_count == 2
    renamed_case = couchdb.update_doc.call_args_list[1].args[0]
    assert renamed_case["case_id"] == "CASE-NEW"
    assert renamed_case["last_action"] == "rename"
    assert renamed_case["updated_at"] == "2026-01-03T00:00:00+00:00"


def test_rename_case_docs_includes_memory_database(monkeypatch):
    couchdb = Mock()
    memory_couchdb = Mock()
    couchdb.find.return_value = [{"_id": "dep:1", "_rev": "1-a", "case_id": "CASE-OLD", "type": "deposition"}]
    memory_couchdb.find.return_value = [{"_id": "mem:1", "_rev": "1-b", "case_id": "CASE-OLD", "type": "case_memory"}]
    monkeypatch.setattr(main, "couchdb", couchdb)
    monkeypatch.setattr(main, "memory_couchdb", memory_couchdb)

    moved = main._rename_case_docs("CASE-OLD", "CASE-NEW")

    assert moved == 2
    assert couchdb.update_doc.call_args.args[0]["case_id"] == "CASE-NEW"
    assert memory_couchdb.update_doc.call_args.args[0]["case_id"] == "CASE-NEW"


def test_rename_case_docs_skips_duplicate_or_blank_memory_ids(monkeypatch):
    couchdb = Mock()
    memory_couchdb = Mock()
    couchdb.find.return_value = [{"_id": "dep:1", "_rev": "1-a", "case_id": "CASE-OLD", "type": "deposition"}]
    memory_couchdb.find.return_value = [
        {"_id": "dep:1", "_rev": "2-a", "case_id": "CASE-OLD", "type": "case_memory"},
        {"_id": "", "_rev": "2-b", "case_id": "CASE-OLD", "type": "case_memory"},
    ]
    monkeypatch.setattr(main, "couchdb", couchdb)
    monkeypatch.setattr(main, "memory_couchdb", memory_couchdb)

    moved = main._rename_case_docs("CASE-OLD", "CASE-NEW")

    assert moved == 1
    assert couchdb.update_doc.call_count == 1
    memory_couchdb.update_doc.assert_not_called()


def test_rename_case_endpoint_branches(monkeypatch):
    monkeypatch.setattr(main, "_safe_case_depositions", lambda _case_id: [{"_id": "dep:1"}])
    upsert = Mock()
    monkeypatch.setattr(main, "_upsert_case_doc", upsert)

    couchdb = Mock()
    couchdb.find.side_effect = [
        [],
        [
            {"_id": "dep:1", "_rev": "1-a", "case_id": "CASE-OLD"},
            {"_id": "case:case-old", "_rev": "1-b", "type": "case", "case_id": "CASE-OLD"},
        ],
    ]
    monkeypatch.setattr(main, "couchdb", couchdb)

    payload = main.rename_case("CASE-OLD", RenameCaseRequest(new_case_id="CASE-NEW"))
    assert payload.old_case_id == "CASE-OLD"
    assert payload.new_case_id == "CASE-NEW"
    assert payload.moved_docs == 2
    assert couchdb.update_doc.call_count == 2
    upsert.assert_called_once_with("CASE-NEW", deposition_count=1, last_action="rename")

    same_payload = main.rename_case("CASE-NEW", RenameCaseRequest(new_case_id="CASE-NEW"))
    assert same_payload.moved_docs == 0

    with pytest.raises(HTTPException) as blank_exc:
        main.rename_case("CASE-NEW", RenameCaseRequest(new_case_id=" "))
    assert blank_exc.value.status_code == 400

    couchdb = Mock()
    couchdb.find.return_value = [{"_id": "dep:exists", "case_id": "CASE-TAKEN"}]
    monkeypatch.setattr(main, "couchdb", couchdb)
    with pytest.raises(HTTPException) as conflict_exc:
        main.rename_case("CASE-OLD", RenameCaseRequest(new_case_id="CASE-TAKEN"))
    assert conflict_exc.value.status_code == 409

    couchdb = Mock()
    couchdb.find.side_effect = [[], []]
    monkeypatch.setattr(main, "couchdb", couchdb)
    with pytest.raises(HTTPException) as missing_exc:
        main.rename_case("CASE-OLD", RenameCaseRequest(new_case_id="CASE-MISSING"))
    assert missing_exc.value.status_code == 404

    couchdb = Mock()
    couchdb.find.side_effect = RuntimeError("find failed")
    monkeypatch.setattr(main, "couchdb", couchdb)
    with pytest.raises(HTTPException) as fail_exc:
        main.rename_case("CASE-OLD", RenameCaseRequest(new_case_id="CASE-ERR"))
    assert fail_exc.value.status_code == 502


def test_list_case_versions_and_save_case_version(monkeypatch):
    version_docs = [
        {
            "_id": "case_version:1",
            "type": "case_version",
            "case_id": "CASE-001",
            "version": 2,
            "created_at": "2026-02-25T00:00:02+00:00",
            "directory": "/data/depositions/default",
            "llm_provider": "openai",
            "llm_model": "gpt-5.2",
            "snapshot": {"status": "ok2"},
        },
        {
            "_id": "case_version:0",
            "type": "case_version",
            "case_id": "CASE-001",
            "version": 1,
            "created_at": "2026-02-25T00:00:01+00:00",
            "directory": "/data/depositions/default",
            "llm_provider": "openai",
            "llm_model": "gpt-5.2",
            "snapshot": {"status": "ok1"},
        },
    ]
    couchdb = Mock()
    couchdb.find.side_effect = [
        version_docs,
        version_docs,
    ]
    monkeypatch.setattr(main, "couchdb", couchdb)
    monkeypatch.setattr(main, "_utc_now_iso", lambda: "2026-02-25T00:00:03+00:00")
    monkeypatch.setattr(main, "_safe_case_depositions", lambda _case_id: [{"_id": "dep:1"}])
    upsert = Mock()
    monkeypatch.setattr(main, "_upsert_case_doc", upsert)

    listed = main.list_case_versions("CASE-001")
    assert listed.case_id == "CASE-001"
    assert [item.version for item in listed.versions] == [2, 1]

    payload = main.save_case_version(
        SaveCaseVersionRequest(
            case_id="CASE-001",
            directory="/data/depositions/default",
            llm_provider="openai",
            llm_model="gpt-5.2",
            snapshot={"status": "saved"},
        )
    )
    assert payload.version == 3
    assert payload.snapshot["status"] == "saved"
    saved_doc = couchdb.save_doc.call_args.args[0]
    assert saved_doc["type"] == "case_version"
    assert saved_doc["version"] == 3
    upsert.assert_called_once_with(
        "CASE-001",
        deposition_count=1,
        last_action="save_version",
        last_directory="/data/depositions/default",
        llm_provider="openai",
        llm_model="gpt-5.2",
    )


def test_case_has_docs_and_clone_case_contents(monkeypatch):
    couchdb = Mock()
    memory_couchdb = Mock()
    couchdb.find.side_effect = [
        [{"_id": "dep:1", "case_id": "CASE-001"}],
        RuntimeError("find failed"),
        [
            {"_id": "dep:1", "_rev": "1-a", "case_id": "CASE-001", "type": "deposition", "file_name": "a.txt"},
            {"_id": "case:1", "_rev": "1-c", "case_id": "CASE-001", "type": "case"},
        ],
    ]
    memory_couchdb.find.side_effect = [
        RuntimeError("find failed"),
        [{"_id": "mem:1", "_rev": "1-b", "case_id": "CASE-001", "type": "case_memory", "payload": {"x": 1}}],
    ]
    monkeypatch.setattr(main, "couchdb", couchdb)
    monkeypatch.setattr(main, "memory_couchdb", memory_couchdb)

    assert main._case_has_docs("CASE-001") is True
    assert main._case_has_docs("CASE-ERR") is False

    cloned = main._clone_case_contents("CASE-001", "CASE-NEW")
    assert cloned == 2
    assert couchdb.save_doc.call_count == 1
    assert memory_couchdb.save_doc.call_count == 1
    first_saved = couchdb.save_doc.call_args_list[0].args[0]
    assert first_saved["case_id"] == "CASE-NEW"
    assert "_id" not in first_saved
    assert "_rev" not in first_saved
    memory_saved = memory_couchdb.save_doc.call_args_list[0].args[0]
    assert memory_saved["case_id"] == "CASE-NEW"
    assert memory_saved["type"] == "case_memory"


def test_case_has_docs_uses_memory_database_fallback(monkeypatch):
    couchdb = Mock()
    memory_couchdb = Mock()
    couchdb.find.side_effect = RuntimeError("primary down")
    memory_couchdb.find.return_value = [{"_id": "mem:1", "case_id": "CASE-001"}]
    monkeypatch.setattr(main, "couchdb", couchdb)
    monkeypatch.setattr(main, "memory_couchdb", memory_couchdb)

    assert main._case_has_docs("CASE-001") is True


def test_save_case_version_clones_when_case_id_changes(monkeypatch):
    monkeypatch.setattr(main, "_utc_now_iso", lambda: "2026-02-25T00:00:04+00:00")
    monkeypatch.setattr(main, "_safe_case_depositions", lambda _case_id: [{"_id": "dep:new"}])
    upsert = Mock()
    monkeypatch.setattr(main, "_upsert_case_doc", upsert)
    monkeypatch.setattr(main, "_case_has_docs", lambda _case_id: False)
    clone = Mock(return_value=2)
    monkeypatch.setattr(main, "_clone_case_contents", clone)
    monkeypatch.setattr(main, "_list_case_versions", lambda _case_id: [])
    couchdb = Mock()
    monkeypatch.setattr(main, "couchdb", couchdb)

    payload = main.save_case_version(
        SaveCaseVersionRequest(
            case_id="CASE-NEW",
            source_case_id="CASE-BASE",
            directory="/data/depositions/default",
            llm_provider="openai",
            llm_model="gpt-5.2",
            snapshot={"status": "saved"},
        )
    )
    assert payload.version == 1
    clone.assert_called_once_with("CASE-BASE", "CASE-NEW")
    upsert.assert_called_once_with(
        "CASE-NEW",
        deposition_count=1,
        last_action="save_version",
        last_directory="/data/depositions/default",
        llm_provider="openai",
        llm_model="gpt-5.2",
    )


def test_list_case_versions_skips_invalid_docs(monkeypatch):
    couchdb = Mock()
    couchdb.find.return_value = [
        {
            "_id": "case_version:bad",
            "type": "case_version",
            "case_id": "CASE-001",
            "version": 3,
            "created_at": "2026-02-25T00:00:03+00:00",
            "directory": "/data/depositions/default",
            "llm_provider": "invalid-provider",
            "llm_model": "gpt-5.2",
        },
        {
            "_id": "case_version:ok",
            "type": "case_version",
            "case_id": "CASE-001",
            "version": 2,
            "created_at": "2026-02-25T00:00:02+00:00",
            "directory": "/data/depositions/default",
            "llm_provider": "openai",
            "llm_model": "gpt-5.2",
            "snapshot": {"status": "ok"},
        },
    ]
    monkeypatch.setattr(main, "couchdb", couchdb)

    versions = main._list_case_versions("CASE-001")
    assert len(versions) == 1
    assert versions[0].version == 2


def test_list_case_versions_and_save_case_version_errors(monkeypatch):
    couchdb = Mock()
    couchdb.find.side_effect = RuntimeError("find failed")
    monkeypatch.setattr(main, "couchdb", couchdb)

    with pytest.raises(HTTPException) as list_exc:
        main.list_case_versions("CASE-001")
    assert list_exc.value.status_code == 502

    with pytest.raises(HTTPException) as blank_case_exc:
        main.save_case_version(
            SaveCaseVersionRequest(
                case_id=" ",
                directory="/data/depositions/default",
                llm_provider="openai",
                llm_model="gpt-5.2",
            )
        )
    assert blank_case_exc.value.status_code == 400

    with pytest.raises(HTTPException) as blank_dir_exc:
        main.save_case_version(
            SaveCaseVersionRequest(
                case_id="CASE-001",
                directory=" ",
                llm_provider="openai",
                llm_model="gpt-5.2",
            )
        )
    assert blank_dir_exc.value.status_code == 400

    monkeypatch.setattr(main, "_case_has_docs", lambda _case_id: True)
    with pytest.raises(HTTPException) as conflict_exc:
        main.save_case_version(
            SaveCaseVersionRequest(
                case_id="CASE-NEW",
                source_case_id="CASE-BASE",
                directory="/data/depositions/default",
                llm_provider="openai",
                llm_model="gpt-5.2",
            )
        )
    assert conflict_exc.value.status_code == 409

    monkeypatch.setattr(main, "_case_has_docs", lambda _case_id: False)
    monkeypatch.setattr(main, "_clone_case_contents", lambda _src, _dst: 0)
    with pytest.raises(HTTPException) as missing_source_exc:
        main.save_case_version(
            SaveCaseVersionRequest(
                case_id="CASE-NEW",
                source_case_id="CASE-BASE",
                directory="/data/depositions/default",
                llm_provider="openai",
                llm_model="gpt-5.2",
            )
        )
    assert missing_source_exc.value.status_code == 404

    couchdb = Mock()
    couchdb.find.return_value = []
    couchdb.save_doc.side_effect = RuntimeError("save failed")
    monkeypatch.setattr(main, "couchdb", couchdb)
    monkeypatch.setattr(main, "_safe_case_depositions", lambda _case_id: [])
    monkeypatch.setattr(main, "_upsert_case_doc", Mock())

    with pytest.raises(HTTPException) as save_exc:
        main.save_case_version(
            SaveCaseVersionRequest(
                case_id="CASE-001",
                directory="/data/depositions/default",
                llm_provider="openai",
                llm_model="gpt-5.2",
            )
        )
    assert save_exc.value.status_code == 502


def test_get_llm_options_returns_payload(monkeypatch):
    monkeypatch.setattr(main, "resolve_llm_selection", lambda *_args: ("openai", "gpt-5.2"))
    monkeypatch.setattr(
        main,
        "list_llm_options",
        lambda _settings: [{"provider": "openai", "model": "gpt-5.2", "label": "ChatGPT - gpt-5.2"}],
    )
    monkeypatch.setattr(
        main,
        "get_llm_option_status",
        lambda _settings, _provider, _model, *, force_probe=False: {
            "operational": True,
            "error": None,
            "possible_fix": None,
        },
    )

    payload = main.get_llm_options()

    assert payload.selected_provider == "openai"
    assert payload.selected_model == "gpt-5.2"
    assert payload.options[0].label == "ChatGPT - gpt-5.2"


def test_get_llm_options_force_probe_forwards_flag(monkeypatch):
    monkeypatch.setattr(main, "resolve_llm_selection", lambda *_args: ("openai", "gpt-5.2"))
    monkeypatch.setattr(
        main,
        "list_llm_options",
        lambda _settings: [{"provider": "openai", "model": "gpt-5.2", "label": "ChatGPT - gpt-5.2"}],
    )
    status = Mock(
        return_value={
            "operational": True,
            "error": None,
            "possible_fix": None,
        }
    )
    monkeypatch.setattr(main, "get_llm_option_status", status)

    payload = main.get_llm_options(force_probe=True)

    assert payload.options[0].label == "ChatGPT - gpt-5.2"
    status.assert_called_once_with(main.settings, "openai", "gpt-5.2", force_probe=True)


def test_get_llm_options_force_probe_parallel_branch(monkeypatch):
    monkeypatch.setattr(main, "settings", SimpleNamespace(llm_options_probe_workers=2))
    monkeypatch.setattr(main, "resolve_llm_selection", lambda *_args: ("openai", "gpt-5.2"))
    monkeypatch.setattr(
        main,
        "list_llm_options",
        lambda _settings: [
            {"provider": "openai", "model": "gpt-5.2", "label": "ChatGPT - gpt-5.2"},
            {"provider": "ollama", "model": "llama3.3", "label": "Ollama - llama3.3"},
        ],
    )
    monkeypatch.setattr(
        main,
        "get_llm_option_status",
        lambda _settings, _provider, _model, *, force_probe=False: {
            "operational": True,
            "error": None,
            "possible_fix": None,
        },
    )

    payload = main.get_llm_options(force_probe=True)

    assert len(payload.options) == 2
    assert payload.options[1].label == "Ollama - llama3.3"


def test_resolve_request_llm_wraps_validation_errors(monkeypatch):
    monkeypatch.setattr(main, "resolve_llm_selection", Mock(side_effect=ValueError("Unsupported")))

    with pytest.raises(HTTPException) as exc:
        main._resolve_request_llm("bad", "x")

    assert exc.value.status_code == 400
    assert "Unsupported" in exc.value.detail


def test_ensure_request_llm_operational_wraps_errors(monkeypatch):
    monkeypatch.setattr(main, "ensure_llm_operational", Mock(side_effect=main.LLMOperationalError("bad", "fix it")))

    with pytest.raises(HTTPException) as exc:
        main._ensure_request_llm_operational("openai", "gpt-5.2")

    assert exc.value.status_code == 503
    assert "Possible fix: fix it" in exc.value.detail


def test_ensure_startup_llm_connectivity_success(monkeypatch):
    monkeypatch.setattr(main, "_resolve_request_llm", lambda *_args: ("openai", "gpt-5.2"))
    ready = Mock()
    monkeypatch.setattr(main, "ensure_llm_operational", ready)

    main._ensure_startup_llm_connectivity()

    ready.assert_called_once_with(main.settings, "openai", "gpt-5.2")


def test_ensure_startup_llm_connectivity_failure(monkeypatch):
    monkeypatch.setattr(main, "_resolve_request_llm", lambda *_args: ("openai", "gpt-5.2"))
    monkeypatch.setattr(
        main,
        "ensure_llm_operational",
        Mock(side_effect=main.LLMOperationalError("bad", "set OPENAI_API_KEY")),
    )

    with pytest.raises(RuntimeError) as exc:
        main._ensure_startup_llm_connectivity()

    assert "Startup failed" in str(exc.value)
    assert "Possible fix: set OPENAI_API_KEY" in str(exc.value)


def test_ingest_case_success(monkeypatch):
    file_a = Path("/tmp/a.txt")
    file_b = Path("/tmp/b.txt")
    monkeypatch.setattr(main, "_resolve_ingest_txt_files", lambda _path: [file_a, file_b])

    workflow = Mock()
    workflow.run.side_effect = [
        {"deposition_doc": {"_id": "dep:1"}},
        {"deposition_doc": {"_id": "dep:2"}},
    ]
    workflow.reassess_case.return_value = [
        {
            "_id": "dep:1",
            "file_name": "a.txt",
            "witness_name": "Jane",
            "contradiction_score": 10,
            "flagged": False,
        },
        {
            "_id": "dep:2",
            "file_name": "b.txt",
            "witness_name": "Alan",
            "contradiction_score": 20,
            "flagged": True,
        },
    ]
    couchdb = Mock()
    couchdb.list_depositions.return_value = []

    monkeypatch.setattr(main, "workflow", workflow)
    monkeypatch.setattr(main, "couchdb", couchdb)

    response = main.ingest_case(
        IngestCaseRequest(
            case_id="case-1",
            directory="/tmp",
            llm_provider="ollama",
            llm_model="llama3.3",
        )
    )

    assert response.case_id == "case-1"
    assert len(response.ingested) == 2
    assert response.thought_stream is not None
    assert workflow.run.call_count == 2
    workflow.run.assert_any_call(
        case_id="case-1",
        file_path="/tmp/a.txt",
        llm_provider="ollama",
        llm_model="llama3.3",
    )
    workflow.reassess_case.assert_called_once_with(
        "case-1",
        llm_provider="ollama",
        llm_model="llama3.3",
    )


def test_ingest_case_wraps_processing_errors(monkeypatch):
    file_a = Path("/tmp/a.txt")
    monkeypatch.setattr(main, "_resolve_ingest_txt_files", lambda _path: [file_a])
    workflow = Mock()
    workflow.run.side_effect = RuntimeError("parse failed")
    monkeypatch.setattr(main, "workflow", workflow)

    with pytest.raises(HTTPException) as exc:
        main.ingest_case(IngestCaseRequest(case_id="case-1", directory="/tmp"))

    assert exc.value.status_code == 502
    assert "a.txt" in exc.value.detail


def test_ingest_case_skip_reassess_uses_per_file_docs(monkeypatch):
    file_a = Path("/tmp/a.txt")
    file_b = Path("/tmp/b.txt")
    monkeypatch.setattr(main, "_resolve_ingest_txt_files", lambda _path: [file_a, file_b])

    workflow = Mock()
    workflow.run.side_effect = [
        {"deposition_doc": {"_id": "dep:1"}},
        {"deposition_doc": {"_id": "dep:2"}},
    ]
    couchdb = Mock()
    couchdb.list_depositions.return_value = []
    couchdb.get_doc.side_effect = [
        {
            "_id": "dep:1",
            "file_name": "a.txt",
            "witness_name": "Jane",
            "contradiction_score": 5,
            "flagged": False,
        },
        {
            "_id": "dep:2",
            "file_name": "b.txt",
            "witness_name": "Alan",
            "contradiction_score": 9,
            "flagged": True,
        },
    ]

    monkeypatch.setattr(main, "workflow", workflow)
    monkeypatch.setattr(main, "couchdb", couchdb)

    response = main.ingest_case(
        IngestCaseRequest(
            case_id="case-1",
            directory="/tmp",
            llm_provider="ollama",
            llm_model="llama3.3",
            skip_reassess=True,
        )
    )

    assert response.case_id == "case-1"
    assert [item.deposition_id for item in response.ingested] == ["dep:1", "dep:2"]
    workflow.reassess_case.assert_not_called()


def test_ingest_case_purges_stale_case_docs(monkeypatch):
    file_a = Path("/tmp/oj_witness.txt")
    monkeypatch.setattr(main, "_resolve_ingest_txt_files", lambda _path: [file_a])

    workflow = Mock()
    canonical_doc_id = "dep:case-1:oj-witness"
    workflow.run.return_value = {"deposition_doc": {"_id": canonical_doc_id}}
    workflow.reassess_case.return_value = [
        {
            "_id": canonical_doc_id,
            "file_name": "oj_witness.txt",
            "witness_name": "Witness A",
            "contradiction_score": 12,
            "flagged": False,
        }
    ]
    couchdb = Mock()
    couchdb.list_depositions.return_value = [
        {"_id": "dep:old", "_rev": "3-old", "file_name": "witness_jane_doe.txt"},
        {"_id": canonical_doc_id, "_rev": "5-new", "file_name": "oj_witness.txt"},
    ]

    monkeypatch.setattr(main, "workflow", workflow)
    monkeypatch.setattr(main, "couchdb", couchdb)

    response = main.ingest_case(IngestCaseRequest(case_id="case-1", directory="/tmp"))

    assert response.case_id == "case-1"
    couchdb.delete_doc.assert_called_once_with("dep:old", rev="3-old")


def test_ingest_case_purge_wraps_delete_errors(monkeypatch):
    file_a = Path("/tmp/oj_witness.txt")
    monkeypatch.setattr(main, "_resolve_ingest_txt_files", lambda _path: [file_a])
    couchdb = Mock()
    couchdb.list_depositions.return_value = [
        {"_id": "dep:old", "_rev": "3-old", "file_name": "witness_jane_doe.txt"}
    ]
    couchdb.delete_doc.side_effect = RuntimeError("delete failed")
    monkeypatch.setattr(main, "couchdb", couchdb)

    with pytest.raises(HTTPException) as exc:
        main.ingest_case(IngestCaseRequest(case_id="case-1", directory="/tmp"))

    assert exc.value.status_code == 502
    assert "Failed to remove stale deposition" in exc.value.detail


def test_safe_case_depositions_handles_errors_and_non_iterable(monkeypatch):
    couchdb = Mock()
    couchdb.list_depositions.side_effect = RuntimeError("db down")
    monkeypatch.setattr(main, "couchdb", couchdb)
    assert main._safe_case_depositions("case-1") == []

    couchdb = Mock()
    couchdb.list_depositions.return_value = Mock()  # non-iterable
    monkeypatch.setattr(main, "couchdb", couchdb)
    assert main._safe_case_depositions("case-1") == []

    couchdb = Mock()
    couchdb.list_depositions.return_value = ({"_id": "dep:1"},)
    monkeypatch.setattr(main, "couchdb", couchdb)
    assert main._safe_case_depositions("case-1") == [{"_id": "dep:1"}]


def test_purge_stale_case_depositions_skips_docs_without_id(monkeypatch):
    couchdb = Mock()
    couchdb.list_depositions.return_value = [{"file_name": "old.txt"}]
    monkeypatch.setattr(main, "couchdb", couchdb)

    main._purge_stale_case_depositions("case-1", [Path("/tmp/new.txt")])

    couchdb.delete_doc.assert_not_called()


def test_canonical_deposition_doc_id_uses_case_and_file_stem():
    assert (
        main._canonical_deposition_doc_id("CASE 001 / Alpha", "witness_jane_doe.txt")
        == "dep:case-001-alpha:witness-jane-doe"
    )


def test_purge_noncanonical_case_depositions_removes_legacy_duplicates(monkeypatch):
    couchdb = Mock()
    couchdb.list_depositions.return_value = [
        {"_id": "dep:legacy-1", "_rev": "1-a", "file_name": "oj_witness.txt"},
        {"_id": "dep:case-1:oj-witness", "_rev": "2-b", "file_name": "oj_witness.txt"},
        {"_id": "dep:legacy-other", "_rev": "1-c", "file_name": "other.txt"},
    ]
    monkeypatch.setattr(main, "couchdb", couchdb)

    main._purge_noncanonical_case_depositions("case-1", [Path("/tmp/oj_witness.txt")])

    couchdb.delete_doc.assert_called_once_with("dep:legacy-1", rev="1-a")


def test_purge_noncanonical_case_depositions_wraps_delete_errors(monkeypatch):
    couchdb = Mock()
    couchdb.list_depositions.return_value = [
        {"_id": "dep:legacy-1", "_rev": "1-a", "file_name": "oj_witness.txt"}
    ]
    couchdb.delete_doc.side_effect = RuntimeError("delete failed")
    monkeypatch.setattr(main, "couchdb", couchdb)

    with pytest.raises(HTTPException) as exc:
        main._purge_noncanonical_case_depositions("case-1", [Path("/tmp/oj_witness.txt")])

    assert exc.value.status_code == 502
    assert "Failed to remove duplicate deposition" in exc.value.detail


def test_purge_noncanonical_case_depositions_skips_docs_without_id(monkeypatch):
    couchdb = Mock()
    couchdb.list_depositions.return_value = [{"_id": "", "file_name": "oj_witness.txt"}]
    monkeypatch.setattr(main, "couchdb", couchdb)

    main._purge_noncanonical_case_depositions("case-1", [Path("/tmp/oj_witness.txt")])

    couchdb.delete_doc.assert_not_called()


def test_list_depositions_sorts_descending(monkeypatch):
    couchdb = Mock()
    couchdb.list_depositions.return_value = [
        {"_id": "dep:1", "contradiction_score": 10},
        {"_id": "dep:2", "contradiction_score": 80},
    ]
    monkeypatch.setattr(main, "couchdb", couchdb)

    result = main.list_depositions("case-1")

    assert [doc["_id"] for doc in result] == ["dep:2", "dep:1"]


def test_list_depositions_dashboard_handles_missing_and_none_scores(monkeypatch):
    couchdb = Mock()
    couchdb.list_depositions.return_value = [
        {"_id": "dep:1", "contradiction_score": None},
        {"_id": "dep:2", "contradiction_score": 25},
        {"_id": "dep:3"},
        {"_id": "dep:4", "contradiction_score": "high"},
    ]
    monkeypatch.setattr(main, "couchdb", couchdb)

    result = main.list_depositions("case-1")

    assert [doc["_id"] for doc in result] == ["dep:2", "dep:1", "dep:3", "dep:4"]


def test_get_deposition_success_and_not_found(monkeypatch):
    couchdb = Mock()
    couchdb.get_doc.return_value = {"_id": "dep:1"}
    monkeypatch.setattr(main, "couchdb", couchdb)

    assert main.get_deposition("dep:1") == {"_id": "dep:1"}

    couchdb.get_doc.side_effect = RuntimeError("missing")
    with pytest.raises(HTTPException) as exc:
        main.get_deposition("dep:2")
    assert exc.value.status_code == 404


def test_chat_success(monkeypatch):
    couchdb = Mock()
    couchdb.get_doc.return_value = {"_id": "dep:1", "case_id": "case-1"}
    couchdb.list_depositions.return_value = [{"_id": "dep:1"}, {"_id": "dep:2"}]
    chat_service = Mock()
    chat_service.respond.return_value = "Short answer: ok\nDetails:\n- a\n- b"

    monkeypatch.setattr(main, "couchdb", couchdb)
    monkeypatch.setattr(main, "chat_service", chat_service)

    response = main.chat(
        ChatRequest(
            case_id="case-1",
            deposition_id="dep:1",
            message="hello",
            llm_provider="ollama",
            llm_model="llama3.3",
        )
    )

    assert response.response.startswith("Short answer:")
    assert response.thought_stream is not None
    assert response.thought_stream.attorney == []
    chat_service.respond.assert_called_once()
    assert chat_service.respond.call_args.kwargs["llm_provider"] == "ollama"
    assert chat_service.respond.call_args.kwargs["llm_model"] == "llama3.3"


def test_chat_success_with_trace_provider(monkeypatch):
    class TraceChatService:
        def respond_with_trace(self, *_args, **_kwargs):
            return (
                "Short answer: traced\nDetails:\n- a\n- b",
                [
                    {
                        "persona": "Persona:Attorney",
                        "phase": "chat_response",
                        "llm_provider": "openai",
                        "llm_model": "gpt-5.2",
                        "notes": "trace note",
                    }
                ],
            )

    couchdb = Mock()
    couchdb.get_doc.return_value = {"_id": "dep:1", "case_id": "case-1"}
    couchdb.list_depositions.return_value = [{"_id": "dep:1"}, {"_id": "dep:2"}]
    monkeypatch.setattr(main, "couchdb", couchdb)
    monkeypatch.setattr(main, "chat_service", TraceChatService())

    response = main.chat(ChatRequest(case_id="case-1", deposition_id="dep:1", message="hello"))
    assert response.response.startswith("Short answer:")
    assert response.thought_stream is not None
    assert len(response.thought_stream.attorney) == 1
    assert response.thought_stream.attorney[0].phase == "chat_response"


def test_chat_wraps_service_errors(monkeypatch):
    couchdb = Mock()
    couchdb.get_doc.return_value = {"_id": "dep:1", "case_id": "case-1"}
    couchdb.list_depositions.return_value = [{"_id": "dep:1"}, {"_id": "dep:2"}]
    chat_service = Mock()
    chat_service.respond.side_effect = RuntimeError("model failed")

    monkeypatch.setattr(main, "couchdb", couchdb)
    monkeypatch.setattr(main, "chat_service", chat_service)

    with pytest.raises(HTTPException) as exc:
        main.chat(ChatRequest(case_id="case-1", deposition_id="dep:1", message="hello"))

    assert exc.value.status_code == 502
    assert "model failed" in exc.value.detail


def test_chat_errors(monkeypatch):
    couchdb = Mock()
    couchdb.get_doc.side_effect = RuntimeError("missing")
    monkeypatch.setattr(main, "couchdb", couchdb)

    with pytest.raises(HTTPException) as missing:
        main.chat(ChatRequest(case_id="case-1", deposition_id="dep:1", message="hello"))
    assert missing.value.status_code == 404

    couchdb.get_doc.side_effect = None
    couchdb.get_doc.return_value = {"_id": "dep:1", "case_id": "other-case"}
    with pytest.raises(HTTPException) as mismatch:
        main.chat(ChatRequest(case_id="case-1", deposition_id="dep:1", message="hello"))
    assert mismatch.value.status_code == 400


def test_reason_contradiction_success(monkeypatch):
    couchdb = Mock()
    couchdb.get_doc.return_value = {"_id": "dep:1", "case_id": "case-1"}
    couchdb.list_depositions.return_value = [{"_id": "dep:1"}, {"_id": "dep:2"}]
    chat_service = Mock()
    chat_service.reason_about_contradiction.return_value = "Short answer: ok\nDetails:\n- a\n- b\n- c"

    monkeypatch.setattr(main, "couchdb", couchdb)
    monkeypatch.setattr(main, "chat_service", chat_service)

    request = ContradictionReasonRequest(
        case_id="case-1",
        deposition_id="dep:1",
        llm_provider="ollama",
        llm_model="llama3.3",
        contradiction=ContradictionFinding(
            other_deposition_id="dep:2",
            other_witness_name="Peer",
            topic="Timeline",
            rationale="Mismatch",
            severity=60,
        ),
    )

    response = main.reason_contradiction(request)

    assert response.response.startswith("Short answer:")
    chat_service.reason_about_contradiction.assert_called_once()
    assert chat_service.reason_about_contradiction.call_args.kwargs["llm_provider"] == "ollama"
    assert chat_service.reason_about_contradiction.call_args.kwargs["llm_model"] == "llama3.3"


def test_reason_contradiction_wraps_service_errors(monkeypatch):
    couchdb = Mock()
    couchdb.get_doc.return_value = {"_id": "dep:1", "case_id": "case-1"}
    couchdb.list_depositions.return_value = [{"_id": "dep:1"}, {"_id": "dep:2"}]
    chat_service = Mock()
    chat_service.reason_about_contradiction.side_effect = RuntimeError("model failed")

    monkeypatch.setattr(main, "couchdb", couchdb)
    monkeypatch.setattr(main, "chat_service", chat_service)

    request = ContradictionReasonRequest(
        case_id="case-1",
        deposition_id="dep:1",
        contradiction=ContradictionFinding(
            other_deposition_id="dep:2",
            other_witness_name="Peer",
            topic="Timeline",
            rationale="Mismatch",
            severity=60,
        ),
    )

    with pytest.raises(HTTPException) as exc:
        main.reason_contradiction(request)

    assert exc.value.status_code == 502
    assert "model failed" in exc.value.detail


def test_reason_contradiction_errors(monkeypatch):
    couchdb = Mock()
    couchdb.get_doc.side_effect = RuntimeError("missing")
    monkeypatch.setattr(main, "couchdb", couchdb)

    request = ContradictionReasonRequest(
        case_id="case-1",
        deposition_id="dep:1",
        contradiction=ContradictionFinding(
            other_deposition_id="dep:2",
            other_witness_name="Peer",
            topic="Timeline",
            rationale="Mismatch",
            severity=60,
        ),
    )

    with pytest.raises(HTTPException) as missing:
        main.reason_contradiction(request)
    assert missing.value.status_code == 404

    couchdb.get_doc.side_effect = None
    couchdb.get_doc.return_value = {"_id": "dep:1", "case_id": "other-case"}
    with pytest.raises(HTTPException) as mismatch:
        main.reason_contradiction(request)
    assert mismatch.value.status_code == 400


def test_summarize_focused_reasoning_success(monkeypatch):
    couchdb = Mock()
    couchdb.get_doc.return_value = {"_id": "dep:1", "case_id": "case-1"}
    chat_service = Mock()
    chat_service.summarize_focused_reasoning.return_value = "Short answer: Condensed conflict summary."
    save_memory = Mock()
    upsert_case = Mock()

    monkeypatch.setattr(main, "couchdb", couchdb)
    monkeypatch.setattr(main, "chat_service", chat_service)
    monkeypatch.setattr(main, "_save_case_memory", save_memory)
    monkeypatch.setattr(main, "_upsert_case_doc", upsert_case)
    monkeypatch.setattr(main, "_safe_case_depositions", lambda _case_id: [{"_id": "dep:1"}])

    response = main.summarize_focused_reasoning(
        FocusedReasoningSummaryRequest(
            case_id="case-1",
            deposition_id="dep:1",
            reasoning_text="Short answer: Full focused reasoning.",
            llm_provider="ollama",
            llm_model="llama3.3",
        )
    )

    assert response.summary == "Short answer: Condensed conflict summary."
    chat_service.summarize_focused_reasoning.assert_called_once()
    assert chat_service.summarize_focused_reasoning.call_args.kwargs["llm_provider"] == "ollama"
    assert chat_service.summarize_focused_reasoning.call_args.kwargs["llm_model"] == "llama3.3"
    save_memory.assert_called_once()
    upsert_case.assert_called_once()


def test_summarize_focused_reasoning_wraps_service_errors(monkeypatch):
    couchdb = Mock()
    couchdb.get_doc.return_value = {"_id": "dep:1", "case_id": "case-1"}
    chat_service = Mock()
    chat_service.summarize_focused_reasoning.side_effect = RuntimeError("summary failed")

    monkeypatch.setattr(main, "couchdb", couchdb)
    monkeypatch.setattr(main, "chat_service", chat_service)

    with pytest.raises(HTTPException) as exc:
        main.summarize_focused_reasoning(
            FocusedReasoningSummaryRequest(
                case_id="case-1",
                deposition_id="dep:1",
                reasoning_text="Short answer: Full focused reasoning.",
            )
        )

    assert exc.value.status_code == 502
    assert "summary failed" in exc.value.detail


def test_summarize_focused_reasoning_errors(monkeypatch):
    couchdb = Mock()
    couchdb.get_doc.side_effect = RuntimeError("missing")
    monkeypatch.setattr(main, "couchdb", couchdb)

    with pytest.raises(HTTPException) as missing:
        main.summarize_focused_reasoning(
            FocusedReasoningSummaryRequest(
                case_id="case-1",
                deposition_id="dep:1",
                reasoning_text="Short answer: Full focused reasoning.",
            )
        )
    assert missing.value.status_code == 404

    couchdb.get_doc.side_effect = None
    couchdb.get_doc.return_value = {"_id": "dep:1", "case_id": "other-case"}
    with pytest.raises(HTTPException) as mismatch:
        main.summarize_focused_reasoning(
            FocusedReasoningSummaryRequest(
                case_id="case-1",
                deposition_id="dep:1",
                reasoning_text="Short answer: Full focused reasoning.",
            )
        )
    assert mismatch.value.status_code == 400


def test_compute_deposition_sentiment_labels_negative_and_neutral():
    negative = main._compute_deposition_sentiment(
        "The witness described fear, retaliation, conflict, and a violent forced encounter."
    )
    neutral = main._compute_deposition_sentiment("This deposition states dates, names, and routine observations.")

    assert negative.label == "negative"
    assert negative.negative_matches >= 4
    assert negative.score < 0
    assert neutral.label == "neutral"
    assert neutral.score == 0


def test_deposition_sentiment_success(monkeypatch):
    couchdb = Mock()
    couchdb.get_doc.return_value = {
        "_id": "dep:1",
        "case_id": "case-1",
        "raw_text": "The witness remained calm and consistent, but described one conflict.",
    }
    save_memory = Mock()
    upsert_case = Mock()

    monkeypatch.setattr(main, "couchdb", couchdb)
    monkeypatch.setattr(main, "_save_case_memory", save_memory)
    monkeypatch.setattr(main, "_upsert_case_doc", upsert_case)
    monkeypatch.setattr(main, "_safe_case_depositions", lambda _case_id: [{"_id": "dep:1"}])

    response = main.deposition_sentiment(
        DepositionSentimentRequest(case_id="case-1", deposition_id="dep:1")
    )

    assert response.label in {"positive", "neutral", "negative"}
    assert response.word_count > 0
    save_memory.assert_called_once()
    upsert_case.assert_called_once()


def test_deposition_sentiment_errors(monkeypatch):
    couchdb = Mock()
    couchdb.get_doc.side_effect = RuntimeError("missing")
    monkeypatch.setattr(main, "couchdb", couchdb)

    with pytest.raises(HTTPException) as missing:
        main.deposition_sentiment(
            DepositionSentimentRequest(case_id="case-1", deposition_id="dep:1")
        )
    assert missing.value.status_code == 404

    couchdb.get_doc.side_effect = None
    couchdb.get_doc.return_value = {"_id": "dep:1", "case_id": "other-case", "raw_text": "text"}
    with pytest.raises(HTTPException) as mismatch:
        main.deposition_sentiment(
            DepositionSentimentRequest(case_id="case-1", deposition_id="dep:1")
        )
    assert mismatch.value.status_code == 400
