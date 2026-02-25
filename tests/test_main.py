from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import HTTPException

from backend.app import main
from backend.app.models import (
    ChatRequest,
    ContradictionFinding,
    ContradictionReasonRequest,
    DepositionRootRequest,
    IngestCaseRequest,
    RenameCaseRequest,
    SaveCaseRequest,
    SaveCaseVersionRequest,
)


@pytest.fixture(autouse=True)
def stub_llm_readiness(monkeypatch):
    monkeypatch.setattr(main, "ensure_llm_operational", lambda *_args, **_kwargs: None)


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
    startup = Mock()
    monkeypatch.setattr(main, "couchdb", couchdb)
    monkeypatch.setattr(main, "_ensure_startup_llm_connectivity", startup)

    async def run_lifespan() -> None:
        async with main.lifespan(main.app):
            pass

    asyncio.run(run_lifespan())

    startup.assert_called_once_with()
    couchdb.ensure_db.assert_called_once_with()
    couchdb.close.assert_called_once_with()


def test_lifespan_stops_when_startup_llm_check_fails(monkeypatch):
    couchdb = Mock()
    monkeypatch.setattr(main, "couchdb", couchdb)
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
    monkeypatch.setattr(main, "couchdb", couchdb)
    monkeypatch.setattr(main, "_utc_now_iso", lambda: "2026-01-01T00:00:00+00:00")

    main._save_case_memory("case-1", "langgraph", {"x": 1})

    saved = couchdb.save_doc.call_args.args[0]
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
    )

    updated = couchdb.update_doc.call_args.args[0]
    assert updated["deposition_count"] == 7
    assert updated["memory_entries"] == 4
    assert updated["last_action"] == "ingest"
    assert updated["last_directory"] == "/data/depositions/oj_simpson"
    assert updated["last_llm_provider"] == "ollama"
    assert updated["last_llm_model"] == "llama3.3"


def test_save_case_memory_and_upsert_case_doc_wrap_errors(monkeypatch):
    couchdb = Mock()
    couchdb.save_doc.side_effect = RuntimeError("save failed")
    monkeypatch.setattr(main, "couchdb", couchdb)

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
    )

    with pytest.raises(HTTPException) as save_exc:
        main.save_case(SaveCaseRequest(case_id="   "))
    assert save_exc.value.status_code == 400

    monkeypatch.setattr(main, "_list_case_summaries", lambda: [])
    fallback_payload = main.save_case(SaveCaseRequest(case_id="CASE-FALLBACK"))
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
    couchdb.find.side_effect = [
        [{"_id": "dep:1", "case_id": "CASE-001"}],
        RuntimeError("find failed"),
        [
            {"_id": "dep:1", "_rev": "1-a", "case_id": "CASE-001", "type": "deposition", "file_name": "a.txt"},
            {"_id": "mem:1", "_rev": "1-b", "case_id": "CASE-001", "type": "case_memory", "payload": {"x": 1}},
            {"_id": "case:1", "_rev": "1-c", "case_id": "CASE-001", "type": "case"},
        ],
    ]
    monkeypatch.setattr(main, "couchdb", couchdb)

    assert main._case_has_docs("CASE-001") is True
    assert main._case_has_docs("CASE-ERR") is False

    cloned = main._clone_case_contents("CASE-001", "CASE-NEW")
    assert cloned == 2
    assert couchdb.save_doc.call_count == 2
    first_saved = couchdb.save_doc.call_args_list[0].args[0]
    assert first_saved["case_id"] == "CASE-NEW"
    assert "_id" not in first_saved
    assert "_rev" not in first_saved


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
    chat_service.respond.assert_called_once()
    assert chat_service.respond.call_args.kwargs["llm_provider"] == "ollama"
    assert chat_service.respond.call_args.kwargs["llm_model"] == "llama3.3"


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
