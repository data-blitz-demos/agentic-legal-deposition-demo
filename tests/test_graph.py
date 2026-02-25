from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from backend.app import graph as graph_module
from backend.app.graph import DepositionWorkflow
from backend.app.models import Claim, ContradictionAssessment, ContradictionFinding, DepositionSchema


def make_workflow(max_context_depositions: int = 2) -> DepositionWorkflow:
    workflow = object.__new__(DepositionWorkflow)
    workflow.settings = SimpleNamespace(
        max_context_depositions=max_context_depositions,
        default_llm_provider="openai",
        model_name="gpt-5.2",
        openai_api_key="key",
        openai_models="gpt-5.2",
        ollama_url="http://localhost:11434",
        ollama_default_model="llama3.3",
        ollama_models="llama3.3",
    )
    workflow.couchdb = Mock()
    workflow.llm = Mock()
    workflow.graph = Mock()
    return workflow


def sample_assessment(score: int = 50) -> ContradictionAssessment:
    return ContradictionAssessment(
        contradiction_score=score,
        flagged=score >= 35,
        explanation="assessment",
        contradictions=[
            ContradictionFinding(
                other_deposition_id="dep:peer",
                other_witness_name="Peer",
                topic="Timeline",
                rationale="Mismatch",
                severity=score,
            )
        ],
    )


def test_init_sets_llm_and_graph(monkeypatch):
    llm_obj = object()
    graph_obj = object()

    monkeypatch.setattr(graph_module, "ChatOpenAI", lambda **kwargs: llm_obj)
    monkeypatch.setattr(DepositionWorkflow, "_build_graph", lambda self: graph_obj)

    settings = SimpleNamespace(model_name="gpt-test", openai_api_key="key")
    couchdb = Mock()
    workflow = DepositionWorkflow(settings, couchdb)

    assert workflow.llm is llm_obj
    assert workflow.graph is graph_obj
    assert workflow.couchdb is couchdb


def test_build_graph_registers_nodes_and_edges(monkeypatch):
    calls = {"nodes": [], "edges": [], "entry": None}

    class StubBuilder:
        def add_node(self, name, fn):
            calls["nodes"].append(name)

        def set_entry_point(self, name):
            calls["entry"] = name

        def add_edge(self, src, dst):
            calls["edges"].append((src, dst))

        def compile(self):
            return "compiled"

    monkeypatch.setattr(graph_module, "StateGraph", lambda _state: StubBuilder())
    workflow = make_workflow()

    result = workflow._build_graph()

    assert result == "compiled"
    assert calls["entry"] == "read_file"
    assert "map_deposition" in calls["nodes"]
    assert ("persist_assessment", graph_module.END) in calls["edges"]


def test_run_invokes_compiled_graph():
    workflow = make_workflow()
    workflow.graph.invoke.return_value = {"ok": True}

    result = workflow.run(case_id="case-1", file_path="/tmp/file.txt")

    assert result == {"ok": True}
    workflow.graph.invoke.assert_called_once_with(
        {
            "case_id": "case-1",
            "file_path": "/tmp/file.txt",
            "llm_provider": "",
            "llm_model": "",
        }
    )


def test_run_invokes_compiled_graph_with_selected_llm():
    workflow = make_workflow()
    workflow.graph.invoke.return_value = {"ok": True}

    workflow.run(
        case_id="case-1",
        file_path="/tmp/file.txt",
        llm_provider="ollama",
        llm_model="llama3.3",
    )

    workflow.graph.invoke.assert_called_once_with(
        {
            "case_id": "case-1",
            "file_path": "/tmp/file.txt",
            "llm_provider": "ollama",
            "llm_model": "llama3.3",
        }
    )


def test_reassess_case_updates_each_doc():
    workflow = make_workflow()
    docs = [
        {"_id": "dep:1", "case_id": "c"},
        {"_id": "dep:2", "case_id": "c"},
    ]
    workflow.couchdb.list_depositions.return_value = docs
    workflow._assess_deposition_against_peers = Mock(return_value=sample_assessment(60))
    workflow.couchdb.update_doc.side_effect = lambda doc: {**doc, "_rev": "1-a"}

    result = workflow.reassess_case("c", llm_provider="ollama", llm_model="llama3.3")

    assert len(result) == 2
    assert workflow._assess_deposition_against_peers.call_count == 2
    assert workflow.couchdb.update_doc.call_count == 2
    for call in workflow._assess_deposition_against_peers.call_args_list:
        assert call.kwargs["llm_provider"] == "ollama"
        assert call.kwargs["llm_model"] == "llama3.3"


def test_read_file_decodes_fallback_encoding(tmp_path):
    workflow = make_workflow()
    file_path = tmp_path / "dep.txt"
    file_path.write_bytes("caf\xe9".encode("cp1252"))

    state = workflow._read_file({"file_path": str(file_path)})

    assert state["raw_text"] == "café"


def test_read_file_raises_when_all_decodes_fail(monkeypatch):
    workflow = make_workflow()

    class BadBytes:
        def decode(self, _encoding: str) -> str:
            raise UnicodeDecodeError("codec", b"a", 0, 1, "bad")

    monkeypatch.setattr(Path, "read_bytes", lambda _self: BadBytes())

    with pytest.raises(ValueError, match="Could not decode deposition file"):
        workflow._read_file({"file_path": "/tmp/bad.txt"})


def test_map_deposition_success_sets_case_and_file(monkeypatch):
    workflow = make_workflow()
    parser = Mock()
    parser.invoke.return_value = DepositionSchema(
        case_id="x",
        file_name="x",
        witness_name="Jane",
        witness_role="Manager",
        deposition_date="2025-01-01",
        summary="Summary",
        claims=[],
    )
    workflow.llm.with_structured_output.return_value = parser
    monkeypatch.setattr(graph_module, "load_schema", lambda _name: {"title": "DepositionSchema", "type": "object"})

    result = workflow._map_deposition(
        {"case_id": "case-1", "file_path": "/tmp/witness.txt", "raw_text": "raw"}
    )

    deposition = result["deposition"]
    assert deposition.case_id == "case-1"
    assert deposition.file_name == "witness.txt"


def test_map_deposition_uses_prompt_templates(monkeypatch):
    workflow = make_workflow()
    parser = Mock()
    parser.invoke.return_value = DepositionSchema(
        case_id="x",
        file_name="x",
        witness_name="Jane",
        witness_role="Manager",
        deposition_date="2025-01-01",
        summary="Summary",
        claims=[],
    )
    workflow.llm.with_structured_output.return_value = parser
    calls: list[str] = []

    def fake_render(name: str, **_kwargs):
        calls.append(name)
        return f"PROMPT::{name}"

    monkeypatch.setattr(graph_module, "render_prompt", fake_render)
    monkeypatch.setattr(graph_module, "load_schema", lambda _name: {"title": "DepositionSchema", "type": "object"})

    workflow._map_deposition(
        {"case_id": "case-1", "file_path": "/tmp/witness.txt", "raw_text": "raw"}
    )

    assert calls == ["map_deposition_system", "map_deposition_user"]
    messages = parser.invoke.call_args.args[0]
    assert str(messages[0].content) == "PROMPT::map_deposition_system"
    assert str(messages[1].content) == "PROMPT::map_deposition_user"


def test_map_deposition_raises_with_fix_on_exception(monkeypatch):
    workflow = make_workflow()
    parser = Mock()
    parser.invoke.side_effect = RuntimeError("quota")
    workflow.llm.with_structured_output.return_value = parser
    monkeypatch.setattr(graph_module, "llm_failure_message", lambda *_args: "Possible fix: test")
    monkeypatch.setattr(graph_module, "load_schema", lambda _name: {"title": "DepositionSchema", "type": "object"})

    with pytest.raises(RuntimeError, match="Possible fix: test"):
        workflow._map_deposition(
            {"case_id": "case-1", "file_path": "/tmp/fallback.txt", "raw_text": "raw"}
        )


def test_map_deposition_uses_selected_llm_override(monkeypatch):
    workflow = make_workflow()
    selected_llm = Mock()
    parser = Mock()
    parser.invoke.return_value = DepositionSchema(
        case_id="x",
        file_name="x",
        witness_name="Jane",
        witness_role="Manager",
        deposition_date="2025-01-01",
        summary="Summary",
        claims=[],
    )
    selected_llm.with_structured_output.return_value = parser
    build = Mock(return_value=selected_llm)
    monkeypatch.setattr(graph_module, "build_chat_model", build)
    monkeypatch.setattr(graph_module, "load_schema", lambda _name: {"title": "DepositionSchema", "type": "object"})

    workflow._map_deposition(
        {
            "case_id": "case-1",
            "file_path": "/tmp/witness.txt",
            "raw_text": "raw",
            "llm_provider": "ollama",
            "llm_model": "llama3.3",
        }
    )

    build.assert_called_once()
    selected_llm.with_structured_output.assert_called_once()


def test_map_deposition_backfills_claims_when_llm_returns_empty(monkeypatch):
    workflow = make_workflow()
    parser = Mock()
    parser.invoke.return_value = DepositionSchema(
        case_id="x",
        file_name="x",
        witness_name="Jane",
        witness_role="Manager",
        deposition_date="2025-01-01",
        summary="Summary",
        claims=[],
    )
    workflow.llm.with_structured_output.return_value = parser
    monkeypatch.setattr(graph_module, "load_schema", lambda _name: {"title": "DepositionSchema", "type": "object"})
    fallback_claim = Claim(
        topic="Timeline",
        statement="I arrived at 8:45 a.m.",
        confidence=0.55,
        source_quote="I arrived at 8:45 a.m.",
    )
    workflow._fallback_map_deposition = Mock(
        return_value=DepositionSchema(
            case_id="case-1",
            file_name="witness.txt",
            witness_name="Jane",
            witness_role="Manager",
            summary="Summary",
            claims=[fallback_claim],
        )
    )

    result = workflow._map_deposition(
        {"case_id": "case-1", "file_path": "/tmp/witness.txt", "raw_text": "raw"}
    )

    assert result["deposition"].claims == [fallback_claim]
    workflow._fallback_map_deposition.assert_called_once()


def test_map_deposition_validates_dict_payload_from_json_schema(monkeypatch):
    workflow = make_workflow()
    parser = Mock()
    parser.invoke.return_value = {
        "case_id": "x",
        "file_name": "x",
        "witness_name": "Jane",
        "witness_role": "Manager",
        "deposition_date": "2025-01-01",
        "summary": "Summary",
        "claims": [],
    }
    workflow.llm.with_structured_output.return_value = parser
    monkeypatch.setattr(graph_module, "load_schema", lambda _name: {"title": "DepositionSchema", "type": "object"})

    result = workflow._map_deposition(
        {"case_id": "case-1", "file_path": "/tmp/witness.txt", "raw_text": "raw"}
    )

    assert isinstance(result["deposition"], DepositionSchema)


def test_fallback_map_deposition_extracts_metadata_and_claims():
    workflow = make_workflow()
    raw_text = (
        "Deposition of Jane Doe\n"
        "Date: April 14, 2025\n"
        "Role: Operations Manager\n\n"
        "I arrived at 8:45 a.m.\n"
        "Camera was not functioning.\n"
    )

    mapped = workflow._fallback_map_deposition({"case_id": "case-1", "raw_text": raw_text}, "jane.txt")

    assert mapped.witness_name == "Jane Doe"
    assert mapped.witness_role == "Operations Manager"
    assert mapped.deposition_date == "April 14, 2025"
    assert mapped.claims


def test_fallback_map_deposition_without_headers_uses_filename_defaults():
    workflow = make_workflow()
    mapped = workflow._fallback_map_deposition({"case_id": "case-1", "raw_text": ""}, "john_doe.txt")

    assert mapped.witness_name == "John Doe"
    assert mapped.witness_role == "Unknown role"
    assert mapped.summary == "No narrative testimony was available."
    assert mapped.claims == []


def test_extract_first_group_and_split_sentences():
    workflow = make_workflow()

    assert workflow._extract_first_group("Date: Today", r"^Date:\s*(.+)$") == "Today"
    assert workflow._extract_first_group("Nothing", r"^Date:\s*(.+)$") is None
    assert workflow._split_sentences("One. Two? Three!") == ["One.", "Two?", "Three!"]


def test_infer_topic_known_and_unknown():
    workflow = make_workflow()

    assert workflow._infer_topic("The loading dock camera was down") == "Camera status"
    assert workflow._infer_topic("Unexpected phrasing token") == "Unexpected Phrasing Token"
    assert workflow._infer_topic("!!!") == "General testimony"


def test_save_deposition_calls_couchdb_save():
    workflow = make_workflow()
    deposition = DepositionSchema(
        case_id="case-1",
        file_name="witness file.txt",
        witness_name="Jane",
        witness_role="Manager",
        summary="Summary",
        claims=[Claim(topic="T", statement="S", confidence=0.5, source_quote="Q")],
    )
    workflow.couchdb.save_doc.return_value = {"_id": "dep:case-1:witness-file", "_rev": "1-a"}

    result = workflow._save_deposition({"case_id": "case-1", "deposition": deposition, "raw_text": "raw"})

    assert result["deposition_doc"]["_id"] == "dep:case-1:witness-file"
    workflow.couchdb.save_doc.assert_called_once()


def test_load_other_depositions_respects_limit():
    workflow = make_workflow(max_context_depositions=1)
    workflow.couchdb.list_depositions.return_value = [
        {"_id": "dep:1"},
        {"_id": "dep:2"},
        {"_id": "dep:3"},
    ]

    result = workflow._load_other_depositions({"case_id": "case", "deposition_doc": {"_id": "dep:1"}})

    assert result["other_depositions"] == [{"_id": "dep:2"}]


def test_evaluate_contradictions_uses_assessor():
    workflow = make_workflow()
    workflow._assess_deposition_against_peers = Mock(return_value=sample_assessment(44))

    result = workflow._evaluate_contradictions(
        {
            "deposition_doc": {"_id": "dep:1"},
            "other_depositions": [{"_id": "dep:2"}],
            "llm_provider": "ollama",
            "llm_model": "llama3.3",
        }
    )

    assert result["assessment"].contradiction_score == 44
    workflow._assess_deposition_against_peers.assert_called_once()
    assert workflow._assess_deposition_against_peers.call_args.kwargs["llm_provider"] == "ollama"


def test_assess_deposition_returns_zero_when_no_peers():
    workflow = make_workflow()

    result = workflow._assess_deposition_against_peers({"_id": "dep:1"}, [])

    assert result.contradiction_score == 0
    assert result.flagged is False


def test_assess_deposition_uses_llm_when_available():
    workflow = make_workflow()
    assessment = sample_assessment(70)
    assessor = Mock()
    assessor.invoke.return_value = assessment
    workflow.llm.with_structured_output.return_value = assessor

    result = workflow._assess_deposition_against_peers(
        {
            "_id": "dep:1",
            "witness_name": "Jane",
            "witness_role": "Manager",
            "summary": "Summary",
            "claims": [],
        },
        [{"_id": "dep:2", "witness_name": "Alan", "summary": "Peer", "claims": []}],
    )

    assert result == assessment


def test_assess_deposition_uses_fallback_when_llm_returns_zero(monkeypatch):
    workflow = make_workflow()
    assessor = Mock()
    assessor.invoke.return_value = ContradictionAssessment(
        contradiction_score=0,
        flagged=False,
        explanation="No direct contradictions found.",
        contradictions=[],
    )
    workflow.llm.with_structured_output.return_value = assessor
    fallback = sample_assessment(65)
    workflow._fallback_assess_deposition = Mock(return_value=fallback)

    result = workflow._assess_deposition_against_peers(
        {
            "_id": "dep:1",
            "witness_name": "Jane",
            "witness_role": "Manager",
            "summary": "Summary",
            "claims": [{"topic": "Timeline", "statement": "I arrived at 8:45 a.m."}],
        },
        [
            {
                "_id": "dep:2",
                "witness_name": "Alan",
                "summary": "Peer",
                "claims": [{"topic": "Timeline", "statement": "She arrived at 9:15 a.m."}],
            }
        ],
    )

    assert result == fallback
    workflow._fallback_assess_deposition.assert_called_once()


def test_assess_deposition_skips_fallback_when_llm_has_contradictions(monkeypatch):
    workflow = make_workflow()
    assessment = sample_assessment(70)
    assessor = Mock()
    assessor.invoke.return_value = assessment
    workflow.llm.with_structured_output.return_value = assessor
    workflow._fallback_assess_deposition = Mock()

    result = workflow._assess_deposition_against_peers(
        {
            "_id": "dep:1",
            "witness_name": "Jane",
            "witness_role": "Manager",
            "summary": "Summary",
            "claims": [{"topic": "Timeline", "statement": "I arrived at 8:45 a.m."}],
        },
        [
            {
                "_id": "dep:2",
                "witness_name": "Alan",
                "summary": "Peer",
                "claims": [{"topic": "Timeline", "statement": "She arrived at 9:15 a.m."}],
            }
        ],
    )

    assert result == assessment
    workflow._fallback_assess_deposition.assert_not_called()


def test_assess_deposition_keeps_llm_result_when_fallback_not_stronger(monkeypatch):
    workflow = make_workflow()
    assessor = Mock()
    llm_assessment = ContradictionAssessment(
        contradiction_score=10,
        flagged=False,
        explanation="Weak inconsistency noted.",
        contradictions=[],
    )
    assessor.invoke.return_value = llm_assessment
    workflow.llm.with_structured_output.return_value = assessor
    workflow._fallback_assess_deposition = Mock(
        return_value=ContradictionAssessment(
            contradiction_score=0,
            flagged=False,
            explanation="No clear direct contradictions were found in fallback analysis.",
            contradictions=[],
        )
    )

    result = workflow._assess_deposition_against_peers(
        {
            "_id": "dep:1",
            "witness_name": "Jane",
            "witness_role": "Manager",
            "summary": "Summary",
            "claims": [{"topic": "Timeline", "statement": "I arrived at 8:45 a.m."}],
        },
        [
            {
                "_id": "dep:2",
                "witness_name": "Alan",
                "summary": "Peer",
                "claims": [{"topic": "Timeline", "statement": "She arrived around 8:50 a.m."}],
            }
        ],
    )

    assert result == llm_assessment
    workflow._fallback_assess_deposition.assert_called_once()


def test_assess_deposition_uses_prompt_templates(monkeypatch):
    workflow = make_workflow()
    assessor = Mock()
    assessor.invoke.return_value = sample_assessment(50)
    workflow.llm.with_structured_output.return_value = assessor
    calls: list[str] = []

    def fake_render(name: str, **_kwargs):
        calls.append(name)
        return f"PROMPT::{name}"

    monkeypatch.setattr(graph_module, "render_prompt", fake_render)

    workflow._assess_deposition_against_peers(
        {
            "_id": "dep:1",
            "witness_name": "Jane",
            "witness_role": "Manager",
            "summary": "Summary",
            "claims": [],
        },
        [{"_id": "dep:2", "witness_name": "Alan", "summary": "Peer", "claims": []}],
    )

    assert calls == ["assess_contradictions_system", "assess_contradictions_user"]
    messages = assessor.invoke.call_args.args[0]
    assert str(messages[0].content) == "PROMPT::assess_contradictions_system"
    assert str(messages[1].content) == "PROMPT::assess_contradictions_user"


def test_assess_deposition_raises_with_fix_when_llm_errors(monkeypatch):
    workflow = make_workflow()
    assessor = Mock()
    assessor.invoke.side_effect = RuntimeError("quota")
    workflow.llm.with_structured_output.return_value = assessor
    monkeypatch.setattr(graph_module, "llm_failure_message", lambda *_args: "Possible fix: test")

    with pytest.raises(RuntimeError, match="Possible fix: test"):
        workflow._assess_deposition_against_peers(
            {
                "_id": "dep:1",
                "witness_name": "Jane",
                "witness_role": "Manager",
                "summary": "Summary",
                "claims": [],
            },
            [{"_id": "dep:2", "witness_name": "Alan", "summary": "Peer", "claims": []}],
        )


def test_assess_deposition_uses_selected_llm_override(monkeypatch):
    workflow = make_workflow()
    selected_llm = Mock()
    assessor = Mock()
    assessor.invoke.return_value = sample_assessment(55)
    selected_llm.with_structured_output.return_value = assessor
    build = Mock(return_value=selected_llm)
    monkeypatch.setattr(graph_module, "build_chat_model", build)

    result = workflow._assess_deposition_against_peers(
        {
            "_id": "dep:1",
            "witness_name": "Jane",
            "witness_role": "Manager",
            "summary": "Summary",
            "claims": [],
        },
        [{"_id": "dep:2", "witness_name": "Alan", "summary": "Peer", "claims": []}],
        llm_provider="ollama",
        llm_model="llama3.3",
    )

    assert result.contradiction_score == 55
    build.assert_called_once()


def test_fallback_assess_detects_conflict_and_scores():
    workflow = make_workflow()
    target = {
        "claims": [{"topic": "Timeline", "statement": "I did not drive forklift before 10:15 a.m."}],
    }
    peers = [
        {
            "_id": "dep:2",
            "witness_name": "Alan",
            "claims": [{"topic": "Timeline", "statement": "I saw him drive forklift at 9:10 a.m."}],
        }
    ]

    result = workflow._fallback_assess_deposition(target, peers)

    assert result.flagged is True
    assert result.contradictions


def test_fallback_assess_returns_zero_when_no_overlap():
    workflow = make_workflow()
    target = {"claims": [{"topic": "Topic A", "statement": "Apples are red."}]}
    peers = [{"_id": "dep:2", "witness_name": "Alan", "claims": [{"topic": "Topic B", "statement": "Cars are blue."}]}]

    result = workflow._fallback_assess_deposition(target, peers)

    assert result.contradiction_score == 0
    assert result.contradictions == []


def test_fallback_assess_handles_empty_claims_duplicate_keys_and_limit_breaks():
    workflow = make_workflow()
    target = {
        "claims": [{"topic": "SkipEmpty", "statement": ""}]
        + [
            {
                "topic": f"Topic{chr(65 + idx)}",
                "statement": "I did not drive forklift near loading dock at 9:00 a.m.",
            }
            for idx in range(12)
        ],
    }
    peers = [
        {
            "_id": "dep:2",
            "witness_name": "Alan",
            "claims": [
                {"topic": "SkipPeerEmpty", "statement": ""},
                {"topic": "NoOverlap", "statement": "Cars are blue and boats are green."},
                {"topic": "TopicA", "statement": "I drove forklift near loading dock at 8:30 a.m."},
                {"topic": "TopicA", "statement": "I drove forklift near loading dock at 8:30 a.m."},
            ],
        }
    ]

    result = workflow._fallback_assess_deposition(target, peers)

    assert len(result.contradictions) == 8
    assert result.contradiction_score >= 70


def test_topics_overlap_and_keyword_tokens():
    workflow = make_workflow()

    assert workflow._topics_overlap("Forklift incident", "forklift", "a", "b") is True
    assert workflow._topics_overlap("alpha", "beta", "red green blue", "cyan magenta yellow") is False

    tokens = workflow._keyword_tokens("The witness said the forklift and camera were operating")
    assert "forklift" in tokens
    assert "the" not in tokens


@pytest.mark.parametrize(
    "left,right,expected",
    [
        ("I did not strike rack C.", "I struck rack C.", True),
        ("I arrived at 9:10 a.m.", "I arrived at 9:40 a.m.", True),
        ("There were 2 drums.", "There were 5 drums.", True),
        ("The rack was damaged.", "The rack was damaged.", False),
    ],
)
def test_claims_conflict_cases(left, right, expected):
    workflow = make_workflow()
    conflict, _, _ = workflow._claims_conflict(left, right)
    assert conflict is expected


def test_contains_negation_and_extract_time_minutes():
    workflow = make_workflow()

    assert workflow._contains_negation("I did not see it") is True
    assert workflow._contains_negation("I saw it") is False
    assert workflow._extract_time_minutes("Event at 1:30 p.m.") == 13 * 60 + 30
    assert workflow._extract_time_minutes("Event at 12:05 am") == 5
    assert workflow._extract_time_minutes("No time") is None


def test_persist_assessment_updates_doc_and_saves():
    workflow = make_workflow()
    workflow.couchdb.update_doc.return_value = {"_id": "dep:1", "_rev": "2-b"}
    assessment = sample_assessment(68)

    result = workflow._persist_assessment(
        {
            "deposition_doc": {"_id": "dep:1", "contradictions": []},
            "assessment": assessment,
        }
    )

    assert result["deposition_doc"]["_rev"] == "2-b"
    workflow.couchdb.update_doc.assert_called_once()
