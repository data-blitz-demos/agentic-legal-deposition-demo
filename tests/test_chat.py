from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from backend.app import chat as chat_module
from backend.app.chat import AttorneyChatService


def build_service_with_mock_llm() -> AttorneyChatService:
    service = object.__new__(AttorneyChatService)
    service.llm = Mock()
    service.settings = SimpleNamespace(
        default_llm_provider="openai",
        model_name="gpt-5.2",
        openai_api_key="key",
        ollama_url="http://localhost:11434",
        ollama_default_model="llama3.3",
        ollama_models="llama3.3",
        openai_models="gpt-5.2",
    )
    return service


def test_init_configures_chat_openai(monkeypatch):
    captured: dict = {}

    class StubChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(chat_module, "ChatOpenAI", StubChatOpenAI)
    settings = SimpleNamespace(model_name="gpt-test", openai_api_key="key")

    service = AttorneyChatService(settings)

    assert isinstance(service.llm, StubChatOpenAI)
    assert captured == {"model": "gpt-test", "api_key": "key", "temperature": 0.2}


def test_build_context_payload_includes_target_and_peers():
    service = build_service_with_mock_llm()
    deposition = {"_id": "dep:1", "witness_name": "Jane", "claims": [{"topic": "T"}]}
    peers = [{"_id": "dep:2", "witness_name": "Alan", "claims": []}]

    payload = service._build_context_payload(deposition, peers)

    assert payload["target_deposition"]["id"] == "dep:1"
    assert payload["peer_depositions"][0]["id"] == "dep:2"


def test_respond_uses_llm_when_available():
    service = build_service_with_mock_llm()
    service.llm.invoke.return_value = SimpleNamespace(content="Short answer: ok\nDetails:\n- a\n- b")

    result = service.respond(
        deposition={"_id": "dep:1", "witness_name": "Jane", "contradictions": []},
        peers=[],
        user_message="Summarize",
        history=[
            {"role": "assistant", "content": "prior"},
            {"role": "user", "content": "follow-up"},
            {"role": "user", "content": ""},
        ],
    )

    assert "Short answer:" in result
    service.llm.invoke.assert_called_once()


def test_respond_with_trace_returns_attorney_event():
    service = build_service_with_mock_llm()
    service.llm.invoke.return_value = SimpleNamespace(content="Short answer: ok\nDetails:\n- a\n- b")

    response, trace = service.respond_with_trace(
        deposition={"_id": "dep:1", "file_name": "witness.txt", "witness_name": "Jane", "contradictions": []},
        peers=[{"_id": "dep:2"}],
        user_message="Summarize",
        history=[{"role": "user", "content": "prior question"}],
    )

    assert response.startswith("Short answer:")
    assert len(trace) == 1
    assert trace[0]["persona"] == "Persona:Attorney"
    assert trace[0]["phase"] == "chat_response"
    assert "System Prompt" not in trace[0]["notes"]


def test_respond_normalizes_json_chat_output_to_descriptive_text():
    service = build_service_with_mock_llm()
    service.llm.invoke.return_value = SimpleNamespace(
        content=json.dumps(
            {
                "short_answer": "The key issue is a timeline contradiction with moderate litigation risk.",
                "details": [
                    "Witness timing conflicts with the peer account by roughly 30 minutes.",
                    "Corroboration should focus on timestamped exhibits before deposition follow-up.",
                ],
            }
        )
    )

    response, trace = service.respond_with_trace(
        deposition={
            "_id": "dep:1",
            "file_name": "witness.txt",
            "witness_name": "Jane",
            "contradiction_score": 48,
            "contradictions": [],
        },
        peers=[],
        user_message="Summarize risk",
        history=[],
    )

    assert response.startswith("Short answer:")
    assert "Details:" in response
    assert "- Witness timing conflicts" in response
    assert not response.strip().startswith("{")
    assert trace[0]["output_preview"].startswith("Short answer:")


def test_normalize_chat_output_falls_back_when_raw_text_has_no_bullets():
    service = build_service_with_mock_llm()

    normalized = service._normalize_chat_output(
        "This is an unstructured answer without any bullet formatting.",
        deposition={"witness_name": "Jane", "contradiction_score": 22, "contradictions": []},
        user_message="focus on credibility",
    )

    assert normalized.startswith("Short answer:")
    assert "Details:" in normalized
    assert "- Requested focus: focus on credibility" in normalized


def test_normalize_chat_output_replaces_placeholder_and_expands_single_bullet():
    service = build_service_with_mock_llm()

    normalized = service._normalize_chat_output(
        "Short answer: <1 sentence>\n- First concrete point.",
        deposition={"witness_name": "Jane", "contradiction_score": 22, "contradictions": []},
        user_message="focus on credibility",
    )

    assert normalized.startswith("Short answer: Jane's testimony is currently low-risk")
    assert "- First concrete point." in normalized
    assert "- Requested focus: focus on credibility" in normalized


def test_default_chat_sections_with_contradictions_uses_conflict_detail():
    service = build_service_with_mock_llm()

    sections = service._default_chat_sections(
        deposition={
            "witness_name": "Jane",
            "contradiction_score": 66,
            "contradictions": [
                {
                    "topic": "Timeline",
                    "other_witness_name": "Alan",
                    "rationale": "Mismatch",
                }
            ],
        },
        user_message="",
    )

    assert "risk at 66/100" in sections["short_answer"]
    assert sections["details"][0].startswith("Primary conflict is on Timeline versus Alan: Mismatch")


def test_render_chat_sections_fills_missing_details():
    service = build_service_with_mock_llm()
    rendered = service._render_chat_sections({"short_answer": "Answer", "details": []})
    assert rendered == "Short answer: Answer\nDetails:\n- No additional details are available yet."


def test_respond_raises_with_fix_on_llm_error():
    service = build_service_with_mock_llm()
    service.llm.invoke.side_effect = RuntimeError("quota")

    with pytest.raises(RuntimeError, match="Possible fix:"):
        service.respond(
            deposition={
                "witness_name": "Jane",
                "contradiction_score": 60,
                "contradictions": [
                    {
                        "topic": "Timeline",
                        "other_witness_name": "Alan",
                        "rationale": "Conflict",
                    }
                ],
            },
            peers=[],
            user_message="Focus on timeline",
            history=[],
        )


def test_respond_uses_selected_llm_override(monkeypatch):
    service = build_service_with_mock_llm()
    service.settings = SimpleNamespace()
    selected_llm = Mock()
    selected_llm.invoke.return_value = SimpleNamespace(content="Short answer: override\nDetails:\n- a\n- b")
    build = Mock(return_value=selected_llm)
    monkeypatch.setattr(chat_module, "build_chat_model", build)

    result = service.respond(
        deposition={"_id": "dep:1", "witness_name": "Jane", "contradictions": []},
        peers=[],
        user_message="Use ollama",
        history=[],
        llm_provider="ollama",
        llm_model="llama3.3",
    )

    assert result.startswith("Short answer:")
    build.assert_called_once()
    selected_llm.invoke.assert_called_once()


def test_respond_uses_prompt_templates(monkeypatch):
    service = build_service_with_mock_llm()
    service.llm.invoke.return_value = SimpleNamespace(content="Short answer: ok\nDetails:\n- a\n- b")
    calls: list[str] = []

    def fake_render(name: str, **_kwargs):
        calls.append(name)
        return f"PROMPT::{name}"

    monkeypatch.setattr(chat_module, "render_prompt", fake_render)

    result = service.respond(
        deposition={"_id": "dep:1", "witness_name": "Jane", "contradictions": []},
        peers=[],
        user_message="Summarize",
        history=[],
    )

    assert result.startswith("Short answer:")
    assert calls == ["chat_system", "chat_user_context"]
    messages = service.llm.invoke.call_args.args[0]
    assert str(messages[0].content) == "PROMPT::chat_system"
    assert str(messages[1].content) == "PROMPT::chat_user_context"


def test_reason_about_contradiction_uses_llm_when_available():
    service = build_service_with_mock_llm()
    service.llm.invoke.return_value = SimpleNamespace(content="Short answer: strong\nDetails:\n- a\n- b\n- c")

    result = service.reason_about_contradiction(
        deposition={"_id": "dep:1"},
        peers=[],
        contradiction={"topic": "Timeline", "severity": 70},
    )

    assert result.startswith("Short answer:")
    service.llm.invoke.assert_called_once()


def test_reason_about_contradiction_raises_with_fix_on_error():
    service = build_service_with_mock_llm()
    service.llm.invoke.side_effect = RuntimeError("quota")

    with pytest.raises(RuntimeError, match="Possible fix:"):
        service.reason_about_contradiction(
            deposition={"_id": "dep:1"},
            peers=[],
            contradiction={
                "topic": "Timeline",
                "severity": 20,
                "other_witness_name": "Alan",
                "rationale": "Mismatch",
            },
        )


def test_reason_about_contradiction_uses_selected_llm_override(monkeypatch):
    service = build_service_with_mock_llm()
    service.settings = SimpleNamespace()
    selected_llm = Mock()
    selected_llm.invoke.return_value = SimpleNamespace(content="Short answer: focused\nDetails:\n- a\n- b\n- c")
    build = Mock(return_value=selected_llm)
    monkeypatch.setattr(chat_module, "build_chat_model", build)

    result = service.reason_about_contradiction(
        deposition={"_id": "dep:1"},
        peers=[],
        contradiction={"topic": "Timeline", "severity": 70},
        llm_provider="ollama",
        llm_model="llama3.3",
    )

    assert result.startswith("Short answer:")
    build.assert_called_once()
    selected_llm.invoke.assert_called_once()


def test_reason_about_contradiction_uses_prompt_templates(monkeypatch):
    service = build_service_with_mock_llm()
    service.llm.invoke.return_value = SimpleNamespace(content="Short answer: ok\nDetails:\n- a\n- b\n- c")
    calls: list[str] = []

    def fake_render(name: str, **_kwargs):
        calls.append(name)
        return f"PROMPT::{name}"

    monkeypatch.setattr(chat_module, "render_prompt", fake_render)

    result = service.reason_about_contradiction(
        deposition={"_id": "dep:1"},
        peers=[],
        contradiction={"topic": "Timeline", "severity": 70},
    )

    assert result.startswith("Short answer:")
    assert calls == ["reason_contradiction_system", "reason_contradiction_user"]
    messages = service.llm.invoke.call_args.args[0]
    assert str(messages[0].content) == "PROMPT::reason_contradiction_system"
    assert str(messages[1].content) == "PROMPT::reason_contradiction_user"


def test_reason_about_contradiction_normalizes_json_output():
    service = build_service_with_mock_llm()
    service.llm.invoke.return_value = SimpleNamespace(
        content=json.dumps(
            {
                "short_answer": "The timeline conflict is material and likely impeachment-worthy.",
                "strength": "strong",
                "conflict_focus": "Arrival time mismatch before incident.",
                "target_evidence": "Target says 8:45 a.m. arrival.",
                "peer_evidence": "Peer says notice at 9:15 a.m. after impact.",
                "insights": ["This discrepancy affects credibility.", "It may alter incident causation framing."],
                "next_steps": ["Lock exact times.", "Confront with logs.", "Tie to exhibits."],
            }
        )
    )

    result = service.reason_about_contradiction(
        deposition={
            "_id": "dep:1",
            "summary": "Target summary",
            "claims": [{"statement": "I arrived at 8:45 a.m."}],
        },
        peers=[
            {
                "_id": "dep:2",
                "witness_name": "Alan",
                "summary": "Peer summary",
                "claims": [{"statement": "I was told at 9:15 a.m."}],
            }
        ],
        contradiction={"topic": "Timeline", "severity": 80, "other_witness_name": "Alan", "rationale": "Time conflict"},
    )

    assert result.startswith("Short answer:")
    assert "Strength: Strong" in result
    assert "Evidence comparison:" in result
    assert "Recommended next steps:" in result


def test_reason_about_contradiction_replaces_placeholder_output():
    service = build_service_with_mock_llm()
    service.llm.invoke.return_value = SimpleNamespace(
        content="Short answer: <1 sentence>\nDetails:\n- <bullet>\n- <bullet>\n- <bullet>"
    )

    result = service.reason_about_contradiction(
        deposition={"_id": "dep:1", "summary": "Target summary", "claims": []},
        peers=[],
        contradiction={"topic": "Camera", "severity": 50, "rationale": "Opposite statements"},
    )

    assert "<1 sentence>" not in result
    assert "<bullet>" not in result
    assert "Strength:" in result
    assert "Conflict focus:" in result


def test_normalize_reasoning_output_handles_single_bullet():
    service = build_service_with_mock_llm()
    context = service._build_context_payload(
        deposition={"_id": "dep:1", "summary": "Target summary", "claims": []},
        peers=[],
    )

    result = service._normalize_reasoning_output(
        "Short answer: Focused concern.\n- One insight only.",
        {"topic": "Timeline", "severity": 45},
        context,
    )

    assert result.startswith("Short answer: Focused concern.")
    assert "Why this matters:" in result


def test_normalize_reasoning_output_uses_default_insights_when_no_bullets():
    service = build_service_with_mock_llm()
    context = service._build_context_payload(
        deposition={"_id": "dep:1", "summary": "Target summary", "claims": []},
        peers=[],
    )

    result = service._normalize_reasoning_output(
        "Short answer: Narrative without bullets.",
        {"topic": "Timeline", "severity": 45},
        context,
    )

    assert result.startswith("Short answer: Narrative without bullets.")
    assert "- The conflict goes to witness credibility" in result


def test_parse_json_payload_supports_fenced_and_empty():
    service = build_service_with_mock_llm()

    assert service._parse_json_payload("   ") is None
    parsed = service._parse_json_payload("```json\n{\"short_answer\":\"ok\"}\n```")
    assert parsed == {"short_answer": "ok"}


def test_extract_short_answer_falls_back_to_first_line_and_empty():
    service = build_service_with_mock_llm()

    assert service._extract_short_answer("Line one\nLine two") == "Line one"
    assert service._extract_short_answer("   \n  ") == ""


def test_extract_bullets_skips_blank_lines():
    service = build_service_with_mock_llm()
    bullets = service._extract_bullets("\n- a\n\n* b\nplain")
    assert bullets == ["a", "b"]


def test_default_reasoning_sections_uses_weak_strength_and_summary_fallback():
    service = build_service_with_mock_llm()
    context = {
        "target_deposition": {"summary": "Target summary", "claims": []},
        "peer_depositions": [{"witness_name": "Peer", "summary": "Peer summary", "claims": []}],
    }

    sections = service._default_reasoning_sections(
        {"topic": "Timeline", "severity": 10, "rationale": "Mismatch"},
        context,
    )

    assert sections["strength"] == "weak"
    assert sections["target_evidence"] == "Target summary"
    assert sections["peer_evidence"] == "Peer summary"


def test_pick_peer_for_contradiction_prefers_match_or_first():
    service = build_service_with_mock_llm()
    context = {
        "peer_depositions": [
            {"witness_name": "First"},
            {"witness_name": "Alan"},
        ]
    }
    matched = service._pick_peer_for_contradiction(context, {"other_witness_name": "Alan"})
    fallback = service._pick_peer_for_contradiction(context, {"other_witness_name": "Unknown"})
    empty = service._pick_peer_for_contradiction({"peer_depositions": []}, {"other_witness_name": "Any"})

    assert matched["witness_name"] == "Alan"
    assert fallback["witness_name"] == "First"
    assert empty == {}


def test_first_claim_statement_reads_first_item_or_empty():
    service = build_service_with_mock_llm()
    assert service._first_claim_statement({"claims": [{"statement": "First claim"}]}) == "First claim"
    assert service._first_claim_statement({"claims": []}) == ""


def test_merge_json_reasoning_handles_string_and_list_fields():
    service = build_service_with_mock_llm()
    defaults = service._default_reasoning_sections({"topic": "T", "severity": 80}, {"target_deposition": {}, "peer_depositions": []})

    merged = service._merge_json_reasoning(
        {
            "answer": "Concise answer",
            "strength": "moderate",
            "focus": "Focus line",
            "target_evidence": "Target fact",
            "peer_evidence": "Peer fact",
            "insights": "One insight",
            "recommended_actions": "One action",
        },
        defaults,
    )

    assert merged["short_answer"] == "Concise answer"
    assert merged["strength"] == "moderate"
    assert merged["conflict_focus"] == "Focus line"
    assert merged["insights"] == ["One insight"]
    assert merged["next_steps"] == ["One action"]


def test_render_reasoning_sections_fills_missing_insights_and_steps():
    service = build_service_with_mock_llm()
    rendered = service._render_reasoning_sections(
        {
            "short_answer": "Answer",
            "strength": "weak",
            "conflict_focus": "Focus",
            "target_evidence": "Target",
            "peer_evidence": "Peer",
            "insights": [],
            "next_steps": [],
        }
    )

    assert "Why this matters:" in rendered
    assert "Recommended next steps:" in rendered
    assert "1. Run targeted follow-up questioning" in rendered


def test_fallback_chat_response_without_contradictions():
    service = build_service_with_mock_llm()
    result = service._fallback_chat_response(
        deposition={"witness_name": "Jane", "contradiction_score": 0, "contradictions": []},
        user_message="",
    )

    assert "low-risk" in result


def test_fallback_chat_response_with_contradictions_and_user_focus():
    service = build_service_with_mock_llm()
    result = service._fallback_chat_response(
        deposition={
            "witness_name": "Jane",
            "contradiction_score": 66,
            "contradictions": [
                {
                    "topic": "Timeline",
                    "other_witness_name": "Alan",
                    "rationale": "Mismatch",
                }
            ],
        },
        user_message="check timeline conflict",
    )

    assert "risk at 66/100" in result
    assert "Primary conflict is on Timeline" in result
    assert "Requested focus: check timeline conflict" in result


def test_merge_json_chat_accepts_string_detail_and_preserves_defaults():
    service = build_service_with_mock_llm()
    defaults = service._default_chat_sections(
        deposition={"witness_name": "Jane", "contradiction_score": 0, "contradictions": []},
        user_message="",
    )

    merged = service._merge_json_chat(
        {
            "response": "Concise answer",
            "analysis": "One concrete detail.",
        },
        defaults,
    )

    assert merged["short_answer"] == "Concise answer"
    assert merged["details"] == ["One concrete detail."]


def test_fallback_contradiction_reasoning_strong_threshold():
    service = build_service_with_mock_llm()
    result = service._fallback_contradiction_reasoning(
        {
            "topic": "Camera",
            "severity": 80,
            "other_witness_name": "Mark",
            "rationale": "Opposite statements",
        }
    )

    assert "appears strong" in result
    assert "Camera" in result


def test_fallback_contradiction_reasoning_moderate_threshold():
    service = build_service_with_mock_llm()
    result = service._fallback_contradiction_reasoning({"topic": "Timeline", "severity": 50})

    assert "appears moderate" in result


def test_fallback_contradiction_reasoning_weak_threshold():
    service = build_service_with_mock_llm()
    result = service._fallback_contradiction_reasoning({"topic": "Timeline", "severity": 10})

    assert "appears weak" in result


def test_preview_text_truncates_when_limit_exceeded():
    service = build_service_with_mock_llm()
    preview = service._preview_text("x" * 32, 10)
    assert preview == ("x" * 10) + "...(truncated)"
