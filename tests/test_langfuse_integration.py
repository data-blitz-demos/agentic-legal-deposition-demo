from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

from backend.app import langfuse_integration as lf


def make_settings(**overrides):
    base = {
        "langfuse_enabled": True,
        "langfuse_public_key": "pk-test",
        "langfuse_secret_key": "sk-test",
        "langfuse_base_url": "http://langfuse:3000",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def reset_langfuse_state():
    lf._LANGFUSE_INITIALIZED = False


def test_langfuse_sdk_installed_false_when_any_dependency_missing(monkeypatch):
    reset_langfuse_state()
    monkeypatch.setattr(lf, "Langfuse", object())
    monkeypatch.setattr(lf, "CallbackHandler", None)
    monkeypatch.setattr(lf, "get_client", object())
    monkeypatch.setattr(lf, "propagate_attributes", object())

    assert lf.langfuse_sdk_installed() is False


def test_langfuse_enabled_requires_flag_sdk_and_credentials(monkeypatch):
    reset_langfuse_state()
    monkeypatch.setattr(lf, "Langfuse", object())
    monkeypatch.setattr(lf, "CallbackHandler", object())
    monkeypatch.setattr(lf, "get_client", object())
    monkeypatch.setattr(lf, "propagate_attributes", object())

    assert lf.langfuse_enabled(make_settings()) is True
    assert lf.langfuse_enabled(make_settings(langfuse_enabled=False)) is False
    assert lf.langfuse_enabled(make_settings(langfuse_public_key="")) is False


def test_initialize_langfuse_returns_true_when_already_initialized():
    lf._LANGFUSE_INITIALIZED = True

    assert lf.initialize_langfuse(make_settings()) is True

    reset_langfuse_state()


def test_initialize_langfuse_returns_false_when_not_enabled(monkeypatch):
    reset_langfuse_state()
    monkeypatch.setattr(lf, "langfuse_enabled", lambda _settings: False)

    assert lf.initialize_langfuse(make_settings()) is False


def test_initialize_langfuse_initializes_client(monkeypatch):
    reset_langfuse_state()
    init = Mock()
    monkeypatch.setattr(lf, "Langfuse", init)
    monkeypatch.setattr(lf, "langfuse_enabled", lambda _settings: True)

    assert lf.initialize_langfuse(make_settings()) is True
    init.assert_called_once_with(public_key="pk-test", secret_key="sk-test", base_url="http://langfuse:3000")


def test_initialize_langfuse_logs_and_returns_false_on_error(monkeypatch):
    reset_langfuse_state()
    monkeypatch.setattr(lf, "Langfuse", Mock(side_effect=RuntimeError("boom")))
    monkeypatch.setattr(lf, "langfuse_enabled", lambda _settings: True)
    logged = Mock()
    monkeypatch.setattr(lf.logger, "exception", logged)

    assert lf.initialize_langfuse(make_settings()) is False
    logged.assert_called_once_with("langfuse initialization failed")


def test_shutdown_langfuse_noops_when_not_initialized(monkeypatch):
    reset_langfuse_state()
    client = Mock()
    monkeypatch.setattr(lf, "get_client", Mock(return_value=client))

    lf.shutdown_langfuse(make_settings())

    client.shutdown.assert_not_called()


def test_shutdown_langfuse_logs_when_shutdown_fails(monkeypatch):
    lf._LANGFUSE_INITIALIZED = True
    client = Mock()
    client.shutdown.side_effect = RuntimeError("boom")
    monkeypatch.setattr(lf, "get_client", Mock(return_value=client))
    monkeypatch.setattr(lf, "langfuse_enabled", lambda _settings: True)
    logged = Mock()
    monkeypatch.setattr(lf.logger, "exception", logged)

    lf.shutdown_langfuse(make_settings())

    logged.assert_called_once_with("langfuse shutdown failed")
    reset_langfuse_state()


def test_trim_string_and_normalizers():
    assert lf._trim_string("  abc  ", limit=2) == "ab"
    assert lf._trim_string("   ") is None
    assert lf._normalize_tags([" a ", "", "a", "b"]) == ["a", "b"]
    assert lf._normalize_tags(None) is None
    assert lf._normalize_metadata(None) is None
    assert lf._normalize_metadata({"a": "  x  ", "b": "", "c": None, "d": 3}) == {"a": "x", "d": 3}


def test_observe_operation_yields_none_when_initialization_disabled(monkeypatch):
    monkeypatch.setattr(lf, "initialize_langfuse", lambda _settings: False)

    with lf.observe_operation(make_settings(), "op") as observation:
        assert observation is None


def test_observe_operation_uses_propagation_and_observation_contexts(monkeypatch):
    monkeypatch.setattr(lf, "initialize_langfuse", lambda _settings: True)
    propagation_context = nullcontext()
    observation_context = nullcontext("obs")
    propagate = Mock(return_value=propagation_context)
    client = Mock()
    client.start_as_current_observation.return_value = observation_context
    monkeypatch.setattr(lf, "propagate_attributes", propagate)
    monkeypatch.setattr(lf, "get_client", Mock(return_value=client))

    with lf.observe_operation(
        make_settings(),
        "op",
        session_id=" s1 ",
        user_id=" u1 ",
        tags=["a", "a", "b"],
        metadata={"x": " y ", "n": 2, "skip": None},
    ) as observation:
        assert observation == "obs"

    propagate.assert_called_once_with(
        session_id="s1",
        user_id="u1",
        tags=["a", "b"],
        metadata={"x": "y", "n": 2},
    )
    client.start_as_current_observation.assert_called_once_with(as_type="span", name="op")


def test_observe_operation_logs_and_yields_none_on_context_failure(monkeypatch):
    monkeypatch.setattr(lf, "initialize_langfuse", lambda _settings: True)
    monkeypatch.setattr(lf, "propagate_attributes", Mock(side_effect=RuntimeError("boom")))
    logged = Mock()
    monkeypatch.setattr(lf.logger, "exception", logged)

    with lf.observe_operation(make_settings(), "op", session_id="abc") as observation:
        assert observation is None

    logged.assert_called_once_with("langfuse observation failed name=%s", "op")


def test_build_langchain_config_returns_empty_when_disabled(monkeypatch):
    monkeypatch.setattr(lf, "initialize_langfuse", lambda _settings: False)

    assert lf.build_langchain_config(make_settings(), operation="chat") == {}


def test_build_langchain_config_includes_callback_and_metadata(monkeypatch):
    monkeypatch.setattr(lf, "initialize_langfuse", lambda _settings: True)
    callback = object()
    monkeypatch.setattr(lf, "CallbackHandler", Mock(return_value=callback))

    config = lf.build_langchain_config(
        make_settings(),
        operation="chat",
        session_id=" case-1 ",
        user_id=" user-1 ",
        tags=["attorneyos", "attorneyos", "ollama"],
        metadata={"a": " x ", "blank": "", "num": 2},
    )

    assert config == {
        "callbacks": [callback],
        "metadata": {
            "a": "x",
            "num": 2,
            "langfuse_session_id": "case-1",
            "langfuse_user_id": "user-1",
            "langfuse_tags": ["attorneyos", "ollama"],
            "operation": "chat",
        },
    }


def test_build_langchain_config_sets_operation_when_optional_fields_missing(monkeypatch):
    monkeypatch.setattr(lf, "initialize_langfuse", lambda _settings: True)
    monkeypatch.setattr(lf, "CallbackHandler", Mock(return_value="cb"))

    config = lf.build_langchain_config(make_settings(), operation="ingest", metadata={})

    assert config["metadata"] == {"operation": "ingest"}


def test_prompt_and_response_heuristics_cover_structure_and_placeholders():
    assert lf._word_count("One two three") == 3
    assert lf._contains_placeholder_text("Short answer: <1 sentence>") is True
    assert lf._response_has_structure("Short answer: Yes\nDetails:\n- A") is True
    assert lf._content_terms("What is the witness timeline for contract breach?") >= {"witness", "timeline", "contract", "breach"}
    assert lf._clamp01(2) == 1.0
    assert lf._to_rubric_5(0.5) == 3.0
    assert lf._overall_quality_label(0.1) == "poor"
    assert lf._overall_quality_label(0.9) == "excellent"
    assert lf._prompt_quality_heuristic("   ") == 0.0
    assert lf._prompt_clarity_heuristic("   ") == 0.0
    assert lf._prompt_specificity_heuristic("   ") == 0.0
    assert lf._response_quality_heuristic("   ") == 0.0
    assert lf._response_relevance_heuristic("", "timeline mismatch found") == 0.0
    assert lf._response_completeness_heuristic("   ") == 0.0
    assert lf._response_relevance_heuristic("compare timeline", "timeline mismatch found") > 0.0
    assert lf._response_completeness_heuristic("Short answer: Yes\nDetails:\n- A\n- B") > 0.5
    assert lf._response_helpfulness_heuristic("compare timeline", "Short answer: review the timeline.\nRecommended action:\n- Compare exhibits.") > 0.5
    assert lf._response_actionable("Recommended action:\n- Compare exhibits.") is True
    assert lf._prompt_quality_heuristic("Compare the witness timeline against the contract notice?") > 0.5
    assert lf._response_quality_heuristic("Short answer: Yes\nDetails:\n- A\n- B") > 0.6


def test_score_current_trace_returns_false_when_disabled(monkeypatch):
    monkeypatch.setattr(lf, "initialize_langfuse", lambda _settings: False)

    assert lf.score_current_trace(make_settings(), name="score", value=1.0) is False


def test_score_current_trace_logs_and_returns_false_on_sdk_error(monkeypatch):
    monkeypatch.setattr(lf, "initialize_langfuse", lambda _settings: True)
    client = SimpleNamespace(
        score_current_trace=Mock(side_effect=RuntimeError("boom")),
        score_current_span=Mock(side_effect=RuntimeError("boom")),
    )
    monkeypatch.setattr(lf, "get_client", Mock(return_value=client))
    logged = Mock()
    monkeypatch.setattr(lf.logger, "exception", logged)

    assert lf.score_current_trace(make_settings(), name="quality", value=0.8, data_type="NUMERIC") is False
    assert len(logged.call_args_list) == 2
    assert logged.call_args_list[0].args == ("langfuse score_current_trace failed name=%s", "quality")
    assert logged.call_args_list[1].args == ("langfuse score_current_span failed name=%s", "quality")


def test_score_current_trace_handles_missing_method_and_success(monkeypatch):
    monkeypatch.setattr(lf, "initialize_langfuse", lambda _settings: True)
    monkeypatch.setattr(lf, "get_client", Mock(return_value=SimpleNamespace()))
    assert lf.score_current_trace(make_settings(), name="quality", value=0.8) is False

    trace_scorer = Mock()
    span_scorer = Mock()
    monkeypatch.setattr(
        lf,
        "get_client",
        Mock(return_value=SimpleNamespace(score_current_trace=trace_scorer, score_current_span=span_scorer)),
    )
    assert lf.score_current_trace(
        make_settings(),
        name="quality",
        value=0.8,
        data_type="NUMERIC",
        comment="helpful comment",
    ) is True
    trace_scorer.assert_called_once_with(
        name="quality",
        value=0.8,
        data_type="NUMERIC",
        comment="helpful comment",
    )
    span_scorer.assert_called_once_with(
        name="quality",
        value=0.8,
        data_type="NUMERIC",
        comment="helpful comment",
    )


def test_score_current_trace_succeeds_when_only_span_scorer_is_available(monkeypatch):
    monkeypatch.setattr(lf, "initialize_langfuse", lambda _settings: True)
    span_scorer = Mock()
    monkeypatch.setattr(lf, "get_client", Mock(return_value=SimpleNamespace(score_current_span=span_scorer)))

    assert lf.score_current_trace(make_settings(), name="quality", value=0.8, data_type="NUMERIC") is True
    span_scorer.assert_called_once_with(
        name="quality",
        value=0.8,
        data_type="NUMERIC",
    )


def test_score_user_prompt_and_response_emits_expected_scores(monkeypatch):
    captured: list[dict] = []
    monkeypatch.setattr(
        lf,
        "score_current_trace",
        lambda _settings, **kwargs: captured.append(kwargs) or True,
    )

    metrics = lf.score_user_prompt_and_response(
        make_settings(),
        operation="chat",
        prompt_text="Compare the witness timeline?",
        response_text="Short answer: Yes\nDetails:\n- Timeline mismatch.\nRecommended action:\n- Review exhibits.",
    )

    assert metrics["user_prompt_word_count"] == 4.0
    assert metrics["response_has_structure"] == 1.0
    assert metrics["response_placeholder_free"] == 1.0
    assert metrics["response_actionable"] == 1.0
    assert metrics["overall_response_quality"] >= 3.0
    assert metrics["overall_quality_rubric"] >= 3.0
    assert [item["name"] for item in captured] == [
        "user_prompt_word_count",
        "user_prompt_quality_heuristic",
        "prompt_clarity_rubric",
        "prompt_specificity_rubric",
        "response_word_count",
        "response_quality_heuristic",
        "response_relevance_rubric",
        "response_helpfulness_rubric",
        "response_completeness_rubric",
        "response_has_structure",
        "response_placeholder_free",
        "response_actionable",
        "overall_response_quality",
        "overall_response_quality_label",
        "overall_quality_rubric",
        "overall_quality_label",
    ]


def test_score_observable_dashboard_emits_summary_metric_values_and_statuses(monkeypatch):
    captured: list[dict] = []
    monkeypatch.setattr(
        lf,
        "score_current_trace",
        lambda _settings, **kwargs: captured.append(kwargs) or True,
    )

    emitted = lf.score_observable_dashboard(
        make_settings(),
        operation="agent_metrics",
        metric_groups={
            "runtime": [
                SimpleNamespace(key="task_success_rate_pct", value=95.5, display="95.5%", status="good"),
                SimpleNamespace(key="golden_set_accuracy", value=None, display="N/A", status="info"),
                SimpleNamespace(key="", value=1.0, display="skip", status="info"),
            ],
            "mcp_tool": [
                {"key": "mcp_neo4j_query_samples", "value": 3.0, "display": "3", "status": "info"},
                {"key": "mcp_boolean_flag", "value": True, "display": "true", "status": "good"},
            ],
        },
        summary={
            "sampled_runs": 12,
            "storage_connected": True,
            "label": "ready",
        },
    )

    assert emitted == len(captured)
    assert any(item["name"] == "observables.summary.sampled_runs" and item["value"] == 12.0 for item in captured)
    assert any(item["name"] == "observables.summary.storage_connected" and item["value"] == 1.0 for item in captured)
    assert any(item["name"] == "observables.summary.label" and item["value"] == "ready" for item in captured)
    assert any(item["name"] == "observables.runtime.task_success_rate_pct" and item["value"] == 95.5 for item in captured)
    assert any(item["name"] == "observables.runtime.task_success_rate_pct_status" and item["value"] == "good" for item in captured)
    assert any(item["name"] == "observables.runtime.golden_set_accuracy_display" and item["value"] == "N/A" for item in captured)
    assert any(item["name"] == "observables.mcp.mcp_neo4j_query_samples" and item["value"] == 3.0 for item in captured)
    assert any(item["name"] == "observables.mcp.mcp_boolean_flag" and item["value"] == 1.0 for item in captured)


def test_normalize_observable_group_name_maps_mcp_aliases():
    assert lf._normalize_observable_group_name("mcp_tool") == "mcp"
    assert lf._normalize_observable_group_name("mcp tools") == "mcp"
    assert lf._normalize_observable_group_name("input_context") == "input_context"
