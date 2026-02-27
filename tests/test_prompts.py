from __future__ import annotations

from pathlib import Path

import pytest

from backend.app import prompts as prompts_module


def test_load_prompt_success_reads_template_file():
    prompts_module.load_prompt.cache_clear()

    content = prompts_module.load_prompt("chat_system")

    assert "You are a seasoned attorney" in content


def test_load_prompt_unknown_name_raises_value_error():
    prompts_module.load_prompt.cache_clear()

    with pytest.raises(ValueError, match="Unknown prompt name"):
        prompts_module.load_prompt("does_not_exist")


def test_load_prompt_missing_file_raises(monkeypatch):
    prompts_module.load_prompt.cache_clear()
    monkeypatch.setitem(prompts_module.PROMPT_FILES, "fake_prompt", "missing_prompt.txt")

    with pytest.raises(FileNotFoundError, match="Prompt file not found"):
        prompts_module.load_prompt("fake_prompt")



def test_render_prompt_success_substitutes_variables():
    prompts_module.load_prompt.cache_clear()

    rendered = prompts_module.render_prompt(
        "map_deposition_user",
        case_id="case-1",
        file_name="a.txt",
        schema_json='{"title":"DepositionSchema"}',
        raw_text="hello",
    )

    assert "Case ID: case-1" in rendered
    assert "File Name: a.txt" in rendered
    assert rendered.endswith("hello")



def test_render_prompt_missing_variable_raises_value_error():
    prompts_module.load_prompt.cache_clear()

    with pytest.raises(ValueError, match="Missing prompt variable"):
        prompts_module.render_prompt("map_deposition_user", case_id="c")


def test_render_graph_rag_prompt_templates():
    prompts_module.load_prompt.cache_clear()

    system_prompt = prompts_module.load_prompt("graph_rag_system")
    user_prompt = prompts_module.render_prompt(
        "graph_rag_user",
        question="What is contract?",
        context_text="Resource: Contract",
    )

    assert "retrieval-augmented mode" in system_prompt
    assert "Question:" in user_prompt
    assert "Resource: Contract" in user_prompt


def test_prompt_dir_points_to_backend_prompts_folder():
    prompt_dir = prompts_module._prompt_dir()

    assert prompt_dir.name == "prompts"
    assert prompt_dir == Path(__file__).resolve().parents[1] / "backend/prompts"
