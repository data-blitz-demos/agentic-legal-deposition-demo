"""Starter DeepEval suite for AttorneyOS workflows.

Run locally with:
    PYTEST_ADDOPTS="-p no:unraisableexception --assert=plain" deepeval test run evals/attorneyos_deepeval.py

Required environment for the chat eval:
    DEEPEVAL_CASE_ID
    DEEPEVAL_DEPOSITION_ID

Optional environment for the graph RAG eval:
    DEEPEVAL_ENABLE_GRAPH_RAG=1
    DEEPEVAL_GRAPH_QUESTION="What contradictions matter most?"

Judge-model configuration is handled by DeepEval itself. See:
    https://deepeval.com/docs/getting-started
"""

from __future__ import annotations

import asyncio
import os

import httpx
import pytest

deepeval = pytest.importorskip("deepeval")

assert_test = deepeval.assert_test
AnswerRelevancyMetric = pytest.importorskip("deepeval.metrics").AnswerRelevancyMetric
LLMTestCase = pytest.importorskip("deepeval.test_case").LLMTestCase

API_BASE_URL = str(os.getenv("DEEPEVAL_TARGET_URL", "http://localhost:8000") or "http://localhost:8000").rstrip("/")
CASE_ID = str(os.getenv("DEEPEVAL_CASE_ID", "") or "").strip()
DEPOSITION_ID = str(os.getenv("DEEPEVAL_DEPOSITION_ID", "") or "").strip()
GRAPH_RAG_ENABLED = str(os.getenv("DEEPEVAL_ENABLE_GRAPH_RAG", "") or "").strip() == "1"
GRAPH_QUESTION = str(
    os.getenv("DEEPEVAL_GRAPH_QUESTION", "What are the main contradiction themes in this case?") or ""
).strip()


def _ensure_event_loop() -> asyncio.AbstractEventLoop:
    """Create a current event loop so DeepEval stays compatible with Python 3.12."""

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def _post_json(path: str, payload: dict) -> dict:
    """Submit one JSON request to the running AttorneyOS API."""

    response = httpx.post(f"{API_BASE_URL}{path}", json=payload, timeout=90.0)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise AssertionError(f"Expected JSON object from {path}, received: {type(data)!r}")
    return data


@pytest.mark.skipif(
    not CASE_ID or not DEPOSITION_ID,
    reason="Set DEEPEVAL_CASE_ID and DEEPEVAL_DEPOSITION_ID to run the Attorney chat DeepEval suite.",
)
def test_attorney_chat_answer_relevancy():
    """Check that attorney-chat answers stay relevant to the user prompt."""

    _ensure_event_loop()
    prompt = "Summarize the main contradiction themes from this deposition in the usual attorney format."
    payload = _post_json(
        "/api/chat",
        {
            "case_id": CASE_ID,
            "deposition_id": DEPOSITION_ID,
            "message": prompt,
        },
    )
    answer = str(payload.get("response") or "").strip()
    assert answer, "Attorney chat returned an empty response."
    assert_test(
        LLMTestCase(input=prompt, actual_output=answer),
        [AnswerRelevancyMetric(threshold=0.7)],
    )


@pytest.mark.skipif(
    not GRAPH_RAG_ENABLED,
    reason="Set DEEPEVAL_ENABLE_GRAPH_RAG=1 to run the Graph RAG DeepEval suite.",
)
def test_graph_rag_answer_relevancy():
    """Check that Graph RAG answers remain relevant to the input question."""

    _ensure_event_loop()
    payload = _post_json(
        "/api/graph-rag/query",
        {
            "question": GRAPH_QUESTION,
            "use_rag": True,
            "top_k": 5,
        },
    )
    answer = str(payload.get("answer") or "").strip()
    assert answer, "Graph RAG returned an empty answer."
    assert_test(
        LLMTestCase(input=GRAPH_QUESTION, actual_output=answer),
        [AnswerRelevancyMetric(threshold=0.7)],
    )
