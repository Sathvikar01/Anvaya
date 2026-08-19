from __future__ import annotations

import pytest

from hh_goa_rag.guardrails.types import GuardrailResponse, ReasonCode, Route

pytest.importorskip("langgraph")

from hh_goa_rag.orchestration import build_langgraph  # noqa: E402


def test_langgraph_invokes_the_frozen_pipeline_node() -> None:
    calls: list[object] = []

    def runner(transcript: object, **kwargs: object) -> GuardrailResponse:
        calls.append((transcript, kwargs))
        return GuardrailResponse(
            route=Route.INSUFFICIENT_CONTEXT,
            reason_code=ReasonCode.RETRIEVAL_EMPTY,
        )

    graph = build_langgraph(runner)
    response = graph.invoke(
        {
            "transcript": "manhattan project",
            "operation_started": 123,
            "on_stage": None,
        }
    )["response"]

    assert response.route == Route.INSUFFICIENT_CONTEXT
    assert calls[0][0] == "manhattan project"
    assert calls[0][1]["operation_started"] == 123
