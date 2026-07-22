"""Optional LangGraph orchestration for the frozen pipeline.

The native harness remains the default. This module only adds orchestration and
observability around the existing deterministic pipeline; it does not change
retrieval, generation, or grounding policy.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypedDict

from hh_goa_rag.guardrails.types import GuardrailResponse


class GraphState(TypedDict, total=False):
    transcript: object
    operation_started: int
    stages: dict[str, float] | None
    base_metadata: dict[str, Any] | None
    on_stage: Callable[[str], None] | None
    response: GuardrailResponse


def build_langgraph(runner: Callable[..., GuardrailResponse]) -> Any:
    """Compile a graph that delegates to the unchanged frozen harness pipeline."""
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as error:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "LangGraph orchestration requested but langgraph is not installed; "
            "install the 'orchestration' extra"
        ) from error

    def run_frozen_pipeline(state: GraphState) -> dict[str, GuardrailResponse]:
        return {
            "response": runner(
                state["transcript"],
                operation_started=state["operation_started"],
                stages=state.get("stages"),
                base_metadata=state.get("base_metadata"),
                on_stage=state.get("on_stage"),
            )
        }

    builder = StateGraph(GraphState)
    builder.add_node("frozen_voice_rag", run_frozen_pipeline)
    builder.add_edge(START, "frozen_voice_rag")
    builder.add_edge("frozen_voice_rag", END)
    return builder.compile()
