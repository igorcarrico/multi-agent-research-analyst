"""LangGraph workflow assembly + conditional routing.

Topology:
    research → analyst → writer → critic → evaluator → [route]
        route ─ pass ─→ END
        route ─ revise ─→ reviewer → critic   (loop)
"""
from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from agents import (
    AnalystAgent,
    CriticAgent,
    ResearchAgent,
    ReviewerAgent,
    WriterAgent,
)
from evaluation import Evaluator
from utils import AppConfig, StepLogger

from .state import ResearchState


def _route_after_eval(state: ResearchState) -> Literal["revise", "pass"]:
    """Decide whether to loop back to the reviewer or finish."""
    score = state.get("weighted_score", 0.0)
    threshold = state.get("quality_threshold", 8.0)
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 2)

    if score >= threshold:
        return "pass"
    if iteration >= max_iter:
        return "pass"
    return "revise"


def _finalize(state: ResearchState) -> dict:
    """Promote the latest report to final_report. Pure function, no LLM."""
    return {"final_report": state.get("report", "")}


def build_workflow(cfg: AppConfig, logger: StepLogger):
    """Assemble the StateGraph. Returns a compiled runnable."""

    research = ResearchAgent(cfg, logger)
    analyst = AnalystAgent(cfg, logger)
    writer = WriterAgent(cfg, logger)
    critic = CriticAgent(cfg, logger)
    reviewer = ReviewerAgent(cfg, logger)
    evaluator = Evaluator(cfg, logger)

    g = StateGraph(ResearchState)

    g.add_node("research", research.run)
    g.add_node("analyst", analyst.run)
    g.add_node("writer", writer.run)
    g.add_node("critic", critic.run)
    g.add_node("evaluator", evaluator.run)
    g.add_node("reviewer", reviewer.run)
    g.add_node("finalize", _finalize)

    g.add_edge(START, "research")
    g.add_edge("research", "analyst")
    g.add_edge("analyst", "writer")
    g.add_edge("writer", "critic")
    g.add_edge("critic", "evaluator")

    g.add_conditional_edges(
        "evaluator",
        _route_after_eval,
        {"revise": "reviewer", "pass": "finalize"},
    )

    # Reviewer rewrites → re-critique → re-evaluate. Loop guarded by iteration counter.
    g.add_edge("reviewer", "critic")
    g.add_edge("finalize", END)

    return g.compile()


def run_workflow(
    cfg: AppConfig,
    logger: StepLogger,
    query: str,
    context: str = "",
    language: str = "Portuguese",
) -> ResearchState:
    """Convenience wrapper. Builds, invokes, returns final state."""
    from .state import initial_state

    app = build_workflow(cfg, logger)
    state = initial_state(
        query=query,
        context=context,
        language=language,
        max_iterations=cfg.workflow.max_iterations,
        quality_threshold=cfg.workflow.quality_threshold,
    )
    # recursion_limit is a safety net on top of our own iteration counter.
    final: ResearchState = app.invoke(state, config={"recursion_limit": 25})
    final["logs"] = [e.to_dict() for e in logger.events]
    return final
