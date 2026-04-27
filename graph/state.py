"""Graph state. A single TypedDict shared across all nodes.

Why TypedDict and not Pydantic: LangGraph merges partial dicts returned by each
node into the running state. TypedDict is the idiomatic choice and keeps the
hot path allocation-free.
"""
from __future__ import annotations

from typing import Any, TypedDict


class ResearchState(TypedDict, total=False):
    # Input
    query: str
    context: str                         # optional extra context from the user
    language: str                        # output language (e.g. "Portuguese", "English")

    # Pipeline outputs
    research: str
    sources: list[str]
    analysis: str
    report: str
    critique: str

    # Evaluation
    scores: dict[str, float]             # per-axis scores, 0–10
    weighted_score: float                # aggregate, 0–10

    # Loop control
    iteration: int                       # number of reviewer passes done
    max_iterations: int
    quality_threshold: float

    # Final
    final_report: str

    # Observability — list of LogEvent dicts (kept JSON-serializable)
    logs: list[dict[str, Any]]


def initial_state(
    query: str,
    *,
    context: str = "",
    language: str = "Portuguese",
    max_iterations: int = 2,
    quality_threshold: float = 8.0,
) -> ResearchState:
    return ResearchState(
        query=query,
        context=context,
        language=language,
        research="",
        sources=[],
        analysis="",
        report="",
        critique="",
        scores={},
        weighted_score=0.0,
        iteration=0,
        max_iterations=max_iterations,
        quality_threshold=quality_threshold,
        final_report="",
        logs=[],
    )
