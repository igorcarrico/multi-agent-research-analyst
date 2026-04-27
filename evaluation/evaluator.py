"""LLM-as-judge evaluator.

Returns per-axis scores (0–10) and a weighted aggregate. Falls back to
heuristics if the judge returns malformed JSON, so the workflow never deadlocks
on a parsing error.
"""
from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from agents.base import _extract_usage
from utils import AppConfig, LogEvent, StepLogger, build_llm
from utils.prompts import EVALUATOR_PROMPT

from .metrics import heuristic_score

_AXES = ("coherence", "clarity", "completeness", "factual_consistency")


def _parse_json(raw: str) -> dict[str, Any] | None:
    """Tolerant JSON parse — judges sometimes wrap output in markdown fences."""
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.MULTILINE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


class Evaluator:
    """Scores reports. Used both as a graph node and standalone for comparisons."""

    name = "evaluator"

    def __init__(self, cfg: AppConfig, logger: StepLogger):
        self.cfg = cfg
        self.logger = logger
        self.weights = cfg.evaluation.weights
        # Use the base config — we want a neutral judge.
        self.llm_cfg = cfg.llm
        self.llm: BaseChatModel = build_llm(cfg.llm)

    def score(self, query: str, analysis: str, report: str) -> tuple[dict[str, float], float]:
        """Return (per-axis scores, weighted aggregate)."""
        chain = EVALUATOR_PROMPT | self.llm
        msg = chain.invoke({"query": query, "analysis": analysis, "report": report})
        usage = _extract_usage(msg, self.llm_cfg.model)
        self.logger.log(LogEvent(
            agent=self.name, event="usage",
            payload={"summary": f"{usage['input_tokens']}+{usage['output_tokens']} tok",
                     "usage": usage},
        ))
        raw = msg.content
        parsed = _parse_json(raw)

        if not parsed:
            # Fallback: cheap heuristics if the judge fails.
            heur = heuristic_score(report)
            scores = {ax: heur.get("structure_ok", 5.0) for ax in _AXES}
        else:
            scores = {ax: float(parsed.get(ax, 0.0)) for ax in _AXES}

        weighted = sum(scores[ax] * self.weights.get(ax, 0.0) for ax in _AXES)
        return scores, round(weighted, 2)

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """LangGraph node entrypoint."""
        t0 = self.logger.start(self.name, summary="scoring report")
        scores, weighted = self.score(
            query=state["query"],
            analysis=state.get("analysis", ""),
            report=state.get("report", ""),
        )
        threshold = state.get("quality_threshold", 8.0)
        decision = "pass" if weighted >= threshold else "revise"
        self.logger.end(
            self.name,
            t0,
            summary=f"score={weighted}/10 → {decision}",
            scores=scores,
        )
        self.logger.decision(self.name, weighted=weighted, threshold=threshold, decision=decision)
        return {"scores": scores, "weighted_score": weighted}
