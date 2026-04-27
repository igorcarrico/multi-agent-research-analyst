"""Side-by-side comparator: baseline vs multi-agent.

Returns:
  - scores for each report (axis-by-axis + aggregate)
  - heuristic deltas
  - a short LLM-written verdict explaining which is stronger and why
"""
from __future__ import annotations

from typing import Any

from agents.base import _extract_usage
from utils import AppConfig, LogEvent, StepLogger, build_llm
from utils.prompts import COMPARATOR_PROMPT

from .evaluator import Evaluator
from .metrics import heuristic_score


def compare_outputs(
    cfg: AppConfig,
    logger: StepLogger,
    *,
    query: str,
    analysis: str,
    baseline_report: str,
    multiagent_report: str,
    language: str = "Portuguese",
) -> dict[str, Any]:
    evaluator = Evaluator(cfg, logger)

    t0 = logger.start("comparator", summary="scoring baseline")
    baseline_scores, baseline_weighted = evaluator.score(query, analysis, baseline_report)
    logger.end("comparator", t0, summary=f"baseline={baseline_weighted}")

    t0 = logger.start("comparator", summary="scoring multi-agent")
    multi_scores, multi_weighted = evaluator.score(query, analysis, multiagent_report)
    logger.end("comparator", t0, summary=f"multi-agent={multi_weighted}")

    # LLM verdict — narrative explanation, not a score.
    judge = build_llm(cfg.llm)
    verdict_msg = (COMPARATOR_PROMPT | judge).invoke(
        {
            "query": query,
            "report_a": baseline_report,
            "report_b": multiagent_report,
            "language": language,
        }
    )
    usage = _extract_usage(verdict_msg, cfg.llm.model)
    logger.log(LogEvent(
        agent="comparator", event="usage",
        payload={"summary": f"{usage['input_tokens']}+{usage['output_tokens']} tok",
                 "usage": usage},
    ))
    verdict = verdict_msg.content if hasattr(verdict_msg, "content") else str(verdict_msg)

    return {
        "baseline": {
            "scores": baseline_scores,
            "weighted_score": baseline_weighted,
            "heuristics": heuristic_score(baseline_report),
        },
        "multiagent": {
            "scores": multi_scores,
            "weighted_score": multi_weighted,
            "heuristics": heuristic_score(multiagent_report),
        },
        "delta": round(multi_weighted - baseline_weighted, 2),
        "verdict": verdict,
    }
