"""Reviewer Agent — rewrites the draft to address every critique point.

Increments `iteration` so the conditional router can give up after N attempts.
"""
from __future__ import annotations

from typing import Any

from utils.prompts import REVIEWER_PROMPT

from .base import BaseAgent


class ReviewerAgent(BaseAgent):
    name = "reviewer"

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        iteration = state.get("iteration", 0) + 1
        t0 = self.logger.start(self.name, summary=f"rewrite pass {iteration}")
        improved = self._invoke(
            REVIEWER_PROMPT,
            query=state["query"],
            critique=state.get("critique", ""),
            report=state.get("report", ""),
            analysis=state.get("analysis", ""),
            language=state.get("language", "Portuguese"),
        )
        self.logger.end(self.name, t0, summary=f"{len(improved)} chars")
        return {"report": improved, "iteration": iteration}
