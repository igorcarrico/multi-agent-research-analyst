"""Single-agent baseline.

Lives outside the LangGraph workflow on purpose: it represents the naive
'just send one prompt' approach we want to compare against.
"""
from __future__ import annotations

from typing import Any

from utils.prompts import BASELINE_PROMPT

from .base import BaseAgent


class BaselineAgent(BaseAgent):
    name = "baseline"

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        t0 = self.logger.start(self.name, summary="single-prompt baseline")
        report = self._invoke(
            BASELINE_PROMPT,
            query=state["query"],
            language=state.get("language", "Portuguese"),
        )
        self.logger.end(self.name, t0, summary=f"{len(report)} chars")
        return {"report": report}
