"""Analyst Agent — turns raw notes into structured insights."""
from __future__ import annotations

from typing import Any

from utils.prompts import ANALYST_PROMPT

from .base import BaseAgent


class AnalystAgent(BaseAgent):
    name = "analyst"

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        t0 = self.logger.start(self.name, summary="extracting insights")
        analysis = self._invoke(
            ANALYST_PROMPT,
            query=state["query"],
            research=state.get("research", ""),
            language=state.get("language", "Portuguese"),
        )
        self.logger.end(self.name, t0, summary=f"{len(analysis)} chars")
        return {"analysis": analysis}
