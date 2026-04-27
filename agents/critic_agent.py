"""Critic Agent — sharp adversarial review of the current draft."""
from __future__ import annotations

from typing import Any

from utils.prompts import CRITIC_PROMPT

from .base import BaseAgent


class CriticAgent(BaseAgent):
    name = "critic"

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        t0 = self.logger.start(self.name, summary="critiquing draft")
        critique = self._invoke(
            CRITIC_PROMPT,
            query=state["query"],
            report=state.get("report", ""),
            language=state.get("language", "Portuguese"),
        )
        self.logger.end(self.name, t0, summary=f"{len(critique)} chars")
        return {"critique": critique}
