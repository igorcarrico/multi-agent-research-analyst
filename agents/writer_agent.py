"""Writer Agent — produces the report draft."""
from __future__ import annotations

from typing import Any

from utils.prompts import WRITER_PROMPT

from .base import BaseAgent


class WriterAgent(BaseAgent):
    name = "writer"

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        t0 = self.logger.start(
            self.name,
            summary=f"drafting (iteration {state.get('iteration', 0)})",
        )
        report = self._invoke(
            WRITER_PROMPT,
            query=state["query"],
            research=state.get("research", ""),
            analysis=state.get("analysis", ""),
            critique=state.get("critique", "") or "none — first draft",
            language=state.get("language", "Portuguese"),
        )
        self.logger.end(self.name, t0, summary=f"{len(report)} chars")
        return {"report": report}
