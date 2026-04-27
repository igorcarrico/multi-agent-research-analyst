"""Research Agent.

Two modes:
  - simulated (default): the LLM produces structured notes from its parametric
    knowledge. Good enough for portfolio demos and avoids API keys for search.
  - web: uses DuckDuckGo to fetch real snippets, then asks the LLM to synthesize.
    Toggled via config.workflow.enable_web_search.

Designed so swapping in a real RAG retriever later means changing only
`_gather_context`.
"""
from __future__ import annotations

from typing import Any

from utils.prompts import RESEARCH_PROMPT

from .base import BaseAgent


class ResearchAgent(BaseAgent):
    name = "research"

    def _gather_context(self, query: str) -> tuple[str, list[str]]:
        if not self.cfg.workflow.enable_web_search:
            return "", []

        try:
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                hits = list(ddgs.text(query, max_results=5))
            snippets = [f"- {h.get('title', '')}: {h.get('body', '')}" for h in hits]
            sources = [h.get("href", "") for h in hits if h.get("href")]
            return "\n".join(snippets), sources
        except Exception as e:
            self.logger.error(self.name, e)
            return "", []

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        query = state["query"]
        t0 = self.logger.start(self.name, summary=f"researching: {query[:60]}")

        web_context, sources = self._gather_context(query)
        full_context = (state.get("context", "") + "\n\n" + web_context).strip()

        try:
            notes = self._invoke(
                RESEARCH_PROMPT,
                query=query,
                context=full_context or "none",
                language=state.get("language", "Portuguese"),
            )
        except Exception as e:
            self.logger.error(self.name, e)
            raise

        self.logger.end(self.name, t0, summary=f"{len(notes)} chars, {len(sources)} sources")
        return {"research": notes, "sources": sources}
