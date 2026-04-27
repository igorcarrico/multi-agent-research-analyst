"""Base class for agents.

Each agent owns its own LLM (so per-agent overrides like temperature/model work),
and exposes `.run(state) -> patch`. The patch is a partial dict that LangGraph
merges into ResearchState.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from tenacity import retry, stop_after_attempt, wait_exponential

from utils import AppConfig, LogEvent, StepLogger, build_llm


def _extract_usage(msg: Any, model_name: str) -> dict[str, Any]:
    """Pull token counts from an AIMessage in a provider-agnostic way."""
    usage_meta = getattr(msg, "usage_metadata", None) or {}
    if usage_meta:
        return {
            "model": model_name,
            "input_tokens": usage_meta.get("input_tokens", 0),
            "output_tokens": usage_meta.get("output_tokens", 0),
        }
    # Fallback: try response_metadata (older LangChain shape).
    rm = getattr(msg, "response_metadata", {}) or {}
    usage = rm.get("usage") or rm.get("token_usage") or {}
    return {
        "model": model_name,
        "input_tokens": usage.get("input_tokens", usage.get("prompt_tokens", 0)),
        "output_tokens": usage.get("output_tokens", usage.get("completion_tokens", 0)),
    }


class BaseAgent(ABC):
    name: str = "base"

    def __init__(self, cfg: AppConfig, logger: StepLogger):
        self.cfg = cfg
        self.logger = logger
        self.llm_cfg = cfg.llm_for(self.name)
        self.llm: BaseChatModel = build_llm(self.llm_cfg)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def _invoke(self, prompt, **kwargs) -> str:
        """Invoke the LLM with retry, log token usage, return content."""
        chain = prompt | self.llm
        msg = chain.invoke(kwargs)
        usage = _extract_usage(msg, self.llm_cfg.model)
        # Emit a usage event so the UI/logs can aggregate cost per agent.
        self.logger.log(LogEvent(
            agent=self.name,
            event="usage",
            payload={"summary": f"{usage['input_tokens']}+{usage['output_tokens']} tok",
                     "usage": usage},
        ))
        return msg.content if hasattr(msg, "content") else str(msg)

    @abstractmethod
    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Process state and return a partial dict to merge in."""
        ...
