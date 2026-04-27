"""LLM factory. Centralizes provider choice so agents stay provider-agnostic."""
from __future__ import annotations

import os
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from .config import LLMConfig


def build_llm(cfg: LLMConfig, **extra: Any) -> BaseChatModel:
    """Return a chat model for the given config.

    Switching providers is a one-line change in config.yaml.
    """
    provider = cfg.provider.lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            api_key=os.getenv("OPENAI_API_KEY"),
            **extra,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            **extra,
        )

    raise ValueError(f"Unsupported provider: {cfg.provider}")
