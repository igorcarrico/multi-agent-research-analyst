"""Typed configuration loader. Reads config.yaml + .env once at startup."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.3
    max_tokens: int = 2000


class AgentOverride(BaseModel):
    temperature: float | None = None
    model: str | None = None
    max_tokens: int | None = None


class WorkflowConfig(BaseModel):
    max_iterations: int = 2
    quality_threshold: float = 8.0
    enable_web_search: bool = False


class EvaluationConfig(BaseModel):
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "coherence": 0.30,
            "clarity": 0.25,
            "completeness": 0.30,
            "factual_consistency": 0.15,
        }
    )


class AppConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    agents: dict[str, AgentOverride] = Field(default_factory=dict)
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    def llm_for(self, agent_name: str) -> LLMConfig:
        """Resolve effective LLM config for one agent (base + override)."""
        base = self.llm.model_dump()
        override = self.agents.get(agent_name)
        if override:
            for k, v in override.model_dump().items():
                if v is not None:
                    base[k] = v
        return LLMConfig(**base)


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    p = Path(path)
    if not p.exists():
        return AppConfig()
    raw: dict[str, Any] = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return AppConfig(**raw)
