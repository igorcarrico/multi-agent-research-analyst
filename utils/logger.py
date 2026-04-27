"""Structured step logger.

Each agent emits a LogEvent. The UI consumes the same list to render the
execution trace, so logs are first-class data, not stderr noise.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
_file_logger = logging.getLogger("multi_agent")


@dataclass
class LogEvent:
    agent: str
    event: str                     # "start" | "end" | "decision" | "error"
    duration_ms: float | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StepLogger:
    """Collects events in memory AND tees to a JSONL file for post-mortem."""

    def __init__(self, log_file: str | Path | None = "logs/run.jsonl"):
        self.events: list[LogEvent] = []
        self.log_file = Path(log_file) if log_file else None
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: LogEvent) -> None:
        self.events.append(event)
        line = json.dumps(event.to_dict(), ensure_ascii=False, default=str)
        _file_logger.info(f"[{event.agent}] {event.event} {event.payload.get('summary', '')}")
        if self.log_file:
            with self.log_file.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def start(self, agent: str, **payload: Any) -> float:
        self.log(LogEvent(agent=agent, event="start", payload=payload))
        return time.time()

    def end(self, agent: str, t0: float, **payload: Any) -> None:
        self.log(
            LogEvent(
                agent=agent,
                event="end",
                duration_ms=(time.time() - t0) * 1000,
                payload=payload,
            )
        )

    def decision(self, agent: str, **payload: Any) -> None:
        self.log(LogEvent(agent=agent, event="decision", payload=payload))

    def error(self, agent: str, err: Exception) -> None:
        self.log(LogEvent(agent=agent, event="error", payload={"error": str(err)}))
