"""Cost tracking for LLM API calls — accumulates token usage per stage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMResult:
    """Result from a single LLM call, including usage metadata."""

    content: str
    usage: dict[str, int] = field(default_factory=dict)
    model: str = ""
    latency_ms: float = 0.0

    @property
    def prompt_tokens(self) -> int:
        return self.usage.get("prompt_tokens", 0)

    @property
    def completion_tokens(self) -> int:
        return self.usage.get("completion_tokens", 0)

    @property
    def total_tokens(self) -> int:
        return self.usage.get("total_tokens", 0) or (
            self.prompt_tokens + self.completion_tokens
        )


@dataclass
class StageCost:
    """Accumulated cost for one pipeline stage."""

    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    num_calls: int = 0
    total_latency_ms: float = 0.0

    def record(self, result: LLMResult) -> None:
        self.total_tokens += result.total_tokens
        self.prompt_tokens += result.prompt_tokens
        self.completion_tokens += result.completion_tokens
        self.num_calls += 1
        self.total_latency_ms += result.latency_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "num_calls": self.num_calls,
            "total_latency_ms": round(self.total_latency_ms, 1),
        }


class CostTracker:
    """Accumulates LLM costs across all pipeline stages."""

    def __init__(self) -> None:
        self._stages: dict[str, StageCost] = {}
        self._current_stage: str = ""

    def set_stage(self, stage_name: str) -> None:
        self._current_stage = stage_name
        if stage_name not in self._stages:
            self._stages[stage_name] = StageCost()

    def record(self, result: LLMResult) -> None:
        """Record an LLM call result to the current stage."""
        if not self._current_stage:
            return
        if self._current_stage not in self._stages:
            self._stages[self._current_stage] = StageCost()
        self._stages[self._current_stage].record(result)

    def summary(self) -> dict[str, Any]:
        """Return a full cost summary across all stages."""
        stages = {name: cost.to_dict() for name, cost in self._stages.items()}
        total_tokens = sum(c.total_tokens for c in self._stages.values())
        total_calls = sum(c.num_calls for c in self._stages.values())
        total_latency = sum(c.total_latency_ms for c in self._stages.values())
        return {
            "stages": stages,
            "total_tokens": total_tokens,
            "total_calls": total_calls,
            "total_latency_ms": round(total_latency, 1),
        }
