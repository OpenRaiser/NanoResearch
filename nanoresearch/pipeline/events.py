"""Typed event system for pipeline progress reporting.

Inspired by EvoScientist's stream/events.py. Events flow from agents
through the orchestrator to UI consumers (CLI Live, Feishu bot, etc.).

Event types:
    stage_start, stage_complete, stage_failed, stage_skip
    substep          — fine-grained progress within a stage
    thinking         — LLM is generating (with token count)
    tool_call        — tool invocation started
    tool_result      — tool invocation completed
    reflection       — stage reflection result
    pipeline_start, pipeline_complete, pipeline_failed
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable


class EventType(str, Enum):
    PIPELINE_START = "pipeline_start"
    PIPELINE_COMPLETE = "pipeline_complete"
    PIPELINE_FAILED = "pipeline_failed"
    STAGE_START = "stage_start"
    STAGE_COMPLETE = "stage_complete"
    STAGE_FAILED = "stage_failed"
    STAGE_SKIP = "stage_skip"
    SUBSTEP = "substep"
    THINKING = "thinking"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    REFLECTION = "reflection"
    RETRY = "retry"


@dataclass
class PipelineEvent:
    """A single pipeline event."""
    type: EventType
    stage: str = ""
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    elapsed_s: float = 0.0


# Listener type: receives a PipelineEvent
EventListener = Callable[[PipelineEvent], None]


class EventEmitter:
    """Emits typed events to registered listeners.

    Usage:
        emitter = EventEmitter()
        emitter.on(listener_fn)  # register
        emitter.emit(PipelineEvent(type=EventType.STAGE_START, stage="ideation"))
    """

    def __init__(self) -> None:
        self._listeners: list[EventListener] = []
        self._start_time = time.monotonic()

    def on(self, listener: EventListener) -> None:
        """Register an event listener."""
        self._listeners.append(listener)

    def off(self, listener: EventListener) -> None:
        """Remove an event listener."""
        self._listeners = [l for l in self._listeners if l is not listener]

    def emit(self, event: PipelineEvent) -> None:
        """Emit an event to all listeners."""
        event.elapsed_s = round(time.monotonic() - self._start_time, 1)
        for listener in self._listeners:
            try:
                listener(event)
            except Exception:
                pass  # listeners must not crash the pipeline

    # ─── convenience methods ───

    def pipeline_start(self, topic: str, total_stages: int) -> None:
        self.emit(PipelineEvent(
            type=EventType.PIPELINE_START,
            message=f"Starting pipeline: {topic}",
            data={"topic": topic, "total_stages": total_stages},
        ))

    def pipeline_complete(self, success: bool, message: str = "") -> None:
        self.emit(PipelineEvent(
            type=EventType.PIPELINE_COMPLETE if success else EventType.PIPELINE_FAILED,
            message=message or ("Pipeline completed" if success else "Pipeline failed"),
            data={"success": success},
        ))

    def stage_start(self, stage: str, index: int, total: int) -> None:
        self.emit(PipelineEvent(
            type=EventType.STAGE_START,
            stage=stage,
            message=f"[{index+1}/{total}] Running {stage}...",
            data={"index": index, "total": total},
        ))

    def stage_complete(self, stage: str, duration_s: float) -> None:
        self.emit(PipelineEvent(
            type=EventType.STAGE_COMPLETE,
            stage=stage,
            message=f"{stage} completed in {duration_s:.1f}s",
            data={"duration_s": duration_s},
        ))

    def stage_failed(self, stage: str, error: str) -> None:
        self.emit(PipelineEvent(
            type=EventType.STAGE_FAILED,
            stage=stage,
            message=f"{stage} failed: {error}",
            data={"error": error},
        ))

    def substep(self, stage: str, message: str) -> None:
        self.emit(PipelineEvent(
            type=EventType.SUBSTEP,
            stage=stage,
            message=message,
        ))

    def thinking(self, stage: str, token_count: int = 0) -> None:
        self.emit(PipelineEvent(
            type=EventType.THINKING,
            stage=stage,
            message=f"Thinking... ({token_count} tokens)" if token_count else "Thinking...",
            data={"token_count": token_count},
        ))

    def tool_call(self, stage: str, tool_name: str, args_summary: str = "") -> None:
        self.emit(PipelineEvent(
            type=EventType.TOOL_CALL,
            stage=stage,
            message=f"Tool: {tool_name}({args_summary})",
            data={"tool_name": tool_name, "args_summary": args_summary},
        ))

    def tool_result(self, stage: str, tool_name: str, success: bool, summary: str = "") -> None:
        self.emit(PipelineEvent(
            type=EventType.TOOL_RESULT,
            stage=stage,
            message=f"{'OK' if success else 'ERR'} {tool_name}: {summary}",
            data={"tool_name": tool_name, "success": success, "summary": summary},
        ))

    def reflection(self, stage: str, quality_score: int, suggestions: list[str]) -> None:
        self.emit(PipelineEvent(
            type=EventType.REFLECTION,
            stage=stage,
            message=f"Reflection: quality={quality_score}/10",
            data={"quality_score": quality_score, "suggestions": suggestions},
        ))

    def retry(self, stage: str, attempt: int, max_attempts: int, delay_s: float) -> None:
        self.emit(PipelineEvent(
            type=EventType.RETRY,
            stage=stage,
            message=f"Retrying {stage} in {delay_s:.0f}s ({attempt}/{max_attempts})",
            data={"attempt": attempt, "max_attempts": max_attempts, "delay_s": delay_s},
        ))
