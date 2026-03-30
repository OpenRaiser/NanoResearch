"""Base agent — common LLM call logic for all research agents."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import traceback as _tb_mod
from abc import ABC, abstractmethod
from typing import Any, Callable

from nanoresearch.config import ResearchConfig, StageModelConfig
from nanoresearch.exceptions import LLMError
from nanoresearch.pipeline.multi_model import ModelDispatcher
from nanoresearch.pipeline.workspace import Workspace
from nanoresearch.schemas.manifest import PipelineStage

# Import all free functions from the helpers module so they remain accessible
# at their original locations (e.g. ``from nanoresearch.agents.base import detect_truncation``).
from nanoresearch.agents._base_helpers import (  # noqa: F401 — re-exports
    _VALID_JSON_ESCAPES,
    _LATEX_CMD_PREFIXES,
    _MAX_TOOL_RESULT_CHARS,
    _HEAD_CHARS,
    _TAIL_CHARS,
    _CONTEXT_COMPACT_THRESHOLD_CHARS,
    _PROTECTED_TAIL_TURNS,
    _truncate_tool_result,
    _compact_messages_if_needed,
    _fix_json_escapes,
    _extract_balanced_json_segment,
    _extract_json_candidates,
    _scan_json_fragment,
    _close_json_fragment,
    _trim_json_fragment,
    _repair_truncated_json,
    _json_error_msg,
    detect_truncation,
)

logger = logging.getLogger(__name__)


class BaseResearchAgent(ABC):
    """Abstract base class for all NanoResearch agents."""

    stage: PipelineStage  # subclass must set this

    def __init__(self, workspace: Workspace, config: ResearchConfig) -> None:
        self.workspace = workspace
        self.config = config
        self._dispatcher = ModelDispatcher(config)
        self._substep_callback: Callable[[str], None] | None = None

    def report_substep(self, msg: str) -> None:
        """Report a sub-step progress update (e.g. 'Searching paper 3/10')."""
        if self._substep_callback is not None:
            try:
                self._substep_callback(msg)
            except Exception:
                pass  # progress is cosmetic, never crash

    def _remember_mutation_snapshot_entry(self, entry: dict[str, Any] | None) -> None:
        self._last_mutation_snapshot_entry = dict(entry) if isinstance(entry, dict) else None

    def consume_last_mutation_snapshot_entry(self) -> dict[str, Any] | None:
        entry = getattr(self, "_last_mutation_snapshot_entry", None)
        self._last_mutation_snapshot_entry = None
        return dict(entry) if isinstance(entry, dict) else None

    @property
    def stage_config(self) -> StageModelConfig:
        return self.config.for_stage(self.stage.value)

    async def close(self) -> None:
        await self._dispatcher.close()

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
        stage_override: StageModelConfig | None = None,
    ) -> str:
        """Call the LLM configured for this agent's stage."""
        cfg = stage_override if stage_override is not None else self.stage_config
        return await self._dispatcher.generate(
            cfg, system_prompt, user_prompt, json_mode
        )

    async def generate_with_image(
        self,
        system_prompt: str,
        user_prompt: str,
        image_bytes: bytes,
        mime_type: str = "image/png",
        json_mode: bool = False,
        stage_override: StageModelConfig | None = None,
    ) -> str:
        """Call the LLM with an image attachment (vision)."""
        cfg = stage_override if stage_override is not None else self.stage_config
        return await self._dispatcher.generate_with_image(
            cfg, system_prompt, user_prompt, image_bytes, mime_type, json_mode
        )

    async def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        stage_override: StageModelConfig | None = None,
    ) -> dict | list:
        """Call LLM and parse the response as JSON.

        Handles LaTeX backslash sequences that break strict JSON parsing.
        """
        raw = await self.generate(
            system_prompt, user_prompt, json_mode=True,
            stage_override=stage_override,
        )
        last_attempt = raw.strip()
        for text in _extract_json_candidates(raw):
            last_attempt = text
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

            fixed = _fix_json_escapes(text)
            last_attempt = fixed
            try:
                return json.loads(fixed, strict=False)
            except json.JSONDecodeError:
                pass

            repaired = _repair_truncated_json(fixed)
            if repaired is not None and repaired != fixed:
                last_attempt = repaired
                try:
                    return json.loads(repaired, strict=False)
                except json.JSONDecodeError:
                    pass

        # All attempts failed
        logger.error(
            "JSON parse failed even after escape fixing. First 500 chars: %s",
            last_attempt[:500],
        )
        raise LLMError(
            f"LLM output is not valid JSON: "
            f"{_json_error_msg(last_attempt)}. "
            f"Raw output starts with: {raw[:200]!r}"
        ) from None

    _MAX_JSON_RETRIES = 3

    async def generate_json_validated(
        self,
        system_prompt: str,
        user_prompt: str,
        model_class: type,
        stage_override: StageModelConfig | None = None,
    ) -> Any:
        """Call LLM, parse as JSON, and validate against a Pydantic model.

        On validation failure, feeds accumulated errors back for up to 3 retries.
        Returns a validated Pydantic model instance.
        """
        error_history: list[str] = []
        prompt = user_prompt

        for attempt in range(1 + self._MAX_JSON_RETRIES):
            try:
                raw_dict = await self.generate_json(
                    system_prompt, prompt, stage_override=stage_override,
                )
                return model_class.model_validate(raw_dict)
            except Exception as exc:
                error_history.append(f"Attempt {attempt + 1}: {exc}")
                self.log(f"  JSON validation failed (attempt {attempt + 1}/{1 + self._MAX_JSON_RETRIES}): {exc}")

                if attempt >= self._MAX_JSON_RETRIES:
                    logger.error("JSON validation failed after %d retries: %s", self._MAX_JSON_RETRIES, exc)
                    raise LLMError(
                        f"JSON schema validation failed after {self._MAX_JSON_RETRIES} retries:\n"
                        + "\n".join(error_history)
                    ) from exc

                # Build retry prompt with ALL accumulated errors
                prompt = (
                    f"Your previous JSON responses had validation errors:\n"
                    + "\n".join(f"  - {e}" for e in error_history)
                    + f"\n\nOriginal request:\n{user_prompt}\n\n"
                    f"Fix the JSON to match the required schema and try again."
                )

        raise RuntimeError("Unreachable")  # pragma: no cover

    async def generate_with_tools(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: Any,  # ToolRegistry
        max_tool_rounds: int = 10,
        stage_override: StageModelConfig | None = None,
        reminder_text: str | None = None,
        reminder_interval: int = 3,
    ) -> str:
        """Run a ReAct loop: let the LLM call tools until it produces text."""
        cfg = stage_override if stage_override is not None else self.stage_config
        openai_tools = tools.to_openai_tools()

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Track repeated failures to avoid infinite retry loops (OpenClaw pattern)
        _failure_counts: dict[str, int] = {}
        _MAX_IDENTICAL_FAILURES = 2

        for round_idx in range(max_tool_rounds):
            msg = await self._dispatcher.generate_with_tools(cfg, messages, openai_tools)

            tool_calls = getattr(msg, "tool_calls", None)
            if not tool_calls:
                return self._dispatcher._strip_think_blocks(msg.content or "")

            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }
            if msg.content:
                assistant_msg["content"] = msg.content
            messages.append(assistant_msg)

            async def _execute_tool_call(tc):
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError as exc:
                    logger.warning("Invalid JSON in tool args for %s: %s", name, exc)
                    args = {}

                self.log(f"Tool call: {name}({args})")
                self.report_substep(f"Tool: {name}")
                try:
                    result = await tools.call(name, args)
                    result_str = json.dumps(result, ensure_ascii=False, default=str)
                    status = "success"
                except Exception as e:
                    # Structured error with traceback (EvoScientist pattern)
                    tb = _tb_mod.format_exc()
                    logger.error("Tool %r raised an exception:\n%s", name, tb)
                    error_str = (
                        f"[TOOL ERROR] Tool '{name}' failed with {type(e).__name__}: {e}\n\n"
                        f"Traceback (most recent call last):\n{tb[-500:]}"
                    )
                    error_sig = type(e).__name__
                    try:
                        args_hash = hash(json.dumps(args, sort_keys=True, default=str))
                    except (TypeError, ValueError):
                        args_hash = hash(str(sorted(args.items())) if isinstance(args, dict) else str(args))
                    fail_key = f"{name}|{args_hash}|{error_sig}"
                    _failure_counts[fail_key] = _failure_counts.get(fail_key, 0) + 1
                    if _failure_counts[fail_key] >= _MAX_IDENTICAL_FAILURES:
                        error_str = (
                            f"[NON-RETRYABLE] {error_str}\n\n"
                            f"This exact call has failed {_failure_counts[fail_key]} times. "
                            f"Do NOT retry with the same arguments. Try a different approach."
                        )
                    result_str = error_str
                    status = "error"

                result_content = _truncate_tool_result(result_str)
                return {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_content,
                    # Status field (EvoScientist pattern) — allows agents
                    # to distinguish success from error without string parsing.
                    # Note: OpenAI API ignores extra fields; they're for internal use.
                    "_status": status,
                }

            if len(tool_calls) > 1:
                tool_results = await asyncio.gather(
                    *(_execute_tool_call(tc) for tc in tool_calls),
                    return_exceptions=True,
                )
                for i, tr in enumerate(tool_results):
                    if isinstance(tr, Exception):
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_calls[i].id,
                            "content": f"Error: {type(tr).__name__}: {tr}",
                        })
                    else:
                        messages.append(tr)
            else:
                messages.append(await _execute_tool_call(tool_calls[0]))

            _compact_messages_if_needed(messages)

            if (round_idx + 1) % reminder_interval == 0 and round_idx + 1 <= max_tool_rounds:
                _reminder = reminder_text or (
                    "[REMINDER] You are writing academic content for a top-tier venue. "
                    "Focus on producing the final output now. Use the information "
                    "gathered from tools to write high-quality content. "
                    "Do NOT continue searching indefinitely."
                )
                messages.append({"role": "system", "content": _reminder})

        self.log(f"Exceeded {max_tool_rounds} tool rounds, forcing final answer")
        final_msg = await self._dispatcher.generate_with_tools(cfg, messages, tools=None)
        if hasattr(final_msg, 'tool_calls') and final_msg.tool_calls:
            return self._dispatcher._strip_think_blocks(
                final_msg.content or "Agent completed but produced no text summary."
            )
        return self._dispatcher._strip_think_blocks(final_msg.content or "")

    @abstractmethod
    async def run(self, **inputs: Any) -> dict[str, Any]:
        """Execute this agent's stage. Returns output data dict."""
        ...

    @staticmethod
    def _inject_error_context(prompt: str, last_error: str | None) -> str:
        """Prepend error context from a previous failed attempt to a prompt.

        This lets agents adapt their behavior on retry instead of blindly
        repeating the same approach.
        """
        if not last_error:
            return prompt
        return (
            f"[PREVIOUS ATTEMPT FAILED]\n"
            f"The previous attempt at this stage failed with:\n"
            f"  {last_error}\n"
            f"Adapt your approach to avoid the same failure.\n\n"
            f"{prompt}"
        )

    def log(self, msg: str) -> None:
        logger.info(f"[{self.stage.value}] {msg}")

    def save_log(self, filename: str, content: str) -> None:
        self.workspace.write_text(f"logs/{filename}", content)

    def _resolve_experiment_python(self) -> str:
        """Return the experiment Python path.

        Resolution order:
        1. config.experiment_python (user-managed environment)
        2. experiment/.venv python (auto-created venv)
        3. sys.executable (fallback)
        """
        import os
        import sys
        from pathlib import Path as _Path

        # Priority 1: user-specified python
        user_spec = (self.config.experiment_python or "").strip()
        if user_spec:
            from nanoresearch.agents.runtime_env import RuntimeEnvironmentManager
            mgr = RuntimeEnvironmentManager(self.config)
            resolved = mgr._resolve_user_python(user_spec)
            if resolved and _Path(resolved).exists():
                return resolved

        # Priority 2: experiment venv
        exp_dir = self.workspace.path / "experiment"
        if os.name == "nt":
            venv_py = exp_dir / ".venv" / "Scripts" / "python.exe"
        else:
            venv_py = exp_dir / ".venv" / "bin" / "python"
        if venv_py.exists():
            return str(venv_py)
        return sys.executable
