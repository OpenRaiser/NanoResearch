"""Base agent — common LLM call logic for all research agents."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any

from nanoresearch.config import ResearchConfig, StageModelConfig
from nanoresearch.exceptions import LLMError
from nanoresearch.pipeline.multi_model import ModelDispatcher
from nanoresearch.pipeline.workspace import Workspace
from nanoresearch.schemas.manifest import PipelineStage

logger = logging.getLogger(__name__)

# JSON valid escape characters (after the backslash)
_VALID_JSON_ESCAPES = frozenset('"\\/bfnrtu')


def _fix_json_escapes(text: str) -> str:
    """Fix invalid JSON escape sequences produced by LaTeX content.

    LLM outputs often contain raw LaTeX like \\cite{}, \\textbf{}, \\frac{}
    inside JSON strings. These produce invalid \\c, \\t, \\f escapes.
    We double-escape them so json.loads() can parse them.
    """
    result = []
    i = 0
    while i < len(text):
        if text[i] == '\\' and i + 1 < len(text):
            next_char = text[i + 1]
            if next_char in _VALID_JSON_ESCAPES:
                # Valid JSON escape — keep as-is
                result.append(text[i])
                result.append(next_char)
                i += 2
            elif next_char == '\\':
                # Already escaped backslash
                result.append('\\\\')
                i += 2
            else:
                # Invalid escape (e.g. \c from \cite) — double the backslash
                result.append('\\\\')
                i += 1  # re-process next_char as a normal character
        else:
            result.append(text[i])
            i += 1
    return ''.join(result)


class BaseResearchAgent(ABC):
    """Abstract base class for all NanoResearch agents."""

    stage: PipelineStage  # subclass must set this

    def __init__(self, workspace: Workspace, config: ResearchConfig) -> None:
        self.workspace = workspace
        self.config = config
        self._dispatcher = ModelDispatcher(config)

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
        # Try to extract JSON from markdown code blocks if present
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove opening fence (e.g. ```json, ```python, ```)
            lines = lines[1:]
            # Remove only the last closing fence (not all triple-backtick lines)
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines)

        # First try strict parsing
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Fix LaTeX backslash escapes and retry with strict=False
        # (strict=False allows raw control characters like \n \t inside strings)
        fixed = _fix_json_escapes(text)
        try:
            return json.loads(fixed, strict=False)
        except json.JSONDecodeError as exc:
            logger.error(
                "JSON parse failed even after escape fixing. First 500 chars: %s",
                fixed[:500],
            )
            raise LLMError(
                f"LLM output is not valid JSON: {exc}. "
                f"Raw output starts with: {raw[:200]!r}"
            ) from exc

    async def generate_with_tools(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: Any,  # ToolRegistry
        max_tool_rounds: int = 10,
        stage_override: StageModelConfig | None = None,
    ) -> str:
        """Run a ReAct loop: let the LLM call tools until it produces text.

        Args:
            system_prompt: System message.
            user_prompt: Initial user message.
            tools: A ``ToolRegistry`` instance.
            max_tool_rounds: Max tool-call round-trips before forcing a final answer.
            stage_override: Optional stage config override.

        Returns:
            Final assistant text content.
        """
        cfg = stage_override if stage_override is not None else self.stage_config
        openai_tools = tools.to_openai_tools()

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        for round_idx in range(max_tool_rounds):
            msg = await self._dispatcher.generate_with_tools(cfg, messages, openai_tools)

            # If the model returns text without tool calls, we're done
            tool_calls = getattr(msg, "tool_calls", None)
            if not tool_calls:
                return msg.content or ""

            # Append the assistant message with tool calls.
            # NOTE: omit "content" when empty — some backends (e.g. AWS
            # Bedrock) reject empty content blocks.
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

            # Execute tool calls in parallel when multiple are requested
            async def _execute_tool_call(tc):
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "Invalid JSON in tool args for %s: %s", name, exc,
                    )
                    args = {}

                self.log(f"Tool call: {name}({args})")
                try:
                    result = await tools.call(name, args)
                    result_str = json.dumps(result, ensure_ascii=False, default=str)
                except Exception as e:
                    result_str = f"Error: {type(e).__name__}: {e}"

                return {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str[:8000],  # truncate large results
                }

            if len(tool_calls) > 1:
                # Run independent tool calls concurrently
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
                # Single tool call — no concurrency overhead
                messages.append(await _execute_tool_call(tool_calls[0]))

            # Inject system reminder every 3 rounds to prevent instruction drift
            if (round_idx + 1) % 3 == 0 and round_idx + 1 < max_tool_rounds:
                messages.append({
                    "role": "user",
                    "content": (
                        "[REMINDER] You are writing academic content for a top-tier venue. "
                        "Focus on producing the final output now. Use the information "
                        "gathered from tools to write high-quality content. "
                        "Do NOT continue searching indefinitely."
                    ),
                })

        # Exceeded max rounds — do a final summary call without tools
        self.log(f"Exceeded {max_tool_rounds} tool rounds, forcing final answer")
        final_msg = await self._dispatcher.generate_with_tools(cfg, messages, tools=None)
        return final_msg.content or ""

    @abstractmethod
    async def run(self, **inputs: Any) -> dict[str, Any]:
        """Execute this agent's stage. Returns output data dict."""
        ...

    def log(self, msg: str) -> None:
        logger.info(f"[{self.stage.value}] {msg}")

    def save_log(self, filename: str, content: str) -> None:
        self.workspace.write_text(f"logs/{filename}", content)
