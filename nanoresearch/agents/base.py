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

# Known LaTeX command prefixes — used to distinguish \textbf from JSON \t escape
_LATEX_CMD_PREFIXES = frozenset([
    "cite", "textbf", "textit", "frac", "ref", "label", "sqrt", "sum",
    "int", "alpha", "beta", "gamma", "delta", "epsilon", "theta", "lambda",
    "sigma", "omega", "text", "math", "begin", "end", "item", "section",
    "subsection", "paragraph", "emph", "url", "href", "footnote",
    "caption", "includegraphics", "usepackage", "newcommand",
])

# ---- Tool result management (OpenClaw-inspired patterns) ----

_MAX_TOOL_RESULT_CHARS = 6000
_HEAD_CHARS = 2000
_TAIL_CHARS = 1500
# Approximate token limit for proactive compaction (chars ≈ tokens * 4)
_CONTEXT_COMPACT_THRESHOLD_CHARS = 100_000
_PROTECTED_TAIL_TURNS = 6  # keep last N messages intact during compaction


def _truncate_tool_result(text: str) -> str:
    """Head/tail truncation for large tool results.

    Keeps the first 2000 and last 1500 chars, truncating the middle.
    Prevents large search results from flooding the context window.
    """
    if len(text) <= _MAX_TOOL_RESULT_CHARS:
        return text
    mid_len = len(text) - _HEAD_CHARS - _TAIL_CHARS
    return (
        text[:_HEAD_CHARS]
        + f"\n\n... [{mid_len} chars truncated] ...\n\n"
        + text[-_TAIL_CHARS:]
    )


def _compact_messages_if_needed(messages: list[dict]) -> None:
    """Proactive context compaction: trim old tool results when context grows large.

    Inspired by OpenClaw's cache-aware pruning: when total content exceeds
    the threshold, truncate tool results in older messages (keeping the last
    N turns intact). Modifies messages in-place.
    """
    total_chars = sum(len(m.get("content", "") or "") for m in messages)
    if total_chars < _CONTEXT_COMPACT_THRESHOLD_CHARS:
        return

    # Compact older tool results (skip system prompt + last N turns)
    protect_start = max(1, len(messages) - _PROTECTED_TAIL_TURNS)
    compacted = 0
    for i in range(1, protect_start):
        msg = messages[i]
        content = msg.get("content", "") or ""
        if msg.get("role") == "tool" and len(content) > 500:
            # Keep first 200 + last 200 chars
            msg["content"] = (
                content[:200]
                + f"\n[compacted: {len(content)} chars → 400]\n"
                + content[-200:]
            )
            compacted += 1

    if compacted:
        logger.info("Proactive compaction: trimmed %d old tool results", compacted)


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
                # Check if this is actually a LaTeX command (e.g. \textbf, \boldsymbol)
                # rather than a JSON escape (\t, \n, \b, \f, \r, \u)
                cmd_match = re.match(r'([a-zA-Z]+)', text[i + 1:])
                if cmd_match and (
                    cmd_match.group(1) in _LATEX_CMD_PREFIXES
                    or len(cmd_match.group(1)) > 1
                ):
                    # LaTeX command (known set, or 2+ alpha chars) — double-escape
                    result.append('\\\\')
                    i += 1  # re-process next_char as normal
                else:
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


def _extract_balanced_json_segment(text: str, start: int) -> str | None:
    """Extract a balanced JSON object/array substring starting at ``start``."""
    if start < 0 or start >= len(text) or text[start] not in "{[":
        return None

    stack: list[str] = []
    in_string = False
    escape = False

    for index in range(start, len(text)):
        ch = text[index]
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch == '}' and stack and stack[-1] == '{':
            stack.pop()
        elif ch == ']' and stack and stack[-1] == '[':
            stack.pop()
        elif ch in '}]':
            return None
        if not stack:
            return text[start:index + 1].strip()
    return None


def _extract_json_candidates(text: str) -> list[str]:
    """Return likely JSON substrings from raw LLM output."""
    stripped = text.strip()
    if not stripped:
        return []

    candidates: list[str] = []

    def _add(candidate: str | None) -> None:
        if candidate is None:
            return
        value = candidate.strip()
        if value and value not in candidates:
            candidates.append(value)

    _add(stripped)

    for match in re.finditer(r"```(?:json|JSON|javascript|js)?\s*([\s\S]*?)```", stripped):
        block = match.group(1).strip()
        if block.startswith("{") or block.startswith("["):
            _add(block)

    start_count = 0
    for index, ch in enumerate(stripped):
        if ch not in "{[":
            continue
        start_count += 1
        if start_count > 20:
            break
        _add(_extract_balanced_json_segment(stripped, index))
        tail = stripped[index:].strip()
        if tail.startswith("{") or tail.startswith("["):
            _add(tail)

    return candidates


def _scan_json_fragment(
    text: str,
) -> tuple[list[tuple[str, int]], bool, bool, int | None]:
    """Scan a possibly truncated JSON fragment."""
    stack: list[tuple[str, int]] = []
    in_string = False
    escape = False
    last_comma_index: int | None = None

    for index, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            stack.append((ch, index))
        elif ch == '}' and stack and stack[-1][0] == '{':
            stack.pop()
        elif ch == ']' and stack and stack[-1][0] == '[':
            stack.pop()
        elif ch == ',':
            last_comma_index = index

    return stack, in_string, escape, last_comma_index


def _close_json_fragment(text: str) -> str:
    """Close open string/bracket state for a truncated JSON fragment."""
    candidate = re.sub(r',\s*$', '', text.strip())
    stack, in_string, escape, _ = _scan_json_fragment(candidate)

    if escape and in_string:
        candidate += '\\'
        stack, in_string, _, _ = _scan_json_fragment(candidate)
    if in_string:
        candidate += '"'
        stack, _, _, _ = _scan_json_fragment(candidate)

    closers = {'[': ']', '{': '}'}
    for opener, _ in reversed(stack):
        candidate += closers[opener]

    return re.sub(r',\s*([}\]])', r'\1', candidate)


def _trim_json_fragment(text: str) -> str | None:
    """Trim the last incomplete JSON element from a fragment."""
    candidate = text.rstrip()
    if not candidate:
        return None

    stack, _, _, last_comma_index = _scan_json_fragment(candidate)
    if last_comma_index is not None:
        return candidate[:last_comma_index]
    if stack:
        return candidate[:stack[-1][1] + 1]
    return None


def _repair_truncated_json(text: str) -> str | None:
    """Attempt to repair JSON that was truncated by output token limit.

    Strategy: close any open strings, arrays, and objects from the end.
    """
    # Only try if it looks like it starts as valid JSON
    stripped = text.strip()
    if not stripped or stripped[0] not in ('{', '['):
        return None

    candidate = stripped
    for _ in range(12):
        repaired = _close_json_fragment(candidate)
        try:
            json.loads(repaired, strict=False)
            return repaired
        except json.JSONDecodeError:
            trimmed = _trim_json_fragment(candidate)
            if not trimmed or trimmed == candidate:
                return repaired
            candidate = trimmed
    return _close_json_fragment(candidate)


def _json_error_msg(text: str) -> str:
    """Get JSON parse error message for diagnostics."""
    try:
        json.loads(text)
        return "no error"
    except json.JSONDecodeError as exc:
        return str(exc)


def detect_truncation(text: str) -> bool:
    """Detect if LLM output was likely truncated mid-generation.

    Checks for unbalanced braces, incomplete environments, and
    sentences ending abruptly without terminal punctuation.
    """
    if not text or len(text) < 20:
        return True
    text = text.rstrip()
    # Unbalanced JSON braces
    if text.count('{') > text.count('}') + 2:
        return True
    # Unbalanced LaTeX environments
    begins = len(re.findall(r'\\begin\{', text))
    ends = len(re.findall(r'\\end\{', text))
    if begins > ends + 1:
        return True
    # Ends mid-word or mid-sentence (no terminal punctuation)
    last_char = text[-1]
    if last_char not in '.!?}])\'"–—':
        # Check if it looks like a code/latex block (ok to end with command)
        if not re.search(r'\\[a-zA-Z]+[\{\[]?$', text[-30:]):
            return True
    return False


class BaseResearchAgent(ABC):
    """Abstract base class for all NanoResearch agents."""

    stage: PipelineStage  # subclass must set this

    def __init__(self, workspace: Workspace, config: ResearchConfig) -> None:
        self.workspace = workspace
        self.config = config
        self._dispatcher = ModelDispatcher(config)

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

    async def generate_json_validated(
        self,
        system_prompt: str,
        user_prompt: str,
        model_class: type,
        stage_override: StageModelConfig | None = None,
    ) -> Any:
        """Call LLM, parse as JSON, and validate against a Pydantic model.

        On validation failure, feeds the error back to the LLM for one retry.
        Returns a validated Pydantic model instance.
        """
        raw_dict = await self.generate_json(
            system_prompt, user_prompt, stage_override=stage_override,
        )
        try:
            return model_class.model_validate(raw_dict)
        except Exception as first_exc:
            # Single retry: feed validation error back to LLM
            self.log(f"  JSON schema validation failed: {first_exc}, retrying...")
            retry_prompt = (
                f"Your previous JSON response had validation errors:\n"
                f"{first_exc}\n\n"
                f"Original request:\n{user_prompt}\n\n"
                f"Fix the JSON to match the required schema and try again."
            )
            try:
                raw_dict = await self.generate_json(
                    system_prompt, retry_prompt, stage_override=stage_override,
                )
                return model_class.model_validate(raw_dict)
            except Exception as retry_exc:
                logger.error(
                    "JSON validation failed after retry: %s", retry_exc,
                )
                raise LLMError(
                    f"JSON schema validation failed after retry: {retry_exc}"
                ) from retry_exc

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
        """Run a ReAct loop: let the LLM call tools until it produces text.

        Args:
            system_prompt: System message.
            user_prompt: Initial user message.
            tools: A ``ToolRegistry`` instance.
            max_tool_rounds: Max tool-call round-trips before forcing a final answer.
            stage_override: Optional stage config override.
            reminder_text: Custom periodic reminder (default: academic writing focus).
            reminder_interval: Inject reminder every N rounds (default: 3).

        Returns:
            Final assistant text content.
        """
        cfg = stage_override if stage_override is not None else self.stage_config
        openai_tools = tools.to_openai_tools()

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Track repeated failures to avoid infinite retry loops (OpenClaw pattern)
        _failure_counts: dict[str, int] = {}  # "name|args_hash|error_sig" -> count
        _MAX_IDENTICAL_FAILURES = 2

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
                    error_str = f"Error: {type(e).__name__}: {e}"
                    # Classify error: rate limits and server errors are retryable,
                    # but repeated identical failures get tagged [NON-RETRYABLE]
                    error_sig = type(e).__name__
                    try:
                        args_hash = hash(json.dumps(args, sort_keys=True, default=str))
                    except (TypeError, ValueError):
                        args_hash = hash(str(sorted(args.items())) if isinstance(args, dict) else str(args))
                    fail_key = f"{name}|{args_hash}|{error_sig}"
                    _failure_counts[fail_key] = _failure_counts.get(fail_key, 0) + 1
                    if _failure_counts[fail_key] >= _MAX_IDENTICAL_FAILURES:
                        error_str = (
                            f"[NON-RETRYABLE] {error_str} — "
                            f"This exact call has failed {_failure_counts[fail_key]} times. "
                            f"Do NOT retry with the same arguments. Try a different query or approach."
                        )
                    result_str = error_str

                # Head/tail truncation for large tool results (OpenClaw pruner pattern)
                result_content = _truncate_tool_result(result_str)
                return {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_content,
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

            # Proactive context compaction: if messages are getting large,
            # trim older tool results to keep context manageable
            _compact_messages_if_needed(messages)

            # Inject system reminder periodically to prevent instruction drift
            if (round_idx + 1) % reminder_interval == 0 and round_idx + 1 <= max_tool_rounds:
                _reminder = reminder_text or (
                    "[REMINDER] You are writing academic content for a top-tier venue. "
                    "Focus on producing the final output now. Use the information "
                    "gathered from tools to write high-quality content. "
                    "Do NOT continue searching indefinitely."
                )
                messages.append({"role": "user", "content": _reminder})

        # Exceeded max rounds — do a final summary call without tools
        self.log(f"Exceeded {max_tool_rounds} tool rounds, forcing final answer")
        final_msg = await self._dispatcher.generate_with_tools(cfg, messages, tools=None)
        # Guard: if LLM still returns tool_calls in summary round, force text extraction
        if hasattr(final_msg, 'tool_calls') and final_msg.tool_calls:
            return final_msg.content or "Agent completed but produced no text summary."
        return final_msg.content or ""

    @abstractmethod
    async def run(self, **inputs: Any) -> dict[str, Any]:
        """Execute this agent's stage. Returns output data dict."""
        ...

    def log(self, msg: str) -> None:
        logger.info(f"[{self.stage.value}] {msg}")

    def save_log(self, filename: str, content: str) -> None:
        self.workspace.write_text(f"logs/{filename}", content)
