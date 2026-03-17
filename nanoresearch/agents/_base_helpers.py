"""Base agent helper functions — JSON parsing, truncation, context management.

All free functions used by BaseResearchAgent are defined here to keep base.py
under 500 lines.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

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
# Approximate token limit for proactive compaction (chars ~ tokens * 4)
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
    # BUG-12 fix: handle multimodal messages where content is a list
    # of dicts (e.g. [{"type":"text","text":"..."}, {"type":"image_url",...}]).
    def _content_len(content: Any) -> int:
        if isinstance(content, str):
            return len(content)
        if isinstance(content, list):
            return sum(
                len(item.get("text", ""))
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            )
        return 0
    total_chars = sum(_content_len(m.get("content")) for m in messages)
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
                + f"\n[compacted: {len(content)} chars -> 400]\n"
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
    if last_char not in '.!?}])\'\"':
        # Check if it looks like a code/latex block (ok to end with command)
        if not re.search(r'\\[a-zA-Z]+[\{\[]?$', text[-30:]):
            return True
    return False
