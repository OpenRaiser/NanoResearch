"""Internal helpers for LaTeX sanitization -- free functions used by latex_assembler."""
from __future__ import annotations

import re

from . import _escape_latex_text

_TEXT_ARGUMENT_COMMANDS = (
    "title",
    "caption",
    "section",
    "subsection",
    "subsubsection",
    "paragraph",
    "author",
)

_SYNTAX_HEAVY_ENVIRONMENTS = {
    "tabular",
    "tabular*",
    "array",
    "align",
    "align*",
    "equation",
    "equation*",
    "gather",
    "gather*",
    "multline",
    "multline*",
    "eqnarray",
    "eqnarray*",
    "verbatim",
    "lstlisting",
    "minted",
    "tikzpicture",
}


def _find_matching_brace(text: str, open_brace_index: int) -> int | None:
    """Find the matching closing brace for ``text[open_brace_index] == '{'``."""
    if open_brace_index < 0 or open_brace_index >= len(text) or text[open_brace_index] != "{":
        return None

    depth = 0
    escape = False
    for index in range(open_brace_index, len(text)):
        ch = text[index]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return index
    return None


def _sanitize_command_text_argument(text: str, command: str) -> str:
    """Escape plain-text special chars inside a command's main text argument."""
    pattern = re.compile(rf"\\{command}(?:\[[^\]]*\])?\{{")
    result: list[str] = []
    cursor = 0

    while True:
        match = pattern.search(text, cursor)
        if not match:
            result.append(text[cursor:])
            break

        open_brace_index = match.end() - 1
        close_brace_index = _find_matching_brace(text, open_brace_index)
        if close_brace_index is None:
            result.append(text[cursor:])
            break

        result.append(text[cursor:match.end()])
        body = text[match.end():close_brace_index]
        result.append(_escape_latex_text(body))
        result.append("}")
        cursor = close_brace_index + 1

    return "".join(result)


def _update_environment_stack(line: str, env_stack: list[str]) -> None:
    """Track LaTeX environments while scanning document lines."""
    for match in re.finditer(r"\\begin\{([^}]+)\}", line):
        env_stack.append(match.group(1))
    for match in re.finditer(r"\\end\{([^}]+)\}", line):
        env_name = match.group(1)
        for idx in range(len(env_stack) - 1, -1, -1):
            if env_stack[idx] == env_name:
                del env_stack[idx]
                break


def _sanitize_prose_line(line: str, env_stack: list[str]) -> str:
    """Escape unsafe prose characters without touching syntax-heavy blocks."""
    result = line
    for command in _TEXT_ARGUMENT_COMMANDS:
        result = _sanitize_command_text_argument(result, command)

    stripped = result.lstrip()
    if not stripped or stripped.startswith("%"):
        return result
    if any(env in _SYNTAX_HEAVY_ENVIRONMENTS for env in env_stack):
        return result

    item_match = re.match(r"^(\s*\\item(?:\[[^\]]*\])?\s*)(.*)$", result)
    if item_match:
        prefix, body = item_match.groups()
        return f"{prefix}{_escape_latex_text(body)}"

    if not stripped.startswith("\\"):
        return _escape_latex_text(result)

    return result


# ---------------------------------------------------------------------------
# LLM thinking leak filter
# ---------------------------------------------------------------------------

# Patterns that indicate LLM meta-commentary leaked into paper content.
# Each pattern is anchored to a standalone line (possibly preceded by whitespace).
_LLM_THINKING_PATTERNS = [
    # Direct self-reference / chain-of-thought leaks
    re.compile(r'^\s*(?:Now )?I have enough context\b.*$', re.MULTILINE),
    re.compile(r'^\s*(?:Let me|I\'ll now|I will now|I need to|I should)\b.*$', re.MULTILINE),
    re.compile(r'^\s*(?:Looking at|Based on this|Here is|Sure,|OK,|Hmm,|Okay,)\b.*$', re.MULTILINE),
    re.compile(r'^\s*(?:Now,? let\'?s|First,? I|Next,? I)\b.*$', re.MULTILINE),
    # "I have/I'll write/I can" self-narration
    re.compile(r'^\s*I (?:have|will|can|shall|am going to) (?:write|draft|compose|generate|create|produce)\b.*$', re.MULTILINE),
    # Common LLM preamble / postamble
    re.compile(r'^\s*Here (?:is|are) the (?:LaTeX|content|section|text|revised|updated|complete)\b.*$', re.MULTILINE),
    re.compile(r'^\s*(?:Below is|The following is)\b.*$', re.MULTILINE),
    # "This section will/should" meta-narration (not part of paper prose)
    re.compile(r'^\s*This section (?:will|should|needs to|is going to) (?:discuss|describe|present|cover|outline)\b.*$', re.MULTILINE),
]


def _strip_llm_thinking(text: str) -> str:
    """Remove LLM meta-commentary lines from section content.

    Only removes lines that are *standalone* (i.e. the entire line matches
    a thinking pattern).  Does not touch lines embedded in paragraphs.
    """
    for pat in _LLM_THINKING_PATTERNS:
        text = pat.sub('', text)
    # Collapse runs of 3+ blank lines into 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()
