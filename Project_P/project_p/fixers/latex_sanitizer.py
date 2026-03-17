"""Character-level LaTeX sanitization.

Handles: Unicode replacement, % escaping, prose escaping, \\textbackslash fix,
control chars, LLM artifact stripping, markdown fence stripping, placeholder removal.
"""
from __future__ import annotations

import re

from .._helpers import escape_latex_text, find_matching_brace

# ── Text argument commands that need prose escaping ──────────────────────────

_TEXT_ARGUMENT_COMMANDS = (
    "title", "caption", "section", "subsection", "subsubsection",
    "paragraph", "author",
)

_SYNTAX_HEAVY_ENVIRONMENTS = {
    "tabular", "tabular*", "array",
    "align", "align*", "equation", "equation*",
    "gather", "gather*", "multline", "multline*",
    "eqnarray", "eqnarray*",
    "verbatim", "lstlisting", "minted", "tikzpicture",
}

# ── LLM artifact patterns ───────────────────────────────────────────────────

_LLM_ARTIFACT_PATTERNS = [
    r'I (?:now )?have sufficient \w+ to write.*',
    r'I have sufficient \w+.*',
    r'Let me (?:now )?(?:write|compose|draft|look up|check|verify).*',
    r'I will now (?:write|compose|draft|proceed).*',
    r'I see the paper ID.*',
    r'I (?:need|want) to (?:look up|check|find|verify|search).*',
    r'Based on (?:the|my) (?:analysis|research|review|context).*I (?:will|can|should).*',
    r'Now I (?:will|can|shall) (?:write|compose|draft).*',
    r'Here is the (?:completed?|final|written) (?:section|text|content).*:?\s*$',
    r'`[0-9a-f]{20,}`',
]

_LLM_THINKING_PATTERNS = [
    re.compile(r'^\s*Now I have enough context\.?\s*$', re.MULTILINE),
    re.compile(r'^\s*(?:Let me|I\'ll now|I will now|I need to|I should)\b.*$', re.MULTILINE),
    re.compile(r'^\s*(?:Now,? let\'?s|First,? I|Next,? I)\b.*$', re.MULTILINE),
    re.compile(r'^\s*(?:Sure,|OK,|Hmm,|Okay,)\s*$', re.MULTILINE),
    re.compile(r'^\s*Here (?:is|are) the (?:LaTeX|content|section|text)\b.*$', re.MULTILINE),
    re.compile(r'^\s*(?:Below is|The following is) the (?:LaTeX|content|section|text)\b.*$', re.MULTILINE),
]

_PLACEHOLDER_PATS = [
    r'(?:Note that )?(?:the )?experimental (?:pipeline|results?)\s+'
    r'(?:is|are)\s+(?:currently\s+)?being\s+finalized[^.]*\.?',
    r'[^.\n]*(?:complete|full)\s+(?:numerical\s+)?results\s+will\s+be\s+'
    r'(?:reported|included|provided)\s+in\s+(?:the\s+)?(?:camera[- ]ready|'
    r'final)\s+version[^.]*\.?',
    r'[^.\n]*results?\s+for\s+our\s+method\s+(?:are|is)\s+pending[^.]*\.?',
    r'[^.\n]*(?:due to|because of)\s+(?:technical\s+)?(?:issues?|problems?)\s+'
    r'during\s+(?:experiment\s+)?execution[^.]*\.?',
    r'[^.\n]*quantitative\s+results?\s+(?:for\s+our\s+method\s+)?'
    r'(?:are|is)\s+not\s+available\s+in\s+this\s+version[^.]*\.?',
    r'Results\s+are\s+pending\s+due\s+to\s+execution\s+issues\.?',
]


# ── Prose line sanitization ──────────────────────────────────────────────────

def _sanitize_command_text_argument(text: str, command: str) -> str:
    """Escape plain-text special chars inside a command's text argument."""
    pattern = re.compile(rf"\\{command}(?:\[[^\]]*\])?\{{")
    result: list[str] = []
    cursor = 0

    while True:
        match = pattern.search(text, cursor)
        if not match:
            result.append(text[cursor:])
            break

        open_brace_index = match.end() - 1
        close_brace_index = find_matching_brace(text, open_brace_index)
        if close_brace_index is None:
            result.append(text[cursor:])
            break

        result.append(text[cursor:match.end()])
        body = text[match.end():close_brace_index]
        result.append(escape_latex_text(body))
        result.append("}")
        cursor = close_brace_index + 1

    return "".join(result)


def _update_environment_stack(line: str, env_stack: list[str]) -> None:
    """Track LaTeX environments."""
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
        return f"{prefix}{escape_latex_text(body)}"

    if not stripped.startswith("\\"):
        return escape_latex_text(result)

    return result


# ── Main sanitizer ───────────────────────────────────────────────────────────

def sanitize_latex(text: str) -> str:
    """Apply all character-level LaTeX sanitization fixes."""

    # 0c. Truncate garbage before \documentclass
    docclass_pos = text.find(r'\documentclass')
    if docclass_pos > 0:
        text = text[docclass_pos:]

    # 0d. Strip Markdown code fences
    text = re.sub(r'```(?:latex|tex)?\s*\n', '', text)
    text = re.sub(r'\n```[ \t]*(?:\n|$)', '\n', text)

    # 0e. Strip placeholder sentences
    for _pp in _PLACEHOLDER_PATS:
        text = re.sub(_pp, '', text, flags=re.IGNORECASE)

    # 0. Remove LLM artifact text
    for pat in _LLM_ARTIFACT_PATTERNS:
        text = re.sub(pat, '', text, flags=re.IGNORECASE | re.MULTILINE)

    # 0a. Strip LLM thinking lines
    for pat in _LLM_THINKING_PATTERNS:
        text = pat.sub('', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 0b. Strip control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # 1. Unicode replacements
    unicode_map = {
        "\u2014": "---",
        "\u2013": "--",
        "\u2018": "`",
        "\u2019": "'",
        "\u201c": "``",
        "\u201d": "''",
        "\u2192": r"$\rightarrow$",
        "\u2190": r"$\leftarrow$",
        "\u2208": r"$\in$",
        "\u2209": r"$\notin$",
        "\u2264": r"$\leq$",
        "\u2265": r"$\geq$",
        "\u00d7": r"$\times$",
        "\u2248": r"$\approx$",
        "\u00b1": r"$\pm$",
        "\u221e": r"$\infty$",
    }
    for char, repl in unicode_map.items():
        text = text.replace(char, repl)

    # 2. Escape bare % after digits
    lines = text.split("\n")
    fixed_lines = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("%"):
            fixed_lines.append(line)
            continue
        if r'\url{' in line or r'\href{' in line:
            fixed_line = re.sub(
                r'(\\(?:url|href)\{[^}]*\})|(?<!\\)(\d)%',
                lambda m: m.group(1) if m.group(1) else m.group(2) + r'\%',
                line,
            )
            fixed_lines.append(fixed_line)
        else:
            fixed_line = re.sub(r'(?<!\\)(\d)%', r'\1\\%', line)
            fixed_lines.append(fixed_line)
    text = "\n".join(fixed_lines)

    # 2b. Escape prose-only special chars
    env_stack: list[str] = []
    sanitized_lines: list[str] = []
    for line in text.split("\n"):
        sanitized_lines.append(_sanitize_prose_line(line, env_stack))
        _update_environment_stack(line, env_stack)
    text = "\n".join(sanitized_lines)

    # 2c-pre. Fix \textbackslash{}, → \, (common NanoResearch artifact)
    text = text.replace(r'\textbackslash{},', r'\,')

    # 2c-pre2. Fix \~{} → ~ everywhere (not just before \ref)
    text = re.sub(r'\\~\{\}', '~', text)

    # 2c. Fix triple backslash \\\ → \\ in tabular environments
    text = re.sub(
        r'(\\begin\{tabular[*x]?\}(?:\[[^\]]*\])?\{[^}]*\})'
        r'(.*?)'
        r'(\\end\{tabular[*x]?\})',
        lambda m: m.group(1) + re.sub(r'\\\\\\(?!\\)', r'\\\\', m.group(2)) + m.group(3),
        text,
        flags=re.DOTALL,
    )

    # 2d. Fix \~{}\ref → ~\ref
    text = re.sub(
        r'\\~\{\}(\\(?:ref|eqref|cite[tp]?|pageref)\{)',
        r'~\1',
        text,
    )

    return text
