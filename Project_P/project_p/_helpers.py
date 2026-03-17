"""Shared utility functions for Project_P fixers."""
from __future__ import annotations

import re

# ── LaTeX text escaping ──────────────────────────────────────────────────────

_LATEX_TEXT_ESCAPES = {
    "&": r"\&",
    "%": r"\%",
    "#": r"\#",
    "_": r"\_",
    "^": r"\^{}",
    "~": r"\~{}",
}

_IDENTIFIER_COMMANDS = frozenset({
    "ref", "eqref", "autoref", "nameref", "pageref",
    "label",
    "cite", "citet", "citep", "citealp", "citeauthor", "citeyear",
    "bibliography", "bibliographystyle",
    "input", "include", "includegraphics",
    "url",
})


def escape_latex_text(text: str) -> str:
    """Escape LaTeX special characters in plain text.

    Preserves existing LaTeX commands and already-escaped sequences.
    Reference-type commands have their braced arguments preserved verbatim.
    """
    if not isinstance(text, str):
        text = str(text)

    result: list[str] = []
    i = 0
    in_math = False
    preservable_after_backslash = set(r"\$%#&_{}~^()[]")

    while i < len(text):
        ch = text[i]

        if ch == "\\":
            if i + 1 >= len(text):
                result.append(r"\textbackslash{}")
                break

            next_char = text[i + 1]
            if next_char.isalpha():
                j = i + 2
                while j < len(text) and text[j].isalpha():
                    j += 1
                cmd_name = text[i + 1:j]
                result.append(text[i:j])
                i = j

                if cmd_name in _IDENTIFIER_COMMANDS:
                    while i < len(text) and text[i] == '[':
                        close_bracket = text.find(']', i)
                        if close_bracket == -1:
                            break
                        result.append(text[i:close_bracket + 1])
                        i = close_bracket + 1
                    if i < len(text) and text[i] == '{':
                        depth = 0
                        k = i
                        while k < len(text):
                            if text[k] == '{':
                                depth += 1
                            elif text[k] == '}':
                                depth -= 1
                                if depth == 0:
                                    result.append(text[i:k + 1])
                                    i = k + 1
                                    break
                            k += 1
                        else:
                            result.append(text[i:])
                            i = len(text)
                continue

            if next_char in preservable_after_backslash:
                result.append(text[i:i + 2])
                i += 2
                if next_char in ('(', '['):
                    in_math = True
                elif next_char in (')', ']'):
                    in_math = False
                continue

            result.append(r"\textbackslash{}")
            i += 1
            continue

        if ch == "$":
            if in_math:
                result.append(ch)
                in_math = False
            else:
                has_closing = False
                j = i + 1
                esc = False
                while j < len(text):
                    if esc:
                        esc = False
                    elif text[j] == "\\":
                        esc = True
                    elif text[j] == "$":
                        has_closing = True
                        break
                    j += 1
                if has_closing:
                    result.append(ch)
                    in_math = True
                else:
                    result.append(r"\$")
            i += 1
            continue

        if in_math:
            result.append(ch)
            i += 1
            continue

        result.append(_LATEX_TEXT_ESCAPES.get(ch, ch))
        i += 1

    return "".join(result)


# ── Brace matching ───────────────────────────────────────────────────────────

def find_matching_brace(text: str, open_brace_index: int) -> int | None:
    """Find the matching closing brace for text[open_brace_index] == '{'."""
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


# ── Section finder ───────────────────────────────────────────────────────────

def find_section_end(content: str, section_heading: str) -> int | None:
    r"""Find the position at the END of a \section{heading}'s content.

    Returns position just before next \section or bibliography/end.
    Uses keyword matching for flexibility.
    """
    heading_lower = section_heading.lower()
    stem = heading_lower.rstrip('s')
    sec_pattern = re.compile(
        r'\\section\*?\{([^}]*(?:\{[^}]*(?:\{[^}]*\}[^}]*)*\}[^}]*)*)\}',
        re.IGNORECASE,
    )
    exact_match = None
    keyword_match = None
    for m in sec_pattern.finditer(content):
        sec_title = m.group(1).lower()
        if sec_title == heading_lower:
            exact_match = m
            break
        if keyword_match is None and (stem in sec_title or heading_lower in sec_title):
            keyword_match = m
    sec_m = exact_match or keyword_match
    if not sec_m:
        return None
    after = content[sec_m.end():]
    end_re = re.search(
        r'\\section\*?\{|\\bibliographystyle\{|\\bibliography\{'
        r'|\\begin\{thebibliography\}|\\end\{document\}',
        after,
    )
    if end_re:
        return sec_m.end() + end_re.start()
    return len(content)


# ── Bibliography position finder ─────────────────────────────────────────────

def find_bib_position(text: str) -> int:
    """Find the start position of bibliography commands. Returns len(text) if not found."""
    bib_pos = len(text)
    for pat in (r'\\bibliographystyle\{', r'\\bibliography\{', r'\\begin\{thebibliography\}'):
        m = re.search(pat, text)
        if m and m.start() < bib_pos:
            bib_pos = m.start()
    return bib_pos
