"""Shared 2-level LaTeX fix pipeline.

Level 1: Deterministic fixes (Unicode, packages, preamble, env matching) — no LLM.
Level 2: LLM search-replace using JSON [{"old", "new"}] format.

Both writing.py and review.py import from this module.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Callable

logger = logging.getLogger(__name__)

# ── Unicode map (common compilation killers) ──────────────────────────────

UNICODE_REPLACEMENTS: dict[str, str] = {
    "\u2018": "`", "\u2019": "'", "\u201c": "``", "\u201d": "''",
    "\u2014": "---", "\u2013": "--", "\u2026": "\\ldots{}",
    "\u00d7": "$\\times$", "\u2264": "$\\leq$", "\u2265": "$\\geq$",
    "\u2260": "$\\neq$", "\u221e": "$\\infty$", "\u03b1": "$\\alpha$",
    "\u03b2": "$\\beta$", "\u03b3": "$\\gamma$", "\u03b4": "$\\delta$",
    "\u03bb": "$\\lambda$", "\u03c0": "$\\pi$", "\u03c3": "$\\sigma$",
    "\u2192": "$\\rightarrow$", "\u2190": "$\\leftarrow$",
    "\u00b1": "$\\pm$", "\u2248": "$\\approx$",
    "\u00e9": "{\\'e}", "\u00e8": "{\\`e}",
    "\u00f6": '{\\"o}', "\u00fc": '{\\"u}', "\u00e4": '{\\"a}',
}

# ── Package-to-command map ────────────────────────────────────────────────

_PACKAGE_FIXES: dict[str, tuple[str, str]] = {
    "\\multirow":    ("multirow",  "\\usepackage{multirow}"),
    "\\toprule":     ("booktabs",  "\\usepackage{booktabs}"),
    "\\midrule":     ("booktabs",  "\\usepackage{booktabs}"),
    "\\bottomrule":  ("booktabs",  "\\usepackage{booktabs}"),
    "\\FloatBarrier": ("placeins", "\\usepackage{placeins}"),
    "\\url{":        ("url",       "\\usepackage{url}"),
    "\\href{":       ("hyperref",  "\\usepackage{hyperref}"),
    "\\textcolor":   ("xcolor",    "\\usepackage{xcolor}"),
    "\\cellcolor":   ("xcolor",    "\\usepackage[table]{xcolor}"),
}

# ── Environment list for mismatch detection ───────────────────────────────

_ENV_NAMES = ("figure", "figure*", "table", "table*", "align", "equation",
              "tabular", "tabular*", "abstract")


# ============================================================================
#  Level 1: Deterministic fixes (no LLM)
# ============================================================================

def deterministic_fix(
    tex_source: str,
    error_log: str = "",
    error_line: int | None = None,
    log_fn: Callable[[str], None] | None = None,
) -> str | None:
    """Apply deterministic LaTeX fixes. Returns modified tex or None if unchanged.

    Args:
        tex_source: LaTeX source code.
        error_log: Compilation error output (used to decide which fixes to apply).
        error_line: Line number from error (1-based), if known.
        log_fn: Optional logging callback (e.g. ``self.log``).

    Returns:
        Modified tex string if any changes were made, None otherwise.
    """
    def _log(msg: str) -> None:
        if log_fn:
            log_fn(msg)

    modified = tex_source
    error_lower = error_log.lower()

    # 1. Garbage before \documentclass
    dc_match = re.search(r'\\documentclass[\[{]', modified)
    if dc_match and dc_match.start() > 0:
        prefix = modified[:dc_match.start()]
        if re.search(r'[a-zA-Z]', prefix.replace('%', '')):
            modified = modified[dc_match.start():]
            _log("  Deterministic: removed junk before \\documentclass")

    # 2. Missing \end{document}
    if "\\begin{document}" in modified and "\\end{document}" not in modified:
        modified += "\n\\end{document}\n"
        _log("  Deterministic: added missing \\end{document}")

    # 3. Unicode replacements (conditional on error type for efficiency)
    if (not error_lower
            or "invalid" in error_lower
            or "character" in error_lower
            or "unicode" in error_lower):
        for char, repl in UNICODE_REPLACEMENTS.items():
            if char in modified:
                modified = modified.replace(char, repl)
                _log(f"  Deterministic: replaced U+{ord(char):04X}")

        # Control characters (0x00-0x1F except \t \n \r)
        for code in range(0x20):
            if code in (0x09, 0x0A, 0x0D):  # tab, newline, CR
                continue
            char = chr(code)
            if char in modified:
                modified = modified.replace(char, '')
                _log(f"  Deterministic: removed control char 0x{code:02X}")

    # 4. Missing packages — check for known undefined control sequences
    preamble_end = modified.find("\\begin{document}")
    if preamble_end > 0:
        preamble = modified[:preamble_end]
        if "undefined control sequence" in error_lower:
            for cmd, (pkg, use_line) in _PACKAGE_FIXES.items():
                if cmd in modified and pkg not in preamble:
                    modified = (modified[:preamble_end]
                                + use_line + "\n"
                                + modified[preamble_end:])
                    preamble_end += len(use_line) + 1
                    preamble = modified[:preamble_end]
                    _log(f"  Deterministic: added {use_line}")

    # 5a. Duplicate consecutive \begin{env}\n\begin{env} or \end{env}\n\end{env}
    _ALL_ENVS = (*_ENV_NAMES, "document", "itemize", "enumerate")
    for env in _ALL_ENVS:
        dup_begin = f"\\begin{{{env}}}\n\\begin{{{env}}}"
        while dup_begin in modified:
            modified = modified.replace(dup_begin, f"\\begin{{{env}}}", 1)
            _log(f"  Deterministic: removed duplicate \\begin{{{env}}}")
        dup_end = f"\\end{{{env}}}\n\\end{{{env}}}"
        while dup_end in modified:
            modified = modified.replace(dup_end, f"\\end{{{env}}}", 1)
            _log(f"  Deterministic: removed duplicate \\end{{{env}}}")

    # 5b. Escaped underscores inside \ref{}, \eqref{}, \label{} etc.
    # e.g. \ref{fig:framework\_overview} → \ref{fig:framework_overview}
    def _unescape_identifier_args(src: str) -> str:
        _ID_CMD_PATTERN = re.compile(
            r'(\\(?:ref|eqref|autoref|nameref|pageref|label'
            r'|cite[tp]?|citealp|citeauthor|citeyear))\{([^}]*)\}'
        )
        def _fix(m: re.Match) -> str:
            cmd = m.group(1)
            arg = m.group(2).replace('\\_', '_')
            return f"{cmd}{{{arg}}}"
        return _ID_CMD_PATTERN.sub(_fix, src)

    prev = modified
    modified = _unescape_identifier_args(modified)
    if modified != prev:
        _log("  Deterministic: unescaped underscores in \\ref/\\label/\\cite arguments")

    # 5c. Mismatched environments at the error line
    modified_lines = modified.split('\n')
    if error_line and error_line <= len(modified_lines):
        err_line_text = modified_lines[error_line - 1]
        for env in _ENV_NAMES:
            begin_tag = f"\\begin{{{env}}}"
            end_tag = f"\\end{{{env}}}"
            if begin_tag in err_line_text or end_tag in err_line_text:
                begin_count = modified.count(begin_tag)
                end_count = modified.count(end_tag)
                if begin_count > end_count:
                    end_doc = modified.rfind("\\end{document}")
                    if end_doc > 0:
                        modified = (modified[:end_doc]
                                    + end_tag + "\n\n"
                                    + modified[end_doc:])
                        _log(f"  Deterministic: added missing {end_tag}")
                elif end_count > begin_count:
                    if (error_line - 1 < len(modified_lines)
                            and end_tag in modified_lines[error_line - 1]):
                        modified_lines[error_line - 1] = modified_lines[
                            error_line - 1
                        ].replace(end_tag, '', 1)
                        modified = '\n'.join(modified_lines)
                        _log(f"  Deterministic: removed extra {end_tag}"
                             f" at line {error_line}")

    return modified if modified != tex_source else None


# ============================================================================
#  Error classification
# ============================================================================

def classify_error(error_lower: str) -> str:
    """Classify LaTeX error for targeted LLM guidance.

    Args:
        error_lower: Lowercased error log text.

    Returns:
        Human-readable hint string, or empty string if unclassified.
    """
    if ("invalid character" in error_lower
            or "unicode" in error_lower
            or "character" in error_lower):
        return (
            "Likely cause: Unicode characters (em-dash, en-dash, smart quotes, "
            "non-ASCII). Replace with LaTeX equivalents: --- for em-dash, -- for "
            "en-dash, standard quotes, \\alpha etc."
        )
    if "undefined control sequence" in error_lower:
        return "Likely cause: Typo in command name or missing \\usepackage."
    if "ended by" in error_lower:
        return (
            "Likely cause: \\begin{X} ended by \\end{Y} — typo in environment name. "
            "Check for misspelled environment names like 'equaton' vs 'equation'."
        )
    if "missing" in error_lower and ("begin" in error_lower
                                      or "end" in error_lower):
        return "Likely cause: Mismatched \\begin/\\end environments."
    if "missing \\begin{document}" in error_lower:
        return "Likely cause: Non-LaTeX content before \\begin{document}."
    if "missing $" in error_lower:
        return "Likely cause: Math symbols used outside math mode. Wrap with $...$."
    if ("extra }" in error_lower
            or "missing {" in error_lower
            or "missing }" in error_lower):
        return (
            "Likely cause: Mismatched braces { }. Look for an extra } or missing { "
            "near the error line. Count braces carefully in math formulas like \\frac{}{}."
        )
    if "extra alignment" in error_lower or "misplaced" in error_lower:
        return ("Likely cause: & used outside tabular, or wrong number of "
                "columns in table.")
    return ""


# ============================================================================
#  Level 2 helpers: snippet building, JSON parsing, edit application
# ============================================================================

def build_error_snippet(
    tex_lines: list[str],
    error_line: int | None,
) -> tuple[int, int, str]:
    """Build a numbered code snippet around the error line.

    Expands the window to include complete LaTeX environments
    (figure, table, align, equation, tabular).

    Returns:
        (win_start, win_end, numbered_snippet)
    """
    if error_line and error_line <= len(tex_lines):
        err_idx = error_line - 1
        win_start = max(0, err_idx - 20)
        win_end = min(len(tex_lines), err_idx + 20 + 1)

        # Expand to environment boundaries
        _env_begin = re.compile(
            r'\\begin\{(?:figure|table|align|equation|tabular)'
        )
        _env_end = re.compile(
            r'\\end\{(?:figure|table|align|equation|tabular)'
        )
        env_stack: list[int] = []
        for i in range(win_start, win_end):
            if _env_begin.search(tex_lines[i]):
                env_stack.append(i)
            if _env_end.search(tex_lines[i]):
                if env_stack:
                    env_stack.pop()
        if env_stack:
            for i in range(win_end, min(len(tex_lines), win_end + 30)):
                if _env_end.search(tex_lines[i]):
                    win_end = i + 1
                    env_stack.pop()
                    if not env_stack:
                        break
            for i in range(win_start - 1, max(-1, win_start - 30), -1):
                if i < 0:
                    break
                if _env_begin.search(tex_lines[i]):
                    win_start = i
                    break
    else:
        # No line number — show preamble + first 40 lines
        win_start = 0
        win_end = min(len(tex_lines), 40)

    snippet_lines = tex_lines[win_start:win_end]
    numbered = "\n".join(
        f"{win_start + i + 1:>5}: {line}"
        for i, line in enumerate(snippet_lines)
    )
    return win_start, win_end, numbered


def parse_edit_json(raw: str) -> list[dict]:
    """Parse LLM output as JSON array of {"old": ..., "new": ...} edits."""
    raw = raw.strip()

    # Strip markdown fences
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()

    # Try direct JSON parse
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [e for e in data
                    if isinstance(e, dict) and "old" in e and "new" in e]
        if isinstance(data, dict) and "old" in data and "new" in data:
            return [data]
    except json.JSONDecodeError:
        pass

    # Try to extract JSON array from surrounding text
    arr_match = re.search(r'\[.*\]', raw, re.DOTALL)
    if arr_match:
        try:
            data = json.loads(arr_match.group(0))
            if isinstance(data, list):
                return [e for e in data
                        if isinstance(e, dict) and "old" in e and "new" in e]
        except json.JSONDecodeError:
            pass

    return []


def apply_edits(
    tex_source: str,
    edits: list[dict],
    log_fn: Callable[[str], None] | None = None,
    search_window: tuple[int, int] | None = None,
) -> str | None:
    """Apply search-replace edits to LaTeX source.

    Each edit is {"old": "exact text", "new": "replacement"}.
    Uses whitespace-normalized matching as fallback.

    Returns:
        Modified tex if any edits applied, None otherwise.
    """
    def _log(msg: str) -> None:
        if log_fn:
            log_fn(msg)

    def _window_span(text: str) -> tuple[int, int] | None:
        if search_window is None:
            return None
        start_line, end_line = search_window
        start_line = max(0, start_line)
        end_line = max(start_line, end_line)
        lines = text.splitlines(keepends=True)
        if not lines:
            return (0, len(text))
        start_char = sum(len(line) for line in lines[:start_line])
        end_char = sum(len(line) for line in lines[:end_line])
        return (start_char, end_char)

    def _replace_exact(source: str, old: str, new: str, local_only: bool) -> tuple[str, bool]:
        span = _window_span(source) if local_only else None
        if span is not None:
            start_char, end_char = span
            local = source[start_char:end_char]
            local_index = local.find(old)
            if local_index == -1:
                return source, False
            abs_start = start_char + local_index
            abs_end = abs_start + len(old)
            return source[:abs_start] + new + source[abs_end:], True
        if old not in source:
            return source, False
        return source.replace(old, new, 1), True

    def _replace_ws_normalized(source: str, old: str, new: str, local_only: bool) -> tuple[str, bool]:
        old_tokens = old.split()
        if not old_tokens:
            return source, False

        ws_pattern = re.compile(r'\s+'.join(re.escape(token) for token in old_tokens))
        span = _window_span(source) if local_only else None

        if span is not None:
            start_char, end_char = span
            local = source[start_char:end_char]
            match = ws_pattern.search(local)
            if match is None:
                return source, False
            abs_start = start_char + match.start()
            abs_end = start_char + match.end()
            return source[:abs_start] + new + source[abs_end:], True

        match = ws_pattern.search(source)
        if match is None:
            return source, False
        return source[:match.start()] + new + source[match.end():], True

    result = tex_source
    applied = 0

    for edit in edits:
        old = edit.get("old", "")
        new = edit.get("new", "")
        if not old or old == new:
            continue
        for local_only, mode in (
            (True, "local exact"),
            (True, "local whitespace-normalized"),
            (False, "global exact"),
            (False, "global whitespace-normalized"),
        ):
            if local_only and search_window is None:
                continue

            if "whitespace" in mode:
                result, matched = _replace_ws_normalized(result, old, new, local_only)
            else:
                result, matched = _replace_exact(result, old, new, local_only)

            if matched:
                applied += 1
                _log(f"  Level 2: applied edit with {mode} match")
                break
        else:
            _log("  Level 2: old text not found, skipping edit")

    if applied > 0 and result != tex_source:
        _log(f"  Level 2: {applied} edit(s) applied successfully")
        return result
    return None


# ============================================================================
#  Utility: error line extraction, error signature
# ============================================================================

def extract_error_lines(error_log: str) -> list[int]:
    """Extract line numbers from a LaTeX error log.

    Prioritizes actual error lines over warnings.
    """
    error_lines: list[int] = []

    # First pass: lines containing 'error:'
    for log_line in error_log.split('\n'):
        if 'error:' in log_line.lower():
            for m in re.finditer(r'\.tex:(\d+):', log_line):
                ln = int(m.group(1))
                if ln not in error_lines:
                    error_lines.append(ln)
            for m in re.finditer(
                r'(?:input line\s+|line\s+|l\.)(\d+)', log_line
            ):
                ln = int(m.group(1))
                if ln not in error_lines:
                    error_lines.append(ln)

    # Fallback: any line number references
    if not error_lines:
        for m in re.finditer(r'(?:\.tex:(\d+):)', error_log):
            ln = int(m.group(1))
            if ln not in error_lines:
                error_lines.append(ln)
        for m in re.finditer(r'(?:line\s+|l\.)(\d+)', error_log):
            ln = int(m.group(1))
            if ln not in error_lines:
                error_lines.append(ln)

    return error_lines


def error_signature(error_log: str) -> str:
    """Compute a short hash of the error log tail for dedup."""
    return hashlib.md5(error_log[-500:].encode()).hexdigest()[:8]


def truncate_error_log(error_log: str, max_chars: int = 3000) -> str:
    """Truncate long error logs keeping head and tail."""
    if len(error_log) <= max_chars:
        return error_log
    half = max_chars // 2
    return error_log[:half] + "\n...[truncated]...\n" + error_log[-half:]


# ============================================================================
#  LLM prompt construction (shared between writing.py and review.py)
# ============================================================================

SEARCH_REPLACE_SYSTEM_PROMPT = (
    "You are a LaTeX error fixer. You will see a code snippet with an error.\n\n"
    "Your job: identify the EXACT broken text and provide a replacement.\n\n"
    "Reply with ONLY a JSON array of edit operations:\n"
    '[\n'
    '  {"old": "exact broken text from the snippet", '
    '"new": "fixed replacement text"}\n'
    ']\n\n'
    "Rules:\n"
    "- old MUST be an EXACT substring copied from the snippet "
    "(including whitespace)\n"
    "- old should be as SHORT as possible — just the broken part + "
    "enough context to be unique\n"
    "- new is the corrected version of old\n"
    "- You may include multiple edits if there are multiple errors\n"
    "- Output ONLY the JSON array, nothing else\n"
    "- Do NOT wrap in markdown fences"
)


def build_search_replace_prompt(
    error_log: str,
    error_line: int | None,
    targeted_hint: str,
    win_start: int,
    win_end: int,
    numbered_snippet: str,
) -> str:
    """Build the user prompt for Level 2 LLM search-replace fix."""
    line_ref = f" at line {error_line}" if error_line else ""
    return (
        f"LaTeX compilation error{line_ref}:\n"
        f"{error_log}\n\n"
        f"{targeted_hint}\n\n"
        f"Code snippet (lines {win_start + 1}-{win_end}):\n"
        f"{numbered_snippet}\n\n"
        f"Provide the search-replace edits as a JSON array."
    )
