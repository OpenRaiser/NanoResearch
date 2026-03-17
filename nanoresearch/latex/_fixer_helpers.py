"""LaTeX fixer helpers — error line extraction, signatures, prompt construction."""

from __future__ import annotations

import hashlib
import re


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
