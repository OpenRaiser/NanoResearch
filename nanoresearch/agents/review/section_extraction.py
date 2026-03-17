"""Section extraction and JSON repair helpers for the review package."""

from __future__ import annotations

import json
import re

from ._constants import _SECTION_PATTERN


class _SectionExtractionMixin:
    """Mixin — section extraction and JSON repair methods."""

    @staticmethod
    def _extract_sections(tex: str) -> list[tuple[str, str, int]]:
        """Extract (heading, content, level) tuples from LaTeX source.

        Handles \\section{} (level=0), \\subsection{} (level=1),
        and \\subsubsection{} (level=2).
        """
        # BUG-37 fix: support 2 levels of nested braces in section titles
        # (e.g. \section{Method for \textbf{Computing \textit{X}}}).
        # Inner group: [^{}]  |  {  ( [^{}] | { [^{}]* } )*  }
        matches = list(_SECTION_PATTERN.finditer(tex))
        if not matches:
            return [("Full Paper", tex, 0)]

        sections: list[tuple[str, str, int]] = []
        for i, m in enumerate(matches):
            prefix = m.group(1)  # "", "sub", or "subsub"
            level = prefix.count("sub")
            heading = m.group(2).strip()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(tex)
            content = tex[start:end].strip()
            sections.append((heading, content, level))
        return sections

    @staticmethod
    def _get_full_section_content(
        sections: list[tuple[str, str, int]], heading: str
    ) -> str:
        """Get the full content of a top-level section including its subsections.

        Merges subsection content back into the parent section so the reviewer
        sees the complete section, not just the intro paragraph.
        """
        for i, (h, c, level) in enumerate(sections):
            if h != heading:
                continue
            if level != 0:
                return c  # Subsection — return as-is
            # Merge all following subsections until the next level=0
            parts = [c]
            for j in range(i + 1, len(sections)):
                if sections[j][2] == 0:
                    break
                sub_h, sub_c, sub_lvl = sections[j]
                sub_prefix = "\\sub" * sub_lvl + "section"
                parts.append(f"\\{sub_prefix}{{{sub_h}}}\n{sub_c}")
            return "\n\n".join(parts)
        return ""

    @staticmethod
    def _repair_truncated_json(text: str) -> dict | None:
        """Attempt to repair JSON that was truncated mid-output.

        Handles common truncation patterns: unterminated strings, missing
        closing brackets/braces.
        """
        # Try parsing as-is first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Strategy: close any open strings, arrays, and objects
        repaired = text.rstrip()
        # If inside a string, close it
        in_string = False
        escaped = False
        for ch in repaired:
            if escaped:
                escaped = False
                continue
            if ch == '\\':
                escaped = True
                continue
            if ch == '"':
                in_string = not in_string
        if in_string:
            repaired += '"'

        # Count open brackets/braces using a stack (close in correct LIFO order)
        stack: list[str] = []
        in_str = False
        esc = False
        for ch in repaired:
            if esc:
                esc = False
                continue
            if ch == '\\':
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
            if in_str:
                continue
            if ch in ('{', '['):
                stack.append(ch)
            elif ch == '}' and stack and stack[-1] == '{':
                stack.pop()
            elif ch == ']' and stack and stack[-1] == '[':
                stack.pop()

        closers = {'{': '}', '[': ']'}
        for bracket in reversed(stack):
            repaired += closers.get(bracket, '')

        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

        # Last resort: extract score via regex from the raw text
        score_match = re.search(r'"score"\s*:\s*(\d+)', text)
        if score_match:
            # BUG-31 fix: clamp score to valid range [1, 10].
            # LLM may output e.g. "score": 99 in truncated JSON.
            score = max(1, min(10, int(score_match.group(1))))
            # BUG-38 fix: lower min-length from 10 to 3 so short but
            # valid issues (e.g. "Use \\cite{}") are not silently dropped.
            issues = re.findall(r'"issues"\s*:\s*\[(.*?)\]', text, re.DOTALL)
            issue_list = []
            if issues:
                issue_list = re.findall(r'"([^"]{3,})"', issues[0])
            suggestions = re.findall(r'"suggestions"\s*:\s*\[(.*?)\]', text, re.DOTALL)
            suggestion_list = []
            if suggestions:
                suggestion_list = re.findall(r'"([^"]{3,})"', suggestions[0])
            return {
                "score": score,
                "issues": issue_list[:5],
                "suggestions": suggestion_list[:3],
            }
        return None
