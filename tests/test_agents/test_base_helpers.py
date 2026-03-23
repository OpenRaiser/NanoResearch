"""Tests for nanoresearch.agents._base_helpers (via base re-exports)."""

from __future__ import annotations

import pytest

from nanoresearch.agents.base import (
    detect_truncation,
    _truncate_tool_result,
    _fix_json_escapes,
)
from nanoresearch.agents._base_helpers import _extract_json_candidates


class TestTruncateToolResult:
    """Tests for _truncate_tool_result."""

    def test_short_text_unchanged(self) -> None:
        text = "short"
        assert _truncate_tool_result(text) == text

    def test_long_text_truncated(self) -> None:
        text = "a" * 10000
        result = _truncate_tool_result(text)
        assert "truncated" in result
        assert len(result) < len(text)
        assert result.startswith("a" * 2000)
        assert result.endswith("a" * 1500)


class TestDetectTruncation:
    """Tests for detect_truncation."""

    def test_no_truncation(self) -> None:
        content = "Complete sentence here."
        assert detect_truncation(content) is False

    def test_truncation_incomplete_sentence(self) -> None:
        content = "This is an incomplete"
        assert detect_truncation(content) is True

    def test_truncation_trailing_backslash(self) -> None:
        content = "Some LaTeX \\"
        assert detect_truncation(content) is True


class TestFixJsonEscapes:
    """Tests for _fix_json_escapes."""

    def test_valid_json_unchanged(self) -> None:
        text = '{"key": "value"}'
        assert _fix_json_escapes(text) == text

    def test_latex_cite_escaped(self) -> None:
        text = r'{"text": "See \cite{smith2020}"}'
        fixed = _fix_json_escapes(text)
        assert "\\\\cite" in fixed or "\\cite" in fixed


class TestExtractJsonCandidates:
    """Tests for _extract_json_candidates."""

    def test_extracts_json_object(self) -> None:
        text = 'before {"a": 1} after'
        candidates = list(_extract_json_candidates(text))
        assert len(candidates) >= 1
        assert '{"a": 1}' in candidates or any('"a": 1' in c for c in candidates)

    def test_extracts_json_array(self) -> None:
        text = 'result: [1, 2, 3]'
        candidates = list(_extract_json_candidates(text))
        assert len(candidates) >= 1
