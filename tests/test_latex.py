"""Tests for nanoresearch.latex.fixer."""

from __future__ import annotations

import pytest

from nanoresearch.latex.fixer import deterministic_fix, UNICODE_REPLACEMENTS


class TestDeterministicFix:
    """Tests for deterministic_fix."""

    def test_unchanged_tex_returns_none(self) -> None:
        tex = r"\documentclass{article}\begin{document}Hello\end{document}"
        result = deterministic_fix(tex)
        assert result is None or result == tex

    def test_adds_end_document_if_missing(self) -> None:
        tex = r"\documentclass{article}\begin{document}Hi"
        result = deterministic_fix(tex)
        assert result is not None
        assert r"\end{document}" in result

    def test_unicode_replacement(self) -> None:
        tex = r"\documentclass{article}\begin{document}α and β\end{document}"
        result = deterministic_fix(tex)
        assert result is not None
        assert "\\alpha" in result or "alpha" in result
        assert "\\beta" in result or "beta" in result


class TestUnicodeReplacements:
    """Tests for UNICODE_REPLACEMENTS constant."""

    def test_smart_quotes_mapped(self) -> None:
        assert "\u2018" in UNICODE_REPLACEMENTS
        assert "\u2019" in UNICODE_REPLACEMENTS
        assert UNICODE_REPLACEMENTS["\u2018"] == "`"
        assert UNICODE_REPLACEMENTS["\u2019"] == "'"

    def test_math_symbols_mapped(self) -> None:
        assert "\u2264" in UNICODE_REPLACEMENTS
        assert "$\\leq$" in UNICODE_REPLACEMENTS["\u2264"] or "leq" in UNICODE_REPLACEMENTS["\u2264"]
