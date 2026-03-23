"""Tests for nanoresearch.agents.checkers."""

from __future__ import annotations

import pytest

from nanoresearch.agents.checkers import check_latex_consistency


class TestCheckLatexConsistency:
    """Tests for check_latex_consistency."""

    def test_empty_tex_no_issues(self) -> None:
        issues = check_latex_consistency("")
        assert issues == []

    def test_ref_without_label(self) -> None:
        tex = r"See \ref{fig:main} for details."
        issues = check_latex_consistency(tex)
        assert any(i["issue_type"] == "ref_mismatch" for i in issues)
        assert any("fig:main" in i["description"] for i in issues)

    def test_ref_with_label_no_issue(self) -> None:
        tex = r"\label{fig:main}\begin{figure}...\end{figure}See \ref{fig:main}."
        issues = check_latex_consistency(tex)
        ref_issues = [i for i in issues if i["issue_type"] == "ref_mismatch"]
        assert not any("fig:main" in i["description"] for i in ref_issues)

    def test_env_mismatch(self) -> None:
        tex = r"\begin{equation}x=1\end{align}"
        issues = check_latex_consistency(tex)
        assert any(i["issue_type"] == "env_mismatch" for i in issues)

    def test_balanced_env_no_mismatch(self) -> None:
        tex = r"\begin{equation}x=1\end{equation}"
        issues = check_latex_consistency(tex)
        env_issues = [i for i in issues if i["issue_type"] == "env_mismatch"]
        assert not any("equation" in i["description"] for i in env_issues)
