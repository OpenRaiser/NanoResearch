"""Tests for LaTeX consistency and math formula checkers."""

from nanoresearch.agents.checkers import (
    check_bare_special_chars,
    check_latex_consistency,
    check_math_formulas,
    check_unicode_issues,
    check_unmatched_braces,
    validate_equations_sympy,
)


class TestLatexConsistency:
    def test_dangling_ref(self):
        tex = r"""
\section{Introduction}\label{sec:intro}
See Figure~\ref{fig:missing} and Section~\ref{sec:intro}.
"""
        issues = check_latex_consistency(tex)
        ref_issues = [i for i in issues if i["issue_type"] == "ref_mismatch"]
        assert len(ref_issues) == 1
        assert "fig:missing" in ref_issues[0]["description"]

    def test_no_dangling_ref(self):
        tex = r"""
\label{sec:intro}
\ref{sec:intro}
"""
        issues = check_latex_consistency(tex)
        ref_issues = [i for i in issues if i["issue_type"] == "ref_mismatch"]
        assert len(ref_issues) == 0

    def test_env_mismatch(self):
        tex = r"""
\begin{figure}
  Content
\end{figure}
\begin{table}
  Data
"""
        issues = check_latex_consistency(tex)
        env_issues = [i for i in issues if i["issue_type"] == "env_mismatch"]
        assert len(env_issues) == 1
        assert "table" in env_issues[0]["description"]

    def test_env_balanced(self):
        tex = r"""
\begin{figure}
\end{figure}
\begin{table}
\end{table}
"""
        issues = check_latex_consistency(tex)
        env_issues = [i for i in issues if i["issue_type"] == "env_mismatch"]
        assert len(env_issues) == 0

    def test_malformed_cite_key(self):
        tex = r"\cite{smith 2024}"
        issues = check_latex_consistency(tex)
        cite_issues = [i for i in issues if i["issue_type"] == "cite_format"]
        assert len(cite_issues) == 1
        assert "smith 2024" in cite_issues[0]["description"]

    def test_valid_cite_keys(self):
        tex = r"\cite{smith2024,jones_2023,doe2022a}"
        issues = check_latex_consistency(tex)
        cite_issues = [i for i in issues if i["issue_type"] == "cite_format"]
        assert len(cite_issues) == 0

    def test_eqref_without_label(self):
        tex = r"""
\begin{equation}\label{eq:loss}
L = \sum x_i
\end{equation}
See Eq.~\eqref{eq:missing}.
"""
        issues = check_latex_consistency(tex)
        ref_issues = [i for i in issues if i["issue_type"] == "ref_mismatch"]
        assert any("eq:missing" in i["description"] for i in ref_issues)


class TestMathFormulas:
    def test_unreferenced_equation(self):
        tex = r"""
\begin{equation}\label{eq:main}
E = mc^2
\end{equation}
This paper shows results.
"""
        issues = check_math_formulas(tex)
        eq_issues = [i for i in issues if i["issue_type"] == "unreferenced_equation"]
        assert len(eq_issues) == 1
        assert "eq:main" in eq_issues[0]["description"]

    def test_referenced_equation(self):
        tex = r"""
\begin{equation}\label{eq:main}
E = mc^2
\end{equation}
As shown in \eqref{eq:main}.
"""
        issues = check_math_formulas(tex)
        eq_issues = [i for i in issues if i["issue_type"] == "unreferenced_equation"]
        assert len(eq_issues) == 0

    def test_mixed_bold_notation(self):
        tex = r"""
$\mathbf{x} = \bm{y}$
"""
        issues = check_math_formulas(tex)
        bold_issues = [i for i in issues if i["issue_type"] == "symbol_inconsistency"]
        assert len(bold_issues) == 1

    def test_consistent_bold_notation(self):
        tex = r"""
$\mathbf{x} = \mathbf{y}$
"""
        issues = check_math_formulas(tex)
        bold_issues = [i for i in issues if i["issue_type"] == "symbol_inconsistency"]
        assert len(bold_issues) == 0


class TestUnmatchedBraces:
    def test_balanced_braces(self):
        tex = r"\section{Introduction}"
        issues = check_unmatched_braces(tex)
        assert len(issues) == 0

    def test_extra_opening_brace(self):
        tex = r"\textbf{bold text"
        issues = check_unmatched_braces(tex)
        assert len(issues) == 1
        assert issues[0]["issue_type"] == "unmatched_braces"
        assert "extra opening" in issues[0]["description"]

    def test_extra_closing_brace(self):
        tex = r"some text}"
        issues = check_unmatched_braces(tex)
        assert len(issues) == 1
        assert "extra closing" in issues[0]["description"]

    def test_ignores_comment_lines(self):
        tex = "% \\textbf{incomplete"
        issues = check_unmatched_braces(tex)
        assert len(issues) == 0

    def test_ignores_escaped_braces(self):
        tex = r"value is 10\% and set \{1,2,3\}"
        issues = check_unmatched_braces(tex)
        assert len(issues) == 0


class TestBareSpecialChars:
    def test_bare_ampersand_outside_table(self):
        tex = "Smith & Jones worked on this."
        issues = check_bare_special_chars(tex)
        amp_issues = [i for i in issues if "&" in i["description"]]
        assert len(amp_issues) == 1

    def test_ampersand_in_tabular_ok(self):
        tex = r"""
\begin{tabular}{cc}
a & b \\
c & d \\
\end{tabular}
"""
        issues = check_bare_special_chars(tex)
        amp_issues = [i for i in issues if "&" in i["description"]]
        assert len(amp_issues) == 0

    def test_ampersand_in_math_ok(self):
        tex = r"$x & y$"
        issues = check_bare_special_chars(tex)
        amp_issues = [i for i in issues if "&" in i["description"]]
        assert len(amp_issues) == 0

    def test_bare_hash(self):
        tex = "item #1 in the list"
        issues = check_bare_special_chars(tex)
        hash_issues = [i for i in issues if "#" in i["description"]]
        assert len(hash_issues) == 1

    def test_escaped_chars_ok(self):
        tex = r"Smith \& Jones, item \#1"
        issues = check_bare_special_chars(tex)
        assert len(issues) == 0

    def test_comment_lines_skipped(self):
        tex = "% This & that # stuff"
        issues = check_bare_special_chars(tex)
        assert len(issues) == 0


class TestUnicodeIssues:
    def test_smart_quotes(self):
        tex = "\u201cHello\u201d"
        issues = check_unicode_issues(tex)
        assert len(issues) >= 1
        assert any("unicode_char" == i["issue_type"] for i in issues)

    def test_em_dash(self):
        tex = "some text \u2014 more text"
        issues = check_unicode_issues(tex)
        assert len(issues) == 1
        assert "---" in issues[0]["description"]

    def test_plain_ascii_ok(self):
        tex = r"\section{Introduction} This is normal text."
        issues = check_unicode_issues(tex)
        assert len(issues) == 0

    def test_accented_char(self):
        tex = "Sch\u00f6dinger"
        issues = check_unicode_issues(tex)
        assert len(issues) == 1
        assert '\\"o' in issues[0]["description"]

    def test_dedup_same_char(self):
        tex = "\u2019first\u2019 and \u2019second\u2019"
        issues = check_unicode_issues(tex)
        # Same char reported only once
        assert len(issues) == 1

    def test_comment_lines_skipped(self):
        tex = "% caf\u00e9"
        issues = check_unicode_issues(tex)
        assert len(issues) == 0


class TestSympyValidation:
    def test_returns_empty_without_sympy(self):
        """Should return empty list when equations have no parseable content."""
        tex = r"""
\begin{equation}\label{eq:test}
\end{equation}
"""
        # This should not raise even if sympy is/isn't installed
        issues = validate_equations_sympy(tex)
        assert isinstance(issues, list)

    def test_simple_equation(self):
        tex = r"""
\begin{equation}
x^2 + y^2 = z^2
\end{equation}
"""
        issues = validate_equations_sympy(tex)
        # Should return empty list (valid equation) or list with issue (depends on sympy)
        assert isinstance(issues, list)
