"""Tests for _check_global_consistency in the Writing module."""

import pytest

from nanoresearch.agents.writing import _check_global_consistency
from nanoresearch.schemas.paper import Section


def _make_sections(**kwargs) -> list[Section]:
    """Helper: build Section objects from label=content pairs."""
    return [
        Section(heading=label.replace("sec:", "").title(), label=label, content=content)
        for label, content in kwargs.items()
    ]


# ── 1. ref/label matching ──


class TestRefLabelMatching:
    def test_broken_ref_detected(self):
        tex = r"\ref{fig:missing} and \label{fig:existing}"
        issues = _check_global_consistency(tex, "", [])
        assert any("fig:missing" in i and "no matching" in i for i in issues)

    def test_eqref_without_label(self):
        tex = r"See Eq.~\eqref{eq:loss}."
        issues = _check_global_consistency(tex, "", [])
        assert any("eq:loss" in i for i in issues)

    def test_all_refs_matched(self):
        tex = r"\ref{fig:arch} and \label{fig:arch} and \ref{tab:main} and \label{tab:main}"
        issues = _check_global_consistency(tex, "", [])
        ref_issues = [i for i in issues if "no matching" in i]
        assert ref_issues == []

    def test_autoref_checked(self):
        tex = r"\autoref{sec:ghost}"
        issues = _check_global_consistency(tex, "", [])
        assert any("sec:ghost" in i for i in issues)


# ── 2. Duplicate labels ──


class TestDuplicateLabels:
    def test_duplicate_label_detected(self):
        tex = r"\label{fig:arch} some text \label{fig:arch}"
        issues = _check_global_consistency(tex, "", [])
        assert any("Duplicate" in i and "fig:arch" in i for i in issues)

    def test_unique_labels_ok(self):
        tex = r"\label{fig:a} \label{fig:b} \label{tab:c}"
        issues = _check_global_consistency(tex, "", [])
        dup_issues = [i for i in issues if "Duplicate" in i]
        assert dup_issues == []


# ── 3. Abstract number consistency ──


class TestAbstractNumbers:
    def test_abstract_pct_missing_in_body(self):
        abstract = "Our method achieves 93.2\\% accuracy."
        sections = _make_sections(**{"sec:experiments": "The baseline gets 85.0\\%."})
        tex = abstract + "\n" + sections[0].content
        issues = _check_global_consistency(tex, abstract, sections)
        assert any("93.2" in i and "fabrication" in i for i in issues)

    def test_abstract_pct_present_in_body(self):
        abstract = "Our method achieves 93.2\\% accuracy."
        sections = _make_sections(
            **{"sec:experiments": "Our method achieves 93.2\\% accuracy on the test set."}
        )
        tex = abstract + "\n" + sections[0].content
        issues = _check_global_consistency(tex, abstract, sections)
        pct_issues = [i for i in issues if "93.2" in i]
        assert pct_issues == []

    def test_no_abstract_no_crash(self):
        issues = _check_global_consistency("some tex", "", [])
        pct_issues = [i for i in issues if "fabrication" in i]
        assert pct_issues == []


# ── 4. Contribution bullet count ──


class TestContributionCount:
    def test_too_many_bullets_warned(self):
        intro_content = (
            "\\begin{itemize}\n"
            + "\\item A\n" * 7
            + "\\end{itemize}"
        )
        sections = _make_sections(**{"sec:intro": intro_content})
        issues = _check_global_consistency(intro_content, "", sections)
        assert any("7" in i and "merging" in i for i in issues)

    def test_normal_bullet_count_ok(self):
        intro_content = (
            "\\begin{itemize}\n"
            "\\item We propose X.\n"
            "\\item We introduce Y.\n"
            "\\item Experiments show Z.\n"
            "\\end{itemize}"
        )
        sections = _make_sections(**{"sec:intro": intro_content})
        issues = _check_global_consistency(intro_content, "", sections)
        bullet_issues = [i for i in issues if "merging" in i]
        assert bullet_issues == []


# ── 5. Floats without labels ──


class TestFloatLabels:
    def test_figure_without_label_warned(self):
        tex = (
            "\\begin{figure}[t!]\n"
            "\\includegraphics{fig.pdf}\n"
            "\\caption{A figure}\n"
            "\\end{figure}"
        )
        issues = _check_global_consistency(tex, "", [])
        assert any("figure" in i and "no \\label" in i for i in issues)

    def test_table_without_label_warned(self):
        tex = (
            "\\begin{table}[t!]\n"
            "\\caption{Results}\n"
            "\\begin{tabular}{cc} a & b \\end{tabular}\n"
            "\\end{table}"
        )
        issues = _check_global_consistency(tex, "", [])
        assert any("table" in i and "no \\label" in i for i in issues)

    def test_figure_with_label_ok(self):
        tex = (
            "\\begin{figure}[t!]\n"
            "\\includegraphics{fig.pdf}\n"
            "\\caption{A figure}\n"
            "\\label{fig:test}\n"
            "\\end{figure}"
        )
        issues = _check_global_consistency(tex, "", [])
        float_issues = [i for i in issues if "no \\label" in i]
        assert float_issues == []

    def test_figure_star_checked(self):
        tex = (
            "\\begin{figure*}\n"
            "\\includegraphics{wide.pdf}\n"
            "\\caption{Wide figure}\n"
            "\\end{figure*}"
        )
        issues = _check_global_consistency(tex, "", [])
        assert any("figure*" in i and "no \\label" in i for i in issues)


# ── Combined: clean document has no issues ──


class TestCleanDocument:
    def test_clean_doc_no_issues(self):
        abstract = "We achieve 95.1\\% accuracy."
        intro = (
            "\\begin{itemize}\n"
            "\\item We propose X.\n"
            "\\item Experiments demonstrate 95.1\\% accuracy.\n"
            "\\end{itemize}"
        )
        experiments = (
            "Our method reaches 95.1\\% accuracy.\n"
            "\\begin{table}[t!]\n"
            "\\caption{Main results}\\label{tab:main}\n"
            "\\begin{tabular}{cc} Method & Acc \\\\ Ours & 95.1 \\end{tabular}\n"
            "\\end{table}\n"
            "\\begin{figure}[t!]\n"
            "\\includegraphics{fig.pdf}\n"
            "\\caption{Results}\\label{fig:results}\n"
            "\\end{figure}"
        )
        sections = [
            Section(heading="Introduction", label="sec:intro", content=intro),
            Section(heading="Experiments", label="sec:experiments", content=experiments),
        ]
        tex = (
            abstract + "\n" + intro + "\n" + experiments
            + "\nSee Table~\\ref{tab:main} and Figure~\\ref{fig:results}."
        )
        issues = _check_global_consistency(tex, abstract, sections)
        assert issues == []
