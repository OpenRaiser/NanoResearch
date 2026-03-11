"""Quick test for _relocate_intro_figures (Bug 4)."""
import re
from nanoresearch.agents.writing.latex_assembler import _LaTeXAssemblerMixin


def test_ablation_figure_moved_out_of_intro():
    t = (
        "\\section{Introduction}\n"
        "We propose a method.\n"
        "\\begin{figure}[t!]\n"
        "\\includegraphics{fig_ablation.pdf}\n"
        "\\label{fig:ablation}\n"
        "\\end{figure}\n"
        "Some more intro text.\n"
        "\\section{Experiments}\n"
        "Results are in Figure~\\ref{fig:ablation}.\n"
        "\\section{Conclusion}\n"
        "Done.\n"
        "\\end{document}\n"
    )
    r = _LaTeXAssemblerMixin._relocate_intro_figures(t)

    intro = re.search(
        r"\\section\{Introduction\}(.*?)\\section\{Experiments\}", r, re.DOTALL
    )
    assert intro, "Could not find Introduction section"
    assert "fig_ablation" not in intro.group(1), "Figure still in Introduction"

    post = r[intro.end():]
    assert "fig_ablation" in post, "Figure lost entirely"


def test_architecture_figure_kept_in_intro():
    t = (
        "\\section{Introduction}\n"
        "Overview.\n"
        "\\begin{figure}[t!]\n"
        "\\includegraphics{fig1_architecture.pdf}\n"
        "\\label{fig:architecture}\n"
        "\\end{figure}\n"
        "\\section{Method}\n"
        "Details.\n"
    )
    r = _LaTeXAssemblerMixin._relocate_intro_figures(t)

    intro = re.search(
        r"\\section\{Introduction\}(.*?)\\section\{Method\}", r, re.DOTALL
    )
    assert intro, "Could not find Introduction section"
    assert "architecture" in intro.group(1), "Architecture fig wrongly removed"


def test_sanitize_step0c_garbage_before_documentclass():
    t = "for comments #>\n\\documentclass{article}\n\\begin{document}\nHello"
    r = _LaTeXAssemblerMixin._sanitize_latex(t)
    assert r.startswith("\\documentclass"), f"Starts with: {r[:30]}"


def test_sanitize_step0d_markdown_fences():
    t = "```latex\n\\section{Intro}\nHello\n```\n"
    r = _LaTeXAssemblerMixin._sanitize_latex(t)
    assert "```" not in r


def test_sanitize_step2c_tilde_ref():
    t = "Figure\\~{}\\ref{fig:arch} and Table\\~{}\\ref{tab:main}"
    r = _LaTeXAssemblerMixin._sanitize_latex(t)
    assert "~\\ref{fig:arch}" in r
    assert "~\\ref{tab:main}" in r
    assert "\\~{}" not in r
