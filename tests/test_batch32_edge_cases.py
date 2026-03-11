"""Edge-case tests for Batch 32 fixes (Bugs 1-6 from IMPROVEMENTS.md)."""
import re

import pytest

from nanoresearch.agents.writing.latex_assembler import _LaTeXAssemblerMixin

_san = _LaTeXAssemblerMixin._sanitize_latex
_reloc = _LaTeXAssemblerMixin._relocate_intro_figures


# =====================================================================
# Bug 1 — step 0c: garbage before \documentclass
# =====================================================================

class TestStep0c:
    def test_no_documentclass_leaves_text_unchanged(self):
        """Section-level text (no \\documentclass) must not be truncated."""
        t = "We propose a novel method.\nOur key contributions are:"
        assert _san(t) == t

    def test_documentclass_at_position_zero(self):
        """If \\documentclass is already at the start, nothing is removed."""
        t = "\\documentclass{article}\n\\begin{document}\nHello"
        r = _san(t)
        assert r.startswith("\\documentclass")

    def test_multi_line_garbage_removed(self):
        t = "I will now write the LaTeX.\nHere it is:\n\\documentclass{article}\nBody"
        r = _san(t)
        assert r.startswith("\\documentclass{article}")
        assert "I will now" not in r


# =====================================================================
# Bug 2 — step 0d: Markdown code fences
# =====================================================================

class TestStep0d:
    def test_latex_fence_stripped(self):
        t = "```latex\n\\section{Intro}\nContent\n```\n"
        r = _san(t)
        assert "```" not in r
        assert "\\section{Intro}" in r

    def test_bare_fence_stripped(self):
        t = "```\nSome code\n```\n"
        r = _san(t)
        assert "```" not in r

    def test_lstlisting_backticks_preserved(self):
        """Backticks inside \\verb or lstlisting should NOT be stripped.
        Current regex only matches fences at line-start with a newline
        after the opening fence, so inline \\verb+```+ is safe."""
        t = "Use \\verb+```+ for code."
        r = _san(t)
        assert "\\verb+```+" in r

    def test_no_fence_no_change(self):
        t = "Normal LaTeX text with no fences."
        assert _san(t) == t


# =====================================================================
# Bug 3 — step 2c: \~{}\ref → ~\ref
# =====================================================================

class TestStep2c:
    def test_figure_ref(self):
        assert "~\\ref{fig:x}" in _san("Figure\\~{}\\ref{fig:x}")

    def test_table_ref(self):
        assert "~\\ref{tab:y}" in _san("Table\\~{}\\ref{tab:y}")

    def test_eqref(self):
        assert "~\\eqref{eq:1}" in _san("Eq.\\~{}\\eqref{eq:1}")

    def test_cite(self):
        assert "~\\cite{foo}" in _san("work\\~{}\\cite{foo}")

    def test_citep(self):
        assert "~\\citep{bar}" in _san("work\\~{}\\citep{bar}")

    def test_bare_tilde_not_affected(self):
        """A correct ~\\ref should NOT be double-modified."""
        t = "Figure~\\ref{fig:x}"
        r = _san(t)
        assert "~\\ref{fig:x}" in r
        # No double tilde or other corruption
        assert "~~" not in r

    def test_tilde_accent_not_in_ref_context_preserved(self):
        """\\~{n} for ñ should not be affected (different pattern)."""
        t = "\\~{n}i\\~{n}o"
        r = _san(t)
        assert "\\~{n}" in r  # not matched because no \ref follows


# =====================================================================
# Bug 4 — step 8: _relocate_intro_figures
# =====================================================================

class TestRelocateIntroFigures:
    def test_no_introduction_section(self):
        t = "\\section{Method}\nHello\n\\section{Results}\nBye"
        assert _reloc(t) == t

    def test_no_figures_in_intro(self):
        t = (
            "\\section{Introduction}\n"
            "Just text, no figures.\n"
            "\\section{Method}\n"
            "Details.\n"
        )
        assert _reloc(t) == t

    def test_architecture_figure_stays(self):
        t = (
            "\\section{Introduction}\n"
            "\\begin{figure}[t!]\n"
            "\\label{fig:architecture}\n"
            "\\end{figure}\n"
            "\\section{Method}\n"
        )
        r = _reloc(t)
        intro = re.search(
            r"\\section\{Introduction\}(.*?)\\section\{Method\}",
            r, re.DOTALL,
        )
        assert "architecture" in intro.group(1)

    def test_overview_figure_stays(self):
        t = (
            "\\section{Introduction}\n"
            "\\begin{figure}[t!]\n"
            "\\label{fig:system_overview}\n"
            "\\end{figure}\n"
            "\\section{Method}\n"
        )
        r = _reloc(t)
        intro = re.search(
            r"\\section\{Introduction\}(.*?)\\section\{Method\}",
            r, re.DOTALL,
        )
        assert "system_overview" in intro.group(1)

    def test_fig1_stays(self):
        """fig1 is commonly the architecture/overview figure."""
        t = (
            "\\section{Introduction}\n"
            "\\begin{figure}[t!]\n"
            "\\label{fig:fig1_something}\n"
            "\\end{figure}\n"
            "\\section{Method}\n"
        )
        r = _reloc(t)
        intro = re.search(
            r"\\section\{Introduction\}(.*?)\\section\{Method\}",
            r, re.DOTALL,
        )
        assert "fig1_something" in intro.group(1)

    def test_ablation_figure_moved(self):
        t = (
            "\\section{Introduction}\n"
            "Text.\n"
            "\\begin{figure}[t!]\n"
            "\\includegraphics{ablation.pdf}\n"
            "\\label{fig:ablation}\n"
            "\\end{figure}\n"
            "More text.\n"
            "\\section{Experiments}\n"
            "See Figure~\\ref{fig:ablation}.\n"
            "\\end{document}\n"
        )
        r = _reloc(t)
        intro = re.search(
            r"\\section\{Introduction\}(.*?)\\section\{Experiments\}",
            r, re.DOTALL,
        )
        assert "ablation" not in intro.group(1), "Ablation fig still in intro"
        # Should appear after the \ref in Experiments
        post = r[intro.end():]
        assert "ablation" in post, "Ablation fig lost"

    def test_no_ref_fallback_to_end_document(self):
        """If no \\ref exists for the label, figure goes before \\end{document}."""
        t = (
            "\\section{Introduction}\n"
            "\\begin{figure}[t!]\n"
            "\\label{fig:orphan_results}\n"
            "\\end{figure}\n"
            "\\section{Method}\n"
            "No ref to orphan.\n"
            "\\end{document}\n"
        )
        r = _reloc(t)
        intro = re.search(
            r"\\section\{Introduction\}(.*?)\\section\{Method\}",
            r, re.DOTALL,
        )
        assert "orphan_results" not in intro.group(1)
        # Should be before \end{document}
        end_doc_pos = r.find("\\end{document}")
        fig_pos = r.find("orphan_results")
        assert fig_pos < end_doc_pos, "Figure not before \\end{document}"

    def test_figure_without_label_moved(self):
        """A figure with no \\label should still be moved out of intro."""
        t = (
            "\\section{Introduction}\n"
            "\\begin{figure}[t!]\n"
            "\\includegraphics{random.pdf}\n"
            "\\end{figure}\n"
            "\\section{Method}\n"
            "\\end{document}\n"
        )
        r = _reloc(t)
        intro = re.search(
            r"\\section\{Introduction\}(.*?)\\section\{Method\}",
            r, re.DOTALL,
        )
        assert "random.pdf" not in intro.group(1)

    def test_multiple_figures_mixed(self):
        """Architecture stays, ablation + results move."""
        t = (
            "\\section{Introduction}\n"
            "\\begin{figure}[t!]\n"
            "\\label{fig:framework}\n"
            "\\end{figure}\n"
            "\\begin{figure}[t!]\n"
            "\\label{fig:ablation_study}\n"
            "\\end{figure}\n"
            "\\begin{figure}[t!]\n"
            "\\label{fig:results_comparison}\n"
            "\\end{figure}\n"
            "\\section{Experiments}\n"
            "Figure~\\ref{fig:ablation_study} and Figure~\\ref{fig:results_comparison}.\n"
            "\\end{document}\n"
        )
        r = _reloc(t)
        intro = re.search(
            r"\\section\{Introduction\}(.*?)\\section\{Experiments\}",
            r, re.DOTALL,
        )
        assert "framework" in intro.group(1), "Framework fig wrongly removed"
        assert "ablation_study" not in intro.group(1), "Ablation still in intro"
        assert "results_comparison" not in intro.group(1), "Results still in intro"


# =====================================================================
# Bug 5 — import re in figure_gen.py
# =====================================================================

def test_figure_gen_has_import_re():
    """figure_gen.py must have `import re` at module level."""
    import nanoresearch.agents.figure_gen as fg
    import re as _re
    # The module must have `re` in its namespace
    assert hasattr(fg, 're'), "figure_gen.py missing import re"
    assert fg.re is _re


# =====================================================================
# Bug 6 — chart prompt prohibition (check prompt content)
# =====================================================================

def test_chart_prompt_prohibits_error_messages():
    """chart_code.yaml system prompt should include prohibition keywords."""
    from pathlib import Path
    import yaml

    yaml_path = (
        Path(__file__).resolve().parent.parent
        / "nanoresearch" / "prompts" / "figure_gen" / "chart_code.yaml"
    )
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    prompt = data.get("system_prompt", "")
    assert "ABSOLUTE PROHIBITION" in prompt
    assert "NEVER" in prompt
    assert "failed" in prompt.lower()
    assert "synthetic" in prompt.lower()
    assert "error" in prompt.lower()
