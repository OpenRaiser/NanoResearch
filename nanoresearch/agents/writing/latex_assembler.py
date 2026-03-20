"""LaTeX assembly: rendering, compilation, sanitization, figure handling."""
from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
from nanoresearch.latex import fixer as latex_fixer
from nanoresearch.schemas.paper import PaperSkeleton
from . import _escape_latex_text
from ._latex_helpers import (
    _sanitize_prose_line,
    _update_environment_stack,
    _strip_llm_thinking,
)

MAX_LATEX_FIX_ATTEMPTS = 3

# ---------------------------------------------------------------------------
# Sub-mixin imports
# ---------------------------------------------------------------------------
from .latex_figure_placement import _LaTeXFigurePlacementMixin
from .latex_bib_figures import _LaTeXBibFiguresMixin


class _LaTeXAssemblerMixin(
    _LaTeXFigurePlacementMixin,
    _LaTeXBibFiguresMixin,
):
    """Mixin -- LaTeX rendering, compilation, and sanitization.

    Inherits figure-placement methods from ``_LaTeXFigurePlacementMixin``
    and BibTeX/figure-validation methods from ``_LaTeXBibFiguresMixin``.
    """

    def _render_latex(self, skeleton: PaperSkeleton) -> str:
        """Render the paper skeleton to LaTeX string."""
        try:
            from mcp_server.tools.latex_gen import generate_full_paper
            data = skeleton.model_dump(mode="json")
            return generate_full_paper(data, skeleton.template_format)
        except ImportError:
            logger.debug("latex_gen module not available, using fallback")
            return self._fallback_latex(skeleton)
        except Exception as exc:
            logger.warning("LaTeX rendering failed, using fallback: %s", exc)
            return self._fallback_latex(skeleton)

    def _fallback_latex(self, skeleton: PaperSkeleton) -> str:
        """Generate LaTeX without templates as a fallback.

        Uses the NeurIPS 2025 document class by default.  The neurips_2025.sty
        file is copied into the compilation directory by ``_copy_style_files``.
        """
        lines = [
            r"\documentclass{article}",
            "",
            r"%% ---- NeurIPS 2025 Style ----",
            r"\usepackage[preprint]{neurips_2025}",
            "",
            r"%% ---- Standard Packages ----",
            r"\usepackage[utf8]{inputenc}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage{amsmath,amssymb}",
            r"\usepackage{graphicx}",
            r"\usepackage{hyperref}",
            r"\usepackage{booktabs}",
            r"\usepackage{xcolor}",
            r"\usepackage{float}",
            r"\usepackage[section]{placeins}",  # prevent floats drifting across sections
            r"\usepackage{multirow}",  # for multi-row table cells
            r"\graphicspath{{figures/}}",
            "",
            f"\\title{{{skeleton.title}}}",
            "\\author{{{}}}".format(" \\and ".join(skeleton.authors)),
            "",
            r"\begin{document}",
            r"\maketitle",
            "",
            r"\begin{abstract}",
            _strip_llm_thinking(skeleton.abstract),
            r"\end{abstract}",
            "",
        ]

        for section in skeleton.sections:
            lines.append(f"\\section{{{section.heading}}}")
            if section.label:
                lines.append(f"\\label{{{section.label}}}")
            lines.append(_strip_llm_thinking(section.content))
            lines.append("")
            for sub in section.subsections:
                lines.append(f"\\subsection{{{sub.heading}}}")
                if sub.label:
                    lines.append(f"\\label{{{sub.label}}}")
                lines.append(_strip_llm_thinking(sub.content))
                lines.append("")

        lines.extend([
            r"\bibliographystyle{plainnat}",
            r"\bibliography{references}",
            "",
            r"\end{document}",
        ])
        return "\n".join(lines)

    async def _compile_pdf(
        self,
        tex_path,
        max_fix_attempts: int = MAX_LATEX_FIX_ATTEMPTS,
        template_format: str = "neurips2025",
    ) -> dict:
        """Compile LaTeX to PDF with automatic error-fix loop.

        If compilation fails, feed the error back to the LLM, apply the fix,
        and retry up to *max_fix_attempts* times.

        Safety features (OpenClaw-inspired):
        - Backs up original tex before fix loop; restores on total failure
        - Post-write verification: re-reads file to confirm write succeeded
        """
        import shutil

        self._copy_figures_to_drafts()
        self._copy_style_files(template_format)

        try:
            from mcp_server.tools.pdf_compile import compile_pdf
        except ImportError as exc:
            logger.warning("Cannot import pdf_compile: %s", exc)
            return {"error": f"PDF compiler module not available: {exc}"}

        tex_path = Path(tex_path)

        # Backup original tex before any fix attempts
        backup_path = tex_path.with_suffix('.tex.bak')
        try:
            shutil.copy2(tex_path, backup_path)
        except OSError:
            pass  # non-fatal

        result: dict = {}
        for attempt in range(max_fix_attempts + 1):
            result = await compile_pdf(str(tex_path))

            if "pdf_path" in result:
                if attempt > 0:
                    self.log(f"PDF compiled successfully after {attempt} fix(es)")
                return result

            error_msg = result.get("error", "Unknown compilation error")

            # Don't retry if the problem isn't fixable via LaTeX edits
            if "No LaTeX compiler found" in error_msg or "not found" in error_msg.lower():
                self.log("No LaTeX compiler available, skipping fix loop")
                return result

            if attempt >= max_fix_attempts:
                self.log(f"PDF compilation failed after {max_fix_attempts} fix attempts")
                return result

            # Ask LLM to fix the LaTeX
            self.log(f"PDF compilation failed (attempt {attempt + 1}), asking LLM to fix...")
            self.save_log(
                f"latex_compile_error_{attempt}.log", error_msg
            )

            try:
                current_tex = tex_path.read_text(encoding="utf-8")
            except OSError as exc:
                logger.error("Cannot read tex file for fixing: %s", exc)
                return result

            fixed_tex = await self._fix_latex_errors(current_tex, error_msg)

            if fixed_tex and fixed_tex != current_tex:
                # Sanitize again after the LLM fix
                fixed_tex = self._sanitize_latex(fixed_tex)
                try:
                    tex_path.write_text(fixed_tex, encoding="utf-8")
                except OSError as exc:
                    logger.error("Cannot write fixed tex file: %s", exc)
                    return result
                # Post-write verification
                try:
                    verify = tex_path.read_text(encoding="utf-8")
                    if verify != fixed_tex:
                        self.log("  WARNING: post-write verification failed, reverting")
                        tex_path.write_text(current_tex, encoding="utf-8")
                        return result
                except OSError:
                    pass
                self.log(f"  Applied LLM fix (attempt {attempt + 1})")
            else:
                self.log("  LLM returned no changes, aborting fix loop")
                return result

        return result  # pragma: no cover

    async def _fix_latex_errors(self, tex_source: str, error_log: str) -> str | None:
        """Fix LaTeX compilation errors using a 2-level strategy.

        Level 1: Deterministic fixes (no LLM) -- via shared latex_fixer module.
        Level 2: Search-replace LLM fix -- LLM outputs {"old":"...","new":"..."} pairs.

        Inspired by OpenClaw's edit tool. NEVER sends full document for rewriting.
        """
        error_log = latex_fixer.truncate_error_log(error_log)

        error_lines = latex_fixer.extract_error_lines(error_log)
        error_line = error_lines[0] if error_lines else None

        tex_lines = tex_source.split('\n')
        error_lower = error_log.lower()

        # Level 1: Deterministic
        fixed = latex_fixer.deterministic_fix(
            tex_source, error_log, error_line, log_fn=self.log,
        )
        if fixed and fixed != tex_source:
            self.log("  Level 1: deterministic fix applied")
            return fixed

        targeted_hint = latex_fixer.classify_error(error_lower)

        # Level 2: Search-replace LLM fix
        result = await self._search_replace_llm_fix_writing(
            tex_source, tex_lines, error_line, error_log, targeted_hint
        )
        if result:
            return result

        self.log("  All fix levels exhausted, no fix found")
        return None

    async def _search_replace_llm_fix_writing(
        self, tex_source: str, tex_lines: list[str],
        error_line: int | None, error_log: str, targeted_hint: str,
    ) -> str | None:
        """Level 2 search-replace fix via shared latex_fixer module."""
        win_start, win_end, numbered = latex_fixer.build_error_snippet(
            tex_lines, error_line,
        )
        prompt = latex_fixer.build_search_replace_prompt(
            error_log, error_line, targeted_hint,
            win_start, win_end, numbered,
        )

        try:
            raw = (await self.generate(
                latex_fixer.SEARCH_REPLACE_SYSTEM_PROMPT, prompt,
            )) or ""
            edits = latex_fixer.parse_edit_json(raw)
            if not edits:
                self.log("  Level 2: LLM returned no valid edits")
                return None
            return latex_fixer.apply_edits(
                tex_source, edits, log_fn=self.log,
                search_window=(win_start, win_end),
            )
        except Exception as exc:
            self.log(f"  Level 2 search-replace fix failed: {exc}")
        return None

    # ---- LaTeX sanitization --------------------------------------------------

    @classmethod
    def _sanitize_latex(cls, text: str) -> str:
        """Fix common LLM output issues that break LaTeX compilation.

        Applies, in order:
        1. Unicode replacement (dashes, quotes)
        2. Percent-sign escaping
        3. Normalize float placement to [t!] (not [H])
        4. Auto-fix table overflow (inject \\small / \\tabcolsep / @{})
        5. Enforce max 3 contribution bullets in Introduction
        """
        # -- 0c. Truncate garbage before \documentclass --
        docclass_pos = text.find(r'\documentclass')
        if docclass_pos > 0:
            text = text[docclass_pos:]

        # -- 0d. Strip Markdown code fences --
        text = re.sub(r'```(?:latex|tex)?\s*\n', '', text)
        text = re.sub(r'\n```[ \t]*(?:\n|$)', '\n', text)

        # -- 0. Remove LLM artifact text --
        _LLM_ARTIFACT_PATTERNS = [
            r'I (?:now )?have sufficient \w+ to write.*',
            r'I have sufficient \w+.*',
            r'Let me (?:now )?(?:write|compose|draft|look up|check|verify).*',
            r'I will now (?:write|compose|draft|proceed).*',
            r'I see the paper ID.*',
            r'I (?:need|want) to (?:look up|check|find|verify|search).*',
            r'Based on (?:the|my) (?:analysis|research|review|context).*I (?:will|can|should).*',
            r'Now I (?:will|can|shall) (?:write|compose|draft).*',
            r'Here is the (?:completed?|final|written) (?:section|text|content).*:?\s*$',
            r'`[0-9a-f]{20,}`',
        ]
        for pat in _LLM_ARTIFACT_PATTERNS:
            text = re.sub(pat, '', text, flags=re.IGNORECASE | re.MULTILINE)

        # -- 0b. Strip control characters (U+0000-U+001F except \n \r \t) --
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

        # -- 1. Unicode replacements --
        text = text.replace("\u2014", "---")  # em-dash
        text = text.replace("\u2013", "--")   # en-dash
        text = text.replace("\u2018", "`")    # left single quote
        text = text.replace("\u2019", "'")    # right single quote
        text = text.replace("\u201c", "``")   # left double quote
        text = text.replace("\u201d", "''")   # right double quote
        text = text.replace("\u2192", r"$\rightarrow$")
        text = text.replace("\u2190", r"$\leftarrow$")
        text = text.replace("\u2208", r"$\in$")
        text = text.replace("\u2209", r"$\notin$")
        text = text.replace("\u2264", r"$\leq$")
        text = text.replace("\u2265", r"$\geq$")
        text = text.replace("\u00d7", r"$\times$")
        text = text.replace("\u2248", r"$\approx$")
        text = text.replace("\u00b1", r"$\pm$")
        text = text.replace("\u221e", r"$\infty$")

        # -- 2. Escape bare % after digits --
        lines = text.split("\n")
        fixed_lines = []
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith("%"):
                fixed_lines.append(line)
                continue
            if r'\url{' in line or r'\href{' in line:
                fixed_lines.append(line)
            else:
                fixed_line = re.sub(r'(?<!\\)(\d)%', r'\1\\%', line)
                fixed_lines.append(fixed_line)
        text = "\n".join(fixed_lines)

        # -- 2b. Escape prose-only special chars in lines / captions / titles --
        env_stack: list[str] = []
        sanitized_lines: list[str] = []
        for line in text.split("\n"):
            sanitized_lines.append(_sanitize_prose_line(line, env_stack))
            _update_environment_stack(line, env_stack)
        text = "\n".join(sanitized_lines)

        # -- 2c. Fix \~{}\ref -> ~\ref --
        text = re.sub(
            r'\\~\{\}(\\(?:ref|eqref|cite[tp]?|pageref)\{)',
            r'~\1',
            text,
        )

        # -- 3. Normalize figure placement --
        text = re.sub(
            r'\\begin\{figure\}\s*\[[Hh]!?\]',
            r'\\begin{figure}[t!]',
            text,
        )
        text = re.sub(
            r'\\begin\{figure\}(?!\[)',
            r'\\begin{figure}[t!]',
            text,
        )
        text = re.sub(
            r'\\begin\{figure\*\}\s*\[[Hh]!?\]',
            r'\\begin{figure*}[t!]',
            text,
        )
        text = re.sub(
            r'\\begin\{table\}\s*\[[Hh]!?\]',
            r'\\begin{table}[t!]',
            text,
        )
        text = re.sub(
            r'\\begin\{table\*\}\s*\[[Hh]!?\]',
            r'\\begin{table*}[t!]',
            text,
        )

        # -- 4. Auto-fix table overflow --
        text = cls._fix_table_overflow(text)

        # -- 5. Enforce contribution limit --
        text = cls._enforce_contribution_limit(text)

        # -- 6. Collapse blank lines before/after math environments --
        _math_envs = r'(?:equation|align|gather|multline|eqnarray)\*?'
        text = re.sub(
            rf'\n[ \t]*\n([ \t]*\\begin\{{{_math_envs}\}})',
            r'\n\1',
            text,
        )
        text = re.sub(
            rf'(\\end\{{{_math_envs}\}})[ \t]*\n[ \t]*\n',
            r'\1\n',
            text,
        )

        # -- 7. Extract figure blocks from inside list environments --
        text = cls._extract_figures_from_lists(text)

        # -- 8. Relocate non-architecture figures out of Introduction --
        text = cls._relocate_intro_figures(text)

        # -- 9. Fix stray \end{document} inside body --
        text = cls._fix_end_document_placement(text)

        # -- 10. Relocate figures stranded after bibliography --
        text = cls._relocate_post_bib_figures(text)

        # -- 11. Spread consecutive figures --
        text = cls._spread_consecutive_figures(text)

        return text
