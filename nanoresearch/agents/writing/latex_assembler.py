"""LaTeX assembly: rendering, compilation, sanitization, figure handling."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
from nanoresearch.latex import fixer as latex_fixer
from nanoresearch.schemas.paper import PaperSkeleton
from . import _escape_latex_text

MAX_LATEX_FIX_ATTEMPTS = 3


class _LaTeXAssemblerMixin:
    """Mixin — LaTeX rendering and compilation."""

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
        """Generate LaTeX without templates as a fallback."""
        lines = [
            r"\documentclass{article}",
            r"\usepackage[utf8]{inputenc}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage[a4paper, margin=1in]{geometry}",
            r"\usepackage{amsmath,amssymb}",
            r"\usepackage{graphicx}",
            r"\usepackage{hyperref}",
            r"\usepackage{natbib}",
            r"\usepackage{booktabs}",
            r"\usepackage{float}",
            r"\usepackage[section]{placeins}",  # prevent floats drifting across sections
            r"\usepackage{multirow}",  # for multi-row table cells
            "",
            f"\\title{{{skeleton.title}}}",
            f"\\author{{{' \\and '.join(skeleton.authors)}}}",
            r"\date{}",
            "",
            r"\begin{document}",
            r"\maketitle",
            "",
            r"\begin{abstract}",
            skeleton.abstract,
            r"\end{abstract}",
            "",
        ]

        for section in skeleton.sections:
            lines.append(f"\\section{{{section.heading}}}")
            if section.label:
                lines.append(f"\\label{{{section.label}}}")
            lines.append(section.content)
            lines.append("")
            for sub in section.subsections:
                lines.append(f"\\subsection{{{sub.heading}}}")
                if sub.label:
                    lines.append(f"\\label{{{sub.label}}}")
                lines.append(sub.content)
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
        template_format: str = "arxiv",
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
                # Restore backup on total failure
                if backup_path.exists():
                    try:
                        shutil.copy2(backup_path, tex_path)
                        self.log("  Restored original tex from backup")
                    except OSError:
                        pass
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

        Level 1: Deterministic fixes (no LLM) — via shared latex_fixer module.
        Level 2: Search-replace LLM fix — LLM outputs {"old":"...","new":"..."} pairs.

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
            )
        except Exception as exc:
            self.log(f"  Level 2 search-replace fix failed: {exc}")
        return None

    # ---- LaTeX sanitization --------------------------------------------------

    @staticmethod
    def _sanitize_latex(text: str) -> str:
        """Fix common LLM output issues that break LaTeX compilation.

        Applies, in order:
        1. Unicode replacement (dashes, quotes)
        2. Percent-sign escaping
        3. Normalize float placement to [t!] (not [H])
        4. Auto-fix table overflow (inject \\small / \\tabcolsep / @{})
        5. Enforce max 3 contribution bullets in Introduction
        """
        # ── 0. Remove LLM artifact text ──
        # LLMs sometimes emit "thinking out loud" text that leaks into LaTeX
        _LLM_ARTIFACT_PATTERNS = [
            # "I now have sufficient {information/context/data} to write..."
            r'I (?:now )?have sufficient \w+ to write.*',
            r'I have sufficient \w+.*',
            # "Let me {now} {write/compose/look up/draft}..."
            r'Let me (?:now )?(?:write|compose|draft|look up|check|verify).*',
            r'I will now (?:write|compose|draft|proceed).*',
            # "I see the paper ID..." / "I need to..." / "I should..."
            r'I see the paper ID.*',
            r'I (?:need|want) to (?:look up|check|find|verify|search).*',
            # "Based on {the/my} {analysis/research}..."
            r'Based on (?:the|my) (?:analysis|research|review|context).*I (?:will|can|should).*',
            r'Now I (?:will|can|shall) (?:write|compose|draft).*',
            r'Here is the (?:completed?|final|written) (?:section|text|content).*:?\s*$',
            # Backtick-wrapped paper IDs leaked from tool results
            r'`[0-9a-f]{20,}`',
        ]
        for pat in _LLM_ARTIFACT_PATTERNS:
            text = re.sub(pat, '', text, flags=re.IGNORECASE | re.MULTILINE)

        # ── 0b. Strip control characters (U+0000–U+001F except \n \r \t) ──
        # LLMs occasionally emit invisible control chars that crash tectonic
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

        # ── 1. Unicode replacements ──
        text = text.replace("\u2014", "---")  # em-dash
        text = text.replace("\u2013", "--")   # en-dash
        text = text.replace("\u2018", "`")    # left single quote
        text = text.replace("\u2019", "'")    # right single quote
        text = text.replace("\u201c", "``")   # left double quote
        text = text.replace("\u201d", "''")   # right double quote
        text = text.replace("\u2192", r"$\rightarrow$")  # →
        text = text.replace("\u2190", r"$\leftarrow$")   # ←
        text = text.replace("\u2208", r"$\in$")           # ∈
        text = text.replace("\u2209", r"$\notin$")        # ∉
        text = text.replace("\u2264", r"$\leq$")          # ≤
        text = text.replace("\u2265", r"$\geq$")          # ≥
        text = text.replace("\u00d7", r"$\times$")        # ×
        text = text.replace("\u2248", r"$\approx$")       # ≈
        text = text.replace("\u00b1", r"$\pm$")           # ±
        text = text.replace("\u221e", r"$\infty$")        # ∞

        # ── 2. Escape bare % after digits ──
        lines = text.split("\n")
        fixed_lines = []
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith("%"):
                fixed_lines.append(line)
                continue
            fixed_line = re.sub(r'(?<!\\)(\d)%', r'\1\\%', line)
            fixed_lines.append(fixed_line)
        text = "\n".join(fixed_lines)

        # ── 3. Normalize figure placement ──
        # Use [t!] instead of [H] to let LaTeX optimize float placement.
        # placeins package with [section] option prevents cross-section drift.
        # Normalize [H], [h], [h!] → [t!] for figure and figure*
        text = re.sub(
            r'\\begin\{figure\}\s*\[[Hh]!?\]',
            r'\\begin{figure}[t!]',
            text,
        )
        # Handle bare \begin{figure} without any placement arg
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
        # Normalize table placement too
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

        # ── 4. Auto-fix table overflow ──
        text = _LaTeXAssemblerMixin._fix_table_overflow(text)

        # ── 5. Enforce contribution limit ──
        text = _LaTeXAssemblerMixin._enforce_contribution_limit(text)

        # ── 6. Collapse blank lines before/after math environments ──
        # A blank line before \begin{equation} creates a paragraph break → extra vertical space
        _math_envs = r'(?:equation|align|gather|multline|eqnarray)\*?'
        text = re.sub(
            rf'\n[ \t]*\n([ \t]*\\begin\{{{_math_envs}\}})',
            r'\n\1',
            text,
        )
        # Also collapse blank lines after \end{math_env}
        text = re.sub(
            rf'(\\end\{{{_math_envs}\}})[ \t]*\n[ \t]*\n',
            r'\1\n',
            text,
        )

        # ── 7. Extract figure blocks from inside list environments ──
        text = _LaTeXAssemblerMixin._extract_figures_from_lists(text)

        return text

    # ---- table / contribution post-processors --------------------------------

    @staticmethod
    def _fix_table_overflow(text: str) -> str:
        """Inject \\small, \\tabcolsep, and @{} into tables that lack them."""

        def _patch_table(match: re.Match) -> str:
            block = match.group(0)
            # Inject \small after \begin{table}[...] or \begin{table*}[...] if missing
            if "\\small" not in block:
                block = re.sub(
                    r'(\\begin\{table\*?\}\[[^\]]*\])',
                    r'\1\n\\small',
                    block,
                )
            # Inject \setlength{\tabcolsep}{4pt} before \begin{tabular} if missing
            if "\\tabcolsep" not in block:
                block = block.replace(
                    "\\begin{tabular}",
                    "\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}",
                )
            # Add @{} to tabular column spec if missing (opening and closing)
            # Uses balanced-brace search to correctly handle @{} in column specs
            def _fix_tabular_at_braces(text):
                result = []
                i = 0
                tag = "\\begin{tabular}{"
                while i < len(text):
                    pos = text.find(tag, i)
                    if pos == -1:
                        result.append(text[i:])
                        break
                    result.append(text[i:pos])
                    # Find matching closing brace using balanced counting
                    brace_start = pos + len(tag) - 1  # index of opening {
                    depth = 0
                    brace_end = brace_start
                    for j in range(brace_start, len(text)):
                        if text[j] == '{':
                            depth += 1
                        elif text[j] == '}':
                            depth -= 1
                            if depth == 0:
                                brace_end = j
                                break
                    if brace_end <= brace_start:
                        # Can't parse, leave unchanged
                        result.append(tag)
                        i = pos + len(tag)
                        continue
                    spec = text[brace_start + 1:brace_end]
                    # Clean up garbled patterns from prior runs
                    while "@{@{}}" in spec:
                        spec = spec.replace("@{@{}}", "@{}")
                    if not spec.startswith("@{}"):
                        spec = "@{}" + spec
                    if not spec.endswith("@{}"):
                        spec = spec + "@{}"
                    result.append(f"\\begin{{tabular}}{{{spec}}}")
                    i = brace_end + 1
                return "".join(result)

            block = _fix_tabular_at_braces(block)
            return block

        # Match entire table environments (non-greedy), including table*
        text = re.sub(
            r'\\begin\{table\*?\}.*?\\end\{table\*?\}',
            _patch_table,
            text,
            flags=re.DOTALL,
        )
        return text

    @staticmethod
    def _enforce_contribution_limit(text: str, max_items: int = 3) -> str:
        """Truncate itemize blocks to *max_items* in the Introduction section.

        Only targets the first itemize block found between \\section{Introduction}
        and the next \\section{}.
        """
        intro_match = re.search(
            r'\\section\{Introduction\}(.*?)(?=\\section\{)',
            text,
            re.DOTALL,
        )
        if not intro_match:
            return text

        intro = intro_match.group(1)
        item_env = re.search(
            r'(\\begin\{itemize\})(.*?)(\\end\{itemize\})',
            intro,
            re.DOTALL,
        )
        if not item_env:
            return text

        items = list(re.finditer(r'\\item\b', item_env.group(2)))
        if len(items) <= max_items:
            return text

        # Keep only the first max_items items
        keep_end = items[max_items].start()
        new_body = item_env.group(2)[:keep_end].rstrip()
        new_env = f"{item_env.group(1)}{new_body}\n{item_env.group(3)}"
        new_intro = intro[:item_env.start()] + new_env + intro[item_env.end():]
        text = text[:intro_match.start(1)] + new_intro + text[intro_match.end(1):]
        return text

    @staticmethod
    def _extract_figures_from_lists(text: str) -> str:
        """Move figure/figure* blocks out of itemize/enumerate environments.

        Figures inside list environments cause severe formatting issues,
        especially with [H] placement. This extracts them and places
        them immediately after the closing \\end{itemize/enumerate}.
        """
        fig_pattern = re.compile(
            r'\\begin\{figure\*?\}.*?\\end\{figure\*?\}', re.DOTALL
        )
        for env in ('itemize', 'enumerate'):
            # Use re.sub with callback to handle ALL list instances in one pass
            # Match innermost (non-nested) list environments
            pat = re.compile(
                rf'(\\begin\{{{env}\}})'
                rf'((?:(?!\\begin\{{{env}\}})(?!\\end\{{{env}\}}).)*?)'
                rf'(\\end\{{{env}\}})',
                re.DOTALL,
            )

            def _move_figs(m: re.Match) -> str:
                body = m.group(2)
                figs = list(fig_pattern.finditer(body))
                if not figs:
                    return m.group(0)  # no figures, leave unchanged
                # Extract figures from list body
                extracted: list[str] = []
                new_body = body
                for fm in reversed(figs):
                    extracted.insert(0, fm.group(0))
                    new_body = new_body[:fm.start()] + new_body[fm.end():]
                new_body = re.sub(r'\n{3,}', '\n\n', new_body)
                return (
                    m.group(1) + new_body + m.group(3)
                    + '\n\n' + '\n\n'.join(extracted)
                )

            text = pat.sub(_move_figs, text)
        return text

    @staticmethod
    def _sanitize_bibtex(bib: str) -> str:
        """Fix common Unicode issues in BibTeX entries and deduplicate."""
        # ── 0. Deduplicate BibTeX entries by key ──
        # Split into individual entries and keep only the first occurrence of each key.
        entry_pattern = re.compile(r'(@\w+\s*\{[^,\s]+\s*,.*?\n\}\s*\n?)', re.DOTALL)
        key_pattern = re.compile(r'@\w+\s*\{\s*([^,\s]+)\s*,')
        entries = entry_pattern.findall(bib)
        if entries:
            seen_bib_keys: set[str] = set()
            deduped_entries: list[str] = []
            for entry in entries:
                key_match = key_pattern.match(entry.strip())
                if key_match:
                    bib_key = key_match.group(1).strip()
                    if bib_key in seen_bib_keys:
                        continue
                    seen_bib_keys.add(bib_key)
                deduped_entries.append(entry.strip())
            bib = "\n\n".join(deduped_entries) + "\n"

        # ── 1. HTML entity decoding ──
        # APIs (Semantic Scholar, OpenAlex) sometimes return HTML entities in titles.
        # Must convert BEFORE Unicode replacements since some entities decode to Unicode.
        import html as _html
        bib = _html.unescape(bib)
        # After unescape, bare '&' needs LaTeX escaping in TEXT fields (title,
        # booktitle, journal, etc.) but NOT in url/doi/eprint fields where '&'
        # is a valid query-string separator.
        _URL_FIELDS = {"url", "doi", "eprint", "howpublished", "note"}

        def _escape_ampersand_in_entry(entry_text: str) -> str:
            """Escape bare & only in non-URL BibTeX fields."""
            def _field_repl(fm: re.Match) -> str:
                field_name = fm.group(1).strip().lower()
                field_body = fm.group(2)
                if field_name in _URL_FIELDS:
                    return fm.group(0)  # leave URL fields untouched
                # Escape bare & (not already-escaped \&) in text fields
                return fm.group(0).replace(
                    field_body,
                    re.sub(r'(?<!\\)&', r'\\&', field_body),
                )
            # Match field = {value} or field = "value"
            return re.sub(
                r'(\b\w+)\s*=\s*(\{(?:[^{}]|\{[^{}]*\})*\}|"[^"]*")',
                _field_repl, entry_text,
            )

        bib = _escape_ampersand_in_entry(bib)

        replacements = {
            "\u00e9": r"{\'e}",
            "\u00e8": r"{\`e}",
            "\u00eb": r'{\"e}',
            "\u00fc": r'{\"u}',
            "\u00f6": r'{\"o}',
            "\u00e4": r'{\"a}',
            "\u00df": r"{\ss}",
            "\u00e7": r"{\c{c}}",
            "\u00c7": r"{\c{C}}",
            "\u00f1": r"{\~n}",
            "\u011f": r"{\u{g}}",
            "\u0131": r"{\i}",
            "\u015f": r"{\c{s}}",
            "\u0151": r"{\H{o}}",
            "\u0171": r"{\H{u}}",
            "\u017e": r"{\v{z}}",
            "\u0161": r"{\v{s}}",
            "\u0107": r"{\'c}",
            "\u2014": "---",
            "\u2013": "--",
        }
        for char, repl in replacements.items():
            bib = bib.replace(char, repl)

        # Fix bare underscores in title fields (cause "Missing $ inserted")
        # Only target title = {...} lines; leave other fields alone
        def _fix_title_underscores(m: re.Match) -> str:
            key = m.group(1)  # "title" or "booktitle"
            val = m.group(2)
            # Replace bare _ with \_ (but not already-escaped \_)
            val = re.sub(r'(?<!\\)_', r'\\_', val)
            return f'{key} = {{{val}}}'

        bib = re.sub(
            r'((?:book)?title)\s*=\s*\{((?:[^{}]|\{[^{}]*\})*)\}',
            _fix_title_underscores,
            bib,
            flags=re.IGNORECASE,
        )
        return bib

    # ---- smart figure placement -----------------------------------------------

    @staticmethod
    def _insert_figure_near_ref(
        content: str,
        fig_key: str,
        figure_block: str,
    ) -> tuple[str, bool]:
        """Insert *figure_block* after the paragraph that references *fig_key*.

        Returns (new_content, was_inserted).
        """
        # Build possible label patterns: fig:architecture, fig:results, etc.
        label = fig_key  # already the suffix like "architecture"
        pattern = re.compile(
            rf'\\ref\{{fig:{re.escape(label)}\}}', re.IGNORECASE,
        )
        match = pattern.search(content)
        if not match:
            return content, False

        # Find end of the paragraph (next blank line or \subsection/\paragraph)
        search_start = match.end()
        para_end = re.search(
            r'\n\s*\n|\\subsection\{|\\paragraph\{|\\begin\{table\}|\\begin\{figure\}',
            content[search_start:],
        )
        if para_end:
            insert_pos = search_start + para_end.start()
        else:
            insert_pos = len(content)

        new_content = (
            content[:insert_pos]
            + "\n\n"
            + figure_block
            + "\n"
            + content[insert_pos:]
        )
        return new_content, True

    def _validate_figures_in_latex(
        self, latex_content: str, figure_output: dict | None
    ) -> str:
        """Validate that every figure file from figure_output has \\includegraphics in the LaTeX.

        If a figure is missing, inject a figure block before \\end{document}.
        Returns the (possibly modified) LaTeX content.
        """
        figures = (figure_output or {}).get("figures", {})
        if not figures:
            return latex_content

        missing_blocks: list[str] = []
        for fig_key, fig_data in figures.items():
            if "error" in fig_data and "png_path" not in fig_data:
                continue

            pdf_name = f"{fig_key}.pdf"
            png_name = f"{fig_key}.png"
            # Check if either file name appears in an \includegraphics
            if pdf_name in latex_content or png_name in latex_content:
                continue

            # This figure is missing — build an emergency block
            self.log(f"  VALIDATION: '{fig_key}' missing from LaTeX, injecting")
            caption = _escape_latex_text(fig_data.get("caption", f"Figure: {fig_key}"))
            parts = fig_key.split("_", 1)
            label_suffix = parts[1] if len(parts) > 1 else fig_key
            include_name = pdf_name if fig_data.get("pdf_path") else png_name

            block = (
                "\\begin{figure}[t!]\n"
                "\\centering\n"
                f"\\includegraphics[width=0.85\\textwidth, "
                f"height=0.32\\textheight, keepaspectratio]"
                f"{{{include_name}}}\n"
                f"\\caption{{{caption}}}\n"
                f"\\label{{fig:{label_suffix}}}\n"
                "\\end{figure}"
            )
            missing_blocks.append(block)

        if missing_blocks:
            self.log(f"  VALIDATION: injecting {len(missing_blocks)} missing figure(s)")
            # Insert before \end{document}
            inject_text = "\n\n".join(missing_blocks)
            end_doc_pos = latex_content.rfind("\\end{document}")
            if end_doc_pos >= 0:
                latex_content = (
                    latex_content[:end_doc_pos]
                    + "\n\n% --- Auto-injected missing figures ---\n"
                    + inject_text
                    + "\n\n"
                    + latex_content[end_doc_pos:]
                )
            else:
                latex_content += "\n\n" + inject_text
        else:
            self.log("  VALIDATION: all figures present in LaTeX ✓")

        # Global pass: ensure ALL \includegraphics have a height cap.
        # LLM-written sections may include \includegraphics with only width=...
        # which can cause tall images to fill an entire page.
        latex_content = self._enforce_figure_height_cap(latex_content)

        return latex_content

    @staticmethod
    def _enforce_figure_height_cap(latex: str) -> str:
        r"""Ensure every \includegraphics has a height cap.

        Handles two cases:
        1. \includegraphics[width=...]{file} — add height if missing
        2. \includegraphics{file} — add full [width+height] options
        """
        import re

        _HEIGHT_OPTS = "height=0.32\\textheight, keepaspectratio"

        # Case 1: has [options] but no height= → append height
        def _add_height(m: re.Match) -> str:
            opts = m.group(1)
            if "height=" in opts:
                return m.group(0)
            new_opts = opts + ", " + _HEIGHT_OPTS
            return f"\\includegraphics[{new_opts}]" + m.group(2)

        latex = re.sub(
            r'\\includegraphics\[([^\]]+)\](\{[^}]+\})',
            _add_height,
            latex,
        )

        # Case 2: no [options] at all → add default options
        def _add_full_opts(m: re.Match) -> str:
            filename = m.group(1)
            return (f"\\includegraphics"
                    f"[width=0.85\\textwidth, {_HEIGHT_OPTS}]"
                    f"{{{filename}}}")

        latex = re.sub(
            r'\\includegraphics\{([^}]+)\}',
            _add_full_opts,
            latex,
        )

        return latex

    def _copy_style_files(self, template_format: str) -> None:
        """Copy .sty/.cls/.bst files bundled with *template_format* to drafts/."""
        from nanoresearch.templates import get_style_files

        drafts_dir = self.workspace.path / "drafts"
        for f in get_style_files(template_format):
            dst = drafts_dir / f.name
            if not dst.exists():
                try:
                    shutil.copy2(str(f), str(dst))
                except OSError as exc:
                    logger.warning("Failed to copy style %s -> %s: %s", f, dst, exc)

    def _copy_figures_to_drafts(self) -> None:
        """Copy figure PDF/PNG files from figures/ to drafts/ for compilation."""
        fig_dir = self.workspace.path / "figures"
        drafts_dir = self.workspace.path / "drafts"
        if not fig_dir.exists():
            return
        for ext in ("*.pdf", "*.png"):
            for f in fig_dir.glob(ext):
                dst = drafts_dir / f.name
                try:
                    if not dst.exists() or f.stat().st_mtime > dst.stat().st_mtime:
                        shutil.copy2(str(f), str(dst))
                except OSError as exc:
                    logger.warning("Failed to copy figure %s -> %s: %s", f, dst, exc)
