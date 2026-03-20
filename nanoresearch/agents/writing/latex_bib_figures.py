"""LaTeX BibTeX sanitization, figure validation, and utility methods."""
from __future__ import annotations

import logging
import re
import shutil
from typing import Any

logger = logging.getLogger(__name__)
from . import _escape_latex_text


class _LaTeXBibFiguresMixin:
    """Mixin -- BibTeX sanitization, figure validation, and file-copy helpers."""

    @staticmethod
    def _sanitize_bibtex(bib: str) -> str:
        """Fix common Unicode issues in BibTeX entries and deduplicate."""
        # -- 0. Deduplicate BibTeX entries by key --
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

        # -- 1. HTML entity decoding --
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

        # BUG-32 fix: escape '#' and '%' in BibTeX text fields.
        # '#' is BibTeX string-concatenation operator; '%' starts a comment.
        # Both break BibTeX parsing when they appear bare in titles/venues.
        def _escape_hash_percent_in_entry(entry_text: str) -> str:
            def _field_repl_hp(fm: re.Match) -> str:
                field_name = fm.group(1).strip().lower()
                if field_name in _URL_FIELDS:
                    return fm.group(0)
                field_body = fm.group(2)
                escaped = re.sub(r'(?<!\\)#', r'\\#', field_body)
                escaped = re.sub(r'(?<!\\)%', r'\\%', escaped)
                return fm.group(0).replace(field_body, escaped)
            return re.sub(
                r'(\b\w+)\s*=\s*(\{(?:[^{}]|\{[^{}]*\})*\}|"[^"]*")',
                _field_repl_hp, entry_text,
            )

        bib = _escape_hash_percent_in_entry(bib)

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

    # Section-hint map for fallback placement when no \ref is found.
    # ORDER MATTERS: more specific keywords first, ambiguous ones last.
    # Experiments keywords come before Method keywords so that
    # "model_comparison" matches "comparison" (Experiments) before
    # "model" (Method).
    _FIGURE_SECTION_HINTS: list[tuple[str, str]] = [
        # -- Experiments (check first -- most figures go here) --
        ("result", "Experiments"),
        ("comparison", "Experiments"),
        ("performance", "Experiments"),
        ("ablation", "Experiments"),
        ("efficiency", "Experiments"),
        ("tradeoff", "Experiments"),
        ("training", "Experiments"),
        ("convergence", "Experiments"),
        ("qualitative", "Experiments"),
        ("visualization", "Experiments"),
        ("accuracy", "Experiments"),
        ("loss", "Experiments"),
        # -- Introduction --
        ("overview", "Introduction"),
        ("task", "Introduction"),
        ("motivation", "Introduction"),
        ("teaser", "Introduction"),
        ("intuition", "Introduction"),
        ("illustration", "Introduction"),
        # -- Method (check last -- "model"/"main" are ambiguous) --
        ("architecture", "Method"),
        ("framework", "Method"),
        ("pipeline", "Method"),
        ("diagram", "Method"),
        ("workflow", "Method"),
        ("detail", "Method"),
        ("model", "Method"),  # ambiguous -- but only reached if no Experiments keyword matched
    ]

    @staticmethod
    def _insert_figure_near_ref(
        content: str,
        fig_key: str,
        figure_block: str,
    ) -> tuple[str, bool]:
        r"""Insert *figure_block* after the paragraph that first references *fig_key*.

        Searches for \ref{fig:KEY}, \autoref{fig:KEY}, \cref{fig:KEY}.
        Places the figure after the end of that paragraph (next blank line
        or structural command), which matches top-venue placement conventions:
        figures appear right after the paragraph that discusses them.

        Returns (new_content, was_inserted).
        """
        label = fig_key  # suffix like "architecture"
        pattern = re.compile(
            rf'\\(?:ref|autoref|cref)\{{fig:{re.escape(label)}\}}',
            re.IGNORECASE,
        )
        match = pattern.search(content)
        if not match:
            return content, False

        # Find the end of the paragraph containing the \ref.
        # Top-venue convention: figure floats go right after the paragraph
        # that first discusses them. LaTeX [t!] places at top of the NEXT
        # column/page, so visually the figure lands close to the reference.
        search_start = match.end()
        para_end = re.search(
            r'\n\s*\n|\\(?:sub){0,2}section\{|\\paragraph\{'
            r'|\\begin\{table\}|\\begin\{figure\}',
            content[search_start:],
        )
        if para_end:
            insert_pos = search_start + para_end.start()
        else:
            insert_pos = len(content)

        # Avoid inserting inside a bibliography or after \end{document}
        bib_pos = len(content)
        for anchor in (r'\bibliographystyle{', r'\bibliography{',
                        r'\begin{thebibliography}', r'\end{document}'):
            p = content.find(anchor)
            if p >= 0:
                bib_pos = min(bib_pos, p)
        if insert_pos > bib_pos:
            insert_pos = bib_pos

        new_content = (
            content[:insert_pos]
            + "\n\n"
            + figure_block
            + "\n"
            + content[insert_pos:]
        )
        return new_content, True

    @staticmethod
    def _find_section_end(content: str, section_heading: str) -> int | None:
        r"""Find the character position at the END of a \section{heading}'s content.

        Returns the position just before the next \section{} or bibliography/end.
        Returns None if section not found.

        Uses keyword-based matching: \section{Proposed Method} matches
        heading="Method", \section{Experimental Results} matches
        heading="Experiments", etc. This handles the many section heading
        variations that LLMs produce.
        """
        # Build keyword from heading (core word for fuzzy matching)
        heading_lower = section_heading.lower()
        stem = heading_lower.rstrip('s')  # "Experiments" -> "Experiment"
        # Find all \section{...} commands; prefer exact match over keyword
        sec_pattern = re.compile(
            r'\\section\*?\{([^}]*(?:\{[^}]*\}[^}]*)*)\}', re.IGNORECASE,
        )
        exact_match = None
        keyword_match = None
        for m in sec_pattern.finditer(content):
            sec_title = m.group(1).lower()
            if sec_title == heading_lower:
                exact_match = m
                break  # Exact is best, stop immediately
            if keyword_match is None and (
                stem in sec_title or heading_lower in sec_title
            ):
                keyword_match = m  # Keep scanning for a possible exact match
        sec_m = exact_match or keyword_match
        if not sec_m:
            return None
        after = content[sec_m.end():]
        # Find the next \section or bibliography or \end{document}
        end_re = re.search(
            r'\\section\*?\{|\\bibliographystyle\{|\\bibliography\{'
            r'|\\begin\{thebibliography\}|\\end\{document\}',
            after,
        )
        if end_re:
            return sec_m.end() + end_re.start()
        return len(content)

    @classmethod
    def _smart_place_figure(
        cls,
        content: str,
        figure_block: str,
    ) -> str:
        r"""Place a figure block at the best position in the document.

        Strategy (matches top-venue conventions):
        1. Near first \ref{fig:label} -- after the paragraph that discusses it.
        2. In the correct \section based on figure type (architecture->Method,
           results->Experiments, etc.) -- at the END of that section.
        3. Last resort: before \bibliographystyle (never after bibliography).
        """
        # Extract the label from the figure block
        label_m = re.search(r'\\label\{fig:([^}]+)\}', figure_block)
        if label_m:
            fig_label = label_m.group(1)  # suffix only, e.g. "arch"
        else:
            label_m = re.search(r'\\label\{([^}]+)\}', figure_block)
            if label_m:
                raw = label_m.group(1)
                # Strip fig: prefix if present to avoid double-prefix in \ref search
                fig_label = raw[4:] if raw.startswith("fig:") else raw
            else:
                fig_label = ""

        # Strategy 1: place near first \ref
        if fig_label:
            new_content, placed = cls._insert_figure_near_ref(
                content, fig_label, figure_block,
            )
            if placed:
                return new_content

        # Strategy 2: place at end of the appropriate section
        target_section = None
        fig_key_lower = fig_label.lower() if fig_label else ""
        # Also check includegraphics filename for hints
        incl_m = re.search(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', figure_block)
        file_hint = (incl_m.group(1).lower() if incl_m else "")

        for keyword, section in cls._FIGURE_SECTION_HINTS:
            if keyword in fig_key_lower or keyword in file_hint:
                target_section = section
                break

        if not target_section:
            # Default: figures without hints go in Experiments
            target_section = "Experiments"

        sec_end = cls._find_section_end(content, target_section)
        if sec_end is not None:
            # Insert at the end of the section (before the next \section)
            # But back up before any trailing blank lines
            while sec_end > 0 and content[sec_end - 1] in ('\n', '\r', ' ', '\t'):
                sec_end -= 1
            content = (
                content[:sec_end]
                + "\n\n" + figure_block + "\n\n"
                + content[sec_end:]
            )
            logger.info(
                "Placed figure (label=%s) at end of \\section{%s}",
                fig_label, target_section,
            )
            return content

        # Strategy 3: before bibliography (last resort)
        for anchor in (r'\bibliographystyle{', r'\bibliography{',
                        r'\begin{thebibliography}', r'\end{document}'):
            pos = content.find(anchor)
            if pos >= 0:
                content = (
                    content[:pos]
                    + "\n\n" + figure_block + "\n\n"
                    + content[pos:]
                )
                logger.info(
                    "Placed figure (label=%s) before bibliography (last resort)",
                    fig_label,
                )
                return content

        # Truly last resort: append
        return content + "\n\n" + figure_block + "\n"

    def _validate_figures_in_latex(
        self, latex_content: str, figure_output: dict | None
    ) -> str:
        r"""Validate that every figure file from figure_output has \includegraphics in the LaTeX.

        Missing figures are injected near their \ref{fig:...} citation when
        possible, or into an appropriate section based on the figure key.
        Failed/empty figures are skipped entirely.
        Returns the (possibly modified) LaTeX content.
        """
        figures = (figure_output or {}).get("figures", {})
        if not figures:
            return latex_content

        missing_figures: list[tuple[str, str, str]] = []  # (label_suffix, block, fig_key)
        for fig_key, fig_data in figures.items():
            # Skip failed/empty figures (Fix 6)
            if fig_data.get("status") == "failed":
                self.log(f"  VALIDATION: skipping failed figure '{fig_key}'")
                continue
            if "error" in fig_data and "png_path" not in fig_data:
                self.log(f"  VALIDATION: skipping errored figure '{fig_key}'")
                continue
            # Skip figures whose image file doesn't exist
            png_path = fig_data.get("png_path")
            pdf_path = fig_data.get("pdf_path")
            if not png_path and not pdf_path:
                self.log(f"  VALIDATION: skipping '{fig_key}' -- no image file")
                continue

            pdf_name = f"{fig_key}.pdf"
            png_name = f"{fig_key}.png"
            # Check if either file name appears in an \includegraphics
            if pdf_name in latex_content or png_name in latex_content:
                continue

            # This figure is missing -- build an emergency block
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
            missing_figures.append((label_suffix, block, fig_key))

        if missing_figures:
            self.log(f"  VALIDATION: injecting {len(missing_figures)} missing figure(s)")
            for label_suffix, block, fig_key in missing_figures:
                # Use smart placement: near \ref -> correct section -> before bib
                latex_content = self._smart_place_figure(latex_content, block)
                self.log(f"    Placed '{fig_key}' using smart placement")
        else:
            self.log("  VALIDATION: all figures present in LaTeX")

        # Global pass: ensure ALL \includegraphics have a height cap.
        # LLM-written sections may include \includegraphics with only width=...
        # which can cause tall images to fill an entire page.
        latex_content = self._enforce_figure_height_cap(latex_content)

        return latex_content

    @staticmethod
    def _enforce_figure_height_cap(latex: str) -> str:
        r"""Ensure every \includegraphics has a height cap.

        Handles two cases:
        1. \includegraphics[width=...]{file} -- add height if missing
        2. \includegraphics{file} -- add full [width+height] options
        """
        _HEIGHT_OPTS = "height=0.32\\textheight, keepaspectratio"

        # Case 1: has [options] but no height= -> append height
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

        # Case 2: no [options] at all -> add default options
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
        for ext in ("*.pdf", "*.png", "*.jpg", "*.jpeg"):
            for f in fig_dir.glob(ext):
                dst = drafts_dir / f.name
                try:
                    if not dst.exists() or f.stat().st_mtime > dst.stat().st_mtime:
                        shutil.copy2(str(f), str(dst))
                except OSError as exc:
                    logger.warning("Failed to copy figure %s -> %s: %s", f, dst, exc)
