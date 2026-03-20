"""LaTeX figure placement: document structure fixes and figure relocation."""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


class _LaTeXFigurePlacementMixin:
    """Mixin -- end-document fix, figure relocation, table overflow, contribution limit."""

    @staticmethod
    def _fix_end_document_placement(text: str) -> str:
        r"""Ensure exactly one \end{document} at the very end, with
        \bibliographystyle and \bibliography just before it.

        LLMs sometimes emit \end{document} inside section content (e.g.
        after an equation block), causing LaTeX to stop processing before
        reaching the bibliography commands -- all \cite{} become (?).

        Only operates on full documents (must contain \begin{document}).
        """
        # Guard: only operate on full LaTeX documents
        if r'\begin{document}' not in text:
            return text

        # Count \end{document} -- if exactly one and it's at the end,
        # and bibliography is before it, nothing to fix.
        end_doc_positions = [
            m.start() for m in re.finditer(r'\\end\{document\}', text)
        ]
        if not end_doc_positions:
            # No \end{document} at all -- add bibliography + end
            text = text.rstrip()
            text += "\n\n\\bibliographystyle{plainnat}"
            text += "\n\\bibliography{references}"
            text += "\n\n\\end{document}\n"
            return text

        if len(end_doc_positions) == 1:
            # Check if bibliography is before \end{document}
            end_pos = end_doc_positions[0]
            has_bib_before = (
                re.search(r'\\bibliography\{', text[:end_pos])
                or re.search(r'\\begin\{thebibliography\}', text[:end_pos])
            )
            if has_bib_before:
                return text  # All good -- single \end{document} with bib before it

        # ---- Need to fix: multiple \end{document} or bib after it ----

        # 1. Extract bibliography commands (preserve style & file name)
        bib_style_m = re.search(
            r'\\bibliographystyle\{([^}]+)\}', text,
        )
        bib_file_m = re.search(
            r'\\bibliography\{([^}]+)\}', text,
        )
        bib_style = bib_style_m.group(1) if bib_style_m else "plainnat"
        bib_file = bib_file_m.group(1) if bib_file_m else "references"

        # Also check for inline \begin{thebibliography}...\end{thebibliography}
        inline_bib_m = re.search(
            r'(\\begin\{thebibliography\}.*?\\end\{thebibliography\})',
            text, re.DOTALL,
        )
        inline_bib = inline_bib_m.group(1) if inline_bib_m else ""

        # 2. Remove ALL \end{document} and bibliography commands from body
        text = re.sub(r'\\end\{document\}\s*', '', text)
        text = re.sub(r'\\bibliographystyle\{[^}]*\}\s*', '', text)
        text = re.sub(r'\\bibliography\{[^}]*\}\s*', '', text)
        if inline_bib:
            text = text.replace(inline_bib, '')

        # 3. Strip trailing whitespace
        text = text.rstrip()

        # 4. Re-append in correct order: bibliography -> \end{document}
        if inline_bib:
            text += "\n\n" + inline_bib
        else:
            text += f"\n\n\\bibliographystyle{{{bib_style}}}"
            text += f"\n\\bibliography{{{bib_file}}}"
        text += "\n\n\\end{document}\n"

        return text

    @classmethod
    def _relocate_post_bib_figures(cls, text: str) -> str:
        r"""Move any \begin{figure}...\end{figure} blocks that appear after
        \bibliography back to their proper positions in the document body.

        Instead of dumping all figures right before \bibliography (which is
        NOT how top-venue papers are formatted), each figure is placed near
        its first \ref{fig:...} citation, or in the appropriate section
        based on figure type. This matches the convention in NeurIPS/ICML/CVPR
        papers where figures appear close to where they are discussed.
        """
        # Find bibliography start position
        bib_patterns = [
            r'\\bibliographystyle\{',
            r'\\bibliography\{',
            r'\\begin\{thebibliography\}',
        ]
        bib_pos = -1
        for pat in bib_patterns:
            m = re.search(pat, text)
            if m:
                if bib_pos < 0 or m.start() < bib_pos:
                    bib_pos = m.start()
        if bib_pos < 0:
            return text

        after_bib = text[bib_pos:]
        # Find all figure blocks after bibliography
        fig_blocks = list(re.finditer(
            r'\\begin\{figure\*?\}.*?\\end\{figure\*?\}',
            after_bib, re.DOTALL,
        ))
        if not fig_blocks:
            return text

        # Step 1: Extract and remove all figures from after-bib region
        extracted: list[str] = []
        for m in reversed(fig_blocks):
            extracted.insert(0, m.group(0))
            start = bib_pos + m.start()
            end = bib_pos + m.end()
            while end < len(text) and text[end] in (' ', '\n', '\r'):
                end += 1
            text = text[:start] + text[end:]

        # Step 2: Re-place each figure using smart placement
        # (near its \ref, or in correct section, or before bib as last resort)
        for fig_block in extracted:
            text = cls._smart_place_figure(text, fig_block)

        logger.info(
            "Relocated %d figure(s) from after bibliography to proper positions",
            len(extracted),
        )
        return text

    # ---- table / contribution post-processors --------------------------------

    @staticmethod
    def _fix_table_overflow(text: str) -> str:
        r"""Inject \small, \tabcolsep, and @{} into tables that lack them."""

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
        r"""Truncate itemize blocks to *max_items* in the Introduction section.

        Only targets the first itemize block found between \section{Introduction}
        and the next \section{}.
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
        r"""Move figure/figure* blocks out of itemize/enumerate environments.

        Figures inside list environments cause severe formatting issues,
        especially with [H] placement. This extracts them and places
        them immediately after the closing \end{itemize/enumerate}.
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

    # Labels that are legitimately placed in Introduction (architecture/overview)
    _INTRO_KEEP_LABELS = re.compile(
        r'arch|overview|model|framework|pipeline|system|fig1|fig_1',
        re.IGNORECASE,
    )

    @classmethod
    def _relocate_intro_figures(cls, text: str) -> str:
        r"""Move non-architecture figures out of the Introduction section.

        Ablation, results, and training-curve figures don't belong in
        Introduction -- they should appear near their first \ref in
        later sections (Experiments, Results, etc.).
        """
        # Find Introduction section boundaries
        intro_match = re.search(
            r'(\\section\{Introduction\})',
            text, re.IGNORECASE,
        )
        if not intro_match:
            return text

        intro_start = intro_match.end()

        # Find the next \section{...} after Introduction
        next_section = re.search(r'\\section\{', text[intro_start:])
        if not next_section:
            return text  # no following section -- nothing to do

        intro_end = intro_start + next_section.start()
        intro_text = text[intro_start:intro_end]

        # Extract all figure environments from Introduction
        fig_pattern = re.compile(
            r'\\begin\{figure\*?\}.*?\\end\{figure\*?\}', re.DOTALL
        )
        figures_in_intro = list(fig_pattern.finditer(intro_text))
        if not figures_in_intro:
            return text

        # Classify: keep architecture figures, relocate the rest
        to_relocate: list[tuple[str, str]] = []  # (label, figure_block)
        # Process in reverse to preserve indices when removing
        new_intro = intro_text
        for m in reversed(figures_in_intro):
            fig_block = m.group(0)
            # Extract label
            label_match = re.search(r'\\label\{([^}]+)\}', fig_block)
            label = label_match.group(1) if label_match else ""

            if cls._INTRO_KEEP_LABELS.search(label):
                continue  # architecture figure -- keep in Intro

            to_relocate.insert(0, (label, fig_block))
            # Remove from Introduction
            new_intro = new_intro[:m.start()] + new_intro[m.end():]

        if not to_relocate:
            return text

        # Rebuild text with figures removed from Introduction
        text = text[:intro_start] + new_intro + text[intro_end:]
        # Collapse excess blank lines left behind
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Re-insert each figure using smart placement
        # (near its \ref, or in correct section based on figure type)
        for _label, fig_block in to_relocate:
            text = cls._smart_place_figure(text, fig_block)

        return text

    @classmethod
    def _spread_consecutive_figures(cls, text: str) -> str:
        r"""Detect consecutive \begin{figure}...\end{figure} blocks with no
        text paragraph between them and spread them apart.

        Strategy (content-aware):
        1. Remove the second figure from its current position.
        2. Re-insert it via _smart_place_figure() which places near
           its \ref{fig:label} citation or in the appropriate section
           based on figure-type hints -- matching top-venue conventions.
        3. If re-placement still leaves them consecutive (both referenced
           in the same paragraph), alternate the placement specifier
           ([t!] -> [b!]) so LaTeX separates them vertically.
        """
        fig_env = re.compile(
            r'(\\begin\{figure\*?\})(.*?)(\\end\{figure\*?\})',
            re.DOTALL,
        )
        max_passes = 20  # safety: prevent infinite loop
        pass_count = 0
        i = 0
        while pass_count < max_passes:
            figures = list(fig_env.finditer(text))
            if i >= len(figures) - 1:
                break
            pass_count += 1
            fig_a = figures[i]
            fig_b = figures[i + 1]
            between = text[fig_a.end():fig_b.start()]
            if between.strip() != "":
                i += 1
                continue

            # Consecutive -- remove fig_b and let smart placement decide
            block_b = fig_b.group(0)
            text = text[:fig_b.start()] + text[fig_b.end():]
            text = re.sub(r'\n{3,}', '\n\n', text)

            # Re-insert using content-aware placement (near \ref or
            # correct section based on figure label/type)
            text = cls._smart_place_figure(text, block_b)

            # Check if they are still consecutive after re-placement
            figures_new = list(fig_env.finditer(text))
            still_consecutive = False
            for j in range(len(figures_new) - 1):
                if figures_new[j + 1].group(0) == block_b:
                    gap = text[figures_new[j].end():figures_new[j + 1].start()]
                    if gap.strip() == "":
                        still_consecutive = True
                    break

            if still_consecutive:
                # Both referenced in the same area -- alternate specifier
                text = text.replace(
                    block_b,
                    re.sub(
                        r'\\begin\{(figure\*?)\}\[([^\]]*)\]',
                        lambda m: (
                            f'\\begin{{{m.group(1)}}}['
                            + m.group(2).replace('t', 'b')
                            + ']'
                        ),
                        block_b,
                        count=1,
                    ),
                    1,
                )
                logger.info(
                    "Spread consecutive figures: smart placement insufficient, "
                    "changed to [b] specifier"
                )
                i += 1
            else:
                logger.info(
                    "Spread consecutive figures: re-placed via smart placement"
                )
                # Don't advance -- re-check in case removal shifted another pair

        text = re.sub(r'\n{4,}', '\n\n\n', text)
        return text
