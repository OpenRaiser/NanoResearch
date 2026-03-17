r"""Structural fixes: \end{document} uniqueness, contribution limit, blank lines."""
from __future__ import annotations

import re


def fix_structure(text: str) -> str:
    """Apply structural fixes."""
    text = _enforce_contribution_limit(text)
    text = _collapse_blank_lines(text)
    text = _strip_stray_end_document(text)
    text = _optimize_last_page_spacing(text)
    return text


def _enforce_contribution_limit(text: str, max_items: int = 3) -> str:
    """Truncate itemize blocks to max_items in the Introduction section."""
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

    keep_end = items[max_items].start()
    new_body = item_env.group(2)[:keep_end].rstrip()
    new_env = f"{item_env.group(1)}{new_body}\n{item_env.group(3)}"
    new_intro = intro[:item_env.start()] + new_env + intro[item_env.end():]
    text = text[:intro_match.start(1)] + new_intro + text[intro_match.end(1):]
    return text


def _collapse_blank_lines(text: str) -> str:
    """Collapse runs of 3+ blank lines into 2."""
    return re.sub(r'\n{4,}', '\n\n\n', text)


def _strip_stray_end_document(text: str) -> str:
    r"""Remove \end{document} that appears in the middle of the document body.

    Only operates on full documents. Preserves the final \end{document}.
    This complements figures.py's _fix_end_document_placement.
    """
    if r'\begin{document}' not in text:
        return text

    # Find all \end{document} occurrences
    positions = list(re.finditer(r'\\end\{document\}', text))
    if len(positions) <= 1:
        return text

    # Remove all but the last one
    for m in reversed(positions[:-1]):
        text = text[:m.start()] + text[m.end():]

    return text


def _optimize_last_page_spacing(text: str) -> str:
    r"""Reduce empty space on the last page.

    1. Add ``\raggedbottom`` to preamble (prevents page-stretching)
    2. Shrink bibliography font/spacing for compact references
    3. Allow last content page to stretch to absorb more lines
    """
    begin_doc = text.find(r'\begin{document}')
    if begin_doc < 0:
        return text

    # 1. Ensure \raggedbottom is in preamble
    if r'\raggedbottom' not in text:
        text = text[:begin_doc] + '\\raggedbottom\n' + text[begin_doc:]
        begin_doc = text.find(r'\begin{document}')

    # 2. Wrap bibliography in compact formatting
    bib_m = re.search(r'\\bibliographystyle\{', text)
    if bib_m and r'\bibsep' not in text:
        bib_pos = bib_m.start()
        spacing_cmds = (
            '\\begingroup\n'
            '\\footnotesize\n'
            '\\setlength{\\bibsep}{1pt plus 0.5pt minus 0.5pt}\n'
            '\\setlength{\\itemsep}{0pt}\n'
            '\\setlength{\\parsep}{0pt}\n'
            '\\setlength{\\parskip}{0pt}\n'
        )
        text = text[:bib_pos] + spacing_cmds + text[bib_pos:]

        end_doc = text.rfind(r'\end{document}')
        if end_doc > 0:
            text = text[:end_doc] + '\\endgroup\n' + text[end_doc:]

    # 3. Add \enlargethispage right before bibliography to stretch the
    #    references page and avoid a near-empty overflow page
    if r'\enlargethispage' not in text:
        bib_m2 = re.search(r'\\begingroup\s*\n\\footnotesize', text)
        if not bib_m2:
            bib_m2 = re.search(r'\\bibliographystyle\{', text)
        if bib_m2:
            text = (text[:bib_m2.start()]
                    + '\\enlargethispage{4\\baselineskip}\n'
                    + text[bib_m2.start():])

    return text
