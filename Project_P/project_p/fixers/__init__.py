"""LaTeX fixers — deterministic repairs for common paper formatting issues."""
from __future__ import annotations

import logging
from pathlib import Path

from .latex_sanitizer import sanitize_latex
from .environments import fix_environments
from .floats import fix_floats
from .figures import fix_figure_placement
from .structure import fix_structure
from .bibtex import fix_bibtex
from .figure_trim import trim_all_figures

logger = logging.getLogger(__name__)


def run_all_fixes(
    tex: str,
    bib: str,
    figures_dir: Path | None,
) -> tuple[str, list[str]]:
    """Run all deterministic LaTeX fixes in order.

    Returns (fixed_tex, list_of_fix_descriptions).
    """
    fixes: list[str] = []

    original = tex

    tex = sanitize_latex(tex)
    if tex != original:
        fixes.append("latex_sanitizer: character-level fixes applied")

    prev = tex
    tex = fix_environments(tex)
    if tex != prev:
        fixes.append("environments: fixed mismatched/double environments")

    prev = tex
    tex = fix_floats(tex)
    if tex != prev:
        fixes.append("floats: normalized placement specifiers and table overflow")

    prev = tex
    tex = fix_figure_placement(tex, figures_dir)
    if tex != prev:
        fixes.append("figures: fixed figure placement/relocation/missing")

    prev = tex
    tex = fix_structure(tex)
    if tex != prev:
        fixes.append("structure: fixed \\end{document}, contributions, blank lines")

    logger.info("Applied %d fix categories", len(fixes))
    return tex, fixes
