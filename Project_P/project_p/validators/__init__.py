"""Validators — cross-reference and citation consistency checks."""
from __future__ import annotations

from pathlib import Path

from .crossref import fix_crossrefs, fix_includegraphics_paths
from ..fixers.figures import remove_empty_figure_blocks


def run_all_checks(
    tex: str, bib: str, figures_dir: Path | None = None,
) -> tuple[str, list[str]]:
    """Run all validators; return (fixed_tex, fix_descriptions)."""
    fixes: list[str] = []

    tex, ref_fixes = fix_crossrefs(tex, bib)
    fixes.extend(ref_fixes)

    if figures_dir and figures_dir.exists():
        tex, fig_fixes = fix_includegraphics_paths(tex, figures_dir)
        fixes.extend(fig_fixes)

    # Clean up figure blocks left empty by includegraphics path fixing
    tex = remove_empty_figure_blocks(tex)

    return tex, fixes
