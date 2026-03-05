"""Template discovery utilities for pluggable LaTeX templates.

Each template is a subdirectory under ``nanoresearch/templates/`` containing at
least a ``paper.tex.j2`` Jinja2 template.  Templates may optionally include a
``styles/`` subdirectory with ``.sty``, ``.cls``, and ``.bst`` files that will
be copied into the compilation directory automatically.
"""

from __future__ import annotations

from pathlib import Path

_TEMPLATES_DIR = Path(__file__).parent


def get_available_formats() -> list[str]:
    """Auto-discover available template formats.

    Scans for subdirectories (excluding ``base``) that contain a
    ``paper.tex.j2`` file.
    """
    formats: list[str] = []
    for d in sorted(_TEMPLATES_DIR.iterdir()):
        if d.is_dir() and d.name != "base" and not d.name.startswith("__") and (d / "paper.tex.j2").exists():
            formats.append(d.name)
    return formats


def get_style_files(template_format: str) -> list[Path]:
    """Return ``.sty`` / ``.cls`` / ``.bst`` files bundled with *template_format*."""
    styles_dir = _TEMPLATES_DIR / template_format / "styles"
    if not styles_dir.is_dir():
        return []
    return [f for f in sorted(styles_dir.iterdir()) if f.suffix in (".sty", ".cls", ".bst")]
