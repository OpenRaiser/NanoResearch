"""LaTeX generation tool using Jinja2 templates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

# Template root: nanoresearch/nanoresearch/templates/
_TEMPLATES_ROOT = Path(__file__).resolve().parent.parent.parent / "nanoresearch" / "templates"


def _get_jinja_env(template_dir: Path | None = None) -> Environment:
    """Create a Jinja2 environment for LaTeX rendering."""
    base_dir = template_dir or _TEMPLATES_ROOT
    loader = FileSystemLoader([
        str(base_dir),
        str(base_dir / "base"),
    ])
    env = Environment(
        loader=loader,
        autoescape=select_autoescape([]),  # no HTML escaping for LaTeX
        block_start_string="<%",
        block_end_string="%>",
        variable_start_string="<<",
        variable_end_string=">>",
        comment_start_string="<#",
        comment_end_string="#>",
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return env


def generate_latex(
    template_name: str,
    data: dict[str, Any],
    template_format: str = "arxiv",
    template_dir: Path | None = None,
) -> str:
    """Render a LaTeX document from a Jinja2 template.

    Args:
        template_name: Template file name (e.g. "paper.tex.j2").
        data: Template variables.
        template_format: "arxiv", "neurips", or "icml".
        template_dir: Override template directory.

    Returns:
        Rendered LaTeX string.
    """
    env = _get_jinja_env(template_dir)

    # Try format-specific template first, fall back to base
    candidates = [
        f"{template_format}/{template_name}",
        f"base/{template_name}",
        template_name,
    ]
    template = None
    for candidate in candidates:
        try:
            template = env.get_template(candidate)
            break
        except Exception:
            continue
    if template is None:
        raise FileNotFoundError(
            f"Template '{template_name}' not found in {candidates}"
        )
    return template.render(**data)


def generate_full_paper(data: dict[str, Any], template_format: str = "arxiv") -> str:
    """Generate a complete paper LaTeX document.

    Args:
        data: Must include 'title', 'authors', 'abstract', 'sections',
              'references_bibtex'. Sections is a list of dicts with
              'heading' and 'content' keys.
        template_format: Target venue format.

    Returns:
        Complete LaTeX document string.
    """
    return generate_latex("paper.tex.j2", data, template_format)
