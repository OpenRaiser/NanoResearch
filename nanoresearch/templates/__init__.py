"""Template management for different paper formats."""

import os
from pathlib import Path


def get_available_formats() -> list[str]:
    """Get list of available template formats.

    Returns:
        List of available format names (e.g., ['arxiv', 'neurips', 'icml'])
    """
    templates_dir = Path(__file__).parent
    formats = []

    for item in templates_dir.iterdir():
        if item.is_dir() and not item.name.startswith('_') and not item.name.startswith('.'):
            # Check if it's a valid template directory (has paper.tex.j2)
            if (item / "paper.tex.j2").exists():
                formats.append(item.name)

    return sorted(formats)


def get_template_path(format_name: str) -> Path:
    """Get the path to a specific template format.

    Args:
        format_name: Name of the format (e.g., 'neurips', 'icml', 'arxiv')

    Returns:
        Path to the template directory

    Raises:
        ValueError: If format is not available
    """
    templates_dir = Path(__file__).parent
    template_path = templates_dir / format_name

    if not template_path.exists() or not (template_path / "paper.tex.j2").exists():
        available = get_available_formats()
        raise ValueError(f"Unknown format '{format_name}'. Available: {available}")

    return template_path
