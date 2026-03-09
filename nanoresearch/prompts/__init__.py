"""Prompt template loader — loads YAML prompt files with optional variable substitution.

Usage:
    from nanoresearch.prompts import load_prompt, get_prompt_version

    prompt = load_prompt("writing", "introduction")
    prompt = load_prompt("writing", "method", variables={"contribution_guidance": "..."})
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

_PROMPTS_DIR = Path(__file__).parent
_CACHE: dict[str, dict] = {}


def load_prompt(category: str, name: str, variables: dict | None = None) -> str:
    """Load a prompt template and optionally fill variables.

    Args:
        category: Subdirectory (e.g., "writing", "review", "ideation")
        name: File name without .yaml extension
        variables: Dict of {placeholder: value} for string formatting

    Returns:
        Rendered prompt string.
    """
    cache_key = f"{category}/{name}"
    if cache_key not in _CACHE:
        path = _PROMPTS_DIR / category / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Prompt template not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            _CACHE[cache_key] = yaml.safe_load(f)

    template = _CACHE[cache_key]
    prompt_text: str = template.get("prompt", template.get("system_prompt", ""))

    if variables:
        for key, value in variables.items():
            prompt_text = prompt_text.replace(f"{{{key}}}", str(value))

    return prompt_text


def get_prompt_version(category: str, name: str) -> Optional[str]:
    """Get version string of a prompt template."""
    load_prompt(category, name)  # Ensure cached
    return _CACHE.get(f"{category}/{name}", {}).get("version")


def clear_cache() -> None:
    """Clear the prompt cache (useful for testing or hot-reloading)."""
    _CACHE.clear()
