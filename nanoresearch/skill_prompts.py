"""Per-section / per-task system prompts for every pipeline agent.

Design principles:
  1. Each section or sub-task gets a SPECIALIZED system prompt (not one generic prompt).
  2. Prompt text lives in YAML files under nanoresearch/prompts/ — loaded once, cached.
  3. This module provides the same public API as before (dicts, getter functions, constants),
     so existing code imports are unchanged.
  4. User prompts carry the data/context; system prompts carry the persona/rules.
"""

from __future__ import annotations

from nanoresearch.prompts import load_prompt

# ═══════════════════════════════════════════════════════════════════════════════
# WRITING — per-section system prompts (loaded from prompts/writing/*.yaml)
# ═══════════════════════════════════════════════════════════════════════════════
WRITING_SYSTEM_PROMPTS: dict[str, str] = {
    "Introduction": load_prompt("writing", "introduction"),
    "Related Work": load_prompt("writing", "related_work"),
    "Method": load_prompt("writing", "method"),
    "Experiments": load_prompt("writing", "experiments"),
    "Conclusion": load_prompt("writing", "conclusion"),
}

_WRITING_DEFAULT = load_prompt("writing", "_default")


def get_writing_system_prompt(section_heading: str) -> str:
    """Return the specialized system prompt for a writing section."""
    for key, prompt in WRITING_SYSTEM_PROMPTS.items():
        if key.lower() in section_heading.lower():
            return prompt
    return _WRITING_DEFAULT


# ═══════════════════════════════════════════════════════════════════════════════
# REVIEW — per-section system prompts (loaded from prompts/review/*.yaml)
# ═══════════════════════════════════════════════════════════════════════════════
REVIEW_SYSTEM_PROMPTS: dict[str, str] = {
    "Abstract": load_prompt("review", "abstract"),
    "Introduction": load_prompt("review", "introduction"),
    "Related Work": load_prompt("review", "related_work"),
    "Method": load_prompt("review", "method"),
    "Experiments": load_prompt("review", "experiments"),
    "Conclusion": load_prompt("review", "conclusion"),
}

_REVIEW_DEFAULT = load_prompt("review", "_default")


def get_review_system_prompt(section_heading: str) -> str:
    """Return the specialized system prompt for reviewing a section."""
    for key, prompt in REVIEW_SYSTEM_PROMPTS.items():
        if key.lower() in section_heading.lower():
            return prompt
    return _REVIEW_DEFAULT


# ═══════════════════════════════════════════════════════════════════════════════
# IDEATION — per-task system prompts (loaded from prompts/ideation/*.yaml)
# ═══════════════════════════════════════════════════════════════════════════════
IDEATION_QUERY_SYSTEM = load_prompt("ideation", "query_generation")
IDEATION_ANALYSIS_SYSTEM = load_prompt("ideation", "analysis")
IDEATION_MUST_CITE_SYSTEM = load_prompt("ideation", "must_cite")
IDEATION_EVIDENCE_SYSTEM = load_prompt("ideation", "evidence")


# ═══════════════════════════════════════════════════════════════════════════════
# TITLE & ABSTRACT — specialized system prompts
# ═══════════════════════════════════════════════════════════════════════════════
TITLE_SYSTEM = load_prompt("writing", "title")
ABSTRACT_SYSTEM = load_prompt("writing", "abstract")


# ═══════════════════════════════════════════════════════════════════════════════
# REVISION — system prompt for the REVIEW agent's revision step
# ═══════════════════════════════════════════════════════════════════════════════
REVISION_SYSTEM = load_prompt("review", "revision")
