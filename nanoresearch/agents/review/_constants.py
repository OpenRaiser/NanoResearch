"""Shared constants, regex patterns, and free functions for the review package."""

from __future__ import annotations

import re

from nanoresearch.skill_prompts import get_review_system_prompt, REVISION_SYSTEM

# ---- Shared BibTeX helpers (BUG-4/22/26 fix) ----

_CONFERENCE_KEYWORDS = frozenset({
    "neurips", "nips", "icml", "iclr", "cvpr", "iccv", "eccv",
    "acl", "emnlp", "naacl", "aaai", "ijcai", "sigir", "kdd",
    "chi", "uist", "sigmod", "vldb", "www", "cikm", "wsdm",
    "proceedings", "conference", "workshop", "symposium",
})


def _detect_bib_entry_type(venue: str) -> tuple[str, str]:
    """Return (bibtex_type, venue_field) based on venue name."""
    if not venue:
        return "article", "journal"
    venue_lower = venue.lower()
    if any(kw in venue_lower for kw in _CONFERENCE_KEYWORDS):
        return "inproceedings", "booktitle"
    return "article", "journal"


MAX_REVISION_ROUNDS = 5
MAX_LATEX_FIX_ATTEMPTS = 3  # compile-fix loop iterations
MIN_SECTION_SCORE = 8  # Sections scoring below this get revised
CONVERGENCE_THRESHOLD = 0.3  # Stop if avg score improves by less than this

# ---- Pre-compiled regex patterns (R2 perf fix) ----

_SECTION_PATTERN = re.compile(
    r"\\((?:sub){0,2})section\*?\{"
    r"((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})+)"
    r"\}",
)

_CITE_PATTERN = re.compile(r"\\[Cc]ite[tp]?(?:\w*)(?:\*)?(?:\[[^\]]*\])*\{([^}]+)\}")

_RELATED_WORK_SECTION_PATTERN = re.compile(
    r'\\section\{(?:Related Works?|Prior Work|Literature Review'
    r'|Background(?:\s+and\s+Related\s+Work)?)\}'
    r'(.*?)(?=\\section\{|\\end\{document\})',
    re.DOTALL | re.IGNORECASE,
)

_ABSTRACT_PATTERN = re.compile(
    r'(\\begin\{abstract\})(.*?)(\\end\{abstract\})',
    re.DOTALL,
)

# Generic fallback (used by compile-fix and other non-section calls)
REVIEW_SYSTEM_PROMPT = get_review_system_prompt("_default")
REVISION_SYSTEM_PROMPT = REVISION_SYSTEM
