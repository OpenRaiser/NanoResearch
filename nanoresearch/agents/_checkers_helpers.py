"""Checker helpers — anti-AI writing patterns and utility functions."""

from __future__ import annotations

import re
from collections import Counter

# Reuse math env pattern from checkers.py
_MATH_ENV_RE = re.compile(
    r"(?:"
    r"\$[^$]+\$"
    r"|\$\$[^$]+\$\$"
    r"|\\begin\{(?:equation|align|gather|math|displaymath)\*?\}.*?"
    r"\\end\{(?:equation|align|gather|math|displaymath)\*?\}"
    r")",
    re.DOTALL,
)

# Phrases that are strong indicators of AI-generated text
_AI_PHRASES = [
    r"delve\s+into",
    r"it\s+is\s+worth\s+noting\s+that",
    r"in\s+the\s+realm\s+of",
    r"harness(?:ing)?\s+the\s+power\s+of",
    r"pave(?:s|d)?\s+the\s+way",
    r"shed(?:s|ding)?\s+light\s+on",
    r"play(?:s|ing)?\s+a\s+(?:crucial|pivotal|vital)\s+role",
    r"stand(?:s|ing)?\s+as\s+a\s+testament",
    r"serves?\s+as\s+a\s+cornerstone",
    r"a\s+myriad\s+of",
    r"in\s+(?:today's|an)\s+(?:rapidly\s+)?(?:evolving|changing)\s+(?:landscape|world)",
    r"rich\s+(?:tapestry|heritage)",
    r"a\s+testament\s+to",
    r"navigat(?:e|ing)\s+the\s+(?:complexities|landscape|challenges)",
    r"embark(?:s|ing)?\s+on",
]

# Words that are overused by AI but rare in human academic writing
_AI_OVERUSED_WORDS = [
    "utilize", "leverage", "facilitate", "underscores", "encompasses",
    "groundbreaking", "transformative", "synergy", "holistic", "seamless",
    "streamline", "cutting-edge", "delve",
]

_AI_PHRASE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _AI_PHRASES]


def check_ai_writing_patterns(tex: str) -> list[dict]:
    """Detect AI-generated writing patterns in LaTeX source.

    Checks for:
    - Banned AI phrases (strong indicators)
    - Overused AI vocabulary (weak indicators, only flag if many)
    - Formulaic paragraph openings (consecutive same transition words)
    """
    issues: list[dict] = []

    # Strip comments and math environments to avoid false positives
    stripped = _MATH_ENV_RE.sub(lambda m: " " * len(m.group()), tex)

    # --- Banned AI phrases ---
    found_phrases: list[tuple[str, str]] = []
    for pat in _AI_PHRASE_PATTERNS:
        for match in pat.finditer(stripped):
            phrase = match.group()
            line_start = stripped[:match.start()].count("\n") + 1
            found_phrases.append((phrase, f"line {line_start}"))

    if found_phrases:
        examples = found_phrases[:5]
        desc_parts = [f"'{p}' ({loc})" for p, loc in examples]
        issues.append({
            "issue_type": "ai_writing_pattern",
            "description": (
                f"Found {len(found_phrases)} AI-typical phrase(s): "
                + "; ".join(desc_parts)
                + (f" (and {len(found_phrases) - 5} more)" if len(found_phrases) > 5 else "")
                + ". Replace with direct, specific language."
            ),
            "locations": [loc for _, loc in examples],
            "severity": "medium",
        })

    # --- Overused AI vocabulary ---
    word_counts: dict[str, int] = {}
    lower_stripped = stripped.lower()
    for word in _AI_OVERUSED_WORDS:
        count = len(re.findall(r"\b" + word + r"\b", lower_stripped))
        if count >= 3:
            word_counts[word] = count

    if len(word_counts) >= 4:
        top_words = sorted(word_counts.items(), key=lambda x: -x[1])[:5]
        desc = ", ".join(f"'{w}' ({c}x)" for w, c in top_words)
        issues.append({
            "issue_type": "ai_vocabulary",
            "description": (
                f"High density of AI-typical vocabulary: {desc}. "
                "Consider replacing with more specific alternatives."
            ),
            "locations": [],
            "severity": "low",
        })

    # --- Formulaic paragraph openings ---
    lines = tex.splitlines()
    para_openers: list[tuple[str, int]] = []
    for i, line in enumerate(lines, 1):
        s = line.strip()
        if not s or s.startswith("%") or s.startswith("\\"):
            continue
        first_word = s.split()[0].lower().rstrip(".,;:") if s.split() else ""
        if first_word in ("additionally", "furthermore", "moreover", "however",
                         "consequently", "therefore", "meanwhile", "nonetheless"):
            para_openers.append((first_word, i))

    opener_counts = Counter(w for w, _ in para_openers)
    for word, count in opener_counts.items():
        if count >= 3:
            issues.append({
                "issue_type": "repetitive_transitions",
                "description": (
                    f"'{word.title()}' used to start {count} paragraphs. "
                    "Vary transitions or remove them — topic sentences "
                    "often work better without transition words."
                ),
                "locations": [f"line {ln}" for _, ln in para_openers if _ == word][:3],
                "severity": "low",
            })

    return issues


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_lines(tex: str, needle: str, max_hits: int = 3) -> list[str]:
    """Return up to *max_hits* ``"line N"`` location strings."""
    locations: list[str] = []
    for lineno, line in enumerate(tex.splitlines(), 1):
        if needle in line:
            locations.append(f"line {lineno}")
            if len(locations) >= max_hits:
                break
    return locations
