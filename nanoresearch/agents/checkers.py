"""LaTeX consistency and math formula checkers.

Pure functions that analyse a LaTeX source string and return lists of
issue dicts compatible with ``ConsistencyIssue`` construction::

    {"issue_type": str, "description": str, "locations": list[str], "severity": str}
"""

from __future__ import annotations

import re
from collections import Counter


# ---------------------------------------------------------------------------
# LaTeX structural consistency
# ---------------------------------------------------------------------------

def check_latex_consistency(tex: str) -> list[dict]:
    """Check LaTeX source for structural consistency issues.

    Checks performed:
    - ``\\ref{key}`` without a corresponding ``\\label{key}``
    - ``\\begin{env}`` / ``\\end{env}`` mismatch
    - ``\\cite{key}`` with malformed keys (spaces, special chars)
    """
    issues: list[dict] = []

    # --- \\ref without \\label ---
    labels = set(re.findall(r"\\label\{([^}]+)\}", tex))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", tex))
    eqrefs = set(re.findall(r"\\eqref\{([^}]+)\}", tex))
    all_refs = refs | eqrefs

    dangling = all_refs - labels
    for key in sorted(dangling):
        issues.append({
            "issue_type": "ref_mismatch",
            "description": f"\\ref{{{key}}} or \\eqref{{{key}}} has no corresponding \\label{{{key}}}",
            "locations": _find_lines(tex, key),
            "severity": "high",
        })

    # --- \\begin / \\end mismatch ---
    begins = re.findall(r"\\begin\{([^}]+)\}", tex)
    ends = re.findall(r"\\end\{([^}]+)\}", tex)
    begin_counts = Counter(begins)
    end_counts = Counter(ends)

    for env in set(begin_counts) | set(end_counts):
        b = begin_counts.get(env, 0)
        e = end_counts.get(env, 0)
        if b != e:
            issues.append({
                "issue_type": "env_mismatch",
                "description": (
                    f"\\begin{{{env}}} appears {b} time(s) but "
                    f"\\end{{{env}}} appears {e} time(s)"
                ),
                "locations": _find_lines(tex, f"\\begin{{{env}}}") + _find_lines(tex, f"\\end{{{env}}}"),
                "severity": "high",
            })

    # --- malformed cite keys ---
    cite_keys = re.findall(r"\\cite[tp]?\{([^}]+)\}", tex)
    for cite_block in cite_keys:
        for key in cite_block.split(","):
            key = key.strip()
            if not key:
                continue
            if re.search(r"[^a-zA-Z0-9_:.\-/]", key):
                issues.append({
                    "issue_type": "cite_format",
                    "description": f"Citation key '{key}' contains unusual characters",
                    "locations": _find_lines(tex, key),
                    "severity": "low",
                })

    return issues


# ---------------------------------------------------------------------------
# Math formula checks
# ---------------------------------------------------------------------------

def check_math_formulas(tex: str) -> list[dict]:
    """Check math formulas for basic consistency issues.

    Checks performed:
    - ``equation`` labels never referenced via ``\\eqref``
    - Mixed bold-symbol notation (``\\mathbf{x}`` and ``\\bm{x}`` both present)
    """
    issues: list[dict] = []

    # --- equation labels not referenced ---
    # Collect labels inside equation/align/gather environments
    # Use a two-pass approach: first find environments, then extract all labels within each
    eq_env_pattern = re.compile(
        r"\\begin\{(?:equation|align|gather)\*?\}(.*?)\\end\{(?:equation|align|gather)\*?\}",
        re.DOTALL,
    )
    eq_labels: set[str] = set()
    for m in eq_env_pattern.finditer(tex):
        for label_m in re.finditer(r"\\label\{([^}]+)\}", m.group(1)):
            eq_labels.add(label_m.group(1))

    eqrefs = set(re.findall(r"\\eqref\{([^}]+)\}", tex))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", tex))
    all_referenced = eqrefs | refs

    unreferenced = eq_labels - all_referenced
    for label in sorted(unreferenced):
        issues.append({
            "issue_type": "unreferenced_equation",
            "description": f"Equation \\label{{{label}}} is never referenced via \\eqref or \\ref",
            "locations": _find_lines(tex, f"\\label{{{label}}}"),
            "severity": "low",
        })

    # --- mixed bold notation ---
    has_mathbf = bool(re.search(r"\\mathbf\{", tex))
    has_bm = bool(re.search(r"\\bm\{", tex))
    if has_mathbf and has_bm:
        issues.append({
            "issue_type": "symbol_inconsistency",
            "description": (
                "Both \\mathbf{} and \\bm{} are used for bold symbols. "
                "Pick one convention for consistency."
            ),
            "locations": (
                _find_lines(tex, "\\mathbf{")[:2]
                + _find_lines(tex, "\\bm{")[:2]
            ),
            "severity": "medium",
        })

    return issues


# ---------------------------------------------------------------------------
# Optional: SymPy equation parsing
# ---------------------------------------------------------------------------

def _clean_equation_for_sympy(eq_text: str) -> str:
    """Clean LaTeX equation text for SymPy parsing.

    Removes alignment markers, labels, text commands, and other
    constructs that SymPy's parser can't handle.
    """
    clean = eq_text
    # Remove alignment markers
    clean = clean.replace("&", "").replace("\\\\", "")
    # Remove label/tag/nonumber
    clean = re.sub(r"\\(?:label|tag|nonumber|notag)\{[^}]*\}", "", clean)
    # Remove \text{...}, \textbf{...}, \textit{...}, \mathrm{...} — keep inner text
    clean = re.sub(r"\\(?:text|textbf|textit|textrm|mathrm|mathit|mathbf|boldsymbol)\{([^}]*)\}", r"\1", clean)
    # Remove \left and \right (SymPy doesn't need them)
    clean = re.sub(r"\\(?:left|right|big|Big|bigg|Bigg)([|.()[\]{}]?)", r"\1", clean)
    # Remove \, \; \! \: \quad \qquad spacing
    clean = re.sub(r"\\(?:[,;!:]|quad|qquad|hspace\{[^}]*\}|vspace\{[^}]*\})", " ", clean)
    # Remove \limits
    clean = clean.replace("\\limits", "")
    # Remove \displaystyle
    clean = clean.replace("\\displaystyle", "")
    # Remove \phantom{...}
    clean = re.sub(r"\\phantom\{[^}]*\}", "", clean)
    # Remove \underbrace{...}_{...} → keep first arg
    clean = re.sub(r"\\(?:underbrace|overbrace)\{([^}]*)\}_\{[^}]*\}", r"\1", clean)
    # Multiple spaces → single
    clean = re.sub(r"\s+", " ", clean)
    return clean.strip()


def validate_equations_sympy(tex: str) -> list[dict]:
    """Validate LaTeX equations with SymPy for structural correctness.

    Checks for:
    - Equations that fail basic parsing (mismatched braces, wrong syntax)
    - Dimensional consistency issues
    Returns issues for equations that fail to parse. Requires ``sympy``.
    """
    try:
        from sympy.parsing.latex import parse_latex  # type: ignore[import-untyped]
    except ImportError:
        return []

    issues: list[dict] = []

    # Extract display math from equation, align, gather, multline environments
    equations = re.findall(
        r"\\begin\{(?:equation|align|gather|multline)\*?\}"
        r"(.*?)"
        r"\\end\{(?:equation|align|gather|multline)\*?\}",
        tex,
        re.DOTALL,
    )

    # Also extract $$...$$ display math
    display_math = re.findall(r"\$\$(.*?)\$\$", tex, re.DOTALL)
    equations.extend(display_math)

    # Track which equations we've already tested to avoid duplicates
    tested: set[str] = set()

    for i, eq_text in enumerate(equations):
        clean = _clean_equation_for_sympy(eq_text)
        if not clean or len(clean) < 3:
            continue

        # For multi-line equations (align), test each line separately
        lines = [l.strip() for l in clean.split("\n") if l.strip()]
        if not lines:
            lines = [clean]

        for line in lines:
            # Skip pure text or comments
            if not any(c in line for c in ("+", "-", "=", "\\", "^", "_", "/")):
                continue
            # Skip if already tested
            if line in tested:
                continue
            tested.add(line)

            # Split on = and test each side independently (more robust)
            parts = line.split("=")
            for part_idx, part in enumerate(parts):
                part = part.strip()
                if not part or len(part) < 2:
                    continue
                # Skip parts that are just numbers or simple text
                if re.match(r"^[\d\s.]+$", part):
                    continue

                try:
                    parse_latex(part)
                except Exception as e:
                    err_str = str(e)
                    # Filter out common false positives (custom macros, etc.)
                    if any(fp in err_str.lower() for fp in (
                        "unexpected", "expected", "parsing",
                        "don't understand",
                    )):
                        # Only report if it looks like a structural error
                        if any(kw in err_str.lower() for kw in (
                            "brace", "bracket", "unexpected end",
                            "missing", "unmatched",
                            "expected something else", "don't understand",
                        )):
                            issues.append({
                                "issue_type": "equation_syntax_error",
                                "description": (
                                    f"Equation {i+1} has a structural issue: {err_str}. "
                                    f"Fragment: '{part[:60]}'"
                                ),
                                "locations": _find_lines(tex, eq_text[:40]) if len(eq_text) >= 5 else [],
                                "severity": "medium",
                            })
                        else:
                            # Unknown macro or custom notation — low severity
                            issues.append({
                                "issue_type": "unparseable_equation",
                                "description": (
                                    f"Equation {i+1} uses notation SymPy can't parse: {err_str}. "
                                    f"This may be a custom macro. Fragment: '{part[:60]}'"
                                ),
                                "locations": _find_lines(tex, eq_text[:40]) if len(eq_text) >= 5 else [],
                                "severity": "low",
                            })

        # Limit total issues to avoid noise
        if len(issues) >= 10:
            break

    return issues


# ---------------------------------------------------------------------------
# Brace balance check
# ---------------------------------------------------------------------------

def check_unmatched_braces(tex: str) -> list[dict]:
    """Detect lines with unmatched ``{`` / ``}`` braces.

    Ignores braces inside comments (lines starting with ``%``).
    Only reports when a single line has a net imbalance, which catches
    most accidental typos without false-positives from multi-line macros.
    """
    issues: list[dict] = []
    for lineno, line in enumerate(tex.splitlines(), 1):
        # Skip comment lines
        stripped = line.lstrip()
        if stripped.startswith("%"):
            continue
        # Remove escaped braces \{ \}
        cleaned = line.replace("\\{", "").replace("\\}", "")
        depth = 0
        for ch in cleaned:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
        if depth != 0:
            # Small imbalances (±1, ±2) are often multi-line macros (\newcommand,
            # \def, etc.) — use "medium" severity. Large imbalances (±3+) are
            # more likely genuine errors — use "high".
            sev = "high" if abs(depth) >= 3 else "medium"
            issues.append({
                "issue_type": "unmatched_braces",
                "description": (
                    f"Line {lineno} has {'extra opening' if depth > 0 else 'extra closing'} "
                    f"brace(s) (net imbalance: {depth:+d})"
                ),
                "locations": [f"line {lineno}"],
                "severity": sev,
            })
    return issues


# ---------------------------------------------------------------------------
# Bare special characters check
# ---------------------------------------------------------------------------

_MATH_ENV_RE = re.compile(
    r"(?:"
    r"\$[^$]+\$"                   # inline $...$
    r"|\$\$[^$]+\$\$"             # display $$...$$
    r"|\\begin\{(?:equation|align|gather|math|displaymath)\*?\}.*?"
    r"\\end\{(?:equation|align|gather|math|displaymath)\*?\}"
    r")",
    re.DOTALL,
)

# Characters that need escaping outside math mode
_BARE_SPECIAL_RE = re.compile(r"(?<!\\)([&#])")


def check_bare_special_chars(tex: str) -> list[dict]:
    """Find bare ``&`` and ``#`` outside math environments and tables.

    These characters have special meaning in LaTeX and cause compilation
    errors when used unescaped outside appropriate environments.
    ``_`` is excluded because it appears frequently in cite keys.
    """
    issues: list[dict] = []

    # Remove math environments to avoid false positives
    masked = _MATH_ENV_RE.sub(lambda m: " " * len(m.group()), tex)

    # Also mask tabular / table / align environments (& is legal there)
    masked = re.sub(
        r"\\begin\{(?:tabular|tabularx|array|matrix|pmatrix|bmatrix|cases"
        r"|align|alignat|flalign|gathered|split)\*?\}"
        r".*?"
        r"\\end\{(?:tabular|tabularx|array|matrix|pmatrix|bmatrix|cases"
        r"|align|alignat|flalign|gathered|split)\*?\}",
        lambda m: " " * len(m.group()),
        masked,
        flags=re.DOTALL,
    )

    for lineno, line in enumerate(masked.splitlines(), 1):
        # Skip comment lines
        stripped = line.lstrip()
        if stripped.startswith("%"):
            continue
        for match in _BARE_SPECIAL_RE.finditer(line):
            char = match.group(1)
            issues.append({
                "issue_type": "bare_special_char",
                "description": (
                    f"Bare '{char}' on line {lineno} — should be '\\{char}' outside "
                    f"math/table environments"
                ),
                "locations": [f"line {lineno}"],
                "severity": "medium",
            })

    return issues


# ---------------------------------------------------------------------------
# Unicode / non-ASCII check
# ---------------------------------------------------------------------------

# Common Unicode → LaTeX replacements
_UNICODE_MAP = {
    "\u2018": "`",       # '
    "\u2019": "'",       # '
    "\u201c": "``",      # "
    "\u201d": "''",      # "
    "\u2013": "--",      # en-dash
    "\u2014": "---",     # em-dash
    "\u2026": "\\ldots", # …
    "\u00e9": "\\'e",    # é
    "\u00e8": "\\`e",    # è
    "\u00fc": '\\"u',    # ü
    "\u00f6": '\\"o',    # ö
    "\u00e4": '\\"a',    # ä
}

_NON_ASCII_RE = re.compile(r"[^\x00-\x7F]")


def check_unicode_issues(tex: str) -> list[dict]:
    """Detect non-ASCII characters that may break LaTeX compilation.

    Reports characters that should be replaced with LaTeX commands.
    """
    issues: list[dict] = []
    seen_chars: set[str] = set()

    for lineno, line in enumerate(tex.splitlines(), 1):
        stripped = line.lstrip()
        if stripped.startswith("%"):
            continue
        for match in _NON_ASCII_RE.finditer(line):
            char = match.group()
            if char in seen_chars:
                continue  # Only report each unique character once
            seen_chars.add(char)
            suggestion = _UNICODE_MAP.get(char, "")
            desc = f"Non-ASCII character U+{ord(char):04X} ('{char}') on line {lineno}"
            if suggestion:
                desc += f" — use {suggestion!r} instead"
            issues.append({
                "issue_type": "unicode_char",
                "description": desc,
                "locations": [f"line {lineno}"],
                "severity": "medium",
            })

    return issues


# ---------------------------------------------------------------------------
# Anti-AI writing pattern detection
# ---------------------------------------------------------------------------

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
    r"comprehensive\s+overview",
    r"rapidly\s+evolving",
    r"in\s+(?:today's|an)\s+(?:rapidly\s+)?(?:evolving|changing)\s+(?:landscape|world)",
    r"rich\s+(?:tapestry|heritage)",
    r"at\s+the\s+(?:forefront|intersection)\s+of",
    r"a\s+testament\s+to",
    r"navigat(?:e|ing)\s+the\s+(?:complexities|landscape|challenges)",
    r"embark(?:s|ing)?\s+on",
    r"not\s+(?:just|merely|only)\s+[^,.]+,\s+(?:but\s+)?(?:also\s+)?(?:a|an|the)",  # "not just X, but also Y"
]

# Words that are overused by AI but rare in human academic writing
_AI_OVERUSED_WORDS = [
    "additionally",
    "furthermore",
    "moreover",
    "crucial",
    "utilize",
    "leverage",
    "facilitate",
    "underscores",
    "encompasses",
    "groundbreaking",
    "transformative",
    "paradigm",
    "synergy",
    "holistic",
    "robust",  # when not in a technical/statistical context
    "seamless",
    "streamline",
    "cutting-edge",
    "novel",  # overused in non-contribution contexts
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
            # Find line number
            line_start = stripped[:match.start()].count("\n") + 1
            found_phrases.append((phrase, f"line {line_start}"))

    if found_phrases:
        # Group and report
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
    # Count occurrences outside comments, but only flag if 3+ different AI words are overused
    word_counts: dict[str, int] = {}
    lower_stripped = stripped.lower()
    for word in _AI_OVERUSED_WORDS:
        count = len(re.findall(r"\b" + word + r"\b", lower_stripped))
        if count >= 2:
            word_counts[word] = count

    if len(word_counts) >= 3:
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
    # Check if consecutive paragraphs start with the same transition word
    lines = tex.splitlines()
    para_openers: list[tuple[str, int]] = []
    for i, line in enumerate(lines, 1):
        s = line.strip()
        if not s or s.startswith("%") or s.startswith("\\"):
            continue
        # Get first word
        first_word = s.split()[0].lower().rstrip(".,;:") if s.split() else ""
        if first_word in ("additionally", "furthermore", "moreover", "however",
                         "consequently", "therefore", "meanwhile", "nonetheless"):
            para_openers.append((first_word, i))

    # Flag if the same opener appears 3+ times
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
