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
    eq_label_pattern = re.compile(
        r"\\begin\{(?:equation|align|gather)\*?\}.*?\\label\{([^}]+)\}.*?\\end\{(?:equation|align|gather)\*?\}",
        re.DOTALL,
    )
    eq_labels = set(eq_label_pattern.findall(tex))

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

def validate_equations_sympy(tex: str) -> list[dict]:
    """Attempt to parse LaTeX equations with SymPy.

    Returns issues for equations that fail to parse. Requires ``sympy``.
    """
    try:
        from sympy.parsing.latex import parse_latex  # type: ignore[import-untyped]
    except ImportError:
        return []

    issues: list[dict] = []

    # Extract inline and display math
    equations = re.findall(
        r"\\begin\{(?:equation|align)\*?\}(.*?)\\end\{(?:equation|align)\*?\}",
        tex,
        re.DOTALL,
    )

    for i, eq_text in enumerate(equations):
        # Clean up alignment markers
        clean = eq_text.replace("&", "").replace("\\\\", "").strip()
        # Remove label/tag commands
        clean = re.sub(r"\\(?:label|tag|nonumber)\{[^}]*\}", "", clean)
        clean = clean.strip()
        if not clean:
            continue
        try:
            parse_latex(clean)
        except Exception as e:
            issues.append({
                "issue_type": "unparseable_equation",
                "description": f"Equation {i+1} failed SymPy parsing: {e}",
                "locations": _find_lines(tex, clean[:40]) if len(clean) >= 5 else [],
                "severity": "low",
            })

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
            issues.append({
                "issue_type": "unmatched_braces",
                "description": (
                    f"Line {lineno} has {'extra opening' if depth > 0 else 'extra closing'} "
                    f"brace(s) (net imbalance: {depth:+d})"
                ),
                "locations": [f"line {lineno}"],
                "severity": "high",
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

    # Also mask tabular / table environments (& is legal there)
    masked = re.sub(
        r"\\begin\{(?:tabular|tabularx|array|matrix|pmatrix|bmatrix|cases)\*?\}"
        r".*?"
        r"\\end\{(?:tabular|tabularx|array|matrix|pmatrix|bmatrix|cases)\*?\}",
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
