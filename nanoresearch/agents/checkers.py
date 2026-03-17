"""LaTeX consistency and math formula checkers.

Pure functions that analyse a LaTeX source string and return lists of
issue dicts compatible with ``ConsistencyIssue`` construction::

    {"issue_type": str, "description": str, "locations": list[str], "severity": str}
"""

from __future__ import annotations

import re
from collections import Counter

# Re-export helpers so existing imports continue to work
from nanoresearch.agents._checkers_helpers import (  # noqa: F401
    check_ai_writing_patterns,
    _find_lines,
    _AI_PHRASES,
    _AI_OVERUSED_WORDS,
    _AI_PHRASE_PATTERNS,
)


# ---------------------------------------------------------------------------
# LaTeX structural consistency
# ---------------------------------------------------------------------------

def check_latex_consistency(tex: str) -> list[dict]:
    """Check LaTeX source for structural consistency issues."""
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
    """Check math formulas for basic consistency issues."""
    issues: list[dict] = []

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
    """Clean LaTeX equation text for SymPy parsing."""
    clean = eq_text
    clean = clean.replace("&", "").replace("\\\\", "")
    clean = re.sub(r"\\(?:label|tag|nonumber|notag)\{[^}]*\}", "", clean)
    clean = re.sub(r"\\(?:text|textbf|textit|textrm|mathrm|mathit|mathbf|boldsymbol)\{([^}]*)\}", r"\1", clean)
    clean = re.sub(r"\\(?:left|right|big|Big|bigg|Bigg)([|.()[\]{}]?)", r"\1", clean)
    clean = re.sub(r"\\(?:[,;!:]|quad|qquad|hspace\{[^}]*\}|vspace\{[^}]*\})", " ", clean)
    clean = clean.replace("\\limits", "")
    clean = clean.replace("\\displaystyle", "")
    clean = re.sub(r"\\phantom\{[^}]*\}", "", clean)
    clean = re.sub(r"\\(?:underbrace|overbrace)\{([^}]*)\}_\{[^}]*\}", r"\1", clean)
    clean = re.sub(r"\s+", " ", clean)
    return clean.strip()


def validate_equations_sympy(tex: str) -> list[dict]:
    """Validate LaTeX equations with SymPy for structural correctness."""
    try:
        from sympy.parsing.latex import parse_latex  # type: ignore[import-untyped]
    except ImportError:
        return []

    issues: list[dict] = []
    equations = re.findall(
        r"\\begin\{(?:equation|align|gather|multline)\*?\}"
        r"(.*?)"
        r"\\end\{(?:equation|align|gather|multline)\*?\}",
        tex, re.DOTALL,
    )
    display_math = re.findall(r"\$\$(.*?)\$\$", tex, re.DOTALL)
    equations.extend(display_math)
    tested: set[str] = set()

    for i, eq_text in enumerate(equations):
        clean = _clean_equation_for_sympy(eq_text)
        if not clean or len(clean) < 3:
            continue
        lines = [l.strip() for l in clean.split("\n") if l.strip()]
        if not lines:
            lines = [clean]
        for line in lines:
            if not any(c in line for c in ("+", "-", "=", "\\", "^", "_", "/")):
                continue
            if line in tested:
                continue
            tested.add(line)
            parts = line.split("=")
            for part_idx, part in enumerate(parts):
                part = part.strip()
                if not part or len(part) < 2:
                    continue
                if re.match(r"^[\d\s.]+$", part):
                    continue
                try:
                    parse_latex(part)
                except Exception as e:
                    err_str = str(e)
                    if any(fp in err_str.lower() for fp in ("unexpected", "expected", "parsing", "don't understand")):
                        if any(kw in err_str.lower() for kw in ("brace", "bracket", "unexpected end", "missing", "unmatched", "expected something else", "don't understand")):
                            issues.append({
                                "issue_type": "equation_syntax_error",
                                "description": f"Equation {i+1} has a structural issue: {err_str}. Fragment: '{part[:60]}'",
                                "locations": _find_lines(tex, eq_text[:40]) if len(eq_text) >= 5 else [],
                                "severity": "medium",
                            })
                        else:
                            issues.append({
                                "issue_type": "unparseable_equation",
                                "description": f"Equation {i+1} uses notation SymPy can't parse: {err_str}. Fragment: '{part[:60]}'",
                                "locations": _find_lines(tex, eq_text[:40]) if len(eq_text) >= 5 else [],
                                "severity": "low",
                            })
        if len(issues) >= 10:
            break

    return issues


# ---------------------------------------------------------------------------
# Brace balance check
# ---------------------------------------------------------------------------

def check_unmatched_braces(tex: str) -> list[dict]:
    """Detect lines with unmatched ``{`` / ``}`` braces."""
    issues: list[dict] = []
    for lineno, line in enumerate(tex.splitlines(), 1):
        stripped = line.lstrip()
        if stripped.startswith("%"):
            continue
        cleaned = line.replace("\\{", "").replace("\\}", "")
        depth = 0
        for ch in cleaned:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
        if depth != 0:
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
    r"\$[^$]+\$"
    r"|\$\$[^$]+\$\$"
    r"|\\begin\{(?:equation|align|gather|math|displaymath)\*?\}.*?"
    r"\\end\{(?:equation|align|gather|math|displaymath)\*?\}"
    r")",
    re.DOTALL,
)

_BARE_SPECIAL_RE = re.compile(r"(?<!\\)([&#])")


def check_bare_special_chars(tex: str) -> list[dict]:
    """Find bare ``&`` and ``#`` outside math environments and tables."""
    issues: list[dict] = []
    masked = _MATH_ENV_RE.sub(lambda m: " " * len(m.group()), tex)
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

_UNICODE_MAP = {
    "\u2018": "`", "\u2019": "'", "\u201c": "``", "\u201d": "''",
    "\u2013": "--", "\u2014": "---", "\u2026": "\\ldots",
    "\u00e9": "\\'e", "\u00e8": "\\`e",
    "\u00fc": '\\"u', "\u00f6": '\\"o', "\u00e4": '\\"a',
}

_NON_ASCII_RE = re.compile(r"[^\x00-\x7F]")


def check_unicode_issues(tex: str) -> list[dict]:
    """Detect non-ASCII characters that may break LaTeX compilation."""
    issues: list[dict] = []
    seen_chars: set[str] = set()
    for lineno, line in enumerate(tex.splitlines(), 1):
        stripped = line.lstrip()
        if stripped.startswith("%"):
            continue
        for match in _NON_ASCII_RE.finditer(line):
            char = match.group()
            if char in seen_chars:
                continue
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
