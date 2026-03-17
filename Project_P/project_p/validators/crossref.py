r"""Cross-reference validation and auto-fix.

Checks:
1. \ref{X} ↔ \label{X} consistency (with fuzzy matching)
2. \cite{X} ↔ BibTeX key consistency
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


def fix_crossrefs(tex: str, bib: str) -> tuple[str, list[str]]:
    """Validate and fix cross-references. Returns (fixed_tex, fix_descriptions)."""
    fixes: list[str] = []

    tex, ref_fixes = _fix_ref_label_mismatches(tex)
    fixes.extend(ref_fixes)

    tex, sec_fixes = _inject_missing_section_labels(tex)
    fixes.extend(sec_fixes)

    tex, cite_fixes = _fix_cite_mismatches(tex, bib)
    fixes.extend(cite_fixes)

    return tex, fixes


def _fix_ref_label_mismatches(tex: str) -> tuple[str, list[str]]:
    r"""Fix \ref{X} without matching \label{X} via fuzzy matching."""
    fixes: list[str] = []

    # Extract all labels and refs
    labels = set(re.findall(r'\\label\{([^}]+)\}', tex))
    refs = set(re.findall(r'\\(?:ref|autoref|cref|eqref)\{([^}]+)\}', tex))

    unmatched_refs = refs - labels
    if not unmatched_refs:
        return tex, fixes

    for ref_key in unmatched_refs:
        best_match = _fuzzy_match_label(ref_key, labels)
        if best_match:
            # Replace this ref with the matched label
            tex = re.sub(
                rf'(\\(?:ref|autoref|cref|eqref))\{{{re.escape(ref_key)}\}}',
                rf'\1{{{best_match}}}',
                tex,
            )
            fixes.append(f"crossref: \\ref{{{ref_key}}} → \\ref{{{best_match}}}")
            logger.info("Fixed ref: %s → %s", ref_key, best_match)
        else:
            logger.warning("Unresolved ref: \\ref{%s} has no matching label", ref_key)

    return tex, fixes


# Semantic aliases: ref suffix → set of label suffixes that are equivalent
_LABEL_ALIASES: dict[str, set[str]] = {
    "architecture": {"method_detail", "method", "framework", "model", "pipeline", "overview"},
    "framework": {"architecture", "method_detail", "model", "pipeline", "overview"},
    "model": {"architecture", "method_detail", "framework"},
    "overview": {"problem_overview", "architecture", "framework"},
    "results": {"main_results", "results_comparison", "baseline_main_results"},
    "comparison": {"results_comparison", "main_results", "baseline_main_results"},
}


def _fuzzy_match_label(ref_key: str, labels: set[str]) -> str | None:
    """Try to match a ref key to available labels using suffix matching."""
    # Exact suffix match: fig:arch → fig:architecture
    for label in labels:
        if label.endswith(ref_key) or ref_key.endswith(label):
            return label

    # Strip prefix and try
    ref_parts = ref_key.split(":")
    ref_suffix = ref_parts[-1] if len(ref_parts) > 1 else ref_key

    for label in labels:
        label_parts = label.split(":")
        label_suffix = label_parts[-1] if len(label_parts) > 1 else label

        # Same prefix type (fig:, tab:, sec:, eq:)
        if len(ref_parts) > 1 and len(label_parts) > 1 and ref_parts[0] == label_parts[0]:
            # Substring containment
            if ref_suffix in label_suffix or label_suffix in ref_suffix:
                return label

    # Levenshtein-like: try common typos (underscore vs hyphen)
    normalized_ref = ref_key.replace("-", "_").lower()
    for label in labels:
        normalized_label = label.replace("-", "_").lower()
        if normalized_ref == normalized_label:
            return label

    # Semantic alias matching (e.g., "fig:architecture" → "fig:method_detail")
    # Only match labels with the same prefix type (fig: → fig:, tab: → tab:)
    ref_prefix = ref_parts[0] if len(ref_parts) > 1 else ""
    ref_suffix_lower = ref_suffix.lower() if ref_suffix else ref_key.lower()
    aliases = _LABEL_ALIASES.get(ref_suffix_lower, set())
    if aliases and ref_prefix:
        for label in labels:
            label_parts = label.split(":")
            label_prefix = label_parts[0] if len(label_parts) > 1 else ""
            if label_prefix != ref_prefix:
                continue  # skip cross-type matches (fig: vs sec:)
            label_suffix = (label_parts[-1] if len(label_parts) > 1 else label).lower()
            if label_suffix in aliases:
                return label

    return None


def _inject_missing_section_labels(tex: str) -> tuple[str, list[str]]:
    r"""Auto-inject \label{sec:X} when \ref{sec:X} exists but no \label{sec:X}.

    Finds the section/subsection/paragraph heading whose title best matches
    the label suffix and injects the label after the heading.
    """
    fixes: list[str] = []
    labels = set(re.findall(r'\\label\{([^}]+)\}', tex))
    refs = set(re.findall(r'\\(?:ref|autoref|cref|eqref)\{([^}]+)\}', tex))

    # Only handle sec: prefixed refs
    missing_sec_refs = {r for r in refs - labels if r.startswith("sec:")}
    if not missing_sec_refs:
        return tex, fixes

    # Build index of section/subsection/paragraph headings
    heading_pat = re.compile(
        r'(\\(?:section|subsection|subsubsection|paragraph)\*?'
        r'\{((?:[^{}]|\{[^{}]*\})*)\})'
        r'(\s*(?:\\label\{[^}]+\})?)',
    )
    headings = list(heading_pat.finditer(tex))

    for sec_ref in sorted(missing_sec_refs):
        suffix = sec_ref[4:]  # strip "sec:"
        suffix_lower = suffix.lower().replace("_", " ").replace("-", " ")
        suffix_words = set(suffix_lower.split())

        best_heading = None
        best_score = 0
        for hm in headings:
            title = hm.group(2)
            title_lower = title.lower().replace("_", " ").replace("-", " ")
            # Clean LaTeX commands from title for matching
            title_clean = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', title_lower)
            title_clean = re.sub(r'\\[a-zA-Z]+', '', title_clean).strip()
            title_words = set(title_clean.split())

            # Score: substring match or word overlap
            score = 0
            if suffix_lower in title_clean or title_clean in suffix_lower:
                score = 10
            else:
                overlap = len(suffix_words & title_words)
                if overlap > 0:
                    score = overlap

            if score > best_score:
                best_score = score
                best_heading = hm

        if best_heading and best_score > 0:
            # Check if this heading already has a label
            existing_label = re.search(
                r'\\label\{[^}]+\}',
                best_heading.group(3),
            )
            if existing_label:
                continue

            insert_pos = best_heading.end()
            label_str = f"\\label{{{sec_ref}}}"
            tex = tex[:insert_pos] + label_str + tex[insert_pos:]
            fixes.append(f"crossref: injected \\label{{{sec_ref}}} at "
                         f"\\{best_heading.group(0)[:40]}...")
            logger.info("Injected \\label{%s} at heading: %s",
                        sec_ref, best_heading.group(2)[:50])
            # Re-parse headings since positions shifted
            headings = list(heading_pat.finditer(tex))

    return tex, fixes


def _fix_cite_mismatches(tex: str, bib: str) -> tuple[str, list[str]]:
    r"""Check \cite{X} keys against BibTeX entries."""
    fixes: list[str] = []

    if not bib.strip():
        return tex, fixes

    # Extract bib keys
    bib_keys = set(re.findall(r'@\w+\s*\{\s*([^,\s]+)\s*,', bib))

    # Extract all cite keys (handle multi-cite: \cite{a,b,c})
    cite_matches = re.findall(r'\\cite[tp]?\{([^}]+)\}', tex)
    cited_keys: set[str] = set()
    for match in cite_matches:
        for key in match.split(","):
            cited_keys.add(key.strip())

    undefined_cites = cited_keys - bib_keys
    if not undefined_cites:
        return tex, fixes

    for cite_key in undefined_cites:
        # Try fuzzy match
        best = _fuzzy_match_bib_key(cite_key, bib_keys)
        if best:
            # Replace in all cite commands
            tex = re.sub(
                rf'(?<=[{{,])\s*{re.escape(cite_key)}\s*(?=[}},])',
                best,
                tex,
            )
            fixes.append(f"crossref: \\cite{{{cite_key}}} → \\cite{{{best}}}")
            logger.info("Fixed cite: %s → %s", cite_key, best)
        else:
            logger.warning("Undefined cite key: %s", cite_key)

    return tex, fixes


def _fuzzy_match_bib_key(cite_key: str, bib_keys: set[str]) -> str | None:
    """Try to match a cite key to available bib keys."""
    # Exact case-insensitive match
    key_lower = cite_key.lower()
    for bk in bib_keys:
        if bk.lower() == key_lower:
            return bk

    # Substring containment (min length match)
    for bk in bib_keys:
        shorter = min(cite_key, bk, key=len)
        longer = max(cite_key, bk, key=len)
        if shorter.lower() in longer.lower() and len(shorter) >= max(3, len(longer) // 2):
            return bk

    return None


# ── \includegraphics file existence check ────────────────────────────────────

def fix_includegraphics_paths(tex: str, figures_dir) -> tuple[str, list[str]]:
    r"""Fix \includegraphics paths that reference non-existent files.

    Tries to fuzzy-match against actual files in figures_dir.
    """
    from pathlib import Path
    fixes: list[str] = []
    figures_dir = Path(figures_dir)

    # Collect actual image files (exclude compile output like paper.pdf)
    _EXCLUDE_STEMS = {"paper", "main", "output", "draft"}
    actual_files: dict[str, str] = {}  # stem_lower → actual filename
    for ext in ("*.png", "*.pdf", "*.jpg", "*.jpeg"):
        for f in figures_dir.glob(ext):
            if f.stem.lower() not in _EXCLUDE_STEMS:
                actual_files[f.stem.lower()] = f.name

    pattern = re.compile(r'(\\includegraphics(?:\[[^\]]*\])?)\{([^}]+)\}')

    def _fix_path(m: re.Match) -> str:
        cmd = m.group(1)
        ref_path = m.group(2)
        ref_name = Path(ref_path).name
        ref_stem = Path(ref_name).stem.lower()

        # Check exact match
        if ref_stem in actual_files:
            actual = actual_files[ref_stem]
            if ref_name != actual:
                fixes.append(f"includegraphics: {ref_path} -> {actual}")
                return f"{cmd}{{{actual}}}"
            return m.group(0)

        # Fuzzy match
        best = _fuzzy_match_filename(ref_stem, actual_files)
        if best:
            actual = actual_files[best]
            fixes.append(f"includegraphics: {ref_path} -> {actual}")
            return f"{cmd}{{{actual}}}"

        # Truly missing — comment out to prevent compile crash
        logger.warning("Missing figure file: %s", ref_path)
        fixes.append(f"includegraphics: {ref_path} (MISSING, commented out)")
        return f"% MISSING: {cmd}{{{ref_path}}}"

    tex = pattern.sub(_fix_path, tex)
    return tex, fixes


def _fuzzy_match_filename(stem: str, actual_files: dict[str, str]) -> str | None:
    """Fuzzy match a filename stem against actual files."""
    # Substring containment
    for actual_stem in actual_files:
        if stem in actual_stem or actual_stem in stem:
            return actual_stem

    # Word overlap
    stem_words = set(stem.split("_"))
    best_overlap = 0
    best_match = None
    for actual_stem in actual_files:
        actual_words = set(actual_stem.split("_"))
        overlap = len(stem_words & actual_words)
        if overlap > best_overlap and overlap >= 2:
            best_overlap = overlap
            best_match = actual_stem

    return best_match
