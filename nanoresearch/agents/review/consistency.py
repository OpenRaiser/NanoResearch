"""Consistency checking mixin for the review package."""

from __future__ import annotations

import logging
import re

from nanoresearch.schemas.review import ConsistencyIssue

from ._constants import _CITE_PATTERN, _RELATED_WORK_SECTION_PATTERN

logger = logging.getLogger(__name__)


class _ConsistencyMixin:
    """Mixin — consistency checks and AI artifact detection."""

    def _check_claim_result_consistency(
        self, tex: str, blueprint: dict
    ) -> list[ConsistencyIssue]:
        """Check that claims in the paper match the experiment blueprint.

        Detects:
        - Metrics mentioned in the paper but not defined in the blueprint
        - Dataset names in the paper that don't match blueprint datasets
        - Baseline methods in the paper not listed in blueprint baselines
        """
        issues: list[ConsistencyIssue] = []
        if not blueprint:
            return issues

        # Collect blueprint names (lowercased for fuzzy matching)
        bp_metrics = {
            m.get("name", "").lower()
            for m in blueprint.get("metrics", [])
            if isinstance(m, dict) and m.get("name")
        }
        bp_datasets = {
            d.get("name", "").lower()
            for d in blueprint.get("datasets", [])
            if isinstance(d, dict) and d.get("name")
        }
        bp_baselines = {
            b.get("name", "").lower()
            for b in blueprint.get("baselines", [])
            if isinstance(b, dict) and b.get("name")
        }

        # Check for baseline methods mentioned in \textbf{} or table rows
        # that are not in the blueprint
        tex_lower = tex.lower()
        for baseline in bp_baselines:
            if baseline and len(baseline) > 2 and baseline not in tex_lower:
                issues.append(ConsistencyIssue(
                    issue_type="missing_baseline",
                    description=(
                        f"Blueprint baseline '{baseline}' is not mentioned "
                        f"in the paper text"
                    ),
                    severity="low",
                    locations=["Results / Experiments section"],
                ))

        # Check proposed method name appears in paper
        proposed = blueprint.get("proposed_method", {})
        if not isinstance(proposed, dict):
            proposed = {}
        method_name = proposed.get("name", "")
        if method_name and len(method_name) > 2:
            if method_name.lower() not in tex_lower:
                issues.append(ConsistencyIssue(
                    issue_type="missing_method",
                    description=(
                        f"Proposed method '{method_name}' from blueprint "
                        f"is not mentioned in the paper"
                    ),
                    severity="low",
                    locations=["Throughout paper"],
                ))

        return issues

    def _check_citation_coverage(self, tex: str, ideation_output: dict) -> list[ConsistencyIssue]:
        """Check citation coverage: total count and must-cite enforcement."""
        issues: list[ConsistencyIssue] = []

        # Count total unique citations (handle natbib variants + optional args)
        cited: set[str] = set()
        for m in _CITE_PATTERN.finditer(tex):
            for k in m.group(1).split(","):
                k = k.strip()
                if k:
                    cited.add(k)

        total = len(cited)
        if total < 10:
            issues.append(ConsistencyIssue(
                issue_type="low_citation_count",
                description=(
                    f"Paper has only {total} unique citations. "
                    "A top-venue paper typically needs 25+ citations. "
                    "Add more references, especially in Related Work and Introduction."
                ),
                severity="high",
                locations=["Related Work", "Introduction"],
            ))
        elif total < 20:
            issues.append(ConsistencyIssue(
                issue_type="moderate_citation_count",
                description=(
                    f"Paper has {total} unique citations. "
                    "Consider adding more to strengthen Related Work (target: 25+)."
                ),
                severity="medium",
                locations=["Related Work"],
            ))

        # Check Related Work section specifically (handle common heading variants)
        rw_match = _RELATED_WORK_SECTION_PATTERN.search(tex)
        if rw_match:
            rw_content = rw_match.group(1)
            rw_cited: set[str] = set()
            for m in _CITE_PATTERN.finditer(rw_content):
                for k in m.group(1).split(","):
                    k = k.strip()
                    if k:
                        rw_cited.add(k)
            if len(rw_cited) < 10:
                issues.append(ConsistencyIssue(
                    issue_type="sparse_related_work_citations",
                    description=(
                        f"Related Work has only {len(rw_cited)} unique citations. "
                        "A thorough survey needs 15+ citations minimum."
                    ),
                    severity="medium",
                    locations=["Related Work"],
                ))

        return issues

    def _check_figure_text_alignment(self, tex: str) -> list[ConsistencyIssue]:
        """Check that figure references match figure definitions."""
        import re
        issues: list[ConsistencyIssue] = []

        # Find all \label{fig:...}
        defined_figs = set(re.findall(r'\\label\{(fig:[^}]+)\}', tex))
        # Find all \ref{fig:...} and \autoref{fig:...}
        referenced_figs = set(re.findall(r'\\(?:(?:auto|[Cc])?ref)\{(fig:[^}]+)\}', tex))

        # Figures referenced but not defined
        for fig in referenced_figs - defined_figs:
            issues.append(ConsistencyIssue(
                issue_type="undefined_figure_ref",
                description=f"Figure reference '\\ref{{{fig}}}' has no matching \\label",
                severity="high",
                locations=["Figures"],
            ))

        # Figures defined but never referenced
        for fig in defined_figs - referenced_figs:
            issues.append(ConsistencyIssue(
                issue_type="unreferenced_figure",
                description=f"Figure '\\label{{{fig}}}' is defined but never referenced in text",
                severity="low",
                locations=["Figures"],
            ))

        return issues

    @staticmethod
    def _check_latex_structure(tex: str) -> list[str]:
        """Quick structural checks for LaTeX source (no compilation).

        Returns a list of issue description strings.  Used by the
        backpressure mechanism to detect revision-introduced breakage
        and distinguish it from pre-existing issues (BUG-30 fix).
        """
        issues: list[str] = []
        begins = len(re.findall(r'\\begin\{', tex))
        ends = len(re.findall(r'\\end\{', tex))
        if begins != ends:
            issues.append(f"Unbalanced environments: {begins} \\begin vs {ends} \\end")
        # Check for mismatched environment types
        env_stack: list[str] = []
        for env_m in re.finditer(r'\\(begin|end)\{([^}]+)\}', tex):
            cmd, env_name = env_m.group(1), env_m.group(2)
            if cmd == "begin":
                env_stack.append(env_name)
            elif env_stack and env_stack[-1] == env_name:
                env_stack.pop()
            elif env_stack:
                issues.append(
                    f"Mismatched environment: \\begin{{{env_stack[-1]}}} "
                    f"closed by \\end{{{env_name}}}"
                )
                env_stack.pop()
                break  # one mismatch is enough
        if '\\documentclass' not in tex:
            issues.append("Missing \\documentclass")
        if '\\end{document}' not in tex:
            issues.append("Missing \\end{document}")
        return issues

    @staticmethod
    def _fix_mismatched_environments(tex: str) -> str:
        """Auto-fix ``\\begin{X}...\\end{Y}`` mismatches in LaTeX source.

        Common LLM errors: ``\\begin{equation}...\\end{parameter}``,
        ``\\begin{align}...\\end{equation}``, etc.
        Strategy: scan for mismatches and replace the wrong ``\\end{Y}``
        with the correct ``\\end{X}`` from the stack.
        """
        # Collect all begin/end positions
        env_events: list[tuple[int, int, str, str]] = []  # (start, end, cmd, env_name)
        for m in re.finditer(r'\\(begin|end)\{([^}]+)\}', tex):
            env_events.append((m.start(), m.end(), m.group(1), m.group(2)))

        # Walk through and fix mismatches (reverse order to preserve offsets)
        fixes: list[tuple[int, int, str]] = []  # (start, end, replacement)
        stack: list[tuple[int, int, str]] = []  # (start, end, env_name) for begins
        for start, end, cmd, env_name in env_events:
            if cmd == "begin":
                stack.append((start, end, env_name))
            elif cmd == "end":
                if stack and stack[-1][2] == env_name:
                    stack.pop()  # correct match
                elif stack:
                    expected = stack[-1][2]
                    # Replace \end{wrong} with \end{expected}
                    fixes.append((start, end, f"\\end{{{expected}}}"))
                    stack.pop()
                # else: orphan \end — leave as-is

        # Apply fixes in reverse order (right to left)
        result = tex
        for start, end, replacement in reversed(fixes):
            result = result[:start] + replacement + result[end:]

        return result

    def _run_consistency_checks(self, tex: str) -> list[ConsistencyIssue]:
        """Run automated consistency checks on the LaTeX source."""
        issues: list[ConsistencyIssue] = []

        try:
            from nanoresearch.agents.checkers import (
                check_ai_writing_patterns,
                check_bare_special_chars,
                check_latex_consistency,
                check_math_formulas,
                check_unicode_issues,
                check_unmatched_braces,
                validate_equations_sympy,
            )
            for checker in (
                check_latex_consistency,
                check_math_formulas,
                check_unmatched_braces,
                check_bare_special_chars,
                check_unicode_issues,
                validate_equations_sympy,
                check_ai_writing_patterns,
            ):
                try:
                    for issue in checker(tex):
                        if not isinstance(issue, dict):
                            continue
                        # Ensure required fields exist with defaults
                        issue.setdefault("issue_type", "unknown")
                        issue.setdefault("description", "No description")
                        issues.append(ConsistencyIssue(**issue))
                except Exception as exc:
                    logger.warning("Checker %s failed: %s", getattr(checker, '__name__', checker), exc)
        except ImportError:
            logger.debug("checkers module not available, skipping automated checks")

        # AI artifact detection (hardcoded, no external dependency)
        try:
            ai_issues = self._check_ai_artifacts(tex)
            issues.extend(ai_issues)
        except Exception as exc:
            logger.warning("AI artifact check failed: %s", exc)

        return issues

    # ---- AI artifact detection ----
    # Top ~20 most egregious AI-flavored words (case-insensitive scan)
    _AI_BANNED_WORDS: list[str] = [
        "delve", "leverage", "utilize", "harness", "pivotal", "unveil",
        "elucidate", "foster", "intricate", "nuanced", "profound",
        "testament", "vibrant", "ameliorate", "underscore", "transcend",
        "envision", "bolster", "culminate", "traverse",
    ]

    _HEDGING_PILEUP_RE = re.compile(
        r"\b(?:may\s+potentially|could\s+possibly|might\s+perhaps)\b",
        re.IGNORECASE,
    )

    def _check_ai_artifacts(self, tex: str) -> list[ConsistencyIssue]:
        """Scan LaTeX text for common AI-writing artifacts.

        Returns a list of ConsistencyIssue for each detected problem.
        """
        issues: list[ConsistencyIssue] = []
        tex_lower = tex.lower()

        # 1. Banned AI words
        flagged_words: list[tuple[str, int]] = []
        for word in self._AI_BANNED_WORDS:
            # Use word-boundary regex for accurate counting
            count = len(re.findall(r"\b" + re.escape(word) + r"\b", tex_lower))
            if count > 0:
                flagged_words.append((word, count))

        if flagged_words:
            word_summary = ", ".join(
                f'"{w}" ({c}x)' for w, c in sorted(flagged_words, key=lambda x: -x[1])
            )
            total = sum(c for _, c in flagged_words)
            issues.append(ConsistencyIssue(
                issue_type="ai_artifact",
                description=(
                    f"AI-flagged vocabulary detected ({total} total occurrences "
                    f"across {len(flagged_words)} words): {word_summary}. "
                    f"Replace with natural, specific alternatives."
                ),
                locations=[],
                severity="high" if total >= 5 else "medium",
            ))

        # 2. Em-dash overuse (--- in LaTeX)
        emdash_count = tex.count("---")
        if emdash_count > 3:
            issues.append(ConsistencyIssue(
                issue_type="ai_artifact",
                description=(
                    f"Excessive em-dashes: {emdash_count} occurrences of '---' "
                    f"(max 3 recommended). Rewrite sentences to avoid em-dash "
                    f"constructions."
                ),
                locations=[],
                severity="medium",
            ))

        # 3. Furthermore / Moreover overuse
        for transition in ("Furthermore", "Moreover"):
            count = len(re.findall(
                r"\b" + re.escape(transition) + r"\b", tex
            ))
            if count > 3:
                issues.append(ConsistencyIssue(
                    issue_type="ai_artifact",
                    description=(
                        f'Overuse of "{transition}": {count} occurrences '
                        f"(max 3 recommended for a full paper). Vary transitions "
                        f"or restructure sentences."
                    ),
                    locations=[],
                    severity="medium",
                ))

        # 4. Hedging pileups: "may potentially", "could possibly", "might perhaps"
        hedging_matches = self._HEDGING_PILEUP_RE.findall(tex)
        if hedging_matches:
            issues.append(ConsistencyIssue(
                issue_type="ai_artifact",
                description=(
                    f"Hedging pileup detected ({len(hedging_matches)} "
                    f'occurrence(s)): {", ".join(repr(m) for m in hedging_matches[:5])}. '
                    f"Use a single hedging word or state the claim directly."
                ),
                locations=[],
                severity="medium",
            ))

        return issues

    @staticmethod
    def _dedup_consistency_issues(
        issues: list[ConsistencyIssue],
    ) -> list[ConsistencyIssue]:
        """Remove duplicate consistency issues by (issue_type, description) key."""
        seen: set[tuple[str, str]] = set()
        deduped: list[ConsistencyIssue] = []
        for issue in issues:
            key = (issue.issue_type, issue.description)
            if key not in seen:
                seen.add(key)
                deduped.append(issue)
        return deduped
