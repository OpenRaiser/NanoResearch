"""Single-model paper review and citation fact-checking mixin."""

from __future__ import annotations

import logging
import re
from typing import Any

from nanoresearch.schemas.review import (
    ReviewOutput,
    SectionReview,
)
from nanoresearch.skill_prompts import get_review_system_prompt

logger = logging.getLogger(__name__)


class _SingleReviewMixin:
    """Mixin — single-model review and citation fact-checking methods."""

    # ── Citation fact-checking helpers ────────────────────────────────

    @staticmethod
    def _build_bibtex_key_to_paper_map(
        paper_tex: str, papers: list,
    ) -> dict[str, dict]:
        """Build mapping from BibTeX cite key -> paper dict (with title/abstract).

        Matches BibTeX entries in the paper to papers from ideation
        by title similarity.
        """
        if not isinstance(papers, list) or not papers:
            return {}

        # Extract bibtex entries: key -> title
        # Regex handles one level of nested braces: title = {{Nested}} or {A {B} C}
        bib_entries: dict[str, str] = {}
        for m in re.finditer(
            r'@\w+\s*\{\s*([^,\s]+)\s*,.*?title\s*=\s*\{((?:[^{}]|\{[^{}]*\})*)\}',
            paper_tex, re.DOTALL | re.IGNORECASE,
        ):
            title = m.group(2).strip().lower()
            # Strip outer braces from BibTeX-style {{Title}}
            if title.startswith("{") and title.endswith("}"):
                title = title[1:-1].strip()
            bib_entries[m.group(1).strip()] = title

        if not bib_entries:
            return {}

        # Build title -> paper dict index from ideation papers
        title_to_paper: dict[str, dict] = {}
        for p in papers:
            if isinstance(p, dict) and p.get("title"):
                title_to_paper[p["title"].lower().strip()] = p

        # Match bibtex entries to papers by title
        result: dict[str, dict] = {}
        for key, bib_title in bib_entries.items():
            # Exact match first
            if bib_title in title_to_paper:
                result[key] = title_to_paper[bib_title]
                continue
            # Fuzzy: check if bib_title is a significant substring
            for ptitle, paper in title_to_paper.items():
                if len(bib_title) > 10 and (
                    bib_title in ptitle or ptitle in bib_title
                ):
                    result[key] = paper
                    break

        return result

    async def _review_single_section(
        self,
        heading: str,
        content: str,
        ideation_output: dict,
        experiment_blueprint: dict,
        review_config: Any,
    ) -> SectionReview:
        """Review a single section of the paper.

        Returns a SectionReview with detailed feedback including strengths
        (to be preserved during revision) and structured issues.
        """
        # Per-section specialized system prompt
        section_review_system = get_review_system_prompt(heading)

        prompt = f"""Review the following section of an academic paper for a top-tier AI venue.

Section: {heading}

```latex
{content[:12000]}
```

Research context:
- Topic: {str(ideation_output.get('topic', 'Unknown'))[:500]}
- Hypothesis: {str(ideation_output.get('selected_hypothesis', 'Unknown'))[:500]}
- Method: {str((experiment_blueprint.get('proposed_method') or {{}}).get('name', 'Unknown'))[:500]}

Provide a thorough review with:
1. **Score** (1-10): Use the rubric strictly. Justify your score.
2. **Strengths** (up to 3): What this section does WELL — these must be PRESERVED during any revision.
3. **Must-fix issues** (up to 5): Each must state: [PROBLEM] what is wrong -> [IMPACT] why it matters -> [FIX] specific action to take.
4. **Optional suggestions** (up to 3): Nice-to-have improvements.

IMPORTANT scoring guidelines:
- 9-10: Publication-ready, only cosmetic tweaks needed
- 7-8: Solid work with minor fixable issues
- 5-6: Significant problems but recoverable
- 3-4: Major rewrite needed
- 1-2: Fundamentally flawed
- Score should reflect SEVERITY, not just the NUMBER of issues
  (1 critical flaw like missing experiments > 5 minor typos)
- Justify your score — explain which specific issues drive it down

Return JSON:
{{
    "section": "{heading}",
    "score": 7,
    "score_justification": "Brief explanation of why this score",
    "strengths": ["Strength 1: what is good and must be preserved"],
    "issues": ["Issue 1: [PROBLEM] ... [IMPACT] ... [FIX] ..."],
    "suggestions": ["Suggestion 1"]
}}"""

        try:
            result = await self.generate_json(
                section_review_system, prompt, stage_override=review_config
            )
        except Exception:
            # JSON parse failed — try repair
            raw = await self.generate(
                section_review_system, prompt, json_mode=True,
                stage_override=review_config,
            )
            raw = raw or ""
            result = self._repair_truncated_json(raw)
            if result is None or isinstance(result, list):
                logger.warning("Could not parse review for section %s, using defaults", heading)
                result = {"section": heading, "score": 5, "issues": [], "suggestions": []}

        # Safely coerce score to int (LLM may return float or string)
        raw_score = result.get("score", 5)
        try:
            score = max(1, min(10, int(float(raw_score))))
        except (TypeError, ValueError):
            score = 5

        # Coerce issues/strengths to list[str] — LLM may return list[dict]
        raw_issues = result.get("issues", [])[:5]
        issues = []
        for item in raw_issues:
            if isinstance(item, str):
                issues.append(item)
            elif isinstance(item, dict):
                # Flatten dict like {"issue": "...", "impact": "...", "fix": "..."}
                parts = [f"[{k.upper()}] {v}" for k, v in item.items() if v]
                issues.append(" -> ".join(parts) if parts else str(item))
            else:
                issues.append(str(item))

        raw_strengths = result.get("strengths", [])[:3]
        strengths = [s if isinstance(s, str) else str(s) for s in raw_strengths]

        # Coerce suggestions to list[str] — same dict issue as issues
        raw_suggestions = result.get("suggestions", [])[:3]
        suggestions = []
        for item in raw_suggestions:
            if isinstance(item, str):
                suggestions.append(item)
            elif isinstance(item, dict):
                parts = [f"[{k.upper()}] {v}" for k, v in item.items() if v]
                suggestions.append(" -> ".join(parts) if parts else str(item))
            else:
                suggestions.append(str(item))

        # BUG-6 fix: when issues is empty but score is low, preserve the
        # original low score to convey severity to the revision LLM.
        # The downstream Step 2b (line ~145) will inject generic issues
        # for sections below MIN_SECTION_SCORE, so the section won't be
        # skipped — but keeping the real score gives the revision LLM a
        # stronger signal about how much improvement is needed.
        # (Previously this inflated score to 7, masking severity.)

        sr = SectionReview(
            section=result.get("section", heading),
            score=max(1, min(10, score)),
            issues=issues,
            suggestions=suggestions,
            strengths=strengths,
            score_justification=result.get("score_justification", ""),
        )
        return sr

    async def _review_paper(
        self,
        paper_tex: str,
        ideation_output: dict,
        experiment_blueprint: dict,
    ) -> ReviewOutput:
        """Review the paper section-by-section to avoid JSON truncation."""
        sections = self._extract_sections(paper_tex)
        review_config = self.config.for_stage("review")

        # Build top-level sections with full content (including subsections).
        # _extract_sections splits at every \section/\subsection boundary,
        # so a \section{Method} followed by \subsection{...} would only contain
        # the intro paragraph. We merge subsection content back into the parent.
        main_sections: list[tuple[str, str]] = []

        # Extract abstract for review (it's in \begin{abstract}...\end{abstract},
        # not in \section{}, so _extract_sections misses it)
        abs_match = re.search(
            r'\\begin\{abstract\}(.*?)\\end\{abstract\}',
            paper_tex, re.DOTALL,
        )
        if abs_match:
            main_sections.append(("Abstract", abs_match.group(1).strip()))

        for h, _c, level in sections:
            if level == 0:
                merged = self._get_full_section_content(sections, h)
                main_sections.append((h, merged))

        # If no main sections found, use all (capped at 10 to avoid runaway)
        if not main_sections:
            main_sections = [(h, c) for h, c, _lvl in sections[:10]]

        # Review each section individually
        section_reviews: list[SectionReview] = []
        for heading, content in main_sections:  # No artificial cap
            try:
                sr = await self._review_single_section(
                    heading, content, ideation_output,
                    experiment_blueprint, review_config,
                )
                section_reviews.append(sr)
                self.log(f"  Reviewed '{heading}': score={sr.score}")
            except Exception as e:
                logger.warning("Failed to review section '%s': %s", heading, e)
                section_reviews.append(SectionReview(
                    section=heading, score=5, issues=[str(e)], suggestions=[],
                ))

        # Generate overall assessment with tool-augmented verification
        overall_score = (
            sum(sr.score for sr in section_reviews) / len(section_reviews)
            if section_reviews else 5.0
        )

        major_revisions = []
        minor_revisions = []
        for sr in section_reviews:
            if sr.score < 5:
                major_revisions.extend(sr.issues[:2])
            elif sr.score < 7:
                minor_revisions.extend(sr.suggestions[:2])

        return ReviewOutput(
            overall_score=overall_score,
            section_reviews=section_reviews,
            major_revisions=major_revisions,
            minor_revisions=minor_revisions,
        )
