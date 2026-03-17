"""Revision mixin — section revision, grounding, and the revision loop."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from nanoresearch.schemas.review import SectionReview

from ._constants import (
    CONVERGENCE_THRESHOLD,
    MAX_REVISION_ROUNDS,
    MIN_SECTION_SCORE,
    REVISION_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


class _RevisionMixin:
    """Mixin — revision loop and section revision methods."""

    async def _run_revision_loop(
        self,
        paper_tex: str,
        current_tex: str,
        review: Any,
        ideation_output: dict,
        experiment_blueprint: dict,
    ) -> str:
        """Run the revision loop with convergence detection.

        Modifies ``review`` in place. Returns updated ``current_tex``.
        """
        revision_round = 0
        # BUG-15 fix: initialize prev_avg_score from section_reviews average
        # (same metric used for new_avg), not overall_score (different metric).
        prev_avg_score = (
            sum(sr.score for sr in review.section_reviews)
            / len(review.section_reviews)
            if review.section_reviews else 0
        )
        # BUG-9 fix: track consecutive stalls to avoid infinite revision loops
        _stall_count = 0
        _MAX_STALL_ROUNDS = 2
        # BUG-30 fix: track consecutive backpressure reverts to prevent
        # infinite loops when the original paper_tex has structural errors
        # (e.g. \begin{equation}...\end{parameter}).
        _bp_revert_count = 0
        _MAX_BP_REVERTS = 2
        # Pre-check: detect structural issues in the original paper_tex
        # so backpressure can distinguish pre-existing vs revision-introduced.
        _original_bp_issues = self._check_latex_structure(paper_tex)

        # Auto-fix pre-existing mismatched environments from WRITING stage
        if any("Mismatched environment" in i for i in _original_bp_issues):
            candidate_tex = self._fix_mismatched_environments(paper_tex)
            fixed_issues = self._check_latex_structure(candidate_tex)
            if len(fixed_issues) < len(_original_bp_issues):
                n_fixed = len(_original_bp_issues) - len(fixed_issues)
                self.log(
                    f"  Auto-fixed {n_fixed} pre-existing mismatched "
                    f"environment(s) from WRITING stage"
                )
                paper_tex = candidate_tex
                current_tex = candidate_tex
                _original_bp_issues = fixed_issues

        while revision_round < MAX_REVISION_ROUNDS:
            low_sections = [
                sr for sr in review.section_reviews if sr.score < MIN_SECTION_SCORE
            ]
            # Only continue if there are sections to revise.
            # Consistency issues alone cannot drive revision (no sections to modify),
            # they are informational and included in the revision prompt.
            if not low_sections:
                break

            revision_round += 1
            # Clear per-round revised sections to avoid stale accumulation
            round_revised: dict[str, str] = {}
            consistency = review.consistency_issues
            self.log(
                f"Revision round {revision_round}: "
                f"revising {len(low_sections)} sections, "
                f"{len(consistency)} consistency issues"
            )

            for section_review in low_sections:
                # Collect consistency issues relevant to this section
                # Include issues without locations as potentially relevant to any section
                section_consistency = [
                    ci for ci in consistency
                    if not getattr(ci, 'locations', []) or any(
                        section_review.section.lower() in loc.lower()
                        for loc in getattr(ci, 'locations', [])
                    )
                ]
                revised = await self._revise_section(
                    current_tex, section_review, ideation_output,
                    consistency_issues=section_consistency,
                )
                if revised:
                    round_revised[section_review.section] = revised
                    review.revised_sections[section_review.section] = revised

            # Apply this round's revisions to get updated tex for re-review
            if round_revised:
                current_tex = self._apply_revisions(current_tex, round_revised)

                # Backpressure: verify revision didn't break LaTeX structure
                bp_issues = self._check_latex_structure(current_tex)
                # BUG-30 fix: only revert if the revision INTRODUCED new
                # structural issues.  Pre-existing issues (from WRITING)
                # should not trigger revert — the revision didn't cause them.
                new_bp_issues = [i for i in bp_issues if i not in _original_bp_issues]

                # Auto-fix mismatched environments before reverting
                if new_bp_issues and any("Mismatched environment" in i for i in new_bp_issues):
                    fixed_tex = self._fix_mismatched_environments(current_tex)
                    fixed_issues = self._check_latex_structure(fixed_tex)
                    still_new = [i for i in fixed_issues if i not in _original_bp_issues]
                    if len(still_new) < len(new_bp_issues):
                        self.log(
                            f"  Auto-fixed {len(new_bp_issues) - len(still_new)} "
                            f"mismatched environment(s)"
                        )
                        current_tex = fixed_tex
                        new_bp_issues = still_new

                if new_bp_issues:
                    _bp_revert_count += 1
                    self.log(
                        f"  Backpressure FAILED (new issues): {new_bp_issues}, "
                        f"reverting round {revision_round} "
                        f"(consecutive reverts: {_bp_revert_count})"
                    )
                    for sec_name in round_revised:
                        review.revised_sections.pop(sec_name, None)
                    current_tex = self._apply_revisions(paper_tex, review.revised_sections)
                    if _bp_revert_count >= _MAX_BP_REVERTS:
                        self.log(
                            f"  {_bp_revert_count} consecutive backpressure reverts — "
                            f"stopping revision loop to avoid wasting LLM calls"
                        )
                        break
                    continue  # skip re-review, try next round
                else:
                    _bp_revert_count = 0  # reset on success
                    if bp_issues:
                        # Pre-existing structural issues — log but don't revert
                        self.log(
                            f"  Backpressure: {len(bp_issues)} pre-existing "
                            f"structural issue(s), not reverting"
                        )

            # Re-run all consistency checks after revision
            review.consistency_issues = self._run_consistency_checks(current_tex)
            review.consistency_issues.extend(
                self._check_claim_result_consistency(current_tex, experiment_blueprint)
            )
            review.consistency_issues.extend(
                self._check_figure_text_alignment(current_tex)
            )
            review.consistency_issues = self._dedup_consistency_issues(review.consistency_issues)
            if review.consistency_issues:
                self.log(f"  {len(review.consistency_issues)} consistency issues remain after revision")

            # Re-review revised sections with LLM
            re_review = await self._review_paper(current_tex, ideation_output, experiment_blueprint)

            # Monotonic score guarantee: if a section's score decreased after
            # revision, try meta-refine (diagnose + retry), then revert if still no good.
            sections_to_meta_refine: list[tuple[SectionReview, SectionReview, str]] = []
            for new_sr in re_review.section_reviews:
                for old_sr in review.section_reviews:
                    if old_sr.section != new_sr.section:
                        continue

                    if new_sr.score < old_sr.score:
                        # Score decreased — queue for meta-refine
                        failed_text = round_revised.get(old_sr.section, "")
                        if failed_text:
                            sections_to_meta_refine.append((old_sr, new_sr, failed_text))
                        else:
                            # No revision text to analyze, just revert
                            logger.warning(
                                "Score regression: '%s' %d -> %d, reverting",
                                old_sr.section, old_sr.score, new_sr.score,
                            )
                    else:
                        # Score maintained or improved — accept new review
                        old_sr.score = new_sr.score
                        old_sr.issues = new_sr.issues
                        old_sr.suggestions = new_sr.suggestions
                        if new_sr.strengths:
                            old_sr.strengths = new_sr.strengths
                    break

            # Meta-refine: diagnose failed revisions, retry with improved prompt
            reverted_any = False
            for old_sr, new_sr, failed_text in sections_to_meta_refine:
                self.log(
                    f"  '{old_sr.section}' score dropped {old_sr.score}->{new_sr.score}, "
                    f"running meta-refine"
                )
                refined = await self._meta_refine_revision(
                    current_tex, old_sr, new_sr, failed_text,
                    ideation_output,
                )
                if refined:
                    # Apply refined revision and re-score just this section
                    test_tex = self._apply_revisions(
                        paper_tex,
                        {**review.revised_sections, old_sr.section: refined},
                    )
                    sections_list = self._extract_sections(test_tex)
                    section_content = self._get_full_section_content(
                        sections_list, old_sr.section
                    )
                    if section_content:
                        review_config = self.config.for_stage("review")
                        rescore = await self._review_single_section(
                            old_sr.section, section_content,
                            ideation_output, experiment_blueprint, review_config,
                        )
                        if rescore.score >= old_sr.score:
                            # Meta-refine succeeded
                            self.log(
                                f"  '{old_sr.section}' meta-refine succeeded: "
                                f"{old_sr.score}->{rescore.score}"
                            )
                            old_sr.score = rescore.score
                            old_sr.issues = rescore.issues
                            old_sr.suggestions = rescore.suggestions
                            if rescore.strengths:
                                old_sr.strengths = rescore.strengths
                            round_revised[old_sr.section] = refined
                            review.revised_sections[old_sr.section] = refined
                            continue

                # Meta-refine failed or not attempted — revert
                self.log(
                    f"  '{old_sr.section}' meta-refine failed, reverting to original"
                )
                if old_sr.section in round_revised:
                    del round_revised[old_sr.section]
                    review.revised_sections.pop(old_sr.section, None)
                reverted_any = True

            # If we reverted any sections, re-apply revisions from scratch
            if reverted_any:
                current_tex = self._apply_revisions(paper_tex, review.revised_sections)
                review.consistency_issues = self._run_consistency_checks(current_tex)
                review.consistency_issues.extend(
                    self._check_claim_result_consistency(current_tex, experiment_blueprint)
                )
                review.consistency_issues.extend(
                    self._check_figure_text_alignment(current_tex)
                )
                review.consistency_issues = self._dedup_consistency_issues(review.consistency_issues)
            elif round_revised:
                # All sections accepted (no reverts) — update current_tex
                current_tex = self._apply_revisions(paper_tex, review.revised_sections)

            # Convergence check: stop if score barely improved or degraded
            new_avg = (
                sum(sr.score for sr in review.section_reviews)
                / len(review.section_reviews)
                if review.section_reviews else 0
            )
            improvement = new_avg - prev_avg_score
            self.log(
                f"  Round {revision_round} score: {new_avg:.1f} "
                f"(delta: {improvement:+.1f})"
            )
            if improvement < CONVERGENCE_THRESHOLD:
                _stall_count += 1
                # BUG-9 fix: if stalled for _MAX_STALL_ROUNDS consecutive
                # rounds, stop even if sections are still below threshold.
                still_low = [
                    sr for sr in review.section_reviews
                    if sr.score < MIN_SECTION_SCORE
                ]
                if _stall_count >= _MAX_STALL_ROUNDS:
                    self.log(
                        f"  Stalled for {_stall_count} rounds — stopping "
                        f"({len(still_low)} section(s) still below threshold)"
                    )
                    break
                elif still_low and revision_round < MAX_REVISION_ROUNDS:
                    self.log(
                        f"  Improvement stalled ({improvement:.2f} < "
                        f"{CONVERGENCE_THRESHOLD}), but {len(still_low)} section(s) "
                        f"still below {MIN_SECTION_SCORE}: "
                        f"{[sr.section for sr in still_low]}. Continuing."
                    )
                else:
                    self.log(
                        f"  Convergence reached (improvement {improvement:.2f} < "
                        f"{CONVERGENCE_THRESHOLD}), stopping revision loop"
                    )
                    break
            else:
                _stall_count = 0  # reset on improvement
            prev_avg_score = new_avg

        review.revision_rounds = revision_round
        return current_tex

    async def _revise_section(
        self,
        paper_tex: str,
        section_review: SectionReview,
        ideation_output: dict,
        consistency_issues: list | None = None,
    ) -> str:
        """Revise a single section based on reviewer feedback.

        Uses the WRITING-stage LLM (typically a strong generation model) rather
        than the review-stage LLM, because revision is a generation task, not
        an evaluation task.
        """
        consistency_block = ""
        if consistency_issues:
            ci_texts = [
                f"- [{getattr(ci, 'issue_type', 'unknown')}] {getattr(ci, 'description', str(ci))}"
                for ci in consistency_issues[:20]  # Cap to avoid prompt overflow
            ]
            consistency_block = (
                "\n\nConsistency issues to fix:\n"
                + "\n".join(ci_texts)
            )

        # Truncate issues/suggestions to avoid prompt overflow
        issues_json = json.dumps(section_review.issues[:10], indent=2)
        suggestions_json = json.dumps(section_review.suggestions[:10], indent=2)

        # Extract strengths if available (set by _review_single_section)
        strengths = section_review.strengths
        strengths_block = ""
        if strengths:
            strengths_json = json.dumps(strengths, indent=2)
            strengths_block = (
                f"\n\nStrengths to PRESERVE (do NOT change these aspects):\n{strengths_json}\n"
                f"These were identified as good by the reviewer. Your revision must keep them intact."
            )

        # Section-specific revision guidance
        section_guidance = self._get_section_revision_guidance(section_review.section)

        # Extract bibliography from paper_tex so the LLM knows available citations
        bib_keys = ""
        bib_match = re.findall(r'\\bibitem\{([^}]+)\}|@\w+\{([^,]+),', paper_tex)
        if bib_match:
            keys = [m[0] or m[1] for m in bib_match[:50]]
            bib_keys = f"\n\nAvailable citation keys: {', '.join(keys)}"

        # Smart truncation: section-boundary-aware to avoid losing Method/Experiment
        tex_for_prompt = self._smart_truncate(paper_tex, max_chars=20000)

        prompt = f"""Revise the "{section_review.section}" section of this paper.

=== REVIEWER FEEDBACK ===
Issues to FIX (mandatory):
{issues_json}

Suggestions (optional improvements):
{suggestions_json}{consistency_block}{strengths_block}

=== REVISION GUIDELINES ===
{section_guidance}
{bib_keys}

=== CRITICAL RULES ===
1. Fix ALL listed issues — each one must be addressed
2. PRESERVE all strengths identified by the reviewer
3. Do NOT introduce new problems (vague claims, broken LaTeX, removed content)
4. Do NOT remove or modify \\begin{{figure}}...\\end{{figure}} or \\begin{{table}}...\\end{{table}} blocks
5. Keep the same overall structure and length (+-20%)
6. Use ONLY citation keys from the paper's bibliography
7. GROUNDING: Do NOT change any concrete numbers (accuracy, F1, loss, etc.) that appear in tables or experimental results. These come from real experiments. Do NOT "improve" them, round them, or replace them with different values. Do NOT add new result numbers that were not in the original text.

{self._build_revision_grounding_block()}

Current paper (LaTeX):
```latex
{tex_for_prompt}
```

Research topic: {ideation_output.get('topic', '')}

Write an improved version of the "{section_review.section}" section.
Output ONLY the LaTeX content for this section (no \\section{{}} command, just the body text).
If the section contains \\subsection{{}} commands, include them in your output."""

        # Use REVISION-stage LLM — strong at generation + reasoning
        revision_config = self.config.for_stage("revision")
        try:
            revised = await self.generate(
                REVISION_SYSTEM_PROMPT, prompt, stage_override=revision_config
            )
            return (revised or "").strip()
        except Exception as e:
            logger.warning("Failed to revise section '%s': %s", section_review.section, e)
            return ""

