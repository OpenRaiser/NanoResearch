"""Review agent — automated paper review, consistency checking, and revision."""

from __future__ import annotations

import re
import logging
from typing import Any

from nanoresearch.agents.base import BaseResearchAgent
from nanoresearch.schemas.manifest import PipelineStage
from nanoresearch.schemas.review import (
    ConsistencyIssue,
    ReviewOutput,
    SectionReview,
)

from ._constants import (  # noqa: F401 — re-exported for backward compat
    _CONFERENCE_KEYWORDS,
    _detect_bib_entry_type,
    MAX_REVISION_ROUNDS,
    MAX_LATEX_FIX_ATTEMPTS,
    MIN_SECTION_SCORE,
    CONVERGENCE_THRESHOLD,
    _SECTION_PATTERN,
    _CITE_PATTERN,
    _RELATED_WORK_SECTION_PATTERN,
    _ABSTRACT_PATTERN,
    REVIEW_SYSTEM_PROMPT,
    REVISION_SYSTEM_PROMPT,
)

from .section_extraction import _SectionExtractionMixin
from .multi_reviewer import _MultiReviewerMixin
from .single_review import _SingleReviewMixin
from .revision import _RevisionMixin
from .apply_revisions import _ApplyRevisionsMixin
from .consistency import _ConsistencyMixin
from .latex_compile import _LaTeXCompileMixin

__all__ = ["ReviewAgent"]

logger = logging.getLogger(__name__)


class ReviewAgent(
    _SectionExtractionMixin,
    _MultiReviewerMixin,
    _SingleReviewMixin,
    _RevisionMixin,
    _ApplyRevisionsMixin,
    _ConsistencyMixin,
    _LaTeXCompileMixin,
    BaseResearchAgent,
):
    stage = PipelineStage.REVIEW

    async def run(self, **inputs: Any) -> dict[str, Any]:
        paper_tex = inputs.get("paper_tex", "")
        if not isinstance(paper_tex, str):
            paper_tex = str(paper_tex) if paper_tex else ""
        ideation_output = inputs.get("ideation_output") or {}
        if not isinstance(ideation_output, dict):
            ideation_output = {}
        experiment_blueprint = inputs.get("experiment_blueprint") or {}
        if not isinstance(experiment_blueprint, dict):
            experiment_blueprint = {}

        # Grounding metadata from writing stage — used to protect real results
        self._writing_grounding: dict = inputs.get("writing_grounding") or {}
        self._experiment_results: dict = inputs.get("experiment_results") or {}
        self._experiment_analysis: dict = inputs.get("experiment_analysis") or {}
        self._experiment_status: str = inputs.get("experiment_status", "pending")

        if not paper_tex:
            self.log("No paper.tex content available, skipping review")
            return ReviewOutput().model_dump(mode="json")

        self.log("Starting automated review")

        # Step 1: LLM review — multi-model if committee configured, else single
        committee = getattr(self.config, "review_committee", [])
        if isinstance(committee, list) and len(committee) >= 2:
            review = await self._multi_reviewer_assessment(
                paper_tex, ideation_output, experiment_blueprint, committee
            )
        else:
            review = await self._review_paper(
                paper_tex, ideation_output, experiment_blueprint
            )
        self.log(
            f"Initial review: overall score {review.overall_score:.1f}, "
            f"{len(review.section_reviews)} sections reviewed"
        )

        # Step 2: Consistency checks (automated, no LLM)
        consistency_issues = self._run_consistency_checks(paper_tex)
        review.consistency_issues.extend(consistency_issues)
        self.log(f"Found {len(consistency_issues)} consistency issues")

        # Step 2a: Claim-result consistency check
        claim_issues = self._check_claim_result_consistency(
            paper_tex, experiment_blueprint
        )
        review.consistency_issues.extend(claim_issues)
        if claim_issues:
            self.log(f"Found {len(claim_issues)} claim-result mismatches")

        # Step 2c: Figure-text alignment check
        figure_issues = self._check_figure_text_alignment(paper_tex)
        review.consistency_issues.extend(figure_issues)
        if figure_issues:
            self.log(f"Found {len(figure_issues)} figure alignment issues")

        # Step 2d: Citation coverage check
        citation_issues = self._check_citation_coverage(paper_tex, ideation_output)
        review.consistency_issues.extend(citation_issues)
        if citation_issues:
            self.log(f"Found {len(citation_issues)} citation coverage issues")

        # Step 2e: Citation fact-checking (LLM-based)
        try:
            from nanoresearch.agents.review_citation_checker import (
                verify_citation_claims,
            )

            bibtex_map = self._build_bibtex_key_to_paper_map(
                paper_tex, ideation_output.get("papers", [])
            )
            if bibtex_map:
                cite_verifications = await verify_citation_claims(
                    self, paper_tex, bibtex_map
                )
                inaccurate = [v for v in cite_verifications if not v["accurate"]]
                if inaccurate:
                    self.log(
                        f"Citation fact-check: {len(inaccurate)} "
                        f"potentially inaccurate claims"
                    )
                    for v in inaccurate:
                        review.consistency_issues.append(
                            ConsistencyIssue(
                                issue_type="citation_inaccuracy",
                                description=(
                                    f"Claim about [{v['cite_key']}] may be "
                                    f"inaccurate: {v.get('issue', 'unspecified')}"
                                ),
                                locations=[],
                                severity="medium",
                            )
                        )
                else:
                    self.log(
                        f"Citation fact-check: {len(cite_verifications)} "
                        f"claims verified, all accurate"
                    )
        except Exception as exc:
            logger.warning("Citation fact-checking failed: %s", exc)

        # Deduplicate consistency issues before entering revision loop
        review.consistency_issues = self._dedup_consistency_issues(review.consistency_issues)

        # Step 2b: Fix incoherent reviews (low score but no issues)
        for sr in review.section_reviews:
            if sr.score < MIN_SECTION_SCORE and not sr.issues:
                sr.issues = [
                    f"Section '{sr.section}' scored {sr.score}/10 — "
                    "it needs substantial improvement in clarity, depth, "
                    "and technical rigor to reach publication quality."
                ]
                sr.suggestions = [
                    "Rewrite the section with more detailed technical content, "
                    "proper citations, and clear exposition. Remove any placeholder "
                    "or 'results pending' language. Fill tables with concrete data."
                ]

        # Step 3: Revision loop with convergence detection
        current_tex = await self._run_revision_loop(
            paper_tex, paper_tex, review, ideation_output, experiment_blueprint
        )

        # Recalculate overall score
        if review.section_reviews:
            review.overall_score = sum(
                sr.score for sr in review.section_reviews
            ) / len(review.section_reviews)

        # Save outputs
        output_data = review.model_dump(mode="json")
        self.workspace.write_json("drafts/review_output.json", output_data)
        self.workspace.register_artifact(
            "review_output",
            self.workspace.path / "drafts" / "review_output.json",
            self.stage,
        )

        # If we have revised sections, write revised paper back to paper.tex
        # current_tex already has all revisions applied from the loop above.
        if review.revised_sections:
            revised_tex = current_tex

            # Sanitize the revised LaTeX (fix Unicode, LLM artifacts, etc.)
            revised_tex = self._sanitize_revised_tex(revised_tex)

            # Deduplicate figures: keep only the first occurrence of each figure
            seen_fig_labels: set[str] = set()
            seen_fig_files: set[str] = set()
            def _dedup_fig(m: re.Match) -> str:
                block = m.group(0)
                label_m = re.search(r'\\label\{(fig:[^}]+)\}', block)
                lbl = label_m.group(1) if label_m else None
                if lbl and lbl in seen_fig_labels:
                    return ""
                file_m = re.search(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', block)
                if file_m:
                    fname = file_m.group(1)
                    if fname in seen_fig_files:
                        return ""
                    seen_fig_files.add(fname)
                # Register label AFTER both checks pass
                if lbl:
                    seen_fig_labels.add(lbl)
                return block
            revised_tex = re.sub(
                r'\\begin\{figure\*?\}.*?\\end\{figure\*?\}',
                _dedup_fig, revised_tex, flags=re.DOTALL,
            )

            # Deduplicate tables: same logic as figures
            seen_tab_labels: set[str] = set()
            def _dedup_tab(m: re.Match) -> str:
                block = m.group(0)
                label_m = re.search(r'\\label\{(tab:[^}]+)\}', block)
                lbl = label_m.group(1) if label_m else None
                if lbl and lbl in seen_tab_labels:
                    return ""
                if lbl:
                    seen_tab_labels.add(lbl)
                return block
            revised_tex = re.sub(
                r'\\begin\{table\*?\}.*?\\end\{table\*?\}',
                _dedup_tab, revised_tex, flags=re.DOTALL,
            )

            revised_tex = re.sub(r'\n{3,}', '\n\n', revised_tex)

            # Resolve any new citations introduced during revision
            bib_path = self.workspace.path / "drafts" / "references.bib"
            if bib_path.exists():
                revised_tex, _ = await self._resolve_missing_citations(
                    revised_tex, bib_path
                )

            # Overwrite original paper.tex with revised version
            tex_path = self.workspace.path / "drafts" / "paper.tex"
            self.workspace.write_text("drafts/paper.tex", revised_tex)
            # Also save a backup copy
            self.workspace.write_text("drafts/paper_revised.tex", revised_tex)
            self.workspace.register_artifact(
                "paper_tex",
                tex_path,
                self.stage,
            )
            self.log("Saved revised paper to drafts/paper.tex")

            # Compile PDF with error-fix loop (like WritingAgent)
            pdf_result = await self._compile_pdf_with_fix_loop(tex_path)
            if "pdf_path" in pdf_result:
                self.log("PDF compiled successfully after revision")
            else:
                self.log(f"PDF compilation failed: {pdf_result.get('error', 'unknown')}")

        self.log(
            f"Review complete: score={review.overall_score:.1f}, "
            f"rounds={review.revision_rounds}, "
            f"revised={len(review.revised_sections)} sections"
        )
        return output_data
