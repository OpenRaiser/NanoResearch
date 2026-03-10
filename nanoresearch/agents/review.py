"""Review agent — automated paper review, consistency checking, and revision."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from nanoresearch.agents.base import BaseResearchAgent
from nanoresearch.agents.tools import ToolDefinition, ToolRegistry
from nanoresearch.latex import fixer as latex_fixer
from nanoresearch.schemas.manifest import PipelineStage
from nanoresearch.schemas.review import (
    ConsistencyIssue,
    ReviewOutput,
    SectionReview,
)

logger = logging.getLogger(__name__)

MAX_REVISION_ROUNDS = 5
MAX_LATEX_FIX_ATTEMPTS = 3  # compile-fix loop iterations
MIN_SECTION_SCORE = 8  # Sections scoring below this get revised
CONVERGENCE_THRESHOLD = 0.3  # Stop if avg score improves by less than this

from nanoresearch.skill_prompts import get_review_system_prompt, REVISION_SYSTEM

# Generic fallback (used by compile-fix and other non-section calls)
REVIEW_SYSTEM_PROMPT = get_review_system_prompt("_default")
REVISION_SYSTEM_PROMPT = REVISION_SYSTEM


class ReviewAgent(BaseResearchAgent):
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
        revision_round = 0
        current_tex = paper_tex
        prev_avg_score = review.overall_score

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
                # Quick structural checks (no compilation — that happens at the end)
                bp_issues = []
                begins = len(re.findall(r'\\begin\{', current_tex))
                ends = len(re.findall(r'\\end\{', current_tex))
                if begins != ends:
                    bp_issues.append(f"Unbalanced environments: {begins} \\begin vs {ends} \\end")
                # Check for mismatched environment types (e.g. \begin{itemize}...\end{enumerate})
                env_stack: list[str] = []
                for env_m in re.finditer(r'\\(begin|end)\{([^}]+)\}', current_tex):
                    cmd, env_name = env_m.group(1), env_m.group(2)
                    if cmd == "begin":
                        env_stack.append(env_name)
                    elif env_stack and env_stack[-1] == env_name:
                        env_stack.pop()
                    elif env_stack:
                        bp_issues.append(
                            f"Mismatched environment: \\begin{{{env_stack[-1]}}} "
                            f"closed by \\end{{{env_name}}}"
                        )
                        env_stack.pop()  # consume it to avoid cascading errors
                        break  # one mismatch is enough to revert
                if '\\documentclass' not in current_tex:
                    bp_issues.append("Missing \\documentclass")
                if '\\end{document}' not in current_tex:
                    bp_issues.append("Missing \\end{document}")
                if bp_issues:
                    self.log(f"  Backpressure FAILED: {bp_issues}, reverting round {revision_round}")
                    # Revert this round's changes
                    for sec_name in round_revised:
                        review.revised_sections.pop(sec_name, None)
                    current_tex = self._apply_revisions(paper_tex, review.revised_sections)
                    continue  # skip re-review, try next round

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
                                "Score regression: '%s' %d → %d, reverting",
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
                    f"  '{old_sr.section}' score dropped {old_sr.score}→{new_sr.score}, "
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
                                f"{old_sr.score}→{rescore.score}"
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
                # Check if there are still low-scoring sections before exiting
                still_low = [
                    sr for sr in review.section_reviews
                    if sr.score < MIN_SECTION_SCORE
                ]
                if still_low and revision_round < MAX_REVISION_ROUNDS:
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
            prev_avg_score = new_avg

        review.revision_rounds = revision_round

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

    # ------------------------------------------------------------------
    # Section extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_sections(tex: str) -> list[tuple[str, str, int]]:
        """Extract (heading, content, level) tuples from LaTeX source.

        Handles \\section{} (level=0), \\subsection{} (level=1),
        and \\subsubsection{} (level=2).
        """
        # Use a pattern that handles nested braces (e.g. \section{Method for \textbf{X}})
        pattern = re.compile(
            r"\\((?:sub){0,2})section\*?\{((?:[^{}]|\{[^{}]*\})+)\}",
        )
        matches = list(pattern.finditer(tex))
        if not matches:
            return [("Full Paper", tex, 0)]

        sections: list[tuple[str, str, int]] = []
        for i, m in enumerate(matches):
            prefix = m.group(1)  # "", "sub", or "subsub"
            level = prefix.count("sub")
            heading = m.group(2).strip()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(tex)
            content = tex[start:end].strip()
            sections.append((heading, content, level))
        return sections

    @staticmethod
    def _get_full_section_content(
        sections: list[tuple[str, str, int]], heading: str
    ) -> str:
        """Get the full content of a top-level section including its subsections.

        Merges subsection content back into the parent section so the reviewer
        sees the complete section, not just the intro paragraph.
        """
        for i, (h, c, level) in enumerate(sections):
            if h != heading:
                continue
            if level != 0:
                return c  # Subsection — return as-is
            # Merge all following subsections until the next level=0
            merged = c
            for j in range(i + 1, len(sections)):
                if sections[j][2] == 0:
                    break
                sub_h, sub_c, sub_lvl = sections[j]
                sub_prefix = "\\sub" * sub_lvl + "section"
                merged += f"\n\n\\{sub_prefix}{{{sub_h}}}\n{sub_c}"
            return merged
        return ""

    async def _build_review_tools(self) -> ToolRegistry | None:
        """Build a ToolRegistry with search tools for reviewing.

        Returns None if no tools could be registered.
        """
        registry = ToolRegistry()

        try:
            from mcp_server.tools.arxiv_search import search_arxiv
            from mcp_server.tools.semantic_scholar import search_semantic_scholar

            async def _search_papers(query: str, max_results: int = 5) -> list[dict]:
                results: list[dict] = []
                try:
                    results.extend(await search_arxiv(query, max_results=max_results))
                except Exception as exc:
                    logger.debug("arxiv search failed: %s", exc)
                try:
                    results.extend(await search_semantic_scholar(query, max_results=max_results))
                except Exception as exc:
                    logger.debug("semantic scholar search failed: %s", exc)
                return results

            registry.register(ToolDefinition(
                name="search_papers",
                description="Search for academic papers to verify SOTA claims and find latest results.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "max_results": {"type": "integer", "description": "Max papers", "default": 5},
                    },
                    "required": ["query"],
                },
                handler=_search_papers,
            ))
        except ImportError:
            pass

        try:
            from mcp_server.tools.paperswithcode import get_sota
            registry.register(ToolDefinition(
                name="get_sota",
                description="Query Papers With Code SOTA leaderboard for a task/dataset.",
                parameters={
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string", "description": "PapersWithCode task ID or name"},
                        "dataset": {"type": "string", "description": "Dataset name", "default": ""},
                    },
                    "required": ["task_id"],
                },
                handler=lambda task_id, dataset="": get_sota(task_id, dataset=dataset),
            ))
        except ImportError:
            pass

        try:
            from mcp_server.tools.web_search import search_web
            registry.register(ToolDefinition(
                name="search_web",
                description="Search the web for latest benchmark results and technical information.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "max_results": {"type": "integer", "description": "Max results", "default": 5},
                    },
                    "required": ["query"],
                },
                handler=lambda query, max_results=5: search_web(query, max_results=max_results),
            ))
        except ImportError:
            pass

        return registry if len(registry) > 0 else None

    @staticmethod
    def _repair_truncated_json(text: str) -> dict | None:
        """Attempt to repair JSON that was truncated mid-output.

        Handles common truncation patterns: unterminated strings, missing
        closing brackets/braces.
        """
        # Try parsing as-is first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Strategy: close any open strings, arrays, and objects
        repaired = text.rstrip()
        # If inside a string, close it
        in_string = False
        escaped = False
        for ch in repaired:
            if escaped:
                escaped = False
                continue
            if ch == '\\':
                escaped = True
                continue
            if ch == '"':
                in_string = not in_string
        if in_string:
            repaired += '"'

        # Count open brackets/braces using a stack (close in correct LIFO order)
        stack: list[str] = []
        in_str = False
        esc = False
        for ch in repaired:
            if esc:
                esc = False
                continue
            if ch == '\\':
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
            if in_str:
                continue
            if ch in ('{', '['):
                stack.append(ch)
            elif ch == '}' and stack and stack[-1] == '{':
                stack.pop()
            elif ch == ']' and stack and stack[-1] == '[':
                stack.pop()

        closers = {'{': '}', '[': ']'}
        for bracket in reversed(stack):
            repaired += closers.get(bracket, '')

        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

        # Last resort: extract score via regex from the raw text
        score_match = re.search(r'"score"\s*:\s*(\d+)', text)
        if score_match:
            score = int(score_match.group(1))
            # Try to extract issues array elements
            issues = re.findall(r'"issues"\s*:\s*\[(.*?)\]', text, re.DOTALL)
            issue_list = []
            if issues:
                issue_list = re.findall(r'"([^"]{10,})"', issues[0])
            suggestions = re.findall(r'"suggestions"\s*:\s*\[(.*?)\]', text, re.DOTALL)
            suggestion_list = []
            if suggestions:
                suggestion_list = re.findall(r'"([^"]{10,})"', suggestions[0])
            return {
                "score": score,
                "issues": issue_list[:5],
                "suggestions": suggestion_list[:3],
            }
        return None

    # ── Multi-model review committee ──────────────────────────────────

    async def _multi_reviewer_assessment(
        self,
        paper_tex: str,
        ideation_output: dict,
        experiment_blueprint: dict,
        committee: list[dict],
    ) -> ReviewOutput:
        """Run parallel reviews from multiple model personas, merge results.

        Falls back to single-model review if all reviewers fail.
        """
        import asyncio as _aio

        tasks = []
        for reviewer in committee:
            tasks.append(
                self._review_as_role(
                    paper_tex, ideation_output, experiment_blueprint, reviewer
                )
            )
        results = await _aio.gather(*tasks, return_exceptions=True)

        valid_reviews: list[ReviewOutput] = []
        weights: list[float] = []
        for review_result, reviewer in zip(results, committee):
            if isinstance(review_result, Exception):
                self.log(
                    f"Reviewer '{reviewer.get('role', '?')}' failed: "
                    f"{review_result}"
                )
                continue
            valid_reviews.append(review_result)
            weights.append(reviewer.get("weight", 1.0 / len(committee)))

        if not valid_reviews:
            self.log("All reviewers failed, falling back to single-model")
            return await self._review_paper(
                paper_tex, ideation_output, experiment_blueprint
            )

        # Normalize weights (fallback to equal weights if all zero)
        total_w = sum(weights)
        if total_w > 0:
            weights = [w / total_w for w in weights]
        else:
            weights = [1.0 / len(weights)] * len(weights)

        # Weighted overall score
        overall = sum(
            r.overall_score * w for r, w in zip(valid_reviews, weights)
        )

        # Merge section reviews: per-section min score + union issues
        merged_sections = self._merge_section_reviews(valid_reviews)

        # Union major/minor revisions (dedup by first 80 chars)
        major: list[str] = []
        minor: list[str] = []
        seen_major: set[str] = set()
        seen_minor: set[str] = set()
        for r in valid_reviews:
            for issue in r.major_revisions:
                key = issue[:80].lower()
                if key not in seen_major:
                    major.append(issue)
                    seen_major.add(key)
            for sug in r.minor_revisions:
                key = sug[:80].lower()
                if key not in seen_minor:
                    minor.append(sug)
                    seen_minor.add(key)

        self.log(
            f"Multi-reviewer assessment: {len(valid_reviews)} reviewers, "
            f"weighted score {overall:.1f}"
        )

        return ReviewOutput(
            overall_score=round(overall, 2),
            section_reviews=merged_sections,
            major_revisions=major,
            minor_revisions=minor,
        )

    async def _review_as_role(
        self,
        paper_tex: str,
        ideation_output: dict,
        experiment_blueprint: dict,
        reviewer: dict,
    ) -> ReviewOutput:
        """Run a full review using a specific reviewer persona and model."""
        from nanoresearch.config import StageModelConfig

        role = reviewer.get("role", "Reviewer")
        focus = reviewer.get("focus", "overall paper quality")

        reviewer_config = StageModelConfig(
            model=reviewer.get("model", self.config.for_stage("review").model),
            base_url=reviewer.get("base_url"),
            api_key=reviewer.get("api_key"),
            temperature=reviewer.get("temperature", 0.3),
            max_tokens=reviewer.get("max_tokens", 16384),
            timeout=reviewer.get("timeout", 300.0),
        )

        # Extract sections (same logic as _review_paper)
        sections = self._extract_sections(paper_tex)
        main_sections: list[tuple[str, str]] = []
        for h, _c, level in sections:
            if level == 0:
                merged = self._get_full_section_content(sections, h)
                main_sections.append((h, merged))
        if not main_sections:
            main_sections = [(h, c) for h, c, _lvl in sections[:10]]

        section_reviews: list[SectionReview] = []
        for heading, content in main_sections:
            try:
                sr = await self._review_single_section_as_role(
                    heading, content, ideation_output, experiment_blueprint,
                    reviewer_config, role, focus,
                )
                section_reviews.append(sr)
            except Exception as e:
                logger.warning(
                    "Reviewer %s failed on section '%s': %s", role, heading, e
                )
                section_reviews.append(
                    SectionReview(section=heading, score=5, issues=[str(e)])
                )

        overall_score = (
            sum(sr.score for sr in section_reviews) / len(section_reviews)
            if section_reviews else 5.0
        )
        major = []
        minor = []
        for sr in section_reviews:
            if sr.score < 5:
                major.extend(sr.issues[:2])
            elif sr.score < 7:
                minor.extend(sr.suggestions[:2])

        return ReviewOutput(
            overall_score=overall_score,
            section_reviews=section_reviews,
            major_revisions=major,
            minor_revisions=minor,
        )

    async def _review_single_section_as_role(
        self,
        heading: str,
        content: str,
        ideation_output: dict,
        experiment_blueprint: dict,
        reviewer_config,
        role: str,
        focus: str,
    ) -> SectionReview:
        """Review a section using a specific reviewer persona."""
        system_prompt = (
            f"You are a top-tier {role} at a major ML conference "
            f"(NeurIPS/ICML/ICLR). Your primary focus: {focus}.\n"
            f"Review the paper section and provide structured feedback. "
            f"Be rigorous but constructive."
        )

        prompt = f"""Review the following section of an academic paper.

Section: {heading}

```latex
{content[:12000]}
```

Research context:
- Topic: {str(ideation_output.get('topic', 'Unknown'))[:500]}
- Method: {str((experiment_blueprint.get('proposed_method') or {}).get('name', 'Unknown'))[:500]}

Focus on: {focus}

Return JSON:
{{
    "section": "{heading}",
    "score": 7,
    "issues": ["Issue 1: specific problem and how to fix it"],
    "suggestions": ["Suggestion 1"]
}}

Score rubric: 9-10 publication-ready, 7-8 solid with minor issues, 5-6 significant problems, 3-4 major rewrite, 1-2 fundamentally flawed."""

        try:
            result = await self.generate_json(
                system_prompt, prompt, stage_override=reviewer_config
            )
        except Exception:
            raw = await self.generate(
                system_prompt, prompt, json_mode=True,
                stage_override=reviewer_config,
            )
            result = self._repair_truncated_json(raw or "")
            if result is None or isinstance(result, list):
                result = {"section": heading, "score": 5, "issues": [], "suggestions": []}

        raw_score = result.get("score", 5)
        try:
            score = max(1, min(10, int(float(raw_score))))
        except (TypeError, ValueError):
            score = 5

        issues = [
            str(i) for i in result.get("issues", [])[:5]
            if i
        ]
        suggestions = [
            str(s) for s in result.get("suggestions", [])[:3]
            if s
        ]

        if not issues and score < 7:
            score = 7

        return SectionReview(
            section=result.get("section", heading),
            score=score,
            issues=issues,
            suggestions=suggestions,
        )

    @staticmethod
    def _merge_section_reviews(reviews: list[ReviewOutput]) -> list[SectionReview]:
        """Merge section reviews from multiple reviewers.

        Strategy: per-section min score (strictest reviewer wins),
        union all issues/suggestions with dedup.
        """
        section_map: dict[str, dict] = {}
        for review in reviews:
            for sr in review.section_reviews:
                name = sr.section.lower().strip()
                if name not in section_map:
                    section_map[name] = {
                        "section": sr.section,
                        "score": sr.score,
                        "issues": list(sr.issues),
                        "suggestions": list(sr.suggestions),
                    }
                else:
                    existing = section_map[name]
                    existing["score"] = min(existing["score"], sr.score)
                    seen_i = {i[:80].lower() for i in existing["issues"]}
                    for issue in sr.issues:
                        if issue[:80].lower() not in seen_i:
                            existing["issues"].append(issue)
                            seen_i.add(issue[:80].lower())
                    seen_s = {s[:80].lower() for s in existing["suggestions"]}
                    for sug in sr.suggestions:
                        if sug[:80].lower() not in seen_s:
                            existing["suggestions"].append(sug)
                            seen_s.add(sug[:80].lower())

        return [
            SectionReview(
                section=d["section"],
                score=max(1, min(10, d["score"])),
                issues=d["issues"],
                suggestions=d["suggestions"],
            )
            for d in section_map.values()
        ]

    # ── Citation fact-checking helpers ────────────────────────────────

    @staticmethod
    def _build_bibtex_key_to_paper_map(
        paper_tex: str, papers: list,
    ) -> dict[str, dict]:
        """Build mapping from BibTeX cite key → paper dict (with title/abstract).

        Matches BibTeX entries in the paper to papers from ideation
        by title similarity.
        """
        if not isinstance(papers, list) or not papers:
            return {}

        # Extract bibtex entries: key → title
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

        # Build title → paper dict index from ideation papers
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
- Method: {str((experiment_blueprint.get('proposed_method') or {}).get('name', 'Unknown'))[:500]}

Provide a thorough review with:
1. **Score** (1-10): Use the rubric strictly. Justify your score.
2. **Strengths** (up to 3): What this section does WELL — these must be PRESERVED during any revision.
3. **Must-fix issues** (up to 5): Each must state: [PROBLEM] what is wrong → [IMPACT] why it matters → [FIX] specific action to take.
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
            score = int(float(raw_score))
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
                issues.append(" → ".join(parts) if parts else str(item))
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
                suggestions.append(" → ".join(parts) if parts else str(item))
            else:
                suggestions.append(str(item))

        # Only fix the one truly contradictory case: zero issues but low score
        if len(issues) == 0 and score < 7:
            score = 7

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
5. Keep the same overall structure and length (±20%)
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

    def _build_revision_grounding_block(self) -> str:
        """Build a grounding context block for revision prompts.

        Tells the revision LLM which numbers are real and must be preserved.
        """
        grounding = getattr(self, '_writing_grounding', {})
        analysis = getattr(self, '_experiment_analysis', {})
        status = getattr(self, '_experiment_status', 'pending')

        completeness = grounding.get('result_completeness', 'none') if grounding else 'none'
        has_real = grounding.get('has_real_results', False) if grounding else False

        if not has_real:
            return (
                "=== GROUNDING STATUS: NO REAL RESULTS ===\n"
                "This paper has NO real experiment results. During revision:\n"
                "- Do NOT add any experimental numbers\n"
                "- Do NOT fill empty table cells with fabricated values\n"
                "- Preserve any 'results pending' or 'future work' language\n"
                "=== END GROUNDING ==="
            )

        lines = [
            f"=== GROUNDING STATUS: {completeness.upper()} RESULTS ===",
            "This paper contains REAL experiment results. During revision:",
            "- PRESERVE all numbers in tables (they come from real experiments)",
            "- Do NOT round, adjust, or 'improve' any metric values",
            "- Do NOT add new baseline numbers that weren't in the original",
        ]

        # Add real metric values for reference
        final_metrics = analysis.get('final_metrics', {}) if isinstance(analysis, dict) else {}
        if isinstance(final_metrics, dict) and final_metrics:
            lines.append("Real metrics (for reference, do NOT modify):")
            for k, v in list(final_metrics.items())[:10]:
                lines.append(f"  {k} = {v}")

        lines.append("=== END GROUNDING ===")
        return "\n".join(lines)

    async def _meta_refine_revision(
        self,
        paper_tex: str,
        old_review: SectionReview,
        new_review: SectionReview,
        failed_revision: str,
        ideation_output: dict,
    ) -> str | None:
        """Diagnose why a revision degraded quality, then retry with extra constraints.

        This is the runtime prompt self-optimization loop:
        1. LLM analyzes old review vs new review to find what the revision broke
        2. Generates extra constraints to prevent the same mistakes
        3. Retries revision with the augmented prompt
        """
        section_name = old_review.section
        old_strengths = old_review.strengths

        # Guard: if failed_revision is empty, nothing to diagnose
        if not failed_revision or not failed_revision.strip():
            logger.warning("Meta-refine: empty failed revision for '%s', skipping", section_name)
            return None

        # Step 1: Diagnose with review-stage LLM (cheap + fast)
        diagnosis_prompt = f"""A revision of the "{section_name}" section made the paper WORSE.

BEFORE revision (score {old_review.score}/10):
- Strengths: {json.dumps(old_strengths, ensure_ascii=False)}
- Issues: {json.dumps(old_review.issues[:5], ensure_ascii=False)}

AFTER revision (score {new_review.score}/10):
- New issues: {json.dumps(new_review.issues[:5], ensure_ascii=False)}

Failed revision text (first 3000 chars):
{failed_revision[:3000]}

Analyze what the revision did WRONG. Common mistakes:
- Removed specific data/numbers and replaced with vague language
- Deleted citations or technical details
- Introduced generic filler ("it is worth noting", "interestingly")
- Broke LaTeX formatting or removed figures/tables
- Over-simplified technical content
- Changed notation inconsistently

Return JSON:
{{
    "diagnosis": "What specifically went wrong with this revision",
    "extra_constraints": [
        "Constraint 1: Do NOT ...",
        "Constraint 2: MUST keep ...",
        "Constraint 3: ..."
    ]
}}"""

        review_config = self.config.for_stage("review")
        try:
            result = await self.generate_json(
                "You are a meta-reviewer analyzing why a paper revision failed. "
                "Be specific about what went wrong and provide actionable constraints.",
                diagnosis_prompt,
                stage_override=review_config,
            )
        except Exception as exc:
            logger.warning("Meta-refine diagnosis failed for '%s': %s", section_name, exc)
            return None

        if isinstance(result, list):
            result = {"extra_constraints": result}
        diagnosis = result.get("diagnosis", "unknown")
        extra_constraints = result.get("extra_constraints", [])
        self.log(f"  '{section_name}' diagnosis: {diagnosis[:120]}")

        if not extra_constraints:
            return None

        # Step 2: Retry revision with extra constraints appended
        constraints_block = "\n".join(f"- {c}" for c in extra_constraints[:5])
        augmented_review = SectionReview(
            section=old_review.section,
            score=old_review.score,
            issues=old_review.issues,
            suggestions=old_review.suggestions + [
                f"[META-REFINE] Previous revision failed because: {diagnosis}. "
                f"Extra constraints:\n{constraints_block}"
            ],
        )
        # Carry over strengths and justification
        augmented_review.strengths = old_strengths
        augmented_review.score_justification = old_review.score_justification

        self.log(f"  '{section_name}' retrying revision with {len(extra_constraints)} extra constraints")
        try:
            return await self._revise_section(
                paper_tex, augmented_review, ideation_output,
            )
        except Exception as exc:
            logger.warning("Meta-refine retry failed for '%s': %s", section_name, exc)
            return None

    @staticmethod
    def _smart_truncate(text: str, max_chars: int = 20000) -> str:
        """Truncate paper text preserving high-priority sections.

        Instead of naive head/tail split (which often drops Method/Experiment),
        keep preamble + sections by priority order.
        """
        if len(text) <= max_chars:
            return text

        # Find section boundaries
        section_starts = [
            (m.start(), m.group(1))
            for m in re.finditer(r'\\section\{([^}]+)\}', text)
        ]
        if not section_starts:
            # No sections found, fall back to head/tail
            return text[:12000] + "\n\n[...truncated...]\n\n" + text[-8000:]

        # Always keep preamble (title, abstract, etc.) up to first \section
        preamble = text[:section_starts[0][0]]
        remaining = max_chars - len(preamble)
        if remaining <= 0:
            return preamble[:max_chars]

        # Collect sections by priority, record their original position for ordering
        priority = ["introduction", "method", "experiment",
                     "result", "conclusion", "related"]
        # (original_start_pos, section_text) — for document-order output
        kept: list[tuple[int, str]] = []

        for pname in priority:
            if remaining <= 0:
                break
            for i, (start, title) in enumerate(section_starts):
                if pname in title.lower():
                    end = (section_starts[i + 1][0]
                           if i + 1 < len(section_starts)
                           else len(text))
                    content = text[start:end]
                    if len(content) <= remaining:
                        kept.append((start, content))
                        remaining -= len(content)
                    elif remaining > 500:
                        kept.append((start,
                                     content[:remaining] + "\n[...truncated...]"))
                        remaining = 0
                    break

        if not kept:
            return text[:max_chars]

        # Sort by original document position so LLM sees correct order
        kept.sort(key=lambda x: x[0])
        return preamble + "\n\n".join(s for _, s in kept)

    @staticmethod
    def _get_section_revision_guidance(section_name: str) -> str:
        """Return section-specific revision guidance for top-tier venue standards."""
        section_lower = section_name.lower()
        if "related" in section_lower or "prior" in section_lower or "background" in section_lower:
            return (
                "SECTION-SPECIFIC GUIDANCE (Related Work):\n"
                "- Organize by THEME/APPROACH, not chronologically\n"
                "- Each paragraph should cover one research direction with 3-5 citations\n"
                "- For each cited work, briefly state its approach AND its limitation\n"
                "- End each paragraph by explaining how your work addresses these limitations\n"
                "- The final paragraph should clearly differentiate your approach from ALL prior work\n"
                "- Use \\citet{} when the author is the subject, \\citep{} for parenthetical\n"
                "- Minimum 15 citations total for a strong Related Work section\n"
                "- Cover at minimum: (1) the main task, (2) the key technique you use, "
                "(3) closely related approaches you improve upon"
            )
        elif "intro" in section_lower:
            return (
                "SECTION-SPECIFIC GUIDANCE (Introduction):\n"
                "- Start with the broader problem and its importance (1 paragraph)\n"
                "- Describe the specific challenge your work addresses (1 paragraph)\n"
                "- Briefly outline your approach and key contributions (1 paragraph)\n"
                "- List 3-4 concrete contributions as a bulleted list\n"
                "- Include at least 5-8 citations to establish context"
            )
        elif "method" in section_lower or "approach" in section_lower:
            return (
                "SECTION-SPECIFIC GUIDANCE (Method):\n"
                "- Include a formal problem definition with mathematical notation\n"
                "- Describe each component with equations\n"
                "- Use \\begin{equation} for key formulas, number them for reference\n"
                "- Explain the intuition behind each design choice\n"
                "- Reference the architecture figure if available (Figure~\\ref{fig:framework})"
            )
        elif "experiment" in section_lower or "result" in section_lower:
            return (
                "SECTION-SPECIFIC GUIDANCE (Experiments):\n"
                "- Describe datasets with statistics (size, splits, metrics)\n"
                "- List all baselines with brief descriptions and citations\n"
                "- Present main results in a table (Table~\\ref{tab:main_results})\n"
                "- Include ablation study results\n"
                "- Discuss why your method outperforms baselines\n"
                "- ALL numeric values in tables must be concrete, never '--' or 'N/A'"
            )
        elif "conclusion" in section_lower:
            return (
                "SECTION-SPECIFIC GUIDANCE (Conclusion):\n"
                "- Summarize the key contributions (2-3 sentences)\n"
                "- State the main experimental findings with numbers\n"
                "- Discuss limitations honestly\n"
                "- Suggest 2-3 specific future work directions"
            )
        return ""

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
        import re
        issues: list[ConsistencyIssue] = []

        # Count total unique citations (handle natbib variants + optional args)
        cite_pattern = re.compile(r"\\[Cc]ite[tp]?(?:\w*)(?:\*)?(?:\[[^\]]*\])*\{([^}]+)\}")
        cited: set[str] = set()
        for m in cite_pattern.finditer(tex):
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
        rw_pattern = re.compile(
            r'\\section\{(?:Related Works?|Prior Work|Literature Review'
            r'|Background(?:\s+and\s+Related\s+Work)?)\}'
            r'(.*?)(?=\\section\{|\\end\{document\})',
            re.DOTALL | re.IGNORECASE,
        )
        rw_match = rw_pattern.search(tex)
        if rw_match:
            rw_content = rw_match.group(1)
            rw_cited: set[str] = set()
            for m in cite_pattern.finditer(rw_content):
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
        referenced_figs = set(re.findall(r'\\(?:auto)?ref\{(fig:[^}]+)\}', tex))

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

    async def _resolve_missing_citations(
        self, latex: str, bib_path: Path
    ) -> tuple[str, bool]:
        """Find \\cite keys missing from bib and auto-fill them.

        Delegates to WritingAgent's resolver. Returns (latex, changed).
        """
        try:
            from nanoresearch.agents.writing import WritingAgent
            bib_content = bib_path.read_text(encoding="utf-8")

            # Reuse WritingAgent's citation regex
            cited: set[str] = set()
            for m in WritingAgent._CITE_KEY_RE.finditer(latex):
                for k in m.group(1).split(","):
                    k = k.strip()
                    if k:
                        cited.add(k)

            defined: set[str] = set()
            for m in WritingAgent._BIB_KEY_RE.finditer(bib_content):
                defined.add(m.group(1).strip())

            missing = cited - defined
            if not missing:
                return latex, False

            self.log(f"Resolving {len(missing)} missing citation(s) in review: {sorted(missing)}")

            # Create a temporary WritingAgent-like resolver
            new_entries: list[str] = []
            for key in sorted(missing):
                # Skip if already present (guards against duplicate calls)
                if re.search(r'@\w+\s*\{\s*' + re.escape(key) + r'\s*,', bib_content):
                    continue
                entry = await self._resolve_single_citation_key(key)
                new_entries.append(entry)

            if new_entries:
                bib_content = bib_content.rstrip() + "\n\n" + "\n".join(new_entries)
                # Atomic write: write to temp file then rename
                tmp_path = bib_path.with_suffix(".bib.tmp")
                tmp_path.write_text(bib_content, encoding="utf-8")
                tmp_path.replace(bib_path)
                self.log(f"Added {len(new_entries)} bib entries during review")

            return latex, bool(new_entries)
        except Exception as exc:
            logger.warning("Citation resolution failed: %s", exc)
            return latex, False

    async def _resolve_single_citation_key(self, key: str) -> str:
        """Resolve a missing citation key via S2 search or stub."""
        m = re.match(r"([a-z]+)(\d{4})([a-z]?)$", key, re.IGNORECASE)
        surname = m.group(1).capitalize() if m else key
        year = m.group(2) if m else ""
        query = f"{surname} {year}" if m else key

        try:
            from mcp_server.tools.semantic_scholar import search_semantic_scholar
            results = await search_semantic_scholar(query, max_results=5)
            best = None
            for r in results:
                r_year = str(r.get("year", ""))
                r_authors = " ".join(a.get("name", str(a)) if isinstance(a, dict) else str(a) for a in r.get("authors", []))
                if year and r_year == year and surname.lower() in r_authors.lower():
                    best = r
                    break
            if not best:
                for r in results:
                    if year and str(r.get("year", "")) == year:
                        best = r
                        break
            if not best and results:
                best = results[0]

            if best:
                authors = best.get("authors", [])
                author_str = " and ".join(a.get("name", str(a)) if isinstance(a, dict) else str(a) for a in authors[:5]) if authors else surname
                title = best.get("title", "Unknown")
                venue = best.get("venue", "") or "arXiv preprint"
                r_year = best.get("year", year or 2024)
                return (
                    f"@article{{{key},\n"
                    f"  title={{{title}}},\n"
                    f"  author={{{author_str}}},\n"
                    f"  year={{{r_year}}},\n"
                    f"  journal={{{venue}}},\n"
                    f"}}\n"
                )
        except Exception as exc:
            logger.debug("S2 search failed for '%s': %s", key, exc)

        return (
            f"@misc{{{key},\n"
            f"  title={{{surname} et al.}},\n"
            f"  author={{{surname}}},\n"
            f"  year={{{year or 2024}}},\n"
            f"  note={{Citation auto-generated}},\n"
            f"}}\n"
        )

    @staticmethod
    def _sanitize_revised_tex(tex: str) -> str:
        """Sanitize revised LaTeX using WritingAgent's sanitizer.

        Falls back to inline critical fixes if WritingAgent import fails.
        """
        try:
            from nanoresearch.agents.writing import WritingAgent
            tex = WritingAgent._sanitize_latex(tex)
        except Exception as exc:
            logger.warning("WritingAgent._sanitize_latex failed (%s), applying inline fixes", exc)
            # Inline fallback: at minimum fix the most critical issues
            import re as _re
            # [H]/[h]/[h!] → [t!] (preserve * for column-spanning variants)
            tex = _re.sub(r'\\begin\{figure\}\s*\[[Hh]!?\]', r'\\begin{figure}[t!]', tex)
            tex = _re.sub(r'\\begin\{figure\*\}\s*\[[Hh]!?\]', r'\\begin{figure*}[t!]', tex)
            tex = _re.sub(r'\\begin\{table\}\s*\[[Hh]!?\]', r'\\begin{table}[t!]', tex)
            tex = _re.sub(r'\\begin\{table\*\}\s*\[[Hh]!?\]', r'\\begin{table*}[t!]', tex)
            # Unicode dashes
            tex = tex.replace("\u2014", "---").replace("\u2013", "--")
            tex = tex.replace("\u201c", "``").replace("\u201d", "''")
        return tex

    async def _compile_pdf_with_fix_loop(self, tex_path: Path) -> dict:
        """Compile LaTeX to PDF with automatic error-fix loop.

        If compilation fails, feed the error back to the LLM, apply the fix,
        and retry up to MAX_LATEX_FIX_ATTEMPTS times.

        Safety features (OpenClaw-inspired):
        - Backs up original tex before fix loop; restores on total failure
        - Post-write verification: re-reads file to confirm write succeeded
        """
        import shutil

        try:
            from mcp_server.tools.pdf_compile import compile_pdf
        except ImportError:
            return {"error": "PDF compiler module not available"}

        tex_path = Path(tex_path)

        # Backup original tex before any fix attempts
        backup_path = tex_path.with_suffix('.tex.bak')
        try:
            shutil.copy2(tex_path, backup_path)
        except OSError:
            pass  # non-fatal

        import hashlib
        result: dict = {}
        seen_error_sigs: set[str] = set()
        for attempt in range(MAX_LATEX_FIX_ATTEMPTS + 1):
            try:
                result = await compile_pdf(str(tex_path))
                if not isinstance(result, dict):
                    result = {"error": f"Unexpected compile_pdf return type: {type(result).__name__}"}
            except Exception as e:
                result = {"error": str(e)}

            if "pdf_path" in result:
                if attempt > 0:
                    self.log(f"PDF compiled successfully after {attempt} fix(es)")
                return result

            error_msg = result.get("error", "Unknown compilation error")

            # Detect repeated identical errors to avoid infinite loops
            error_sig = hashlib.md5(error_msg[-500:].encode()).hexdigest()[:8]
            if error_sig in seen_error_sigs:
                self.log("LaTeX fix loop: same error repeated, stopping")
                return result
            seen_error_sigs.add(error_sig)

            # Don't retry if the problem isn't fixable via LaTeX edits
            if "not found" in error_msg.lower() or "not available" in error_msg.lower():
                self.log("No LaTeX compiler available, skipping fix loop")
                return result

            if attempt >= MAX_LATEX_FIX_ATTEMPTS:
                self.log(f"PDF compilation failed after {MAX_LATEX_FIX_ATTEMPTS} fix attempts")
                # Restore backup on total failure
                if backup_path.exists():
                    try:
                        shutil.copy2(backup_path, tex_path)
                        self.log("  Restored original tex from backup")
                    except OSError:
                        pass
                return result

            # ── Check if error originates from .bbl (BibTeX) ──
            # Errors like "paper.bbl:64: Misplaced alignment tab character &"
            # can only be fixed by editing references.bib, not paper.tex.
            if '.bbl' in error_msg or 'alignment tab' in error_msg.lower():
                bib_path = tex_path.parent / "references.bib"
                if bib_path.exists():
                    try:
                        from nanoresearch.agents.writing import WritingAgent
                        bib_content = bib_path.read_text(encoding="utf-8")
                        fixed_bib = WritingAgent._sanitize_bibtex(bib_content)
                        if fixed_bib != bib_content:
                            bib_path.write_text(fixed_bib, encoding="utf-8")
                            self.log(f"  Fixed BibTeX file (attempt {attempt + 1})")
                            continue  # retry compilation with fixed .bib
                    except Exception as bib_exc:
                        self.log(f"  BibTeX fix failed: {bib_exc}")

            # Feed error to LLM and fix
            self.log(
                f"PDF compilation failed (attempt {attempt + 1}), "
                f"feeding error to LLM for fix..."
            )

            try:
                current_tex = tex_path.read_text(encoding="utf-8")
            except OSError as exc:
                logger.error("Cannot read tex file for fixing: %s", exc)
                return result

            fixed_tex = await self._fix_latex_errors(current_tex, error_msg)

            if fixed_tex and fixed_tex != current_tex:
                fixed_tex = self._sanitize_revised_tex(fixed_tex)
                try:
                    tex_path.write_text(fixed_tex, encoding="utf-8")
                except OSError as exc:
                    logger.error("Cannot write fixed tex file: %s", exc)
                    return result
                # Post-write verification
                try:
                    verify = tex_path.read_text(encoding="utf-8")
                    if verify != fixed_tex:
                        self.log("  WARNING: post-write verification failed, reverting")
                        tex_path.write_text(current_tex, encoding="utf-8")
                        return result
                except OSError:
                    pass
                self.log(f"  Applied LLM fix (attempt {attempt + 1})")
            else:
                self.log("  LLM returned no changes, aborting fix loop")
                return result

        return result

    async def _fix_latex_errors(self, tex_source: str, error_log: str) -> str | None:
        """Fix LaTeX compilation errors using a 2-level strategy.

        Level 1: Deterministic fixes (no LLM) — via shared latex.fixer module.
        Level 2: Search-replace LLM fix — LLM outputs {"old":"...","new":"..."} pairs.

        Inspired by OpenClaw's edit tool: minimal LLM output, exact text matching.
        NEVER sends the full document to the LLM for rewriting.
        """
        error_log = latex_fixer.truncate_error_log(error_log)

        error_lines = latex_fixer.extract_error_lines(error_log)
        error_line = error_lines[0] if error_lines else None

        tex_lines = tex_source.split('\n')
        error_lower = error_log.lower()

        # ──────────── Level 1: Deterministic fixes ────────────
        fixed = latex_fixer.deterministic_fix(
            tex_source, error_log, error_line, log_fn=self.log,
        )
        if fixed and fixed != tex_source:
            self.log("  Level 1: deterministic fix applied")
            return fixed

        # Classify error for LLM hint
        targeted_hint = latex_fixer.classify_error(error_lower)

        # ──────────── Level 2: Search-replace LLM fix ────────────
        result = await self._search_replace_llm_fix(
            tex_source, tex_lines, error_line, error_log, targeted_hint
        )
        if result:
            return result

        self.log("  All fix levels exhausted, no fix found")
        return None

    def _try_deterministic_fix(
        self,
        tex_source: str,
        tex_lines: list[str],
        error_log: str,
        error_lower: str,
        error_line: int | None,
    ) -> str | None:
        """Level 1: Delegate to shared latex_fixer.deterministic_fix()."""
        return latex_fixer.deterministic_fix(
            tex_source, error_log, error_line, log_fn=self.log,
        )

    @staticmethod
    def _classify_error(error_lower: str) -> str:
        """Delegate to shared latex_fixer.classify_error()."""
        return latex_fixer.classify_error(error_lower)

    async def _search_replace_llm_fix(
        self,
        tex_source: str,
        tex_lines: list[str],
        error_line: int | None,
        error_log: str,
        targeted_hint: str,
    ) -> str | None:
        """Level 2: Search-replace fix via shared latex_fixer module."""
        win_start, win_end, numbered = latex_fixer.build_error_snippet(
            tex_lines, error_line,
        )
        prompt = latex_fixer.build_search_replace_prompt(
            error_log, error_line, targeted_hint,
            win_start, win_end, numbered,
        )

        revision_config = self.config.for_stage("revision")
        try:
            raw = await self.generate(
                latex_fixer.SEARCH_REPLACE_SYSTEM_PROMPT, prompt,
                stage_override=revision_config,
            )
            edits = latex_fixer.parse_edit_json(raw)
            if not edits:
                self.log("  Level 2: LLM returned no valid edits")
                return None
            return latex_fixer.apply_edits(
                tex_source, edits, log_fn=self.log,
                search_window=(win_start, win_end),
            )
        except Exception as exc:
            self.log(f"  Level 2 search-replace fix failed: {exc}")

        return None

    @staticmethod
    def _parse_edit_json(raw: str) -> list[dict]:
        """Delegate to shared latex_fixer.parse_edit_json()."""
        return latex_fixer.parse_edit_json(raw)

    @staticmethod
    def _apply_revisions(paper_tex: str, revised_sections: dict[str, str]) -> str:
        """Apply revised sections back into the paper LaTeX source.

        Handles \\section{}, \\subsection{}, and \\subsubsection{}.
        Searches only after \\begin{document} to avoid matching ToC entries.
        Preserves \\begin{figure}...\\end{figure} blocks from the original
        when the revised content doesn't include them.
        """
        result = paper_tex

        # Find body start to avoid ToC matches
        body_marker = r"\begin{document}"
        body_start = result.find(body_marker)
        if body_start >= 0:
            body_start += len(body_marker)
        else:
            body_start = 0

        for heading, new_content in revised_sections.items():
            # Recalculate body_start each iteration because prior replacements
            # shift all character offsets in `result`
            body_start = result.find(body_marker)
            if body_start >= 0:
                body_start += len(body_marker)
            else:
                body_start = 0

            # Special case: Abstract lives in \begin{abstract}...\end{abstract}
            if heading == "Abstract":
                abs_pat = re.compile(
                    r'(\\begin\{abstract\})(.*?)(\\end\{abstract\})',
                    re.DOTALL,
                )
                abs_m = abs_pat.search(result)
                if abs_m:
                    result = (
                        result[:abs_m.start(2)]
                        + "\n" + new_content.strip() + "\n"
                        + result[abs_m.end(2):]
                    )
                    logger.info("Applied abstract revision")
                else:
                    logger.warning("Cannot find abstract in paper — revision discarded")
                continue

            # Determine the level of the section being revised
            # by finding its command in the document
            esc_heading = re.escape(heading)
            heading_match = re.search(
                r"\\((?:sub){0,2})section\*?\{" + esc_heading + r"\}",
                result[body_start:],
            )
            if not heading_match:
                logger.warning(
                    "Cannot find section '%s' in paper — revision discarded", heading
                )
                continue

            section_level = heading_match.group(1).count("sub")

            # For top-level sections (\section{}), match everything up to the
            # next \section{} (same level), \end{document}, or \bibliography.
            # This includes all subsections within it.
            if section_level == 0:
                pattern = (
                    r"(\\section\*?\{" + esc_heading + r"\})"
                    r"(.*?)"
                    r"(?=\\section\*?\{|\\end\{document\}|\\bibliography)"
                )
            else:
                # For subsections, match up to the next same-or-higher-level section
                pattern = (
                    r"(\\(?:sub){0,2}section\*?\{" + esc_heading + r"\})"
                    r"(.*?)"
                    r"(?=\\(?:sub){0,2}section\*?\{|\\end\{document\}|\\bibliography)"
                )
            match = re.search(pattern, result[body_start:], re.DOTALL)
            if not match:
                logger.warning(
                    "Cannot find section '%s' in paper — revision discarded", heading
                )
                continue

            old_content = match.group(2)
            abs_start = body_start + match.start(2)
            abs_end = body_start + match.end(2)

            # Preserve figure/table environments from old content
            # that may have been dropped by the revision LLM
            old_figures = re.findall(
                r'(\\begin\{figure\*?\}.*?\\end\{figure\*?\})',
                old_content, re.DOTALL,
            )
            old_tables = re.findall(
                r'(\\begin\{table\*?\}.*?\\end\{table\*?\})',
                old_content, re.DOTALL,
            )
            preserved = []
            for fig_block in old_figures:
                # Only preserve if new content doesn't already have this figure
                label_match = re.search(r'\\label\{([^}]+)\}', fig_block)
                if label_match:
                    label = label_match.group(1)
                    # Exact label match (not substring) to avoid fig:method matching fig:method_base
                    if not re.search(r'\\label\{' + re.escape(label) + r'\}', new_content):
                        preserved.append(fig_block)
                else:
                    # No label — check if specific includegraphics file is already present
                    file_m = re.search(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', fig_block)
                    if file_m:
                        fname = re.escape(file_m.group(1))
                        if not re.search(r'\\includegraphics(?:\[[^\]]*\])?\{' + fname + r'\}', new_content):
                            preserved.append(fig_block)
                    elif 'includegraphics' not in new_content:
                        preserved.append(fig_block)
            for tbl_block in old_tables:
                label_match = re.search(r'\\label\{([^}]+)\}', tbl_block)
                if label_match:
                    label = label_match.group(1)
                    # Exact label match (not substring)
                    if not re.search(r'\\label\{' + re.escape(label) + r'\}', new_content):
                        preserved.append(tbl_block)
                elif 'caption' in tbl_block and '\\begin{tabular}' not in new_content:
                    preserved.append(tbl_block)

            suffix = ""
            if preserved:
                suffix = "\n\n" + "\n\n".join(preserved)

            result = (
                result[:abs_start]
                + "\n" + new_content + suffix + "\n\n"
                + result[abs_end:]
            )
        return result
