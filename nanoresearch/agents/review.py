"""Review agent — automated paper review, consistency checking, and revision."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from nanoresearch.agents.base import BaseResearchAgent
from nanoresearch.agents.tools import ToolDefinition, ToolRegistry
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

REVIEW_SYSTEM_PROMPT = """You are an expert academic paper reviewer for a top-tier venue (NeurIPS, ICML, CVPR, ACL).

Scoring rubric per section:
  9-10: Publication-ready, minor polishing only
  7-8:  Solid but has identifiable weaknesses
  5-6:  Significant issues that need addressing
  3-4:  Major rewrite needed
  1-2:  Fundamentally flawed

Evaluation criteria:
- **Technical correctness**: Are claims supported? Are equations correct?
- **Completeness**: Are all necessary details present (datasets, baselines, ablations)?
- **Clarity**: Is the writing clear, precise, and well-organized?
- **Citations**: Are key works cited? Are citation keys valid?
- **Reproducibility**: Could someone reproduce the experiments from the description?
- **Consistency**: Same notation and terminology across sections?

Always respond in valid JSON format."""

REVISION_SYSTEM_PROMPT = """You are an expert academic paper writer revising a section for a top-tier AI venue (NeurIPS, ICML, CVPR, ACL).

Your revision must meet publication-ready standards:
- Fix ALL issues listed by the reviewer
- Maintain LaTeX formatting and notation consistency with the rest of the paper
- Use \\citet{} for grammatical-subject citations, \\citep{} for parenthetical
- Keep notation consistent: \\mathbf{x} for vectors, \\mathbf{W} for matrices
- Include equations with \\begin{equation} and reference them with Eq.~\\eqref{}
- Be specific and quantitative — avoid vague claims like "significant improvement"
- Every claim must be supported by evidence or citations
- Use ONLY citation keys from the paper's bibliography — do NOT invent new ones
- Match the writing quality of published papers at NeurIPS/ICML

Output ONLY the revised section content (LaTeX formatted). No explanation, no markdown fences."""


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

        if not paper_tex:
            self.log("No paper.tex content available, skipping review")
            return ReviewOutput().model_dump(mode="json")

        self.log("Starting automated review")

        # Step 1: LLM review — score each section (split-based for full coverage)
        review = await self._review_paper(paper_tex, ideation_output, experiment_blueprint)
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
            if not low_sections and not consistency_issues:
                break

            revision_round += 1
            # Clear per-round revised sections to avoid stale accumulation
            round_revised: dict[str, str] = {}
            self.log(
                f"Revision round {revision_round}: "
                f"revising {len(low_sections)} sections, "
                f"{len(consistency_issues)} consistency issues"
            )

            for section_review in low_sections:
                # Collect consistency issues relevant to this section
                section_consistency = [
                    ci for ci in consistency_issues
                    if any(
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

            # Re-run consistency checks after revision
            consistency_issues = self._run_consistency_checks(current_tex)
            if consistency_issues:
                self.log(f"  {len(consistency_issues)} consistency issues remain after revision")

            # Re-review revised sections with LLM
            re_review = await self._review_paper(current_tex, ideation_output, experiment_blueprint)
            for new_sr in re_review.section_reviews:
                for old_sr in review.section_reviews:
                    if old_sr.section == new_sr.section:
                        old_sr.score = new_sr.score
                        old_sr.issues = new_sr.issues
                        old_sr.suggestions = new_sr.suggestions
                        break

            # Validate review coherence: flag score-vs-issues mismatches
            for sr in review.section_reviews:
                if sr.score >= 8 and len(sr.issues) > 3:
                    logger.warning(
                        "Review coherence: %s scored %d but has %d issues",
                        sr.section, sr.score, len(sr.issues),
                    )
                    sr.score = min(sr.score, 7)  # cap score if too many issues
                elif sr.score < MIN_SECTION_SCORE and len(sr.issues) == 0:
                    logger.warning(
                        "Review coherence: %s scored %d but has 0 issues, injecting generic issues",
                        sr.section, sr.score,
                    )
                    sr.issues = [
                        f"Section '{sr.section}' scored {sr.score}/10. "
                        "Improve clarity, depth, citations, and remove placeholder text."
                    ]
                    sr.suggestions = [
                        "Rewrite with concrete data, proper references, and "
                        "remove any 'results pending' or LLM artifact text."
                    ]

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
        pattern = re.compile(
            r"\\((?:sub){0,2})section\*?\{([^}]+)\}",
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

        # Count open brackets/braces and close them
        opens = {'[': 0, '{': 0}
        closes = {']': '[', '}': '{'}
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
            if ch in opens:
                opens[ch] += 1
            elif ch in closes:
                opens[closes[ch]] = max(0, opens[closes[ch]] - 1)

        repaired += ']' * opens['[']
        repaired += '}' * opens['{']

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

    async def _review_single_section(
        self,
        heading: str,
        content: str,
        ideation_output: dict,
        experiment_blueprint: dict,
        review_config: Any,
    ) -> SectionReview:
        """Review a single section of the paper."""
        prompt = f"""Review the following section of an academic paper:

Section: {heading}

```latex
{content[:8000]}
```

Research context:
- Topic: {str(ideation_output.get('topic', 'Unknown'))[:500]}
- Hypothesis: {str(ideation_output.get('selected_hypothesis', 'Unknown'))[:500]}
- Method: {str((experiment_blueprint.get('proposed_method') or {}).get('name', 'Unknown'))[:500]}

Provide:
1. A quality score (1-10) using the rubric
2. Up to 5 specific issues (be concrete)
3. Up to 3 actionable suggestions

Return JSON:
{{
    "section": "{heading}",
    "score": 7,
    "issues": ["Issue 1"],
    "suggestions": ["Suggestion 1"]
}}"""

        try:
            result = await self.generate_json(
                REVIEW_SYSTEM_PROMPT, prompt, stage_override=review_config
            )
        except Exception:
            # JSON parse failed — try repair
            raw = await self.generate(
                REVIEW_SYSTEM_PROMPT, prompt, json_mode=True,
                stage_override=review_config,
            )
            result = self._repair_truncated_json(raw)
            if result is None:
                logger.warning("Could not parse review for section %s, using defaults", heading)
                result = {"section": heading, "score": 5, "issues": [], "suggestions": []}

        # Safely coerce score to int (LLM may return float or string)
        raw_score = result.get("score", 5)
        try:
            score = int(float(raw_score))
        except (TypeError, ValueError):
            score = 5
        return SectionReview(
            section=result.get("section", heading),
            score=max(1, min(10, score)),
            issues=result.get("issues", [])[:5],
            suggestions=result.get("suggestions", [])[:3],
        )

    async def _review_paper(
        self,
        paper_tex: str,
        ideation_output: dict,
        experiment_blueprint: dict,
    ) -> ReviewOutput:
        """Review the paper section-by-section to avoid JSON truncation."""
        sections = self._extract_sections(paper_tex)
        review_config = self.config.for_stage("review")

        # Review only top-level sections (level=0), skip sub/subsub sections
        main_sections = [
            (h, c) for h, c, level in sections if level == 0
        ]
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
        """Revise a single section based on reviewer feedback."""
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

        # Section-specific revision guidance
        section_guidance = self._get_section_revision_guidance(section_review.section)

        # Extract bibliography from paper_tex so the LLM knows available citations
        bib_keys = ""
        bib_match = re.findall(r'\\bibitem\{([^}]+)\}|@\w+\{([^,]+),', paper_tex)
        if bib_match:
            keys = [m[0] or m[1] for m in bib_match[:50]]
            bib_keys = f"\n\nAvailable citation keys: {', '.join(keys)}"

        # Smart truncation: preserve both content and bibliography
        tex_for_prompt = paper_tex
        if len(paper_tex) > 20000:
            # Keep first 12000 chars (content) + last 8000 chars (bibliography)
            tex_for_prompt = paper_tex[:12000] + "\n...[middle truncated]...\n" + paper_tex[-8000:]

        prompt = f"""Revise the "{section_review.section}" section of this paper.

Issues found:
{issues_json}

Suggestions:
{suggestions_json}{consistency_block}

{section_guidance}
{bib_keys}

Current paper (LaTeX):
```latex
{tex_for_prompt}
```

Research topic: {ideation_output.get('topic', '')}

Write an improved version of the "{section_review.section}" section.
Output ONLY the LaTeX content for this section (no \\section command, just the body text).
IMPORTANT: If the section contains \\begin{{figure}}...\\end{{figure}} or \\begin{{table}}...\\end{{table}} blocks, you MUST preserve them in your output. Do NOT remove figures or tables."""

        review_config = self.config.for_stage("review")
        try:
            revised = await self.generate(
                REVISION_SYSTEM_PROMPT, prompt, stage_override=review_config
            )
            return revised.strip()
        except Exception as e:
            logger.warning("Failed to revise section '%s': %s", section_review.section, e)
            return ""

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
            if m.get("name")
        }
        bp_datasets = {
            d.get("name", "").lower()
            for d in blueprint.get("datasets", [])
            if d.get("name")
        }
        bp_baselines = {
            b.get("name", "").lower()
            for b in blueprint.get("baselines", [])
            if b.get("name")
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
                r_authors = " ".join(r.get("authors", []))
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
                author_str = " and ".join(authors[:5]) if authors else surname
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
        """Sanitize revised LaTeX using WritingAgent's sanitizer."""
        try:
            from nanoresearch.agents.writing import WritingAgent
            tex = WritingAgent._sanitize_latex(tex)
        except ImportError:
            logger.debug("WritingAgent not available for sanitization, skipping")
        return tex

    async def _compile_pdf_with_fix_loop(self, tex_path: Path) -> dict:
        """Compile LaTeX to PDF with automatic error-fix loop.

        If compilation fails, feed the error back to the LLM, apply the fix,
        and retry up to MAX_LATEX_FIX_ATTEMPTS times. This mirrors the
        WritingAgent's _compile_pdf logic.
        """
        try:
            from mcp_server.tools.pdf_compile import compile_pdf
        except ImportError:
            return {"error": "PDF compiler module not available"}

        tex_path = Path(tex_path)

        result: dict = {}
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

            # Don't retry if the problem isn't fixable via LaTeX edits
            if "not found" in error_msg.lower() or "not available" in error_msg.lower():
                self.log("No LaTeX compiler available, skipping fix loop")
                return result

            if attempt >= MAX_LATEX_FIX_ATTEMPTS:
                self.log(f"PDF compilation failed after {MAX_LATEX_FIX_ATTEMPTS} fix attempts")
                return result

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
                self.log(f"  Applied LLM fix (attempt {attempt + 1})")
            else:
                self.log("  LLM returned no changes, aborting fix loop")
                return result

        return result

    async def _fix_latex_errors(self, tex_source: str, error_log: str) -> str | None:
        """Ask the LLM to fix LaTeX compilation errors with root-cause analysis.

        Feeds the compilation error + LaTeX source to the LLM, gets back
        the corrected full LaTeX document.
        """
        # Truncate very long error logs to fit in context
        if len(error_log) > 3000:
            error_log = error_log[:1500] + "\n...[truncated]...\n" + error_log[-1500:]

        # Root-cause analysis: classify the error type for targeted fix
        error_lower = error_log.lower()
        targeted_guidance = ""
        if "missing \\begin{document}" in error_lower or "missing begin{document}" in error_lower:
            targeted_guidance = (
                "\nDIAGNOSIS: The LaTeX file is missing or has a corrupted preamble. "
                "This usually means non-LaTeX content (e.g., HTML comments, XML tags, "
                "meta-commentary text) appears BEFORE \\begin{document}. "
                "FIX: Ensure the file starts with \\documentclass and that NOTHING except "
                "valid LaTeX preamble commands appears before \\begin{document}.\n"
            )
        elif "undefined control sequence" in error_lower:
            targeted_guidance = (
                "\nDIAGNOSIS: An undefined LaTeX command is used. "
                "FIX: Check for typos in command names, ensure required packages are loaded, "
                "or replace with standard alternatives.\n"
            )
        elif "mismatched" in error_lower or "begin" in error_lower and "end" in error_lower:
            targeted_guidance = (
                "\nDIAGNOSIS: Mismatched \\begin/\\end environments. "
                "FIX: Ensure every \\begin{X} has a matching \\end{X}. Check for nested "
                "environments that are not properly closed.\n"
            )
        elif "unicode" in error_lower or "utf" in error_lower or "character" in error_lower:
            targeted_guidance = (
                "\nDIAGNOSIS: Unicode characters that LaTeX cannot handle. "
                "FIX: Replace em-dash (\u2014) with ---, en-dash (\u2013) with --, "
                "smart quotes with standard quotes, Greek letters with \\alpha etc.\n"
            )

        system = (
            "You are a LaTeX expert. Fix the compilation errors in the given LaTeX source.\n"
            "Return the COMPLETE fixed LaTeX document. Do NOT omit any sections.\n"
            "Do NOT wrap in markdown fences. Output ONLY the LaTeX source.\n"
            "The output must start with \\documentclass and end with \\end{document}.\n"
            "Do NOT include any HTML comments, XML tags, or non-LaTeX content.\n\n"
            "Common fixes:\n"
            "- Replace Unicode characters (em-dash, en-dash, smart quotes, Greek letters) "
            "with LaTeX commands\n"
            "- Fix unescaped special characters: % # & _ $ { }\n"
            "- Fix mismatched \\begin/\\end environments\n"
            "- Fix malformed table/tabular environments\n"
            "- Remove or replace unsupported packages\n"
            "- Add missing packages (e.g. \\usepackage{bm} for \\bm command)\n"
            "- Fix \\includegraphics paths\n"
            "- Fix malformed \\cite, \\ref, \\label commands\n\n"
            "Overfull hbox fixes:\n"
            "- Tables: add \\small, \\setlength{\\tabcolsep}{4pt}, @{} in column spec, "
            "or wrap in \\resizebox{\\textwidth}{!}{...}\n"
            "- Long inline math: break into \\begin{align} or \\begin{multline}\n"
            "- Long text: rewrite the sentence shorter or add \\linebreak"
        )

        # Smart truncation: extract error line and send a focused window
        # This avoids sending the entire 60KB+ document and expecting 60KB+ back
        error_line = None
        line_match = re.search(r'(?:line\s+|l\.)(\d+)', error_log)
        if line_match:
            error_line = int(line_match.group(1))

        tex_lines = tex_source.split('\n')
        if error_line and len(tex_source) > 30000:
            # Send preamble (first 50 lines) + window around error + last 30 lines
            preamble_end = min(50, len(tex_lines))
            window_start = max(preamble_end, error_line - 30)
            window_end = min(len(tex_lines), error_line + 30)
            tail_start = max(window_end, len(tex_lines) - 30)

            focused_lines = tex_lines[:preamble_end]
            if window_start > preamble_end:
                focused_lines.append(f"% ... [lines {preamble_end+1}-{window_start} omitted] ...")
            focused_lines.extend(tex_lines[window_start:window_end])
            if tail_start > window_end:
                focused_lines.append(f"% ... [lines {window_end+1}-{tail_start} omitted] ...")
            focused_lines.extend(tex_lines[tail_start:])
            tex_for_prompt = '\n'.join(focused_lines)

            prompt = (
                f"The following LaTeX document failed to compile at line {error_line}.\n\n"
                f"=== COMPILATION ERROR ===\n{error_log}\n=== END ERROR ===\n"
                f"{targeted_guidance}\n"
                f"I'm showing the preamble + a window around the error line + the end of the document.\n"
                f"=== LATEX SOURCE (focused) ===\n{tex_for_prompt}\n=== END SOURCE ===\n\n"
                f"Fix the error and return the COMPLETE fixed LaTeX document.\n"
                f"The document MUST start with \\documentclass and end with \\end{{document}}.\n"
                f"For omitted sections, reproduce them exactly as they were."
            )
        else:
            # Fallback for short documents or no line number
            if len(tex_source) > 50000:
                tex_source = tex_source[:25000] + "\n...[truncated]...\n" + tex_source[-25000:]
            prompt = (
                f"The following LaTeX document failed to compile.\n\n"
                f"=== COMPILATION ERROR ===\n{error_log}\n=== END ERROR ===\n"
                f"{targeted_guidance}\n"
                f"=== LATEX SOURCE ===\n{tex_source}\n=== END SOURCE ===\n\n"
                f"Fix ALL errors and return the complete corrected LaTeX document.\n"
                f"The document MUST start with \\documentclass and end with \\end{{document}}."
            )

        review_config = self.config.for_stage("review")
        try:
            fixed = await self.generate(
                system, prompt, stage_override=review_config
            )
            # Strip markdown fences if present
            fixed = fixed.strip()
            if fixed.startswith("```"):
                lines = fixed.split("\n")
                lines = lines[1:]
                if lines and lines[-1].strip().startswith("```"):
                    lines = lines[:-1]
                fixed = "\n".join(lines)
            # Strip any leading non-LaTeX content (meta-commentary, HTML, etc.)
            if "\\documentclass" in fixed:
                idx = fixed.index("\\documentclass")
                if idx > 0:
                    discarded = fixed[:idx].strip()
                    if discarded:
                        self.log(f"  Stripping {len(discarded)} chars of non-LaTeX preamble")
                    fixed = fixed[idx:]
            # Sanity check: must contain \documentclass
            if "\\documentclass" not in fixed:
                self.log("  LLM fix missing \\documentclass, discarding")
                return None
            return fixed
        except Exception as e:
            logger.warning("LLM latex fix failed: %s", e)
            return None

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
            pattern = (
                r"(\\(?:sub){0,2}section\*?\{" + re.escape(heading) + r"\})"
                r"(.*?)"
                r"(?=\\(?:sub){0,2}section\*?\{|\\end\{document\}|\\bibliography)"
            )
            match = re.search(pattern, result[body_start:], re.DOTALL)
            if match:
                old_content = match.group(2)
                abs_start = body_start + match.start(2)
                abs_end = body_start + match.end(2)

                # Preserve figure/table environments from old content
                # that may have been dropped by the revision LLM
                old_figures = re.findall(
                    r'(\\begin\{figure\}.*?\\end\{figure\})',
                    old_content, re.DOTALL,
                )
                old_tables = re.findall(
                    r'(\\begin\{table\}.*?\\end\{table\})',
                    old_content, re.DOTALL,
                )
                preserved = []
                for fig_block in old_figures:
                    # Only preserve if new content doesn't already have this figure
                    label_match = re.search(r'\\label\{([^}]+)\}', fig_block)
                    if label_match:
                        label = label_match.group(1)
                        if label not in new_content:
                            preserved.append(fig_block)
                    elif 'includegraphics' in fig_block and 'includegraphics' not in new_content:
                        preserved.append(fig_block)
                for tbl_block in old_tables:
                    label_match = re.search(r'\\label\{([^}]+)\}', tbl_block)
                    if label_match:
                        label = label_match.group(1)
                        if label not in new_content:
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
