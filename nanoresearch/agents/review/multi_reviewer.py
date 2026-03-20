"""Multi-model review committee mixin."""

from __future__ import annotations

import logging
from typing import Any

from nanoresearch.agents.tools import ToolDefinition, ToolRegistry
from nanoresearch.schemas.review import (
    ReviewOutput,
    SectionReview,
)

logger = logging.getLogger(__name__)


class _MultiReviewerMixin:
    """Mixin — multi-model review committee methods."""

    async def _build_review_tools(self) -> ToolRegistry | None:
        """Build a ToolRegistry with search tools for reviewing.

        Returns None if no tools could be registered.
        """
        registry = ToolRegistry()

        try:
            from mcp_server.tools.arxiv_search import search_arxiv
            from mcp_server.tools.openalex import search_openalex

            async def _search_papers(query: str, max_results: int = 5) -> list[dict]:
                results: list[dict] = []
                try:
                    results.extend(await search_arxiv(query, max_results=max_results))
                except Exception as exc:
                    logger.debug("arxiv search failed: %s", exc)
                try:
                    results.extend(await search_openalex(query, max_results=max_results))
                except Exception as exc:
                    logger.debug("openalex search failed: %s", exc)
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
- Method: {str((experiment_blueprint.get('proposed_method') or {{}}).get('name', 'Unknown'))[:500]}

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

        # BUG-6 fix: do NOT inflate score when issues list is empty.
        # Preserve the original low score so downstream revision logic
        # receives the correct severity signal.  Step 2b will inject
        # generic issues for sections below MIN_SECTION_SCORE anyway.

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
