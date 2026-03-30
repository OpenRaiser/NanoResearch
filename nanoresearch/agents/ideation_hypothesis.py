"""Ideation hypothesis mixin -- tool building, analysis, GitHub, evidence."""

from __future__ import annotations

import json
import logging
from typing import Any

from nanoresearch.agents.tools import ToolDefinition, ToolRegistry
from nanoresearch.schemas.evidence import EvidenceBundle, ExtractedMetric
from nanoresearch.schemas.ideation import IdeationOutput, PaperReference

logger = logging.getLogger(__name__)

from nanoresearch.agents.ideation import (
    MAX_PAPERS_FOR_ANALYSIS,
    MAX_ABSTRACT_LENGTH,
    MAX_METHOD_TEXT_PER_PAPER,
    MAX_EXPERIMENT_TEXT_PER_PAPER,
    MAX_GITHUB_REPOS,
    MAX_GITHUB_QUERIES,
    IDEATION_SYSTEM_PROMPT,
    _get_arxiv_search,
    _get_s2_search,
    _get_oa_search,
    _get_github_search,
)

from nanoresearch.skill_prompts import (
    IDEATION_ANALYSIS_SYSTEM,
    IDEATION_EVIDENCE_SYSTEM,
)


class _IdeationHypothesisMixin:
    """Mixin with tool building, analysis/hypothesis, GitHub, evidence methods."""

    async def _build_search_tools(self) -> ToolRegistry:
        registry = ToolRegistry()
        search_arxiv = await _get_arxiv_search()
        search_s2 = await _get_s2_search()

        _arxiv_categories = [
            "cs.LG", "cs.AI", "cs.CV", "cs.CL",
            "q-bio.BM", "q-bio.QM", "physics.chem-ph",
            "cond-mat.mtrl-sci", "stat.ML",
        ]

        async def _handle_arxiv(query, max_results=10):
            return await search_arxiv(
                query, max_results=max_results, categories=_arxiv_categories,
            )

        async def _handle_s2(query, max_results=10):
            return await search_s2(query, max_results=max_results)

        registry.register(ToolDefinition(
            name="search_arxiv",
            description="Search arXiv for academic papers by query. Returns paper metadata.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "description": "Max papers", "default": 10},
                },
                "required": ["query"],
            },
            handler=_handle_arxiv,
        ))

        registry.register(ToolDefinition(
            name="search_semantic_scholar",
            description="Search Semantic Scholar for papers with citation data.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "description": "Max papers", "default": 10},
                },
                "required": ["query"],
            },
            handler=_handle_s2,
        ))

        search_oa = await _get_oa_search()
        if search_oa:
            async def _handle_oa(query, max_results=10):
                return await search_oa(query, max_results=max_results)

            registry.register(ToolDefinition(
                name="search_openalex",
                description="Search OpenAlex for papers. Large quota, good citation counts. Covers ~250M works.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "max_results": {"type": "integer", "description": "Max papers", "default": 10},
                    },
                    "required": ["query"],
                },
                handler=_handle_oa,
            ))

        try:
            from mcp_server.tools.web_search import search_web
            registry.register(ToolDefinition(
                name="search_web",
                description="Search the web for general information. Returns titles, URLs, snippets.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "max_results": {"type": "integer", "default": 5},
                    },
                    "required": ["query"],
                },
                handler=search_web,
            ))
        except ImportError:
            pass

        try:
            from mcp_server.tools.semantic_scholar import get_paper_details
            registry.register(ToolDefinition(
                name="get_paper_details",
                description="Get detailed info about a paper by Semantic Scholar or arXiv ID.",
                parameters={
                    "type": "object",
                    "properties": {
                        "paper_id": {"type": "string", "description": "Paper ID"},
                    },
                    "required": ["paper_id"],
                },
                handler=get_paper_details,
            ))
        except ImportError:
            pass

        return registry

    async def _analyze_and_hypothesize(
        self, topic: str, queries: list[str], papers: list[dict]
    ) -> IdeationOutput:
        paper_summaries = []
        for i, p in enumerate(papers[:MAX_PAPERS_FOR_ANALYSIS]):
            abstract_text = (p.get('abstract', '') or '')[:300]
            method_text = (p.get('method_text', '') or '')[:MAX_METHOD_TEXT_PER_PAPER]
            experiment_text = (p.get('experiment_text', '') or '')[:MAX_EXPERIMENT_TEXT_PER_PAPER]

            summary = (
                f"[{i+1}] {p.get('title', 'Unknown')} ({p.get('year', '?')})\n"
                f"    Authors: {', '.join(a.get('name', str(a)) if isinstance(a, dict) else str(a) for a in (p.get('authors') or [])[:3])}\n"
                f"    Citations: {p.get('citation_count', 0)}\n"
                f"    Abstract: {abstract_text}..."
            )
            if method_text:
                summary += f"\n    Method Summary: {method_text}..."
            if experiment_text:
                summary += f"\n    Experiment Summary: {experiment_text}..."
            paper_summaries.append(summary)

        papers_text = "\n\n".join(paper_summaries)

        prompt = f"""Research Topic: "{topic}"

I searched using these queries: {json.dumps(queries)}

Here are the retrieved papers:
{papers_text}

Analyze these papers and produce a JSON object with:
1. "survey_summary": A 300-500 word narrative summarizing the state of the field.
   Include what methods dominate (e.g. "80% of papers use X"), which datasets are standard,
   and what the current SOTA performance is.

2. "gaps": Array of 3-5 research gaps, each with:
   - "gap_id": "GAP-001", "GAP-002", etc.
   - "description": What is missing/underexplored
   - "supporting_refs": List of paper indices that support this gap
   - "severity": "low", "medium", or "high"
   - "quantitative_evidence": e.g. "Only 2/15 papers address X" or "No paper combines A with B"
   - "future_work_mention": Which paper(s) explicitly mention this as future work (if any)
   Gaps should be categorized: method gap, dataset gap, application gap, or theory gap.

3. "hypotheses": Array of 2-4 hypotheses, each with:
   - "hypothesis_id": "HYP-001", "HYP-002", etc.
   - "statement": Concise hypothesis
   - "gap_refs": Which gaps this addresses
   - "novelty_justification": Why this is novel. MUST explain how it differs from the closest
     existing work. Name the closest paper and state the specific difference.
   - "feasibility_notes": Practical feasibility -- required compute (GPU type, hours),
     required data (publicly available?), implementation complexity (simple/moderate/hard)
   - "closest_existing_work": Title of the most similar published paper and how your idea differs
4. "selected_hypothesis": The hypothesis_id of the most promising one
5. "rationale": Why this hypothesis was selected (2-3 sentences)

NOVELTY VERIFICATION (critical):
Before finalizing hypotheses, you MUST use the search tools to:
1. Search for papers with ideas similar to EACH hypothesis
2. If a highly similar paper exists, either refine the hypothesis to be clearly different or discard it
3. The novelty_justification must reference actual searched papers, not speculation

Return ONLY valid JSON."""

        try:
            tools = await self._build_search_tools()
            if len(tools) > 0:
                react_result = await self.generate_with_tools(
                    IDEATION_ANALYSIS_SYSTEM, prompt, tools, max_tool_rounds=2
                )
                text = react_result.strip()
                if text.startswith("```"):
                    lines = text.split("\n")
                    lines = lines[1:]
                    if lines and lines[-1].strip().startswith("```"):
                        lines = lines[:-1]
                    text = "\n".join(lines)
                try:
                    result = json.loads(text)
                except json.JSONDecodeError as e:
                    logger.warning(
                        "[%s] ReAct output was not valid JSON (%s), "
                        "falling back to standard generation. Output preview: %r",
                        self.stage.value, e, text[:200],
                    )
                    result = await self.generate_json(IDEATION_ANALYSIS_SYSTEM, prompt)
            else:
                result = await self.generate_json(IDEATION_ANALYSIS_SYSTEM, prompt)
        except Exception as e:
            logger.warning("[%s] ReAct tool-use failed, falling back to standard generation: %s",
                           self.stage.value, e)
            result = await self.generate_json(IDEATION_ANALYSIS_SYSTEM, prompt)

        if isinstance(result, list):
            logger.warning(
                "[%s] LLM returned a list instead of dict; wrapping as {'hypotheses': ...}",
                self.stage.value,
            )
            result = {"hypotheses": result}

        paper_refs = []
        for p in papers:
            try:
                paper_refs.append(PaperReference(
                    paper_id=p.get("paper_id", p.get("arxiv_id", "")),
                    title=p.get("title", ""),
                    authors=p.get("authors") or [],
                    year=p.get("year"),
                    abstract=(p.get("abstract", "") or "")[:MAX_ABSTRACT_LENGTH],
                    venue=p.get("venue", ""),
                    citation_count=p.get("citation_count") or 0,
                    url=p.get("url", ""),
                    method_text=(p.get("method_text", "") or "")[:MAX_METHOD_TEXT_PER_PAPER],
                    experiment_text=(p.get("experiment_text", "") or "")[:MAX_EXPERIMENT_TEXT_PER_PAPER],
                ))
            except Exception as exc:
                logger.warning("Skipping malformed paper entry: %s (error: %s)", p.get("title", "?"), exc)

        return IdeationOutput(
            topic=topic,
            search_queries=queries,
            papers=paper_refs,
            survey_summary=result.get("survey_summary", ""),
            gaps=[
                {
                    "gap_id": g.get("gap_id", f"GAP-{i+1:03d}"),
                    "description": g.get("description", ""),
                    "supporting_refs": [str(r) for r in g.get("supporting_refs", [])],
                    "severity": g.get("severity", "medium"),
                    "quantitative_evidence": g.get("quantitative_evidence", ""),
                    "future_work_mention": g.get("future_work_mention", ""),
                }
                for i, g in enumerate(result.get("gaps", []))
                if isinstance(g, dict)
            ],
            hypotheses=[
                {
                    "hypothesis_id": h.get("hypothesis_id", f"HYP-{i+1:03d}"),
                    "statement": h.get("statement", ""),
                    "gap_refs": h.get("gap_refs", []),
                    "novelty_justification": h.get("novelty_justification", ""),
                    "feasibility_notes": (
                        json.dumps(h["feasibility_notes"], ensure_ascii=False)
                        if isinstance(h.get("feasibility_notes"), dict)
                        else str(h.get("feasibility_notes", ""))
                    ),
                    "closest_existing_work": h.get("closest_existing_work", ""),
                }
                for i, h in enumerate(result.get("hypotheses", []))
                if isinstance(h, dict)
            ],
            selected_hypothesis=result.get("selected_hypothesis", ""),
            rationale=result.get("rationale", ""),
        )

    # ── Idea Tournament: pairwise comparison for hypothesis selection ──

    _TOURNAMENT_SYSTEM = (
        "You are a research hypothesis evaluator. Compare two hypotheses and pick the stronger one.\n"
        "Consider: novelty, feasibility, potential impact, and clarity.\n"
        "Reply with ONLY the hypothesis_id of the winner (e.g. HYP-001). Nothing else."
    )

    async def _tournament_select(
        self, hypotheses: list[dict], topic: str
    ) -> str:
        """Pairwise tournament to select the best hypothesis.

        Each pair is compared via a cheap LLM call. The hypothesis with the
        most wins is selected. Ties are broken by original order.

        Returns:
            hypothesis_id of the winner.
        """
        if len(hypotheses) <= 1:
            return hypotheses[0].get("hypothesis_id", "HYP-001") if hypotheses else ""

        ids = [h.get("hypothesis_id", f"HYP-{i+1:03d}") for i, h in enumerate(hypotheses)]
        wins: dict[str, int] = {hid: 0 for hid in ids}

        # Use the cheapest available model for comparisons
        from nanoresearch.config import StageModelConfig
        cheap_config = StageModelConfig(
            model=self.config.evidence_extraction.model,
            temperature=0.1,
            max_tokens=32,
        )

        pairs = [(i, j) for i in range(len(hypotheses)) for j in range(i + 1, len(hypotheses))]
        self.log(f"Tournament: {len(hypotheses)} hypotheses, {len(pairs)} comparisons")

        async def _compare(i: int, j: int) -> str | None:
            h_a, h_b = hypotheses[i], hypotheses[j]
            prompt = (
                f"Research topic: {topic}\n\n"
                f"Hypothesis A ({ids[i]}):\n"
                f"  Statement: {h_a.get('statement', '')}\n"
                f"  Novelty: {h_a.get('novelty_justification', '')}\n"
                f"  Feasibility: {h_a.get('feasibility_notes', '')}\n\n"
                f"Hypothesis B ({ids[j]}):\n"
                f"  Statement: {h_b.get('statement', '')}\n"
                f"  Novelty: {h_b.get('novelty_justification', '')}\n"
                f"  Feasibility: {h_b.get('feasibility_notes', '')}\n\n"
                f"Which is stronger? Reply with ONLY the hypothesis_id."
            )
            try:
                result = await self.generate(
                    self._TOURNAMENT_SYSTEM, prompt, stage_override=cheap_config,
                )
                result = result.strip()
                # Extract hypothesis_id from response
                for hid in ids:
                    if hid in result:
                        return hid
                return None
            except Exception as e:
                logger.debug("Tournament comparison failed: %s", e)
                return None

        # Run all comparisons concurrently
        import asyncio
        results = await asyncio.gather(*(_compare(i, j) for i, j in pairs))

        for winner in results:
            if winner and winner in wins:
                wins[winner] += 1

        # Sort by wins (descending), tie-break by original order
        ranked = sorted(ids, key=lambda hid: -wins[hid])
        best = ranked[0]

        win_str = ", ".join(f"{hid}={wins[hid]}W" for hid in ranked)
        self.log(f"Tournament results: {win_str} -> winner: {best}")
        return best

    async def _search_github_repos(self, topic: str, queries: list[str]) -> list[dict]:
        search_repos = await _get_github_search()
        all_repos: dict[str, dict] = {}
        search_terms = [topic] + queries[:MAX_GITHUB_QUERIES]
        for term in search_terms:
            try:
                results = await search_repos(term, max_results=3, language="Python")
                for repo in results:
                    key = repo.get("full_name", "")
                    if key and key not in all_repos:
                        all_repos[key] = repo
            except Exception as e:
                logger.warning("[%s] GitHub search failed for '%s': %s",
                               self.stage.value, term, e)
        repos = sorted(all_repos.values(), key=lambda r: r.get("stars", 0), reverse=True)
        return repos[:MAX_GITHUB_REPOS]

    async def _extract_evidence(self, papers: list[dict]) -> EvidenceBundle:
        paper_blocks = []
        for i, p in enumerate(papers[:MAX_PAPERS_FOR_ANALYSIS]):
            abstract = (p.get("abstract", "") or "")[:MAX_ABSTRACT_LENGTH]
            if not abstract.strip():
                continue
            paper_blocks.append(
                f"[PAPER {i+1}] id={p.get('paper_id', p.get('arxiv_id', 'unknown'))}\n"
                f"  title: {p.get('title', 'Unknown')}\n"
                f"  abstract: {abstract}"
            )

        if not paper_blocks:
            return EvidenceBundle(
                coverage_warnings=["No abstracts available for evidence extraction"]
            )

        papers_text = "\n\n".join(paper_blocks)
        system_prompt = IDEATION_EVIDENCE_SYSTEM

        prompt = f"""Extract quantitative results from these paper abstracts.

{papers_text}

For each explicitly stated metric, produce a JSON object with:
- "paper_id": the paper ID shown above
- "paper_title": the paper title
- "dataset": which dataset/benchmark the result is on (e.g. "QM9", "CASP14", "ImageNet")
- "metric_name": the metric name (e.g. "MAE", "GDT-TS", "Top-1 Accuracy")
- "value": the numeric value (as a number, or string if a range like "0.012-0.015")
- "unit": unit if stated (e.g. "eV", "Angstrom", "%"), empty string if none
- "context": the EXACT sentence or phrase from the abstract containing this number
- "method_name": name of the method that achieved this result
- "higher_is_better": true/false/null if unclear

RULES:
- Extract ONLY numbers explicitly written in the abstracts
- Do NOT estimate or calculate any values
- If no quantitative results are found, return an empty list
- Include the exact quote in "context"

Return JSON: {{"extracted_metrics": [...], "extraction_notes": "brief summary", "coverage_warnings": ["list any gaps"]}}"""

        evidence_config = self.config.for_stage("evidence_extraction")
        result = await self.generate_json(
            system_prompt, prompt, stage_override=evidence_config
        )

        if isinstance(result, list):
            logger.warning(
                "[%s] _extract_evidence: LLM returned a list; wrapping as {'extracted_metrics': ...}",
                self.stage.value,
            )
            result = {"extracted_metrics": result}
        metrics = []
        for m in result.get("extracted_metrics", []):
            try:
                metrics.append(ExtractedMetric(
                    paper_id=str(m.get("paper_id", "")),
                    paper_title=m.get("paper_title", ""),
                    dataset=m.get("dataset", ""),
                    metric_name=m.get("metric_name", ""),
                    value=m.get("value", ""),
                    unit=m.get("unit", ""),
                    context=m.get("context", ""),
                    method_name=m.get("method_name", ""),
                    higher_is_better=m.get("higher_is_better"),
                ))
            except Exception as exc:
                logger.warning(
                    "Skipping malformed metric entry: %s (error: %s)", m, exc
                )
                continue

        return EvidenceBundle(
            extracted_metrics=metrics,
            extraction_notes=result.get("extraction_notes", ""),
            coverage_warnings=result.get("coverage_warnings", []),
        )
