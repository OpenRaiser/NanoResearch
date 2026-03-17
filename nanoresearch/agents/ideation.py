"""Ideation agent -- literature search, gap analysis, hypothesis generation.

Split into 3 modules:
    ideation.py             -- IdeationAgent facade + run() + small helpers
    ideation_search.py      -- _IdeationSearchMixin (literature search/filter)
    ideation_hypothesis.py  -- _IdeationHypothesisMixin (tools, analysis, evidence)
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from nanoresearch.agents.base import BaseResearchAgent
from nanoresearch.agents.tools import ToolDefinition, ToolRegistry
from nanoresearch.schemas.evidence import EvidenceBundle, ExtractedMetric
from nanoresearch.schemas.ideation import IdeationOutput, PaperReference
from nanoresearch.schemas.manifest import PipelineStage

logger = logging.getLogger(__name__)

# --- Configurable limits (magic numbers extracted) ---
MAX_SEARCH_QUERIES = 5
MAX_RESULTS_PER_SEARCH = 10
MAX_PAPERS_FOR_ANALYSIS = 30          # was 50 -- reduced to save tokens
MAX_ABSTRACT_LENGTH = 500
MAX_GITHUB_REPOS = 5
MAX_GITHUB_QUERIES = 2

# Phase 4: Citation quality targets
TARGET_CITATION_COUNT = 30            # was 50 -- reduced to save tokens
MIN_HIGH_CITED_PAPERS = 8             # was 10 -- adjusted for smaller set
HIGH_CITATION_THRESHOLD = 100
TOP_K_FULL_TEXT = 4                   # was 8 -- PDF full-text is expensive

# Token budget limits for LLM prompts
MAX_METHOD_TEXT_PER_PAPER = 1000      # was 3000 -- method_text truncation
MAX_EXPERIMENT_TEXT_PER_PAPER = 1000  # was 3000 -- experiment_text truncation

# Lazy imports to avoid hard dependency on mcp_server at import time
_arxiv_search = None
_s2_search = None
_github_search = None
_oa_search = None
_import_lock = asyncio.Lock()


async def _get_arxiv_search():
    global _arxiv_search
    if _arxiv_search is None:
        async with _import_lock:
            if _arxiv_search is None:
                from mcp_server.tools.arxiv_search import search_arxiv
                _arxiv_search = search_arxiv
    return _arxiv_search


async def _get_s2_search():
    global _s2_search
    if _s2_search is None:
        async with _import_lock:
            if _s2_search is None:
                from mcp_server.tools.semantic_scholar import search_semantic_scholar
                _s2_search = search_semantic_scholar
    return _s2_search


async def _get_github_search():
    global _github_search
    if _github_search is None:
        async with _import_lock:
            if _github_search is None:
                from mcp_server.tools.github_search import search_repos
                _github_search = search_repos
    return _github_search


async def _get_oa_search():
    """Lazy import OpenAlex search (returns None if module unavailable)."""
    global _oa_search
    if _oa_search is None:
        async with _import_lock:
            if _oa_search is None:
                try:
                    from mcp_server.tools.openalex import search_openalex
                    _oa_search = search_openalex
                except ImportError:
                    _oa_search = False  # mark as unavailable
    return _oa_search if _oa_search else None


from nanoresearch.prompts import load_prompt as _load_prompt
from nanoresearch.skill_prompts import (
    IDEATION_QUERY_SYSTEM,
    IDEATION_ANALYSIS_SYSTEM,
    IDEATION_MUST_CITE_SYSTEM,
    IDEATION_EVIDENCE_SYSTEM,
)

# Legacy alias -- some internal methods still reference this.
IDEATION_SYSTEM_PROMPT = IDEATION_QUERY_SYSTEM

SEARCH_COVERAGE_SYSTEM_PROMPT = _load_prompt("ideation", "search_coverage")

# Import mixins (after constants are defined, since mixins import them)
from nanoresearch.agents.ideation_search import _IdeationSearchMixin      # noqa: E402
from nanoresearch.agents.ideation_hypothesis import _IdeationHypothesisMixin  # noqa: E402


class IdeationAgent(_IdeationSearchMixin, _IdeationHypothesisMixin, BaseResearchAgent):
    stage = PipelineStage.IDEATION

    async def run(self, **inputs: Any) -> dict[str, Any]:
        topic: str = inputs.get("topic", "")
        if not topic:
            raise ValueError("IdeationAgent requires a non-empty 'topic' in inputs")
        logger.info("[%s] Starting ideation for topic: %s", self.stage.value, topic)

        # Check for cached search results (from a previous failed attempt)
        cache_path = self.workspace.path / "logs" / "ideation_search_cache.json"
        cached = None
        if cache_path.is_file():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if not isinstance(cached, dict) or "papers" not in cached:
                    raise ValueError("invalid cache structure")
                self.log("Found cached search results from previous attempt, skipping search")
            except (json.JSONDecodeError, ValueError, OSError) as e:
                self.log(f"Search cache invalid ({e}), starting fresh")
                cached = None

        if (cached is not None
                and isinstance(cached, dict)
                and isinstance(cached.get("queries"), list)
                and isinstance(cached.get("papers"), list)):
            queries = cached["queries"]
            papers = cached["papers"]
            logger.info("[%s] Using cached: %d queries, %d papers",
                        self.stage.value, len(queries), len(papers))
            must_cites = cached.get("must_cites", [])
            if not must_cites:
                must_cites = await self._extract_must_cites(
                    [p for p in papers if "survey" in (p.get("title", "") or "").lower()
                     or "review" in (p.get("title", "") or "").lower()]
                )
        else:
            # Step 1: Generate search queries
            queries = await self._generate_queries(topic)
            logger.info("[%s] Generated %d search queries", self.stage.value, len(queries))

            # Step 2: Search literature
            papers = await self._search_literature(queries)
            logger.info("[%s] Retrieved %d papers", self.stage.value, len(papers))

            # Step 2b: Search for surveys and merge
            survey_papers = await self._search_surveys(topic)
            logger.info("[%s] Found %d survey papers", self.stage.value, len(survey_papers))
            existing_keys = {self._dedup_key(p) for p in papers}
            for sp in survey_papers:
                key = self._dedup_key(sp)
                if key and key not in existing_keys:
                    papers.append(sp)
                    existing_keys.add(key)

            # Step 2c: Rank and filter papers by citation quality
            papers = self._rank_and_filter_papers(papers, topic=topic)
            logger.info("[%s] After ranking/filtering: %d papers", self.stage.value, len(papers))

            # Step 2c2: Enrich papers from web/PwC with citation counts
            zero_cite = [p for p in papers if (p.get("citation_count", 0) or 0) == 0]
            if zero_cite:
                self.log(f"Enriching citation counts for {len(zero_cite)} papers")
                await self._enrich_citation_counts(zero_cite)
                papers = self._rank_and_filter_papers(papers, topic=topic)

            # Step 2c3: Citation graph expansion (snowball sampling)
            papers = await self._expand_via_citations(papers, top_k=5, max_new=15)
            logger.info("[%s] After citation expansion: %d papers", self.stage.value, len(papers))

            # Step 2d: Enrich top papers with full-text PDF reading
            papers = await self._enrich_with_full_text(papers)

            # Step 2d2: Search coverage self-evaluation (max 2 rounds)
            all_papers_dict = {self._dedup_key(p): p for p in papers}
            for _eval_round in range(2):
                coverage = await self._evaluate_search_coverage(topic, papers)
                score = coverage.get("coverage_score", 10)
                if score >= 8:
                    self.log(f"Search coverage: {score}/10 -- sufficient")
                    break
                missing = coverage.get("missing_directions", [])
                if not missing:
                    break
                self.log(f"Search coverage: {score}/10 -- supplementing {len(missing)} directions")
                new_papers = await self._supplementary_search(missing, all_papers_dict)
                if new_papers:
                    papers.extend(new_papers)
                    for np in new_papers:
                        all_papers_dict[self._dedup_key(np)] = np
                    papers = self._rank_and_filter_papers(papers, topic=topic)
                    self.log(f"Added {len(new_papers)} papers from supplementary search")

            # Step 2e: Extract must-cite papers from surveys
            must_cites = await self._extract_must_cites(
                [p for p in papers if "survey" in (p.get("title", "") or "").lower()
                 or "review" in (p.get("title", "") or "").lower()]
            )
            if must_cites:
                logger.info("[%s] Identified %d must-cite papers",
                            self.stage.value, len(must_cites))
            else:
                must_cites = []

            # Cache search results for retry (including must_cites)
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(
                    json.dumps({"queries": queries, "papers": papers,
                                "must_cites": must_cites},
                               ensure_ascii=False, default=str),
                    encoding="utf-8",
                )
                self.log("Cached search results for potential retry")
            except Exception as e:
                logger.warning("Failed to cache search results: %s", e)

        # Step 3: LLM analysis -- gaps + hypotheses (with ReAct tool use)
        output = await self._analyze_and_hypothesize(topic, queries, papers)

        # Store must-cite titles and match to actual papers
        output.must_cites = must_cites
        if must_cites:
            mc_matches = self._match_must_cites_to_papers(must_cites, papers)
            output.must_cite_matches = mc_matches
            matched_count = sum(1 for m in mc_matches if m.get("matched"))
            self.log(f"Must-cite matching: {matched_count}/{len(must_cites)} matched to papers")

        # Step 4: Extract quantitative evidence from paper abstracts
        evidence = await self._extract_evidence(papers)
        output.evidence = evidence
        logger.info("[%s] Extracted %d metrics from literature",
                    self.stage.value, len(evidence.extracted_metrics))

        # Step 5: Search GitHub for reference implementations
        reference_repos = await self._search_github_repos(topic, queries)
        logger.info("[%s] Found %d reference GitHub repos",
                    self.stage.value, len(reference_repos))
        output.reference_repos = reference_repos

        # Save output
        output_path = self.workspace.write_json(
            "papers/ideation_output.json",
            output.model_dump(mode="json"),
        )
        self.workspace.register_artifact(
            "ideation_output", output_path, self.stage
        )
        return output.model_dump(mode="json")

    async def _generate_queries(self, topic: str) -> list[str]:
        prompt = f"""Given the research topic: "{topic}"

Generate {MAX_SEARCH_QUERIES} diverse search queries to find relevant academic papers.
Include queries for:
- Direct topic matches
- Related methods and techniques
- Benchmark datasets and evaluation approaches
- Recent surveys or reviews

Return JSON: {{"queries": ["query1", "query2", ...]}}"""

        try:
            result = await self.generate_json(IDEATION_SYSTEM_PROMPT, prompt)
            queries = result.get("queries", [])
            if queries:
                return queries
        except Exception as e:
            logger.warning("[%s] Query generation LLM call failed: %s", self.stage.value, e)
        # Fallback: use topic itself as a search query
        self.log("Using fallback queries derived from topic")
        return [topic, f"{topic} survey", f"{topic} benchmark"]

    def _dedup_key(self, paper: dict) -> str:
        """Return a deduplication key for a paper (prefer ID, fallback to title)."""
        for id_field in ("paper_id", "arxiv_id"):
            pid = (paper.get(id_field, "") or "").strip()
            if pid and pid != "unknown":
                return f"id:{pid}"
        return "title:" + (paper.get("title", "") or "").lower().strip()
