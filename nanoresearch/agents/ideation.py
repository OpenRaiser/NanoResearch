"""Ideation agent — literature search, gap analysis, hypothesis generation."""

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
MAX_PAPERS_FOR_ANALYSIS = 50
MAX_ABSTRACT_LENGTH = 500
MAX_GITHUB_REPOS = 5
MAX_GITHUB_QUERIES = 2

# Phase 4: Citation quality targets
TARGET_CITATION_COUNT = 50
MIN_HIGH_CITED_PAPERS = 10
HIGH_CITATION_THRESHOLD = 100
TOP_K_FULL_TEXT = 8

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


from nanoresearch.skill_prompts import (
    IDEATION_QUERY_SYSTEM,
    IDEATION_ANALYSIS_SYSTEM,
    IDEATION_MUST_CITE_SYSTEM,
    IDEATION_EVIDENCE_SYSTEM,
)

# Legacy alias — some internal methods still reference this.
IDEATION_SYSTEM_PROMPT = IDEATION_QUERY_SYSTEM


class IdeationAgent(BaseResearchAgent):
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
            # Restore must_cites from cache, or re-extract if not cached
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
            papers = self._rank_and_filter_papers(papers)
            logger.info("[%s] After ranking/filtering: %d papers", self.stage.value, len(papers))

            # Step 2c2: Enrich papers from web/PwC with citation counts
            zero_cite = [p for p in papers if (p.get("citation_count", 0) or 0) == 0]
            if zero_cite:
                self.log(f"Enriching citation counts for {len(zero_cite)} papers")
                await self._enrich_citation_counts(zero_cite)
                # Re-rank after enrichment
                papers = self._rank_and_filter_papers(papers)

            # Step 2c3: Citation graph expansion (snowball sampling)
            papers = await self._expand_via_citations(papers, top_k=5, max_new=15)
            logger.info("[%s] After citation expansion: %d papers", self.stage.value, len(papers))

            # Step 2d: Enrich top papers with full-text PDF reading
            papers = await self._enrich_with_full_text(papers)

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

        # Step 3: LLM analysis — gaps + hypotheses (with ReAct tool use)
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
        # Store in output for downstream use by experiment agent
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
        # Try paper IDs first (more reliable than titles)
        for id_field in ("paper_id", "arxiv_id"):
            pid = (paper.get(id_field, "") or "").strip()
            if pid and pid != "unknown":
                return f"id:{pid}"
        # Fallback to normalized title
        return "title:" + (paper.get("title", "") or "").lower().strip()

    async def _search_literature(self, queries: list[str]) -> list[dict]:
        all_papers: dict[str, dict] = {}  # deduplicate by ID or title

        if not queries:
            self.log("No search queries available, skipping literature search")
            return []

        search_arxiv = await _get_arxiv_search()
        search_s2 = await _get_s2_search()
        search_oa = await _get_oa_search()
        success_count = 0

        for query in queries[:MAX_SEARCH_QUERIES]:
            if not query or not query.strip():
                continue

            # --- OpenAlex (primary, large quota) ---
            if search_oa:
                try:
                    oa_results = await search_oa(query, max_results=MAX_RESULTS_PER_SEARCH)
                    for p in oa_results:
                        key = self._dedup_key(p)
                        if key and key not in all_papers:
                            all_papers[key] = p
                    if oa_results:
                        success_count += 1
                        logger.debug("[%s] OpenAlex returned %d results for '%s'",
                                     self.stage.value, len(oa_results), query[:60])
                except Exception as e:
                    logger.warning("[%s] OpenAlex search failed for '%s': %s",
                                   self.stage.value, query, e)

            # --- arXiv (always, good for preprints) ---
            try:
                arxiv_results = await search_arxiv(
                    query, max_results=MAX_RESULTS_PER_SEARCH,
                    categories=["cs.LG", "cs.AI", "cs.CV", "cs.CL",
                                "q-bio.BM", "q-bio.QM", "physics.chem-ph",
                                "cond-mat.mtrl-sci", "stat.ML"],
                )
                for p in arxiv_results:
                    key = self._dedup_key(p)
                    if key and key not in all_papers:
                        all_papers[key] = p
                if arxiv_results:
                    success_count += 1
            except Exception as e:
                logger.warning("[%s] arXiv search failed for '%s': %s",
                               self.stage.value, query, e)

            # --- S2 (supplement, save quota) ---
            try:
                s2_results = await search_s2(query, max_results=MAX_RESULTS_PER_SEARCH)
                for p in s2_results:
                    key = self._dedup_key(p)
                    if key and key not in all_papers:
                        all_papers[key] = p
                if s2_results:
                    success_count += 1
            except Exception as e:
                logger.warning("[%s] S2 search failed for '%s': %s",
                               self.stage.value, query, e)

        if success_count == 0 and queries:
            logger.warning("[%s] All search queries failed, literature coverage may be poor",
                           self.stage.value)

        # Supplement with web search results (convert to paper-like dicts)
        try:
            from mcp_server.tools.web_search import search_web
            for query in queries[:2]:
                web_results = await search_web(f"academic paper {query}", max_results=5)
                for wr in web_results:
                    title = wr.get("title", "").strip()
                    paper_dict = {
                        "title": title,
                        "url": wr.get("url", ""),
                        "abstract": wr.get("snippet", ""),
                        "authors": [],
                        "year": None,
                        "citation_count": 0,
                    }
                    url_lower = wr.get("url", "").lower()
                    is_academic = any(
                        domain in url_lower
                        for domain in ("arxiv", "semanticscholar", "acl", "openreview",
                                       "neurips", "icml", "iclr", "aaai", "ieee", "acm")
                    )
                    key = self._dedup_key(paper_dict)
                    if key and key not in all_papers and is_academic:
                        all_papers[key] = paper_dict
        except Exception as e:
            logger.info("[%s] Web search supplementation skipped: %s", self.stage.value, e)

        # Supplement with Papers With Code SOTA info
        try:
            from mcp_server.tools.paperswithcode import search_tasks
            for query in queries[:2]:
                pwc_tasks = await search_tasks(query)
                for task in pwc_tasks[:3]:
                    task_name = task.get("name", "")
                    if not task_name:
                        continue
                    logger.info("[%s] Found PwC task: %s", self.stage.value, task_name)
                    # Store SOTA papers from PwC tasks
                    for paper in task.get("papers", [])[:3]:
                        title = (paper.get("title", "") or "").strip()
                        if not title:
                            continue
                        paper_dict = {
                            "title": title,
                            "url": paper.get("url", ""),
                            "abstract": paper.get("abstract", ""),
                            "authors": paper.get("authors", []),
                            "year": paper.get("year"),
                            "citation_count": 0,
                            "source": "paperswithcode",
                        }
                        key = self._dedup_key(paper_dict)
                        if key and key not in all_papers:
                            all_papers[key] = paper_dict
        except Exception as e:
            logger.info("[%s] PapersWithCode search skipped: %s", self.stage.value, e)

        return list(all_papers.values())

    async def _search_surveys(self, topic: str) -> list[dict]:
        """Search for survey/review papers on the topic."""
        survey_queries = [f"survey {topic}", f"review {topic}", f"comprehensive overview {topic}"]
        survey_papers: dict[str, dict] = {}

        search_s2 = await _get_s2_search()
        search_oa = await _get_oa_search()

        for q in survey_queries:
            # OpenAlex first (save S2 quota)
            if search_oa:
                try:
                    oa_results = await search_oa(q, max_results=5)
                    for p in oa_results:
                        key = self._dedup_key(p)
                        if key and key not in survey_papers:
                            survey_papers[key] = p
                except Exception as e:
                    logger.warning("[%s] OpenAlex survey search failed for '%s': %s",
                                   self.stage.value, q, e)

            try:
                results = await search_s2(q, max_results=5)
                for p in results:
                    key = self._dedup_key(p)
                    if key and key not in survey_papers:
                        survey_papers[key] = p
            except Exception as e:
                logger.warning("[%s] S2 survey search failed for '%s': %s",
                               self.stage.value, q, e)

        return list(survey_papers.values())

    def _rank_and_filter_papers(self, papers: list[dict]) -> list[dict]:
        """Rank papers by citation count, preserving recent low-citation papers."""
        import datetime

        current_year = datetime.date.today().year
        recent_cutoff = current_year - 2  # papers from last 2 years

        # Separate recent low-citation papers from the rest
        recent_papers = []
        other_papers = []
        for p in papers:
            year = p.get("year") or 0
            citations = p.get("citation_count", 0) or 0
            if year >= recent_cutoff and citations < HIGH_CITATION_THRESHOLD:
                recent_papers.append(p)
            else:
                other_papers.append(p)

        # Sort non-recent by citation count (descending)
        other_papers.sort(key=lambda p: p.get("citation_count", 0) or 0, reverse=True)
        # Sort recent by year then citations
        recent_papers.sort(
            key=lambda p: (p.get("year", 0) or 0, p.get("citation_count", 0) or 0),
            reverse=True,
        )

        # Count high-cited papers
        high_cited = [
            p for p in other_papers
            if (p.get("citation_count", 0) or 0) >= HIGH_CITATION_THRESHOLD
        ]
        logger.info(
            "[%s] Citation ranking: %d total, %d high-cited (>=%d), %d recent (%d+)",
            self.stage.value, len(papers), len(high_cited),
            HIGH_CITATION_THRESHOLD, len(recent_papers), recent_cutoff,
        )

        if len(high_cited) < MIN_HIGH_CITED_PAPERS:
            logger.warning(
                "[%s] Only %d high-cited papers found (target: %d). "
                "Citation quality may be low.",
                self.stage.value, len(high_cited), MIN_HIGH_CITED_PAPERS,
            )

        # Reserve slots for recent papers (up to 20% of target)
        recent_slots = min(len(recent_papers), TARGET_CITATION_COUNT // 5)
        other_slots = TARGET_CITATION_COUNT - recent_slots

        result = other_papers[:other_slots] + recent_papers[:recent_slots]
        return result

    async def _expand_via_citations(self, papers: list[dict], top_k: int = 5, max_new: int = 20) -> list[dict]:
        """Snowball sampling: expand paper set via references of top-cited papers.

        Uses S2 batch API to fetch all top-K papers in ONE request instead of K
        individual calls, dramatically reducing API quota consumption.
        """
        try:
            from mcp_server.tools.semantic_scholar import get_papers_batch
        except ImportError:
            # Fallback to single-paper endpoint
            try:
                from mcp_server.tools.semantic_scholar import get_paper_details
            except ImportError:
                self.log("S2 API not available, skipping citation expansion")
                return papers
            return await self._expand_via_citations_single(papers, top_k, max_new, get_paper_details)

        # Pick top-K papers that have a paper_id
        candidates = [
            p for p in papers
            if (p.get("paper_id") or p.get("arxiv_id", "")).strip()
        ]
        candidates.sort(key=lambda p: p.get("citation_count", 0) or 0, reverse=True)
        candidates = candidates[:top_k]

        if not candidates:
            return papers

        # Build lookup IDs for batch request
        lookup_ids = []
        for p in candidates:
            pid = p.get("paper_id") or ""
            arxiv_id = p.get("arxiv_id") or ""
            lookup_id = pid if pid else f"ArXiv:{arxiv_id}" if arxiv_id else ""
            if lookup_id:
                lookup_ids.append(lookup_id)

        if not lookup_ids:
            return papers

        # ONE batch call instead of N individual calls
        self.log(f"Fetching references for {len(lookup_ids)} papers via batch API")
        try:
            batch_results = await get_papers_batch(
                lookup_ids,
                fields="paperId,title,references.paperId,references.title,references.year",
            )
        except Exception as e:
            logger.warning("S2 batch citation expansion failed: %s", e)
            return papers

        existing_keys = {self._dedup_key(p) for p in papers}
        new_papers: list[dict] = []

        for details in batch_results:
            if not details:
                continue
            for ref in details.get("references", []):
                if isinstance(ref, dict):
                    ref_title = (ref.get("title") or "").strip()
                else:
                    continue
                if not ref_title:
                    continue
                ref_dict = {
                    "paper_id": ref.get("paper_id", "") or ref.get("paperId", ""),
                    "title": ref_title,
                    "year": ref.get("year"),
                    "authors": [],
                    "abstract": "",
                    "citation_count": 0,
                    "source": "citation_expansion",
                }
                key = self._dedup_key(ref_dict)
                if key and key not in existing_keys:
                    new_papers.append(ref_dict)
                    existing_keys.add(key)
                if len(new_papers) >= max_new * 3:
                    break

        if not new_papers:
            self.log("Citation expansion found no new papers")
            return papers

        # Enrich new papers with citation counts via batch API
        enriched = await self._enrich_citation_counts(new_papers)

        # Keep only papers with decent citation counts
        enriched = [p for p in enriched if (p.get("citation_count", 0) or 0) >= 20]
        enriched.sort(key=lambda p: p.get("citation_count", 0) or 0, reverse=True)
        enriched = enriched[:max_new]

        if enriched:
            self.log(f"Citation expansion added {len(enriched)} papers from reference graphs")
            papers.extend(enriched)

        return papers

    async def _expand_via_citations_single(self, papers, top_k, max_new, get_paper_details):
        """Fallback: expand citations one-by-one if batch API unavailable."""
        candidates = [
            p for p in papers
            if (p.get("paper_id") or p.get("arxiv_id", "")).strip()
        ]
        candidates.sort(key=lambda p: p.get("citation_count", 0) or 0, reverse=True)
        candidates = candidates[:top_k]
        if not candidates:
            return papers

        existing_keys = {self._dedup_key(p) for p in papers}
        new_papers: list[dict] = []

        for p in candidates:
            pid = p.get("paper_id") or ""
            arxiv_id = p.get("arxiv_id") or ""
            lookup_id = pid if pid else f"ArXiv:{arxiv_id}" if arxiv_id else ""
            if not lookup_id:
                continue
            try:
                details = await get_paper_details(lookup_id)
            except Exception as e:
                logger.debug("Citation expansion failed for %s: %s", lookup_id, e)
                continue
            for ref in details.get("references", []):
                if not isinstance(ref, dict):
                    continue
                ref_title = (ref.get("title") or "").strip()
                if not ref_title:
                    continue
                ref_dict = {
                    "paper_id": ref.get("paper_id", ""),
                    "title": ref_title,
                    "year": ref.get("year"),
                    "authors": [], "abstract": "", "citation_count": 0,
                    "source": "citation_expansion",
                }
                key = self._dedup_key(ref_dict)
                if key and key not in existing_keys:
                    new_papers.append(ref_dict)
                    existing_keys.add(key)
            if len(new_papers) >= max_new * 3:
                break

        if not new_papers:
            self.log("Citation expansion found no new papers")
            return papers
        enriched = await self._enrich_citation_counts(new_papers)
        enriched = [p for p in enriched if (p.get("citation_count", 0) or 0) >= 20]
        enriched.sort(key=lambda p: p.get("citation_count", 0) or 0, reverse=True)
        enriched = enriched[:max_new]
        if enriched:
            self.log(f"Citation expansion added {len(enriched)} papers from reference graphs")
            papers.extend(enriched)
        return papers

    async def _enrich_citation_counts(self, papers: list[dict]) -> list[dict]:
        """Enrich papers that have citation_count=0.

        Strategy (minimizes S2 API calls):
        0. OpenAlex title match first (free, large quota) — handles most papers
        1. Remaining papers WITH paper_id → S2 batch API (1 call for up to 500)
        2. Remaining papers WITHOUT paper_id → S2 title match API (1 call each)
        """
        need_enrich = [
            p for p in papers
            if (p.get("citation_count", 0) or 0) == 0
            and (p.get("title") or "").strip()
            and len((p.get("title") or "").strip()) >= 10
        ]
        if not need_enrich:
            return papers

        # ── Strategy 0: OpenAlex title match (save S2 quota) ──
        try:
            from mcp_server.tools.openalex import enrich_citation_counts_openalex
            self.log(f"OpenAlex enriching citation counts for {len(need_enrich)} papers")
            await enrich_citation_counts_openalex(need_enrich)
            # Recompute need_enrich: only papers still at 0
            still_zero = [
                p for p in need_enrich
                if (p.get("citation_count", 0) or 0) == 0
            ]
            if still_zero:
                self.log(f"OpenAlex resolved {len(need_enrich) - len(still_zero)}/{len(need_enrich)}, "
                         f"{len(still_zero)} remain for S2")
            else:
                self.log(f"OpenAlex resolved all {len(need_enrich)} papers")
                return papers
            need_enrich = still_zero
        except ImportError:
            logger.debug("OpenAlex not available for enrichment, falling back to S2")
        except Exception as e:
            logger.warning("OpenAlex enrichment failed: %s, falling back to S2", e)

        # ── Strategy 1: S2 batch lookup for papers with paper_id ──
        with_id = [p for p in need_enrich if (p.get("paper_id") or "").strip()]
        without_id = [p for p in need_enrich if not (p.get("paper_id") or "").strip()]

        if with_id:
            try:
                from mcp_server.tools.semantic_scholar import get_papers_batch
                ids = [p["paper_id"] for p in with_id]
                self.log(f"S2 batch enriching {len(ids)} papers with paper_id")
                results = await get_papers_batch(
                    ids,
                    fields="paperId,title,authors,year,abstract,venue,citationCount,url,externalIds",
                )
                for p, r in zip(with_id, results):
                    if r and r.get("citation_count", 0):
                        p["citation_count"] = r.get("citation_count", 0) or 0
                        p["abstract"] = p.get("abstract") or r.get("abstract", "")
                        p["authors"] = p.get("authors") or r.get("authors", [])
                        p["venue"] = p.get("venue") or r.get("venue", "")
                        p["url"] = p.get("url") or r.get("url", "")
            except ImportError:
                without_id.extend(with_id)
            except Exception as e:
                logger.warning("S2 batch enrichment failed: %s", e)
                without_id.extend(with_id)

        # ── Strategy 2: S2 title match for papers without paper_id ──
        if without_id:
            try:
                from mcp_server.tools.semantic_scholar import search_paper_by_title
                self.log(f"S2 title-matching {len(without_id)} papers without paper_id")
                for p in without_id:
                    title = (p.get("title") or "").strip()
                    try:
                        r = await search_paper_by_title(title)
                        if r and (r.get("citation_count", 0) or 0) > 0:
                            t_words = set(title.lower().split())
                            r_words = set((r.get("title") or "").lower().split())
                            if t_words and r_words:
                                overlap = len(t_words & r_words) / max(len(t_words), len(r_words))
                                if overlap > 0.6:
                                    p["citation_count"] = r.get("citation_count", 0) or 0
                                    p["paper_id"] = p.get("paper_id") or r.get("paper_id", "")
                                    p["abstract"] = p.get("abstract") or r.get("abstract", "")
                                    p["authors"] = p.get("authors") or r.get("authors", [])
                                    p["venue"] = p.get("venue") or r.get("venue", "")
                                    p["url"] = p.get("url") or r.get("url", "")
                    except Exception as e:
                        logger.debug("S2 title match failed for '%s': %s", title[:50], e)
            except ImportError:
                search_s2 = await _get_s2_search()
                for p in without_id:
                    title = (p.get("title") or "").strip()
                    try:
                        results = await search_s2(title, max_results=3)
                        title_lower = title.lower()
                        for r in results:
                            r_title = (r.get("title") or "").lower().strip()
                            t_words = set(title_lower.split())
                            r_words = set(r_title.split())
                            if t_words and r_words:
                                overlap = len(t_words & r_words) / max(len(t_words), len(r_words))
                                if overlap > 0.7:
                                    p["citation_count"] = r.get("citation_count", 0) or 0
                                    p["paper_id"] = p.get("paper_id") or r.get("paper_id", "")
                                    p["abstract"] = p.get("abstract") or r.get("abstract", "")
                                    p["authors"] = p.get("authors") or r.get("authors", [])
                                    p["venue"] = p.get("venue") or r.get("venue", "")
                                    p["url"] = p.get("url") or r.get("url", "")
                                    break
                    except Exception as e:
                        logger.debug("S2 citation enrichment failed for '%s': %s", title[:50], e)

        return papers

    async def _enrich_with_full_text(self, papers: list[dict], top_k: int = TOP_K_FULL_TEXT) -> list[dict]:
        """Download and extract full text from top-K high-cited papers."""
        try:
            from mcp_server.tools.pdf_reader import download_and_extract
        except ImportError:
            self.log("PDF reader not available, skipping full-text enrichment")
            return papers

        # Pick top-K by citation count that have a pdf_url
        def _has_pdf(p: dict) -> bool:
            if p.get("pdf_url"):
                return True
            url = p.get("url", "")
            # Match .pdf, .pdf?..., and versioned arXiv URLs like /pdf/2301.12345v2
            return ".pdf" in url or "/pdf/" in url

        candidates = [p for p in papers if _has_pdf(p)]
        candidates = sorted(
            candidates,
            key=lambda p: p.get("citation_count", 0) or 0,
            reverse=True,
        )[:top_k]

        for p in candidates:
            pdf_url = p.get("pdf_url", "")
            if not pdf_url:
                # Try to derive from arXiv URL
                url = p.get("url", "")
                if "arxiv.org/abs/" in url:
                    pdf_url = url.replace("/abs/", "/pdf/")
                    if not pdf_url.endswith(".pdf"):
                        pdf_url += ".pdf"
            if not pdf_url:
                continue

            try:
                logger.info("[%s] Downloading PDF: %s...",
                            self.stage.value, p.get("title", "Unknown")[:60])
                extraction = await download_and_extract(pdf_url, max_pages=20)
                p["method_text"] = extraction.get("method_text", "")
                p["experiment_text"] = extraction.get("experiment_text", "")
                p["full_text_available"] = True
                logger.info("[%s]   Extracted %d chars",
                            self.stage.value, len(extraction.get("full_text", "")))
            except Exception as e:
                logger.warning("[%s]   PDF extraction failed: %s", self.stage.value, e)

        return papers

    async def _extract_must_cites(self, survey_papers: list[dict]) -> list[str]:
        """Ask LLM to extract must-cite papers from survey abstracts."""
        if not survey_papers:
            return []

        survey_text = ""
        for i, p in enumerate(survey_papers[:5]):
            abstract = (p.get("abstract", "") or "")[:500]
            survey_text += f"[Survey {i+1}] {p.get('title', 'Unknown')}\n{abstract}\n\n"

        prompt = f"""Based on these survey paper abstracts, identify 10-15 papers that are
frequently cited and essential for any research in this area.

{survey_text}

Return JSON: {{"must_cite_titles": ["Paper Title 1", "Paper Title 2", ...]}}"""

        try:
            result = await self.generate_json(IDEATION_MUST_CITE_SYSTEM, prompt)
            return result.get("must_cite_titles", [])
        except Exception as e:
            logger.warning("[%s] Must-cite extraction failed: %s", self.stage.value, e)
            return []

    def _match_must_cites_to_papers(
        self, must_cite_titles: list[str], papers: list[dict]
    ) -> list[dict]:
        """Match must-cite titles to actual paper objects in our collection.

        Returns a list of dicts with 'title', 'paper_index', 'matched' (bool).
        Papers not found in our collection are flagged as unmatched.
        """
        results = []
        for mc_title in must_cite_titles:
            mc_lower = mc_title.lower().strip()
            mc_words = set(mc_lower.split())
            best_match = None
            best_score = 0.0

            for i, p in enumerate(papers):
                p_title = (p.get("title") or "").lower().strip()
                p_words = set(p_title.split())
                if not mc_words or not p_words:
                    continue
                overlap = len(mc_words & p_words) / max(len(mc_words), len(p_words))
                if overlap > best_score:
                    best_score = overlap
                    best_match = i

            if best_match is not None and best_score > 0.5:
                results.append({
                    "title": mc_title,
                    "paper_index": best_match,
                    "matched": True,
                    "match_score": best_score,
                })
            else:
                results.append({
                    "title": mc_title,
                    "paper_index": None,
                    "matched": False,
                    "match_score": best_score,
                })

        return results

    async def _build_search_tools(self) -> ToolRegistry:
        """Build a ToolRegistry with search tools for ReAct."""
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

        # OpenAlex search (large quota, good for citation counts)
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

        # Web search (optional, may not be available)
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

        # Paper details
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
        # Prepare paper summaries for the LLM
        paper_summaries = []
        for i, p in enumerate(papers[:MAX_PAPERS_FOR_ANALYSIS]):
            abstract_text = (p.get('abstract', '') or '')[:300]
            method_text = (p.get('method_text', '') or '')[:3000]
            experiment_text = (p.get('experiment_text', '') or '')[:3000]

            summary = (
                f"[{i+1}] {p.get('title', 'Unknown')} ({p.get('year', '?')})\n"
                f"    Authors: {', '.join(p.get('authors', [])[:3])}\n"
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
   - "feasibility_notes": Practical feasibility — required compute (GPU type, hours),
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

        # Try ReAct tool-use if tools are available
        try:
            tools = await self._build_search_tools()
            if len(tools) > 0:
                react_result = await self.generate_with_tools(
                    IDEATION_ANALYSIS_SYSTEM, prompt, tools, max_tool_rounds=5
                )
                # Try to parse as JSON
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
                    # Fall back to standard generate_json
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

        # Guard: LLM may return a bare list (e.g. list of hypotheses) instead of the
        # expected top-level dict.  Wrap it so downstream .get() calls don't crash.
        if isinstance(result, list):
            logger.warning(
                "[%s] LLM returned a list instead of dict; wrapping as {'hypotheses': ...}",
                self.stage.value,
            )
            result = {"hypotheses": result}

        # Build PaperReference list
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
                    method_text=(p.get("method_text", "") or "")[:3000],
                    experiment_text=(p.get("experiment_text", "") or "")[:3000],
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
                    # LLM sometimes returns a dict instead of string — coerce to string
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

    async def _search_github_repos(
        self, topic: str, queries: list[str]
    ) -> list[dict]:
        """Step 5: Search GitHub for reference implementations."""
        search_repos = await _get_github_search()
        all_repos: dict[str, dict] = {}  # deduplicate by full_name

        # Use first few queries + the topic itself
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

        # Keep top 5 by stars
        repos = sorted(all_repos.values(), key=lambda r: r.get("stars", 0), reverse=True)
        return repos[:MAX_GITHUB_REPOS]

    async def _extract_evidence(self, papers: list[dict]) -> EvidenceBundle:
        """Extract quantitative metrics explicitly stated in paper abstracts."""
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
