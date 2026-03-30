"""Ideation search mixin -- literature search, filtering, citation expansion."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Import constants and lazy getters from the main ideation module.
# We import at function call time to avoid circular imports, but the
# module-level constants are safe.
from nanoresearch.agents.ideation import (
    MAX_SEARCH_QUERIES,
    MAX_RESULTS_PER_SEARCH,
    MAX_PAPERS_FOR_ANALYSIS,
    MAX_ABSTRACT_LENGTH,
    HIGH_CITATION_THRESHOLD,
    TARGET_CITATION_COUNT,
    MIN_HIGH_CITED_PAPERS,
    TOP_K_FULL_TEXT,
    SEARCH_COVERAGE_SYSTEM_PROMPT,
    IDEATION_MUST_CITE_SYSTEM,
    _get_arxiv_search,
    _get_s2_search,
    _get_oa_search,
)


class _IdeationSearchMixin:
    """Mixin with literature search, filtering, and citation-expansion methods."""

    def __init_search_warnings(self) -> None:
        if not hasattr(self, "_search_warnings"):
            self._search_warnings: list[str] = []

    # ── safe wrappers for parallel search ──

    async def _search_oa_safe(self, search_oa, query: str, max_results: int = MAX_RESULTS_PER_SEARCH) -> list[dict]:
        self.__init_search_warnings()
        try:
            results = await search_oa(query, max_results=max_results)
            if results:
                logger.debug("[%s] OpenAlex returned %d results for '%s'",
                             self.stage.value, len(results), query[:60])
            return results or []
        except Exception as e:
            self._search_warnings.append(f"OpenAlex failed for '{query[:60]}': {e}")
            logger.warning("[%s] OpenAlex search failed for '%s': %s", self.stage.value, query, e)
            return []

    async def _search_arxiv_safe(self, search_arxiv, query: str) -> list[dict]:
        self.__init_search_warnings()
        try:
            results = await search_arxiv(
                query, max_results=MAX_RESULTS_PER_SEARCH,
                categories=["cs.LG", "cs.AI", "cs.CV", "cs.CL",
                            "q-bio.BM", "q-bio.QM", "physics.chem-ph",
                            "cond-mat.mtrl-sci", "stat.ML"],
            )
            return results or []
        except Exception as e:
            self._search_warnings.append(f"arXiv failed for '{query[:60]}': {e}")
            logger.warning("[%s] arXiv search failed for '%s': %s", self.stage.value, query, e)
            return []

    async def _search_web_safe(self, queries: list[str]) -> list[dict]:
        """Web search for academic papers — safe wrapper."""
        papers: list[dict] = []
        try:
            from mcp_server.tools.web_search import search_web
            for query in queries:
                web_results = await search_web(f"academic paper {query}", max_results=5)
                for wr in web_results:
                    title = wr.get("title", "").strip()
                    url_lower = wr.get("url", "").lower()
                    is_academic = any(
                        domain in url_lower
                        for domain in ("arxiv", "semanticscholar", "acl", "openreview",
                                       "neurips", "icml", "iclr", "aaai", "ieee", "acm")
                    )
                    if is_academic and title:
                        papers.append({
                            "title": title,
                            "url": wr.get("url", ""),
                            "abstract": wr.get("snippet", ""),
                            "authors": [],
                            "year": None,
                            "citation_count": 0,
                        })
        except Exception as e:
            self.__init_search_warnings()
            self._search_warnings.append(f"Web search failed: {e}")
            logger.info("[%s] Web search supplementation skipped: %s", self.stage.value, e)
        return papers

    async def _search_pwc_safe(self, queries: list[str]) -> list[dict]:
        """PapersWithCode search — safe wrapper."""
        papers: list[dict] = []
        try:
            from mcp_server.tools.paperswithcode import search_tasks
            for query in queries:
                pwc_tasks = await search_tasks(query)
                for task in pwc_tasks[:3]:
                    task_name = task.get("name", "")
                    if not task_name:
                        continue
                    logger.info("[%s] Found PwC task: %s", self.stage.value, task_name)
                    for paper in task.get("papers", [])[:3]:
                        title = (paper.get("title", "") or "").strip()
                        if not title:
                            continue
                        papers.append({
                            "title": title,
                            "url": paper.get("url", ""),
                            "abstract": paper.get("abstract", ""),
                            "authors": paper.get("authors", []),
                            "year": paper.get("year"),
                            "citation_count": 0,
                            "source": "paperswithcode",
                        })
        except Exception as e:
            logger.info("[%s] PapersWithCode search skipped: %s", self.stage.value, e)
        return papers

    # ── main search (parallelized) ──

    async def _search_literature(self, queries: list[str]) -> list[dict]:
        all_papers: dict[str, dict] = {}

        if not queries:
            self.log("No search queries available, skipping literature search")
            return []

        search_arxiv = await _get_arxiv_search()
        search_oa = await _get_oa_search()

        # Phase 1: All academic searches concurrently
        # Rate limiters inside each search function handle throttling.
        academic_tasks = []
        valid_queries = [q for q in queries[:MAX_SEARCH_QUERIES] if q and q.strip()]
        for query in valid_queries:
            if search_oa:
                academic_tasks.append(self._search_oa_safe(search_oa, query))
            academic_tasks.append(self._search_arxiv_safe(search_arxiv, query))

        academic_results = await asyncio.gather(*academic_tasks, return_exceptions=True)
        success_count = 0
        for result in academic_results:
            if isinstance(result, list):
                for p in result:
                    key = self._dedup_key(p)
                    if key and key not in all_papers:
                        all_papers[key] = p
                if result:
                    success_count += 1

        if success_count == 0 and valid_queries:
            logger.warning("[%s] All search queries failed, literature coverage may be poor",
                           self.stage.value)

        # Phase 2: Web + PwC in parallel
        web_task = self._search_web_safe(valid_queries[:2])
        pwc_task = self._search_pwc_safe(valid_queries[:2])
        supplementary = await asyncio.gather(web_task, pwc_task, return_exceptions=True)
        for result in supplementary:
            if isinstance(result, list):
                for p in result:
                    key = self._dedup_key(p)
                    if key and key not in all_papers:
                        all_papers[key] = p

        return list(all_papers.values())

    async def _search_surveys(self, topic: str) -> list[dict]:
        survey_queries = [f"survey {topic}", f"review {topic}", f"comprehensive overview {topic}"]
        search_oa = await _get_oa_search()
        if not search_oa:
            return []
        # Run all survey queries in parallel
        tasks = [self._search_oa_safe(search_oa, q, max_results=5) for q in survey_queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        survey_papers: dict[str, dict] = {}
        for result in results:
            if isinstance(result, list):
                for p in result:
                    key = self._dedup_key(p)
                    if key and key not in survey_papers:
                        survey_papers[key] = p
        return list(survey_papers.values())

    @staticmethod
    def _is_proceedings_entry(paper: dict) -> bool:
        title = (paper.get("title") or "").strip()
        if not title:
            return True
        title_lower = title.lower()
        if title_lower.startswith(("proceedings of", "findings of", "advances in")):
            return True
        if len(title) < 15 and any(
            kw in title_lower
            for kw in ("conference", "workshop", "symposium", "journal")
        ):
            return True
        return False

    def _rank_and_filter_papers(self, papers: list[dict], topic: str = "") -> list[dict]:
        import datetime

        original_count = len(papers)
        papers = [p for p in papers if not self._is_proceedings_entry(p)]
        filtered_count = original_count - len(papers)
        if filtered_count > 0:
            logger.info("[%s] Filtered out %d proceedings-level entries",
                        self.stage.value, filtered_count)

        # Relevance filtering: remove papers with < 30% keyword overlap with topic
        if topic:
            relevant = []
            irrelevant_count = 0
            for p in papers:
                score = self._topic_relevance_score(topic, p)
                if score >= 0.30:
                    p["_relevance"] = score
                    relevant.append(p)
                else:
                    irrelevant_count += 1
            if irrelevant_count > 0:
                logger.info(
                    "[%s] Filtered out %d irrelevant papers (relevance < 0.30)",
                    self.stage.value, irrelevant_count,
                )
            papers = relevant

        current_year = datetime.date.today().year
        recent_cutoff = current_year - 2

        recent_papers = []
        other_papers = []
        for p in papers:
            year = p.get("year") or 0
            citations = p.get("citation_count", 0) or 0
            if year >= recent_cutoff and citations < HIGH_CITATION_THRESHOLD:
                recent_papers.append(p)
            else:
                other_papers.append(p)

        other_papers.sort(key=lambda p: p.get("citation_count", 0) or 0, reverse=True)
        recent_papers.sort(
            key=lambda p: (p.get("year", 0) or 0, p.get("citation_count", 0) or 0),
            reverse=True,
        )

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

        recent_slots = min(len(recent_papers), TARGET_CITATION_COUNT // 5)
        other_slots = TARGET_CITATION_COUNT - recent_slots
        return other_papers[:other_slots] + recent_papers[:recent_slots]

    @staticmethod
    def _topic_tokens(topic: str) -> set[str]:
        return {
            tok for tok in re.findall(r"[a-z0-9]+", (topic or "").lower())
            if len(tok) >= 3 and tok not in {"for", "and", "the", "with", "using"}
        }

    def _topic_relevance_score(self, topic: str, paper: dict) -> float:
        topic_tokens = self._topic_tokens(topic)
        if not topic_tokens:
            return 0.0
        text = " ".join([
            str(paper.get("title", "") or ""),
            str(paper.get("abstract", "") or ""),
        ]).lower()
        if not text.strip():
            return 0.0
        paper_tokens = set(re.findall(r"[a-z0-9]+", text))
        if not paper_tokens:
            return 0.0
        overlap = topic_tokens & paper_tokens
        return len(overlap) / len(topic_tokens)

    async def _expand_via_citations(self, papers: list[dict], top_k: int = 5, max_new: int = 20) -> list[dict]:
        try:
            from mcp_server.tools.openalex import get_openalex_references
        except ImportError:
            self.log("OpenAlex not available, skipping citation expansion")
            return papers

        self.log(f"Citation expansion via OpenAlex (top {top_k} papers)")
        try:
            new_papers = await get_openalex_references(papers, top_k=top_k, max_new=max_new)
        except Exception as e:
            logger.warning("OpenAlex citation expansion failed: %s", e)
            return papers

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
        need_enrich = [
            p for p in papers
            if (p.get("citation_count", 0) or 0) == 0
            and (p.get("title") or "").strip()
            and len((p.get("title") or "").strip()) >= 10
        ]
        if not need_enrich:
            return papers

        try:
            from mcp_server.tools.openalex import enrich_citation_counts_openalex
            self.log(f"OpenAlex enriching citation counts for {len(need_enrich)} papers")
            await enrich_citation_counts_openalex(need_enrich)
            still_zero = [
                p for p in need_enrich
                if (p.get("citation_count", 0) or 0) == 0
            ]
            if still_zero:
                self.log(f"OpenAlex resolved {len(need_enrich) - len(still_zero)}/{len(need_enrich)}, "
                         f"{len(still_zero)} papers still at citation_count=0")
            else:
                self.log(f"OpenAlex resolved all {len(need_enrich)} papers")
        except ImportError:
            logger.debug("OpenAlex not available for enrichment")
        except Exception as e:
            logger.warning("OpenAlex enrichment failed: %s", e)
        return papers

    async def _enrich_with_full_text(self, papers: list[dict], top_k: int = TOP_K_FULL_TEXT) -> list[dict]:
        try:
            from mcp_server.tools.pdf_reader import download_and_extract
        except ImportError:
            self.log("PDF reader not available, skipping full-text enrichment")
            return papers

        def _has_pdf(p: dict) -> bool:
            if p.get("pdf_url"):
                return True
            url = p.get("url", "")
            return ".pdf" in url or "/pdf/" in url

        candidates = [p for p in papers if _has_pdf(p)]
        candidates = sorted(
            candidates,
            key=lambda p: p.get("citation_count", 0) or 0,
            reverse=True,
        )[:top_k]

        sem = asyncio.Semaphore(3)  # limit concurrent PDF downloads

        async def _download_one(p: dict) -> None:
            pdf_url = p.get("pdf_url", "")
            if not pdf_url:
                url = p.get("url", "")
                if "arxiv.org/abs/" in url:
                    pdf_url = url.replace("/abs/", "/pdf/")
                    if not pdf_url.endswith(".pdf"):
                        pdf_url += ".pdf"
            if not pdf_url:
                return
            async with sem:
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

        await asyncio.gather(*[_download_one(p) for p in candidates])
        return papers

    async def _evaluate_search_coverage(self, topic: str, papers: list[dict]) -> dict:
        paper_summaries = []
        for p in papers[:MAX_PAPERS_FOR_ANALYSIS]:
            title = (p.get("title") or "")[:120]
            abstract_snippet = (p.get("abstract") or "")[:200]
            paper_summaries.append(f"- {title}: {abstract_snippet}")
        papers_text = "\n".join(paper_summaries)

        user_prompt = f"""Topic: {topic}

Collected papers ({len(papers)} total, showing top {min(len(papers), MAX_PAPERS_FOR_ANALYSIS)}):
{papers_text}

Evaluate the search coverage for this topic. Return JSON:
{{
  "coverage_score": <1-10, where 10 is comprehensive>,
  "missing_directions": ["<specific missing research sub-area>", ...],
  "suggested_queries": ["<search query to fill each gap>", ...],
  "well_covered": ["<research direction that is well represented>", ...]
}}"""

        try:
            result = await self.generate_json(SEARCH_COVERAGE_SYSTEM_PROMPT, user_prompt)
            if not isinstance(result, dict):
                return {"coverage_score": 10}
            score = result.get("coverage_score", 10)
            if isinstance(score, (int, float)):
                result["coverage_score"] = max(1, min(10, int(score)))
            else:
                result["coverage_score"] = 10
            return result
        except Exception as e:
            logger.warning("[%s] Search coverage evaluation failed: %s", self.stage.value, e)
            return {"coverage_score": 10}

    async def _supplementary_search(
        self, missing_directions: list[str], existing_papers_dict: dict[str, dict],
    ) -> list[dict]:
        queries = [d.strip() for d in missing_directions[:3] if d and d.strip()]
        if not queries:
            return []
        try:
            raw_papers = await self._search_literature(queries)
        except Exception as e:
            logger.warning("[%s] Supplementary search failed: %s", self.stage.value, e)
            return []
        new_papers = []
        for p in raw_papers:
            key = self._dedup_key(p)
            if key and key not in existing_papers_dict:
                new_papers.append(p)
        return new_papers

    async def _extract_must_cites(self, survey_papers: list[dict]) -> list[str]:
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
                if mc_lower in p_title or p_title in mc_lower:
                    overlap = 1.0
                else:
                    overlap = len(mc_words & p_words) / min(len(mc_words), len(p_words))
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
