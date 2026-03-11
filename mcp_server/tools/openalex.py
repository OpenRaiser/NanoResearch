"""OpenAlex API search and citation tools.

OpenAlex is a free, open catalog of ~250M scholarly works.
API docs: https://docs.openalex.org/

Rate limits (with API key):
  - 10,000 list queries / day
  - 1,000 full-text search queries / day
  - 10 req/s polite pool
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import httpx

from mcp_server.utils import RateLimiter, fetch_with_retry, get_http_client

logger = logging.getLogger(__name__)

# OpenAlex polite pool allows ~10 req/s with API key
_OA_RATE = 5.0  # conservative to avoid 403s
_limiter = RateLimiter(calls_per_second=_OA_RATE)

OA_API_URL = "https://api.openalex.org"


def _get_api_key() -> str:
    return os.environ.get("OPENALEX_API_KEY", "")


def _common_params() -> dict[str, str]:
    """Build common query params (api_key)."""
    params: dict[str, str] = {}
    key = _get_api_key()
    if key:
        params["api_key"] = key
    return params


async def search_openalex(
    query: str, max_results: int = 20
) -> list[dict[str, Any]]:
    """Search OpenAlex for scholarly works.

    Uses the /works endpoint with search parameter.

    Args:
        query: Search query string.
        max_results: Maximum number of results (max 200 per page).

    Returns:
        List of paper metadata dicts in normalized format.
    """
    await _limiter.acquire()
    params = _common_params()
    params["search"] = query
    params["per_page"] = str(min(max_results, 200))
    params["select"] = (
        "id,doi,title,authorships,publication_year,cited_by_count,"
        "primary_location,abstract_inverted_index,type"
    )
    # Sort by relevance (default), then citation count
    params["sort"] = "relevance_score:desc"

    try:
        async with get_http_client() as client:
            resp = await fetch_with_retry(
                client.get, f"{OA_API_URL}/works",
                params=params,
                max_retries=4,
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.TimeoutException:
        logger.warning("OpenAlex timed out for query: %s", query[:100])
        return []
    except httpx.HTTPStatusError as exc:
        logger.warning("OpenAlex HTTP %d for query: %s", exc.response.status_code, query[:100])
        return []
    except httpx.HTTPError as exc:
        logger.warning("OpenAlex network error: %s", exc)
        return []
    except (ValueError, KeyError) as exc:
        logger.warning("OpenAlex invalid JSON: %s", exc)
        return []

    results = []
    for work in data.get("results", []):
        paper = _normalize_work(work)
        if paper.get("title"):
            results.append(paper)
    return results


async def search_openalex_by_title(
    title: str,
) -> dict[str, Any] | None:
    """Find a single paper by title match.

    Uses filter-based search which is more precise than full-text search
    and does NOT count against the 1,000/day full-text quota.

    Args:
        title: The paper title to search for.

    Returns:
        Paper metadata dict, or None if no match found.
    """
    await _limiter.acquire()
    params = _common_params()
    # Use display_name.search for title matching (list query, not full-text)
    params["filter"] = f"display_name.search:{title[:200]}"
    params["per_page"] = "1"
    params["select"] = (
        "id,doi,title,authorships,publication_year,cited_by_count,"
        "primary_location,abstract_inverted_index,type"
    )

    try:
        async with get_http_client() as client:
            resp = await fetch_with_retry(
                client.get, f"{OA_API_URL}/works",
                params=params,
                max_retries=4,
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        logger.warning("OpenAlex title search HTTP %d for: %s", exc.response.status_code, title[:60])
        return None
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("OpenAlex title search failed for '%s': %s", title[:60], exc)
        return None

    results = data.get("results", [])
    if results:
        return _normalize_work(results[0])
    return None


async def get_openalex_papers_batch(
    openalex_ids: list[str],
) -> list[dict[str, Any]]:
    """Fetch details for multiple papers using OpenAlex filter.

    Uses pipe-separated ID filter: /works?filter=openalex:W123|W456|...
    Up to ~50 IDs per request is practical.

    Args:
        openalex_ids: List of OpenAlex work IDs (e.g. "W2741809807").

    Returns:
        List of paper metadata dicts.
    """
    if not openalex_ids:
        return []

    results: list[dict[str, Any]] = []
    # Chunk into groups of 50 to keep URL length reasonable
    for i in range(0, len(openalex_ids), 50):
        chunk = openalex_ids[i:i + 50]
        if i > 0:
            await _limiter.acquire()
        else:
            await _limiter.acquire()

        id_filter = "|".join(chunk)
        params = _common_params()
        params["filter"] = f"openalex:{id_filter}"
        params["per_page"] = str(len(chunk))
        params["select"] = (
            "id,doi,title,authorships,publication_year,cited_by_count,"
            "primary_location,abstract_inverted_index,type"
        )

        try:
            async with get_http_client() as client:
                resp = await fetch_with_retry(
                    client.get, f"{OA_API_URL}/works",
                    params=params,
                    max_retries=4,
                )
                resp.raise_for_status()
                data = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("OpenAlex batch failed for %d IDs: %s", len(chunk), exc)
            continue

        for work in data.get("results", []):
            paper = _normalize_work(work)
            if paper.get("title"):
                results.append(paper)

    return results


async def enrich_citation_counts_openalex(
    papers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Enrich papers with citation counts from OpenAlex via title matching.

    For each paper with citation_count=0, searches OpenAlex by title and
    updates the paper dict in-place if a good match is found.

    Args:
        papers: List of paper dicts to enrich (modified in-place).

    Returns:
        The same list (modified in-place).
    """
    need_enrich = [
        p for p in papers
        if (p.get("citation_count", 0) or 0) == 0
        and (p.get("title") or "").strip()
        and len((p.get("title") or "").strip()) >= 10
    ]
    if not need_enrich:
        return papers

    for p in need_enrich:
        title = (p.get("title") or "").strip()
        try:
            result = await search_openalex_by_title(title)
            if result and (result.get("citation_count", 0) or 0) > 0:
                # Verify title match quality
                t_words = set(title.lower().split())
                r_words = set((result.get("title") or "").lower().split())
                if t_words and r_words:
                    overlap = len(t_words & r_words) / max(len(t_words), len(r_words))
                    if overlap > 0.6:
                        p["citation_count"] = result.get("citation_count", 0) or 0
                        p["abstract"] = p.get("abstract") or result.get("abstract", "")
                        p["authors"] = p.get("authors") or result.get("authors", [])
                        p["venue"] = p.get("venue") or result.get("venue", "")
                        p["url"] = p.get("url") or result.get("url", "")
                        p["openalex_id"] = result.get("openalex_id", "")
        except Exception as e:
            logger.debug("OpenAlex enrichment failed for '%s': %s", title[:50], e)

    return papers


async def get_openalex_references(
    papers: list[dict[str, Any]],
    top_k: int = 5,
    max_new: int = 20,
) -> list[dict[str, Any]]:
    """Snowball sampling via OpenAlex: get referenced works for top-cited papers.

    For each paper, looks up its OpenAlex record to get `referenced_works`
    (list of OpenAlex IDs), then batch-fetches those works.

    Args:
        papers: Papers to expand from.
        top_k: How many top-cited papers to expand.
        max_new: Max new papers to return.

    Returns:
        List of new paper dicts found via reference expansion.
    """
    # Pick top-K papers that have an openalex_id or DOI for lookup
    candidates = []
    for p in papers:
        oa_id = (p.get("openalex_id") or "").strip()
        doi = (p.get("doi") or "").strip()
        title = (p.get("title") or "").strip()
        if oa_id or doi or len(title) >= 10:
            candidates.append(p)
    candidates.sort(key=lambda p: p.get("citation_count", 0) or 0, reverse=True)
    candidates = candidates[:top_k]

    if not candidates:
        return []

    # Step 1: Resolve each candidate to an OpenAlex work with referenced_works
    all_ref_ids: list[str] = []
    existing_titles = {(p.get("title") or "").lower().strip() for p in papers}

    for p in candidates:
        oa_id = (p.get("openalex_id") or "").strip()
        doi = (p.get("doi") or "").strip()

        # Build lookup URL
        lookup_id = ""
        if oa_id:
            lookup_id = oa_id
        elif doi:
            lookup_id = f"doi:{doi}"

        ref_ids: list[str] = []
        if lookup_id:
            ref_ids = await _fetch_referenced_works(lookup_id)
        else:
            # Try title search to find the OpenAlex ID
            title = (p.get("title") or "").strip()
            match = await search_openalex_by_title(title)
            if match and match.get("openalex_id"):
                p["openalex_id"] = match["openalex_id"]
                ref_ids = await _fetch_referenced_works(match["openalex_id"])

        all_ref_ids.extend(ref_ids)

    if not all_ref_ids:
        return []

    # Deduplicate reference IDs
    unique_ref_ids = list(dict.fromkeys(all_ref_ids))  # preserve order, dedup

    # Step 2: Batch fetch referenced works
    new_papers = await get_openalex_papers_batch(unique_ref_ids)

    # Filter out papers we already have
    result = []
    for p in new_papers:
        title_lower = (p.get("title") or "").lower().strip()
        if title_lower and title_lower not in existing_titles:
            existing_titles.add(title_lower)
            result.append(p)
        if len(result) >= max_new * 3:  # over-fetch, caller will filter by citation count
            break

    return result


async def _fetch_referenced_works(openalex_id: str) -> list[str]:
    """Fetch the referenced_works list for a single OpenAlex work.

    Returns list of short OpenAlex IDs (e.g. ["W123", "W456"]).
    """
    await _limiter.acquire()
    params = _common_params()
    params["select"] = "id,referenced_works"

    try:
        async with get_http_client() as client:
            resp = await fetch_with_retry(
                client.get, f"{OA_API_URL}/works/{openalex_id}",
                params=params,
                max_retries=4,
            )
            resp.raise_for_status()
            data = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("OpenAlex referenced_works failed for %s: %s", openalex_id, exc)
        return []

    refs = data.get("referenced_works", []) or []
    # Convert full URLs to short IDs: "https://openalex.org/W123" → "W123"
    result = []
    for ref_url in refs:
        if isinstance(ref_url, str) and "/" in ref_url:
            result.append(ref_url.rsplit("/", 1)[-1])
        elif isinstance(ref_url, str):
            result.append(ref_url)
    return result


def _reconstruct_abstract(inverted_index: dict | None) -> str:
    """Reconstruct abstract from OpenAlex inverted index format."""
    if not inverted_index or not isinstance(inverted_index, dict):
        return ""
    # Build word→position mapping and reconstruct
    positions: list[tuple[int, str]] = []
    for word, idxs in inverted_index.items():
        if isinstance(idxs, list):
            for idx in idxs:
                positions.append((idx, word))
    positions.sort(key=lambda x: x[0])
    return " ".join(word for _, word in positions)


def _normalize_work(work: dict) -> dict[str, Any]:
    """Normalize OpenAlex work data to common format (matching S2 format)."""
    # Extract DOI
    doi = (work.get("doi") or "").replace("https://doi.org/", "")

    # Extract arXiv ID from primary_location or DOI
    arxiv_id = ""
    primary = work.get("primary_location") or {}
    source = primary.get("source") or {}
    if source.get("display_name", "").lower() == "arxiv":
        landing_url = primary.get("landing_page_url", "")
        if "arxiv.org/abs/" in landing_url:
            arxiv_id = landing_url.split("arxiv.org/abs/")[-1].split("v")[0]

    # Extract venue
    venue = source.get("display_name", "") or ""

    # Authors
    authors = []
    for auth in (work.get("authorships") or []):
        author_obj = auth.get("author") or {}
        name = author_obj.get("display_name", "")
        if name:
            authors.append(name)

    # Abstract
    abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))

    # URL
    url = ""
    landing = primary.get("landing_page_url", "")
    if landing:
        url = landing
    elif doi:
        url = f"https://doi.org/{doi}"

    # OpenAlex ID (e.g. "https://openalex.org/W2741809807" → "W2741809807")
    oa_id = (work.get("id") or "")
    if "/" in oa_id:
        oa_id = oa_id.rsplit("/", 1)[-1]

    return {
        "paper_id": "",  # No S2 paper_id
        "arxiv_id": arxiv_id,
        "openalex_id": oa_id,
        "doi": doi,
        "title": work.get("title", "") or "",
        "authors": authors,
        "year": work.get("publication_year"),
        "abstract": abstract,
        "venue": venue,
        "citation_count": work.get("cited_by_count", 0) or 0,
        "url": url,
        "source": "openalex",
    }
