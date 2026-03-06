"""Semantic Scholar API search and citation tools."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

import httpx

from mcp_server.utils import RateLimiter, fetch_with_retry, get_http_client

logger = logging.getLogger(__name__)

# S2 free tier: ~100 requests per 5 minutes ≈ 0.33 req/s.
# With an API key the limit is higher, so we check for that.
_S2_RATE_FREE = 0.3   # calls per second without API key
_S2_RATE_KEYED = 3.0  # calls per second with API key
_has_key = bool(os.environ.get("S2_API_KEY", ""))
_limiter = RateLimiter(calls_per_second=_S2_RATE_KEYED if _has_key else _S2_RATE_FREE)

# Global circuit breaker: pause ALL S2 calls after consecutive 429s
_CB_THRESHOLD = 3       # consecutive 429s before tripping
_CB_COOLDOWN = 60.0     # seconds to pause after tripping
_consecutive_429s = 0
_circuit_open_until = 0.0  # monotonic timestamp
_circuit_lock = asyncio.Lock()


async def _circuit_breaker_check() -> bool:
    """Check if circuit breaker is open. Returns True if we should skip the call."""
    global _consecutive_429s, _circuit_open_until
    now = time.monotonic()
    if _circuit_open_until > now:
        wait = _circuit_open_until - now
        logger.info("S2 circuit breaker open, waiting %.0fs", wait)
        await asyncio.sleep(wait)
    return False


async def _circuit_breaker_record(status_code: int) -> None:
    """Record a response status for circuit breaker logic."""
    global _consecutive_429s, _circuit_open_until
    async with _circuit_lock:
        if status_code == 429:
            _consecutive_429s += 1
            if _consecutive_429s >= _CB_THRESHOLD:
                _circuit_open_until = time.monotonic() + _CB_COOLDOWN
                logger.warning(
                    "S2 circuit breaker TRIPPED after %d consecutive 429s, "
                    "pausing all S2 calls for %.0fs",
                    _consecutive_429s, _CB_COOLDOWN,
                )
                _consecutive_429s = 0  # reset after tripping
        else:
            _consecutive_429s = 0  # reset on success

S2_API_URL = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS = "paperId,title,authors,year,abstract,venue,citationCount,url,externalIds,citations,references"


async def search_semantic_scholar(
    query: str, max_results: int = 20
) -> list[dict[str, Any]]:
    """Search Semantic Scholar for papers.

    Args:
        query: Search query string.
        max_results: Maximum number of results.

    Returns:
        List of paper metadata dicts.
    """
    await _circuit_breaker_check()
    await _limiter.acquire()
    headers = {}
    api_key = os.environ.get("S2_API_KEY", "")
    if api_key:
        headers["x-api-key"] = api_key

    params = {
        "query": query,
        "limit": min(max_results, 100),
        "fields": "paperId,title,authors,year,abstract,venue,citationCount,url,externalIds",
    }
    try:
        async with get_http_client() as client:
            resp = await fetch_with_retry(
                client.get, f"{S2_API_URL}/paper/search",
                params=params, headers=headers,
                max_retries=6,
            )
            await _circuit_breaker_record(resp.status_code)
            resp.raise_for_status()
            data = resp.json()
    except httpx.TimeoutException:
        logger.warning("S2 API timed out for query: %s", query[:100])
        return []
    except httpx.HTTPStatusError as exc:
        logger.warning("S2 API returned HTTP %d for query: %s", exc.response.status_code, query[:100])
        return []
    except httpx.HTTPError as exc:
        logger.warning("S2 API network error: %s", exc)
        return []
    except (ValueError, KeyError) as exc:
        logger.warning("S2 API returned invalid JSON: %s", exc)
        return []

    return [_normalize_paper(p) for p in data.get("data", [])]


async def get_paper_details(paper_id: str) -> dict[str, Any]:
    """Get detailed paper info including citations and references.

    Args:
        paper_id: Semantic Scholar paper ID or arXiv:XXXX.XXXXX format.

    Returns:
        Paper metadata with citations and references.
    """
    await _circuit_breaker_check()
    await _limiter.acquire()
    headers = {}
    api_key = os.environ.get("S2_API_KEY", "")
    if api_key:
        headers["x-api-key"] = api_key

    fields = "paperId,title,authors,year,abstract,venue,citationCount,url,externalIds,citations.paperId,citations.title,citations.year,references.paperId,references.title,references.year"
    try:
        async with get_http_client() as client:
            resp = await fetch_with_retry(
                client.get, f"{S2_API_URL}/paper/{paper_id}",
                params={"fields": fields},
                headers=headers,
                max_retries=6,
            )
            await _circuit_breaker_record(resp.status_code)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        logger.warning("S2 paper detail HTTP %d for %s", exc.response.status_code, paper_id)
        return {"paper_id": paper_id, "title": "", "authors": [], "error": str(exc)}
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("S2 paper detail failed for %s: %s", paper_id, exc)
        return {"paper_id": paper_id, "title": "", "authors": [], "error": str(exc)}

    result = _normalize_paper(data)
    result["citations"] = [
        {"paper_id": c.get("paperId", ""), "title": c.get("title", ""), "year": c.get("year")}
        for c in data.get("citations", []) or []
    ]
    result["references"] = [
        {"paper_id": r.get("paperId", ""), "title": r.get("title", ""), "year": r.get("year")}
        for r in data.get("references", []) or []
    ]
    return result


async def get_papers_batch(
    paper_ids: list[str],
    fields: str = "paperId,title,authors,year,abstract,venue,citationCount,url,externalIds",
) -> list[dict[str, Any]]:
    """Fetch details for multiple papers in a single request (up to 500).

    Uses POST /paper/batch — consumes only 1 API call for up to 500 papers.

    Args:
        paper_ids: List of paper IDs (S2 IDs, ArXiv:XXX, DOI:XXX, etc.).
        fields: Comma-separated fields to return.

    Returns:
        List of paper metadata dicts (in same order as input; None for not-found).
    """
    if not paper_ids:
        return []

    await _circuit_breaker_check()
    await _limiter.acquire()
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("S2_API_KEY", "")
    if api_key:
        headers["x-api-key"] = api_key

    # S2 batch limit is 500
    results: list[dict[str, Any]] = []
    for i in range(0, len(paper_ids), 500):
        chunk = paper_ids[i:i + 500]
        if i > 0:
            await _limiter.acquire()
        try:
            async with get_http_client() as client:
                resp = await fetch_with_retry(
                    client.post,
                    f"{S2_API_URL}/paper/batch",
                    params={"fields": fields},
                    json={"ids": chunk},
                    headers=headers,
                    max_retries=6,
                )
                await _circuit_breaker_record(resp.status_code)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            logger.warning("S2 batch HTTP %d for %d papers", exc.response.status_code, len(chunk))
            results.extend([{}] * len(chunk))
            continue
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("S2 batch failed: %s", exc)
            results.extend([{}] * len(chunk))
            continue

        for item in data:
            if item is None:
                results.append({})
            else:
                results.append(_normalize_paper(item))
    return results


async def search_paper_by_title(
    title: str,
    fields: str = "paperId,title,authors,year,abstract,venue,citationCount,url,externalIds",
) -> dict[str, Any] | None:
    """Find a single paper by closest title match.

    Uses GET /paper/search/match — more precise than keyword search for
    known titles, and returns only 1 result (1 API call).

    Args:
        title: The paper title to search for.
        fields: Comma-separated fields to return.

    Returns:
        Paper metadata dict, or None if no match found.
    """
    await _circuit_breaker_check()
    await _limiter.acquire()
    headers = {}
    api_key = os.environ.get("S2_API_KEY", "")
    if api_key:
        headers["x-api-key"] = api_key

    try:
        async with get_http_client() as client:
            resp = await fetch_with_retry(
                client.get,
                f"{S2_API_URL}/paper/search/match",
                params={"query": title[:200], "fields": fields},
                headers=headers,
                max_retries=6,
            )
            await _circuit_breaker_record(resp.status_code)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            return None
        logger.warning("S2 title match HTTP %d for: %s", exc.response.status_code, title[:60])
        return None
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("S2 title match failed for '%s': %s", title[:60], exc)
        return None

    matches = data.get("data", [])
    if matches:
        return _normalize_paper(matches[0])
    return None


def _normalize_paper(p: dict) -> dict[str, Any]:
    """Normalize S2 paper data to a common format."""
    external = p.get("externalIds") or {}
    arxiv_id = external.get("ArXiv", "")
    return {
        "paper_id": p.get("paperId", ""),
        "arxiv_id": arxiv_id,
        "title": p.get("title", ""),
        "authors": [a.get("name", "") for a in (p.get("authors") or [])],
        "year": p.get("year"),
        "abstract": p.get("abstract", "") or "",
        "venue": p.get("venue", "") or "",
        "citation_count": p.get("citationCount", 0) or 0,
        "url": p.get("url", ""),
    }
