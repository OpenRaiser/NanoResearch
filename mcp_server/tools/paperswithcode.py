"""Papers With Code search tool.

The original PapersWithCode API (paperswithcode.com/api/v1) now redirects
to HuggingFace and is effectively defunct. This module uses web search
as a fallback to find PwC task/benchmark pages.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from mcp_server.utils import RateLimiter, get_http_client

logger = logging.getLogger(__name__)

_limiter = RateLimiter(calls_per_second=2.0)


async def search_tasks(query: str, max_results: int = 10) -> list[dict[str, Any]]:
    """Search for ML tasks/benchmarks via web search fallback.

    The PaperswithCode API redirects to HuggingFace (defunct), so we use
    DuckDuckGo to search paperswithcode.com directly.

    Args:
        query: Search query for tasks.
        max_results: Maximum number of results.

    Returns:
        List of dicts with keys: name, url, description.
    """
    try:
        from mcp_server.tools.web_search import search_web
    except ImportError:
        logger.warning("Web search not available for PwC fallback")
        return []

    try:
        results = await search_web(
            f"paperswithcode.com {query} benchmark SOTA",
            max_results=max_results * 2,  # over-fetch since we filter
        )
    except Exception as e:
        logger.warning("PwC web search failed: %s", e)
        return []

    tasks: list[dict[str, Any]] = []
    for r in results:
        url = r.get("url", "")
        # Filter to actual PwC pages
        if "paperswithcode.com" not in url:
            continue
        tasks.append({
            "name": r.get("title", ""),
            "url": url,
            "description": r.get("snippet", ""),
        })

    return tasks[:max_results]


async def get_sota(
    task_id: str, dataset: str | None = None, max_results: int = 20
) -> list[dict[str, Any]]:
    """Get SOTA leaderboard results for a given task.

    NOTE: The PapersWithCode API is defunct. This now returns an empty list
    with a warning. Use Semantic Scholar or web search for SOTA info instead.
    """
    logger.warning(
        "PapersWithCode SOTA API is defunct (redirects to HuggingFace). "
        "Use Semantic Scholar or web search for leaderboard data."
    )
    return []
