"""Web search tool using duckduckgo_search library.

The original HTML scraping approach was blocked by DuckDuckGo's CAPTCHA.
This uses the `duckduckgo_search` (ddgs) package which works reliably.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from mcp_server.utils import RateLimiter

logger = logging.getLogger(__name__)

_limiter = RateLimiter(calls_per_second=1.0)


async def search_web(query: str, max_results: int = 10) -> list[dict[str, Any]]:
    """Search the web via DuckDuckGo.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return.

    Returns:
        List of dicts with keys: title, url, snippet.
    """
    await _limiter.acquire()

    try:
        from duckduckgo_search import DDGS
    except ImportError:
        logger.warning("duckduckgo_search not installed. Run: pip install duckduckgo_search")
        return []

    def _sync_search() -> list[dict[str, Any]]:
        try:
            import warnings
            warnings.filterwarnings("ignore", message=".*duckduckgo_search.*renamed.*")
            ddgs = DDGS()
            # Try default backend first, fall back to 'lite' on empty results
            raw = ddgs.text(query, max_results=max_results)
            if not raw:
                raw = ddgs.text(query, max_results=max_results, backend="lite")
            results: list[dict[str, Any]] = []
            for r in raw:
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", r.get("link", "")),
                    "snippet": r.get("body", r.get("snippet", "")),
                })
            return results
        except Exception as exc:
            logger.warning("DuckDuckGo search failed for '%s': %s", query[:100], exc)
            return []

    # Run synchronous DDGS in a thread to avoid blocking the event loop
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _sync_search)
