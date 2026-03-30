"""Web search tool using duckduckgo_search library.

The original HTML scraping approach was blocked by DuckDuckGo's CAPTCHA.
This uses the `duckduckgo_search` (ddgs) package which works reliably.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import warnings
from typing import Any

from mcp_server.utils import RateLimiter

# The duckduckgo_search package bypasses warnings.filterwarnings() when emitting
# its rename RuntimeWarning, so filterwarnings("ignore") alone is not enough.
# We wrap showwarning to silently drop that specific warning while letting all
# others through.  This is process-global and thread-safe (reads are atomic).
_orig_showwarning = warnings.showwarning


def _filtered_showwarning(message, category, filename, lineno, file=None, line=None):
    msg_str = str(message)
    if category is RuntimeWarning and "renamed" in msg_str and "ddgs" in msg_str:
        return  # suppress duckduckgo_search rename warning
    _orig_showwarning(message, category, filename, lineno, file, line)


warnings.showwarning = _filtered_showwarning


@contextlib.contextmanager
def _suppress_stderr():
    """Temporarily suppress stderr at OS fd level (catches C extension output)."""
    saved_fd = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(saved_fd)

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
        with _suppress_stderr():
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS
    except ImportError:
        logger.warning("duckduckgo_search / ddgs not installed. Run: pip install ddgs")
        return []

    def _sync_search() -> list[dict[str, Any]]:
        try:
            with _suppress_stderr():
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
