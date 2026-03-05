"""Web search tool using DuckDuckGo HTML endpoint (no API key required)."""

from __future__ import annotations

import logging
from typing import Any

import httpx
from lxml.html import fromstring as html_fromstring

from mcp_server.utils import RateLimiter, fetch_with_retry, get_http_client

logger = logging.getLogger(__name__)

_limiter = RateLimiter(calls_per_second=1.0)

DUCKDUCKGO_URL = "https://html.duckduckgo.com/html/"


async def search_web(query: str, max_results: int = 10) -> list[dict[str, Any]]:
    """Search the web via DuckDuckGo HTML endpoint.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return.

    Returns:
        List of dicts with keys: title, url, snippet.
    """
    await _limiter.acquire()

    data = {"q": query, "b": ""}
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; NanoResearch/0.1)",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    try:
        async with get_http_client(timeout=15.0) as client:
            resp = await fetch_with_retry(client.post, DUCKDUCKGO_URL, data=data, headers=headers)
            resp.raise_for_status()
    except httpx.TimeoutException:
        logger.warning("DuckDuckGo search timed out for query: %s", query[:100])
        return []
    except httpx.HTTPStatusError as exc:
        logger.warning("DuckDuckGo returned HTTP %d for query: %s", exc.response.status_code, query[:100])
        return []
    except httpx.HTTPError as exc:
        logger.warning("DuckDuckGo network error: %s", exc)
        return []

    return _parse_ddg_html(resp.text, max_results)


def _parse_ddg_html(html_text: str, max_results: int) -> list[dict[str, Any]]:
    """Parse DuckDuckGo HTML results page."""
    results: list[dict[str, Any]] = []
    try:
        tree = html_fromstring(html_text)
    except Exception:
        logger.warning("Failed to parse DuckDuckGo HTML response")
        return results

    # DuckDuckGo HTML results are in <div class="result"> blocks
    # Each has <a class="result__a"> for title/url and <a class="result__snippet"> for snippet
    for result_div in tree.xpath('//div[contains(@class, "result")]'):
        if len(results) >= max_results:
            break

        # Title and URL
        title_el = result_div.xpath('.//a[contains(@class, "result__a")]')
        if not title_el:
            continue

        title = (title_el[0].text_content() or "").strip()
        url = title_el[0].get("href", "")

        # Skip DuckDuckGo internal links
        if not url or url.startswith("/") or "duckduckgo.com" in url:
            continue

        # Snippet
        snippet_el = result_div.xpath(
            './/a[contains(@class, "result__snippet")]'
        )
        snippet = (snippet_el[0].text_content() or "").strip() if snippet_el else ""

        if title:
            results.append({"title": title, "url": url, "snippet": snippet})

    return results
