"""arXiv API search tool."""

from __future__ import annotations

import logging
from typing import Any

import httpx
from lxml import etree

from mcp_server.utils import RateLimiter, fetch_with_retry, get_http_client

logger = logging.getLogger(__name__)

# arXiv API rate limit: ~1 request per 3 seconds recommended
_limiter = RateLimiter(calls_per_second=0.3)

ARXIV_API_URL = "http://export.arxiv.org/api/query"


async def search_arxiv(
    query: str,
    max_results: int = 20,
    categories: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Search arXiv for papers matching the query.

    Args:
        query: Search query string (supports arXiv query syntax).
        max_results: Maximum number of results to return.
        categories: Optional list of arXiv categories to filter by
                    (e.g. ["cs.LG", "cs.AI", "q-bio.BM"]).

    Returns:
        List of paper metadata dicts.
    """
    await _limiter.acquire()

    # Build search query with optional category filter
    search_q = f"all:{query}"
    if categories:
        cat_filter = " OR ".join(f"cat:{c}" for c in categories)
        search_q = f"({search_q}) AND ({cat_filter})"

    params = {
        "search_query": search_q,
        "start": 0,
        "max_results": min(max_results, 50),
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    try:
        async with get_http_client() as client:
            resp = await fetch_with_retry(client.get, ARXIV_API_URL, params=params, max_retries=6)
            resp.raise_for_status()
    except httpx.TimeoutException:
        logger.warning("arXiv API timed out for query: %s", query[:100])
        return []
    except httpx.HTTPStatusError as exc:
        logger.warning("arXiv API returned HTTP %d for query: %s", exc.response.status_code, query[:100])
        return []
    except httpx.HTTPError as exc:
        logger.warning("arXiv API network error: %s", exc)
        return []

    return _parse_atom_feed(resp.text)


def _parse_atom_feed(xml_text: str) -> list[dict[str, Any]]:
    """Parse arXiv Atom XML feed into a list of paper dicts."""
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    try:
        root = etree.fromstring(xml_text.encode())
    except etree.XMLSyntaxError as exc:
        logger.warning("arXiv returned malformed XML: %s", exc)
        return []
    papers = []
    for entry in root.findall("atom:entry", ns):
        # Extract arXiv ID from the <id> URL
        id_text = entry.findtext("atom:id", "", ns)
        arxiv_id = id_text.split("/abs/")[-1] if "/abs/" in id_text else id_text

        title = entry.findtext("atom:title", "", ns).strip().replace("\n", " ")
        summary = entry.findtext("atom:summary", "", ns).strip()
        published = entry.findtext("atom:published", "", ns)[:10]

        authors = []
        for author_el in entry.findall("atom:author", ns):
            name = author_el.findtext("atom:name", "", ns)
            if name:
                authors.append(name)

        # Extract categories
        categories = [
            cat.get("term", "")
            for cat in entry.findall("atom:category", ns)
        ]

        # Extract PDF link
        pdf_url = ""
        for link in entry.findall("atom:link", ns):
            if link.get("title") == "pdf":
                pdf_url = link.get("href", "")

        year = int(published[:4]) if len(published) >= 4 else None

        papers.append({
            "paper_id": arxiv_id,
            "title": title,
            "authors": authors,
            "year": year,
            "abstract": summary,
            "venue": "arXiv",
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "pdf_url": pdf_url,
            "categories": categories,
            "published": published,
        })
    return papers
