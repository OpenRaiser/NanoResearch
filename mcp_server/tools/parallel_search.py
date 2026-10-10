"""Anonymous web search through Parallel's Streamable HTTP MCP server."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

logger = logging.getLogger(__name__)

_ENDPOINT = "https://search.parallel.ai/mcp"
_USER_AGENT = "NanoResearch/0.1.0 (https://github.com/OpenRaiser/NanoResearch)"
_TIMEOUT = 30.0


async def _search(query: str, max_results: int) -> list[dict[str, Any]]:
    async with streamablehttp_client(
        _ENDPOINT,
        headers={"User-Agent": _USER_AGENT},
        timeout=_TIMEOUT,
        sse_read_timeout=_TIMEOUT,
    ) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            response = await session.call_tool(
                "web_search", {"objective": query, "search_queries": [query]},
            )
            if response.isError:
                raise RuntimeError("Parallel MCP returned a tool error")
            payload = response.structuredContent
            if payload is None:
                payload = json.loads("\n".join(
                    block.text for block in response.content if block.type == "text"
                ))
            return [
                {
                    "title": result.get("title") or "",
                    "url": result["url"],
                    "snippet": "\n".join(result.get("excerpts") or []),
                }
                for result in payload["results"][:max_results]
            ]


async def search_parallel(query: str, max_results: int = 10) -> list[dict[str, Any]]:
    """Return title/url/snippet results, or an empty list on service failure.

    No credentials are loaded. Anonymous searches use server-managed Fast mode;
    max_results limits the returned list, not the server's search settings.
    """
    if not query or not query.strip():
        raise ValueError("search_web: 'query' must be a non-empty string")
    if max_results <= 0:
        return []
    try:
        return await asyncio.wait_for(_search(query, max_results), timeout=_TIMEOUT)
    except Exception as exc:
        logger.warning("Parallel search failed for '%s': %s", query[:100], exc)
        return []
