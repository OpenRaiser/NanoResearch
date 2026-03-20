"""GitHub repository search tool — find reference implementations."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from mcp_server.utils import RateLimiter, fetch_with_retry, get_http_client

logger = logging.getLogger(__name__)

_limiter = RateLimiter(calls_per_second=5.0)

GITHUB_API = "https://api.github.com"


async def search_repos(
    query: str,
    max_results: int = 5,
    language: str = "Python",
    sort: str = "stars",
) -> list[dict[str, Any]]:
    """Search GitHub for repositories matching the query.

    Args:
        query: Search query (supports GitHub search syntax).
        max_results: Maximum repos to return (capped at 10).
        language: Filter by programming language.
        sort: Sort by 'stars', 'forks', 'updated', or 'best-match'.

    Returns:
        List of repo metadata dicts with structure and README excerpts.
    """
    await _limiter.acquire()

    q = f"{query} language:{language}" if language else query
    params = {
        "q": q,
        "sort": sort,
        "order": "desc",
        "per_page": min(max_results, 10),
    }

    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "NanoResearch-Agent/1.0",
    }

    repos = []
    try:
        async with get_http_client(timeout=15.0) as client:
            resp = await fetch_with_retry(
                client.get, f"{GITHUB_API}/search/repositories",
                params=params, headers=headers,
            )
            if resp.status_code == 403:
                logger.warning("GitHub API rate limit hit for query: %s", query[:80])
                return []
            resp.raise_for_status()
            data = resp.json()

            for item in data.get("items", [])[:max_results]:
                repo = {
                    "full_name": item.get("full_name", ""),
                    "description": item.get("description", "") or "",
                    "url": item.get("html_url", ""),
                    "stars": item.get("stargazers_count", 0),
                    "language": item.get("language", ""),
                    "topics": item.get("topics", []),
                    "default_branch": item.get("default_branch", "main"),
                    "updated_at": item.get("updated_at", ""),
                }

                # Fetch file tree (lightweight, just top-level + src/)
                tree = await _fetch_file_tree(
                    client, headers, repo["full_name"], repo["default_branch"]
                )
                repo["file_tree"] = tree

                # Fetch README excerpt
                readme = await _fetch_readme(client, headers, repo["full_name"])
                repo["readme_excerpt"] = readme[:2000] if readme else ""

                repos.append(repo)
    except httpx.TimeoutException:
        logger.warning("GitHub API timed out for query: %s", query[:80])
        return []
    except httpx.HTTPStatusError as exc:
        logger.warning("GitHub API returned HTTP %d for query: %s", exc.response.status_code, query[:80])
        return []
    except httpx.HTTPError as exc:
        logger.warning("GitHub API network error: %s", exc)
        return []

    return repos


async def _fetch_file_tree(
    client, headers: dict, full_name: str, branch: str,
) -> list[str]:
    """Fetch the repository file tree (first 100 files)."""
    await _limiter.acquire()
    try:
        resp = await client.get(
            f"{GITHUB_API}/repos/{full_name}/git/trees/{branch}",
            params={"recursive": "1"},
            headers=headers,
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
        # Return only Python files and key config files, limit to 100
        paths = []
        for item in data.get("tree", []):
            p = item.get("path", "")
            if item.get("type") != "blob":
                continue
            if p.endswith((".py", ".yaml", ".yml", ".toml", ".cfg", ".sh", ".md")):
                paths.append(p)
            if len(paths) >= 100:
                break
        return paths
    except Exception as e:
        logger.debug("Failed to fetch tree for %s: %s", full_name, e)
        return []


async def _fetch_readme(client, headers: dict, full_name: str) -> str:
    """Fetch repository README content."""
    await _limiter.acquire()
    try:
        resp = await client.get(
            f"{GITHUB_API}/repos/{full_name}/readme",
            headers={**headers, "Accept": "application/vnd.github.v3.raw"},
        )
        if resp.status_code != 200:
            return ""
        return resp.text
    except Exception as e:
        logger.debug("Failed to fetch README for %s: %s", full_name, e)
        return ""
