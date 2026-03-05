"""Papers With Code public API tool (no API key required)."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from mcp_server.utils import RateLimiter, fetch_with_retry, get_http_client

logger = logging.getLogger(__name__)

_limiter = RateLimiter(calls_per_second=5.0)

PWC_API_BASE = "https://paperswithcode.com/api/v1"


async def search_tasks(query: str, max_results: int = 10) -> list[dict[str, Any]]:
    """Search Papers With Code for ML tasks/benchmarks.

    Args:
        query: Search query for tasks (e.g. "image classification").
        max_results: Maximum number of task results.

    Returns:
        List of dicts with keys: id, name, description, url.
    """
    await _limiter.acquire()

    params = {"q": query, "page_size": min(max_results, 50)}

    try:
        async with get_http_client(timeout=15.0) as client:
            resp = await fetch_with_retry(client.get, f"{PWC_API_BASE}/tasks/", params=params)
            resp.raise_for_status()
            data = resp.json()
    except httpx.TimeoutException:
        logger.warning("PWC API timed out for query: %s", query[:100])
        return []
    except httpx.HTTPStatusError as exc:
        logger.warning("PWC API returned HTTP %d for query: %s", exc.response.status_code, query[:100])
        return []
    except httpx.HTTPError as exc:
        logger.warning("PWC API network error: %s", exc)
        return []

    results = []
    for item in data.get("results", [])[:max_results]:
        results.append({
            "id": item.get("id", ""),
            "name": item.get("name", ""),
            "description": item.get("description", ""),
            "url": f"https://paperswithcode.com/task/{item.get('id', '')}",
        })
    return results


async def get_sota(
    task_id: str, dataset: str | None = None, max_results: int = 20
) -> list[dict[str, Any]]:
    """Get SOTA leaderboard results for a given task.

    Args:
        task_id: Papers With Code task identifier (e.g. "image-classification").
        dataset: Optional dataset name filter.
        max_results: Maximum number of SOTA entries.

    Returns:
        List of dicts with keys: method, paper_title, paper_url, dataset, metric, value, rank.
    """
    await _limiter.acquire()

    # First get datasets for this task
    url = f"{PWC_API_BASE}/tasks/{task_id}/datasets/"
    params: dict[str, Any] = {"page_size": 10}

    try:
        async with get_http_client(timeout=15.0) as client:
            resp = await fetch_with_retry(client.get, url, params=params)
            resp.raise_for_status()
            datasets_data = resp.json()
    except httpx.TimeoutException:
        logger.warning("PWC API timed out for task: %s", task_id)
        return []
    except httpx.HTTPStatusError as exc:
        logger.warning("PWC API returned HTTP %d for task: %s", exc.response.status_code, task_id)
        return []
    except httpx.HTTPError as exc:
        logger.warning("PWC API network error for task %s: %s", task_id, exc)
        return []

    datasets = datasets_data.get("results", [])
    if dataset:
        datasets = [d for d in datasets if dataset.lower() in d.get("name", "").lower()]

    all_results: list[dict[str, Any]] = []
    async with get_http_client(timeout=15.0) as client:
        for ds in datasets[:3]:  # Limit to top 3 datasets
            ds_id = ds.get("id", "")
            ds_name = ds.get("name", "")
            if not ds_id:
                continue

            await _limiter.acquire()
            sota_url = f"{PWC_API_BASE}/datasets/{ds_id}/sota/"
            try:
                resp = await client.get(sota_url)
                resp.raise_for_status()
                sota_data = resp.json()
            except Exception as e:
                logger.warning("Failed to fetch SOTA for dataset %s: %s", ds_id, e)
                continue

            rows = sota_data.get("rows", [])
            for rank, row in enumerate(rows[:max_results], start=1):
                all_results.append({
                    "method": row.get("method", ""),
                    "paper_title": row.get("paper_title", ""),
                    "paper_url": row.get("paper_url", ""),
                    "dataset": ds_name,
                    "metrics": row.get("metrics", {}),
                    "rank": rank,
                })

            if len(all_results) >= max_results:
                break

    return all_results[:max_results]
