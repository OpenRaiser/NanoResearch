"""HTTP client utilities and rate limiter for MCP tools."""

from __future__ import annotations

import asyncio
import logging
import time

import httpx

logger = logging.getLogger(__name__)

# Retry settings for HTTP 429 / 5xx
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 2.0  # seconds
_RETRY_MAX_DELAY = 60.0


class RateLimiter:
    """Simple token-bucket rate limiter."""

    def __init__(self, calls_per_second: float = 5.0) -> None:
        self._interval = 1.0 / calls_per_second
        self._last_call = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._interval - (now - self._last_call)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call = time.monotonic()


def get_http_client(**kwargs) -> httpx.AsyncClient:
    """Create a configured async HTTP client."""
    defaults = {"timeout": 30.0, "follow_redirects": True}
    defaults.update(kwargs)
    return httpx.AsyncClient(**defaults)


async def fetch_with_retry(
    request_fn,
    *args,
    max_retries: int = _MAX_RETRIES,
    **kwargs,
) -> httpx.Response:
    """Make an HTTP request with exponential backoff on 429 and 5xx.

    Args:
        request_fn: An async callable, e.g. ``client.get`` or ``client.post``.
        *args: Positional args forwarded to *request_fn* (typically the URL).
        max_retries: How many times to retry on transient errors.
        **kwargs: Keyword args forwarded to *request_fn* (params, headers, …).

    Returns the response (caller should still call raise_for_status if needed).
    """
    delay = _RETRY_BASE_DELAY
    last_resp: httpx.Response | None = None
    for attempt in range(max_retries + 1):
        resp = await request_fn(*args, **kwargs)
        status = getattr(resp, "status_code", None)
        if isinstance(status, int) and (status == 429 or status >= 500):
            last_resp = resp
            if attempt < max_retries:
                # Respect Retry-After header if present
                retry_after = getattr(resp, "headers", {}).get("Retry-After")
                try:
                    wait = float(retry_after) if retry_after else delay
                except (ValueError, TypeError):
                    wait = delay  # HTTP-date format or invalid — use default backoff
                wait = min(wait, _RETRY_MAX_DELAY)
                logger.debug(
                    "HTTP %d, retrying in %.1fs (attempt %d/%d)",
                    status, wait, attempt + 1, max_retries + 1,
                )
                await asyncio.sleep(wait)
                delay = min(delay * 2, _RETRY_MAX_DELAY)
                continue
        return resp
    return last_resp  # type: ignore[return-value]
