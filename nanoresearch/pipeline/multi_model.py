"""Per-stage model dispatcher — uses OpenAI SDK with a custom base URL.

Supports two image generation backends:
  - "openai": DALL-E via /v1/images/generations
  - "gemini": Gemini native API via /v1beta/models/{model}:generateContent
"""

from __future__ import annotations

import asyncio
import json
import logging
from functools import partial
from typing import Any

import httpx
from openai import OpenAI

from nanoresearch.config import ResearchConfig, StageModelConfig

logger = logging.getLogger(__name__)

# Retry settings for transient API errors
MAX_API_RETRIES = 5
RETRY_BASE_DELAY = 3.0  # seconds
RETRY_BACKOFF = 2.0

# Exceptions worth retrying (strings matched in error message)
_RETRYABLE_PATTERNS = (
    "timeout", "timed out", "rate limit", "429", "502", "503", "504",
    "connection", "server error", "overloaded", "capacity",
)


class ModelDispatcher:
    """Dispatches LLM calls via the OpenAI-compatible API.

    All stages use the same base_url + api_key (your self-hosted endpoint).
    Each stage has its own model name, temperature, max_tokens, and timeout.
    """

    def __init__(self, config: ResearchConfig) -> None:
        self._config = config
        self._clients: dict[float, OpenAI] = {}

    def _get_client(self, timeout: float) -> OpenAI:
        """Get or create an OpenAI client for the given timeout."""
        # Round timeout to avoid float precision creating duplicate clients
        timeout = round(timeout, 1)
        if timeout not in self._clients:
            self._clients[timeout] = OpenAI(
                base_url=self._config.base_url,
                api_key=self._config.api_key,
                timeout=timeout,
            )
        return self._clients[timeout]

    async def close(self) -> None:
        for client in self._clients.values():
            try:
                client.close()
            except Exception as exc:
                logger.debug("Error closing OpenAI client: %s", exc)
        self._clients.clear()

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        """Check if an exception is transient and worth retrying."""
        msg = str(exc).lower()
        return any(pat in msg for pat in _RETRYABLE_PATTERNS)

    async def generate(
        self,
        config: StageModelConfig,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
    ) -> str:
        """Generate a completion using the configured model."""
        timeout = config.timeout or self._config.timeout
        client = self._get_client(timeout)

        _m = config.model.lower()
        is_thinking = "thinking" in _m or _m.startswith("o1") or _m.startswith("o3-")

        kwargs: dict[str, Any] = {
            "model": config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if is_thinking:
            # Thinking models: use max_completion_tokens so thinking budget
            # is separate from visible output (supported by most proxies)
            kwargs["max_completion_tokens"] = config.max_tokens
        else:
            kwargs["max_tokens"] = config.max_tokens
        # Some models (Codex, o-series, thinking) don't support temperature
        if config.temperature is not None and not is_thinking:
            kwargs["temperature"] = config.temperature
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        logger.debug("Calling model=%s temp=%s timeout=%ss", config.model, config.temperature, timeout)

        # Run synchronous OpenAI call in a thread with retry
        loop = asyncio.get_running_loop()
        last_exc: Exception | None = None
        for attempt in range(MAX_API_RETRIES + 1):
            try:
                response = await loop.run_in_executor(
                    None,
                    partial(client.chat.completions.create, **kwargs),
                )
                if not response.choices:
                    raise RuntimeError(
                        f"LLM returned empty choices (model={config.model})"
                    )
                content = response.choices[0].message.content or ""
                logger.debug("Response length: %d chars", len(content))
                return content
            except Exception as exc:
                last_exc = exc
                # Fallback: if proxy rejects max_completion_tokens, switch to max_tokens
                if "max_completion_tokens" in str(exc) and "max_completion_tokens" in kwargs:
                    logger.info("Proxy doesn't support max_completion_tokens, falling back to max_tokens")
                    kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
                    continue
                if attempt < MAX_API_RETRIES and self._is_retryable(exc):
                    delay = RETRY_BASE_DELAY * (RETRY_BACKOFF ** attempt)
                    # Connection errors get extra delay (proxy/network recovery)
                    if "connection" in str(exc).lower():
                        delay = max(delay, 10.0)
                    logger.warning(
                        "LLM call failed (model=%s, attempt %d/%d): %s. Retrying in %.1fs...",
                        config.model, attempt + 1, MAX_API_RETRIES + 1, exc, delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    break

        logger.error("LLM call failed (model=%s): %s", config.model, last_exc)
        raise RuntimeError(
            f"LLM call to model {config.model!r} failed after {MAX_API_RETRIES + 1} attempts: {last_exc}"
        ) from last_exc

    async def generate_with_tools(
        self,
        config: StageModelConfig,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        """Generate a completion with optional tool/function calling.

        Args:
            config: Stage model configuration.
            messages: Full message list (system + user + assistant + tool results).
            tools: OpenAI-format tools list.  Pass ``None`` to do a plain call.

        Returns:
            The raw ``response.choices[0].message`` object so the caller
            can inspect ``tool_calls`` and ``content``.
        """
        timeout = config.timeout or self._config.timeout
        client = self._get_client(timeout)

        _m = config.model.lower()
        is_thinking = "thinking" in _m or _m.startswith("o1") or _m.startswith("o3-")

        kwargs: dict[str, Any] = {
            "model": config.model,
            "messages": messages,
        }
        if is_thinking:
            kwargs["max_completion_tokens"] = config.max_tokens
        else:
            kwargs["max_tokens"] = config.max_tokens
        if config.temperature is not None and not is_thinking:
            kwargs["temperature"] = config.temperature
        if tools:
            kwargs["tools"] = tools

        logger.debug(
            "Calling model=%s with %d messages, %d tools",
            config.model, len(messages), len(tools or []),
        )

        loop = asyncio.get_running_loop()
        last_exc: Exception | None = None
        for attempt in range(MAX_API_RETRIES + 1):
            try:
                response = await loop.run_in_executor(
                    None,
                    partial(client.chat.completions.create, **kwargs),
                )
                if not response.choices:
                    raise RuntimeError(
                        f"LLM returned empty choices (model={config.model})"
                    )
                return response.choices[0].message
            except Exception as exc:
                last_exc = exc
                if "max_completion_tokens" in str(exc) and "max_completion_tokens" in kwargs:
                    logger.info("Proxy doesn't support max_completion_tokens, falling back to max_tokens")
                    kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
                    continue
                if attempt < MAX_API_RETRIES and self._is_retryable(exc):
                    delay = RETRY_BASE_DELAY * (RETRY_BACKOFF ** attempt)
                    if "connection" in str(exc).lower():
                        delay = max(delay, 10.0)
                    logger.warning(
                        "LLM tool-call failed (model=%s, attempt %d/%d): %s. Retrying in %.1fs...",
                        config.model, attempt + 1, MAX_API_RETRIES + 1, exc, delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    break

        logger.error("LLM tool-call failed (model=%s): %s", config.model, last_exc)
        raise RuntimeError(
            f"LLM tool-call to model {config.model!r} failed after {MAX_API_RETRIES + 1} attempts: {last_exc}"
        ) from last_exc

    async def generate_image(
        self,
        config: StageModelConfig,
        prompt: str,
        size: str = "1024x1024",
        quality: str = "hd",
    ) -> list[str]:
        """Generate images — routes to OpenAI images API or Gemini native API.

        Returns a list of base64-encoded image strings.
        """
        if config.image_backend == "gemini":
            return await self._generate_image_gemini(config, prompt)
        return await self._generate_image_openai(config, prompt, size, quality)

    async def _generate_image_openai(
        self,
        config: StageModelConfig,
        prompt: str,
        size: str = "1024x1024",
        quality: str = "hd",
    ) -> list[str]:
        """Generate images via OpenAI /v1/images/generations (DALL-E)."""
        timeout = config.timeout or self._config.timeout
        client = self._get_client(timeout)

        logger.debug("Generating image (openai) model=%s size=%s", config.model, size)

        loop = asyncio.get_running_loop()
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                response = await loop.run_in_executor(
                    None,
                    partial(
                        client.images.generate,
                        model=config.model,
                        prompt=prompt,
                        size=size,
                        quality=quality,
                        n=1,
                        response_format="b64_json",
                    ),
                )
                if not response.data:
                    logger.warning("OpenAI image API returned no images (model=%s)", config.model)
                    return []
                return [img.b64_json for img in response.data if img.b64_json]
            except Exception as exc:
                last_exc = exc
                if attempt < 2 and self._is_retryable(exc):
                    delay = RETRY_BASE_DELAY * (RETRY_BACKOFF ** attempt)
                    logger.warning("Image gen failed (attempt %d/3): %s. Retrying in %.1fs...", attempt + 1, exc, delay)
                    await asyncio.sleep(delay)
                else:
                    break

        logger.error("OpenAI image generation failed (model=%s): %s", config.model, last_exc)
        raise RuntimeError(
            f"Image generation via OpenAI failed (model={config.model}): {last_exc}"
        ) from last_exc

    async def _generate_image_gemini(
        self,
        config: StageModelConfig,
        prompt: str,
    ) -> list[str]:
        """Generate images via Gemini native API (/v1beta/models/{model}:generateContent).

        Uses the Gemini-specific request format with responseModalities=["TEXT","IMAGE"].
        Returns a list of base64-encoded image strings extracted from inlineData parts.
        """
        base_url = (config.base_url or self._config.base_url).rstrip("/")
        # Strip OpenAI-style /v1 suffix — Gemini native API uses /v1beta/ path
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        api_key = config.api_key or self._config.api_key
        timeout = config.timeout or self._config.timeout

        url = f"{base_url}/v1beta/models/{config.model}:generateContent"

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"],
            },
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        logger.debug(
            "Generating image (gemini) model=%s aspect_ratio=%s image_size=%s",
            config.model, config.aspect_ratio, config.image_size,
        )

        last_exc: Exception | None = None
        data: dict = {}
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(url, json=payload, headers=headers)
                    response.raise_for_status()
                    data = response.json()
                break  # success
            except (httpx.TimeoutException, httpx.HTTPError) as exc:
                last_exc = exc
                retryable = isinstance(exc, httpx.TimeoutException) or (
                    isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in (429, 502, 503, 504)
                )
                if attempt < 2 and retryable:
                    delay = RETRY_BASE_DELAY * (RETRY_BACKOFF ** attempt)
                    logger.warning("Gemini image gen failed (attempt %d/3): %s. Retrying in %.1fs...", attempt + 1, exc, delay)
                    await asyncio.sleep(delay)
                else:
                    logger.error("Gemini image API failed: %s", exc)
                    raise RuntimeError(f"Gemini image API failed: {exc}") from exc

        # Extract base64 images from Gemini response
        images: list[str] = []
        candidates = data.get("candidates", [])
        for candidate in candidates:
            parts = candidate.get("content", {}).get("parts", [])
            for part in parts:
                inline_data = part.get("inlineData") or part.get("inline_data")
                if inline_data and "data" in inline_data:
                    images.append(inline_data["data"])

        if not images:
            logger.warning("Gemini response contained no image data. Response keys: %s", list(data.keys()))
            logger.debug("Full Gemini response: %s", json.dumps(data, ensure_ascii=False)[:2000])

        return images
