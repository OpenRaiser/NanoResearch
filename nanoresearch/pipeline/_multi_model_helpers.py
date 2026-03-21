"""ModelDispatcher helpers — image generation, tool-call generation methods."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from functools import partial
from typing import Any

import httpx

from nanoresearch.agents.constants import (
    MAX_API_RETRIES,
    RETRY_BACKOFF_FACTOR,
    RETRY_BASE_DELAY,
)
from nanoresearch.config import StageModelConfig

logger = logging.getLogger(__name__)

RETRY_BACKOFF = RETRY_BACKOFF_FACTOR  # backward compat alias


class _MultiModelHelpersMixin:
    """Mixin — generate_with_image, generate_with_tools, generate_image methods."""

    async def generate_with_image(
        self,
        config: StageModelConfig,
        system_prompt: str,
        user_prompt: str,
        image_bytes: bytes,
        mime_type: str = "image/png",
        json_mode: bool = False,
    ) -> str:
        """Generate a completion with an image attachment (vision)."""
        import base64
        b64 = base64.b64encode(image_bytes).decode("ascii")
        data_url = f"data:{mime_type};base64,{b64}"

        timeout = config.timeout or self._config.timeout
        client = self._get_client(timeout, config.base_url, config.api_key)

        is_thinking = self._is_thinking_model(config.model)

        kwargs: dict[str, Any] = {
            "model": config.model,
            "messages": self._normalize_messages_for_model(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ]},
                ],
                is_thinking,
            ),
        }
        self._apply_completion_limit(kwargs, config, is_thinking)
        if config.temperature is not None and not is_thinking:
            kwargs["temperature"] = config.temperature
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        logger.debug("Calling vision model=%s timeout=%ss", config.model, timeout)

        loop = asyncio.get_running_loop()
        last_exc: Exception | None = None
        for attempt in range(MAX_API_RETRIES + 1):
            t0_img = time.monotonic()
            try:
                response = await loop.run_in_executor(
                    None,
                    partial(client.chat.completions.create, **kwargs),
                )
                latency_img = (time.monotonic() - t0_img) * 1000
                if not response.choices:
                    raise RuntimeError(
                        f"LLM returned empty choices (model={config.model})"
                    )
                content_img = self._strip_think_blocks(
                    response.choices[0].message.content or ""
                )
                self._notify_usage(content_img, self._extract_usage(response),
                                   config.model, latency_img)
                return content_img
            except Exception as exc:
                last_exc = exc
                if "max_completion_tokens" in str(exc) and "max_completion_tokens" in kwargs:
                    kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
                    continue
                if self._json_mode_fallback_supported(exc, kwargs):
                    logger.info(
                        "Vision backend doesn't support response_format=json_object, falling back to prompt-only JSON mode"
                    )
                    kwargs.pop("response_format", None)
                    continue
                if attempt < MAX_API_RETRIES and self._is_retryable(exc):
                    delay = RETRY_BASE_DELAY * (RETRY_BACKOFF ** attempt)
                    if "connection" in str(exc).lower():
                        delay = max(delay, 10.0)
                    logger.warning(
                        "LLM vision call failed (model=%s, attempt %d/%d): %s. Retrying in %.1fs...",
                        config.model, attempt + 1, MAX_API_RETRIES + 1, exc, delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    break

        logger.error("LLM vision call failed (model=%s): %s", config.model, last_exc)
        raise RuntimeError(
            f"LLM vision call to model {config.model!r} failed: {last_exc}"
        ) from last_exc

    async def generate_with_tools(
        self,
        config: StageModelConfig,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        """Generate a completion with optional tool/function calling."""
        timeout = config.timeout or self._config.timeout
        client = self._get_client(timeout, config.base_url, config.api_key)

        is_thinking = self._is_thinking_model(config.model)

        kwargs: dict[str, Any] = {
            "model": config.model,
            "messages": self._normalize_messages_for_model(messages, is_thinking),
        }
        self._apply_completion_limit(kwargs, config, is_thinking)
        if config.temperature is not None and not is_thinking:
            kwargs["temperature"] = config.temperature
        if tools:
            kwargs["tools"] = tools

        logger.debug(
            "Calling model=%s with %d messages, %d tools",
            config.model, len(kwargs["messages"]), len(tools or []),
        )

        loop = asyncio.get_running_loop()
        last_exc: Exception | None = None
        for attempt in range(MAX_API_RETRIES + 1):
            t0_tc = time.monotonic()
            try:
                response = await loop.run_in_executor(
                    None,
                    partial(client.chat.completions.create, **kwargs),
                )
                latency_tc = (time.monotonic() - t0_tc) * 1000
                if not response.choices:
                    raise RuntimeError(
                        f"LLM returned empty choices (model={config.model})"
                    )
                msg = response.choices[0].message
                self._notify_usage(
                    getattr(msg, "content", None) or "",
                    self._extract_usage(response),
                    config.model, latency_tc,
                )
                return msg
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
        """Generate images — routes to OpenAI images API or Gemini native API."""
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
        client = self._get_client(timeout, config.base_url, config.api_key)

        logger.debug("Generating image (openai) model=%s size=%s", config.model, size)

        loop = asyncio.get_running_loop()
        last_exc: Exception | None = None
        for attempt in range(3):
            t0_ig = time.monotonic()
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
                latency_ig = (time.monotonic() - t0_ig) * 1000
                self._notify_usage(
                    f"[image_gen:{size}:{quality}]", {},
                    config.model, latency_ig,
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
        """Generate images via Gemini native API."""
        base_url = (config.base_url or self._config.base_url).rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        api_key = config.api_key or self._config.api_key
        timeout = config.timeout or self._config.timeout

        url = f"{base_url}/v1beta/models/{config.model}:generateContent"

        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
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
            t0_gm = time.monotonic()
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=15.0)) as client:
                    response = await client.post(url, json=payload, headers=headers)
                    response.raise_for_status()
                    data = response.json()
                latency_gm = (time.monotonic() - t0_gm) * 1000
                self._notify_usage("[gemini_image_gen]", {}, config.model, latency_gm)
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

        if not data:
            raise RuntimeError(
                f"Gemini image API failed after 3 attempts: {last_exc}"
            ) from last_exc

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
