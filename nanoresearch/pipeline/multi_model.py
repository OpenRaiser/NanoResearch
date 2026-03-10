"""Per-stage model dispatcher — uses OpenAI SDK with a custom base URL.

Supports two image generation backends:
  - "openai": DALL-E via /v1/images/generations
  - "gemini": Gemini native API via /v1beta/models/{model}:generateContent
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from functools import partial
from typing import Any

import httpx
from openai import OpenAI

from nanoresearch.config import ResearchConfig, StageModelConfig
from nanoresearch.pipeline.cost_tracker import LLMResult

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
        self._clients: dict[tuple, OpenAI] = {}
        # Optional callback for cost tracking: called with (LLMResult,) after
        # each successful LLM call.  Set by orchestrator to feed CostTracker.
        self._usage_callback: Any | None = None

    def _get_client(
        self,
        timeout: float,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> OpenAI:
        """Get or create an OpenAI client for the given endpoint + timeout.

        Args:
            timeout: Request timeout in seconds.
            base_url: Per-stage override (falls back to global config).
            api_key: Per-stage override (falls back to global config).
        """
        resolved_url = base_url or self._config.base_url
        resolved_key = api_key or self._config.api_key
        timeout = round(timeout, 1)
        cache_key = (resolved_url, resolved_key, timeout)
        if cache_key not in self._clients:
            self._clients[cache_key] = OpenAI(
                base_url=resolved_url,
                api_key=resolved_key,
                timeout=timeout,
            )
        return self._clients[cache_key]

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

    @staticmethod
    def _is_thinking_model(model_name: str) -> bool:
        model_name = model_name.lower()
        return (
            "thinking" in model_name
            or model_name == "o1"
            or model_name.startswith("o1-")
            or model_name == "o3"
            or model_name.startswith("o3-")
        )

    @staticmethod
    def _normalize_messages_for_model(
        messages: list[dict[str, Any]],
        is_thinking: bool,
    ) -> list[dict[str, Any]]:
        if not is_thinking:
            return messages

        system_chunks: list[str] = []
        normalized: list[dict[str, Any]] = []
        merged = False

        for msg in messages:
            role = msg.get("role")
            if role == "system":
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    system_chunks.append(content.strip())
                elif content:
                    system_chunks.append(str(content).strip())
                continue

            cloned = dict(msg)
            if not merged and system_chunks and role == "user":
                prefix = "\n\n".join(chunk for chunk in system_chunks if chunk).strip()
                content = cloned.get("content")
                if isinstance(content, str) or content is None:
                    body = (content or "").strip()
                    cloned["content"] = f"{prefix}\n\n{body}" if body else prefix
                elif isinstance(content, list):
                    new_content = [
                        dict(item) if isinstance(item, dict) else item
                        for item in content
                    ]
                    injected = False
                    for item in new_content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            body = item.get("text", "")
                            item["text"] = f"{prefix}\n\n{body}" if body else prefix
                            injected = True
                            break
                    if not injected:
                        new_content.insert(0, {"type": "text", "text": prefix})
                    cloned["content"] = new_content
                else:
                    cloned["content"] = f"{prefix}\n\n{content}"
                merged = True
            normalized.append(cloned)

        if system_chunks and not merged:
            prefix = "\n\n".join(chunk for chunk in system_chunks if chunk).strip()
            if prefix:
                normalized.insert(0, {"role": "user", "content": prefix})
        return normalized

    @staticmethod
    def _apply_completion_limit(
        kwargs: dict[str, Any],
        config: StageModelConfig,
        is_thinking: bool,
    ) -> None:
        if is_thinking:
            kwargs["max_completion_tokens"] = config.max_tokens
        else:
            kwargs["max_tokens"] = config.max_tokens

    @staticmethod
    def _json_mode_fallback_supported(
        exc: Exception,
        kwargs: dict[str, Any],
    ) -> bool:
        if "response_format" not in kwargs:
            return False
        msg = str(exc).lower()
        return (
            "response_format" in msg
            and (
                "not supported" in msg
                or "unsupported" in msg
                or "unknown parameter" in msg
                or "invalid parameter" in msg
            )
        )

    @staticmethod
    def _extract_usage(response: Any) -> dict[str, int]:
        """Extract token usage dict from an OpenAI response object."""
        if hasattr(response, "usage") and response.usage is not None:
            return {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(response.usage, "completion_tokens", 0) or 0,
                "total_tokens": getattr(response.usage, "total_tokens", 0) or 0,
            }
        return {}

    def _notify_usage(self, content: str, usage: dict[str, int],
                      model: str, latency_ms: float) -> None:
        """Invoke usage callback if registered.  Never raises."""
        if self._usage_callback is not None:
            try:
                self._usage_callback(LLMResult(
                    content=content, usage=usage,
                    model=model, latency_ms=round(latency_ms, 1),
                ))
            except Exception as exc:
                logger.debug("Usage callback error (non-fatal): %s", exc)

    async def generate(
        self,
        config: StageModelConfig,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
    ) -> str:
        """Generate a completion using the configured model."""
        result = await self.generate_with_usage(config, system_prompt, user_prompt, json_mode)
        return result.content

    async def generate_with_usage(
        self,
        config: StageModelConfig,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
    ) -> LLMResult:
        """Like generate(), but returns an LLMResult with usage metadata.

        Does NOT change the signature of generate() — callers that only need
        the text can keep using generate() unchanged.
        """
        timeout = config.timeout or self._config.timeout
        client = self._get_client(timeout, config.base_url, config.api_key)

        is_thinking = self._is_thinking_model(config.model)

        kwargs: dict[str, Any] = {
            "model": config.model,
            "messages": self._normalize_messages_for_model(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                is_thinking,
            ),
        }
        self._apply_completion_limit(kwargs, config, is_thinking)
        if config.temperature is not None and not is_thinking:
            kwargs["temperature"] = config.temperature
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        loop = asyncio.get_running_loop()
        last_exc: Exception | None = None
        for attempt in range(MAX_API_RETRIES + 1):
            t0 = time.monotonic()
            try:
                response = await loop.run_in_executor(
                    None,
                    partial(client.chat.completions.create, **kwargs),
                )
                latency = (time.monotonic() - t0) * 1000
                if not response.choices:
                    raise RuntimeError(
                        f"LLM returned empty choices (model={config.model})"
                    )
                content = response.choices[0].message.content or ""
                usage = self._extract_usage(response)
                result = LLMResult(
                    content=content, usage=usage,
                    model=config.model, latency_ms=round(latency, 1),
                )
                self._notify_usage(content, usage, config.model, latency)
                return result
            except Exception as exc:
                last_exc = exc
                if "max_completion_tokens" in str(exc) and "max_completion_tokens" in kwargs:
                    kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
                    continue
                if self._json_mode_fallback_supported(exc, kwargs):
                    logger.info(
                        "Proxy doesn't support response_format=json_object, falling back to prompt-only JSON mode"
                    )
                    kwargs.pop("response_format", None)
                    continue
                if attempt < MAX_API_RETRIES and self._is_retryable(exc):
                    delay = RETRY_BASE_DELAY * (RETRY_BACKOFF ** attempt)
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

    async def generate_with_image(
        self,
        config: StageModelConfig,
        system_prompt: str,
        user_prompt: str,
        image_bytes: bytes,
        mime_type: str = "image/png",
        json_mode: bool = False,
    ) -> str:
        """Generate a completion with an image attachment (vision).

        Uses the OpenAI multimodal message format with base64-encoded inline
        image data.  Falls back to text-only if the model rejects image input.
        """
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
                content_img = response.choices[0].message.content or ""
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
                # Image gen has no token usage, but we track calls + latency
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
            t0_gm = time.monotonic()
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(url, json=payload, headers=headers)
                    response.raise_for_status()
                    data = response.json()
                latency_gm = (time.monotonic() - t0_gm) * 1000
                self._notify_usage(
                    "[gemini_image_gen]", {},
                    config.model, latency_gm,
                )
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
