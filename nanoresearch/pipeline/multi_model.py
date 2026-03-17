"""Per-stage model dispatcher — uses OpenAI SDK with a custom base URL.

Supports two image generation backends:
  - "openai": DALL-E via /v1/images/generations
  - "gemini": Gemini native API via /v1beta/models/{model}:generateContent
"""

from __future__ import annotations

import asyncio
import logging
import time
from functools import partial
from typing import Any

import httpx
from openai import OpenAI

from nanoresearch.config import ResearchConfig, StageModelConfig
from nanoresearch.pipeline.cost_tracker import LLMResult
from nanoresearch.pipeline._multi_model_helpers import _MultiModelHelpersMixin

logger = logging.getLogger(__name__)

# Retry settings for transient API errors (centralised in constants.py)
from nanoresearch.agents.constants import (
    MAX_API_RETRIES,
    RETRY_BACKOFF_FACTOR,
    RETRY_BASE_DELAY,
)

RETRY_BACKOFF = RETRY_BACKOFF_FACTOR  # backward compat alias

# Exceptions worth retrying (strings matched in error message)
_RETRYABLE_PATTERNS = (
    "timeout", "timed out", "rate limit", "429", "502", "503", "504",
    "connection", "server error", "overloaded", "capacity",
)


class ModelDispatcher(_MultiModelHelpersMixin):
    """Dispatches LLM calls via the OpenAI-compatible API.

    All stages use the same base_url + api_key (your self-hosted endpoint).
    Each stage has its own model name, temperature, max_tokens, and timeout.
    """

    def __init__(self, config: ResearchConfig) -> None:
        self._config = config
        self._clients: dict[tuple, OpenAI] = {}
        # Optional callback for cost tracking
        self._usage_callback: Any | None = None

    def _get_client(
        self,
        timeout: float,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> OpenAI:
        """Get or create an OpenAI client for the given endpoint + timeout."""
        resolved_url = base_url or self._config.base_url
        resolved_key = api_key or self._config.api_key
        timeout = round(timeout, 1)
        cache_key = (resolved_url, resolved_key, timeout)
        if cache_key not in self._clients:
            self._clients[cache_key] = OpenAI(
                base_url=resolved_url,
                api_key=resolved_key,
                timeout=httpx.Timeout(timeout, connect=15.0),
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
        """Like generate(), but returns an LLMResult with usage metadata."""
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
