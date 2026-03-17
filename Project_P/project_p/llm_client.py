"""OpenAI-compatible LLM client with text and vision support."""
from __future__ import annotations

import base64
import json
import logging
import re

logger = logging.getLogger(__name__)


class LLMClient:
    """OpenAI-compatible client supporting text and vision."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        vision_model: str | None = None,
        temperature: float = 0.1,
        timeout: int = 60,
    ):
        import openai
        self.client = openai.OpenAI(
            base_url=base_url, api_key=api_key, timeout=timeout,
        )
        self.model = model
        self.vision_model = vision_model or model
        self.temperature = temperature

    def generate(
        self,
        system: str,
        user: str,
        json_mode: bool = False,
    ) -> str:
        """Generate text response."""
        kwargs: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = self.client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""
        except Exception as exc:
            logger.error("LLM generate failed: %s", exc)
            return ""

    def generate_with_image(
        self,
        system: str,
        user_text: str,
        image_bytes: bytes,
        json_mode: bool = False,
    ) -> str:
        """Generate response with an image input (vision)."""
        b64 = base64.b64encode(image_bytes).decode("ascii")
        kwargs: dict = {
            "model": self.vision_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{b64}",
                    }},
                ]},
            ],
            "temperature": self.temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = self.client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""
        except Exception as exc:
            logger.error("LLM vision generate failed: %s", exc)
            return ""

    def safe_parse_json(self, text: str, default: dict | None = None) -> dict:
        """Parse JSON from LLM response, handling markdown fences."""
        if default is None:
            default = {}
        if not text.strip():
            return default
        # Strip markdown fences
        cleaned = re.sub(r'^```(?:json)?\s*\n?', '', text.strip())
        cleaned = re.sub(r'\n?```\s*$', '', cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM JSON: %.200s", text)
            return default
