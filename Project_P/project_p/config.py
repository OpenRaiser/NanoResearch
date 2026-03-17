"""Configuration loading for Project_P."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"


@dataclass
class LLMConfig:
    base_url: str = ""
    api_key: str = ""
    model: str = "claude-sonnet-4-6"
    vision_model: str = "claude-sonnet-4-6"
    temperature: float = 0.1
    timeout: int = 60


@dataclass
class Config:
    tectonic_path: str = "tectonic"
    llm: LLMConfig = field(default_factory=LLMConfig)

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        path = path or _DEFAULT_CONFIG_PATH
        if not path.exists():
            logger.warning("Config file not found at %s, using defaults", path)
            return cls()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to load config: %s", exc)
            return cls()

        llm_raw = raw.get("llm", {})
        llm = LLMConfig(
            base_url=llm_raw.get("base_url", ""),
            api_key=llm_raw.get("api_key", ""),
            model=llm_raw.get("model", "claude-sonnet-4-6"),
            vision_model=llm_raw.get("vision_model", "claude-sonnet-4-6"),
            temperature=llm_raw.get("temperature", 0.1),
            timeout=llm_raw.get("timeout", 60),
        )
        return cls(
            tectonic_path=raw.get("tectonic_path", "tectonic"),
            llm=llm,
        )
