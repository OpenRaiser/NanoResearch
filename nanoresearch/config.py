"""Research configuration — per-stage model routing and global settings."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "deepseek-ai/DeepSeek-V3.2"


class StageModelConfig(BaseModel):
    """Configuration for a single pipeline stage."""

    model: str = DEFAULT_MODEL
    temperature: float | None = 0.3  # None = don't send (for models like Codex/o-series)
    max_tokens: int = 8192
    timeout: float | None = None  # per-stage override; None = use global

    # Image generation backend: "openai" (DALL-E) or "gemini" (native Gemini API)
    image_backend: str = "openai"
    # Per-stage base_url / api_key override (e.g. Gemini image API uses different endpoint)
    base_url: str | None = None
    api_key: str | None = None
    # Gemini-specific image options
    aspect_ratio: str = "1:1"
    image_size: str = "1024x1024"


class ResearchConfig(BaseModel):
    """Top-level research configuration."""

    base_url: str = ""
    api_key: str = ""
    timeout: float = 180.0

    ideation: StageModelConfig = Field(
        default_factory=lambda: StageModelConfig(
            model="deepseek-ai/DeepSeek-V3.2", temperature=0.5,
            max_tokens=16384, timeout=600.0,
        )
    )
    planning: StageModelConfig = Field(
        default_factory=lambda: StageModelConfig(
            model="deepseek-ai/DeepSeek-V3.2", temperature=0.2,
            max_tokens=16384, timeout=600.0,
        )
    )
    experiment: StageModelConfig = Field(
        default_factory=lambda: StageModelConfig(
            model="deepseek-ai/DeepSeek-V3.2", temperature=0.1, timeout=600.0
        )
    )
    writing: StageModelConfig = Field(
        default_factory=lambda: StageModelConfig(
            model="deepseek-ai/DeepSeek-V3.2", temperature=0.4,
            max_tokens=16384, timeout=600.0,
        )
    )
    code_gen: StageModelConfig = Field(
        default_factory=lambda: StageModelConfig(
            model="gpt-5.2-codex", temperature=None,
            max_tokens=16384, timeout=600.0,
        )
    )
    figure_prompt: StageModelConfig = Field(
        default_factory=lambda: StageModelConfig(
            model="claude-sonnet-4-6-20250514", temperature=0.5,
            max_tokens=4096, timeout=300.0,
        )
    )
    figure_code: StageModelConfig = Field(
        default_factory=lambda: StageModelConfig(
            model="claude-sonnet-4-6-20250514", temperature=0.1,
            max_tokens=16384, timeout=600.0,
        )
    )
    figure_gen: StageModelConfig = Field(
        default_factory=lambda: StageModelConfig(
            model="gemini-3.1-flash-preview-image-generation",
            image_backend="gemini",
            temperature=None,
            timeout=300.0,
        )
    )
    evidence_extraction: StageModelConfig = Field(
        default_factory=lambda: StageModelConfig(
            model="deepseek-ai/DeepSeek-V3.2",
            temperature=0.1,
        )
    )
    review: StageModelConfig = Field(
        default_factory=lambda: StageModelConfig(
            model="gemini-3.1-flash-lite-preview",
            temperature=0.3,
            max_tokens=16384,
            timeout=300.0,
        )
    )
    revision: StageModelConfig = Field(
        default_factory=lambda: StageModelConfig(
            model="gemini-3-pro-preview-thinking",
            temperature=0.3,
            max_tokens=16384,
            timeout=600.0,
        )
    )
    skip_stages: list[str] = Field(default_factory=list)
    template_format: str = "neurips2025"
    max_retries: int = 2
    quick_eval_timeout: int = 1200  # seconds for quick-eval execution (20 min)

    # Use an existing conda env instead of creating a new venv.
    # When set, experiment agent skips venv creation and uses this env's Python.
    experiment_conda_env: str = ""  # e.g., "shixun"

    # Cluster execution settings (optional — set in config.json under "research.cluster")
    cluster: dict = Field(default_factory=dict)  # {"enabled":true, "host":..., "user":..., ...}

    # Iteration settings for experiment agent
    experiment_max_rounds: int = 3           # maximum iteration rounds
    experiment_plateau_patience: int = 2     # consecutive rounds with < threshold improvement
    experiment_improvement_threshold: float = 0.005  # 0.5% minimum improvement

    # ReAct experiment mode: "pipeline" (default, hardcoded phases) or "react" (LLM-driven tools)
    experiment_mode: str = "pipeline"
    # Max tool-call rounds in react mode (each round = one LLM ↔ tool exchange)
    react_max_rounds: int = 80
    # SLURM settings for react mode (auto-detected if empty)
    slurm_partition: str = ""               # e.g., "belt_road"
    slurm_max_gpus: int = 2                 # max GPUs per job
    slurm_default_time: str = "4:00:00"     # default wall time
    # Container settings for react mode (for clusters with old glibc)
    container_image: str = ""               # e.g., "docker://ubuntu:22.04" (clean base with glibc 2.35)
    container_path: str = ""                # e.g., "/mnt/shared/ubuntu2204.sif"
    container_bind: str = "/mnt:/mnt"       # bind mounts for apptainer

    @classmethod
    def load(cls, path: Path | None = None) -> "ResearchConfig":
        """Load config from nanobot config file, then overlay env vars."""
        if path is None:
            path = Path.home() / ".nanobot" / "config.json"

        research: dict = {}
        if path.is_file():
            try:
                raw = path.read_text(encoding="utf-8")
            except OSError as exc:
                raise RuntimeError(
                    f"Cannot read config file {path}: {exc}"
                ) from exc
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Config file {path} contains invalid JSON: {exc}"
                ) from exc
            if not isinstance(data, dict):
                raise RuntimeError(
                    f"Config file {path} must be a JSON object, got {type(data).__name__}"
                )
            research = data.get("research", {})

        try:
            cfg = cls.model_validate(research)
        except Exception as exc:
            raise RuntimeError(
                f"Invalid config values in {path}: {exc}"
            ) from exc

        # Environment variable overrides (highest priority)
        if env_url := os.environ.get("NANORESEARCH_BASE_URL"):
            cfg.base_url = env_url.strip()
        if env_key := os.environ.get("NANORESEARCH_API_KEY"):
            cfg.api_key = env_key.strip()
        if env_timeout := os.environ.get("NANORESEARCH_TIMEOUT"):
            try:
                cfg.timeout = float(env_timeout)
            except ValueError:
                logger.warning(
                    "NANORESEARCH_TIMEOUT=%r is not a valid float, using default %.1f",
                    env_timeout, cfg.timeout,
                )

        if not cfg.base_url or not cfg.api_key:
            raise ValueError(
                "API credentials required. Set NANORESEARCH_BASE_URL and "
                "NANORESEARCH_API_KEY environment variables, or configure "
                "them in ~/.nanobot/config.json under 'research'."
            )
        return cfg

    def for_stage(self, stage_name: str) -> StageModelConfig:
        """Return model config for the given stage name."""
        stage_name = stage_name.lower()
        mapping = {
            "ideation": self.ideation,
            "planning": self.planning,
            "experiment": self.experiment,
            "writing": self.writing,
            "code_gen": self.code_gen,
            "figure_prompt": self.figure_prompt,
            "figure_code": self.figure_code,
            "figure_gen": self.figure_gen,
            "evidence_extraction": self.evidence_extraction,
            "review": self.review,
            "revision": self.revision,
        }
        if stage_name not in mapping:
            raise ValueError(f"Unknown stage: {stage_name}. Valid: {list(mapping)}")
        return mapping[stage_name]

    def snapshot(self) -> dict:
        """Return a JSON-serializable snapshot for manifest storage.

        Strips all API keys (global and per-stage) to prevent accidental leaks.
        """
        d = self.model_dump()
        d.pop("api_key", None)  # don't persist global API key
        # Also strip per-stage api_key overrides
        for key, val in d.items():
            if isinstance(val, dict) and "api_key" in val:
                val.pop("api_key", None)
        return d
