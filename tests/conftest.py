"""Pytest fixtures for NanoResearch tests."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from nanoresearch.config import ResearchConfig


@pytest.fixture
def valid_config_dict() -> dict:
    """Minimal valid config dict for ResearchConfig.model_validate."""
    return {
        "base_url": "https://api.example.com",
        "api_key": "test-api-key-123",
    }


@pytest.fixture
def valid_config(valid_config_dict: dict) -> "ResearchConfig":
    """ResearchConfig instance with valid credentials (no file I/O)."""
    from nanoresearch.config import ResearchConfig

    return ResearchConfig.model_validate(valid_config_dict)


@pytest.fixture
def tmp_config_file(tmp_path: Path, valid_config_dict: dict) -> Path:
    """Create a temporary config.json for load() tests."""
    import json

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"research": valid_config_dict}, indent=2),
        encoding="utf-8",
    )
    return config_path
