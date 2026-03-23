"""Tests for nanoresearch.config."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from nanoresearch.config import (
    ExecutionProfile,
    ResearchConfig,
    StageModelConfig,
    WritingMode,
)


class TestExecutionProfile:
    """Tests for ExecutionProfile enum."""

    def test_values(self) -> None:
        assert ExecutionProfile.FAST_DRAFT.value == "fast_draft"
        assert ExecutionProfile.LOCAL_QUICK.value == "local_quick"
        assert ExecutionProfile.CLUSTER_FULL.value == "cluster_full"


class TestWritingMode:
    """Tests for WritingMode enum."""

    def test_values(self) -> None:
        assert WritingMode.DIRECT.value == "direct"
        assert WritingMode.HYBRID.value == "hybrid"
        assert WritingMode.REACT.value == "react"


class TestStageModelConfig:
    """Tests for StageModelConfig model."""

    def test_defaults(self) -> None:
        cfg = StageModelConfig()
        assert cfg.model != ""
        assert cfg.temperature == 0.3
        assert cfg.max_tokens == 8192
        assert cfg.image_backend == "openai"

    def test_custom_values(self) -> None:
        cfg = StageModelConfig(
            model="gpt-4",
            temperature=0.0,
            max_tokens=4096,
        )
        assert cfg.model == "gpt-4"
        assert cfg.temperature == 0.0
        assert cfg.max_tokens == 4096


class TestResearchConfig:
    """Tests for ResearchConfig model (without file I/O)."""

    def test_model_validate_minimal(self, valid_config_dict: dict) -> None:
        cfg = ResearchConfig.model_validate(valid_config_dict)
        assert cfg.base_url == "https://api.example.com"
        assert cfg.api_key == "test-api-key-123"
        assert cfg.timeout == 180.0

    def test_for_stage(self, valid_config: ResearchConfig) -> None:
        cfg = valid_config.for_stage("ideation")
        assert cfg is valid_config.ideation
        cfg2 = valid_config.for_stage("IDEATION")
        assert cfg2 is valid_config.ideation

    def test_for_stage_unknown_raises(self, valid_config: ResearchConfig) -> None:
        with pytest.raises(ValueError, match="Unknown stage"):
            valid_config.for_stage("unknown_stage")

    def test_prefers_cluster_execution_default(self, valid_config: ResearchConfig) -> None:
        assert valid_config.prefers_cluster_execution() is False

    def test_prefers_cluster_with_profile(self, valid_config_dict: dict) -> None:
        valid_config_dict["execution_profile"] = "cluster_full"
        cfg = ResearchConfig.model_validate(valid_config_dict)
        assert cfg.prefers_cluster_execution() is True

    def test_should_use_writing_tools_direct_mode(self, valid_config_dict: dict) -> None:
        valid_config_dict["writing_mode"] = "direct"
        cfg = ResearchConfig.model_validate(valid_config_dict)
        assert cfg.should_use_writing_tools("Introduction") is False
        assert cfg.should_use_writing_tools("Method") is False

    def test_should_use_writing_tools_react_mode(self, valid_config_dict: dict) -> None:
        valid_config_dict["writing_mode"] = "react"
        cfg = ResearchConfig.model_validate(valid_config_dict)
        assert cfg.should_use_writing_tools("Introduction") is True
        assert cfg.should_use_writing_tools("Method") is True

    def test_should_use_writing_tools_hybrid(self, valid_config: ResearchConfig) -> None:
        assert valid_config.should_use_writing_tools("Introduction") is True
        assert valid_config.should_use_writing_tools("Method") is True
        assert valid_config.should_use_writing_tools("Random Section") is False

    def test_snapshot_strips_api_key(self, valid_config: ResearchConfig) -> None:
        snap = valid_config.snapshot()
        assert "api_key" not in snap or snap.get("api_key") != valid_config.api_key


class TestResearchConfigLoad:
    """Tests for ResearchConfig.load() with file I/O."""

    def test_load_from_file(
        self,
        tmp_config_file: Path,
        valid_config_dict: dict,
    ) -> None:
        cfg = ResearchConfig.load(tmp_config_file)
        assert cfg.base_url == valid_config_dict["base_url"]
        assert cfg.api_key == valid_config_dict["api_key"]

    def test_load_missing_credentials_raises(self, tmp_path: Path) -> None:
        import json

        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps({"research": {"base_url": "", "api_key": ""}}, indent=2),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="API credentials required"):
            ResearchConfig.load(config_path)

    def test_load_invalid_json_raises(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.json"
        config_path.write_text("{ invalid }", encoding="utf-8")
        with pytest.raises(RuntimeError, match="invalid JSON"):
            ResearchConfig.load(config_path)

    def test_load_env_override(
        self,
        tmp_config_file: Path,
        valid_config_dict: dict,
    ) -> None:
        os.environ["NANORESEARCH_BASE_URL"] = "https://env-override.com"
        os.environ["NANORESEARCH_API_KEY"] = "env-key"
        try:
            cfg = ResearchConfig.load(tmp_config_file)
            assert cfg.base_url == "https://env-override.com"
            assert cfg.api_key == "env-key"
        finally:
            os.environ.pop("NANORESEARCH_BASE_URL", None)
            os.environ.pop("NANORESEARCH_API_KEY", None)
