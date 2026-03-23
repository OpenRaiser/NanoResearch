"""Tests for nanoresearch.pipeline.workspace."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nanoresearch.pipeline.workspace import WORKSPACE_DIRS, Workspace
from nanoresearch.schemas.manifest import PipelineMode, PipelineStage


class TestWorkspaceCreate:
    """Tests for Workspace.create()."""

    def test_create_standard_mode(self, tmp_path: Path) -> None:
        ws = Workspace.create(
            topic="Test topic",
            root=tmp_path,
            session_id="test123",
            pipeline_mode=PipelineMode.STANDARD,
        )
        assert ws.path == tmp_path / "test123"
        assert ws.path.is_dir()
        assert (ws.path / "manifest.json").is_file()
        for d in WORKSPACE_DIRS:
            assert (ws.path / d).is_dir()

    def test_create_deep_mode(self, tmp_path: Path) -> None:
        ws = Workspace.create(
            topic="Deep test",
            root=tmp_path,
            session_id="deep1",
            pipeline_mode=PipelineMode.DEEP,
        )
        assert ws.manifest.pipeline_mode == PipelineMode.DEEP
        assert PipelineStage.CODING.value in ws.manifest.stages

    def test_manifest_has_correct_topic(self, tmp_path: Path) -> None:
        ws = Workspace.create(topic="My Research", root=tmp_path, session_id="s1")
        assert ws.manifest.topic == "My Research"
        assert ws.manifest.session_id == "s1"
        assert ws.manifest.current_stage == PipelineStage.INIT

    def test_create_with_config_snapshot(self, tmp_path: Path) -> None:
        config = {"base_url": "https://api.example.com", "timeout": 120}
        ws = Workspace.create(
            topic="T",
            root=tmp_path,
            session_id="s1",
            config_snapshot=config,
        )
        assert ws.manifest.config_snapshot == config


class TestWorkspaceLoad:
    """Tests for Workspace.load()."""

    def test_load_existing_workspace(self, tmp_path: Path) -> None:
        ws_created = Workspace.create(topic="Load test", root=tmp_path, session_id="load1")
        ws_loaded = Workspace.load(ws_created.path)
        assert ws_loaded.path == ws_created.path
        assert ws_loaded.manifest.topic == "Load test"

    def test_load_nonexistent_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            Workspace.load(tmp_path / "nonexistent")


class TestWorkspaceManifest:
    """Tests for Workspace manifest property."""

    def test_manifest_cached_after_read(self, tmp_path: Path) -> None:
        ws = Workspace.create(topic="T", root=tmp_path, session_id="s1")
        m1 = ws.manifest
        m2 = ws.manifest
        assert m1 is m2

    def test_invalid_manifest_json_raises(self, tmp_path: Path) -> None:
        ws_path = tmp_path / "bad"
        ws_path.mkdir()
        for d in WORKSPACE_DIRS:
            (ws_path / d).mkdir()
        (ws_path / "manifest.json").write_text("{ invalid }", encoding="utf-8")
        ws = Workspace(ws_path)
        with pytest.raises(RuntimeError, match="invalid JSON"):
            _ = ws.manifest
