"""Tests for workspace management."""

import json
from pathlib import Path

import pytest

from nanoresearch.pipeline.workspace import Workspace, WORKSPACE_DIRS
from nanoresearch.schemas.manifest import PipelineStage, WorkspaceManifest


class TestWorkspaceCreate:
    def test_creates_directories(self, tmp_path: Path):
        ws = Workspace.create(topic="test topic", root=tmp_path, session_id="s001")
        for d in WORKSPACE_DIRS:
            assert (ws.path / d).is_dir()

    def test_creates_manifest(self, tmp_path: Path):
        ws = Workspace.create(topic="test topic", root=tmp_path, session_id="s001")
        manifest = ws.manifest
        assert manifest.session_id == "s001"
        assert manifest.topic == "test topic"
        assert manifest.current_stage == PipelineStage.INIT

    def test_auto_session_id(self, tmp_path: Path):
        ws = Workspace.create(topic="test", root=tmp_path)
        assert len(ws.manifest.session_id) == 12

    def test_config_snapshot(self, tmp_path: Path):
        ws = Workspace.create(topic="test", root=tmp_path, config_snapshot={"model": "gpt-4"})
        assert ws.manifest.config_snapshot["model"] == "gpt-4"


class TestWorkspaceLoad:
    def test_load_existing(self, tmp_workspace: Workspace):
        loaded = Workspace.load(tmp_workspace.path)
        assert loaded.manifest.session_id == tmp_workspace.manifest.session_id

    def test_load_nonexistent(self, tmp_path: Path):
        with pytest.raises(Exception):
            Workspace.load(tmp_path / "nonexistent")


class TestWorkspaceStageTracking:
    def test_mark_running(self, tmp_workspace: Workspace):
        tmp_workspace.mark_stage_running(PipelineStage.IDEATION)
        m = tmp_workspace.manifest
        assert m.current_stage == PipelineStage.IDEATION
        assert m.stages["IDEATION"].status == "running"
        assert m.stages["IDEATION"].started_at is not None

    def test_mark_completed(self, tmp_workspace: Workspace):
        tmp_workspace.mark_stage_running(PipelineStage.IDEATION)
        tmp_workspace.mark_stage_completed(PipelineStage.IDEATION, "papers/output.json")
        m = tmp_workspace.manifest
        assert m.stages["IDEATION"].status == "completed"
        assert m.stages["IDEATION"].output_path == "papers/output.json"

    def test_mark_failed(self, tmp_workspace: Workspace):
        tmp_workspace.mark_stage_running(PipelineStage.IDEATION)
        tmp_workspace.mark_stage_failed(PipelineStage.IDEATION, "API error")
        m = tmp_workspace.manifest
        assert m.stages["IDEATION"].status == "failed"
        assert m.stages["IDEATION"].error_message == "API error"
        assert m.current_stage == PipelineStage.FAILED

    def test_increment_retry(self, tmp_workspace: Workspace):
        tmp_workspace.mark_stage_running(PipelineStage.IDEATION)
        count = tmp_workspace.increment_retry(PipelineStage.IDEATION)
        assert count == 1
        m = tmp_workspace.manifest
        assert m.stages["IDEATION"].retries == 1
        assert m.stages["IDEATION"].status == "pending"


class TestWorkspaceArtifacts:
    def test_register_artifact(self, tmp_workspace: Workspace):
        # Create a test file
        test_file = tmp_workspace.path / "papers" / "test.json"
        test_file.write_text('{"test": true}')

        record = tmp_workspace.register_artifact(
            "test_output", test_file, PipelineStage.IDEATION
        )
        assert record.name == "test_output"
        assert record.checksum  # should have MD5

        m = tmp_workspace.manifest
        assert len(m.artifacts) == 1


class TestWorkspaceIO:
    def test_write_and_read_json(self, tmp_workspace: Workspace):
        data = {"key": "value", "number": 42}
        path = tmp_workspace.write_json("papers/test.json", data)
        assert path.is_file()
        loaded = tmp_workspace.read_json("papers/test.json")
        assert loaded == data

    def test_write_and_read_text(self, tmp_workspace: Workspace):
        text = "Hello, world!"
        path = tmp_workspace.write_text("logs/test.txt", text)
        assert path.is_file()
        assert tmp_workspace.read_text("logs/test.txt") == text

    def test_convenience_paths(self, tmp_workspace: Workspace):
        assert tmp_workspace.papers_dir.name == "papers"
        assert tmp_workspace.plans_dir.name == "plans"
        assert tmp_workspace.drafts_dir.name == "drafts"
        assert tmp_workspace.figures_dir.name == "figures"
        assert tmp_workspace.logs_dir.name == "logs"
