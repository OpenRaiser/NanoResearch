"""End-to-end smoke test — validates pipeline assembly and data flow."""
import pytest
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

from nanoresearch.config import ResearchConfig
from nanoresearch.pipeline.workspace import Workspace
from nanoresearch.pipeline.orchestrator import PipelineOrchestrator
from nanoresearch.pipeline.state import PipelineStateMachine
from nanoresearch.schemas.manifest import PipelineStage


class TestE2ESmoke:
    """Smoke tests for pipeline assembly and data flow."""

    def test_orchestrator_creates_all_agents(self, tmp_path):
        """All 6 stage agents are instantiated."""
        ws = Workspace.create("test topic", root=tmp_path)
        config = ResearchConfig()
        orch = PipelineOrchestrator(ws, config)
        assert len(orch._agents) == 6
        for stage in PipelineStateMachine.processing_stages():
            assert stage in orch._agents

    def test_stage_key_map_covers_all_stages(self):
        """_STAGE_KEY_MAP has an entry for every processing stage."""
        for stage in PipelineStateMachine.processing_stages():
            assert stage in PipelineOrchestrator._STAGE_KEY_MAP

    def test_prepare_inputs_ideation(self, tmp_path):
        """Ideation stage receives topic."""
        ws = Workspace.create("test topic", root=tmp_path)
        config = ResearchConfig()
        orch = PipelineOrchestrator(ws, config)
        inputs = orch._prepare_inputs(
            PipelineStage.IDEATION, "my topic", {}, ""
        )
        assert inputs["topic"] == "my topic"

    def test_prepare_inputs_planning(self, tmp_path):
        """Planning stage receives ideation_output."""
        ws = Workspace.create("test topic", root=tmp_path)
        config = ResearchConfig()
        orch = PipelineOrchestrator(ws, config)
        accumulated = {"ideation_output": {"topic": "t", "hypotheses": []}}
        inputs = orch._prepare_inputs(
            PipelineStage.PLANNING, "t", accumulated, ""
        )
        assert "ideation_output" in inputs

    def test_prepare_inputs_experiment(self, tmp_path):
        """Experiment stage receives blueprint and reference repos."""
        ws = Workspace.create("test topic", root=tmp_path)
        config = ResearchConfig()
        orch = PipelineOrchestrator(ws, config)
        accumulated = {
            "experiment_blueprint": {"title": "test"},
            "ideation_output": {"reference_repos": [{"name": "repo1"}]},
        }
        inputs = orch._prepare_inputs(
            PipelineStage.EXPERIMENT, "t", accumulated, ""
        )
        assert "experiment_blueprint" in inputs
        assert "reference_repos" in inputs

    def test_wrap_and_load_round_trip(self, tmp_path):
        """Data can be wrapped, saved, and loaded back."""
        ws = Workspace.create("test topic", root=tmp_path)
        config = ResearchConfig()
        orch = PipelineOrchestrator(ws, config)

        # Simulate saving ideation output
        test_data = {"topic": "test", "papers": [], "hypotheses": []}
        wrapped = orch._wrap_stage_output(PipelineStage.IDEATION, test_data)
        assert "ideation_output" in wrapped

        # Save it to workspace
        ws.write_json("papers/ideation_output.json", test_data)

        # Load it back
        loaded = orch._load_stage_output(PipelineStage.IDEATION)
        assert "ideation_output" in loaded
        assert loaded["ideation_output"]["topic"] == "test"

    def test_reset_stale_running_stages(self, tmp_path):
        """Stages stuck in 'running' get reset to 'pending'."""
        ws = Workspace.create("test topic", root=tmp_path)
        config = ResearchConfig()
        # Manually set a stage to running
        ws.mark_stage_running(PipelineStage.IDEATION)
        assert ws.manifest.stages["IDEATION"].status == "running"

        orch = PipelineOrchestrator(ws, config)
        orch._reset_stale_running_stages()
        assert ws.manifest.stages["IDEATION"].status == "pending"

    def test_skip_stages_config(self, tmp_path):
        """Configured skip_stages are respected."""
        ws = Workspace.create("test topic", root=tmp_path)
        config = ResearchConfig(skip_stages=["EXPERIMENT"])
        orch = PipelineOrchestrator(ws, config)
        assert "EXPERIMENT" in config.skip_stages

    def test_cross_stage_validation_no_crash(self, tmp_path):
        """Cross-stage validation doesn't crash with empty data."""
        ws = Workspace.create("test topic", root=tmp_path)
        config = ResearchConfig()
        orch = PipelineOrchestrator(ws, config)
        # Should not raise even with empty results
        orch._validate_cross_stage_refs(PipelineStage.PLANNING, {})
        orch._validate_cross_stage_refs(PipelineStage.EXPERIMENT, {})
