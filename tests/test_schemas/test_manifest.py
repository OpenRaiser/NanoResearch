"""Tests for nanoresearch.schemas.manifest."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from nanoresearch.schemas.manifest import (
    ArtifactRecord,
    DEEP_ONLY_STAGES,
    DEEP_PROCESSING_STAGES,
    PipelineMode,
    PipelineStage,
    STANDARD_PROCESSING_STAGES,
    StageRecord,
    WorkspaceManifest,
    processing_stages_for_mode,
)


class TestPipelineStage:
    """Tests for PipelineStage enum."""

    def test_all_stages_exist(self) -> None:
        assert PipelineStage.INIT.value == "INIT"
        assert PipelineStage.IDEATION.value == "IDEATION"
        assert PipelineStage.DONE.value == "DONE"
        assert PipelineStage.FAILED.value == "FAILED"

    def test_standard_stages(self) -> None:
        assert PipelineStage.IDEATION in STANDARD_PROCESSING_STAGES
        assert PipelineStage.PLANNING in STANDARD_PROCESSING_STAGES
        assert PipelineStage.CODING not in STANDARD_PROCESSING_STAGES

    def test_deep_stages_include_coding(self) -> None:
        assert PipelineStage.CODING in DEEP_PROCESSING_STAGES
        assert PipelineStage.EXECUTION in DEEP_PROCESSING_STAGES

    def test_deep_only_stages(self) -> None:
        assert PipelineStage.SETUP in DEEP_ONLY_STAGES
        assert PipelineStage.CODING in DEEP_ONLY_STAGES
        assert PipelineStage.IDEATION not in DEEP_ONLY_STAGES


class TestProcessingStagesForMode:
    """Tests for processing_stages_for_mode."""

    def test_standard_mode(self) -> None:
        stages = processing_stages_for_mode(PipelineMode.STANDARD)
        assert PipelineStage.IDEATION in stages
        assert PipelineStage.CODING not in stages

    def test_deep_mode(self) -> None:
        stages = processing_stages_for_mode(PipelineMode.DEEP)
        assert PipelineStage.CODING in stages
        assert PipelineStage.EXECUTION in stages
        assert len(stages) > len(STANDARD_PROCESSING_STAGES)


class TestStageRecord:
    """Tests for StageRecord model."""

    def test_defaults(self) -> None:
        sr = StageRecord(stage=PipelineStage.IDEATION)
        assert sr.stage == PipelineStage.IDEATION
        assert sr.status == "pending"
        assert sr.started_at is None
        assert sr.retries == 0

    def test_with_timestamps(self) -> None:
        now = datetime.now(timezone.utc)
        sr = StageRecord(
            stage=PipelineStage.PLANNING,
            status="completed",
            started_at=now,
            completed_at=now,
        )
        assert sr.status == "completed"
        assert sr.started_at == now
        assert sr.completed_at == now


class TestArtifactRecord:
    """Tests for ArtifactRecord model."""

    def test_minimal(self) -> None:
        ar = ArtifactRecord(
            name="ideation_output",
            path="papers/ideation_output.json",
            stage=PipelineStage.IDEATION,
        )
        assert ar.name == "ideation_output"
        assert ar.path == "papers/ideation_output.json"
        assert ar.stage == PipelineStage.IDEATION
        assert ar.checksum == ""
        assert ar.created_at is not None


class TestWorkspaceManifest:
    """Tests for WorkspaceManifest model."""

    def test_minimal(self) -> None:
        m = WorkspaceManifest(session_id="abc123", topic="ML research")
        assert m.session_id == "abc123"
        assert m.topic == "ML research"
        assert m.schema_version == "1.1"
        assert m.current_stage == PipelineStage.INIT
        assert m.pipeline_mode == PipelineMode.STANDARD
        assert m.stages == {}
        assert m.artifacts == []
        assert m.created_at is not None
        assert m.updated_at is not None

    def test_with_stages(self) -> None:
        m = WorkspaceManifest(
            session_id="s1",
            topic="T",
            stages={
                "ideation": StageRecord(
                    stage=PipelineStage.IDEATION,
                    status="completed",
                ),
            },
        )
        assert "ideation" in m.stages
        assert m.stages["ideation"].status == "completed"

    def test_serialization_roundtrip(self) -> None:
        m = WorkspaceManifest(session_id="x", topic="Y", pipeline_mode=PipelineMode.DEEP)
        data = m.model_dump(mode="json")
        restored = WorkspaceManifest.model_validate(data)
        assert restored.session_id == m.session_id
        assert restored.pipeline_mode == PipelineMode.DEEP
