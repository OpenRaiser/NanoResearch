"""Workspace manifest and pipeline stage tracking."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class PipelineStage(str, Enum):
    """Stages of the research pipeline."""

    INIT = "INIT"
    IDEATION = "IDEATION"
    PLANNING = "PLANNING"
    EXPERIMENT = "EXPERIMENT"
    FIGURE_GEN = "FIGURE_GEN"
    WRITING = "WRITING"
    REVIEW = "REVIEW"
    DONE = "DONE"
    FAILED = "FAILED"

# Valid stage record status values
StageStatus = Literal["pending", "running", "completed", "failed"]


# Valid forward transitions in the state machine.
STAGE_TRANSITIONS: dict[PipelineStage, list[PipelineStage]] = {
    PipelineStage.INIT: [PipelineStage.IDEATION, PipelineStage.FAILED],
    PipelineStage.IDEATION: [PipelineStage.PLANNING, PipelineStage.FAILED],
    PipelineStage.PLANNING: [PipelineStage.EXPERIMENT, PipelineStage.FAILED],
    PipelineStage.EXPERIMENT: [PipelineStage.FIGURE_GEN, PipelineStage.FAILED],
    PipelineStage.FIGURE_GEN: [PipelineStage.WRITING, PipelineStage.FAILED],
    PipelineStage.WRITING: [PipelineStage.REVIEW, PipelineStage.FAILED],
    PipelineStage.REVIEW: [PipelineStage.DONE, PipelineStage.FAILED],
    PipelineStage.DONE: [],
    PipelineStage.FAILED: [],
}


class StageRecord(BaseModel):
    """Record of a single pipeline stage execution."""

    stage: PipelineStage
    status: StageStatus = Field(default="pending")
    started_at: datetime | None = None
    completed_at: datetime | None = None
    retries: int = Field(default=0, ge=0)
    error_message: str = ""
    output_path: str = ""


class ArtifactRecord(BaseModel):
    """A registered output artifact."""

    name: str
    path: str
    stage: PipelineStage
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    checksum: str = ""


class WorkspaceManifest(BaseModel):
    """Master manifest for a research session workspace."""

    schema_version: str = "1.0"
    session_id: str
    topic: str
    current_stage: PipelineStage = PipelineStage.INIT
    stages: dict[str, StageRecord] = Field(default_factory=dict)
    artifacts: list[ArtifactRecord] = Field(default_factory=list)
    config_snapshot: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
