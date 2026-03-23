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
    SETUP = "SETUP"
    CODING = "CODING"
    EXECUTION = "EXECUTION"
    ANALYSIS = "ANALYSIS"
    FIGURE_GEN = "FIGURE_GEN"
    WRITING = "WRITING"
    REVIEW = "REVIEW"
    DONE = "DONE"
    FAILED = "FAILED"


class PipelineMode(str, Enum):
    """Supported pipeline variants."""

    STANDARD = "standard"
    DEEP = "deep"


class PaperMode(str, Enum):
    """Paper type/mode for the research pipeline."""

    ORIGINAL_RESEARCH = "original_research"
    SURVEY_SHORT = "survey_short"
    SURVEY_STANDARD = "survey_standard"
    SURVEY_LONG = "survey_long"

    @classmethod
    def from_string(cls, s: str) -> "PaperMode":
        """Parse from string like 'survey:short:' or 'original:' prefix."""
        s = s.strip().lower()
        if s.startswith("survey:short:"):
            return cls.SURVEY_SHORT
        if s.startswith("survey:long:"):
            return cls.SURVEY_LONG
        if s.startswith("survey:"):
            return cls.SURVEY_STANDARD
        return cls.ORIGINAL_RESEARCH

    @property
    def is_survey(self) -> bool:
        return self != PaperMode.ORIGINAL_RESEARCH

    @property
    def survey_size(self) -> str | None:
        if not self.is_survey:
            return None
        return {
            PaperMode.SURVEY_SHORT: "short",
            PaperMode.SURVEY_STANDARD: "standard",
            PaperMode.SURVEY_LONG: "long",
        }[self]


STANDARD_PROCESSING_STAGES: list[PipelineStage] = [
    PipelineStage.IDEATION,
    PipelineStage.PLANNING,
    PipelineStage.EXPERIMENT,
    PipelineStage.FIGURE_GEN,
    PipelineStage.WRITING,
    PipelineStage.REVIEW,
]

DEEP_PROCESSING_STAGES: list[PipelineStage] = [
    PipelineStage.IDEATION,
    PipelineStage.PLANNING,
    PipelineStage.SETUP,
    PipelineStage.CODING,
    PipelineStage.EXECUTION,
    PipelineStage.ANALYSIS,
    PipelineStage.FIGURE_GEN,
    PipelineStage.WRITING,
    PipelineStage.REVIEW,
]

DEEP_ONLY_STAGES: tuple[PipelineStage, ...] = (
    PipelineStage.SETUP,
    PipelineStage.CODING,
    PipelineStage.EXECUTION,
    PipelineStage.ANALYSIS,
)


def processing_stages_for_mode(
    mode: PipelineMode = PipelineMode.STANDARD,
) -> list[PipelineStage]:
    """Return the ordered working stages for the selected pipeline mode."""

    if mode == PipelineMode.DEEP:
        return list(DEEP_PROCESSING_STAGES)
    return list(STANDARD_PROCESSING_STAGES)


def _build_transitions(
    stages: list[PipelineStage],
) -> dict[PipelineStage, list[PipelineStage]]:
    transitions: dict[PipelineStage, list[PipelineStage]] = {
        PipelineStage.DONE: [],
        PipelineStage.FAILED: [],
    }
    ordered = [PipelineStage.INIT, *stages, PipelineStage.DONE]
    for current, nxt in zip(ordered, ordered[1:]):
        transitions[current] = [nxt, PipelineStage.FAILED]
    return transitions


def _merge_transitions(
    *transition_sets: dict[PipelineStage, list[PipelineStage]],
) -> dict[PipelineStage, list[PipelineStage]]:
    merged: dict[PipelineStage, list[PipelineStage]] = {
        stage: [] for stage in PipelineStage
    }
    for transitions in transition_sets:
        for stage, allowed in transitions.items():
            for target in allowed:
                if target not in merged[stage]:
                    merged[stage].append(target)
    return merged


STANDARD_STAGE_TRANSITIONS = _build_transitions(STANDARD_PROCESSING_STAGES)
DEEP_STAGE_TRANSITIONS = _build_transitions(DEEP_PROCESSING_STAGES)

# Combined transition table for schema/documentation purposes. Runtime
# validation should use the mode-specific tables above.
STAGE_TRANSITIONS: dict[PipelineStage, list[PipelineStage]] = _merge_transitions(
    STANDARD_STAGE_TRANSITIONS,
    DEEP_STAGE_TRANSITIONS,
)


StageStatus = Literal["pending", "running", "completed", "failed"]


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

    schema_version: str = "1.1"
    session_id: str
    topic: str
    pipeline_mode: PipelineMode = PipelineMode.STANDARD
    paper_mode: PaperMode = PaperMode.ORIGINAL_RESEARCH
    current_stage: PipelineStage = PipelineStage.INIT
    stages: dict[str, StageRecord] = Field(default_factory=dict)
    artifacts: list[ArtifactRecord] = Field(default_factory=list)
    config_snapshot: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
