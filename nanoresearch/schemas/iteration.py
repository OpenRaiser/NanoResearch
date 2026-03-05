"""Pydantic data models for experiment iteration loops."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class ExperimentHypothesis(BaseModel):
    """A hypothesis for a single iteration round."""

    round_number: int
    hypothesis: str
    planned_changes: list[str] = Field(default_factory=list)
    expected_signal: str = ""
    rationale: str = ""

    @field_validator("hypothesis", "expected_signal", "rationale", mode="before")
    @classmethod
    def _coerce_to_str(cls, v):
        if isinstance(v, list):
            return "; ".join(str(x) for x in v) if v else ""
        if isinstance(v, dict):
            return str(v)
        return v if isinstance(v, str) else str(v) if v is not None else ""


class PreflightResult(BaseModel):
    """Result of a single preflight check."""

    check_name: str
    status: str  # "passed" / "failed" / "warning"
    message: str
    details: dict = Field(default_factory=dict)


class PreflightReport(BaseModel):
    """Aggregated preflight check report."""

    overall_status: str  # "passed" / "failed" / "warnings"
    checks: list[PreflightResult] = Field(default_factory=list)
    blocking_failures: list[str] = Field(default_factory=list)


class TrainingDynamics(BaseModel):
    """Heuristic analysis of training behavior from training_log."""

    convergence_speed: str = "normal"  # "fast" / "normal" / "slow" / "not_converging"
    overfitting_detected: bool = False
    train_val_gap: float | None = None
    loss_stability: str = "stable"  # "stable" / "noisy" / "diverging"
    final_train_loss: float | None = None
    final_val_loss: float | None = None


class FeedbackAnalysis(BaseModel):
    """Structured feedback from one iteration round."""

    metric_summary: dict[str, float] = Field(default_factory=dict)
    improvement_delta: dict[str, float] = Field(default_factory=dict)
    error_categories: list[str] = Field(default_factory=list)
    training_dynamics: TrainingDynamics = Field(default_factory=TrainingDynamics)
    attribution: str = ""  # "data_issue" / "training_strategy" / "architecture" / "hyperparameter" / "implementation_bug"
    recommended_action: str = ""
    should_continue: bool = False
    termination_reason: str | None = None  # "target_met" / "plateau" / "max_rounds" / "degradation"

    @field_validator("attribution", "recommended_action", mode="before")
    @classmethod
    def _coerce_to_str(cls, v):
        if isinstance(v, list):
            return "; ".join(str(x) for x in v) if v else ""
        if isinstance(v, dict):
            return str(v)
        return v if isinstance(v, str) else str(v) if v is not None else ""


class RoundResult(BaseModel):
    """Complete record for one iteration round."""

    round_number: int
    hypothesis: ExperimentHypothesis
    preflight: PreflightReport = Field(default_factory=lambda: PreflightReport(overall_status="skipped"))
    execution_status: str = "pending"
    quick_eval_status: str = "pending"
    metrics: dict = Field(default_factory=dict)
    analysis: FeedbackAnalysis | None = None
    files_modified: list[str] = Field(default_factory=list)


class IterationState(BaseModel):
    """Top-level state tracking across all iteration rounds."""

    max_rounds: int = 3
    rounds: list[RoundResult] = Field(default_factory=list)
    best_round: int | None = None
    best_metrics: dict = Field(default_factory=dict)
    final_status: str = "in_progress"  # "target_met" / "plateau" / "max_rounds" / "failed"
