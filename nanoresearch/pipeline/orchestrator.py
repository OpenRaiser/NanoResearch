"""Standard 6-stage pipeline orchestrator."""

from __future__ import annotations

from typing import Any

from nanoresearch.agents.base import BaseResearchAgent
from nanoresearch.agents.experiment import ExperimentAgent
from nanoresearch.agents.figure_gen import FigureAgent
from nanoresearch.agents.ideation import IdeationAgent
from nanoresearch.agents.planning import PlanningAgent
from nanoresearch.agents.review import ReviewAgent
from nanoresearch.agents.writing import WritingAgent
from nanoresearch.pipeline.base_orchestrator import (
    BaseOrchestrator,
    ProgressCallback,  # re-export for backward compatibility
)
from nanoresearch.pipeline.state import PipelineStateMachine
from nanoresearch.schemas.manifest import PipelineStage

# Re-export so existing ``from nanoresearch.pipeline.orchestrator import ProgressCallback`` keeps working.
__all__ = ["PipelineOrchestrator", "ProgressCallback"]


class PipelineOrchestrator(BaseOrchestrator):
    """Runs the standard 6-stage research pipeline."""

    _STAGE_KEY_MAP: dict[PipelineStage, str] = {
        PipelineStage.IDEATION: "ideation_output",
        PipelineStage.PLANNING: "experiment_blueprint",
        PipelineStage.EXPERIMENT: "experiment_output",
        PipelineStage.FIGURE_GEN: "figure_output",
        PipelineStage.WRITING: "writing_output",
        PipelineStage.REVIEW: "review_output",
    }

    _OUTPUT_FILE_MAP: dict[PipelineStage, str] = {
        PipelineStage.IDEATION: "papers/ideation_output.json",
        PipelineStage.PLANNING: "plans/experiment_blueprint.json",
        PipelineStage.EXPERIMENT: "logs/experiment_output.json",
        PipelineStage.FIGURE_GEN: "drafts/figure_output.json",
        PipelineStage.WRITING: "drafts/paper_skeleton.json",
        PipelineStage.REVIEW: "drafts/review_output.json",
    }

    # Standard pipeline uses default mode (None → STANDARD)
    _PIPELINE_MODE = None

    def _build_agents(self) -> dict[PipelineStage, BaseResearchAgent]:
        return {
            PipelineStage.IDEATION: IdeationAgent(self.workspace, self.config),
            PipelineStage.PLANNING: PlanningAgent(self.workspace, self.config),
            PipelineStage.EXPERIMENT: ExperimentAgent(self.workspace, self.config),
            PipelineStage.FIGURE_GEN: FigureAgent(self.workspace, self.config),
            PipelineStage.WRITING: WritingAgent(self.workspace, self.config),
            PipelineStage.REVIEW: ReviewAgent(self.workspace, self.config),
        }

    def _get_processing_stages(self) -> list[PipelineStage]:
        return PipelineStateMachine.processing_stages()

    def _prepare_inputs(
        self,
        stage: PipelineStage,
        topic: str,
        accumulated: dict,
        last_error: str,
    ) -> dict[str, Any]:
        inputs: dict[str, Any] = {}

        if stage == PipelineStage.IDEATION:
            inputs["topic"] = topic

        elif stage == PipelineStage.PLANNING:
            inputs["ideation_output"] = accumulated.get("ideation_output", {})

        elif stage == PipelineStage.EXPERIMENT:
            inputs["experiment_blueprint"] = accumulated.get("experiment_blueprint", {})
            ideation = accumulated.get("ideation_output", {})
            inputs["reference_repos"] = ideation.get("reference_repos", [])

        elif stage == PipelineStage.FIGURE_GEN:
            inputs["experiment_blueprint"] = accumulated.get("experiment_blueprint", {})
            inputs["ideation_output"] = accumulated.get("ideation_output", {})
            exp_out = accumulated.get("experiment_output", {})
            inputs["experiment_results"] = exp_out.get("experiment_results", {})
            inputs["experiment_status"] = exp_out.get("experiment_status", "pending")

        elif stage == PipelineStage.WRITING:
            inputs["ideation_output"] = accumulated.get("ideation_output", {})
            inputs["experiment_blueprint"] = accumulated.get("experiment_blueprint", {})
            inputs["figure_output"] = accumulated.get("figure_output", {})
            inputs["template_format"] = self.config.template_format
            exp_out = accumulated.get("experiment_output", {})
            inputs["experiment_results"] = exp_out.get("experiment_results", {})
            inputs["experiment_status"] = exp_out.get("experiment_status", "pending")

        elif stage == PipelineStage.REVIEW:
            try:
                paper_tex = self.workspace.read_text("drafts/paper.tex")
            except FileNotFoundError:
                paper_tex = ""
            inputs["paper_tex"] = paper_tex
            inputs["ideation_output"] = accumulated.get("ideation_output", {})
            inputs["experiment_blueprint"] = accumulated.get("experiment_blueprint", {})

        if last_error:
            inputs["_retry_error"] = last_error

        return inputs
