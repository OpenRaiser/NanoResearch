"""Deep 9-stage pipeline orchestrator."""

from __future__ import annotations

import logging
import shutil
from typing import Any

from nanoresearch.agents.analysis import AnalysisAgent
from nanoresearch.agents.base import BaseResearchAgent
from nanoresearch.agents.coding import CodingAgent
from nanoresearch.agents.execution import ExecutionAgent
from nanoresearch.agents.figure_gen import FigureAgent
from nanoresearch.agents.ideation import IdeationAgent
from nanoresearch.agents.planning import PlanningAgent
from nanoresearch.agents.review import ReviewAgent
from nanoresearch.agents.setup import SetupAgent
from nanoresearch.agents.writing import WritingAgent
from nanoresearch.pipeline.base_orchestrator import BaseOrchestrator
from nanoresearch.pipeline.state import PipelineStateMachine
from nanoresearch.schemas.manifest import PipelineMode, PipelineStage

logger = logging.getLogger(__name__)


class DeepPipelineOrchestrator(BaseOrchestrator):
    """Runs the deep 9-stage research pipeline with real experiments."""

    _STAGE_KEY_MAP: dict[PipelineStage, str] = {
        PipelineStage.IDEATION: "ideation_output",
        PipelineStage.PLANNING: "experiment_blueprint",
        PipelineStage.SETUP: "setup_output",
        PipelineStage.CODING: "coding_output",
        PipelineStage.EXECUTION: "execution_output",
        PipelineStage.ANALYSIS: "analysis_output",
        PipelineStage.FIGURE_GEN: "figure_gen_output",
        PipelineStage.WRITING: "writing_output",
        PipelineStage.REVIEW: "review_output",
    }

    _OUTPUT_FILE_MAP: dict[PipelineStage, str] = {
        PipelineStage.IDEATION: "papers/ideation_output.json",
        PipelineStage.PLANNING: "plans/experiment_blueprint.json",
        PipelineStage.SETUP: "plans/setup_output.json",
        PipelineStage.CODING: "plans/coding_output.json",
        PipelineStage.EXECUTION: "plans/execution_output.json",
        PipelineStage.ANALYSIS: "plans/analysis_output.json",
        PipelineStage.FIGURE_GEN: "drafts/figure_output.json",
        PipelineStage.WRITING: "drafts/paper_skeleton.json",
        PipelineStage.REVIEW: "drafts/review_output.json",
    }

    _PIPELINE_MODE = PipelineMode.DEEP

    def _build_agents(self) -> dict[PipelineStage, BaseResearchAgent]:
        return {
            PipelineStage.IDEATION: IdeationAgent(self.workspace, self.config),
            PipelineStage.PLANNING: PlanningAgent(self.workspace, self.config),
            PipelineStage.SETUP: SetupAgent(self.workspace, self.config),
            PipelineStage.CODING: CodingAgent(self.workspace, self.config),
            PipelineStage.EXECUTION: ExecutionAgent(self.workspace, self.config),
            PipelineStage.ANALYSIS: AnalysisAgent(self.workspace, self.config),
            PipelineStage.FIGURE_GEN: FigureAgent(self.workspace, self.config),
            PipelineStage.WRITING: WritingAgent(self.workspace, self.config),
            PipelineStage.REVIEW: ReviewAgent(self.workspace, self.config),
        }

    def _get_processing_stages(self) -> list[PipelineStage]:
        return PipelineStateMachine.processing_stages(PipelineMode.DEEP)

    def _get_initial_results(self, topic: str) -> dict[str, Any]:
        return {
            "topic": topic,
            "pipeline_mode": PipelineMode.DEEP.value,
        }

    def _post_pipeline(self, results: dict[str, Any]) -> None:
        """Export project and copy experiment code after successful completion."""
        try:
            export_path = self.workspace.export()
            logger.info("Exported project to: %s", export_path)
            results["export_path"] = str(export_path)

            exp_dir = self.workspace.path / "experiment"
            if exp_dir.exists():
                code_dest = export_path / "code"
                code_dest.mkdir(exist_ok=True)
                for file_path in (
                    list(exp_dir.glob("*.py"))
                    + list(exp_dir.glob("*.txt"))
                    + list(exp_dir.glob("*.slurm"))
                ):
                    shutil.copy2(file_path, code_dest / file_path.name)

                results_src = exp_dir / "results"
                if results_src.exists():
                    results_dest = export_path / "results"
                    results_dest.mkdir(exist_ok=True)
                    for file_path in results_src.iterdir():
                        if file_path.is_file() and file_path.suffix in (
                            ".json", ".csv", ".log",
                        ):
                            shutil.copy2(file_path, results_dest / file_path.name)
        except Exception as exc:
            logger.warning("Export failed (non-fatal): %s", exc)

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

        elif stage == PipelineStage.SETUP:
            inputs["topic"] = topic
            inputs["ideation_output"] = accumulated.get("ideation_output", {})
            inputs["experiment_blueprint"] = accumulated.get("experiment_blueprint", {})

        elif stage == PipelineStage.CODING:
            inputs["topic"] = topic
            inputs["experiment_blueprint"] = accumulated.get("experiment_blueprint", {})
            inputs["setup_output"] = accumulated.get("setup_output", {})

        elif stage == PipelineStage.EXECUTION:
            inputs["topic"] = topic
            inputs["coding_output"] = accumulated.get("coding_output", {})
            inputs["setup_output"] = accumulated.get("setup_output", {})
            inputs["experiment_blueprint"] = accumulated.get("experiment_blueprint", {})

        elif stage == PipelineStage.ANALYSIS:
            inputs["execution_output"] = accumulated.get("execution_output", {})
            inputs["experiment_blueprint"] = accumulated.get("experiment_blueprint", {})

        elif stage == PipelineStage.FIGURE_GEN:
            inputs["ideation_output"] = accumulated.get("ideation_output", {})
            inputs["experiment_blueprint"] = accumulated.get("experiment_blueprint", {})
            exec_output = accumulated.get("execution_output", {})
            analysis_output = accumulated.get("analysis_output", {})
            inputs["experiment_results"] = (
                exec_output.get("metrics")
                or analysis_output.get("execution_output", {}).get("metrics", {})
                or {}
            )
            inputs["experiment_analysis"] = analysis_output.get("analysis", {})
            inputs["experiment_summary"] = (
                analysis_output.get("experiment_summary")
                or exec_output.get("experiment_summary", "")
            )
            inputs["experiment_status"] = (
                exec_output.get("experiment_status")
                or exec_output.get("final_status", "pending")
            )

        elif stage == PipelineStage.WRITING:
            inputs["ideation_output"] = accumulated.get("ideation_output", {})
            inputs["experiment_blueprint"] = accumulated.get("experiment_blueprint", {})
            fig_output = accumulated.get("figure_gen_output", {})
            if not fig_output or not fig_output.get("figures"):
                analysis_figures = accumulated.get("analysis_output", {}).get("figures", {})
                fig_output = {"figures": analysis_figures} if analysis_figures else {}
            inputs["figure_output"] = fig_output
            inputs["template_format"] = self.config.template_format

            exec_output = accumulated.get("execution_output", {})
            analysis_output = accumulated.get("analysis_output", {})
            inputs["experiment_results"] = (
                exec_output.get("metrics")
                or analysis_output.get("execution_output", {}).get("metrics", {})
                or {}
            )
            inputs["experiment_analysis"] = analysis_output.get("analysis", {})
            inputs["experiment_summary"] = (
                analysis_output.get("experiment_summary")
                or exec_output.get("experiment_summary", "")
            )
            inputs["experiment_status"] = (
                exec_output.get("experiment_status")
                or exec_output.get("final_status", "pending")
            )

        elif stage == PipelineStage.REVIEW:
            writing_output = accumulated.get("writing_output", {})
            paper_tex = writing_output.get("paper_tex", "")
            if not paper_tex:
                tex_path = self.workspace.path / "drafts" / "paper.tex"
                if tex_path.exists():
                    paper_tex = tex_path.read_text(errors="replace")
            inputs["paper_tex"] = paper_tex
            inputs["ideation_output"] = accumulated.get("ideation_output", {})
            inputs["experiment_blueprint"] = accumulated.get("experiment_blueprint", {})
            exec_output = accumulated.get("execution_output", {})
            analysis_output = accumulated.get("analysis_output", {})
            inputs["experiment_results"] = (
                exec_output.get("metrics")
                or analysis_output.get("execution_output", {}).get("metrics", {})
                or {}
            )
            inputs["experiment_analysis"] = analysis_output.get("analysis", {})
            inputs["experiment_status"] = (
                exec_output.get("experiment_status")
                or exec_output.get("final_status", "pending")
            )
            inputs["writing_grounding"] = writing_output.get("grounding", {})

        if last_error:
            inputs["_retry_error"] = last_error

        return inputs
