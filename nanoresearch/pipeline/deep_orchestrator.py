"""Deep pipeline orchestrator — runs the full experiment-backed research flow."""

from __future__ import annotations

import asyncio
import logging
import traceback
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
from nanoresearch.config import ResearchConfig
from nanoresearch.pipeline.state import PipelineStateMachine
from nanoresearch.pipeline.workspace import Workspace
from nanoresearch.schemas.manifest import PipelineMode, PipelineStage

logger = logging.getLogger(__name__)


DEEP_PROCESSING_STAGES = PipelineStateMachine.processing_stages(PipelineMode.DEEP)


class DeepPipelineOrchestrator:
    """Runs the deep research pipeline with real setup, execution, and analysis."""

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

    def __init__(self, workspace: Workspace, config: ResearchConfig) -> None:
        self.workspace = workspace
        self.config = config
        self.state_machine = PipelineStateMachine(
            workspace.manifest.current_stage,
            mode=PipelineMode.DEEP,
        )
        self._agents: dict[PipelineStage, BaseResearchAgent] = {
            PipelineStage.IDEATION: IdeationAgent(workspace, config),
            PipelineStage.PLANNING: PlanningAgent(workspace, config),
            PipelineStage.SETUP: SetupAgent(workspace, config),
            PipelineStage.CODING: CodingAgent(workspace, config),
            PipelineStage.EXECUTION: ExecutionAgent(workspace, config),
            PipelineStage.ANALYSIS: AnalysisAgent(workspace, config),
            PipelineStage.FIGURE_GEN: FigureAgent(workspace, config),
            PipelineStage.WRITING: WritingAgent(workspace, config),
            PipelineStage.REVIEW: ReviewAgent(workspace, config),
        }

    async def close(self) -> None:
        for agent in self._agents.values():
            await agent.close()

    async def run(self, topic: str) -> dict[str, Any]:
        """Run the full deep pipeline."""

        if self.workspace.manifest.pipeline_mode != PipelineMode.DEEP:
            self.workspace.update_manifest(pipeline_mode=PipelineMode.DEEP)

        logger.info("Starting DEEP pipeline for topic: %s", topic)
        logger.info("Current stage: %s", self.state_machine.current.value)

        self._reset_stale_running_stages()

        results: dict[str, Any] = {
            "topic": topic,
            "pipeline_mode": PipelineMode.DEEP.value,
        }

        for stage in DEEP_PROCESSING_STAGES:
            stage_record = self.workspace.manifest.stages.get(stage.value)
            if stage_record and stage_record.status == "completed":
                logger.info("Skipping completed stage: %s", stage.value)
                results.update(self._load_stage_output(stage, require=True))
                # Advance state machine so subsequent stages can transition.
                # Use _current directly: transition() would reject if the
                # manifest's current_stage put us out-of-order after a crash.
                self.state_machine._current = stage
                continue

            if not self.state_machine.can_transition(stage):
                if self.state_machine.current == stage:
                    pass
                else:
                    prior = self._load_stage_output(stage)
                    if prior:
                        results.update(prior)
                        logger.info(
                            "Loaded prior output for skipped deep stage %s",
                            stage.value,
                        )
                    else:
                        logger.warning(
                            "Skipping deep stage %s (no transition from %s) and no prior output found",
                            stage.value,
                            self.state_machine.current.value,
                        )
                    continue

            if self.state_machine.current != stage:
                self.state_machine.transition(stage)

            stage_result = await self._run_stage_with_retry(stage, topic, results)
            results.update(stage_result)

        self.state_machine.transition(PipelineStage.DONE)
        self.workspace.update_manifest(current_stage=PipelineStage.DONE)
        logger.info("Deep pipeline completed!")

        try:
            export_path = self.workspace.export()
            logger.info("Exported project to: %s", export_path)
            results["export_path"] = str(export_path)

            exp_dir = self.workspace.path / "experiment"
            if exp_dir.exists():
                import shutil

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
                            ".json",
                            ".csv",
                            ".log",
                        ):
                            shutil.copy2(file_path, results_dest / file_path.name)
        except Exception as exc:
            logger.warning("Export failed (non-fatal): %s", exc)

        return results

    async def _run_stage_with_retry(
        self,
        stage: PipelineStage,
        topic: str,
        accumulated: dict,
    ) -> dict[str, Any]:
        """Run one deep stage with retry logic."""

        max_retries = self.config.max_retries
        last_error = ""

        for attempt in range(max_retries + 1):
            try:
                self.workspace.mark_stage_running(stage)
                logger.info(
                    "Running %s (attempt %d/%d)",
                    stage.value,
                    attempt + 1,
                    max_retries + 1,
                )

                agent = self._agents[stage]
                inputs = self._prepare_inputs(stage, topic, accumulated, last_error)
                result = await agent.run(**inputs)

                self.workspace.mark_stage_completed(
                    stage,
                    self._OUTPUT_FILE_MAP.get(stage, ""),
                )
                logger.info("Stage %s completed", stage.value)
                return self._wrap_stage_output(stage, result)

            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                tb = traceback.format_exc()
                logger.error("Stage %s failed: %s", stage.value, last_error)

                self.workspace.write_text(
                    f"logs/{stage.value.lower()}_error_{attempt}.txt",
                    f"Error: {last_error}\n\nTraceback:\n{tb}",
                )

                if attempt < max_retries:
                    self.workspace.increment_retry(stage)
                    delay = 3.0 * (2.0 ** attempt)  # exponential backoff
                    logger.info("Retrying %s in %.0fs...", stage.value, delay)
                    await asyncio.sleep(delay)
                else:
                    self.workspace.mark_stage_failed(stage, last_error)
                    raise RuntimeError(
                        f"Stage {stage.value} failed after {max_retries + 1} attempts: {last_error}"
                    ) from exc

        raise RuntimeError("Unreachable")

    def _prepare_inputs(
        self,
        stage: PipelineStage,
        topic: str,
        accumulated: dict,
        last_error: str,
    ) -> dict[str, Any]:
        """Prepare inputs for each deep stage."""

        inputs: dict[str, Any] = {}

        if stage == PipelineStage.IDEATION:
            inputs["topic"] = topic

        elif stage == PipelineStage.PLANNING:
            inputs["ideation_output"] = accumulated.get("ideation_output", {})

        elif stage == PipelineStage.SETUP:
            inputs["topic"] = topic
            inputs["ideation_output"] = accumulated.get("ideation_output", {})
            inputs["experiment_blueprint"] = accumulated.get(
                "experiment_blueprint",
                {},
            )

        elif stage == PipelineStage.CODING:
            inputs["topic"] = topic
            inputs["experiment_blueprint"] = accumulated.get(
                "experiment_blueprint",
                {},
            )
            inputs["setup_output"] = accumulated.get("setup_output", {})

        elif stage == PipelineStage.EXECUTION:
            inputs["topic"] = topic
            inputs["coding_output"] = accumulated.get("coding_output", {})
            inputs["setup_output"] = accumulated.get("setup_output", {})
            inputs["experiment_blueprint"] = accumulated.get(
                "experiment_blueprint",
                {},
            )

        elif stage == PipelineStage.ANALYSIS:
            inputs["execution_output"] = accumulated.get("execution_output", {})
            inputs["experiment_blueprint"] = accumulated.get(
                "experiment_blueprint",
                {},
            )

        elif stage == PipelineStage.FIGURE_GEN:
            inputs["ideation_output"] = accumulated.get("ideation_output", {})
            inputs["experiment_blueprint"] = accumulated.get(
                "experiment_blueprint",
                {},
            )
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
            inputs["experiment_blueprint"] = accumulated.get(
                "experiment_blueprint",
                {},
            )
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
            inputs["experiment_blueprint"] = accumulated.get(
                "experiment_blueprint",
                {},
            )
            # Pass grounding metadata so review can protect real results
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

    def _wrap_stage_output(
        self,
        stage: PipelineStage,
        result: dict,
    ) -> dict[str, Any]:
        """Wrap agent output with a stage-specific key."""

        key = self._STAGE_KEY_MAP.get(stage, stage.value.lower())
        return {key: result}

    def _load_stage_output(
        self,
        stage: PipelineStage,
        *,
        require: bool = False,
    ) -> dict[str, Any]:
        """Load previously saved output for resume."""

        path = self._OUTPUT_FILE_MAP.get(stage)
        if path:
            try:
                data = self.workspace.read_json(path)
                key = self._STAGE_KEY_MAP.get(stage, stage.value.lower())
                return {key: data}
            except FileNotFoundError:
                if require:
                    raise RuntimeError(
                        f"Stage {stage.value} is marked completed but output file '{path}' is missing."
                    )
        return {}

    def _reset_stale_running_stages(self) -> None:
        """Reset stages left in running status by a previous crash."""

        manifest = self.workspace.manifest
        changed = False
        for stage_key, record in manifest.stages.items():
            if record.status == "running":
                logger.warning(
                    "Stage %s was left in 'running' status (likely from a crash). Resetting to 'pending'.",
                    stage_key,
                )
                record.status = "pending"
                record.error_message = ""
                changed = True
        if changed:
            self.workspace._write_manifest(manifest)
