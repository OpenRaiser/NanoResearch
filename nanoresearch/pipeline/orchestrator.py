"""Main pipeline orchestrator — checkpoint/resume/retry logic."""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
from typing import Any, Callable

from nanoresearch.agents.base import BaseResearchAgent
from nanoresearch.agents.ideation import IdeationAgent
from nanoresearch.agents.planning import PlanningAgent
from nanoresearch.agents.experiment import ExperimentAgent
from nanoresearch.agents.figure_gen import FigureAgent
from nanoresearch.agents.writing import WritingAgent
from nanoresearch.agents.review import ReviewAgent
from nanoresearch.config import ResearchConfig
from nanoresearch.pipeline.state import PipelineStateMachine
from nanoresearch.pipeline.workspace import Workspace
from nanoresearch.schemas.manifest import PipelineStage

logger = logging.getLogger(__name__)

# Retry backoff settings
RETRY_BASE_DELAY = 5.0   # seconds
RETRY_MAX_DELAY = 60.0   # seconds
RETRY_BACKOFF_FACTOR = 2.0

# Progress callback type: (stage_name, status, message)
ProgressCallback = Callable[[str, str, str], None]


class PipelineOrchestrator:
    """Runs the full research pipeline with checkpoint/resume support."""

    def __init__(
        self,
        workspace: Workspace,
        config: ResearchConfig,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self.workspace = workspace
        self.config = config
        self.progress_callback = progress_callback
        self.state_machine = PipelineStateMachine(workspace.manifest.current_stage)
        self._agents: dict[PipelineStage, BaseResearchAgent] = {
            PipelineStage.IDEATION: IdeationAgent(workspace, config),
            PipelineStage.PLANNING: PlanningAgent(workspace, config),
            PipelineStage.EXPERIMENT: ExperimentAgent(workspace, config),
            PipelineStage.FIGURE_GEN: FigureAgent(workspace, config),
            PipelineStage.WRITING: WritingAgent(workspace, config),
            PipelineStage.REVIEW: ReviewAgent(workspace, config),
        }

    def _report_progress(self, stage: str, status: str, message: str) -> None:
        """Report progress via callback if registered."""
        if self.progress_callback:
            try:
                self.progress_callback(stage, status, message)
            except Exception as exc:
                logger.debug("Progress callback error (non-fatal): %s", exc)

    async def close(self) -> None:
        for agent in self._agents.values():
            await agent.close()

    async def run(self, topic: str) -> dict[str, Any]:
        """Run the full pipeline from current stage to DONE."""
        logger.info("Starting pipeline for topic: %s", topic)
        logger.info("Current stage: %s", self.state_machine.current.value)

        # P0: Reset any "running" stages to "pending" on resume.
        # A crash during execution leaves stages as "running" — they must be re-run.
        self._reset_stale_running_stages()

        results: dict[str, Any] = {"topic": topic}

        try:
            stages = PipelineStateMachine.processing_stages()
            for stage_idx, stage in enumerate(stages):
                # Skip already-completed stages (for resume)
                stage_record = self.workspace.manifest.stages.get(stage.value)
                if stage_record and stage_record.status == "completed":
                    logger.info("Skipping completed stage: %s", stage.value)
                    self._report_progress(
                        stage.value, "skipped",
                        f"[{stage_idx+1}/{len(stages)}] {stage.value} already completed",
                    )
                    # Load previous output — raise if file is missing
                    output = self._load_stage_output(stage, require=True)
                    results.update(output)
                    continue

                # Skip stages configured to be skipped
                if stage.value in self.config.skip_stages:
                    logger.info("Skipping stage %s (configured in skip_stages)", stage.value)
                    self._report_progress(
                        stage.value, "skipped",
                        f"[{stage_idx+1}/{len(stages)}] {stage.value} skipped by config",
                    )
                    continue

                # Check if we've reached the current stage
                if not self.state_machine.can_transition(stage):
                    if self.state_machine.current == stage:
                        pass  # we're resuming this stage
                    else:
                        # Load any available output for skipped-but-not-completed stages
                        prior = self._load_stage_output(stage)
                        if prior:
                            results.update(prior)
                            logger.info("Loaded prior output for skipped stage %s", stage.value)
                        else:
                            logger.warning(
                                "Skipping stage %s (no transition from %s) and no prior output found",
                                stage.value, self.state_machine.current.value,
                            )
                        continue

                # Transition to this stage
                if self.state_machine.current != stage:
                    self.state_machine.transition(stage)

                self._report_progress(
                    stage.value, "started",
                    f"[{stage_idx+1}/{len(stages)}] Running {stage.value}...",
                )

                # Run with retry
                t0 = time.monotonic()
                stage_result = await self._run_stage_with_retry(stage, topic, results)
                duration = time.monotonic() - t0
                logger.info("Stage %s completed in %.1fs", stage.value, duration)
                results.update(stage_result)

                # P0: Cross-stage reference validation after each stage
                self._validate_cross_stage_refs(stage, results)

                self._report_progress(
                    stage.value, "completed",
                    f"[{stage_idx+1}/{len(stages)}] {stage.value} completed",
                )

            # Mark pipeline as DONE
            self.state_machine.transition(PipelineStage.DONE)
            self.workspace.update_manifest(current_stage=PipelineStage.DONE)
            logger.info("Pipeline completed successfully!")
            return results
        except Exception:
            raise  # let caller (CLI) handle close()

    async def _run_stage_with_retry(
        self, stage: PipelineStage, topic: str, accumulated: dict
    ) -> dict[str, Any]:
        """Run a stage with retry logic."""
        max_retries = self.config.max_retries
        last_error = ""

        for attempt in range(max_retries + 1):
            try:
                self.workspace.mark_stage_running(stage)
                logger.info(
                    "Running %s (attempt %d/%d)",
                    stage.value, attempt + 1, max_retries + 1,
                )

                agent = self._agents[stage]
                inputs = self._prepare_inputs(stage, topic, accumulated, last_error)
                result = await agent.run(**inputs)

                self.workspace.mark_stage_completed(stage)
                logger.info("Stage %s completed", stage.value)
                return self._wrap_stage_output(stage, result)

            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                tb = traceback.format_exc()
                logger.error("Stage %s failed: %s", stage.value, last_error)

                # Save error log
                self.workspace.write_text(
                    f"logs/{stage.value.lower()}_error_{attempt}.txt",
                    f"Error: {last_error}\n\nTraceback:\n{tb}",
                )

                if attempt < max_retries:
                    self.workspace.increment_retry(stage)
                    # Exponential backoff with cap
                    delay = min(
                        RETRY_BASE_DELAY * (RETRY_BACKOFF_FACTOR ** attempt),
                        RETRY_MAX_DELAY,
                    )
                    logger.info(
                        "Retrying %s in %.0fs (attempt %d/%d)...",
                        stage.value, delay, attempt + 2, max_retries + 1,
                    )
                    self._report_progress(
                        stage.value, "retrying",
                        f"Retrying in {delay:.0f}s...",
                    )
                    await asyncio.sleep(delay)
                else:
                    self.workspace.mark_stage_failed(stage, last_error)
                    self.state_machine.fail()
                    raise RuntimeError(
                        f"Stage {stage.value} failed after {max_retries + 1} attempts: {last_error}"
                    ) from e

        raise RuntimeError("Unreachable")  # pragma: no cover

    def _prepare_inputs(
        self,
        stage: PipelineStage,
        topic: str,
        accumulated: dict,
        last_error: str,
    ) -> dict[str, Any]:
        """Prepare inputs for an agent based on accumulated results."""
        inputs: dict[str, Any] = {}

        if stage == PipelineStage.IDEATION:
            inputs["topic"] = topic

        elif stage == PipelineStage.PLANNING:
            inputs["ideation_output"] = accumulated.get("ideation_output", {})

        elif stage == PipelineStage.EXPERIMENT:
            inputs["experiment_blueprint"] = accumulated.get("experiment_blueprint", {})
            # Pass reference repos from ideation for grounded code generation
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
            # Read the paper.tex written by the Writing stage
            try:
                paper_tex = self.workspace.read_text("drafts/paper.tex")
            except FileNotFoundError:
                paper_tex = ""
            inputs["paper_tex"] = paper_tex
            inputs["ideation_output"] = accumulated.get("ideation_output", {})
            inputs["experiment_blueprint"] = accumulated.get("experiment_blueprint", {})

        # Inject error context for retries
        if last_error:
            inputs["_retry_error"] = last_error

        return inputs

    # Mapping from stage to the key used in accumulated results dict
    _STAGE_KEY_MAP: dict[PipelineStage, str] = {
        PipelineStage.IDEATION: "ideation_output",
        PipelineStage.PLANNING: "experiment_blueprint",
        PipelineStage.EXPERIMENT: "experiment_output",
        PipelineStage.FIGURE_GEN: "figure_output",
        PipelineStage.WRITING: "writing_output",
        PipelineStage.REVIEW: "review_output",
    }

    def _wrap_stage_output(self, stage: PipelineStage, result: dict) -> dict[str, Any]:
        """Wrap agent output with a stage-specific key."""
        key = self._STAGE_KEY_MAP.get(stage, stage.value.lower())
        return {key: result}

    def _load_stage_output(
        self, stage: PipelineStage, *, require: bool = False
    ) -> dict[str, Any]:
        """Load previously saved output for a completed stage.

        Args:
            stage: The pipeline stage to load output for.
            require: If True, raise RuntimeError when a completed stage's
                     output file is missing (P0 fix for silent data loss).
        """
        file_map = {
            PipelineStage.IDEATION: "papers/ideation_output.json",
            PipelineStage.PLANNING: "plans/experiment_blueprint.json",
            PipelineStage.EXPERIMENT: "logs/experiment_output.json",
            PipelineStage.FIGURE_GEN: "drafts/figure_output.json",
            PipelineStage.WRITING: "drafts/paper_skeleton.json",
            PipelineStage.REVIEW: "drafts/review_output.json",
        }
        path = file_map.get(stage)
        if path:
            try:
                data = self.workspace.read_json(path)
                key = self._STAGE_KEY_MAP.get(stage, stage.value.lower())
                return {key: data}
            except FileNotFoundError:
                if require:
                    raise RuntimeError(
                        f"Stage {stage.value} is marked completed but output "
                        f"file '{path}' is missing. The workspace may be "
                        f"corrupted. Delete the workspace and re-run, or "
                        f"manually reset the stage status in manifest.json."
                    )
                logger.warning(
                    "Stage %s marked completed but output file %s not found",
                    stage.value, path,
                )
        return {}

    def _reset_stale_running_stages(self) -> None:
        """Reset stages stuck in 'running' status back to 'pending'.

        When a pipeline crashes mid-execution, stages may remain as 'running'.
        On resume, these must be reset so they are re-executed.
        """
        manifest = self.workspace.manifest
        changed = False
        for stage_key, record in manifest.stages.items():
            if record.status == "running":
                logger.warning(
                    "Stage %s was left in 'running' status (likely from a crash). "
                    "Resetting to 'pending' for re-execution.",
                    stage_key,
                )
                record.status = "pending"
                record.error_message = ""
                changed = True
        if changed:
            self.workspace._write_manifest(manifest)

    def _validate_cross_stage_refs(
        self, stage: PipelineStage, results: dict[str, Any]
    ) -> None:
        """Validate that cross-stage references are consistent.

        Logs warnings (not errors) for mismatches — the pipeline continues
        but operators are alerted to potential data integrity issues.
        """
        if stage == PipelineStage.PLANNING:
            blueprint = results.get("experiment_blueprint", {})
            ideation = results.get("ideation_output", {})
            hyp_ref = blueprint.get("hypothesis_ref", "")
            if hyp_ref and ideation:
                hyp_ids = {
                    h.get("hypothesis_id", "")
                    for h in ideation.get("hypotheses", [])
                }
                if hyp_ref not in hyp_ids:
                    logger.warning(
                        "Cross-ref mismatch: blueprint.hypothesis_ref=%r "
                        "not found in ideation hypotheses %s",
                        hyp_ref, hyp_ids,
                    )

        elif stage == PipelineStage.EXPERIMENT:
            exp_out = results.get("experiment_output", {})
            blueprint = results.get("experiment_blueprint", {})
            # Validate that experiment metrics match blueprint metrics
            bp_metrics = {
                m.get("name", "") for m in blueprint.get("metrics", [])
            }
            if bp_metrics and exp_out:
                for entry in exp_out.get("experiment_results", {}).get("main_results", []):
                    for metric in entry.get("metrics", []):
                        mname = metric.get("metric_name", "")
                        if mname and mname not in bp_metrics:
                            logger.warning(
                                "Cross-ref mismatch: experiment metric %r "
                                "not defined in blueprint metrics %s",
                                mname, bp_metrics,
                            )
