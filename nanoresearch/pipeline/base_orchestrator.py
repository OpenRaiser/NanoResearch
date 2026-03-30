"""Base pipeline orchestrator — shared checkpoint/resume/retry logic.

Concrete subclasses (PipelineOrchestrator, DeepPipelineOrchestrator) only
need to supply agent construction, input wiring, and stage lists.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import traceback
from abc import ABC, abstractmethod
from typing import Any, Callable

from nanoresearch.agents.base import BaseResearchAgent
from nanoresearch.config import ResearchConfig
from nanoresearch.memory import ResearchMemory
from nanoresearch.pipeline.blueprint_validator import validate_blueprint
from nanoresearch.pipeline.reflection import reflect_on_stage, reflect_on_failure, REFLECTION_STAGES
from nanoresearch.pipeline.cost_tracker import CostTracker
from nanoresearch.pipeline.events import EventEmitter
from nanoresearch.pipeline.progress import ProgressEmitter
from nanoresearch.pipeline.state import PipelineStateMachine
from nanoresearch.pipeline.workspace import Workspace
from nanoresearch.schemas.manifest import PipelineMode, PipelineStage

logger = logging.getLogger(__name__)

# Retry backoff settings (centralised in constants.py)
from nanoresearch.agents.constants import (
    RETRY_BACKOFF_FACTOR,
    RETRY_BASE_DELAY,
    RETRY_MAX_DELAY,
)

# Progress callback type: (stage_name, status, message)
ProgressCallback = Callable[[str, str, str], None]


class BaseOrchestrator(ABC):
    """Abstract base for pipeline orchestrators with checkpoint/resume."""

    # --- subclass must override these class attributes ---
    _STAGE_KEY_MAP: dict[PipelineStage, str] = {}
    _OUTPUT_FILE_MAP: dict[PipelineStage, str] = {}
    _PIPELINE_MODE: PipelineMode | None = None  # None = STANDARD

    def __init__(
        self,
        workspace: Workspace,
        config: ResearchConfig,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self.workspace = workspace
        self.config = config
        self.progress_callback = progress_callback
        self.cost_tracker = CostTracker()
        self.progress_emitter = ProgressEmitter(workspace.path / "progress.json")

        self.state_machine = PipelineStateMachine(
            workspace.manifest.current_stage,
            mode=self._PIPELINE_MODE or PipelineMode.STANDARD,
        )

        self._agents: dict[PipelineStage, BaseResearchAgent] = self._build_agents()
        # Wire each agent's dispatcher to feed the cost tracker
        for agent in self._agents.values():
            agent._dispatcher._usage_callback = self.cost_tracker.record

        # Shared context for cross-stage hints
        self._shared_context: dict[str, Any] = {}

        # Typed event emitter (for advanced consumers)
        self.event_emitter = EventEmitter()

        # Cross-session memory
        self.memory = ResearchMemory()

    # ------------------------------------------------------------------
    # Abstract / hook methods — subclasses must / may override
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_agents(self) -> dict[PipelineStage, BaseResearchAgent]:
        """Create one agent per pipeline stage."""

    @abstractmethod
    def _get_processing_stages(self) -> list[PipelineStage]:
        """Return the ordered list of stages to execute."""

    @abstractmethod
    def _prepare_inputs(
        self,
        stage: PipelineStage,
        topic: str,
        accumulated: dict,
        last_error: str,
    ) -> dict[str, Any]:
        """Build the kwargs dict for ``agent.run()``."""

    def _get_initial_results(self, topic: str) -> dict[str, Any]:
        """Initial results dict.  Override to add pipeline_mode, etc."""
        return {"topic": topic}

    def _post_pipeline(self, results: dict[str, Any]) -> None:
        """Hook called after all stages complete and cost is saved.

        Default is no-op.  Deep pipeline overrides for export logic.
        """

    # ------------------------------------------------------------------
    # Shared logic
    # ------------------------------------------------------------------

    def _report_progress(self, stage: str, status: str, message: str) -> None:
        if self.progress_callback:
            try:
                self.progress_callback(stage, status, message)
            except Exception as exc:
                logger.debug("Progress callback error (non-fatal): %s", exc)

    def _report_substep(self, stage: str, message: str) -> None:
        """Report a sub-step update from an agent."""
        self._report_progress(stage, "substep", message)
        self.progress_emitter.substep(stage, message)

    async def close(self) -> None:
        for agent in self._agents.values():
            await agent.close()

    async def run(self, topic: str) -> dict[str, Any]:
        """Run the full pipeline from current stage to DONE."""
        mode_label = "DEEP" if self._PIPELINE_MODE == PipelineMode.DEEP else "standard"
        logger.info("Starting %s pipeline for topic: %s", mode_label, topic)
        logger.info("Current stage: %s", self.state_machine.current.value)

        # Ensure manifest records the pipeline mode (deep only)
        if (
            self._PIPELINE_MODE == PipelineMode.DEEP
            and self.workspace.manifest.pipeline_mode != PipelineMode.DEEP
        ):
            self.workspace.update_manifest(pipeline_mode=PipelineMode.DEEP)

        self._reset_stale_running_stages()

        results = self._get_initial_results(topic)

        try:
            stages = self._get_processing_stages()
            for stage_idx, stage in enumerate(stages):
                # Skip already-completed stages (for resume)
                stage_record = self.workspace.manifest.stages.get(stage.value)
                if stage_record and stage_record.status == "completed":
                    logger.info("Skipping completed stage: %s", stage.value)
                    self._report_progress(
                        stage.value, "skipped",
                        f"[{stage_idx+1}/{len(stages)}] {stage.value} already completed",
                    )
                    output = self._load_stage_output(stage, require=True)
                    results.update(output)
                    # BUG-21 fix: use force_set() with logging instead of
                    # directly mutating private _current attribute.
                    self.state_machine.force_set(stage)
                    continue

                # Skip stages configured to be skipped
                if stage.value in self.config.skip_stages:
                    logger.info("Skipping stage %s (configured in skip_stages)", stage.value)
                    if self.state_machine.current != stage:
                        if self.state_machine.can_transition(stage):
                            self.state_machine.transition(stage)
                        else:
                            logger.warning(
                                "Skipping stage %s from non-adjacent state %s; "
                                "forcing state machine to match",
                                stage.value,
                                self.state_machine.current.value,
                            )
                            self.state_machine.force_set(stage)
                    self.workspace.update_manifest(current_stage=stage)
                    self._report_progress(
                        stage.value, "skipped",
                        f"[{stage_idx+1}/{len(stages)}] {stage.value} skipped by config",
                    )
                    continue

                # Check transition
                if not self.state_machine.can_transition(stage):
                    if self.state_machine.current == stage:
                        pass  # resuming this stage
                    else:
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

                if self.state_machine.current != stage:
                    self.state_machine.transition(stage)

                self._report_progress(
                    stage.value, "started",
                    f"[{stage_idx+1}/{len(stages)}] Running {stage.value}...",
                )
                self.progress_emitter.stage_start(
                    stage.value, len(stages), stage_idx,
                    f"[{stage_idx+1}/{len(stages)}] Running {stage.value}...",
                )
                self.cost_tracker.set_stage(stage.value)

                # Run with retry
                t0 = time.monotonic()
                stage_result = await self._run_stage_with_retry(stage, topic, results)
                duration = time.monotonic() - t0
                logger.info("Stage %s completed in %.1fs", stage.value, duration)
                results.update(stage_result)

                # Cross-stage reference validation
                self._validate_cross_stage_refs(stage, results)

                # Stage reflection (after key stages)
                if stage.value in REFLECTION_STAGES:
                    try:
                        agent = self._agents[stage]
                        reflection = await reflect_on_stage(
                            stage.value, stage_result, results,
                            agent._dispatcher, agent.stage_config,
                        )
                        if reflection:
                            results[f"_reflection_{stage.value}"] = reflection
                            score = reflection.get("quality_score", 10)
                            if score < 5:
                                self._report_substep(
                                    stage.value,
                                    f"Reflection: quality={score}/10, {len(reflection.get('unmet_signals', []))} unmet signals",
                                )
                    except Exception as ref_exc:
                        logger.debug("Reflection error (non-fatal): %s", ref_exc)

                # Blueprint semantic validation after PLANNING
                if stage == PipelineStage.PLANNING:
                    bp = results.get("experiment_blueprint", {})
                    issues = validate_blueprint(bp)
                    if issues:
                        for issue in issues:
                            logger.warning("Blueprint issue: %s", issue)
                        self.progress_emitter.substep(
                            stage.value,
                            f"Blueprint validation: {len(issues)} issue(s) found",
                        )

                self._report_progress(
                    stage.value, "completed",
                    f"[{stage_idx+1}/{len(stages)}] {stage.value} completed",
                )
                self.progress_emitter.stage_complete(
                    stage.value, len(stages), stage_idx,
                    f"[{stage_idx+1}/{len(stages)}] {stage.value} completed in {duration:.1f}s",
                )

            # Mark pipeline as DONE
            self.state_machine.transition(PipelineStage.DONE)
            self.workspace.update_manifest(current_stage=PipelineStage.DONE)

            # Save cost summary
            cost_summary = self.cost_tracker.summary()
            self.workspace.write_json("logs/cost_summary.json", cost_summary)
            results["cost_summary"] = cost_summary
            if cost_summary["total_tokens"] > 0:
                logger.info(
                    "Cost summary: %d total tokens, %d calls, %.1fs total latency",
                    cost_summary["total_tokens"],
                    cost_summary["total_calls"],
                    cost_summary["total_latency_ms"] / 1000,
                )

            self.progress_emitter.pipeline_complete(
                True, f"{mode_label.capitalize()} pipeline completed successfully",
            )
            logger.info("%s pipeline completed!", mode_label.capitalize())

            self._post_pipeline(results)

            # Extract and persist memory from this run
            try:
                first_agent = next(iter(self._agents.values()), None)
                if first_agent:
                    await self.memory.extract_and_merge(results, first_agent._dispatcher)
            except Exception as mem_exc:
                logger.debug("Memory extraction failed (non-fatal): %s", mem_exc)

            return results
        except Exception:
            # Save cost summary even on failure so users can see token usage
            try:
                cost_summary = self.cost_tracker.summary()
                self.workspace.write_json("logs/cost_summary.json", cost_summary)
                if cost_summary["total_tokens"] > 0:
                    logger.info(
                        "Cost summary (on failure): %d total tokens, %d calls, %.1fs total latency",
                        cost_summary["total_tokens"],
                        cost_summary["total_calls"],
                        cost_summary["total_latency_ms"] / 1000,
                    )
            except Exception as cost_exc:
                logger.debug("Failed to save cost summary on failure: %s", cost_exc)
            self.progress_emitter.pipeline_complete(False, f"{mode_label.capitalize()} pipeline failed")
            raise

    async def _run_stage_with_retry(
        self, stage: PipelineStage, topic: str, accumulated: dict
    ) -> dict[str, Any]:
        """Run a stage with retry logic and failure reflection."""
        max_retries = self.config.max_retries
        last_error = ""
        # Accumulated failure context for progressively smarter retries
        _failure_reflections: list[dict] = []

        for attempt in range(max_retries + 1):
            try:
                self.workspace.mark_stage_running(stage)
                logger.info(
                    "Running %s (attempt %d/%d)",
                    stage.value, attempt + 1, max_retries + 1,
                )

                agent = self._agents[stage]
                # Wire substep callback so agent can report fine-grained progress
                agent._substep_callback = lambda msg, s=stage.value: self._report_substep(s, msg)
                # Share cross-stage context
                agent._shared_context = self._shared_context
                inputs = self._prepare_inputs(stage, topic, accumulated, last_error)

                # Inject failure reflection suggestions from previous attempt
                if _failure_reflections:
                    latest = _failure_reflections[-1]
                    suggestions = latest.get("suggestions", [])
                    if suggestions:
                        inputs["_retry_suggestions"] = suggestions
                        inputs["_failure_category"] = latest.get("category", "unknown")

                result = await agent.run(**inputs)

                self.workspace.mark_stage_completed(
                    stage, self._OUTPUT_FILE_MAP.get(stage, ""),
                )
                logger.info("Stage %s completed", stage.value)
                return self._wrap_stage_output(stage, result)

            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                tb = traceback.format_exc()
                logger.error("Stage %s failed: %s", stage.value, last_error)

                self.workspace.write_text(
                    f"logs/{stage.value.lower()}_error_{attempt}.txt",
                    f"Error: {last_error}\n\nTraceback:\n{tb}",
                )

                if attempt < max_retries:
                    # ── Failure reflection: analyze error before retrying ──
                    try:
                        agent = self._agents[stage]
                        self._report_progress(
                            stage.value, "retrying", "Analyzing failure...",
                        )
                        reflection = await reflect_on_failure(
                            stage.value, last_error, attempt + 1,
                            accumulated, agent._dispatcher, agent.stage_config,
                        )
                        if reflection:
                            _failure_reflections.append(reflection)
                            self.workspace.write_text(
                                f"logs/{stage.value.lower()}_reflection_{attempt}.json",
                                json.dumps(reflection, indent=2, ensure_ascii=False),
                            )
                            # Check if reflection says don't bother retrying
                            if not reflection.get("should_retry", True):
                                logger.warning(
                                    "Failure reflection says not to retry %s: %s",
                                    stage.value, reflection.get("reason", ""),
                                )
                                self.workspace.mark_stage_failed(stage, last_error)
                                self.state_machine.fail()
                                raise RuntimeError(
                                    f"Stage {stage.value} failed (reflection: unrecoverable): "
                                    f"{reflection.get('reason', last_error)}"
                                ) from e
                            # Include reflection suggestions in the error context
                            suggestions = reflection.get("suggestions", [])
                            if suggestions:
                                last_error += "\n\nRecovery suggestions:\n" + "\n".join(
                                    f"  - {s}" for s in suggestions
                                )
                    except RuntimeError:
                        raise  # Re-raise the "unrecoverable" RuntimeError
                    except Exception as ref_exc:
                        logger.debug("Failure reflection error (non-fatal): %s", ref_exc)

                    self.workspace.increment_retry(stage)
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

    def _wrap_stage_output(self, stage: PipelineStage, result: dict) -> dict[str, Any]:
        """Wrap agent output with a stage-specific key."""
        key = self._STAGE_KEY_MAP.get(stage, stage.value.lower())
        return {key: result}

    def _load_stage_output(
        self, stage: PipelineStage, *, require: bool = False
    ) -> dict[str, Any]:
        """Load previously saved output for a completed stage."""
        path = self._OUTPUT_FILE_MAP.get(stage)
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
        """Reset stages stuck in 'running' status back to 'pending'."""
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
        """Validate cross-stage references.  Logs warnings, never errors."""
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
