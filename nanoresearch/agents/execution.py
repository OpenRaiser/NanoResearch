"""Execution agent — submits SLURM jobs, monitors progress, debugs failures, collects results."""

from __future__ import annotations

import asyncio
import csv
import gzip
import json
import logging
import os
import platform
import re
import shlex
import shutil
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Any

from nanoresearch.agents.base import BaseResearchAgent
from nanoresearch.agents.debug import DebugAgent, MAX_DEBUG_ROUNDS
from nanoresearch.agents.experiment import ExperimentAgent
from nanoresearch.agents.feedback_analyzer import FeedbackAnalyzer
from nanoresearch.agents.project_runner import (
    RUNNER_CONFIG_NAME,
    RUNNER_SCRIPT_NAME,
    ensure_project_runner,
    is_python_launcher_token,
    normalize_target_spec,
    repair_launch_contract,
    refresh_project_runner_script,
    validate_launch_contract,
)
from nanoresearch.agents.preflight import PreflightChecker
from nanoresearch.agents.repair_journal import (
    REPAIR_SNAPSHOT_JOURNAL_PATH,
    append_snapshot_journal,
    capture_repair_snapshot,
    rollback_snapshot,
)
from nanoresearch.agents.runtime_env import ExperimentExecutionPolicy, RuntimeEnvironmentManager
from nanoresearch.schemas.iteration import ExperimentHypothesis, IterationState, RoundResult
from nanoresearch.schemas.manifest import PipelineStage

logger = logging.getLogger(__name__)

# Poll interval and max wait time for SLURM jobs
POLL_INTERVAL = 30  # seconds
MAX_WAIT_TIME = 7 * 24 * 3600  # 7 days — real training can run for days
LOCAL_EXECUTION_CHECKPOINT = "plans/execution_iteration_checkpoint.json"
REMEDIATION_LEDGER_PATH = "logs/execution_remediation_ledger.json"
RESOURCE_SUCCESS_STATUSES = {"downloaded", "full", "config_only"}
RESULT_CONTRACT_CRASH_INDICATORS = (
    "RuntimeError",
    "Error(s) in loading",
    "Traceback",
    "CUDA out of memory",
    "OOM",
    "Killed",
    "Exception",
    "FileNotFoundError",
    "ModuleNotFoundError",
)
MODULE_PACKAGE_ALIASES = {
    "cv2": "opencv-python",
    "pil": "Pillow",
    "yaml": "PyYAML",
    "sklearn": "scikit-learn",
    "bio": "biopython",
}
QUICK_EVAL_AUTO_OPTIONS = {
    "--quick-eval",
    "--epochs",
    "--num-epochs",
    "--max-steps",
    "--steps",
    "--batch-size",
    "--batch_size",
    "--num-workers",
    "--num_workers",
    "--workers",
    "--subset-size",
    "--subset_size",
    "--train-size",
    "--quick-eval-train-size",
    "--limit-train-batches",
    "--limit-val-batches",
}


class ExecutionAgent(BaseResearchAgent):
    """Submits SLURM training jobs, monitors them, debugs failures, and collects results."""

    stage = PipelineStage.EXECUTION

    @property
    def stage_config(self):
        """Reuse experiment-stage model routing for execution-time reasoning."""
        return self.config.for_stage("experiment")

    async def run(self, **inputs: Any) -> dict[str, Any]:
        coding_output: dict = inputs.get("coding_output", {})
        experiment_blueprint: dict = inputs.get("experiment_blueprint", {})
        setup_output: dict = inputs.get("setup_output", {})
        topic: str = inputs.get("topic", "")

        code_dir = Path(coding_output.get("code_dir", ""))
        slurm_script = coding_output.get("slurm_script", "")

        if not code_dir.exists():
            raise RuntimeError(f"Code directory not found: {code_dir}")

        self.log(f"Starting execution in: {code_dir}")
        remediation_ledger: list[dict[str, Any]] = []

        # Create logs directory
        (code_dir / "logs").mkdir(exist_ok=True)
        (code_dir / "results").mkdir(exist_ok=True)

        cluster_available = bool(slurm_script) and shutil.which("sbatch") is not None
        if not self.config.prefers_cluster_execution() or not cluster_available:
            if self.config.prefers_cluster_execution() and not cluster_available:
                self.log("Cluster execution requested but sbatch is unavailable, falling back to local mode")
            elif not slurm_script:
                self.log("No SLURM script produced by CODING, falling back to local mode")
            else:
                self.log(f"Execution profile '{self.config.execution_profile.value}' prefers local execution")
            final_result = await self._run_local_mode(
                code_dir,
                coding_output,
                experiment_blueprint,
                setup_output,
                topic,
                remediation_ledger=remediation_ledger,
            )
            self.workspace.write_json("plans/execution_output.json", final_result)
            return final_result

        # Pre-flight: fix common SLURM issues before first submission
        debug_agent = DebugAgent(self.workspace, self.config)
        preflight_fixed = debug_agent._fix_common_slurm_issues(code_dir)
        if preflight_fixed:
            self.log("Pre-flight: fixed common SLURM script issues")
            self._append_remediation_entry(
                remediation_ledger,
                kind="slurm_preflight_fix",
                status="applied",
                scope="cluster_preflight",
                details={"code_dir": str(code_dir)},
            )

        # Pre-flight: local syntax/import check before wasting SLURM queue time
        local_ok, local_err = await self._local_preflight(code_dir)
        if not local_ok:
            self.log(f"Pre-flight import check failed, fixing before submission")
            # Run a mini debug loop locally (no SLURM submission)
            for pre_round in range(MAX_DEBUG_ROUNDS):
                debug_result = await debug_agent.run(
                    code_dir=str(code_dir),
                    stdout_log="",
                    stderr_log=local_err,
                    job_status="IMPORT_ERROR",
                    debug_round=pre_round + 1,
                    previous_fixes=[],
                )
                if not debug_result.get("needs_resubmit", False):
                    break
                self._append_remediation_entry(
                    remediation_ledger,
                    kind="cluster_preflight_debug_fix",
                    status="applied",
                    scope="cluster_preflight",
                    cycle=pre_round + 1,
                    files=list(debug_result.get("fixed_files", []) or []),
                    details={
                        "diagnosis": debug_result.get("diagnosis", ""),
                        "patches": list(debug_result.get("patches", []) or []),
                    },
                )
                local_ok, local_err = await self._local_preflight(code_dir)
                if local_ok:
                    self.log(f"Pre-flight fixed after {pre_round + 1} round(s)")
                    break

        # Debug loop: submit → monitor → if failed, debug & retry
        previous_fixes: list[dict] = []
        final_result = None

        for debug_round in range(MAX_DEBUG_ROUNDS + 1):
            # On first round, check for existing job from a previous run (resume)
            existing = await self._find_existing_job(code_dir) if debug_round == 0 else None
            if existing:
                job_id, existing_status = existing
                self.log(f"Found existing SLURM job {job_id} (status: {existing_status})")
                if existing_status == "COMPLETED":
                    final_status = "COMPLETED"
                else:  # RUNNING or PENDING
                    final_status = await self._monitor_job(job_id, code_dir)
                    self.log(f"Existing job {job_id} finished: {final_status}")
            else:
                # Submit new SLURM job
                job_id = await self._submit_job(slurm_script)
                self.log(f"Submitted SLURM job: {job_id}")
                # Monitor job until completion
                final_status = await self._monitor_job(job_id, code_dir)
                self.log(f"Job {job_id} finished with status: {final_status}")

            # Collect results
            results = await self._collect_results(code_dir, job_id, final_status)
            self.log(f"Collected results: {list(results.keys())}")
            recovered_source = str(results.get("recovered_from") or "").strip()
            if recovered_source and (
                recovered_source == "slurm_logs" or results.get("metrics_artifact_materialized")
            ):
                self._append_remediation_entry(
                    remediation_ledger,
                    kind="metrics_recovery",
                    status="applied",
                    scope="cluster_collect",
                    cycle=debug_round + 1,
                    details={
                        "source": recovered_source,
                        "job_id": job_id,
                        **(
                            {
                                "artifact_path": results.get("metrics_artifact_path", ""),
                                "artifact_materialized": True,
                            }
                            if results.get("metrics_artifact_materialized")
                            else {}
                        ),
                    },
                )
                if results.get("metrics_artifact_materialized"):
                    snapshot_entry = self.consume_last_mutation_snapshot_entry()
                    details = {
                        "source": recovered_source,
                        "job_id": job_id,
                        "artifact_path": str(results.get("metrics_artifact_path", "")),
                    }
                    if snapshot_entry:
                        details.update({
                            "snapshot_entry_id": snapshot_entry.get("entry_id"),
                            "snapshot_count": snapshot_entry.get("snapshot_count", 0),
                            "snapshot_journal_path": REPAIR_SNAPSHOT_JOURNAL_PATH,
                            "snapshots": list(snapshot_entry.get("snapshots", []) or []),
                        })
                    self._append_remediation_entry(
                        remediation_ledger,
                        kind="metrics_artifact_recovery",
                        status="applied",
                        scope="cluster_collect",
                        cycle=debug_round + 1,
                        files=[str(results.get("metrics_artifact_path", ""))],
                        details=details,
                    )

            if final_status != "COMPLETED":
                cluster_resume_fix = self._attempt_cluster_resume_repair(
                    code_dir,
                    final_status,
                    results,
                    setup_output,
                    scope="cluster_resume",
                )
                cluster_resume_snapshot_entry = self.consume_last_mutation_snapshot_entry()
                if cluster_resume_fix:
                    self.log(
                        "Applied deterministic cluster resume repair: "
                        f"{cluster_resume_fix}; resubmitting job"
                    )
                    details = None
                    if cluster_resume_snapshot_entry:
                        details = {
                            "snapshot_entry_id": cluster_resume_snapshot_entry.get("entry_id"),
                            "snapshot_count": cluster_resume_snapshot_entry.get("snapshot_count", 0),
                            "snapshot_journal_path": REPAIR_SNAPSHOT_JOURNAL_PATH,
                            "snapshots": list(cluster_resume_snapshot_entry.get("snapshots", []) or []),
                        }
                    self._append_remediation_entry(
                        remediation_ledger,
                        kind="resume_repair",
                        status="applied",
                        scope="cluster_resume",
                        cycle=debug_round + 1,
                        files=list(cluster_resume_fix),
                        details={
                            **(details or {}),
                            "job_id": job_id,
                            "job_status": final_status,
                        },
                    )
                    continue

            metrics = results.get("metrics") or {}
            execution_status = "success" if final_status == "COMPLETED" else "failed"
            result_contract = self._evaluate_experiment_contract(
                results,
                execution_backend="cluster",
                execution_status=execution_status,
                quick_eval_status="skipped",
                final_status=final_status,
            )
            experiment_status = str(result_contract.get("status", "failed"))
            self._append_remediation_entry(
                remediation_ledger,
                kind="result_contract_validation",
                status=experiment_status,
                scope="cluster_result",
                cycle=debug_round + 1,
                details={
                    "success_path": result_contract.get("success_path", ""),
                    "missing_signals": list(result_contract.get("missing_signals", []) or []),
                    "failure_signals": list(result_contract.get("failure_signals", []) or []),
                },
            )

            final_result = {
                "job_id": job_id,
                "execution_backend": "cluster",
                "runtime_env": {
                    "kind": "cluster",
                    "profile": self.config.execution_profile.value,
                    "partition": self.config.slurm_partition,
                },
                "remediation_ledger": list(remediation_ledger),
                "remediation_ledger_path": REMEDIATION_LEDGER_PATH,
                "repair_snapshot_journal_path": self._repair_snapshot_journal_path(),
                "final_status": final_status,
                "code_dir": str(code_dir),
                "debug_rounds": debug_round,
                "execution_status": execution_status,
                "quick_eval_status": "skipped",
                "experiment_status": experiment_status,
                "result_contract": result_contract,
                "experiment_results": metrics,
                **results,
            }

            # If job succeeded or we've exhausted debug rounds, stop
            if final_status == "COMPLETED":
                if experiment_status in {"success", "partial"}:
                    self.log(
                        f"Job completed with result contract status {experiment_status} "
                        f"after {debug_round} debug round(s)"
                    )
                    break
                self.log(
                    "Job exited with code 0 but failed the explicit result contract. "
                    f"Missing={result_contract.get('missing_signals', [])}, "
                    f"failure_signals={result_contract.get('failure_signals', [])}"
                )
                final_status = "FAILED"
                final_result["final_status"] = "FAILED"
                final_result["experiment_status"] = "failed"
                final_result["result_contract"]["status"] = "failed"
                # Fall through to debug loop

            if debug_round >= MAX_DEBUG_ROUNDS:
                self.log(f"Max debug rounds ({MAX_DEBUG_ROUNDS}) reached, giving up")
                break

            # Job failed — enter debug loop
            self.log(f"Job failed, entering debug round {debug_round + 1}/{MAX_DEBUG_ROUNDS}")

            try:
                debug_result = await debug_agent.run(
                    code_dir=str(code_dir),
                    stdout_log=results.get("stdout_log", ""),
                    stderr_log=results.get("stderr_log", ""),
                    job_status=final_status,
                    debug_round=debug_round + 1,
                    previous_fixes=previous_fixes,
                )

                if not debug_result.get("needs_resubmit", False):
                    self.log("Debug agent determined no fix is possible, stopping")
                    break

                previous_fixes.append({
                    "diagnosis": debug_result.get("diagnosis", ""),
                    "patches": debug_result.get("patches", []),
                    "fixed_files": debug_result.get("fixed_files", []),
                })
                self._append_remediation_entry(
                    remediation_ledger,
                    kind="cluster_debug_fix",
                    status="applied",
                    scope="cluster_debug",
                    cycle=debug_round + 1,
                    files=list(debug_result.get("fixed_files", []) or []),
                    details={
                        "diagnosis": debug_result.get("diagnosis", ""),
                        "patches": list(debug_result.get("patches", []) or []),
                        "job_status": final_status,
                    },
                )

                self.log(f"Debug round {debug_round + 1}: fixed {debug_result.get('fixed_files', [])}, resubmitting...")

            except Exception as e:
                self.log(f"Debug agent failed: {e}")
                break

        await debug_agent.close()

        if final_result is None:
            final_result = {
                "job_id": "",
                "execution_backend": "cluster",
                "runtime_env": {
                    "kind": "cluster",
                    "profile": self.config.execution_profile.value,
                    "partition": self.config.slurm_partition,
                },
                "final_status": "FAILED",
                "code_dir": str(code_dir),
                "debug_rounds": 0,
                "execution_status": "failed",
                "quick_eval_status": "skipped",
                "experiment_status": "failed",
                "experiment_results": {},
                "repair_snapshot_journal_path": self._repair_snapshot_journal_path(),
            }

        final_result["remediation_ledger"] = list(remediation_ledger)
        final_result["remediation_ledger_path"] = self._persist_remediation_ledger(remediation_ledger)
        final_result["repair_snapshot_journal_path"] = self._repair_snapshot_journal_path()
        self.workspace.write_json("plans/execution_output.json", final_result)
        return final_result

    async def _run_local_mode(
        self,
        code_dir: Path,
        coding_output: dict[str, Any],
        experiment_blueprint: dict[str, Any],
        setup_output: dict[str, Any],
        topic: str,
        remediation_ledger: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        runner_script = code_dir / RUNNER_SCRIPT_NAME
        entry_train_command = str(
            coding_output.get("entry_train_command")
            or coding_output.get("train_command")
            or ""
        ).strip()
        if not runner_script.exists() and RUNNER_SCRIPT_NAME not in entry_train_command:
            runner_assets = ensure_project_runner(code_dir, entry_train_command)
            coding_output = {**coding_output, **runner_assets, "train_command": runner_assets["runner_command"]}
            self.log("Injected deterministic execution runner for compatibility")
        elif runner_script.exists():
            refreshed_runner = refresh_project_runner_script(code_dir)
            if refreshed_runner:
                self.log("Refreshed deterministic execution runner to latest template")
                self._append_remediation_entry(
                    remediation_ledger,
                    kind="runner_refresh",
                    status="applied",
                    scope="local_runner",
                    files=list(refreshed_runner),
                )

        runtime_manager = RuntimeEnvironmentManager(self.config, self.log)
        runtime_env = await runtime_manager.prepare(code_dir)
        self._record_runtime_env_ledger(runtime_env, remediation_ledger)
        runtime_python = str(runtime_env.get("python", "python"))
        execution_policy = runtime_manager.build_execution_policy(code_dir)
        helper = ExperimentAgent(self.workspace, self.config)
        analyzer = FeedbackAnalyzer(self.config, self._dispatcher)
        base_command = self._build_local_command(code_dir, coding_output, runtime_python)
        blueprint_summary = self._build_execution_blueprint_summary(
            topic,
            experiment_blueprint,
            setup_output,
            coding_output,
        )
        max_rounds = max(
            1,
            1 if self.config.execution_profile.value == "fast_draft" else self.config.experiment_max_rounds,
        )
        iteration_state = IterationState(max_rounds=max_rounds)
        iteration_state, start_round = helper._load_iteration_checkpoint(
            iteration_state,
            LOCAL_EXECUTION_CHECKPOINT,
        )
        round_artifacts: dict[int, dict[str, Any]] = {}
        last_analysis = iteration_state.rounds[-1].analysis if iteration_state.rounds else None
        latest_hypothesis = ExperimentHypothesis(
            round_number=1,
            hypothesis="Validate generated deep-pipeline experiment locally",
            planned_changes=[],
            expected_signal="Dry-run passes and quick-eval produces metrics",
            rationale="Use the generated code as baseline before iterative repair.",
        )

        try:
            for round_num in range(start_round, max_rounds + 1):
                self.log(f"=== Local iteration round {round_num}/{max_rounds} ===")
                files_modified: list[str] = []

                if round_num > 1:
                    history_summary = helper._build_history_summary(iteration_state.rounds)
                    preflight_error_ctx = ""
                    if last_analysis and last_analysis.recommended_action:
                        preflight_error_ctx = (
                            "The previous round recommended this action:\n"
                            f"{last_analysis.recommended_action}\n"
                        )
                    latest_hypothesis = await helper._generate_iteration_hypothesis(
                        last_analysis,
                        history_summary,
                        blueprint_summary,
                        preflight_error_ctx=preflight_error_ctx,
                        code_dir=code_dir,
                    )
                    if latest_hypothesis.hypothesis == "__NO_NEW_IDEAS__":
                        iteration_state.final_status = "no_new_ideas"
                        self.log("Iteration loop exhausted new ideas, stopping")
                        break

                    files_modified = await helper._apply_iteration_changes(
                        latest_hypothesis,
                        code_dir,
                        runtime_python,
                    )
                    if not files_modified and latest_hypothesis.planned_changes:
                        self.log("Search-replace matched nothing, retrying with full-file rewrite")
                        files_modified = await helper._apply_iteration_changes_fullwrite(
                            latest_hypothesis,
                            code_dir,
                        )

                preflight = PreflightChecker(code_dir).run_all()
                self.workspace.write_json(
                    f"logs/execution_round_{round_num}_preflight.json",
                    preflight.model_dump(),
                )

                if preflight.overall_status == "failed":
                    error_message = "\n".join(preflight.blocking_failures)
                    if preflight.suggested_fixes:
                        error_message += (
                            "\nSuggested fixes:\n- " + "\n- ".join(preflight.suggested_fixes[:8])
                        )
                    analysis = await analyzer.analyze(
                        current_round=round_num,
                        metrics={},
                        previous_rounds=iteration_state.rounds,
                        stderr_snippet=error_message[:1000],
                        max_rounds=max_rounds,
                    )
                    round_result = RoundResult(
                        round_number=round_num,
                        hypothesis=latest_hypothesis,
                        preflight=preflight,
                        execution_status="skipped",
                        quick_eval_status="skipped",
                        metrics={},
                        analysis=analysis,
                        files_modified=files_modified,
                    )
                    iteration_state.rounds.append(round_result)
                    helper._save_iteration_checkpoint(iteration_state, LOCAL_EXECUTION_CHECKPOINT)
                    last_analysis = analysis
                    if not analysis.should_continue:
                        iteration_state.final_status = analysis.termination_reason or "preflight_failed"
                        break
                    continue

                launch_contract = validate_launch_contract(base_command, code_dir)
                launch_contract_repair: dict[str, Any] = {
                    "status": "skipped",
                    "actions": [],
                    "files_modified": [],
                    "command": list(base_command),
                    "initial_contract": launch_contract,
                    "final_contract": launch_contract,
                }
                if launch_contract.get("status") == "failed":
                    launch_contract_repair = repair_launch_contract(base_command, code_dir)
                    self._record_launch_contract_repair_ledger(
                        launch_contract_repair,
                        remediation_ledger,
                        round_number=round_num,
                    )
                    repaired_command = launch_contract_repair.get("command")
                    if isinstance(repaired_command, list) and repaired_command:
                        base_command = [str(token) for token in repaired_command]
                    final_contract = launch_contract_repair.get("final_contract")
                    if isinstance(final_contract, dict):
                        launch_contract = final_contract
                    else:
                        launch_contract = validate_launch_contract(base_command, code_dir)

                self._record_launch_contract_ledger(
                    launch_contract,
                    remediation_ledger,
                    round_number=round_num,
                )
                self.workspace.write_json(
                    f"logs/execution_round_{round_num}_launch_contract.json",
                    launch_contract,
                )
                self.workspace.write_json(
                    f"logs/execution_round_{round_num}_launch_contract_repair.json",
                    launch_contract_repair,
                )
                if launch_contract.get("status") == "failed":
                    error_lines = list(launch_contract.get("failures", []) or [])[:8]
                    warning_lines = list(launch_contract.get("warnings", []) or [])[:8]
                    error_message = "\n".join(error_lines) if error_lines else "Launch contract failed"
                    repair_actions = list(launch_contract_repair.get("actions", []) or [])
                    if repair_actions:
                        error_message += (
                            "\nRepair actions attempted:\n- "
                            + "\n- ".join(str(action) for action in repair_actions[:6])
                        )
                    if warning_lines:
                        error_message += "\nWarnings:\n- " + "\n- ".join(warning_lines)
                    analysis = await analyzer.analyze(
                        current_round=round_num,
                        metrics={},
                        previous_rounds=iteration_state.rounds,
                        stderr_snippet=error_message[:1000],
                        max_rounds=max_rounds,
                    )
                    round_result = RoundResult(
                        round_number=round_num,
                        hypothesis=latest_hypothesis,
                        preflight=preflight,
                        execution_status="skipped",
                        quick_eval_status="skipped",
                        metrics={},
                        analysis=analysis,
                        files_modified=files_modified,
                    )
                    iteration_state.rounds.append(round_result)
                    round_artifacts[round_num] = {
                        "launch_contract": launch_contract,
                        "launch_contract_repair": launch_contract_repair,
                        "execution": {"status": "skipped", "stderr": error_message},
                        "quick_eval": {"status": "skipped", "metrics": {}},
                    }
                    helper._save_iteration_checkpoint(iteration_state, LOCAL_EXECUTION_CHECKPOINT)
                    last_analysis = analysis
                    if not analysis.should_continue:
                        iteration_state.final_status = analysis.termination_reason or "launch_contract_failed"
                        break
                    continue

                execution = await self._run_local_dry_run_loop(
                    code_dir,
                    base_command,
                    blueprint_summary,
                    helper,
                    resource_context=setup_output,
                    runtime_python=runtime_python,
                    execution_policy=execution_policy,
                    remediation_ledger=remediation_ledger,
                    round_number=round_num,
                )
                execution_status = execution.get("status", "failed")
                quick_eval = {"status": "skipped", "metrics": {}}
                if execution_status in ("success", "fixed"):
                    quick_eval = await self._run_local_quick_eval_loop(
                        code_dir,
                        base_command,
                        blueprint_summary,
                        helper,
                        resource_context=setup_output,
                        runtime_python=runtime_python,
                        execution_policy=execution_policy,
                        remediation_ledger=remediation_ledger,
                        round_number=round_num,
                    )

                self.workspace.write_json(
                    f"logs/execution_round_{round_num}_execution.json",
                    execution,
                )
                self.workspace.write_json(
                    f"logs/execution_round_{round_num}_quick_eval.json",
                    quick_eval,
                )
                round_artifacts[round_num] = {
                    "launch_contract": launch_contract,
                    "launch_contract_repair": launch_contract_repair,
                    "execution": execution,
                    "quick_eval": quick_eval,
                }

                stderr_snippet = quick_eval.get("stderr", "") or execution.get("stderr", "")
                analysis = await analyzer.analyze(
                    current_round=round_num,
                    metrics=quick_eval.get("metrics", {}),
                    previous_rounds=iteration_state.rounds,
                    stderr_snippet=str(stderr_snippet)[:1000],
                    max_rounds=max_rounds,
                )

                round_result = RoundResult(
                    round_number=round_num,
                    hypothesis=latest_hypothesis,
                    preflight=preflight,
                    execution_status=execution_status,
                    quick_eval_status=quick_eval.get("status", "skipped"),
                    metrics=quick_eval.get("metrics", {}),
                    analysis=analysis,
                    files_modified=files_modified,
                )
                iteration_state.rounds.append(round_result)
                self._update_best_round(iteration_state, analysis)
                self.workspace.write_json(
                    f"logs/execution_round_{round_num}.json",
                    round_result.model_dump(),
                )
                helper._save_iteration_checkpoint(iteration_state, LOCAL_EXECUTION_CHECKPOINT)
                last_analysis = analysis

                self.log(
                    f"Round {round_num}: execution={execution_status}, "
                    f"quick_eval={quick_eval.get('status', 'skipped')}, "
                    f"continue={analysis.should_continue}"
                )
                if not analysis.should_continue:
                    iteration_state.final_status = analysis.termination_reason or "completed"
                    break
            else:
                iteration_state.final_status = "max_rounds"

            best_round_data = helper._get_best_round(iteration_state)
            best_round_number = iteration_state.best_round or (
                iteration_state.rounds[-1].round_number if iteration_state.rounds else None
            )
            best_artifact = (
                round_artifacts.get(best_round_number or -1)
                or self._load_local_round_artifacts(best_round_number)
            )
            execution = best_artifact.get("execution", {})
            quick_eval = best_artifact.get("quick_eval", {})
            artifact_results = self._collect_result_artifacts(code_dir)
            metrics = best_round_data.get("metrics") or quick_eval.get("metrics") or artifact_results.get("metrics", {})
            stdout_log = str(quick_eval.get("stdout") or execution.get("stdout") or "")[-10000:]
            stderr_log = str(quick_eval.get("stderr") or execution.get("stderr") or "")[-5000:]
            parsed_metrics = self._parse_metrics_from_log(stdout_log) if stdout_log else {}
            result_contract = self._evaluate_experiment_contract(
                {
                    **artifact_results,
                    "metrics": metrics,
                    "parsed_metrics": parsed_metrics,
                    "stdout_log": stdout_log,
                    "stderr_log": stderr_log,
                    "recovered_from": quick_eval.get("recovered_from", "")
                    or artifact_results.get("recovered_from", ""),
                },
                execution_backend="local",
                execution_status=best_round_data.get("execution_status", "failed"),
                quick_eval_status=best_round_data.get("quick_eval_status", "failed"),
                final_status="COMPLETED" if best_round_data.get("quick_eval_status") in ("success", "partial") else "FAILED",
            )
            experiment_status = str(result_contract.get("status", "failed"))
            final_status = "COMPLETED" if experiment_status in {"success", "partial"} else "FAILED"
            self._append_remediation_entry(
                remediation_ledger,
                kind="result_contract_validation",
                status=experiment_status,
                scope="local_final",
                round_number=best_round_number,
                details={
                    "success_path": result_contract.get("success_path", ""),
                    "missing_signals": list(result_contract.get("missing_signals", []) or []),
                    "failure_signals": list(result_contract.get("failure_signals", []) or []),
                },
            )

            final_result = {
                "job_id": "local",
                "execution_backend": "local",
                "runtime_env": runtime_env,
                "remediation_ledger": list(remediation_ledger or []),
                "remediation_ledger_path": self._persist_remediation_ledger(remediation_ledger),
                "repair_snapshot_journal_path": self._repair_snapshot_journal_path(),
                "command": base_command,
                "code_dir": str(code_dir),
                "debug_rounds": max(0, len(iteration_state.rounds) - 1),
                "final_status": final_status,
                "execution_status": best_round_data.get("execution_status", "failed"),
                "quick_eval_status": best_round_data.get("quick_eval_status", "failed"),
                "experiment_status": experiment_status,
                "result_contract": result_contract,
                "launch_contract": best_artifact.get("launch_contract", {}),
                "launch_contract_repair": best_artifact.get("launch_contract_repair", {}),
                "metrics": metrics,
                "parsed_metrics": parsed_metrics,
                "experiment_results": metrics,
                "stdout_log": stdout_log,
                "stderr_log": stderr_log,
                "iteration_state": iteration_state.model_dump(),
                "experiment_summary": self._summarize_local_iteration(
                    iteration_state,
                    experiment_blueprint,
                ),
                **artifact_results,
            }
            final_result["metrics"] = metrics
            final_result["experiment_results"] = metrics
            return final_result
        finally:
            await helper.close()

    async def _run_local_dry_run_loop(
        self,
        code_dir: Path,
        base_command: list[str],
        blueprint_summary: str,
        helper: ExperimentAgent,
        resource_context: dict[str, Any] | None = None,
        runtime_python: str = "python",
        execution_policy: ExperimentExecutionPolicy | None = None,
        remediation_ledger: list[dict[str, Any]] | None = None,
        round_number: int | None = None,
    ) -> dict[str, Any]:
        """Run dry-run with iterative batch-fix cycles."""
        max_fix_cycles = 5
        last_result: dict[str, Any] = {}
        fix_history: list[dict[str, Any]] = []

        for cycle in range(1, max_fix_cycles + 1):
            result = await self._run_subprocess(
                self._command_with_mode(base_command, "--dry-run"),
                cwd=code_dir,
                timeout=120,
            )
            last_result = result
            if result["returncode"] == 0:
                status = "success" if cycle == 1 else "fixed"
                return {"status": status, "attempts": cycle, **result}

            if cycle >= max_fix_cycles:
                break

            repair_text = self._repair_error_text(result)
            signature = self._repair_error_signature(result)
            repeat_count = self._repair_repeat_count(fix_history, signature)
            deterministic_fix = self._attempt_resource_path_repair(
                code_dir,
                repair_text,
                resource_context,
                scope="local_dry_run",
            )
            snapshot_entry = self.consume_last_mutation_snapshot_entry()
            if deterministic_fix:
                self.log(
                    f"Applied deterministic resource-path repair during dry-run: {deterministic_fix}"
                )
                details = None
                if snapshot_entry:
                    details = {
                        "snapshot_entry_id": snapshot_entry.get("entry_id"),
                        "snapshot_count": snapshot_entry.get("snapshot_count", 0),
                        "snapshot_journal_path": REPAIR_SNAPSHOT_JOURNAL_PATH,
                        "snapshots": list(snapshot_entry.get("snapshots", []) or []),
                    }
                self._append_remediation_entry(
                    remediation_ledger,
                    kind="resource_path_repair",
                    status="applied",
                    scope="local_dry_run",
                    round_number=round_number,
                    cycle=cycle,
                    signature=signature,
                    files=list(deterministic_fix),
                    details=details,
                )
                self._record_repair_attempt(
                    fix_history,
                    signature,
                    repair_text,
                    cycle,
                    deterministic_fix,
                )
                continue
            option_value_fix = self._attempt_option_value_repair(
                code_dir,
                repair_text,
                resource_context,
                scope="local_dry_run",
            )
            option_value_snapshot_entry = self.consume_last_mutation_snapshot_entry()
            if option_value_fix:
                self.log(
                    "Applied deterministic option-value repair during dry-run: "
                    f"{option_value_fix}"
                )
                details = None
                if option_value_snapshot_entry:
                    details = {
                        "snapshot_entry_id": option_value_snapshot_entry.get("entry_id"),
                        "snapshot_count": option_value_snapshot_entry.get("snapshot_count", 0),
                        "snapshot_journal_path": REPAIR_SNAPSHOT_JOURNAL_PATH,
                        "snapshots": list(option_value_snapshot_entry.get("snapshots", []) or []),
                    }
                self._append_remediation_entry(
                    remediation_ledger,
                    kind="option_value_repair",
                    status="applied",
                    scope="local_dry_run",
                    round_number=round_number,
                    cycle=cycle,
                    signature=signature,
                    files=list(option_value_fix),
                    details=details,
                )
                self._record_repair_attempt(
                    fix_history,
                    signature,
                    repair_text,
                    cycle,
                    option_value_fix,
                )
                continue
            unknown_arg_fix = self._attempt_unrecognized_argument_repair(
                code_dir,
                repair_text,
                mode="dry-run",
                scope="local_dry_run",
            )
            unknown_arg_snapshot_entry = self.consume_last_mutation_snapshot_entry()
            if unknown_arg_fix:
                self.log(
                    "Applied deterministic unrecognized-argument repair during dry-run: "
                    f"{unknown_arg_fix}"
                )
                details = None
                if unknown_arg_snapshot_entry:
                    details = {
                        "snapshot_entry_id": unknown_arg_snapshot_entry.get("entry_id"),
                        "snapshot_count": unknown_arg_snapshot_entry.get("snapshot_count", 0),
                        "snapshot_journal_path": REPAIR_SNAPSHOT_JOURNAL_PATH,
                        "snapshots": list(unknown_arg_snapshot_entry.get("snapshots", []) or []),
                    }
                self._append_remediation_entry(
                    remediation_ledger,
                    kind="unrecognized_argument_repair",
                    status="applied",
                    scope="local_dry_run",
                    round_number=round_number,
                    cycle=cycle,
                    signature=signature,
                    files=list(unknown_arg_fix),
                    details=details,
                )
                self._record_repair_attempt(
                    fix_history,
                    signature,
                    repair_text,
                    cycle,
                    unknown_arg_fix,
                )
                continue
            required_arg_fix = self._attempt_required_argument_repair(
                code_dir,
                repair_text,
                resource_context,
                scope="local_dry_run",
            )
            required_arg_snapshot_entry = self.consume_last_mutation_snapshot_entry()
            if required_arg_fix:
                self.log(
                    "Applied deterministic required-argument repair during dry-run: "
                    f"{required_arg_fix}"
                )
                details = None
                if required_arg_snapshot_entry:
                    details = {
                        "snapshot_entry_id": required_arg_snapshot_entry.get("entry_id"),
                        "snapshot_count": required_arg_snapshot_entry.get("snapshot_count", 0),
                        "snapshot_journal_path": REPAIR_SNAPSHOT_JOURNAL_PATH,
                        "snapshots": list(required_arg_snapshot_entry.get("snapshots", []) or []),
                    }
                self._append_remediation_entry(
                    remediation_ledger,
                    kind="required_argument_repair",
                    status="applied",
                    scope="local_dry_run",
                    round_number=round_number,
                    cycle=cycle,
                    signature=signature,
                    files=list(required_arg_fix),
                    details=details,
                )
                self._record_repair_attempt(
                    fix_history,
                    signature,
                    repair_text,
                    cycle,
                    required_arg_fix,
                )
                continue
            runtime_fix = await self._attempt_runtime_remediation(
                code_dir,
                repair_text,
                runtime_python=runtime_python,
                fix_history=fix_history,
                execution_policy=execution_policy,
                remediation_ledger=remediation_ledger,
                mode="dry-run",
                cycle=cycle,
                signature=signature,
                round_number=round_number,
            )
            if runtime_fix:
                self.log(
                    f"Applied deterministic runtime remediation during dry-run: {runtime_fix}"
                )
                self._record_repair_attempt(
                    fix_history,
                    signature,
                    repair_text,
                    cycle,
                    runtime_fix,
                )
                continue
            modified = await helper._batch_fix_errors(
                code_dir,
                repair_text,
                blueprint_summary,
                mode="dry-run",
                previous_fixes=[dict(entry) for entry in fix_history],
                extra_context=self._build_repair_context(
                    code_dir,
                    result,
                    mode="dry-run",
                    repeat_count=repeat_count,
                    resource_context=resource_context,
                ),
            )
            self._record_repair_attempt(
                fix_history,
                signature,
                repair_text,
                cycle,
                modified,
            )
            self._append_remediation_entry(
                remediation_ledger,
                kind="llm_batch_fix",
                status="applied" if modified else "skipped",
                scope="local_dry_run",
                round_number=round_number,
                cycle=cycle,
                signature=signature,
                reason="" if modified else "no_files_modified",
                files=list(modified or []),
            )
            if not modified:
                break

        return {"status": "failed", "attempts": cycle, **last_result}

    async def _run_local_quick_eval_loop(
        self,
        code_dir: Path,
        base_command: list[str],
        blueprint_summary: str,
        helper: ExperimentAgent,
        resource_context: dict[str, Any] | None = None,
        runtime_python: str = "python",
        execution_policy: ExperimentExecutionPolicy | None = None,
        remediation_ledger: list[dict[str, Any]] | None = None,
        round_number: int | None = None,
    ) -> dict[str, Any]:
        """Run quick-eval with timeout handling and batch-fix cycles."""
        timeout = self.config.quick_eval_timeout
        max_fix_cycles = 5
        last_result: dict[str, Any] = {}
        fix_history: list[dict[str, Any]] = []

        metrics_path = code_dir / "results" / "metrics.json"
        training_log_path = code_dir / "results" / "training_log.csv"
        for cycle in range(1, max_fix_cycles + 1):
            mtime_before = metrics_path.stat().st_mtime if metrics_path.exists() else None
            training_log_mtime_before = (
                training_log_path.stat().st_mtime if training_log_path.exists() else None
            )
            result = await self._run_subprocess(
                self._command_with_mode(base_command, "--quick-eval"),
                cwd=code_dir,
                timeout=timeout,
            )
            last_result = result
            if result["returncode"] == 0:
                quick_eval = helper._collect_quick_eval_results(code_dir, result, attempt=cycle)
                augmented = self._augment_quick_eval_metrics_from_logs(code_dir, quick_eval, result)
                recovered_source = str(augmented.get("recovered_from") or "").strip()
                snapshot_entry = None
                if recovered_source == "execution_log":
                    materialized = self._materialize_recovered_metrics_artifact(
                        code_dir,
                        augmented.get("metrics", {}),
                        source="execution_log",
                        scope="local_quick_eval",
                    )
                    snapshot_entry = self.consume_last_mutation_snapshot_entry()
                    if materialized.get("metrics"):
                        augmented["metrics"] = materialized["metrics"]
                    if materialized.get("written"):
                        augmented["metrics_artifact_materialized"] = True
                        augmented["metrics_artifact_path"] = materialized.get("artifact_path", "")
                elif recovered_source and augmented.get("metrics_artifact_materialized"):
                    snapshot_entry = self.consume_last_mutation_snapshot_entry()
                if recovered_source:
                    self._append_remediation_entry(
                        remediation_ledger,
                        kind="metrics_recovery",
                        status="applied",
                        scope="local_quick_eval",
                        round_number=round_number,
                        cycle=cycle,
                        details={
                            "source": recovered_source,
                            **(
                                {
                                    "artifact_path": augmented.get("metrics_artifact_path", ""),
                                    "artifact_materialized": True,
                                }
                                if augmented.get("metrics_artifact_materialized")
                                else {}
                            ),
                        },
                    )
                    if augmented.get("metrics_artifact_materialized"):
                        details = {
                            "source": recovered_source,
                            "artifact_path": augmented.get("metrics_artifact_path", ""),
                        }
                        if snapshot_entry:
                            details.update({
                                "snapshot_entry_id": snapshot_entry.get("entry_id"),
                                "snapshot_count": snapshot_entry.get("snapshot_count", 0),
                                "snapshot_journal_path": REPAIR_SNAPSHOT_JOURNAL_PATH,
                                "snapshots": list(snapshot_entry.get("snapshots", []) or []),
                            })
                        self._append_remediation_entry(
                            remediation_ledger,
                            kind="metrics_artifact_recovery",
                            status="applied",
                            scope="local_quick_eval",
                            round_number=round_number,
                            cycle=cycle,
                            files=[str(augmented.get("metrics_artifact_path", ""))],
                            details=details,
                        )
                return augmented

            if result["returncode"] == -1:
                metrics_updated = False
                training_log_updated = False
                if metrics_path.exists():
                    try:
                        mtime_after = metrics_path.stat().st_mtime
                        metrics_updated = mtime_before is None or mtime_after > mtime_before
                    except OSError:
                        metrics_updated = False
                if training_log_path.exists():
                    try:
                        training_log_mtime_after = training_log_path.stat().st_mtime
                        training_log_updated = (
                            training_log_mtime_before is None
                            or training_log_mtime_after > training_log_mtime_before
                        )
                    except OSError:
                        training_log_updated = False

                if metrics_updated or training_log_updated:
                    artifact_results = self._collect_result_artifacts(code_dir)
                    artifact_metrics = artifact_results.get("metrics")
                    if self._metrics_satisfy_contract(artifact_metrics):
                        recovered_source = str(artifact_results.get("recovered_from") or "").strip()
                        snapshot_entry = (
                            self.consume_last_mutation_snapshot_entry()
                            if artifact_results.get("metrics_artifact_materialized")
                            else None
                        )
                        if recovered_source:
                            self._append_remediation_entry(
                                remediation_ledger,
                                kind="metrics_recovery",
                                status="applied",
                                scope="local_quick_eval",
                                round_number=round_number,
                                cycle=cycle,
                                details={
                                    "source": recovered_source,
                                    **(
                                        {
                                            "artifact_path": artifact_results.get("metrics_artifact_path", ""),
                                            "artifact_materialized": True,
                                        }
                                        if artifact_results.get("metrics_artifact_materialized")
                                        else {}
                                    ),
                                },
                            )
                            if artifact_results.get("metrics_artifact_materialized"):
                                details = {
                                    "source": recovered_source,
                                    "artifact_path": artifact_results.get("metrics_artifact_path", ""),
                                }
                                if snapshot_entry:
                                    details.update({
                                        "snapshot_entry_id": snapshot_entry.get("entry_id"),
                                        "snapshot_count": snapshot_entry.get("snapshot_count", 0),
                                        "snapshot_journal_path": REPAIR_SNAPSHOT_JOURNAL_PATH,
                                        "snapshots": list(snapshot_entry.get("snapshots", []) or []),
                                    })
                                self._append_remediation_entry(
                                    remediation_ledger,
                                    kind="metrics_artifact_recovery",
                                    status="applied",
                                    scope="local_quick_eval",
                                    round_number=round_number,
                                    cycle=cycle,
                                    files=[str(artifact_results.get("metrics_artifact_path", ""))],
                                    details=details,
                                )
                        return {
                            "status": "partial" if recovered_source else "success",
                            "metrics": artifact_metrics,
                            "attempts": cycle,
                            "stdout": result.get("stdout", ""),
                            "stderr": result.get("stderr", ""),
                            **({"recovered_from": recovered_source} if recovered_source else {}),
                            **(
                                {
                                    "metrics_artifact_materialized": True,
                                    "metrics_artifact_path": artifact_results.get("metrics_artifact_path", ""),
                                }
                                if artifact_results.get("metrics_artifact_materialized")
                                else {}
                            ),
                        }

                if metrics_updated:
                    recovered = self._recover_metrics_contract_from_logs(result)
                    if recovered:
                        materialized = self._materialize_recovered_metrics_artifact(
                            code_dir,
                            recovered,
                            source="execution_log",
                            scope="local_quick_eval",
                        )
                        snapshot_entry = self.consume_last_mutation_snapshot_entry()
                        self._append_remediation_entry(
                            remediation_ledger,
                            kind="metrics_recovery",
                            status="applied",
                            scope="local_quick_eval",
                            round_number=round_number,
                            cycle=cycle,
                            details={
                                "source": "execution_log",
                                **(
                                    {
                                        "artifact_path": materialized.get("artifact_path", ""),
                                        "artifact_materialized": True,
                                    }
                                    if materialized.get("written")
                                    else {}
                                ),
                            },
                        )
                        if materialized.get("written"):
                            details = {
                                "source": "execution_log",
                                "artifact_path": materialized.get("artifact_path", ""),
                            }
                            if snapshot_entry:
                                details.update({
                                    "snapshot_entry_id": snapshot_entry.get("entry_id"),
                                    "snapshot_count": snapshot_entry.get("snapshot_count", 0),
                                    "snapshot_journal_path": REPAIR_SNAPSHOT_JOURNAL_PATH,
                                    "snapshots": list(snapshot_entry.get("snapshots", []) or []),
                                })
                            self._append_remediation_entry(
                                remediation_ledger,
                                kind="metrics_artifact_recovery",
                                status="applied",
                                scope="local_quick_eval",
                                round_number=round_number,
                                cycle=cycle,
                                files=[str(materialized.get("artifact_path", ""))],
                                details=details,
                            )
                        return {
                            "status": "partial",
                            "metrics": materialized.get("metrics") or recovered,
                            "attempts": cycle,
                            "stdout": result.get("stdout", ""),
                            "stderr": result.get("stderr", ""),
                            "recovered_from": "execution_log",
                            **(
                                {
                                    "metrics_artifact_materialized": True,
                                    "metrics_artifact_path": materialized.get("artifact_path", ""),
                                }
                                if materialized.get("written")
                                else {}
                            ),
                        }

            if cycle >= max_fix_cycles:
                break

            if result["returncode"] == -1 and "timed out" in result.get("stderr", "").lower():
                resume_fix = self._attempt_resume_repair(
                    code_dir,
                    self._repair_error_text(result),
                    resource_context,
                    scope="local_quick_eval",
                )
                resume_snapshot_entry = self.consume_last_mutation_snapshot_entry()
                if resume_fix:
                    self.log(
                        "Applied deterministic resume repair during quick-eval timeout recovery: "
                        f"{resume_fix}"
                    )
                    details = None
                    if resume_snapshot_entry:
                        details = {
                            "snapshot_entry_id": resume_snapshot_entry.get("entry_id"),
                            "snapshot_count": resume_snapshot_entry.get("snapshot_count", 0),
                            "snapshot_journal_path": REPAIR_SNAPSHOT_JOURNAL_PATH,
                            "snapshots": list(resume_snapshot_entry.get("snapshots", []) or []),
                        }
                    self._append_remediation_entry(
                        remediation_ledger,
                        kind="resume_repair",
                        status="applied",
                        scope="local_quick_eval",
                        round_number=round_number,
                        cycle=cycle,
                        signature=self._repair_error_signature(result),
                        files=list(resume_fix),
                        details=details,
                    )
                    self._record_repair_attempt(
                        fix_history,
                        self._repair_error_signature(result),
                        self._repair_error_text(result),
                        cycle,
                        resume_fix,
                    )
                    continue
                modified = await helper._fix_timeout(code_dir)
                timeout_snapshot_entry = helper.consume_last_mutation_snapshot_entry()
                details = None
                if timeout_snapshot_entry:
                    details = {
                        "snapshot_entry_id": timeout_snapshot_entry.get("entry_id"),
                        "snapshot_count": timeout_snapshot_entry.get("snapshot_count", 0),
                        "snapshot_journal_path": REPAIR_SNAPSHOT_JOURNAL_PATH,
                        "snapshots": list(timeout_snapshot_entry.get("snapshots", []) or []),
                    }
                self._append_remediation_entry(
                    remediation_ledger,
                    kind="timeout_fix",
                    status="applied" if modified else "skipped",
                    scope="local_quick_eval",
                    round_number=round_number,
                    cycle=cycle,
                    reason="" if modified else "no_files_modified",
                    files=list(modified or []),
                    details=details,
                )
            else:
                repair_text = self._repair_error_text(result)
                signature = self._repair_error_signature(result)
                repeat_count = self._repair_repeat_count(fix_history, signature)
                resume_fix = self._attempt_resume_repair(
                    code_dir,
                    repair_text,
                    resource_context,
                    scope="local_quick_eval",
                )
                resume_snapshot_entry = self.consume_last_mutation_snapshot_entry()
                if resume_fix:
                    self.log(
                        "Applied deterministic resume repair during quick-eval: "
                        f"{resume_fix}"
                    )
                    details = None
                    if resume_snapshot_entry:
                        details = {
                            "snapshot_entry_id": resume_snapshot_entry.get("entry_id"),
                            "snapshot_count": resume_snapshot_entry.get("snapshot_count", 0),
                            "snapshot_journal_path": REPAIR_SNAPSHOT_JOURNAL_PATH,
                            "snapshots": list(resume_snapshot_entry.get("snapshots", []) or []),
                        }
                    self._append_remediation_entry(
                        remediation_ledger,
                        kind="resume_repair",
                        status="applied",
                        scope="local_quick_eval",
                        round_number=round_number,
                        cycle=cycle,
                        signature=signature,
                        files=list(resume_fix),
                        details=details,
                    )
                    self._record_repair_attempt(
                        fix_history,
                        signature,
                        repair_text,
                        cycle,
                        resume_fix,
                    )
                    continue
                deterministic_fix = self._attempt_resource_path_repair(
                    code_dir,
                    repair_text,
                    resource_context,
                    scope="local_quick_eval",
                )
                snapshot_entry = self.consume_last_mutation_snapshot_entry()
                if deterministic_fix:
                    self.log(
                        "Applied deterministic resource-path repair during quick-eval: "
                        f"{deterministic_fix}"
                    )
                    details = None
                    if snapshot_entry:
                        details = {
                            "snapshot_entry_id": snapshot_entry.get("entry_id"),
                            "snapshot_count": snapshot_entry.get("snapshot_count", 0),
                            "snapshot_journal_path": REPAIR_SNAPSHOT_JOURNAL_PATH,
                            "snapshots": list(snapshot_entry.get("snapshots", []) or []),
                        }
                    self._append_remediation_entry(
                        remediation_ledger,
                        kind="resource_path_repair",
                        status="applied",
                        scope="local_quick_eval",
                        round_number=round_number,
                        cycle=cycle,
                        signature=signature,
                        files=list(deterministic_fix),
                        details=details,
                    )
                    self._record_repair_attempt(
                        fix_history,
                        signature,
                        repair_text,
                        cycle,
                        deterministic_fix,
                    )
                    continue
                option_value_fix = self._attempt_option_value_repair(
                    code_dir,
                    repair_text,
                    resource_context,
                    scope="local_quick_eval",
                )
                option_value_snapshot_entry = self.consume_last_mutation_snapshot_entry()
                if option_value_fix:
                    self.log(
                        "Applied deterministic option-value repair during quick-eval: "
                        f"{option_value_fix}"
                    )
                    details = None
                    if option_value_snapshot_entry:
                        details = {
                            "snapshot_entry_id": option_value_snapshot_entry.get("entry_id"),
                            "snapshot_count": option_value_snapshot_entry.get("snapshot_count", 0),
                            "snapshot_journal_path": REPAIR_SNAPSHOT_JOURNAL_PATH,
                            "snapshots": list(option_value_snapshot_entry.get("snapshots", []) or []),
                        }
                    self._append_remediation_entry(
                        remediation_ledger,
                        kind="option_value_repair",
                        status="applied",
                        scope="local_quick_eval",
                        round_number=round_number,
                        cycle=cycle,
                        signature=signature,
                        files=list(option_value_fix),
                        details=details,
                    )
                    self._record_repair_attempt(
                        fix_history,
                        signature,
                        repair_text,
                        cycle,
                        option_value_fix,
                    )
                    continue
                unknown_arg_fix = self._attempt_unrecognized_argument_repair(
                    code_dir,
                    repair_text,
                    mode="quick-eval",
                    scope="local_quick_eval",
                )
                unknown_arg_snapshot_entry = self.consume_last_mutation_snapshot_entry()
                if unknown_arg_fix:
                    self.log(
                        "Applied deterministic unrecognized-argument repair during quick-eval: "
                        f"{unknown_arg_fix}"
                    )
                    details = None
                    if unknown_arg_snapshot_entry:
                        details = {
                            "snapshot_entry_id": unknown_arg_snapshot_entry.get("entry_id"),
                            "snapshot_count": unknown_arg_snapshot_entry.get("snapshot_count", 0),
                            "snapshot_journal_path": REPAIR_SNAPSHOT_JOURNAL_PATH,
                            "snapshots": list(unknown_arg_snapshot_entry.get("snapshots", []) or []),
                        }
                    self._append_remediation_entry(
                        remediation_ledger,
                        kind="unrecognized_argument_repair",
                        status="applied",
                        scope="local_quick_eval",
                        round_number=round_number,
                        cycle=cycle,
                        signature=signature,
                        files=list(unknown_arg_fix),
                        details=details,
                    )
                    self._record_repair_attempt(
                        fix_history,
                        signature,
                        repair_text,
                        cycle,
                        unknown_arg_fix,
                    )
                    continue
                required_arg_fix = self._attempt_required_argument_repair(
                    code_dir,
                    repair_text,
                    resource_context,
                    scope="local_quick_eval",
                )
                required_arg_snapshot_entry = self.consume_last_mutation_snapshot_entry()
                if required_arg_fix:
                    self.log(
                        "Applied deterministic required-argument repair during quick-eval: "
                        f"{required_arg_fix}"
                    )
                    details = None
                    if required_arg_snapshot_entry:
                        details = {
                            "snapshot_entry_id": required_arg_snapshot_entry.get("entry_id"),
                            "snapshot_count": required_arg_snapshot_entry.get("snapshot_count", 0),
                            "snapshot_journal_path": REPAIR_SNAPSHOT_JOURNAL_PATH,
                            "snapshots": list(required_arg_snapshot_entry.get("snapshots", []) or []),
                        }
                    self._append_remediation_entry(
                        remediation_ledger,
                        kind="required_argument_repair",
                        status="applied",
                        scope="local_quick_eval",
                        round_number=round_number,
                        cycle=cycle,
                        signature=signature,
                        files=list(required_arg_fix),
                        details=details,
                    )
                    self._record_repair_attempt(
                        fix_history,
                        signature,
                        repair_text,
                        cycle,
                        required_arg_fix,
                    )
                    continue
                runtime_fix = await self._attempt_runtime_remediation(
                    code_dir,
                    repair_text,
                    runtime_python=runtime_python,
                    fix_history=fix_history,
                    execution_policy=execution_policy,
                    remediation_ledger=remediation_ledger,
                    mode="quick-eval",
                    cycle=cycle,
                    signature=signature,
                    round_number=round_number,
                )
                if runtime_fix:
                    self.log(
                        "Applied deterministic runtime remediation during quick-eval: "
                        f"{runtime_fix}"
                    )
                    self._record_repair_attempt(
                        fix_history,
                        signature,
                        repair_text,
                        cycle,
                        runtime_fix,
                    )
                    continue
                modified = await helper._batch_fix_errors(
                    code_dir,
                    repair_text,
                    blueprint_summary,
                    mode="quick-eval",
                    previous_fixes=[dict(entry) for entry in fix_history],
                    extra_context=self._build_repair_context(
                        code_dir,
                        result,
                        mode="quick-eval",
                        repeat_count=repeat_count,
                        resource_context=resource_context,
                    ),
                )
                self._record_repair_attempt(
                    fix_history,
                    signature,
                    repair_text,
                    cycle,
                    modified,
                )
                self._append_remediation_entry(
                    remediation_ledger,
                    kind="llm_batch_fix",
                    status="applied" if modified else "skipped",
                    scope="local_quick_eval",
                    round_number=round_number,
                    cycle=cycle,
                    signature=signature,
                    reason="" if modified else "no_files_modified",
                    files=list(modified or []),
                )
            if not modified:
                break

        return {"status": "failed", "metrics": {}, "attempts": cycle, **last_result}

    @staticmethod
    def _repair_error_text(result: dict[str, Any]) -> str:
        stderr_text = str(result.get("stderr") or "").strip()
        if stderr_text:
            return stderr_text
        stdout_text = str(result.get("stdout") or "").strip()
        if stdout_text:
            return stdout_text
        return f"Process exited with return code {result.get('returncode', 'unknown')} and produced no output."

    @classmethod
    def _repair_error_signature(cls, result: dict[str, Any]) -> str:
        error_text = cls._repair_error_text(result)
        for raw_line in reversed(error_text.splitlines()):
            line = raw_line.strip()
            if not line or line.startswith("File ") or line.startswith("Traceback"):
                continue
            return f"rc={result.get('returncode', 'unknown')}|{line[:240]}"
        return f"rc={result.get('returncode', 'unknown')}|empty"

    @staticmethod
    def _repair_repeat_count(
        fix_history: list[dict[str, Any]],
        signature: str,
    ) -> int:
        for entry in fix_history:
            if entry.get("signature") == signature:
                return int(entry.get("repeat_count", 1)) + 1
        return 1

    @staticmethod
    def _record_repair_attempt(
        fix_history: list[dict[str, Any]],
        signature: str,
        error_text: str,
        cycle: int,
        modified: list[str],
    ) -> None:
        for entry in fix_history:
            if entry.get("signature") == signature:
                entry["repeat_count"] = int(entry.get("repeat_count", 1)) + 1
                entry["cycle"] = cycle
                entry["error_msg"] = error_text[:300]
                if modified:
                    seen = list(entry.get("fixed_files", []))
                    for rel_path in modified:
                        if rel_path not in seen:
                            seen.append(rel_path)
                    entry["fixed_files"] = seen
                return

        fix_history.append(
            {
                "signature": signature,
                "error_msg": error_text[:300],
                "cycle": cycle,
                "repeat_count": 1,
                "fixed_files": list(modified or []),
            }
        )

    @staticmethod
    def _append_remediation_entry(
        remediation_ledger: list[dict[str, Any]] | None,
        *,
        kind: str,
        status: str,
        scope: str,
        round_number: int | None = None,
        cycle: int | None = None,
        signature: str = "",
        reason: str = "",
        files: list[str] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        if remediation_ledger is None:
            return

        entry: dict[str, Any] = {
            "entry_id": len(remediation_ledger) + 1,
            "kind": kind,
            "status": status,
            "scope": scope,
        }
        if round_number is not None:
            entry["round_number"] = round_number
        if cycle is not None:
            entry["cycle"] = cycle
        if signature:
            entry["signature"] = signature
        if reason:
            entry["reason"] = reason
        if files:
            entry["files"] = list(files)
        if details:
            entry["details"] = dict(details)
        remediation_ledger.append(entry)

    def _persist_remediation_ledger(
        self,
        remediation_ledger: list[dict[str, Any]] | None,
    ) -> str:
        payload = {
            "entry_count": len(remediation_ledger or []),
            "entries": list(remediation_ledger or []),
        }
        self.workspace.write_json(REMEDIATION_LEDGER_PATH, payload)
        return REMEDIATION_LEDGER_PATH

    def _repair_snapshot_journal_path(self) -> str:
        journal_path = self.workspace.path / REPAIR_SNAPSHOT_JOURNAL_PATH
        return REPAIR_SNAPSHOT_JOURNAL_PATH if journal_path.is_file() else ""

    def _record_snapshot_batch(
        self,
        *,
        mutation_kind: str,
        scope: str,
        snapshots: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if not snapshots:
            self._remember_mutation_snapshot_entry(None)
            return None

        entry = append_snapshot_journal(
            self.workspace.path,
            agent=self.__class__.__name__,
            mutation_kind=mutation_kind,
            scope=scope,
            snapshots=snapshots,
            metadata=metadata,
        )
        self._remember_mutation_snapshot_entry(entry)
        return entry

    def _record_runtime_env_ledger(
        self,
        runtime_env: dict[str, Any],
        remediation_ledger: list[dict[str, Any]] | None,
    ) -> None:
        if remediation_ledger is None or not isinstance(runtime_env, dict):
            return

        if runtime_env.get("created"):
            self._append_remediation_entry(
                remediation_ledger,
                kind="runtime_env_create",
                status="applied",
                scope="local_environment",
                details={
                    "env_kind": runtime_env.get("kind", ""),
                    "env_name": runtime_env.get("env_name", ""),
                    "env_path": runtime_env.get("env_path", ""),
                    "recreated": bool(runtime_env.get("recreated", False)),
                },
            )

        dependency_install = runtime_env.get("dependency_install")
        if not isinstance(dependency_install, dict):
            dependency_install = {}
        status = str(dependency_install.get("status") or "").strip()
        if status:
            self._append_remediation_entry(
                remediation_ledger,
                kind="dependency_install",
                status=status,
                scope="local_environment",
                details={
                    "source": dependency_install.get("source", ""),
                    "manifest": dependency_install.get("manifest", ""),
                    "strategy": dependency_install.get("strategy", ""),
                    "error": dependency_install.get("error", ""),
                    "stderr": dependency_install.get("stderr", ""),
                    "returncode": dependency_install.get("returncode"),
                },
            )

        runtime_validation = runtime_env.get("runtime_validation")
        if not isinstance(runtime_validation, dict):
            return
        validation_status = str(runtime_validation.get("status") or "").strip()
        if not validation_status:
            return
        python_smoke = runtime_validation.get("python_smoke")
        pip_probe = runtime_validation.get("pip_probe")
        import_probe = runtime_validation.get("import_probe")
        self._append_remediation_entry(
            remediation_ledger,
            kind="runtime_env_validation",
            status=validation_status,
            scope="local_environment",
            details={
                "python_smoke_status": python_smoke.get("status", "") if isinstance(python_smoke, dict) else "",
                "python_executable": python_smoke.get("executable", "") if isinstance(python_smoke, dict) else "",
                "python_version": python_smoke.get("version", "") if isinstance(python_smoke, dict) else "",
                "pip_status": pip_probe.get("status", "") if isinstance(pip_probe, dict) else "",
                "pip_version": pip_probe.get("version", "") if isinstance(pip_probe, dict) else "",
                "import_status": import_probe.get("status", "") if isinstance(import_probe, dict) else "",
                "failed_imports": list(import_probe.get("failures", []) or []) if isinstance(import_probe, dict) else [],
                "skipped_reason": import_probe.get("skipped_reason", "") if isinstance(import_probe, dict) else "",
            },
        )

        runtime_validation_repair = runtime_env.get("runtime_validation_repair")
        if not isinstance(runtime_validation_repair, dict):
            return
        repair_status = str(runtime_validation_repair.get("status") or "").strip()
        if not repair_status or repair_status == "skipped":
            return
        self._append_remediation_entry(
            remediation_ledger,
            kind="runtime_env_repair",
            status=repair_status,
            scope="local_environment",
            details={
                "actions": list(runtime_validation_repair.get("actions", []) or []),
            },
        )

    def _record_launch_contract_ledger(
        self,
        launch_contract: dict[str, Any],
        remediation_ledger: list[dict[str, Any]] | None,
        *,
        round_number: int | None = None,
    ) -> None:
        if remediation_ledger is None or not isinstance(launch_contract, dict):
            return
        status = str(launch_contract.get("status") or "").strip()
        if not status:
            return
        self._append_remediation_entry(
            remediation_ledger,
            kind="launch_contract",
            status=status,
            scope="local_launch",
            round_number=round_number,
            details={
                "target_kind": launch_contract.get("target_kind", ""),
                "target": launch_contract.get("target", ""),
                "resolved_target": launch_contract.get("resolved_target", ""),
                "created_dirs": list(launch_contract.get("created_dirs", []) or []),
                "warnings": list(launch_contract.get("warnings", []) or []),
                "failures": list(launch_contract.get("failures", []) or []),
            },
        )

    def _record_launch_contract_repair_ledger(
        self,
        repair_result: dict[str, Any],
        remediation_ledger: list[dict[str, Any]] | None,
        *,
        round_number: int | None = None,
    ) -> None:
        if remediation_ledger is None or not isinstance(repair_result, dict):
            return
        status = str(repair_result.get("status") or "").strip()
        if not status or status == "skipped":
            return
        self._append_remediation_entry(
            remediation_ledger,
            kind="launch_contract_repair",
            status=status,
            scope="local_launch",
            round_number=round_number,
            details={
                "actions": list(repair_result.get("actions", []) or []),
                "files_modified": list(repair_result.get("files_modified", []) or []),
                "command": list(repair_result.get("command", []) or []),
                "initial_failures": list(
                    repair_result.get("initial_contract", {}).get("failures", [])
                    if isinstance(repair_result.get("initial_contract"), dict)
                    else []
                ),
                "final_failures": list(
                    repair_result.get("final_contract", {}).get("failures", [])
                    if isinstance(repair_result.get("final_contract"), dict)
                    else []
                ),
            },
        )

    def _build_repair_context(
        self,
        code_dir: Path,
        result: dict[str, Any],
        *,
        mode: str,
        repeat_count: int,
        resource_context: dict[str, Any] | None = None,
    ) -> str:
        report = PreflightChecker(code_dir).run_all()
        context_parts: list[str] = []

        stdout_text = str(result.get("stdout") or "").strip()
        stderr_text = str(result.get("stderr") or "").strip()
        if stdout_text and stdout_text != stderr_text:
            stdout_lines = stdout_text.splitlines()
            stdout_snippet = "\n".join(stdout_lines[-20:])[:1200]
            context_parts.append(f"Recent stdout ({mode}):\n{stdout_snippet}")

        if report.blocking_failures:
            context_parts.append(
                "Preflight blocking diagnostics:\n- " + "\n- ".join(report.blocking_failures[:8])
            )
        elif report.warning_messages:
            context_parts.append(
                "Preflight warnings:\n- " + "\n- ".join(report.warning_messages[:8])
            )

        if report.suggested_fixes:
            context_parts.append(
                "Suggested preflight fixes:\n- " + "\n- ".join(report.suggested_fixes[:8])
            )

        resource_summary = self._summarize_available_resources(code_dir, resource_context)
        if resource_summary:
            context_parts.append(resource_summary)

        if repeat_count > 1:
            context_parts.append(
                f"This failure signature has repeated {repeat_count} times. "
                "Do not repeat the same patch strategy; target a different root cause."
            )

        return "\n\n".join(part for part in context_parts if part)

    @staticmethod
    def _extract_missing_resource_targets(error_text: str) -> list[str]:
        patterns = [
            r"""No such file or directory:\s*['"]([^'"]+)['"]""",
            r"""can't open file\s+['"]([^'"]+)['"]""",
            r"""does not exist:\s*['"]([^'"]+)['"]""",
            r"""FileNotFoundError:.*?['"]([^'"]+)['"]""",
            r"""Can't load [^'"]+ for ['"]([^'"]+)['"]""",
            r"""Incorrect path_or_model_id:\s*['"]([^'"]+)['"]""",
        ]
        targets: list[str] = []
        for pattern in patterns:
            for match in re.finditer(pattern, error_text, re.IGNORECASE):
                candidate = str(match.group(1)).strip()
                if candidate and candidate not in targets:
                    targets.append(candidate)
        return targets

    @staticmethod
    def _resource_kind_from_path(path_text: str) -> str:
        lower = path_text.lower()
        if any(token in lower for token in ("/models/", "\\models\\", ".pt", ".bin", ".ckpt", ".safetensors")):
            return "model"
        return "dataset"

    @staticmethod
    def _normalized_resource_key(path_text: str) -> str:
        name = Path(path_text).name.lower()
        for suffix in (".tar.gz", ".tar.bz2", ".tar.xz"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        else:
            for suffix in (".gz", ".bz2", ".xz", ".zip"):
                if name.endswith(suffix):
                    name = name[: -len(suffix)]
                    break

        while True:
            stem, ext = os.path.splitext(name)
            if ext.lower() in {
                ".csv",
                ".tsv",
                ".txt",
                ".json",
                ".jsonl",
                ".pkl",
                ".pickle",
                ".npy",
                ".npz",
                ".pt",
                ".pth",
                ".bin",
                ".ckpt",
                ".h5",
                ".hdf5",
                ".parquet",
                ".fa",
                ".fasta",
            }:
                name = stem
                continue
            break
        return name

    @classmethod
    def _collect_resource_candidates(
        cls,
        code_dir: Path,
        resource_context: dict[str, Any] | None,
    ) -> list[dict[str, str]]:
        candidates: list[dict[str, str]] = []
        seen_paths: set[str] = set()

        def add_candidate(path_value: str, kind: str, name: str) -> None:
            normalized = str(path_value or "").strip()
            if not normalized or normalized in seen_paths:
                return
            candidate_path = Path(normalized)
            if not candidate_path.exists():
                return
            seen_paths.add(normalized)
            candidates.append(
                {
                    "path": normalized,
                    "kind": kind,
                    "name": str(name or "").strip().lower(),
                    "basename": candidate_path.name.lower(),
                    "normalized_key": cls._normalized_resource_key(normalized),
                }
            )

        def scan_root(root_path: Path, kind: str) -> None:
            if not root_path.exists():
                return
            add_candidate(str(root_path), kind, root_path.name)
            try:
                children = sorted(root_path.iterdir())
            except OSError:
                return
            for child in children:
                add_candidate(str(child), kind, child.name)
                if child.is_dir():
                    try:
                        nested_children = sorted(child.iterdir())[:20]
                    except OSError:
                        continue
                    for nested in nested_children:
                        add_candidate(str(nested), kind, child.name)

        if isinstance(resource_context, dict):
            for resource in resource_context.get("downloaded_resources", []):
                if not isinstance(resource, dict):
                    continue
                if resource.get("status") not in RESOURCE_SUCCESS_STATUSES:
                    continue
                kind = str(resource.get("type", "dataset")).strip().lower()
                name = str(resource.get("name", "")).strip()
                for key in ("workspace_path", "path"):
                    value = resource.get(key)
                    if isinstance(value, str):
                        add_candidate(value, kind, name)
                for value in resource.get("workspace_files", []) or []:
                    if isinstance(value, str):
                        add_candidate(value, kind, name)

            for alias in resource_context.get("workspace_resource_aliases", []):
                if not isinstance(alias, dict):
                    continue
                kind = str(alias.get("type", "dataset")).strip().lower()
                name = str(alias.get("name", "")).strip()
                workspace_path = alias.get("workspace_path")
                if isinstance(workspace_path, str):
                    add_candidate(workspace_path, kind, name)
                for value in alias.get("workspace_files", []) or []:
                    if isinstance(value, str):
                        add_candidate(value, kind, name)

            for root_key, kind in (("data_dir", "dataset"), ("models_dir", "model")):
                root_value = str(resource_context.get(root_key, "")).strip()
                if not root_value:
                    continue
                root_path = Path(root_value)
                scan_root(root_path, kind)

        for root_path, kind in (
            (code_dir / "data", "dataset"),
            (code_dir / "datasets", "dataset"),
            (code_dir / "models", "model"),
            (code_dir / "checkpoints", "model"),
        ):
            scan_root(root_path, kind)

        for config_candidate in (
            code_dir / "config.py",
            code_dir / "config.yaml",
            code_dir / "config.yml",
            code_dir / "config.json",
            code_dir / "config.toml",
            code_dir / "config" / "default.yaml",
            code_dir / "config" / "default.yml",
            code_dir / "config" / "default.json",
            code_dir / "config" / "default.toml",
            code_dir / "configs" / "default.yaml",
            code_dir / "configs" / "default.yml",
            code_dir / "configs" / "default.json",
            code_dir / "configs" / "default.toml",
            code_dir / ".nanoresearch_autofix" / "config_auto.yaml",
            code_dir / ".nanoresearch_autofix" / "config_auto.json",
            code_dir / ".nanoresearch_autofix" / "config_auto.toml",
        ):
            if config_candidate.exists():
                add_candidate(str(config_candidate), "config", config_candidate.name)

        return candidates

    @classmethod
    def _match_resource_target(
        cls,
        code_dir: Path,
        missing_target: str,
        resource_context: dict[str, Any] | None,
    ) -> str | None:
        candidates = cls._collect_resource_candidates(code_dir, resource_context)
        if not candidates:
            return None

        missing_path = Path(missing_target)
        missing_name = missing_path.name.lower()
        missing_kind = (
            "config"
            if "config" in missing_target.lower()
            else cls._resource_kind_from_path(missing_target)
        )

        def filter_kind(items: list[dict[str, str]]) -> list[dict[str, str]]:
            typed = [item for item in items if item["kind"] == missing_kind]
            return typed or items

        cache_to_workspace = [
            ("cache_data_dir", "data_dir"),
            ("cache_models_dir", "models_dir"),
        ]
        for cache_key, workspace_key in cache_to_workspace:
            cache_dir = str(resource_context.get(cache_key, "") if isinstance(resource_context, dict) else "").strip()
            workspace_dir = str(resource_context.get(workspace_key, "") if isinstance(resource_context, dict) else "").strip()
            if cache_dir and workspace_dir and missing_target.startswith(cache_dir):
                suffix = missing_target[len(cache_dir):].lstrip("/\\")
                candidate = Path(workspace_dir) / suffix
                if candidate.exists():
                    return str(candidate)

        basename_matches = filter_kind(
            [item for item in candidates if item["basename"] == missing_name]
        )
        if len(basename_matches) == 1:
            return basename_matches[0]["path"]

        normalized_key = cls._normalized_resource_key(missing_target)
        normalized_matches = filter_kind(
            [item for item in candidates if item.get("normalized_key") == normalized_key]
        )
        if len(normalized_matches) == 1:
            return normalized_matches[0]["path"]

        name_matches = filter_kind(
            [item for item in candidates if item["name"] and item["name"] in missing_target.lower()]
        )
        if len(name_matches) == 1:
            return name_matches[0]["path"]

        if missing_kind == "config":
            config_files = [item for item in candidates if item["kind"] == "config" and Path(item["path"]).is_file()]
            if len(config_files) == 1:
                return config_files[0]["path"]

        if missing_kind == "dataset":
            dataset_files = [item for item in candidates if item["kind"] == "dataset" and Path(item["path"]).is_file()]
            if len(dataset_files) == 1:
                return dataset_files[0]["path"]

        return None

    @classmethod
    def _resource_replacement_map(
        cls,
        code_dir: Path,
        error_text: str,
        resource_context: dict[str, Any] | None,
    ) -> dict[str, str]:
        if not isinstance(resource_context, dict):
            resource_context = {}

        replacements: dict[str, str] = {}
        for old_key, new_key in (("cache_data_dir", "data_dir"), ("cache_models_dir", "models_dir")):
            old_value = str(resource_context.get(old_key, "")).strip()
            new_value = str(resource_context.get(new_key, "")).strip()
            if old_value and new_value and old_value != new_value:
                replacements[old_value] = new_value

        for target in cls._extract_missing_resource_targets(error_text):
            replacement = cls._match_resource_target(code_dir, target, resource_context)
            if replacement and replacement != target:
                replacements[target] = replacement

        return replacements

    @classmethod
    def _materialize_missing_resource_targets(
        cls,
        code_dir: Path,
        error_text: str,
        resource_context: dict[str, Any] | None,
    ) -> list[str]:
        if not isinstance(resource_context, dict):
            return []

        created: list[str] = []
        candidates = cls._collect_resource_candidates(code_dir, resource_context)
        for target_text in cls._extract_missing_resource_targets(error_text):
            target_path = Path(target_text)
            if target_path.exists():
                continue
            if target_path.suffix.lower() in {".gz", ".bz2", ".xz", ".zip"}:
                continue

            normalized_key = cls._normalized_resource_key(target_text)
            gz_matches = [
                item
                for item in candidates
                if Path(item["path"]).is_file()
                and item.get("normalized_key") == normalized_key
                and item["path"].lower().endswith(".gz")
                and not item["path"].lower().endswith(".tar.gz")
            ]
            if len(gz_matches) == 1:
                source_path = Path(gz_matches[0]["path"])
                try:
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    with gzip.open(source_path, "rb") as src, open(target_path, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                except OSError:
                    pass
                else:
                    created.append(str(target_path))
                    continue

            zip_matches = [
                item
                for item in candidates
                if Path(item["path"]).is_file()
                and item["path"].lower().endswith(".zip")
            ]
            extracted = False
            for zip_candidate in zip_matches[:5]:
                source_path = Path(zip_candidate["path"])
                try:
                    with zipfile.ZipFile(source_path) as archive:
                        members = [
                            member
                            for member in archive.namelist()
                            if member and not member.endswith("/")
                        ]
                        matching_members = [
                            member
                            for member in members
                            if cls._normalized_resource_key(member) == normalized_key
                        ]
                        if len(matching_members) != 1:
                            continue
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(matching_members[0]) as src, open(target_path, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                except (OSError, zipfile.BadZipFile, KeyError):
                    continue
                created.append(str(target_path))
                extracted = True
                break
            if extracted:
                continue

        return created

    def _attempt_resource_path_repair(
        self,
        code_dir: Path,
        error_text: str,
        resource_context: dict[str, Any] | None,
        *,
        scope: str = "",
    ) -> list[str]:
        self._remember_mutation_snapshot_entry(None)
        materialized = self._materialize_missing_resource_targets(code_dir, error_text, resource_context)
        replacements = self._resource_replacement_map(code_dir, error_text, resource_context)
        snapshot_batch: list[dict[str, Any]] = []
        for created_path_text in materialized:
            snapshot_batch.append(
                capture_repair_snapshot(
                    self.workspace.path,
                    Path(created_path_text),
                    namespace="resource_path_repair",
                    root_dir=self.workspace.path,
                    existed_before=False,
                    operation="create",
                )
            )
        if not replacements:
            self._record_snapshot_batch(
                mutation_kind="resource_path_repair",
                scope=scope or "resource_path_repair",
                snapshots=snapshot_batch,
                metadata={
                    "modified_files": [],
                    "materialized_files": list(materialized),
                },
            )
            return materialized

        text_suffixes = {".py", ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".txt"}
        modified_files: list[str] = []
        for candidate in code_dir.rglob("*"):
            if not candidate.is_file():
                continue
            if candidate.suffix.lower() not in text_suffixes:
                continue
            try:
                original = candidate.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            updated = original
            for old_value, new_value in replacements.items():
                if old_value and old_value in updated:
                    updated = updated.replace(old_value, new_value)

            if updated != original:
                snapshot = capture_repair_snapshot(
                    self.workspace.path,
                    candidate,
                    namespace="resource_path_repair",
                    root_dir=self.workspace.path,
                    operation="rewrite",
                )
                try:
                    candidate.write_text(updated, encoding="utf-8")
                except OSError:
                    continue

                if candidate.suffix.lower() == ".py" and not ExperimentAgent._check_syntax(candidate):
                    self.log(f"Resource-path repair produced invalid syntax in {candidate}, rolling back")
                    rollback_snapshot(self.workspace.path, candidate, snapshot)
                    snapshot["rolled_back"] = True
                    snapshot["rollback_reason"] = "syntax_error"
                    snapshot_batch.append(snapshot)
                    continue

                modified_files.append(str(candidate.relative_to(code_dir)))
                snapshot_batch.append(snapshot)

        self._record_snapshot_batch(
            mutation_kind="resource_path_repair",
            scope=scope or "resource_path_repair",
            snapshots=snapshot_batch,
            metadata={
                "modified_files": list(modified_files),
                "materialized_files": list(materialized),
            },
        )
        return [*materialized, *modified_files]

    @staticmethod
    def _extract_missing_required_options(error_text: str) -> list[str]:
        options: list[str] = []
        patterns = [
            r"""the following arguments are required:\s*([^\n\r]+)""",
            r"""Missing option ['"]?(--[A-Za-z0-9][A-Za-z0-9_-]*)['"]?""",
            r"""argument\s+(--[A-Za-z0-9][A-Za-z0-9_-]*)\s*:\s*expected one argument""",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, error_text, re.IGNORECASE):
                payload = str(match.group(1)).strip()
                if not payload:
                    continue
                if payload.startswith("--"):
                    extracted = re.findall(r"--[A-Za-z0-9][A-Za-z0-9_-]*", payload)
                    if extracted:
                        for option in extracted:
                            if option not in options:
                                options.append(option)
                        continue
                    if payload not in options:
                        options.append(payload)
                    continue
                for option in re.findall(r"--[A-Za-z0-9][A-Za-z0-9_-]*", payload):
                    if option not in options:
                        options.append(option)
        return options

    @staticmethod
    def _extract_unrecognized_options(error_text: str) -> list[str]:
        options: list[str] = []
        patterns = [
            r"""unrecognized arguments:\s*([^\n\r]+)""",
            r"""No such option:\s*(--[A-Za-z0-9][A-Za-z0-9_-]*)""",
            r"""no such option:\s*(--[A-Za-z0-9][A-Za-z0-9_-]*)""",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, error_text, re.IGNORECASE):
                payload = str(match.group(1)).strip()
                if not payload:
                    continue
                extracted = re.findall(r"--[A-Za-z0-9][A-Za-z0-9_-]*", payload)
                if extracted:
                    for option in extracted:
                        if option not in options:
                            options.append(option)
                    continue
                if payload.startswith("--") and payload not in options:
                    options.append(payload)
        return options

    @staticmethod
    def _strip_command_option(tokens: list[str], option: str) -> list[str]:
        updated: list[str] = []
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if token == option:
                index += 1
                if index < len(tokens) and not tokens[index].startswith("--"):
                    index += 1
                continue
            if token.startswith(f"{option}="):
                index += 1
                continue
            updated.append(token)
            index += 1
        return updated

    @staticmethod
    def _command_option_present(tokens: list[str], options: list[str]) -> tuple[str, int, str]:
        for option in options:
            for index, token in enumerate(tokens):
                if token == option:
                    value = tokens[index + 1] if index + 1 < len(tokens) else ""
                    return option, index, value
                if token.startswith(f"{option}="):
                    return option, index, token.split("=", 1)[1]
        return "", -1, ""

    @staticmethod
    def _path_variants(code_dir: Path, path_value: str) -> set[str]:
        normalized = str(path_value or "").strip()
        if not normalized:
            return set()

        variants: set[str] = {os.path.normcase(os.path.normpath(normalized))}
        candidate = Path(normalized)
        if not candidate.is_absolute():
            candidate = code_dir / candidate
        variants.add(os.path.normcase(os.path.normpath(str(candidate))))
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        variants.add(os.path.normcase(os.path.normpath(str(resolved))))
        return variants

    @classmethod
    def _option_value_matches_missing_target(
        cls,
        code_dir: Path,
        option_value: str,
        missing_targets: list[str],
    ) -> bool:
        option_variants = cls._path_variants(code_dir, option_value)
        if not option_variants:
            return False
        for target in missing_targets:
            target_variants = cls._path_variants(code_dir, target)
            if option_variants & target_variants:
                return True
        return False

    @staticmethod
    def _command_entry_script(tokens: list[str], code_dir: Path) -> Path | None:
        for token in tokens:
            normalized = str(token or "").strip()
            if not normalized:
                continue
            if normalized in {"-m", "-c"}:
                return None
            if normalized.endswith(".py"):
                candidate = Path(normalized)
                return candidate if candidate.is_absolute() else code_dir / candidate
        return None

    @staticmethod
    def _entry_script_supports_flag(entry_script: Path | None, flag: str) -> bool:
        if not entry_script or not entry_script.exists():
            return False
        try:
            content = entry_script.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False
        normalized_flag = flag.lstrip("-").replace("-", "_")
        return flag in content or normalized_flag in content

    @classmethod
    def _resume_failure_signals(cls, error_text: str) -> list[str]:
        lower = str(error_text or "").lower()
        signals: list[str] = []
        signal_map = {
            "timed out": "timeout",
            "timeout": "timeout",
            "keyboardinterrupt": "keyboard_interrupt",
            "interrupted": "interrupted",
            "sigterm": "sigterm",
            "terminated": "terminated",
            "preempt": "preempted",
            "cancelled": "cancelled",
            "node_fail": "node_fail",
            "node fail": "node_fail",
        }
        for token, label in signal_map.items():
            if token in lower and label not in signals:
                signals.append(label)
        return signals

    @staticmethod
    def _choose_single_path(candidates: list[Path]) -> Path | None:
        unique: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                resolved = candidate.resolve()
            except OSError:
                resolved = candidate
            key = str(resolved)
            if key in seen:
                continue
            seen.add(key)
            unique.append(candidate)
        if len(unique) == 1:
            return unique[0]
        return None

    @staticmethod
    def _choose_latest_path(candidates: list[Path]) -> Path | None:
        unique: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                resolved = candidate.resolve()
            except OSError:
                resolved = candidate
            key = str(resolved)
            if key in seen:
                continue
            seen.add(key)
            unique.append(candidate)
        if not unique:
            return None

        def sort_key(path: Path) -> tuple[float, str]:
            try:
                mtime = path.stat().st_mtime
            except OSError:
                mtime = -1.0
            try:
                resolved = path.resolve()
            except OSError:
                resolved = path
            return mtime, str(resolved)

        return sorted(unique, key=sort_key, reverse=True)[0]

    @classmethod
    def _keyword_path_candidate(
        cls,
        candidates: list[Path],
        keywords: tuple[str, ...],
        *,
        files_only: bool = False,
        dirs_only: bool = False,
        allow_latest: bool = False,
    ) -> Path | None:
        normalized_keywords = tuple(str(keyword or "").strip().lower() for keyword in keywords if keyword)
        if not normalized_keywords:
            return None

        scored: list[tuple[int, Path]] = []
        for candidate in candidates:
            if files_only and not candidate.is_file():
                continue
            if dirs_only and not candidate.is_dir():
                continue
            haystacks = {
                candidate.name.lower(),
                cls._normalized_resource_key(str(candidate)),
            }
            parent = candidate.parent
            if parent != candidate:
                haystacks.add(parent.name.lower())
                haystacks.add(cls._normalized_resource_key(str(parent)))
            score = sum(1 for keyword in normalized_keywords if any(keyword in value for value in haystacks))
            if score > 0:
                scored.append((score, candidate))

        if not scored:
            return None
        best_score = max(score for score, _candidate in scored)
        best_candidates = [candidate for score, candidate in scored if score == best_score]
        match = cls._choose_single_path(best_candidates)
        if match is not None:
            return match
        if allow_latest:
            return cls._choose_latest_path(best_candidates)
        return None

    @classmethod
    def _runtime_config_candidate(
        cls,
        code_dir: Path,
        resource_context: dict[str, Any] | None,
    ) -> str | None:
        resources = cls._collect_resource_candidates(code_dir, resource_context)
        config_paths = [
            Path(item["path"])
            for item in resources
            if item.get("kind") == "config" and Path(item["path"]).is_file()
        ]
        preferred = [
            path
            for path in config_paths
            if path.name.lower().startswith("config_auto")
            or path.name.lower().startswith("default.")
        ]
        match = cls._choose_single_path(preferred) or cls._choose_single_path(config_paths)
        return str(match) if match is not None else None

    @classmethod
    def _runtime_dataset_candidates(
        cls,
        code_dir: Path,
        resource_context: dict[str, Any] | None,
    ) -> list[Path]:
        resources = cls._collect_resource_candidates(code_dir, resource_context)
        candidates = [
            Path(item["path"])
            for item in resources
            if item.get("kind") == "dataset" and Path(item["path"]).is_file()
        ]
        unique: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except OSError:
                resolved = candidate
            key = str(resolved)
            if key in seen:
                continue
            seen.add(key)
            unique.append(candidate)
        return unique

    @classmethod
    def _runtime_dataset_directory_candidates(
        cls,
        code_dir: Path,
        resource_context: dict[str, Any] | None,
    ) -> list[Path]:
        resources = cls._collect_resource_candidates(code_dir, resource_context)
        candidates = [
            Path(item["path"])
            for item in resources
            if item.get("kind") == "dataset" and Path(item["path"]).is_dir()
        ]
        unique: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except OSError:
                resolved = candidate
            key = str(resolved)
            if key in seen:
                continue
            seen.add(key)
            unique.append(candidate)
        return unique

    @classmethod
    def _runtime_model_candidates(
        cls,
        code_dir: Path,
        resource_context: dict[str, Any] | None,
    ) -> list[Path]:
        resources = cls._collect_resource_candidates(code_dir, resource_context)
        candidates = [
            Path(item["path"])
            for item in resources
            if item.get("kind") == "model" and Path(item["path"]).is_file()
        ]
        unique: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except OSError:
                resolved = candidate
            key = str(resolved)
            if key in seen:
                continue
            seen.add(key)
            unique.append(candidate)
        return unique

    @classmethod
    def _runtime_option_candidate(
        cls,
        code_dir: Path,
        option: str,
        resource_context: dict[str, Any] | None,
    ) -> str | None:
        normalized = str(option or "").strip().lower()
        if not normalized:
            return None

        config_options = {"--config", "--config-path", "--cfg", "--config-file"}
        data_dir_options = {"--data-dir", "--data-root", "--dataset-dir", "--dataset-root", "--data", "--dataset"}
        data_file_options = {"--data-path", "--dataset-path", "--input-path", "--input-file", "--dataset-file"}
        train_file_options = {"--train-file", "--train-data", "--train-path"}
        val_file_options = {
            "--val-file",
            "--valid-file",
            "--validation-file",
            "--val-data",
            "--valid-data",
            "--validation-data",
            "--val-path",
            "--valid-path",
            "--dev-file",
            "--dev-data",
            "--dev-path",
        }
        test_file_options = {"--test-file", "--test-data", "--test-path"}
        labels_options = {"--labels-path", "--label-file", "--labels-file", "--label-path"}
        annotations_options = {"--annotations", "--annotation-file", "--annotation-path", "--annotations-file"}
        split_file_options = {"--split-file", "--splits-file", "--split-path", "--fold-file", "--folds-file"}
        metadata_options = {"--metadata-path", "--meta-path", "--metadata-file", "--meta-file"}
        image_dir_options = {"--image-dir", "--images-dir", "--image-root", "--images-root"}
        label_dir_options = {"--label-dir", "--labels-dir", "--label-root", "--labels-root"}
        model_dir_options = {"--model-dir", "--model-root"}
        model_file_options = {"--model-path", "--model-file", "--pretrained-model"}
        tokenizer_options = {"--tokenizer-path", "--tokenizer-name-or-path"}
        checkpoint_options = {"--checkpoint", "--ckpt", "--checkpoint-path"}
        resume_options = {"--resume", "--resume-from", "--resume-path"}
        checkpoint_dir_options = {"--checkpoint-dir", "--ckpt-dir"}
        output_dir_options = {"--output-dir", "--results-dir", "--save-dir"}
        log_dir_options = {"--log-dir", "--logging-dir"}

        if normalized in config_options:
            return cls._runtime_config_candidate(code_dir, resource_context)

        if normalized in output_dir_options:
            return str((code_dir / "results").resolve())
        if normalized in checkpoint_dir_options:
            return str((code_dir / "checkpoints").resolve())
        if normalized in log_dir_options:
            return str((code_dir / "logs").resolve())

        if normalized in data_dir_options:
            resource_dir = str(resource_context.get("data_dir", "")).strip() if isinstance(resource_context, dict) else ""
            if resource_dir and Path(resource_dir).exists():
                return str(Path(resource_dir).resolve())
            return_value = cls._choose_single_path([code_dir / "data", code_dir / "datasets"])
            return str(return_value.resolve()) if return_value is not None else None

        if normalized in model_dir_options:
            resource_dir = str(resource_context.get("models_dir", "")).strip() if isinstance(resource_context, dict) else ""
            if resource_dir and Path(resource_dir).exists():
                return str(Path(resource_dir).resolve())
            return_value = cls._choose_single_path([code_dir / "models", code_dir / "checkpoints"])
            return str(return_value.resolve()) if return_value is not None else None

        dataset_files = cls._runtime_dataset_candidates(code_dir, resource_context)
        dataset_dirs = cls._runtime_dataset_directory_candidates(code_dir, resource_context)
        if normalized in train_file_options:
            train_match = cls._keyword_path_candidate(dataset_files, ("train",), files_only=True)
            if train_match is not None:
                return str(train_match.resolve())
            fallback = cls._choose_single_path(dataset_files)
            return str(fallback.resolve()) if fallback is not None else None
        if normalized in val_file_options:
            val_match = cls._keyword_path_candidate(dataset_files, ("val", "valid", "validation", "dev"), files_only=True)
            return str(val_match.resolve()) if val_match is not None else None
        if normalized in test_file_options:
            test_match = cls._keyword_path_candidate(dataset_files, ("test",), files_only=True)
            return str(test_match.resolve()) if test_match is not None else None
        if normalized in labels_options:
            label_match = cls._keyword_path_candidate(dataset_files, ("label", "labels"), files_only=True)
            return str(label_match.resolve()) if label_match is not None else None
        if normalized in annotations_options:
            annotations_match = cls._keyword_path_candidate(
                dataset_files,
                ("annot", "annotation", "annotations", "anno"),
                files_only=True,
            )
            return str(annotations_match.resolve()) if annotations_match is not None else None
        if normalized in split_file_options:
            split_match = cls._keyword_path_candidate(
                dataset_files,
                ("split", "splits", "fold", "folds"),
                files_only=True,
            )
            return str(split_match.resolve()) if split_match is not None else None
        if normalized in metadata_options:
            meta_match = cls._keyword_path_candidate(dataset_files, ("meta", "metadata"), files_only=True)
            return str(meta_match.resolve()) if meta_match is not None else None
        if normalized in image_dir_options:
            image_dir_match = cls._keyword_path_candidate(dataset_dirs, ("image", "images", "img"), dirs_only=True)
            return str(image_dir_match.resolve()) if image_dir_match is not None else None
        if normalized in label_dir_options:
            label_dir_match = cls._keyword_path_candidate(
                dataset_dirs,
                ("label", "labels", "mask", "masks"),
                dirs_only=True,
            )
            return str(label_dir_match.resolve()) if label_dir_match is not None else None
        if normalized in data_file_options:
            fallback = cls._choose_single_path(dataset_files)
            return str(fallback.resolve()) if fallback is not None else None

        model_files = cls._runtime_model_candidates(code_dir, resource_context)
        if normalized in model_file_options:
            match = cls._choose_single_path(model_files)
            return str(match.resolve()) if match is not None else None
        if normalized in tokenizer_options:
            match = cls._choose_single_path([path for path in model_files if "token" in path.name.lower()])
            return str(match.resolve()) if match is not None else None
        if normalized in checkpoint_options or normalized in resume_options:
            checkpoint_files = [
                path for path in model_files if path.suffix.lower() in {".pt", ".pth", ".ckpt", ".bin", ".safetensors"}
            ]
            preferred = [
                path
                for path in checkpoint_files
                if any(token in str(path).lower() for token in ("checkpoint", "checkpoints", "ckpt"))
            ]
            match = cls._choose_single_path(preferred)
            if match is None and preferred:
                match = cls._choose_latest_path(preferred)
            if match is None:
                match = cls._choose_single_path(checkpoint_files)
            if match is None and checkpoint_files:
                match = cls._choose_latest_path(checkpoint_files)
            if match is not None:
                return str(match.resolve())
            fallback = cls._choose_single_path(model_files)
            return str(fallback.resolve()) if fallback is not None else None

        return None

    @staticmethod
    def _upsert_command_option(tokens: list[str], option: str, value: str) -> list[str]:
        updated = list(tokens)
        for index, token in enumerate(updated):
            if token == option:
                if index + 1 < len(updated) and not updated[index + 1].startswith("--"):
                    updated[index + 1] = value
                else:
                    updated.insert(index + 1, value)
                return updated
            if token.startswith(f"{option}="):
                updated[index] = f"{option}={value}"
                return updated
        return [*updated, option, value]

    def _attempt_required_argument_repair(
        self,
        code_dir: Path,
        error_text: str,
        resource_context: dict[str, Any] | None,
        *,
        scope: str = "",
    ) -> list[str]:
        self._remember_mutation_snapshot_entry(None)
        required_options = self._extract_missing_required_options(error_text)
        if not required_options:
            self._record_snapshot_batch(
                mutation_kind="required_argument_repair",
                scope=scope or "required_argument_repair",
                snapshots=[],
                metadata={"modified_files": [], "required_options": []},
            )
            return []

        runner_config_path = code_dir / RUNNER_CONFIG_NAME
        if not runner_config_path.exists():
            self._record_snapshot_batch(
                mutation_kind="required_argument_repair",
                scope=scope or "required_argument_repair",
                snapshots=[],
                metadata={"modified_files": [], "required_options": list(required_options)},
            )
            return []

        try:
            payload = json.loads(runner_config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._record_snapshot_batch(
                mutation_kind="required_argument_repair",
                scope=scope or "required_argument_repair",
                snapshots=[],
                metadata={"modified_files": [], "required_options": list(required_options)},
            )
            return []

        target_command = payload.get("target_command")
        if not isinstance(target_command, list):
            target_command = []
        updated_command = [str(token) for token in target_command]
        repairs: list[dict[str, str]] = []

        for option in required_options:
            candidate = self._runtime_option_candidate(code_dir, option, resource_context)
            if not candidate:
                continue
            new_command = self._upsert_command_option(updated_command, option, candidate)
            if new_command != updated_command:
                updated_command = new_command
                repairs.append({"option": option, "value": candidate})

        if not repairs:
            self._record_snapshot_batch(
                mutation_kind="required_argument_repair",
                scope=scope or "required_argument_repair",
                snapshots=[],
                metadata={"modified_files": [], "required_options": list(required_options)},
            )
            return []

        snapshot = capture_repair_snapshot(
            self.workspace.path,
            runner_config_path,
            namespace="required_argument_repair",
            root_dir=self.workspace.path,
            operation="rewrite",
        )
        payload["target_command"] = updated_command
        try:
            runner_config_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            rollback_snapshot(self.workspace.path, runner_config_path, snapshot)
            snapshot["rolled_back"] = True
            snapshot["rollback_reason"] = "write_error"
            self._record_snapshot_batch(
                mutation_kind="required_argument_repair",
                scope=scope or "required_argument_repair",
                snapshots=[snapshot],
                metadata={"modified_files": [], "required_options": list(required_options)},
            )
            return []

        self._record_snapshot_batch(
            mutation_kind="required_argument_repair",
            scope=scope or "required_argument_repair",
            snapshots=[snapshot],
            metadata={
                "modified_files": [str(runner_config_path.relative_to(code_dir))],
                "required_options": list(required_options),
                "repairs": list(repairs),
            },
        )
        return [str(runner_config_path.relative_to(code_dir))]

    def _attempt_resume_repair(
        self,
        code_dir: Path,
        error_text: str,
        resource_context: dict[str, Any] | None,
        *,
        scope: str = "",
    ) -> list[str]:
        self._remember_mutation_snapshot_entry(None)
        failure_signals = self._resume_failure_signals(error_text)
        if not failure_signals:
            self._record_snapshot_batch(
                mutation_kind="resume_repair",
                scope=scope or "resume_repair",
                snapshots=[],
                metadata={"modified_files": [], "failure_signals": []},
            )
            return []

        runner_config_path = code_dir / RUNNER_CONFIG_NAME
        if not runner_config_path.exists():
            self._record_snapshot_batch(
                mutation_kind="resume_repair",
                scope=scope or "resume_repair",
                snapshots=[],
                metadata={"modified_files": [], "failure_signals": list(failure_signals)},
            )
            return []

        try:
            payload = json.loads(runner_config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._record_snapshot_batch(
                mutation_kind="resume_repair",
                scope=scope or "resume_repair",
                snapshots=[],
                metadata={"modified_files": [], "failure_signals": list(failure_signals)},
            )
            return []

        target_command = payload.get("target_command")
        if not isinstance(target_command, list):
            target_command = []
        updated_command = [str(token) for token in target_command]
        entry_script = self._command_entry_script(updated_command, code_dir)

        option_groups = [
            ["--resume", "--resume-from", "--resume-path"],
            ["--checkpoint", "--ckpt", "--checkpoint-path"],
        ]
        repairs: list[dict[str, str]] = []
        for options in option_groups:
            existing_option, _index, current_value = self._command_option_present(updated_command, options)
            supported = [option for option in options if self._entry_script_supports_flag(entry_script, option)]
            chosen_option = existing_option or (supported[0] if supported else "")
            if not chosen_option:
                continue
            candidate = self._runtime_option_candidate(code_dir, chosen_option, resource_context)
            if not candidate:
                continue
            candidate_variants = self._path_variants(code_dir, candidate)
            current_variants = self._path_variants(code_dir, current_value)
            if current_value and current_variants & candidate_variants and Path(candidate).exists():
                continue
            new_command = self._upsert_command_option(updated_command, chosen_option, candidate)
            if new_command != updated_command:
                repairs.append(
                    {
                        "option": chosen_option,
                        "old_value": current_value,
                        "new_value": candidate,
                    }
                )
                updated_command = new_command
                break

        if not repairs:
            self._record_snapshot_batch(
                mutation_kind="resume_repair",
                scope=scope or "resume_repair",
                snapshots=[],
                metadata={"modified_files": [], "failure_signals": list(failure_signals)},
            )
            return []

        snapshot = capture_repair_snapshot(
            self.workspace.path,
            runner_config_path,
            namespace="resume_repair",
            root_dir=self.workspace.path,
            operation="rewrite",
        )
        payload["target_command"] = updated_command
        try:
            runner_config_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            rollback_snapshot(self.workspace.path, runner_config_path, snapshot)
            snapshot["rolled_back"] = True
            snapshot["rollback_reason"] = "write_error"
            self._record_snapshot_batch(
                mutation_kind="resume_repair",
                scope=scope or "resume_repair",
                snapshots=[snapshot],
                metadata={"modified_files": [], "failure_signals": list(failure_signals)},
            )
            return []

        self._record_snapshot_batch(
            mutation_kind="resume_repair",
            scope=scope or "resume_repair",
            snapshots=[snapshot],
            metadata={
                "modified_files": [str(runner_config_path.relative_to(code_dir))],
                "failure_signals": list(failure_signals),
                "repairs": list(repairs),
            },
        )
        return [str(runner_config_path.relative_to(code_dir))]

    def _attempt_cluster_resume_repair(
        self,
        code_dir: Path,
        final_status: str,
        results: dict[str, Any],
        resource_context: dict[str, Any] | None,
        *,
        scope: str = "",
    ) -> list[str]:
        checkpoints = results.get("checkpoints") if isinstance(results.get("checkpoints"), list) else []
        if not checkpoints and not list((code_dir / "checkpoints").glob("*")):
            self._remember_mutation_snapshot_entry(None)
            self._record_snapshot_batch(
                mutation_kind="resume_repair",
                scope=scope or "cluster_resume_repair",
                snapshots=[],
                metadata={"modified_files": [], "reason": "no_checkpoints"},
            )
            return []

        error_text = "\n".join(
            part
            for part in [
                str(final_status or "").strip(),
                str(results.get("stdout_log") or "").strip(),
                str(results.get("stderr_log") or "").strip(),
            ]
            if part
        )
        return self._attempt_resume_repair(
            code_dir,
            error_text,
            resource_context,
            scope=scope or "cluster_resume_repair",
        )

    def _attempt_option_value_repair(
        self,
        code_dir: Path,
        error_text: str,
        resource_context: dict[str, Any] | None,
        *,
        scope: str = "",
    ) -> list[str]:
        self._remember_mutation_snapshot_entry(None)
        missing_targets = self._extract_missing_resource_targets(error_text)
        if not missing_targets:
            self._record_snapshot_batch(
                mutation_kind="option_value_repair",
                scope=scope or "option_value_repair",
                snapshots=[],
                metadata={"modified_files": [], "missing_targets": []},
            )
            return []

        runner_config_path = code_dir / RUNNER_CONFIG_NAME
        if not runner_config_path.exists():
            self._record_snapshot_batch(
                mutation_kind="option_value_repair",
                scope=scope or "option_value_repair",
                snapshots=[],
                metadata={"modified_files": [], "missing_targets": list(missing_targets)},
            )
            return []

        try:
            payload = json.loads(runner_config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._record_snapshot_batch(
                mutation_kind="option_value_repair",
                scope=scope or "option_value_repair",
                snapshots=[],
                metadata={"modified_files": [], "missing_targets": list(missing_targets)},
            )
            return []

        target_command = payload.get("target_command")
        if not isinstance(target_command, list):
            target_command = []
        updated_command = [str(token) for token in target_command]
        repairs: list[dict[str, str]] = []
        option_groups = [
            ["--config", "--config-path", "--cfg", "--config-file"],
            ["--data-dir", "--data-root", "--dataset-dir", "--dataset-root", "--data", "--dataset"],
            ["--data-path", "--dataset-path", "--input-path", "--input-file", "--dataset-file"],
            ["--train-file", "--train-data", "--train-path"],
            [
                "--val-file",
                "--valid-file",
                "--validation-file",
                "--val-data",
                "--valid-data",
                "--validation-data",
                "--val-path",
                "--valid-path",
                "--dev-file",
                "--dev-data",
                "--dev-path",
            ],
            ["--test-file", "--test-data", "--test-path"],
            ["--labels-path", "--label-file", "--labels-file", "--label-path"],
            ["--annotations", "--annotation-file", "--annotation-path", "--annotations-file"],
            ["--split-file", "--splits-file", "--split-path", "--fold-file", "--folds-file"],
            ["--metadata-path", "--meta-path", "--metadata-file", "--meta-file"],
            ["--image-dir", "--images-dir", "--image-root", "--images-root"],
            ["--label-dir", "--labels-dir", "--label-root", "--labels-root"],
            ["--model-dir", "--model-root"],
            ["--model-path", "--model-file", "--pretrained-model"],
            ["--tokenizer-path", "--tokenizer-name-or-path"],
            ["--checkpoint", "--ckpt", "--checkpoint-path"],
            ["--resume", "--resume-from", "--resume-path"],
        ]
        for options in option_groups:
            option, _index, current_value = self._command_option_present(updated_command, options)
            if not option or not current_value:
                continue
            if not self._option_value_matches_missing_target(code_dir, current_value, missing_targets):
                continue
            candidate = self._runtime_option_candidate(code_dir, option, resource_context)
            if not candidate:
                continue
            if self._path_variants(code_dir, current_value) & self._path_variants(code_dir, candidate):
                continue
            new_command = self._upsert_command_option(updated_command, option, candidate)
            if new_command != updated_command:
                repairs.append({"option": option, "old_value": current_value, "new_value": candidate})
                updated_command = new_command

        if not repairs:
            self._record_snapshot_batch(
                mutation_kind="option_value_repair",
                scope=scope or "option_value_repair",
                snapshots=[],
                metadata={"modified_files": [], "missing_targets": list(missing_targets)},
            )
            return []

        snapshot = capture_repair_snapshot(
            self.workspace.path,
            runner_config_path,
            namespace="option_value_repair",
            root_dir=self.workspace.path,
            operation="rewrite",
        )
        payload["target_command"] = updated_command
        try:
            runner_config_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            rollback_snapshot(self.workspace.path, runner_config_path, snapshot)
            snapshot["rolled_back"] = True
            snapshot["rollback_reason"] = "write_error"
            self._record_snapshot_batch(
                mutation_kind="option_value_repair",
                scope=scope or "option_value_repair",
                snapshots=[snapshot],
                metadata={"modified_files": [], "missing_targets": list(missing_targets)},
            )
            return []

        self._record_snapshot_batch(
            mutation_kind="option_value_repair",
            scope=scope or "option_value_repair",
            snapshots=[snapshot],
            metadata={
                "modified_files": [str(runner_config_path.relative_to(code_dir))],
                "missing_targets": list(missing_targets),
                "repairs": list(repairs),
            },
        )
        return [str(runner_config_path.relative_to(code_dir))]

    def _attempt_unrecognized_argument_repair(
        self,
        code_dir: Path,
        error_text: str,
        *,
        mode: str = "",
        scope: str = "",
    ) -> list[str]:
        self._remember_mutation_snapshot_entry(None)
        unknown_options = self._extract_unrecognized_options(error_text)
        if not unknown_options:
            self._record_snapshot_batch(
                mutation_kind="unrecognized_argument_repair",
                scope=scope or "unrecognized_argument_repair",
                snapshots=[],
                metadata={"modified_files": [], "unknown_options": []},
            )
            return []

        runner_config_path = code_dir / RUNNER_CONFIG_NAME
        if not runner_config_path.exists():
            self._record_snapshot_batch(
                mutation_kind="unrecognized_argument_repair",
                scope=scope or "unrecognized_argument_repair",
                snapshots=[],
                metadata={"modified_files": [], "unknown_options": list(unknown_options)},
            )
            return []

        try:
            payload = json.loads(runner_config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._record_snapshot_batch(
                mutation_kind="unrecognized_argument_repair",
                scope=scope or "unrecognized_argument_repair",
                snapshots=[],
                metadata={"modified_files": [], "unknown_options": list(unknown_options)},
            )
            return []

        target_command = payload.get("target_command")
        if not isinstance(target_command, list):
            target_command = []
        updated_command = [str(token) for token in target_command]
        blocked_quick_eval = {
            str(option).strip()
            for option in payload.get("quick_eval_blocked_options", [])
            if isinstance(option, str) and str(option).strip().startswith("--")
        }
        removed_options: list[str] = []
        blocked_options_added: list[str] = []

        for option in unknown_options:
            new_command = self._strip_command_option(updated_command, option)
            if new_command != updated_command:
                updated_command = new_command
                removed_options.append(option)
                continue
            if mode == "quick-eval" and option in QUICK_EVAL_AUTO_OPTIONS and option not in blocked_quick_eval:
                blocked_quick_eval.add(option)
                blocked_options_added.append(option)

        if not removed_options and not blocked_options_added:
            self._record_snapshot_batch(
                mutation_kind="unrecognized_argument_repair",
                scope=scope or "unrecognized_argument_repair",
                snapshots=[],
                metadata={"modified_files": [], "unknown_options": list(unknown_options)},
            )
            return []

        snapshot = capture_repair_snapshot(
            self.workspace.path,
            runner_config_path,
            namespace="unrecognized_argument_repair",
            root_dir=self.workspace.path,
            operation="rewrite",
        )
        payload["target_command"] = updated_command
        if blocked_quick_eval:
            payload["quick_eval_blocked_options"] = sorted(blocked_quick_eval)
        else:
            payload.pop("quick_eval_blocked_options", None)
        try:
            runner_config_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            rollback_snapshot(self.workspace.path, runner_config_path, snapshot)
            snapshot["rolled_back"] = True
            snapshot["rollback_reason"] = "write_error"
            self._record_snapshot_batch(
                mutation_kind="unrecognized_argument_repair",
                scope=scope or "unrecognized_argument_repair",
                snapshots=[snapshot],
                metadata={"modified_files": [], "unknown_options": list(unknown_options)},
            )
            return []

        self._record_snapshot_batch(
            mutation_kind="unrecognized_argument_repair",
            scope=scope or "unrecognized_argument_repair",
            snapshots=[snapshot],
            metadata={
                "modified_files": [str(runner_config_path.relative_to(code_dir))],
                "unknown_options": list(unknown_options),
                "removed_options": list(removed_options),
                "quick_eval_blocked_options": list(blocked_options_added),
            },
        )
        return [str(runner_config_path.relative_to(code_dir))]

    @staticmethod
    def _extract_missing_modules(error_text: str) -> list[str]:
        modules: list[str] = []
        patterns = [
            r"""No module named ['"]([A-Za-z0-9_.-]+)['"]""",
            r"""ModuleNotFoundError:\s*No module named ['"]([A-Za-z0-9_.-]+)['"]""",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, error_text):
                module_name = str(match.group(1)).strip().split(".")[0]
                if module_name and module_name not in modules:
                    modules.append(module_name)
        return modules

    @staticmethod
    def _extract_nltk_resources(error_text: str) -> list[str]:
        resources: list[str] = []
        for pattern in [
            r"""nltk\.download\(['"]([^'"]+)['"]\)""",
            r"""Resource\s+([A-Za-z0-9_./-]+)\s+not found""",
        ]:
            for match in re.finditer(pattern, error_text, re.IGNORECASE):
                resource_name = str(match.group(1)).strip().strip("/")
                if resource_name and resource_name not in resources:
                    resources.append(resource_name)
        return resources

    @classmethod
    def _candidate_package_names(
        cls,
        module_name: str,
        code_dir: Path,
    ) -> list[str]:
        normalized = module_name.strip()
        if not normalized or not re.fullmatch(r"[A-Za-z0-9_.-]+", normalized):
            return []

        local_module = code_dir / f"{normalized}.py"
        local_package = code_dir / normalized
        if local_module.exists() or local_package.exists():
            return []

        candidates: list[str] = []
        alias = MODULE_PACKAGE_ALIASES.get(normalized.lower())
        if alias:
            candidates.append(alias)
        candidates.append(normalized)
        if "_" in normalized:
            candidates.append(normalized.replace("_", "-"))

        deduped: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            if candidate not in seen:
                deduped.append(candidate)
                seen.add(candidate)
        return deduped

    async def _attempt_runtime_remediation(
        self,
        code_dir: Path,
        error_text: str,
        *,
        runtime_python: str,
        fix_history: list[dict[str, Any]] | None = None,
        execution_policy: ExperimentExecutionPolicy | None = None,
        remediation_ledger: list[dict[str, Any]] | None = None,
        mode: str = "",
        cycle: int | None = None,
        signature: str = "",
        round_number: int | None = None,
    ) -> list[str]:
        policy = execution_policy or RuntimeEnvironmentManager(
            self.config,
            self.log,
        ).build_execution_policy(code_dir)
        actions: list[str] = []

        nltk_resources = self._extract_nltk_resources(error_text)
        remaining_nltk_downloads = policy.remaining_nltk_downloads(fix_history)
        for resource_name in nltk_resources[:remaining_nltk_downloads]:
            result = await self._run_subprocess(
                [
                    runtime_python,
                    "-c",
                    (
                        "import nltk; "
                        f"nltk.download({resource_name!r}, quiet=True, raise_on_error=True)"
                    ),
                ],
                cwd=code_dir,
                timeout=300,
            )
            if result.get("returncode") == 0:
                actions.append(f"nltk:{resource_name}")
                self._append_remediation_entry(
                    remediation_ledger,
                    kind="nltk_download",
                    status="applied",
                    scope=f"local_{mode.replace('-', '_')}" if mode else "local_runtime",
                    round_number=round_number,
                    cycle=cycle,
                    signature=signature,
                    details={"resource": resource_name, "runtime_python": runtime_python},
                )

        if actions:
            return actions
        if nltk_resources and remaining_nltk_downloads <= 0:
            self.log("Skipped NLTK auto-download because the execution policy budget is exhausted")
            self._append_remediation_entry(
                remediation_ledger,
                kind="nltk_download",
                status="skipped",
                scope=f"local_{mode.replace('-', '_')}" if mode else "local_runtime",
                round_number=round_number,
                cycle=cycle,
                signature=signature,
                reason="budget_exhausted",
                details={"resources": list(nltk_resources[:3])},
            )

        missing_modules = self._extract_missing_modules(error_text)
        remaining_package_installs = policy.remaining_runtime_auto_installs(fix_history)
        if missing_modules and not policy.runtime_auto_install_enabled:
            self.log("Skipped runtime pip auto-install because it is disabled by execution policy")
            self._append_remediation_entry(
                remediation_ledger,
                kind="pip_install",
                status="skipped",
                scope=f"local_{mode.replace('-', '_')}" if mode else "local_runtime",
                round_number=round_number,
                cycle=cycle,
                signature=signature,
                reason="disabled_by_policy",
                details={"modules": list(missing_modules[:3])},
            )
            return actions
        if missing_modules and remaining_package_installs <= 0:
            self.log("Skipped runtime pip auto-install because the execution policy budget is exhausted")
            self._append_remediation_entry(
                remediation_ledger,
                kind="pip_install",
                status="skipped",
                scope=f"local_{mode.replace('-', '_')}" if mode else "local_runtime",
                round_number=round_number,
                cycle=cycle,
                signature=signature,
                reason="budget_exhausted",
                details={"modules": list(missing_modules[:3])},
            )
            return actions

        for module_name in missing_modules[:3]:
            allowed_candidates = [
                package_name
                for package_name in self._candidate_package_names(module_name, code_dir)
                if policy.allows_runtime_package(
                    package_name,
                    module_name=module_name,
                    aliases=MODULE_PACKAGE_ALIASES,
                )
            ]
            if not allowed_candidates:
                self.log(
                    "Skipped runtime pip auto-install for missing module "
                    f"{module_name!r}: package is not declared or allowlisted"
                )
                self._append_remediation_entry(
                    remediation_ledger,
                    kind="pip_install",
                    status="skipped",
                    scope=f"local_{mode.replace('-', '_')}" if mode else "local_runtime",
                    round_number=round_number,
                    cycle=cycle,
                    signature=signature,
                    reason="not_declared_or_allowlisted",
                    details={"module": module_name},
                )
                continue
            for package_name in allowed_candidates:
                if remaining_package_installs <= 0:
                    break
                result = await self._run_subprocess(
                    [runtime_python, "-m", "pip", "install", package_name],
                    cwd=code_dir,
                    timeout=900,
                )
                if result.get("returncode") == 0:
                    actions.append(f"pip:{package_name}")
                    remaining_package_installs -= 1
                    self._append_remediation_entry(
                        remediation_ledger,
                        kind="pip_install",
                        status="applied",
                        scope=f"local_{mode.replace('-', '_')}" if mode else "local_runtime",
                        round_number=round_number,
                        cycle=cycle,
                        signature=signature,
                        details={"module": module_name, "package": package_name},
                    )
                    break
                self._append_remediation_entry(
                    remediation_ledger,
                    kind="pip_install",
                    status="failed",
                    scope=f"local_{mode.replace('-', '_')}" if mode else "local_runtime",
                    round_number=round_number,
                    cycle=cycle,
                    signature=signature,
                    details={
                        "module": module_name,
                        "package": package_name,
                        "returncode": result.get("returncode"),
                        "stderr": str(result.get("stderr") or "")[:300],
                    },
                )

        return actions

    @classmethod
    def _summarize_available_resources(
        cls,
        code_dir: Path,
        resource_context: dict[str, Any] | None,
    ) -> str:
        candidates = cls._collect_resource_candidates(code_dir, resource_context)
        if not candidates:
            return ""

        lines: list[str] = []
        for item in candidates[:8]:
            lines.append(f"- [{item['kind']}] {item['path']}")
        return "Available workspace resources:\n" + "\n".join(lines)

    def _augment_quick_eval_metrics_from_logs(
        self,
        code_dir: Path,
        quick_eval: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        if quick_eval.get("metrics"):
            return quick_eval
        artifact_results = self._collect_result_artifacts(code_dir)
        artifact_metrics = artifact_results.get("metrics")
        if isinstance(artifact_metrics, dict) and any(
            key in artifact_metrics for key in ("main_results", "ablation_results", "training_log")
        ):
            augmented = {
                **quick_eval,
                "metrics": artifact_metrics,
            }
            recovered_from = str(artifact_results.get("recovered_from") or "").strip()
            if recovered_from:
                augmented["recovered_from"] = recovered_from
            if artifact_results.get("metrics_artifact_materialized"):
                augmented["metrics_artifact_materialized"] = True
                augmented["metrics_artifact_path"] = artifact_results.get("metrics_artifact_path", "")
            return augmented
        recovered = self._recover_metrics_contract_from_logs(result)
        if not recovered:
            return quick_eval
        return {
            **quick_eval,
            "metrics": recovered,
            "recovered_from": "execution_log",
        }

    def _recover_metrics_contract_from_logs(self, result: dict[str, Any]) -> dict[str, Any]:
        log_text = "\n".join(
            part for part in [
                str(result.get("stdout") or "").strip(),
                str(result.get("stderr") or "").strip(),
            ]
            if part
        )
        parsed = self._parse_metrics_from_log(log_text)
        return self._wrap_log_metrics_for_contract(parsed)

    @staticmethod
    def _wrap_log_metrics_for_contract(parsed_metrics: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(parsed_metrics, dict) or not parsed_metrics:
            return {}

        metric_entries: list[dict[str, Any]] = []
        training_log: list[dict[str, Any]] = []
        epoch_losses = parsed_metrics.get("epoch_losses")
        if isinstance(epoch_losses, list):
            for entry in epoch_losses:
                if not isinstance(entry, dict):
                    continue
                epoch = entry.get("epoch")
                loss = entry.get("loss")
                if epoch is None or loss is None:
                    continue
                try:
                    training_log.append({
                        "epoch": int(epoch),
                        "train_loss": float(loss),
                        "metrics": {},
                    })
                except (TypeError, ValueError):
                    continue

        for key, value in parsed_metrics.items():
            if key == "epoch_losses":
                continue
            if isinstance(key, str) and key.isdigit():
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            metric_entries.append({"metric_name": str(key), "value": numeric_value})

        if not metric_entries:
            return {}

        return {
            "main_results": [
                {
                    "method_name": "QuickEvalLog",
                    "dataset": "UNKNOWN",
                    "is_proposed": True,
                    "metrics": metric_entries,
                }
            ],
            "ablation_results": [],
            "training_log": training_log,
        }

    def _materialize_recovered_metrics_artifact(
        self,
        code_dir: Path,
        recovered_metrics: dict[str, Any],
        *,
        source: str,
        scope: str = "",
    ) -> dict[str, Any]:
        self._remember_mutation_snapshot_entry(None)
        artifact_path = "results/metrics.json"
        if not isinstance(recovered_metrics, dict) or not recovered_metrics:
            self._record_snapshot_batch(
                mutation_kind="metrics_artifact_recovery",
                scope=scope or "metrics_artifact_recovery",
                snapshots=[],
                metadata={"modified_files": [], "source": source, "reason": "no_metrics"},
            )
            return {"written": False, "artifact_path": artifact_path, "metrics": {}}

        existing_metrics = ExperimentAgent._parse_metrics_json(code_dir)
        if existing_metrics:
            self._record_snapshot_batch(
                mutation_kind="metrics_artifact_recovery",
                scope=scope or "metrics_artifact_recovery",
                snapshots=[],
                metadata={
                    "modified_files": [],
                    "source": source,
                    "reason": "artifact_already_valid",
                },
            )
            return {"written": False, "artifact_path": artifact_path, "metrics": existing_metrics}

        metrics_path = code_dir / artifact_path
        try:
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            self._record_snapshot_batch(
                mutation_kind="metrics_artifact_recovery",
                scope=scope or "metrics_artifact_recovery",
                snapshots=[],
                metadata={
                    "modified_files": [],
                    "source": source,
                    "reason": "mkdir_failed",
                },
            )
            return {"written": False, "artifact_path": artifact_path, "metrics": recovered_metrics}

        snapshot = capture_repair_snapshot(
            self.workspace.path,
            metrics_path,
            namespace="metrics_artifact_recovery",
            root_dir=self.workspace.path,
            operation="rewrite" if metrics_path.exists() else "create",
        )
        existing_meta = (
            dict(recovered_metrics.get("_nanoresearch_meta"))
            if isinstance(recovered_metrics.get("_nanoresearch_meta"), dict)
            else {}
        )
        payload = {
            "main_results": list(recovered_metrics.get("main_results") or []),
            "ablation_results": list(recovered_metrics.get("ablation_results") or []),
            "training_log": list(recovered_metrics.get("training_log") or []),
            "_nanoresearch_meta": {
                **existing_meta,
                "recovered_from": source,
                "materialized_by": self.__class__.__name__,
            },
        }
        try:
            metrics_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            rollback_snapshot(self.workspace.path, metrics_path, snapshot)
            snapshot["rolled_back"] = True
            snapshot["rollback_reason"] = "write_error"
            self._record_snapshot_batch(
                mutation_kind="metrics_artifact_recovery",
                scope=scope or "metrics_artifact_recovery",
                snapshots=[snapshot],
                metadata={"modified_files": [], "source": source, "reason": "write_error"},
            )
            return {"written": False, "artifact_path": artifact_path, "metrics": recovered_metrics}

        validated_metrics = ExperimentAgent._parse_metrics_json(code_dir)
        if not validated_metrics:
            rollback_snapshot(self.workspace.path, metrics_path, snapshot)
            snapshot["rolled_back"] = True
            snapshot["rollback_reason"] = "validation_failed"
            self._record_snapshot_batch(
                mutation_kind="metrics_artifact_recovery",
                scope=scope or "metrics_artifact_recovery",
                snapshots=[snapshot],
                metadata={
                    "modified_files": [],
                    "source": source,
                    "reason": "validation_failed",
                },
            )
            return {"written": False, "artifact_path": artifact_path, "metrics": recovered_metrics}

        self._record_snapshot_batch(
            mutation_kind="metrics_artifact_recovery",
            scope=scope or "metrics_artifact_recovery",
            snapshots=[snapshot],
            metadata={
                "modified_files": [artifact_path],
                "source": source,
                "training_log_entries": len(validated_metrics.get("training_log") or []),
            },
        )
        return {"written": True, "artifact_path": artifact_path, "metrics": validated_metrics}

    @staticmethod
    def _csv_column_candidates(*names: str) -> tuple[str, ...]:
        candidates: list[str] = []
        for name in names:
            base = str(name or "").strip()
            if not base:
                continue
            lowered = base.lower()
            normalized = lowered.replace(" ", "_").replace("-", "_").replace("/", "_")
            candidates.extend([base, lowered, normalized])
        return tuple(dict.fromkeys(candidates))

    @classmethod
    def _row_numeric_value(
        cls,
        row: dict[str, Any],
        candidates: tuple[str, ...],
    ) -> float | None:
        values: dict[str, Any] = {}
        for key, value in row.items():
            key_text = str(key or "").strip()
            if not key_text:
                continue
            values[key_text] = value
            values[key_text.lower()] = value
            values[key_text.lower().replace(" ", "_").replace("-", "_").replace("/", "_")] = value
        for candidate in candidates:
            raw_value = values.get(candidate)
            if raw_value is None:
                continue
            text = str(raw_value).strip()
            if not text:
                continue
            try:
                return float(text)
            except (TypeError, ValueError):
                continue
        return None

    @classmethod
    def _parse_training_log_csv(cls, csv_path: Path) -> list[dict[str, Any]]:
        if not csv_path.is_file():
            return []

        excluded_metric_tokens = (
            "epoch",
            "step",
            "iter",
            "iteration",
            "batch",
            "loss",
            "lr",
            "learning_rate",
            "time",
            "second",
            "throughput",
            "speed",
            "memory",
            "seed",
            "sample",
            "grad",
        )
        training_log: list[dict[str, Any]] = []
        try:
            with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    return []
                for index, row in enumerate(reader, start=1):
                    if not isinstance(row, dict):
                        continue
                    epoch_value = cls._row_numeric_value(
                        row,
                        cls._csv_column_candidates("epoch", "step", "global_step", "iteration", "iter"),
                    )
                    train_loss = cls._row_numeric_value(
                        row,
                        cls._csv_column_candidates(
                            "train_loss",
                            "loss",
                            "training_loss",
                            "train/loss",
                            "train-loss",
                        ),
                    )
                    val_loss = cls._row_numeric_value(
                        row,
                        cls._csv_column_candidates(
                            "val_loss",
                            "validation_loss",
                            "valid_loss",
                            "dev_loss",
                            "eval_loss",
                            "val/loss",
                            "validation/loss",
                        ),
                    )
                    metrics: dict[str, float] = {}
                    for key, raw_value in row.items():
                        key_text = str(key or "").strip()
                        if not key_text:
                            continue
                        value_text = str(raw_value or "").strip()
                        if not value_text:
                            continue
                        normalized = key_text.lower().replace(" ", "_").replace("-", "_").replace("/", "_")
                        if any(token in normalized for token in excluded_metric_tokens):
                            continue
                        try:
                            metrics[key_text] = float(value_text)
                        except (TypeError, ValueError):
                            continue

                    entry: dict[str, Any] = {
                        "epoch": int(epoch_value) if epoch_value is not None else index,
                        "metrics": metrics,
                    }
                    if train_loss is not None:
                        entry["train_loss"] = train_loss
                    if val_loss is not None:
                        entry["val_loss"] = val_loss
                    if metrics or train_loss is not None or val_loss is not None:
                        training_log.append(entry)
        except (OSError, csv.Error):
            return []
        return training_log

    @staticmethod
    def _command_with_mode(base_command: list[str], mode_flag: str) -> list[str]:
        """Append a pipeline mode flag if it is not already present."""
        if mode_flag in base_command:
            return list(base_command)
        return [*base_command, mode_flag]

    @staticmethod
    def _build_execution_blueprint_summary(
        topic: str,
        blueprint: dict[str, Any],
        setup_output: dict[str, Any],
        coding_output: dict[str, Any],
    ) -> str:
        """Compact execution context used for iterative repair."""
        payload = {
            "topic": topic,
            "title": blueprint.get("title", ""),
            "proposed_method": blueprint.get("proposed_method", {}),
            "datasets": blueprint.get("datasets", []),
            "metrics": blueprint.get("metrics", []),
            "baselines": blueprint.get("baselines", []),
            "ablation_groups": blueprint.get("ablation_groups", []),
            "downloaded_resources": setup_output.get("downloaded_resources", []),
            "data_dir": setup_output.get("data_dir", ""),
            "models_dir": setup_output.get("models_dir", ""),
            "train_command": coding_output.get("train_command", ""),
        }
        return json.dumps(payload, indent=2, ensure_ascii=False)

    @staticmethod
    def _update_best_round(
        iteration_state: IterationState,
        analysis: Any,
    ) -> None:
        """Track the current best round using the primary metric heuristic."""
        if not analysis or not getattr(analysis, "metric_summary", None):
            return
        primary_key = next(iter(analysis.metric_summary), None)
        primary_value = analysis.metric_summary.get(primary_key) if primary_key else None
        best_value = (
            iteration_state.best_metrics.get(primary_key)
            if iteration_state.best_metrics and primary_key
            else None
        )
        lower_is_better = bool(
            primary_key and any(
                kw in primary_key.lower()
                for kw in ("loss", "error", "perplexity", "mse", "mae", "cer", "wer")
            )
        )
        if best_value is None or primary_value is None:
            is_improvement = best_value is None and primary_value is not None
        elif lower_is_better:
            is_improvement = primary_value < best_value
        else:
            is_improvement = primary_value > best_value
        if is_improvement:
            iteration_state.best_round = iteration_state.rounds[-1].round_number
            iteration_state.best_metrics = analysis.metric_summary

    def _load_local_round_artifacts(self, round_number: int | None) -> dict[str, Any]:
        """Best-effort reload of local round artifacts from disk."""
        if round_number is None:
            return {}
        execution_path = self.workspace.path / "logs" / f"execution_round_{round_number}_execution.json"
        quick_eval_path = self.workspace.path / "logs" / f"execution_round_{round_number}_quick_eval.json"
        data: dict[str, Any] = {}
        if execution_path.exists():
            data["execution"] = json.loads(execution_path.read_text(encoding="utf-8"))
        if quick_eval_path.exists():
            data["quick_eval"] = json.loads(quick_eval_path.read_text(encoding="utf-8"))
        return data

    @staticmethod
    def _summarize_local_iteration(
        iteration_state: IterationState,
        blueprint: dict[str, Any],
    ) -> str:
        """Create a concise experiment summary for downstream writing/analysis."""
        method_name = blueprint.get("proposed_method", {}).get("name", "the proposed method")
        lines = [
            f"Executed local iterative experiment loop for {method_name}.",
            f"Completed rounds: {len(iteration_state.rounds)} / {iteration_state.max_rounds}.",
        ]
        if iteration_state.best_round is not None:
            lines.append(f"Best round: {iteration_state.best_round}.")
        if iteration_state.best_metrics:
            metrics_text = ", ".join(
                f"{key}={value}" for key, value in iteration_state.best_metrics.items()
            )
            lines.append(f"Best metrics: {metrics_text}.")
        if iteration_state.rounds and iteration_state.rounds[-1].analysis:
            analysis = iteration_state.rounds[-1].analysis
            lines.append(f"Latest attribution: {analysis.attribution or 'unknown'}.")
            if analysis.recommended_action:
                lines.append(f"Latest recommended action: {analysis.recommended_action}.")
        lines.append(f"Termination: {iteration_state.final_status}.")
        return "\n".join(lines)

    async def _find_existing_job(self, code_dir: Path) -> tuple[str, str] | None:
        """Check if a previous SLURM job exists (from a crashed run).

        Returns (job_id, status) if found, None otherwise.
        """
        tracker = code_dir / "logs" / "active_job_id.txt"
        if not tracker.exists():
            return None

        job_id = tracker.read_text().strip()
        if not job_id or not job_id.isdigit():
            return None

        status = await self._get_job_status(job_id)
        if status in ("RUNNING", "PENDING", "COMPLETED"):
            return (job_id, status)

        return None  # FAILED/CANCELLED/UNKNOWN — need fresh submit

    async def _submit_job(self, slurm_script: str) -> str:
        """Submit a SLURM batch job and return the job ID."""
        if not Path(slurm_script).exists():
            raise RuntimeError(f"SLURM script not found: {slurm_script}")

        result = await self._run_shell(f"sbatch {slurm_script}")
        stdout = result.get("stdout", "")
        stderr = result.get("stderr", "")

        # Parse job ID from "Submitted batch job 12345"
        match = re.search(r"Submitted batch job (\d+)", stdout)
        if not match:
            raise RuntimeError(
                f"Failed to submit SLURM job. stdout: {stdout}, stderr: {stderr}"
            )

        job_id = match.group(1)

        # Save job ID for resume tracking
        tracker_path = Path(slurm_script).parent / "logs" / "active_job_id.txt"
        tracker_path.parent.mkdir(parents=True, exist_ok=True)
        tracker_path.write_text(job_id)

        return job_id

    async def _monitor_job(self, job_id: str, code_dir: Path) -> str:
        """Poll SLURM until job completes. Returns final status."""
        start_time = time.time()
        last_log_lines = 0

        while time.time() - start_time < MAX_WAIT_TIME:
            status = await self._get_job_status(job_id)

            # Stream training log if available
            log_files = list(code_dir.glob("logs/slurm_*.out"))
            if log_files:
                try:
                    content = log_files[-1].read_text(errors="replace")
                    lines = content.strip().split("\n")
                    if len(lines) > last_log_lines:
                        new_lines = lines[last_log_lines:]
                        for line in new_lines[-5:]:  # show last 5 new lines
                            self.log(f"[TRAIN] {line.strip()}")
                        last_log_lines = len(lines)
                except Exception:
                    pass

            if status in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "PREEMPTED", "OUT_OF_MEMORY"):
                return status

            if status == "PENDING":
                elapsed = int(time.time() - start_time)
                self.log(f"Job {job_id} pending... ({elapsed}s elapsed)")
            elif status == "RUNNING":
                elapsed = int(time.time() - start_time)
                self.log(f"Job {job_id} running... ({elapsed}s elapsed)")

            await asyncio.sleep(POLL_INTERVAL)

        # Timeout — cancel the job
        self.log(f"Job {job_id} exceeded max wait time ({MAX_WAIT_TIME}s), cancelling")
        await self._run_shell(f"scancel {job_id}")
        return "TIMEOUT"

    async def _get_job_status(self, job_id: str) -> str:
        """Query SLURM for job status."""
        result = await self._run_shell(
            f"squeue -j {job_id} -h -o '%T' 2>/dev/null || "
            f"sacct -j {job_id} -n -o State -X 2>/dev/null"
        )
        stdout = result.get("stdout", "").strip()

        if not stdout:
            # Job not in queue and not in accounting — might have just finished
            result2 = await self._run_shell(
                f"sacct -j {job_id} -n -o State -X"
            )
            stdout = result2.get("stdout", "").strip()

        # Parse status
        status = stdout.split("\n")[0].strip().upper() if stdout else "UNKNOWN"
        # Clean up status (sacct sometimes adds '+')
        status = status.rstrip("+").strip()

        return status

    async def _collect_results(
        self, code_dir: Path, job_id: str, status: str
    ) -> dict:
        """Collect training results from output files."""
        results: dict[str, Any] = {
            **self._collect_result_artifacts(code_dir),
            "stdout_log": "",
            "stderr_log": "",
        }

        def _read_log(patterns: list[str], limit: int) -> str:
            candidates: list[Path] = []
            for pattern in patterns:
                candidates.extend(sorted((code_dir / "logs").glob(pattern)))

            for log_file in candidates:
                if job_id in log_file.name:
                    return log_file.read_text(errors="replace")[-limit:]

            for log_file in candidates:
                return log_file.read_text(errors="replace")[-limit:]
            return ""

        results["stdout_log"] = _read_log(
            ["slurm_*.out", f"{job_id}.log", "*.out", "*.log"],
            10000,
        )
        results["stderr_log"] = _read_log(
            ["slurm_*.err", f"{job_id}.err", "*.err"],
            5000,
        )

        if not results["metrics"]:
            log_text = "\n".join(
                part for part in [results["stdout_log"], results["stderr_log"]] if part
            )
            if log_text:
                parsed_metrics = self._parse_metrics_from_log(log_text)
                if parsed_metrics:
                    results["parsed_metrics"] = parsed_metrics
                    recovered_metrics = self._wrap_log_metrics_for_contract(parsed_metrics)
                    if recovered_metrics:
                        materialized = self._materialize_recovered_metrics_artifact(
                            code_dir,
                            recovered_metrics,
                            source="slurm_logs",
                            scope="cluster_collect",
                        )
                        results["metrics"] = materialized.get("metrics") or recovered_metrics
                        results["recovered_from"] = "slurm_logs"
                        results["training_log"] = list(results["metrics"].get("training_log") or [])
                        if materialized.get("written"):
                            results["metrics_artifact_materialized"] = True
                            results["metrics_artifact_path"] = materialized.get("artifact_path", "")

        return results

    def _collect_result_artifacts(self, code_dir: Path) -> dict[str, Any]:
        results: dict[str, Any] = {
            "metrics": {},
            "training_log": [],
        }

        metrics_path = code_dir / "results" / "metrics.json"
        if metrics_path.exists():
            parsed_metrics = ExperimentAgent._parse_metrics_json(code_dir)
            if parsed_metrics:
                results["metrics"] = parsed_metrics
                results["training_log"] = list(parsed_metrics.get("training_log") or [])
                meta = parsed_metrics.get("_nanoresearch_meta")
                if isinstance(meta, dict) and str(meta.get("recovered_from") or "").strip():
                    results["recovered_from"] = str(meta.get("recovered_from") or "").strip()
            else:
                try:
                    results["metrics"] = {"raw": metrics_path.read_text()[:5000]}
                except OSError:
                    results["metrics"] = {}

        log_csv = code_dir / "results" / "training_log.csv"
        if log_csv.exists():
            try:
                with log_csv.open("r", encoding="utf-8", errors="replace") as handle:
                    results["training_log_csv"] = handle.read(10000)
            except OSError:
                results["training_log_csv"] = ""
            parsed_training_log = self._parse_training_log_csv(log_csv)
            if parsed_training_log and not results["training_log"]:
                results["training_log"] = parsed_training_log
            if parsed_training_log and not (
                isinstance(results.get("metrics"), dict)
                and any(
                    key in results["metrics"] for key in ("main_results", "ablation_results", "training_log")
                )
            ):
                materialized = self._materialize_recovered_metrics_artifact(
                    code_dir,
                    {
                        "main_results": [],
                        "ablation_results": [],
                        "training_log": parsed_training_log,
                    },
                    source="training_log_csv",
                    scope="result_artifacts",
                )
                results["metrics"] = materialized.get("metrics") or results.get("metrics", {})
                if results["metrics"]:
                    results["recovered_from"] = "training_log_csv"
                    results["training_log"] = list(results["metrics"].get("training_log") or parsed_training_log)
                if materialized.get("written"):
                    results["metrics_artifact_materialized"] = True
                    results["metrics_artifact_path"] = materialized.get("artifact_path", "")

        for results_file in (code_dir / "results").glob("*"):
            if results_file.is_file() and results_file.name not in ("metrics.json", "training_log.csv"):
                try:
                    content = results_file.read_text(errors="replace")[:5000]
                    results[f"result_file_{results_file.name}"] = content
                except Exception:
                    pass

        checkpoints = (
            list((code_dir / "checkpoints").glob("*.pt"))
            if (code_dir / "checkpoints").exists()
            else []
        )
        results["checkpoints"] = [str(p) for p in checkpoints]
        return results

    def _collect_local_results(
        self,
        code_dir: Path,
        run_result: dict[str, Any],
    ) -> dict[str, Any]:
        results: dict[str, Any] = {
            **self._collect_result_artifacts(code_dir),
            "stdout_log": str(run_result.get("stdout", ""))[-10000:],
            "stderr_log": str(run_result.get("stderr", ""))[-5000:],
        }
        if not results["metrics"] and results["stdout_log"]:
            results["parsed_metrics"] = self._parse_metrics_from_log(results["stdout_log"])
        return results

    @staticmethod
    def _metrics_satisfy_contract(metrics: dict[str, Any] | None) -> bool:
        if not isinstance(metrics, dict):
            return False
        main_results = metrics.get("main_results")
        if not isinstance(main_results, list):
            return False
        for item in main_results:
            if not isinstance(item, dict):
                continue
            metric_entries = item.get("metrics")
            if not isinstance(metric_entries, list):
                continue
            for metric in metric_entries:
                if not isinstance(metric, dict):
                    continue
                if str(metric.get("metric_name", "")).strip() and metric.get("value") is not None:
                    return True
        return False

    @staticmethod
    def _result_file_names(results: dict[str, Any]) -> list[str]:
        return sorted(
            key.removeprefix("result_file_")
            for key, value in results.items()
            if key.startswith("result_file_") and str(value or "").strip()
        )

    @classmethod
    def _detect_contract_failure_signals(
        cls,
        stdout_log: str,
        stderr_log: str,
    ) -> list[str]:
        combined = f"{stdout_log}\n{stderr_log}".lower()
        found: list[str] = []
        for indicator in RESULT_CONTRACT_CRASH_INDICATORS:
            if indicator.lower() in combined and indicator not in found:
                found.append(indicator)
        return found

    @classmethod
    def _evaluate_experiment_contract(
        cls,
        result_payload: dict[str, Any],
        *,
        execution_backend: str,
        execution_status: str,
        quick_eval_status: str,
        final_status: str,
    ) -> dict[str, Any]:
        metrics = result_payload.get("metrics") if isinstance(result_payload.get("metrics"), dict) else {}
        parsed_metrics = result_payload.get("parsed_metrics") if isinstance(result_payload.get("parsed_metrics"), dict) else {}
        training_log = result_payload.get("training_log") if isinstance(result_payload.get("training_log"), list) else []
        training_log_csv = str(result_payload.get("training_log_csv") or "").strip()
        checkpoints = result_payload.get("checkpoints") if isinstance(result_payload.get("checkpoints"), list) else []
        result_files = cls._result_file_names(result_payload)
        recovered_from = str(result_payload.get("recovered_from") or "").strip()
        stdout_log = str(result_payload.get("stdout_log") or "")
        stderr_log = str(result_payload.get("stderr_log") or "")

        has_structured_metrics = cls._metrics_satisfy_contract(metrics)
        has_parsed_metrics = bool(parsed_metrics)
        has_training_trace = bool(training_log) or bool(training_log_csv)
        has_checkpoints = bool(checkpoints)
        has_result_files = bool(result_files)
        failure_signals = cls._detect_contract_failure_signals(stdout_log, stderr_log)
        run_completed = final_status == "COMPLETED" or quick_eval_status in {"success", "partial"}
        has_recovered_artifact_support = (
            has_structured_metrics
            and bool(recovered_from)
            and run_completed
            and not failure_signals
            and (has_checkpoints or has_result_files)
            and (has_parsed_metrics or has_training_trace)
        )

        satisfied_signals: list[str] = []
        if has_structured_metrics and not recovered_from:
            satisfied_signals.append("structured_metrics_artifact")
        elif has_structured_metrics and recovered_from:
            satisfied_signals.append("structured_metrics_recovered")
        if has_parsed_metrics:
            satisfied_signals.append("parsed_metrics")
        if has_training_trace:
            satisfied_signals.append("training_log")
        if has_checkpoints:
            satisfied_signals.append("checkpoints")
        if has_result_files:
            satisfied_signals.append("result_files")

        success_path = ""
        status = "failed"
        if has_structured_metrics and not recovered_from and run_completed:
            status = "success"
            success_path = "structured_metrics_artifact"
        elif has_recovered_artifact_support:
            status = "success"
            success_path = "structured_metrics_recovered"
        elif has_structured_metrics and recovered_from and run_completed:
            status = "partial"
            success_path = "structured_metrics_recovered"
        elif has_parsed_metrics and (has_training_trace or has_checkpoints or has_result_files) and run_completed:
            status = "partial"
            success_path = "parsed_metrics_with_artifacts"
        elif has_training_trace and has_checkpoints and final_status == "COMPLETED":
            status = "partial"
            success_path = "training_log_with_checkpoints"
        elif has_result_files and (has_training_trace or has_checkpoints) and final_status == "COMPLETED":
            status = "partial"
            success_path = "aux_results_with_artifacts"
        elif execution_backend == "cluster" and has_parsed_metrics and final_status == "COMPLETED":
            status = "partial"
            success_path = "parsed_metrics_only"

        missing_signals: list[str] = []
        if status == "failed":
            if not (has_structured_metrics or has_parsed_metrics):
                missing_signals.append("metrics_signal")
            if not (has_training_trace or has_checkpoints or has_result_files or has_structured_metrics):
                missing_signals.append("artifact_signal")
            if failure_signals:
                missing_signals.append("crash_free_logs")

        return {
            "version": "v1",
            "status": status,
            "execution_backend": execution_backend,
            "execution_status": execution_status,
            "quick_eval_status": quick_eval_status,
            "final_status": final_status,
            "recovered_from": recovered_from,
            "success_path": success_path,
            "satisfied_signals": satisfied_signals,
            "missing_signals": missing_signals,
            "failure_signals": failure_signals,
            "artifact_inventory": {
                "structured_metrics": has_structured_metrics,
                "parsed_metrics": has_parsed_metrics,
                "training_log_entries": len(training_log),
                "training_log_csv": bool(training_log_csv),
                "checkpoint_count": len(checkpoints),
                "result_files": result_files,
            },
        }

    def _parse_metrics_from_log(self, log_text: str) -> dict:
        """Try to extract metrics from training log output."""
        metrics: dict[str, Any] = {}
        lines = log_text.split("\n")

        # Common patterns in training logs
        patterns = [
            # "Epoch 10: loss=0.123, accuracy=0.95"
            r"[Ee]poch\s+(\d+).*?loss[=:\s]+([0-9.e-]+)",
            # "Test accuracy: 0.95"
            r"[Tt]est\s+(accuracy|acc)[=:\s]+([0-9.e-]+)",
            # "Best metric: 0.95"
            r"[Bb]est\s+(\w+)[=:\s]+([0-9.e-]+)",
            # "AUC: 0.95" / "F1: 0.85"
            r"(AUC|F1|RMSE|MAE|accuracy|precision|recall)[=:\s]+([0-9.e-]+)",
        ]

        epochs = []
        for line in lines:
            for pattern in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    groups = match.groups()
                    if len(groups) == 2:
                        metrics[groups[0]] = groups[1]

            # Track epoch losses
            epoch_match = re.search(
                r"[Ee]poch\s+(\d+).*?loss[=:\s]+([0-9.e-]+)", line
            )
            if epoch_match:
                epochs.append({
                    "epoch": int(epoch_match.group(1)),
                    "loss": float(epoch_match.group(2)),
                })

        if epochs:
            metrics["epoch_losses"] = epochs
            metrics["final_loss"] = epochs[-1]["loss"]

        return metrics

    async def _local_preflight(self, code_dir: Path, python: str = "python") -> tuple[bool, str]:
        """Run local checks before submitting to SLURM.

        Tests:
        1. Python syntax check (py_compile) on all .py files
        2. Import check — try importing the entry point module
        3. Verify all cross-file imports resolve

        Returns (ok, error_message).
        """
        errors = []

        # 1. Syntax check all .py files
        for py_file in sorted(code_dir.glob("*.py")):
            result = await self._run_subprocess(
                [python, "-c", f"import py_compile; py_compile.compile(r'{py_file}', doraise=True)"],
                timeout=10,
            )
            if result["returncode"] != 0:
                errors.append(f"Syntax error in {py_file.name}:\n{result['stderr']}")

        if errors:
            return False, "\n".join(errors)

        # 2. Try importing the main modules to catch import errors
        # (run in the code directory so local imports work)
        py_modules = [f.stem for f in code_dir.glob("*.py")]
        for module in py_modules:
            result = await self._run_subprocess(
                [python, "-c", f"import {module}"],
                cwd=code_dir,
                timeout=30,
            )
            if result["returncode"] != 0:
                err_text = result["stdout"] + result["stderr"]
                # Ignore errors from missing heavy dependencies (torch, etc.)
                # — those will be installed on the cluster node
                if any(pkg in err_text for pkg in [
                    "No module named 'torch'",
                    "No module named 'torchvision'",
                    "No module named 'torchaudio'",
                    "No module named 'timm'",
                    "No module named 'transformers'",
                    "No module named 'torch_geometric'",
                    "No module named 'torch_scatter'",
                    "No module named 'torch_sparse'",
                    "No module named 'esm'",
                    "No module named 'dgl'",
                    "No module named 'accelerate'",
                    "No module named 'datasets'",
                    "No module named 'einops'",
                    "No module named 'wandb'",
                    "No module named 'scipy'",
                    "No module named 'sklearn'",
                    "No module named 'cv2'",
                    "No module named 'PIL'",
                    "CUDA",
                ]):
                    continue
                errors.append(f"Import error in {module}.py:\n{err_text}")

        if errors:
            return False, "\n".join(errors)

        return True, ""

    def _build_local_command(
        self,
        code_dir: Path,
        coding_output: dict[str, Any],
        runtime_python: str,
    ) -> list[str]:
        runner_script = str(coding_output.get("runner_script", "")).strip()
        if runner_script and Path(runner_script).exists():
            runner_path = Path(runner_script)
            runner_token = runner_path.name if runner_path.parent == code_dir else str(runner_path)
            return [runtime_python, runner_token]
        if (code_dir / RUNNER_SCRIPT_NAME).exists():
            return [runtime_python, RUNNER_SCRIPT_NAME]

        command = str(
            coding_output.get("entry_train_command")
            or coding_output.get("train_command")
            or coding_output.get("code_plan", {}).get("train_command", "")
            or ""
        ).strip()
        if command:
            tokens, _env_vars = normalize_target_spec(command, code_dir)
            if tokens:
                if is_python_launcher_token(tokens[0]):
                    return [runtime_python, *tokens[1:]]
                if tokens[0] in {"-m", "-c"} or tokens[0].endswith(".py"):
                    return [runtime_python, *tokens]
                return tokens

        for candidate in ("main.py", "train.py", "run.py"):
            if (code_dir / candidate).exists():
                return [runtime_python, candidate]
        return [runtime_python, "main.py"]

    async def _run_local_training(
        self,
        code_dir: Path,
        command: list[str],
    ) -> dict[str, Any]:
        timeout = max(60, int(self.config.local_execution_timeout))
        result = await self._run_subprocess(command, cwd=code_dir, timeout=timeout)
        result["command"] = command
        result["timed_out"] = (
            result.get("returncode") == -1
            and "timed out" in result.get("stderr", "").lower()
        )
        return result

    async def _run_shell(self, cmd: str, timeout: int = 60) -> dict:
        """Run a shell command asynchronously with proxy environment."""
        env = self._build_proxy_env()
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return {"returncode": -1, "stdout": "", "stderr": "Command timed out"}
        return {
            "returncode": proc.returncode or 0,
            "stdout": stdout.decode(errors="replace"),
            "stderr": stderr.decode(errors="replace"),
        }

    def _build_proxy_env(self) -> dict[str, str]:
        env = {**os.environ}
        proxy_url = env.get("https_proxy") or env.get("HTTPS_PROXY", "")
        if not proxy_url:
            import re as _re

            bashrc = Path.home() / ".bashrc"
            if bashrc.exists():
                content = bashrc.read_text(errors="replace")
                match = _re.search(r"https_proxy=(http://[^\s;'\"]+)", content)
                if match:
                    proxy_url = match.group(1)
        if proxy_url:
            env.update(
                {
                    "http_proxy": proxy_url,
                    "https_proxy": proxy_url,
                    "HTTP_PROXY": proxy_url,
                    "HTTPS_PROXY": proxy_url,
                }
            )
        env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
        return env

    async def _run_subprocess(
        self,
        command: list[str],
        *,
        cwd: Path | None = None,
        timeout: int = 60,
    ) -> dict[str, Any]:
        env = self._build_proxy_env()
        try:
            proc = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(cwd) if cwd is not None else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
        except PermissionError:
            return await asyncio.to_thread(
                self._run_subprocess_sync,
                command,
                cwd=cwd,
                timeout=timeout,
                env=env,
            )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return {"returncode": -1, "stdout": "", "stderr": "Command timed out"}
        return {
            "returncode": proc.returncode or 0,
            "stdout": stdout.decode(errors="replace"),
            "stderr": stderr.decode(errors="replace"),
        }

    @staticmethod
    def _run_subprocess_sync(
        command: list[str],
        *,
        cwd: Path | None = None,
        timeout: int = 60,
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        try:
            completed = subprocess.run(
                command,
                cwd=str(cwd) if cwd is not None else None,
                capture_output=True,
                timeout=timeout,
                env=env,
                text=False,
            )
        except subprocess.TimeoutExpired:
            return {"returncode": -1, "stdout": "", "stderr": "Command timed out"}
        return {
            "returncode": completed.returncode or 0,
            "stdout": completed.stdout.decode(errors="replace"),
            "stderr": completed.stderr.decode(errors="replace"),
        }

    async def close(self) -> None:
        pass
