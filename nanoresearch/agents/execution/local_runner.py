"""Local execution: dry-run, quick-eval, and full training loops."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from nanoresearch.agents.experiment import ExperimentAgent
from nanoresearch.agents.feedback_analyzer import FeedbackAnalyzer
from nanoresearch.agents.preflight import PreflightChecker
from nanoresearch.agents.project_runner import (
    RUNNER_SCRIPT_NAME,
    ensure_project_runner,
    is_python_launcher_token,
    normalize_target_spec,
    repair_launch_contract,
    refresh_project_runner_script,
    validate_launch_contract,
)
from nanoresearch.agents.repair_journal import (
    REPAIR_SNAPSHOT_JOURNAL_PATH,
)
from nanoresearch.agents.runtime_env import ExperimentExecutionPolicy, RuntimeEnvironmentManager
from nanoresearch.schemas.iteration import ExperimentHypothesis, IterationState, RoundResult

logger = logging.getLogger(__name__)

LOCAL_EXECUTION_CHECKPOINT = "plans/execution_iteration_checkpoint.json"


class _LocalRunnerMixin:

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
