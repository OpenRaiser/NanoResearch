"""Local execution: dry-run, quick-eval, and full training loops."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from nanoresearch.agents.experiment import ExperimentAgent
from nanoresearch.agents.feedback_analyzer import FeedbackAnalyzer
from nanoresearch.agents.preflight import PreflightChecker
from nanoresearch.agents.project_runner import (
    RUNNER_SCRIPT_NAME,
    ensure_project_runner,
    refresh_project_runner_script,
    repair_launch_contract,
    validate_launch_contract,
)
from nanoresearch.agents.runtime_env import RuntimeEnvironmentManager
from nanoresearch.schemas.iteration import ExperimentHypothesis, IterationState, RoundResult

from .local_runner_dry_run import _LocalRunnerDryRunMixin
from .local_runner_helpers import _LocalRunnerHelpersMixin
from .local_runner_quick_eval import _LocalRunnerQuickEvalMixin

logger = logging.getLogger(__name__)

LOCAL_EXECUTION_CHECKPOINT = "plans/execution_iteration_checkpoint.json"


class _LocalRunnerMixin(
    _LocalRunnerDryRunMixin,
    _LocalRunnerQuickEvalMixin,
    _LocalRunnerHelpersMixin,
):

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

        # Build session label for deterministic conda env naming on auto-repair
        session_label = ""
        if hasattr(self, "workspace") and self.workspace:
            m = self.workspace.manifest
            sid = m.session_id[:8]
            slug = m.topic[:20].replace(" ", "_") if m.topic else ""
            session_label = f"{slug}_{sid}" if slug else sid
        runtime_manager = RuntimeEnvironmentManager(
            self.config, self.log, session_label=session_label,
        )
        runtime_env = await runtime_manager.prepare(code_dir)
        self._record_runtime_env_ledger(runtime_env, remediation_ledger)
        runtime_python = str(runtime_env.get("python", "python"))
        execution_policy = runtime_manager.build_execution_policy(code_dir)
        helper = ExperimentAgent(self.workspace, self.config)
        # Wire helper agent cost tracking to parent dispatcher's callback
        if hasattr(self, '_dispatcher') and hasattr(self._dispatcher, '_usage_callback'):
            helper._dispatcher._usage_callback = self._dispatcher._usage_callback
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
