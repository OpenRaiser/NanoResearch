"""Local execution: quick-eval loop with timeout handling and batch-fix cycles."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from nanoresearch.agents.experiment import ExperimentAgent
from nanoresearch.agents.repair_journal import REPAIR_SNAPSHOT_JOURNAL_PATH
from nanoresearch.agents.runtime_env import ExperimentExecutionPolicy

from .local_runner_qe_recovery import _LocalRunnerQERecoveryMixin

logger = logging.getLogger(__name__)


class _LocalRunnerQuickEvalMixin(_LocalRunnerQERecoveryMixin):

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
                recovery = self._handle_quick_eval_timeout_recovery(
                    code_dir,
                    result,
                    cycle,
                    metrics_path,
                    training_log_path,
                    mtime_before,
                    training_log_mtime_before,
                    remediation_ledger,
                    round_number,
                )
                if recovery is not None:
                    return recovery

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
