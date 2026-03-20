"""Local execution: dry-run loop with iterative batch-fix cycles."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from nanoresearch.agents.experiment import ExperimentAgent
from nanoresearch.agents.repair_journal import REPAIR_SNAPSHOT_JOURNAL_PATH
from nanoresearch.agents.runtime_env import ExperimentExecutionPolicy

logger = logging.getLogger(__name__)


class _LocalRunnerDryRunMixin:

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
            # Use local_execution_timeout (default 1800s) instead of the
            # previous hardcoded 120s.  Dataset downloads + model init can
            # easily exceed 2 minutes on first run.
            dry_run_timeout = max(120, int(self.config.local_execution_timeout))
            result = await self._run_subprocess(
                self._command_with_mode(base_command, "--dry-run"),
                cwd=code_dir,
                timeout=dry_run_timeout,
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
