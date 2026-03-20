"""Local execution: quick-eval timeout/partial-metrics recovery helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from nanoresearch.agents.repair_journal import REPAIR_SNAPSHOT_JOURNAL_PATH

logger = logging.getLogger(__name__)


class _LocalRunnerQERecoveryMixin:
    """Helpers extracted from _run_local_quick_eval_loop for timeout/partial recovery."""

    def _handle_quick_eval_timeout_recovery(
        self,
        code_dir: Path,
        result: dict[str, Any],
        cycle: int,
        metrics_path: Path,
        training_log_path: Path,
        mtime_before: float | None,
        training_log_mtime_before: float | None,
        remediation_ledger: list[dict[str, Any]] | None,
        round_number: int | None,
    ) -> dict[str, Any] | None:
        """Handle returncode == -1 (timeout/killed) by checking for partial artifacts.

        Returns a quick-eval result dict if recovery succeeded, or None to
        signal the caller to continue with fix cycles.
        """
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

        return None
