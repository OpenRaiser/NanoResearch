"""Execution agent — submits SLURM jobs, monitors progress, debugs failures, collects results."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from nanoresearch.agents.base import BaseResearchAgent
from nanoresearch.agents.debug import DebugAgent, MAX_DEBUG_ROUNDS
from nanoresearch.agents.repair_journal import REPAIR_SNAPSHOT_JOURNAL_PATH
from nanoresearch.schemas.manifest import PipelineStage

from .cluster_runner import _ClusterRunnerMixin, _kill_process_tree
from .local_runner import _LocalRunnerMixin
from .repair import _RepairMixin, REMEDIATION_LEDGER_PATH
from .result_collector import _ResultCollectorMixin

__all__ = ["ExecutionAgent"]


class ExecutionAgent(
    _LocalRunnerMixin,
    _ClusterRunnerMixin,
    _RepairMixin,
    _ResultCollectorMixin,
    BaseResearchAgent,
):
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

        # Auto-detect: if profile is local_quick but no local GPU and SLURM is
        # available, automatically upgrade to cluster execution.
        use_cluster = self.config.prefers_cluster_execution()
        if not use_cluster and cluster_available:
            try:
                import torch as _torch
                has_gpu = _torch.cuda.is_available() and _torch.cuda.device_count() > 0
            except Exception:
                has_gpu = False
            if not has_gpu:
                use_cluster = True
                self.log(
                    "No local GPU detected but sbatch is available — "
                    "auto-upgrading to cluster (SLURM) execution"
                )

        if not use_cluster or not cluster_available:
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
