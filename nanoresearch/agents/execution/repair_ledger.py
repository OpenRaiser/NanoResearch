"""Repair ledger: error tracking, remediation journal, and context building."""

from __future__ import annotations

import gzip
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from nanoresearch.agents.repair_journal import (
    REPAIR_SNAPSHOT_JOURNAL_PATH,
    append_snapshot_journal,
    capture_repair_snapshot,
    rollback_snapshot,
)
from nanoresearch.agents.preflight import PreflightChecker
from nanoresearch.agents.runtime_env import ExperimentExecutionPolicy, RuntimeEnvironmentManager

logger = logging.getLogger(__name__)

from .repair import REMEDIATION_LEDGER_PATH, RESOURCE_SUCCESS_STATUSES


class _RepairLedgerMixin:
    """Mixin — error tracking, remediation ledger, and repair context."""

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

