"""Runtime remediation strategies."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any

from nanoresearch.agents.preflight import PreflightChecker
from nanoresearch.agents.runtime_env import RuntimeEnvironmentManager

from .repair import MODULE_PACKAGE_ALIASES

from nanoresearch.agents.project_runner import RUNNER_CONFIG_NAME
from nanoresearch.agents.repair_journal import capture_repair_snapshot, rollback_snapshot
from nanoresearch.agents.runtime_env import ExperimentExecutionPolicy
from .repair import QUICK_EVAL_AUTO_OPTIONS

logger = logging.getLogger(__name__)


class _RepairRuntimeMixin:
    """Mixin — runtime remediation (missing modules, NLTK resources)."""

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

        for module_name in missing_modules:
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
