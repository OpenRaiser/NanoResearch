"""Runtime validation and repair mixin."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import re
import subprocess
from pathlib import Path
from typing import Any

from ._constants import (
    MAX_RUNTIME_IMPORT_PROBES,
    MAX_RUNTIME_VALIDATION_REPAIR_PACKAGES,
    PACKAGE_IMPORT_ALIASES,
)
from ._discovery import _canonicalize_dependency_name
from ._types import ExperimentExecutionPolicy

logger = logging.getLogger(__name__)


class _ValidationMixin:
    """Mixin — runtime validation, repair, and execution policy."""

    @staticmethod
    def _package_import_candidates(package_name: str) -> list[str]:
        normalized = _canonicalize_dependency_name(package_name)
        if not normalized:
            return []

        candidates: list[str] = []
        alias = PACKAGE_IMPORT_ALIASES.get(normalized)
        if alias:
            candidates.append(alias)

        direct_candidate = normalized.replace("-", "_")
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", direct_candidate):
            candidates.append(direct_candidate)

        seen: set[str] = set()
        deduped: list[str] = []
        for candidate in candidates:
            if candidate not in seen:
                deduped.append(candidate)
                seen.add(candidate)
        return deduped

    @staticmethod
    def _validation_status(details: dict[str, Any] | None, key: str) -> str:
        if not isinstance(details, dict):
            return ""
        probe = details.get(key)
        if not isinstance(probe, dict):
            return ""
        return str(probe.get("status") or "").strip()

    @classmethod
    def _validation_requires_venv_rebuild(cls, validation: dict[str, Any] | None) -> bool:
        if not isinstance(validation, dict):
            return False
        return (
            cls._validation_status(validation, "python_smoke") == "failed"
            or cls._validation_status(validation, "pip_probe") == "failed"
        )

    @staticmethod
    def _failed_import_packages(validation: dict[str, Any] | None) -> list[str]:
        if not isinstance(validation, dict):
            return []
        import_probe = validation.get("import_probe")
        if not isinstance(import_probe, dict):
            return []

        packages: list[str] = []
        for failure in import_probe.get("failures", []) or []:
            if not isinstance(failure, dict):
                continue
            package_name = _canonicalize_dependency_name(str(failure.get("package") or ""))
            if package_name and package_name not in packages:
                packages.append(package_name)
        return packages

    @staticmethod
    def _extract_requirement_dependency_specs(requirements_file: Path) -> list[str]:
        try:
            lines = requirements_file.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []

        specs: list[str] = []
        for raw_line in lines:
            candidate = raw_line.strip()
            if not candidate or candidate.startswith("#"):
                continue
            if "#egg=" not in candidate and "#" in candidate:
                candidate = candidate.split("#", 1)[0].strip()
            if not candidate:
                continue
            if candidate.startswith(("-r ", "--requirement ", "-c ", "--constraint ")):
                continue
            if candidate.startswith(("-f ", "--find-links ", "--index-url ", "--extra-index-url ")):
                continue
            if candidate.startswith("--"):
                continue
            if candidate.startswith(("-e ", "--editable ")) or "://" in candidate:
                continue
            if _canonicalize_dependency_name(candidate):
                specs.append(candidate)
        return specs

    @classmethod
    def _repairable_dependency_spec_index(
        cls,
        code_dir: Path,
        execution_policy: ExperimentExecutionPolicy | None,
    ) -> dict[str, str]:
        if execution_policy is None or execution_policy.install_plan is None:
            return {}

        source = execution_policy.install_plan.source
        if source not in {"requirements.txt", "environment.yml", "environment.yaml"}:
            return {}

        return cls.collect_repairable_dependency_specs(code_dir)

    @classmethod
    def collect_declared_dependency_names(cls, code_dir: Path) -> list[str]:
        return sorted(cls._collect_declared_dependency_names(code_dir))

    @classmethod
    def collect_repairable_dependency_specs(cls, code_dir: Path) -> dict[str, str]:
        install_plan = cls._select_install_plan(code_dir)
        if install_plan is None:
            return {}

        if install_plan.source == "requirements.txt":
            manifest_file = code_dir / "requirements.txt"
            specs = cls._extract_requirement_dependency_specs(manifest_file)
        elif install_plan.source in {"environment.yml", "environment.yaml"}:
            environment_file = cls._find_environment_file(code_dir)
            specs = cls._extract_pip_dependencies(environment_file) if environment_file is not None else []
        else:
            return {}

        index: dict[str, str] = {}
        for spec in specs:
            normalized = _canonicalize_dependency_name(spec)
            if normalized and normalized not in index:
                index[normalized] = spec
        return index

    async def _repair_runtime_validation(
        self,
        *,
        kind: str,
        python: str,
        code_dir: Path,
        execution_policy: ExperimentExecutionPolicy,
        validation: dict[str, Any],
        env_dir: Path | None = None,
        created: bool = False,
    ) -> dict[str, Any]:
        repair_actions: list[dict[str, Any]] = []
        current_python = str(python)
        current_validation = dict(validation)
        current_install_info: dict[str, Any] | None = None
        recreated = False

        if (
            kind == "venv"
            and env_dir is not None
            and not created
            and self._validation_requires_venv_rebuild(current_validation)
        ):
            recreate_result = await self._recreate_venv(env_dir)
            repair_actions.append(
                {
                    "kind": "recreate_venv",
                    **recreate_result,
                }
            )
            if recreate_result.get("status") == "applied":
                current_python = str(recreate_result.get("python") or current_python)
                recreated = True
                current_install_info = await self.install_requirements(current_python, code_dir)
                repair_actions.append(
                    {
                        "kind": "reinstall_manifest",
                        **current_install_info,
                    }
                )
                current_validation = await self.validate_runtime(
                    current_python,
                    code_dir,
                    execution_policy=execution_policy,
                )

        failed_imports = self._failed_import_packages(current_validation)
        if failed_imports:
            spec_index = self._repairable_dependency_spec_index(code_dir, execution_policy)
            repair_specs: list[str] = []
            unresolved: list[str] = []
            for package_name in failed_imports:
                spec = spec_index.get(package_name)
                if spec and spec not in repair_specs:
                    repair_specs.append(spec)
                else:
                    unresolved.append(package_name)
                if len(repair_specs) >= MAX_RUNTIME_VALIDATION_REPAIR_PACKAGES:
                    break

            if repair_specs:
                targeted_install = await self.install_dependency_specs(
                    current_python,
                    code_dir,
                    repair_specs,
                    source="runtime_validation_import_repair",
                )
                repair_actions.append(
                    {
                        "kind": "import_repair_install",
                        **targeted_install,
                    }
                )
                if targeted_install.get("status") == "installed":
                    current_validation = await self.validate_runtime(
                        current_python,
                        code_dir,
                        execution_policy=execution_policy,
                    )
            elif unresolved:
                repair_actions.append(
                    {
                        "kind": "import_repair_skipped",
                        "status": "skipped",
                        "packages": unresolved,
                    }
                )

        if not repair_actions:
            repair_summary = {
                "status": "skipped",
                "actions": [],
            }
        else:
            final_status = str(current_validation.get("status") or "").strip()
            if final_status == "ready":
                summary_status = "applied"
            elif any(action.get("status") == "failed" for action in repair_actions):
                summary_status = "failed"
            else:
                summary_status = "partial"
            repair_summary = {
                "status": summary_status,
                "actions": repair_actions,
            }

        result: dict[str, Any] = {
            "python": current_python,
            "validation": current_validation,
            "repair": repair_summary,
            "recreated": recreated,
        }
        if current_install_info is not None:
            result["dependency_install"] = current_install_info
        return result

    def _select_import_probe_targets(
        self,
        execution_policy: ExperimentExecutionPolicy | None,
    ) -> tuple[list[dict[str, str]], str]:
        if execution_policy is None:
            return [], "no_execution_policy"

        install_plan = execution_policy.install_plan
        if install_plan is None:
            return [], "no_install_plan"
        if install_plan.source not in {"requirements.txt", "environment.yml", "environment.yaml"}:
            return [], "install_source_not_probe_safe"

        targets: list[dict[str, str]] = []
        for package_name in sorted(execution_policy.declared_dependencies):
            import_candidates = self._package_import_candidates(package_name)
            if not import_candidates:
                continue
            targets.append({"package": package_name, "module": import_candidates[0]})
            if len(targets) >= MAX_RUNTIME_IMPORT_PROBES:
                break

        if not targets:
            return [], "no_probeable_dependencies"
        return targets, ""

    async def validate_runtime(
        self,
        python: str,
        code_dir: Path,
        *,
        execution_policy: ExperimentExecutionPolicy | None = None,
    ) -> dict[str, Any]:
        """Validate that the selected runtime can execute and import key dependencies."""
        loop = asyncio.get_running_loop()

        try:
            smoke_result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    [
                        python,
                        "-c",
                        (
                            "import json, sys; "
                            "print(json.dumps({'executable': sys.executable, 'version': sys.version.split()[0]}))"
                        ),
                    ],
                    cwd=str(code_dir),
                    capture_output=True,
                    text=True,
                    timeout=30,
                ),
            )
        except Exception as exc:
            self._log(f"Runtime smoke probe failed to start for {python}: {exc}")
            return {
                "status": "failed",
                "python_smoke": {
                    "status": "failed",
                    "error": str(exc),
                },
                "pip_probe": {"status": "skipped"},
                "import_probe": {"status": "skipped"},
            }

        smoke_stdout = (smoke_result.stdout or "").strip()
        smoke_stderr = (smoke_result.stderr or "").strip()
        smoke_payload: dict[str, Any] = {}
        if smoke_result.returncode == 0 and smoke_stdout:
            try:
                smoke_payload = json.loads(smoke_stdout.splitlines()[-1])
            except json.JSONDecodeError:
                smoke_payload = {}

        python_smoke = {
            "status": "passed" if smoke_result.returncode == 0 else "failed",
            "returncode": smoke_result.returncode,
            "stderr": smoke_stderr[:300],
            "executable": str(smoke_payload.get("executable") or python),
            "version": str(smoke_payload.get("version") or ""),
        }
        if smoke_result.returncode != 0:
            self._log(
                f"Runtime validation failed for {python}: rc={smoke_result.returncode}, stderr={smoke_stderr[:200]}"
            )
            return {
                "status": "failed",
                "python_smoke": python_smoke,
                "pip_probe": {"status": "skipped"},
                "import_probe": {"status": "skipped"},
            }

        try:
            pip_result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    [python, "-m", "pip", "--version"],
                    cwd=str(code_dir),
                    capture_output=True,
                    text=True,
                    timeout=30,
                ),
            )
        except Exception as exc:
            pip_probe = {
                "status": "failed",
                "error": str(exc),
            }
        else:
            pip_probe = {
                "status": "passed" if pip_result.returncode == 0 else "failed",
                "returncode": pip_result.returncode,
                "version": (pip_result.stdout or "").strip()[:200],
                "stderr": (pip_result.stderr or "").strip()[:300],
            }

        probe_targets, skipped_reason = self._select_import_probe_targets(execution_policy)
        if not probe_targets:
            import_probe = {
                "status": "skipped",
                "targets": [],
                "failures": [],
                "skipped_reason": skipped_reason,
            }
        else:
            import_script = "\n".join(
                [
                    "import importlib",
                    "import json",
                    f"targets = {json.dumps(probe_targets, ensure_ascii=False)}",
                    "results = []",
                    "for item in targets:",
                    "    package = item['package']",
                    "    module = item['module']",
                    "    try:",
                    "        importlib.import_module(module)",
                    "        results.append({'package': package, 'module': module, 'status': 'passed'})",
                    "    except Exception as exc:",
                    "        results.append({",
                    "            'package': package,",
                    "            'module': module,",
                    "            'status': 'failed',",
                    "            'error': f'{exc.__class__.__name__}: {exc}',",
                    "        })",
                    "print(json.dumps({'results': results}, ensure_ascii=False))",
                ]
            )
            try:
                import_result = await loop.run_in_executor(
                    None,
                    lambda: subprocess.run(
                        [python, "-c", import_script],
                        cwd=str(code_dir),
                        capture_output=True,
                        text=True,
                        timeout=60,
                    ),
                )
            except Exception as exc:
                import_probe = {
                    "status": "failed",
                    "targets": list(probe_targets),
                    "failures": [{"package": "", "module": "", "error": str(exc)}],
                }
            else:
                import_stdout = (import_result.stdout or "").strip()
                parsed_results: list[dict[str, Any]] = []
                if import_stdout:
                    try:
                        parsed_payload = json.loads(import_stdout.splitlines()[-1])
                    except json.JSONDecodeError:
                        parsed_payload = {}
                    results_value = parsed_payload.get("results") if isinstance(parsed_payload, dict) else None
                    if isinstance(results_value, list):
                        parsed_results = [item for item in results_value if isinstance(item, dict)]
                failures = [item for item in parsed_results if item.get("status") != "passed"]
                if import_result.returncode != 0 and not failures:
                    failures = [
                        {
                            "package": "",
                            "module": "",
                            "error": (import_result.stderr or "").strip()[:300],
                        }
                    ]
                import_probe = {
                    "status": "passed" if not failures and import_result.returncode == 0 else "partial",
                    "targets": list(probe_targets),
                    "results": parsed_results,
                    "failures": failures,
                    "stderr": (import_result.stderr or "").strip()[:300],
                }

        overall_status = "ready"
        if pip_probe.get("status") != "passed":
            overall_status = "partial"
        if import_probe.get("status") in {"partial", "failed"}:
            overall_status = "partial"

        validation = {
            "status": overall_status,
            "python_smoke": python_smoke,
            "pip_probe": pip_probe,
            "import_probe": import_probe,
        }
        self._log(
            "Runtime validation "
            f"{overall_status} for {python_smoke.get('executable', python)} "
            f"(pip={pip_probe.get('status')}, imports={import_probe.get('status')})"
        )
        return validation

    def build_execution_policy(self, code_dir: Path) -> ExperimentExecutionPolicy:
        """Build a centralized execution/remediation policy for a project."""
        allowlist = {
            normalized
            for item in self.config.runtime_auto_install_allowlist
            if (normalized := _canonicalize_dependency_name(item))
        }
        manifest_snapshot = self.inspect_project_manifests(code_dir)
        return ExperimentExecutionPolicy(
            install_plan=self._select_install_plan(code_dir),
            manifest_source=manifest_snapshot.manifest_source,
            manifest_path=manifest_snapshot.manifest_path,
            declared_dependencies=frozenset(self._collect_declared_dependency_names(code_dir)),
            runtime_auto_install_enabled=bool(self.config.runtime_auto_install_enabled),
            runtime_auto_install_allowlist=frozenset(allowlist),
            max_runtime_auto_installs=max(0, int(self.config.runtime_auto_install_max_packages)),
            max_nltk_downloads=max(0, int(self.config.runtime_auto_install_max_nltk_downloads)),
        )
