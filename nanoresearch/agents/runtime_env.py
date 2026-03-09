"""Shared runtime environment helpers for experiment execution."""

from __future__ import annotations

import asyncio
import configparser
import json
import platform
import re
import shutil
import subprocess
import sys
import venv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from nanoresearch.config import ResearchConfig

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    tomllib = None


PACKAGE_IMPORT_ALIASES = {
    "pyyaml": "yaml",
    "opencv-python": "cv2",
    "pillow": "PIL",
    "scikit-learn": "sklearn",
    "biopython": "Bio",
    "python-dateutil": "dateutil",
    "beautifulsoup4": "bs4",
}
MAX_RUNTIME_IMPORT_PROBES = 5
MAX_RUNTIME_VALIDATION_REPAIR_PACKAGES = 3


def _canonicalize_dependency_name(raw_value: str) -> str | None:
    """Normalize a dependency specifier to its package name."""
    candidate = str(raw_value or "").strip()
    if not candidate:
        return None

    if candidate.startswith(("-r ", "--requirement ", "-c ", "--constraint ")):
        return None
    if candidate.startswith(("-f ", "--find-links ", "--index-url ", "--extra-index-url ")):
        return None
    if candidate.startswith("--"):
        return None

    if candidate.startswith(("-e ", "--editable ")):
        _, _, editable_target = candidate.partition(" ")
        candidate = editable_target.strip()

    egg_marker = "#egg="
    if egg_marker in candidate:
        candidate = candidate.split(egg_marker, 1)[1].strip()
    elif "#" in candidate:
        candidate = candidate.split("#", 1)[0].strip()

    if " @" in candidate:
        candidate = candidate.split(" @", 1)[0].strip()
    candidate = candidate.split(";", 1)[0].strip()
    if not candidate or candidate in {".", ".."}:
        return None
    if candidate.startswith((".", "/", "\\")):
        return None

    match = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", candidate)
    if not match:
        return None

    normalized = re.sub(r"[-_.]+", "-", match.group(0)).lower()
    return normalized or None


@dataclass(frozen=True)
class DependencyInstallPlan:
    """Concrete pip-install strategy for a generated experiment project."""

    source: str
    args: list[str]
    manifest_path: str
    fallback_args: list[str] | None = None


@dataclass(frozen=True)
class ProjectManifestSnapshot:
    """Local manifest inspection snapshot shared across execution backends."""

    manifest_source: str
    manifest_path: str
    environment_source: str
    environment_path: str
    install_source: str
    install_manifest_path: str
    install_kind: str

    def to_dict(self) -> dict[str, str]:
        return {
            "manifest_source": self.manifest_source,
            "manifest_path": self.manifest_path,
            "environment_source": self.environment_source,
            "environment_path": self.environment_path,
            "install_source": self.install_source,
            "install_manifest_path": self.install_manifest_path,
            "install_kind": self.install_kind,
        }


@dataclass(frozen=True)
class ExperimentExecutionPolicy:
    """Central execution-time remediation policy for local experiment runs."""

    install_plan: DependencyInstallPlan | None
    manifest_source: str
    manifest_path: str
    declared_dependencies: frozenset[str]
    runtime_auto_install_enabled: bool
    runtime_auto_install_allowlist: frozenset[str]
    max_runtime_auto_installs: int
    max_nltk_downloads: int

    @staticmethod
    def _count_actions(
        fix_history: list[dict[str, Any]] | None,
        *,
        prefix: str,
    ) -> int:
        if not fix_history:
            return 0

        total = 0
        for entry in fix_history:
            for item in entry.get("fixed_files", []):
                if isinstance(item, str) and item.startswith(prefix):
                    total += 1
        return total

    def remaining_runtime_auto_installs(
        self,
        fix_history: list[dict[str, Any]] | None = None,
    ) -> int:
        used = self._count_actions(fix_history, prefix="pip:")
        return max(0, int(self.max_runtime_auto_installs) - used)

    def remaining_nltk_downloads(
        self,
        fix_history: list[dict[str, Any]] | None = None,
    ) -> int:
        used = self._count_actions(fix_history, prefix="nltk:")
        return max(0, int(self.max_nltk_downloads) - used)

    def allows_runtime_package(
        self,
        package_name: str,
        *,
        module_name: str = "",
        aliases: dict[str, str] | None = None,
    ) -> bool:
        if not self.runtime_auto_install_enabled:
            return False

        normalized_package = _canonicalize_dependency_name(package_name)
        if not normalized_package:
            return False
        if normalized_package in self.declared_dependencies:
            return True
        if normalized_package in self.runtime_auto_install_allowlist:
            return True

        normalized_module = _canonicalize_dependency_name(module_name)
        if not normalized_module or not aliases:
            return False
        alias_target = aliases.get(normalized_module)
        if not alias_target:
            return False
        return _canonicalize_dependency_name(alias_target) == normalized_package

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_source": self.manifest_source,
            "manifest_path": self.manifest_path,
            "declared_dependencies": sorted(self.declared_dependencies),
            "runtime_auto_install_enabled": self.runtime_auto_install_enabled,
            "runtime_auto_install_allowlist": sorted(self.runtime_auto_install_allowlist),
            "max_runtime_auto_installs": self.max_runtime_auto_installs,
            "max_nltk_downloads": self.max_nltk_downloads,
        }


class RuntimeEnvironmentManager:
    """Prepare Python runtimes for local experiment execution."""

    def __init__(
        self,
        config: ResearchConfig,
        log_fn: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self._log = log_fn or (lambda _message: None)

    async def prepare(self, code_dir: Path) -> dict[str, Any]:
        requirements_path = code_dir / "requirements.txt"
        environment_file = self._find_environment_file(code_dir)
        execution_policy = self.build_execution_policy(code_dir)
        conda_env = self.config.experiment_conda_env.strip()
        if conda_env:
            conda_python = self.find_conda_python(conda_env)
            if conda_python:
                self._log(f"Using existing conda env '{conda_env}': {conda_python}")
                install_info = await self.install_requirements(conda_python, code_dir)
                runtime_validation = await self.validate_runtime(
                    conda_python,
                    code_dir,
                    execution_policy=execution_policy,
                )
                runtime_validation_repair = {
                    "status": "skipped",
                    "actions": [],
                }
                return {
                    "kind": "conda",
                    "python": conda_python,
                    "env_name": conda_env,
                    "requirements_path": str(requirements_path) if requirements_path.exists() else "",
                    "environment_file": str(environment_file) if environment_file else "",
                    "dependency_install": install_info,
                    "runtime_validation": runtime_validation,
                    "runtime_validation_repair": runtime_validation_repair,
                    "execution_policy": execution_policy.to_dict(),
                }
            if self.config.auto_create_env and shutil.which("conda"):
                created = await self.create_conda_env(conda_env, code_dir)
                if created:
                    conda_python = self.find_conda_python(conda_env)
                    if conda_python:
                        install_info = await self.install_requirements(conda_python, code_dir)
                        runtime_validation = await self.validate_runtime(
                            conda_python,
                            code_dir,
                            execution_policy=execution_policy,
                        )
                        runtime_validation_repair = {
                            "status": "skipped",
                            "actions": [],
                        }
                        return {
                            "kind": "conda",
                            "python": conda_python,
                            "env_name": conda_env,
                            "created": True,
                            "requirements_path": str(requirements_path) if requirements_path.exists() else "",
                            "environment_file": str(environment_file) if environment_file else "",
                            "dependency_install": install_info,
                            "runtime_validation": runtime_validation,
                            "runtime_validation_repair": runtime_validation_repair,
                            "execution_policy": execution_policy.to_dict(),
                        }
            self._log(f"Conda env '{conda_env}' not found, falling back to venv")

        if not self.config.auto_create_env:
            self._log("Automatic env creation disabled, using system Python")
            install_info = await self.install_requirements(sys.executable, code_dir)
            runtime_validation = await self.validate_runtime(
                sys.executable,
                code_dir,
                execution_policy=execution_policy,
            )
            validation_repair = await self._repair_runtime_validation(
                kind="system",
                python=sys.executable,
                code_dir=code_dir,
                execution_policy=execution_policy,
                validation=runtime_validation,
            )
            runtime_validation = validation_repair["validation"]
            return {
                "kind": "system",
                "python": sys.executable,
                "requirements_path": str(requirements_path) if requirements_path.exists() else "",
                "environment_file": str(environment_file) if environment_file else "",
                "dependency_install": install_info,
                "runtime_validation": runtime_validation,
                "runtime_validation_repair": validation_repair["repair"],
                "execution_policy": execution_policy.to_dict(),
            }

        venv_dir = code_dir / ".venv"
        is_windows = platform.system() == "Windows"
        python_path = venv_dir / ("Scripts/python.exe" if is_windows else "bin/python")
        created = False
        recreated = False

        if not python_path.exists():
            self._log(f"Creating isolated venv at {venv_dir} ...")
            loop = asyncio.get_running_loop()
            try:
                await loop.run_in_executor(
                    None,
                    lambda: venv.create(str(venv_dir), with_pip=True),
                )
                created = True
                self._log(f"Venv created (python: {python_path})")
            except (OSError, subprocess.CalledProcessError) as exc:
                self._log(f"Venv creation failed: {exc}, falling back to system Python")
                install_info = await self.install_requirements(sys.executable, code_dir)
                runtime_validation = await self.validate_runtime(
                    sys.executable,
                    code_dir,
                    execution_policy=execution_policy,
                )
                validation_repair = await self._repair_runtime_validation(
                    kind="system",
                    python=sys.executable,
                    code_dir=code_dir,
                    execution_policy=execution_policy,
                    validation=runtime_validation,
                )
                runtime_validation = validation_repair["validation"]
                return {
                    "kind": "system",
                    "python": sys.executable,
                    "requirements_path": str(requirements_path) if requirements_path.exists() else "",
                    "environment_file": str(environment_file) if environment_file else "",
                    "dependency_install": install_info,
                    "runtime_validation": runtime_validation,
                    "runtime_validation_repair": validation_repair["repair"],
                    "execution_policy": execution_policy.to_dict(),
                }
        else:
            self._log(f"Reusing existing venv at {venv_dir}")

        install_info = await self.install_requirements(str(python_path), code_dir)
        runtime_validation = await self.validate_runtime(
            str(python_path),
            code_dir,
            execution_policy=execution_policy,
        )
        validation_repair = await self._repair_runtime_validation(
            kind="venv",
            python=str(python_path),
            code_dir=code_dir,
            execution_policy=execution_policy,
            validation=runtime_validation,
            env_dir=venv_dir,
            created=created,
        )
        runtime_validation = validation_repair["validation"]
        python_path = Path(str(validation_repair["python"]))
        install_info = validation_repair.get("dependency_install", install_info)
        recreated = bool(validation_repair.get("recreated", False))
        return {
            "kind": "venv",
            "python": str(python_path),
            "env_path": str(venv_dir),
            "created": created or recreated,
            "recreated": recreated,
            "requirements_path": str(requirements_path) if requirements_path.exists() else "",
            "environment_file": str(environment_file) if environment_file else "",
            "dependency_install": install_info,
            "runtime_validation": runtime_validation,
            "runtime_validation_repair": validation_repair["repair"],
            "execution_policy": execution_policy.to_dict(),
        }

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

    async def install_dependency_specs(
        self,
        python: str,
        code_dir: Path,
        specs: list[str],
        *,
        source: str = "runtime_validation",
    ) -> dict[str, Any]:
        filtered_specs = [str(spec).strip() for spec in specs if str(spec).strip()]
        if not filtered_specs:
            return {"status": "skipped", "source": source, "specs": []}

        self._log(f"Installing targeted dependency specs via {source}: {filtered_specs}")
        loop = asyncio.get_running_loop()
        try:
            proc_result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    [python, "-m", "pip", "install", *filtered_specs, "--quiet"],
                    cwd=str(code_dir),
                    capture_output=True,
                    text=True,
                    timeout=600,
                ),
            )
        except Exception as exc:
            self._log(f"Targeted dependency install failed via {source}: {exc}")
            return {
                "status": "error",
                "source": source,
                "specs": filtered_specs,
                "error": str(exc),
            }

        if proc_result.returncode == 0:
            self._log(f"Targeted dependency install OK via {source}")
            return {
                "status": "installed",
                "source": source,
                "specs": filtered_specs,
            }

        stderr = (proc_result.stderr or "").strip()
        self._log(f"Targeted dependency install rc={proc_result.returncode} via {source}: {stderr[:500]}")
        return {
            "status": "failed",
            "source": source,
            "specs": filtered_specs,
            "returncode": proc_result.returncode,
            "stderr": stderr[:500],
        }

    async def _recreate_venv(self, env_dir: Path) -> dict[str, Any]:
        self._log(f"Recreating invalid venv at {env_dir}")
        loop = asyncio.get_running_loop()
        shutil.rmtree(env_dir, ignore_errors=True)
        try:
            await loop.run_in_executor(
                None,
                lambda: venv.create(str(env_dir), with_pip=True),
            )
        except Exception as exc:
            self._log(f"Venv recreation failed at {env_dir}: {exc}")
            return {
                "status": "failed",
                "error": str(exc),
            }

        is_windows = platform.system() == "Windows"
        python_path = env_dir / ("Scripts/python.exe" if is_windows else "bin/python")
        return {
            "status": "applied",
            "python": str(python_path),
        }

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

    @staticmethod
    def find_conda_python(env_name: str) -> str | None:
        """Find the Python executable for a named conda env."""
        try:
            result = subprocess.run(
                ["conda", "run", "-n", env_name, "python", "-c", "import sys; print(sys.executable)"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                path = result.stdout.strip().split("\n")[-1].strip()
                if path and Path(path).exists():
                    return path
        except Exception:
            pass

        is_windows = platform.system() == "Windows"
        for base in [
            Path.home() / "anaconda3",
            Path.home() / "miniconda3",
            Path("D:/anaconda"),
            Path("C:/anaconda3"),
        ]:
            python_path = (
                base / "envs" / env_name / ("python.exe" if is_windows else "bin/python")
            )
            if python_path.exists():
                return str(python_path)
        return None

    async def install_requirements(self, python: str, code_dir: Path) -> dict[str, Any]:
        """Install dependencies from the best available project manifest."""
        install_plan = self._select_install_plan(code_dir)
        if install_plan is None:
            self._log("No dependency manifest found, skipping pip install")
            return {"status": "skipped", "source": "", "manifest": ""}

        self._log(f"Installing dependencies from {install_plan.source} ...")
        loop = asyncio.get_running_loop()
        attempts = [
            ("primary", install_plan.args),
            ("fallback", install_plan.fallback_args or []),
        ]
        last_failure: dict[str, Any] | None = None
        for strategy, install_args in attempts:
            if not install_args:
                continue
            try:
                proc_result = await loop.run_in_executor(
                    None,
                    lambda: subprocess.run(
                        [python, "-m", "pip", "install", *install_args, "--quiet"],
                        cwd=str(code_dir),
                        capture_output=True,
                        text=True,
                        timeout=600,
                    ),
                )
            except Exception as exc:
                self._log(f"pip install error via {install_plan.source} ({strategy}): {exc}")
                last_failure = {
                    "status": "error",
                    "source": install_plan.source,
                    "manifest": install_plan.manifest_path,
                    "strategy": strategy,
                    "error": str(exc),
                }
                continue

            if proc_result.returncode == 0:
                self._log(f"Dependency install OK via {install_plan.source} ({strategy})")
                return {
                    "status": "installed",
                    "source": install_plan.source,
                    "manifest": install_plan.manifest_path,
                    "strategy": strategy,
                }

            stderr = (proc_result.stderr or "").strip()
            self._log(
                f"pip install returned rc={proc_result.returncode} via "
                f"{install_plan.source} ({strategy}): {stderr[:500]}"
            )
            last_failure = {
                "status": "failed",
                "source": install_plan.source,
                "manifest": install_plan.manifest_path,
                "strategy": strategy,
                "returncode": proc_result.returncode,
                "stderr": stderr[:500],
            }

        return last_failure or {
            "status": "skipped",
            "source": install_plan.source,
            "manifest": install_plan.manifest_path,
        }

    async def create_conda_env(self, env_name: str, code_dir: Path) -> bool:
        """Create a conda environment when requested and missing."""
        env_file = self._find_environment_file(code_dir)
        self._log(f"Creating conda env '{env_name}' ...")
        loop = asyncio.get_running_loop()
        try:
            if env_file is not None:
                proc_result = await loop.run_in_executor(
                    None,
                    lambda: subprocess.run(
                        ["conda", "env", "create", "-n", env_name, "-f", str(env_file)],
                        cwd=str(code_dir),
                        capture_output=True,
                        text=True,
                        timeout=1800,
                    ),
                )
            else:
                proc_result = await loop.run_in_executor(
                    None,
                    lambda: subprocess.run(
                        ["conda", "create", "-y", "-n", env_name, "python=3.10"],
                        cwd=str(code_dir),
                        capture_output=True,
                        text=True,
                        timeout=1200,
                    ),
                )
            if proc_result.returncode == 0:
                self._log(f"Conda env '{env_name}' created")
                return True
            stderr = (proc_result.stderr or "").strip()
            self._log(f"Failed to create conda env '{env_name}': {stderr[:500]}")
        except Exception as exc:
            self._log(f"Conda env creation error: {exc}")
        return False

    @staticmethod
    def _find_environment_file(code_dir: Path) -> Path | None:
        """Find the first supported Conda environment manifest."""
        for name in ("environment.yml", "environment.yaml"):
            candidate = code_dir / name
            if candidate.exists():
                return candidate
        return None

    @classmethod
    def _select_install_plan(cls, code_dir: Path) -> DependencyInstallPlan | None:
        """Choose the best available pip install strategy for a project."""
        requirements_path = code_dir / "requirements.txt"
        if requirements_path.exists():
            return DependencyInstallPlan(
                source="requirements.txt",
                args=["-r", str(requirements_path)],
                manifest_path=str(requirements_path),
            )

        environment_file = cls._find_environment_file(code_dir)
        if environment_file is not None:
            pip_dependencies = cls._extract_pip_dependencies(environment_file)
            if pip_dependencies:
                return DependencyInstallPlan(
                    source=environment_file.name,
                    args=pip_dependencies,
                    manifest_path=str(environment_file),
                )

        pyproject_file = code_dir / "pyproject.toml"
        if pyproject_file.exists():
            return DependencyInstallPlan(
                source="pyproject.toml",
                args=["-e", "."],
                fallback_args=["."],
                manifest_path=str(pyproject_file),
            )

        setup_py = code_dir / "setup.py"
        if setup_py.exists():
            return DependencyInstallPlan(
                source="setup.py",
                args=["-e", "."],
                fallback_args=["."],
                manifest_path=str(setup_py),
            )

        setup_cfg = code_dir / "setup.cfg"
        if setup_cfg.exists():
            return DependencyInstallPlan(
                source="setup.cfg",
                args=["-e", "."],
                fallback_args=["."],
                manifest_path=str(setup_cfg),
            )

        return None

    @classmethod
    def _detect_manifest_reference(cls, code_dir: Path) -> tuple[str, str]:
        requirements_path = code_dir / "requirements.txt"
        if requirements_path.exists():
            return "requirements.txt", str(requirements_path)

        environment_file = cls._find_environment_file(code_dir)
        if environment_file is not None:
            return environment_file.name, str(environment_file)

        for name in ("pyproject.toml", "setup.py", "setup.cfg"):
            candidate = code_dir / name
            if candidate.exists():
                return name, str(candidate)

        return "", ""

    @classmethod
    def inspect_project_manifests(cls, code_dir: Path) -> ProjectManifestSnapshot:
        """Inspect local project manifests in the same priority order as local execution."""
        manifest_source, manifest_path = cls._detect_manifest_reference(code_dir)
        environment_file = cls._find_environment_file(code_dir)
        install_plan = cls._select_install_plan(code_dir)

        environment_source = environment_file.name if environment_file is not None else ""
        environment_path = str(environment_file) if environment_file is not None else ""

        install_source = ""
        install_manifest_path = ""
        install_kind = ""
        if install_plan is not None:
            install_source = install_plan.source
            install_manifest_path = install_plan.manifest_path
            if install_plan.source == "requirements.txt":
                install_kind = "requirements"
            elif install_plan.source in {"environment.yml", "environment.yaml"}:
                install_kind = "environment"
            elif install_plan.args[:2] == ["-e", "."]:
                install_kind = "editable"
            else:
                install_kind = "package"

        return ProjectManifestSnapshot(
            manifest_source=manifest_source,
            manifest_path=manifest_path,
            environment_source=environment_source,
            environment_path=environment_path,
            install_source=install_source,
            install_manifest_path=install_manifest_path,
            install_kind=install_kind,
        )

    @classmethod
    def _collect_declared_dependency_names(cls, code_dir: Path) -> set[str]:
        declared: set[str] = set()

        requirements_path = code_dir / "requirements.txt"
        if requirements_path.exists():
            declared.update(cls._extract_requirement_dependency_names(requirements_path))

        environment_file = cls._find_environment_file(code_dir)
        if environment_file is not None:
            for dependency in cls._extract_pip_dependencies(environment_file):
                normalized = _canonicalize_dependency_name(dependency)
                if normalized:
                    declared.add(normalized)

        pyproject_file = code_dir / "pyproject.toml"
        if pyproject_file.exists():
            declared.update(cls._extract_pyproject_dependency_names(pyproject_file))

        setup_cfg = code_dir / "setup.cfg"
        if setup_cfg.exists():
            declared.update(cls._extract_setup_cfg_dependency_names(setup_cfg))

        return declared

    @classmethod
    def _extract_requirement_dependency_names(cls, requirements_file: Path) -> list[str]:
        try:
            lines = requirements_file.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        return cls._normalize_dependency_specs(lines)

    @staticmethod
    def _extract_pip_dependencies(environment_file: Path) -> list[str]:
        """Extract pip-installable dependencies from environment.yml."""
        if not environment_file.exists():
            return []

        try:
            lines = environment_file.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []

        dependencies: list[str] = []
        in_dependencies_block = False
        dependencies_indent = 0
        in_pip_block = False
        pip_indent = 0
        for raw_line in lines:
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            indent = len(raw_line) - len(raw_line.lstrip())
            if stripped == "dependencies:":
                in_dependencies_block = True
                dependencies_indent = indent
                in_pip_block = False
                continue
            if in_dependencies_block and indent <= dependencies_indent and not stripped.startswith("- "):
                in_dependencies_block = False
                in_pip_block = False

            if not in_dependencies_block:
                continue

            if stripped in {"- pip:", "pip:"}:
                in_pip_block = True
                pip_indent = indent
                continue

            if in_pip_block:
                if indent <= pip_indent:
                    in_pip_block = False
                elif stripped.startswith("- "):
                    dependency = stripped[2:].strip()
                    if dependency:
                        dependencies.append(dependency)
                    continue

            if not in_pip_block and stripped.startswith("- "):
                dependency = stripped[2:].strip()
                has_conda_single_equals = bool(
                    re.search(r"(^|[^<>=!~])=([^=]|$)", dependency)
                )
                if (
                    dependency
                    and not dependency.startswith(("python", "pip"))
                    and not has_conda_single_equals
                ):
                    dependencies.append(dependency)

        return dependencies

    @classmethod
    def _extract_pyproject_dependency_names(cls, pyproject_file: Path) -> list[str]:
        if tomllib is None:
            return []

        try:
            parsed = tomllib.loads(pyproject_file.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            return []

        dependencies: list[str] = []
        project = parsed.get("project", {})
        if isinstance(project, dict):
            declared = project.get("dependencies", [])
            if isinstance(declared, list):
                dependencies.extend(str(item) for item in declared)
            optional = project.get("optional-dependencies", {})
            if isinstance(optional, dict):
                for values in optional.values():
                    if isinstance(values, list):
                        dependencies.extend(str(item) for item in values)

        return cls._normalize_dependency_specs(dependencies)

    @classmethod
    def _extract_setup_cfg_dependency_names(cls, setup_cfg_file: Path) -> list[str]:
        parser = configparser.ConfigParser()
        try:
            parser.read(setup_cfg_file, encoding="utf-8")
        except (configparser.Error, OSError):
            return []

        dependency_specs: list[str] = []
        if parser.has_section("options") and parser.has_option("options", "install_requires"):
            dependency_specs.extend(
                line.strip()
                for line in parser.get("options", "install_requires").splitlines()
            )

        if parser.has_section("options.extras_require"):
            for _, value in parser.items("options.extras_require"):
                dependency_specs.extend(line.strip() for line in value.splitlines())

        return cls._normalize_dependency_specs(dependency_specs)

    @staticmethod
    def _normalize_dependency_specs(specs: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for raw_spec in specs:
            dependency_name = _canonicalize_dependency_name(raw_spec)
            if not dependency_name or dependency_name in seen:
                continue
            seen.add(dependency_name)
            normalized.append(dependency_name)
        return normalized
