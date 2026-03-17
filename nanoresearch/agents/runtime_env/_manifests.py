"""Project manifest parsing mixin."""

from __future__ import annotations

import configparser
import json
import logging
import re
from pathlib import Path
from typing import Any

from ._discovery import _canonicalize_dependency_name
from ._types import DependencyInstallPlan, ProjectManifestSnapshot

logger = logging.getLogger(__name__)

try:
    import tomllib
except ModuleNotFoundError:
    tomllib = None


class _ManifestsMixin:
    """Mixin — project manifest detection and dependency extraction."""

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
