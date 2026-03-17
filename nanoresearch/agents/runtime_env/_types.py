"""Shared data types for runtime environment management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ._discovery import _canonicalize_dependency_name

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


