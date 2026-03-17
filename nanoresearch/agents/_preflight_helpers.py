"""Preflight checker helpers — import resolution and environment parsing."""

from __future__ import annotations

import re
from pathlib import Path

from nanoresearch.schemas.iteration import PreflightResult


class _PreflightHelpersMixin:
    """Mixin — import resolution and environment file parsing for PreflightChecker."""

    def check_import_resolution(self) -> PreflightResult:
        """Check that all 'from src.xxx import' can resolve to src/xxx.py files."""
        warnings: list[str] = []
        import_pattern = re.compile(r"from\s+(src\.\w+(?:\.\w+)*)\s+import")

        for py_file in self.code_dir.rglob("*.py"):
            parts = py_file.relative_to(self.code_dir).parts
            if any(p.startswith(".") or p == "__pycache__" for p in parts):
                continue

            try:
                source = py_file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            for match in import_pattern.finditer(source):
                module_path = match.group(1)
                rel_path = module_path.replace(".", "/")
                candidate_file = self.code_dir / (rel_path + ".py")
                candidate_pkg = self.code_dir / rel_path / "__init__.py"
                if not candidate_file.exists() and not candidate_pkg.exists():
                    warnings.append(
                        f"{py_file.name}: 'from {module_path} import ...' "
                        f"— neither {rel_path}.py nor {rel_path}/__init__.py found"
                    )

        if warnings:
            return PreflightResult(
                check_name="import_resolution",
                status="warning",
                message=f"{len(warnings)} unresolved import(s)",
                details={
                    "unresolved": warnings[:10],
                    "suggested_fixes": [
                        "Create the missing src modules/packages or update import paths to match generated file names."
                    ],
                },
            )

        return PreflightResult(
            check_name="import_resolution",
            status="passed",
            message="All src.* imports resolve",
        )

    @staticmethod
    def _extract_environment_pip_dependencies(environment_file: Path) -> list[str]:
        if not environment_file.exists():
            return []
        try:
            lines = environment_file.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []

        dependencies: list[str] = []
        in_pip_block = False
        pip_indent = 0
        for raw_line in lines:
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            indent = len(raw_line) - len(raw_line.lstrip())
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

        return dependencies
