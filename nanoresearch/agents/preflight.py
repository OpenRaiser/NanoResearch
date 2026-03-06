"""Fail-fast preflight checks for experiment code projects.

All checks are pure static/local — no LLM calls, no network access.
Designed to run in < 1 second.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from nanoresearch.schemas.iteration import PreflightReport, PreflightResult

logger = logging.getLogger(__name__)

# Keys that config/default.yaml must contain.
# Each required key maps to a set of accepted aliases (LLMs use varied names).
_REQUIRED_CONFIG_KEYS = {"random_seed"}
_CONFIG_KEY_ALIASES: dict[str, set[str]] = {
    "random_seed": {"random_seed", "seed", "rand_seed", "manual_seed"},
}

# Known framework conflicts (having both simultaneously is suspicious)
_FRAMEWORK_CONFLICTS = [
    ({"torch", "pytorch"}, {"tensorflow", "tf"}),
]


class PreflightChecker:
    """Run static preflight checks on a generated code project."""

    def __init__(self, code_dir: Path) -> None:
        self.code_dir = code_dir

    def run_all(self) -> PreflightReport:
        """Execute all checks and return an aggregated report."""
        checks = [
            self.check_config_yaml(),
            self.check_requirements(),
            self.check_data_references(),
            self.check_main_entrypoint(),
            self.check_import_resolution(),
        ]

        blocking = [c.check_name for c in checks if c.status == "failed"]
        has_warnings = any(c.status == "warning" for c in checks)

        if blocking:
            overall = "failed"
        elif has_warnings:
            overall = "warnings"
        else:
            overall = "passed"

        return PreflightReport(
            overall_status=overall,
            checks=checks,
            blocking_failures=blocking,
        )

    # ------------------------------------------------------------------
    # 1. config/default.yaml — blocking
    # ------------------------------------------------------------------
    def check_config_yaml(self) -> PreflightResult:
        """Verify config/default.yaml exists, is parseable YAML, and has required keys."""
        yaml_path = self.code_dir / "config" / "default.yaml"
        if not yaml_path.exists():
            return PreflightResult(
                check_name="config_yaml",
                status="failed",
                message="config/default.yaml not found",
            )

        try:
            text = yaml_path.read_text(encoding="utf-8")
        except OSError as exc:
            return PreflightResult(
                check_name="config_yaml",
                status="failed",
                message=f"Cannot read config/default.yaml: {exc}",
            )

        # Try to parse YAML (use a simple key-detection approach to avoid
        # hard dependency on PyYAML at import time)
        try:
            import yaml  # type: ignore[import-untyped]

            data = yaml.safe_load(text)
            if not isinstance(data, dict):
                return PreflightResult(
                    check_name="config_yaml",
                    status="failed",
                    message="config/default.yaml does not parse as a YAML mapping",
                )
            # Flatten nested keys for checking (e.g. top-level or one level deep)
            all_keys = set(data.keys())
            for v in data.values():
                if isinstance(v, dict):
                    all_keys.update(v.keys())

            # Check required keys, accepting aliases
            missing = []
            for req_key in _REQUIRED_CONFIG_KEYS:
                aliases = _CONFIG_KEY_ALIASES.get(req_key, {req_key})
                if not (all_keys & aliases):
                    missing.append(req_key)
            if missing:
                return PreflightResult(
                    check_name="config_yaml",
                    status="failed",
                    message=f"config/default.yaml missing required keys: {set(missing)}",
                    details={"missing_keys": sorted(missing)},
                )
        except ImportError:
            # PyYAML not available — do a simple text-based check
            for key in _REQUIRED_CONFIG_KEYS:
                aliases = _CONFIG_KEY_ALIASES.get(key, {key})
                if not any(alias in text for alias in aliases):
                    return PreflightResult(
                        check_name="config_yaml",
                        status="failed",
                        message=f"config/default.yaml appears to be missing key: {key} (PyYAML unavailable for full parse)",
                    )
        except Exception as exc:
            return PreflightResult(
                check_name="config_yaml",
                status="failed",
                message=f"config/default.yaml is not valid YAML: {exc}",
            )

        return PreflightResult(
            check_name="config_yaml",
            status="passed",
            message="config/default.yaml OK",
        )

    # ------------------------------------------------------------------
    # 2. requirements.txt — warning
    # ------------------------------------------------------------------
    def check_requirements(self) -> PreflightResult:
        """Check requirements.txt for parse errors and obvious conflicts."""
        req_path = self.code_dir / "requirements.txt"
        if not req_path.exists():
            return PreflightResult(
                check_name="requirements",
                status="warning",
                message="requirements.txt not found",
            )

        try:
            text = req_path.read_text(encoding="utf-8")
        except OSError as exc:
            return PreflightResult(
                check_name="requirements",
                status="warning",
                message=f"Cannot read requirements.txt: {exc}",
            )

        # Collect package base names (lowercased, before any version specifier)
        pkg_names: set[str] = set()
        warnings: list[str] = []
        for lineno, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            # Extract package name (before >=, ==, <, !, [, etc.)
            match = re.match(r"^([A-Za-z0-9_.-]+)", line)
            if match:
                pkg_names.add(match.group(1).lower().replace("-", "_"))
            else:
                warnings.append(f"Line {lineno}: unparseable requirement '{line}'")

        # Check for framework conflicts
        for group_a, group_b in _FRAMEWORK_CONFLICTS:
            has_a = pkg_names & group_a
            has_b = pkg_names & group_b
            if has_a and has_b:
                warnings.append(
                    f"Possible framework conflict: {has_a} and {has_b} both present"
                )

        if warnings:
            return PreflightResult(
                check_name="requirements",
                status="warning",
                message="; ".join(warnings),
                details={"warnings": warnings},
            )

        return PreflightResult(
            check_name="requirements",
            status="passed",
            message="requirements.txt OK",
        )

    # ------------------------------------------------------------------
    # 3. Data references — warning
    # ------------------------------------------------------------------
    def check_data_references(self) -> PreflightResult:
        """Scan code for data paths/URLs; check for synthetic data fallback."""
        warnings: list[str] = []
        hardcoded_paths: list[str] = []

        for py_file in self.code_dir.rglob("*.py"):
            # Skip venv and hidden dirs
            parts = py_file.relative_to(self.code_dir).parts
            if any(p.startswith(".") or p == "__pycache__" for p in parts):
                continue

            try:
                source = py_file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            # Detect hardcoded absolute data paths
            for match in re.finditer(r"""['"](/(?:data|datasets?|mnt|home)/[^'"]+)['"]""", source):
                hardcoded_paths.append(f"{py_file.name}: {match.group(1)}")

        if hardcoded_paths:
            warnings.append(
                f"Hardcoded data paths found ({len(hardcoded_paths)}): "
                + "; ".join(hardcoded_paths[:3])
            )

        # Check that --quick-eval has a synthetic/fallback path
        main_py = self.code_dir / "main.py"
        if main_py.exists():
            try:
                main_source = main_py.read_text(encoding="utf-8", errors="replace")
                has_quick_eval = "--quick-eval" in main_source
                has_synthetic = any(
                    kw in main_source.lower()
                    for kw in ("synthetic", "random", "randn", "fake_data", "dummy")
                )
                if has_quick_eval and not has_synthetic:
                    warnings.append(
                        "main.py has --quick-eval but no obvious synthetic data fallback"
                    )
            except OSError:
                pass

        if warnings:
            return PreflightResult(
                check_name="data_references",
                status="warning",
                message="; ".join(warnings),
                details={"warnings": warnings},
            )

        return PreflightResult(
            check_name="data_references",
            status="passed",
            message="Data references OK",
        )

    # ------------------------------------------------------------------
    # 4. main.py entrypoint — blocking
    # ------------------------------------------------------------------
    def check_main_entrypoint(self) -> PreflightResult:
        """Verify main.py exists and contains --dry-run and --quick-eval strings."""
        main_py = self.code_dir / "main.py"
        if not main_py.exists():
            return PreflightResult(
                check_name="main_entrypoint",
                status="failed",
                message="main.py not found",
            )

        try:
            source = main_py.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return PreflightResult(
                check_name="main_entrypoint",
                status="failed",
                message=f"Cannot read main.py: {exc}",
            )

        missing_flags: list[str] = []
        if "--dry-run" not in source and "dry_run" not in source:
            missing_flags.append("--dry-run")
        if "--quick-eval" not in source and "quick_eval" not in source:
            missing_flags.append("--quick-eval")

        if missing_flags:
            return PreflightResult(
                check_name="main_entrypoint",
                status="failed",
                message=f"main.py missing flag handling: {missing_flags}",
                details={"missing_flags": missing_flags},
            )

        return PreflightResult(
            check_name="main_entrypoint",
            status="passed",
            message="main.py entrypoint OK",
        )

    # ------------------------------------------------------------------
    # 5. Import resolution — warning
    # ------------------------------------------------------------------
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
                module_path = match.group(1)  # e.g. "src.model"
                # Convert to file path: src.model -> src/model.py
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
                details={"unresolved": warnings[:10]},
            )

        return PreflightResult(
            check_name="import_resolution",
            status="passed",
            message="All src.* imports resolve",
        )
