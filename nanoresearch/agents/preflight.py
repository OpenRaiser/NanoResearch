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

        failed_checks = [check for check in checks if check.status == "failed"]
        warning_checks = [check for check in checks if check.status == "warning"]
        blocking = [self._format_check_summary(check) for check in failed_checks]
        has_warnings = bool(warning_checks)

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
            blocking_check_names=[check.check_name for check in failed_checks],
            warning_messages=[self._format_check_summary(check) for check in warning_checks],
            warning_check_names=[check.check_name for check in warning_checks],
            suggested_fixes=self._collect_suggested_fixes(checks),
        )

    @staticmethod
    def _format_check_summary(check: PreflightResult) -> str:
        return f"{check.check_name}: {check.message}"

    @staticmethod
    def _collect_suggested_fixes(checks: list[PreflightResult]) -> list[str]:
        fixes: list[str] = []
        for check in checks:
            for fix in check.details.get("suggested_fixes", []):
                normalized = str(fix).strip()
                if normalized and normalized not in fixes:
                    fixes.append(normalized)
        return fixes

    def _entrypoint_candidates(self) -> list[Path]:
        return [
            self.code_dir / "main.py",
            self.code_dir / "train.py",
            self.code_dir / "run.py",
            self.code_dir / "run_train.py",
            self.code_dir / "experiment.py",
            self.code_dir / "scripts" / "train.py",
            self.code_dir / "scripts" / "run.py",
            self.code_dir / "src" / "main.py",
        ]

    def _dependency_manifests(self) -> list[Path]:
        manifests: list[Path] = []
        for rel_path in (
            "requirements.txt",
            "environment.yml",
            "environment.yaml",
            "pyproject.toml",
            "setup.py",
            "setup.cfg",
        ):
            candidate = self.code_dir / rel_path
            if candidate.exists():
                manifests.append(candidate)
        return manifests

    # ------------------------------------------------------------------
    # 1. config/default.yaml — blocking
    # ------------------------------------------------------------------
    def check_config_yaml(self) -> PreflightResult:
        """Verify a config file exists (YAML, JSON, or Python config module)."""
        yaml_path = self.code_dir / "config" / "default.yaml"
        # Accept alternative config formats that LLMs commonly generate
        alt_configs = [
            self.code_dir / "config.py",
            self.code_dir / "config.json",
            self.code_dir / "config.yaml",
            self.code_dir / "config" / "config.yaml",
            self.code_dir / "config" / "config.py",
            self.code_dir / "configs" / "default.yaml",
        ]
        if not yaml_path.exists():
            # Check alternative config files
            for alt in alt_configs:
                if alt.exists():
                    return PreflightResult(
                        check_name="config_yaml",
                        status="passed",
                        message=f"Config found at {alt.name} (alternative format)",
                        details={"config_path": str(alt)},
                    )
            return PreflightResult(
                check_name="config_yaml",
                status="warning",
                message="No config file found (config/default.yaml, config.py, etc.)",
                details={
                    "expected_paths": [str(yaml_path), *(str(path) for path in alt_configs)],
                    "suggested_fixes": [
                        "Add config/default.yaml or an alternative config module like config.py.",
                        "Include a random_seed/seed field so dry-run can validate determinism.",
                    ],
                },
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
                    details={
                        "config_path": str(yaml_path),
                        "suggested_fixes": [
                            "Rewrite config/default.yaml as a valid YAML mapping with top-level key/value pairs."
                        ],
                    },
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
                    details={
                        "config_path": str(yaml_path),
                        "missing_keys": sorted(missing),
                        "suggested_fixes": [
                            f"Add the missing config keys to config/default.yaml: {', '.join(sorted(missing))}."
                        ],
                    },
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
                        details={
                            "config_path": str(yaml_path),
                            "missing_keys": [key],
                            "suggested_fixes": [f"Add '{key}' (or one of its aliases) to config/default.yaml."],
                        },
                    )
        except Exception as exc:
            return PreflightResult(
                check_name="config_yaml",
                status="failed",
                message=f"config/default.yaml is not valid YAML: {exc}",
                details={
                    "config_path": str(yaml_path),
                    "suggested_fixes": [
                        "Fix YAML syntax in config/default.yaml so it parses cleanly."
                    ],
                },
            )

        return PreflightResult(
            check_name="config_yaml",
            status="passed",
            message="config/default.yaml OK",
            details={"config_path": str(yaml_path)},
        )

    # ------------------------------------------------------------------
    # 2. requirements.txt — warning
    # ------------------------------------------------------------------
    def check_requirements(self) -> PreflightResult:
        """Check requirements.txt for parse errors and obvious conflicts."""
        req_path = self.code_dir / "requirements.txt"
        manifests = self._dependency_manifests()
        if not req_path.exists():
            for manifest in manifests:
                if manifest.name != "requirements.txt":
                    details = {"manifest": str(manifest)}
                    if manifest.name in {"environment.yml", "environment.yaml"}:
                        pip_dependencies = self._extract_environment_pip_dependencies(manifest)
                        details["pip_dependencies"] = pip_dependencies
                        if not pip_dependencies:
                            return PreflightResult(
                                check_name="requirements",
                                status="warning",
                                message=(
                                    f"{manifest.name} found but it has no pip-installable dependencies; "
                                    "local venv execution may still need a conda env"
                                ),
                                details={
                                    **details,
                                    "suggested_fixes": [
                                        "Add a pip block to environment.yml/environment.yaml or provide requirements.txt for local execution.",
                                        "If the project depends on Conda-only packages, run with experiment_conda_env configured.",
                                    ],
                                },
                            )
                    return PreflightResult(
                        check_name="requirements",
                        status="passed",
                        message=f"Dependency manifest found at {manifest.name}",
                        details=details,
                    )
            return PreflightResult(
                check_name="requirements",
                status="warning",
                message="No dependency manifest found (requirements.txt, environment.yml, pyproject.toml, setup.py)",
                details={
                    "expected_manifests": [
                        "requirements.txt",
                        "environment.yml",
                        "environment.yaml",
                        "pyproject.toml",
                        "setup.py",
                        "setup.cfg",
                    ],
                    "suggested_fixes": [
                        "Add requirements.txt, pyproject.toml, or setup.py so the runtime manager can install dependencies.",
                    ],
                },
            )

        try:
            text = req_path.read_text(encoding="utf-8")
        except OSError as exc:
            return PreflightResult(
                check_name="requirements",
                status="warning",
                message=f"Cannot read requirements.txt: {exc}",
                details={
                    "manifest": str(req_path),
                    "suggested_fixes": ["Fix file permissions or regenerate requirements.txt."],
                },
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
                details={
                    "manifest": str(req_path),
                    "warnings": warnings,
                    "suggested_fixes": [
                        "Remove malformed lines from requirements.txt and avoid mixing conflicting frameworks in one environment."
                    ],
                },
            )

        return PreflightResult(
            check_name="requirements",
            status="passed",
            message="requirements.txt OK",
            details={"manifest": str(req_path)},
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
                details={
                    "warnings": warnings,
                    "hardcoded_paths": hardcoded_paths[:10],
                    "suggested_fixes": [
                        "Replace hardcoded absolute data paths with config-driven or relative paths.",
                        "Add a synthetic or dummy-data fallback for quick-eval mode.",
                    ],
                },
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
        """Verify a Python entrypoint exists (main.py, train.py, run.py, etc.)."""
        main_py: Path | None = None
        for candidate in self._entrypoint_candidates():
            if candidate.exists():
                main_py = candidate
                break
        if main_py is None:
            searched_paths = [str(path.relative_to(self.code_dir)) for path in self._entrypoint_candidates()]
            return PreflightResult(
                check_name="main_entrypoint",
                status="warning",
                message="No standard Python entrypoint found (main.py, train.py, run.py, scripts/train.py)",
                details={
                    "searched_paths": searched_paths,
                    "suggested_fixes": [
                        "Add a runnable entrypoint such as main.py, train.py, or scripts/train.py.",
                        "Make the entrypoint accept --dry-run and --quick-eval so execution can validate it automatically.",
                    ],
                },
            )

        try:
            source = main_py.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return PreflightResult(
                check_name="main_entrypoint",
                status="failed",
                message=f"Cannot read main.py: {exc}",
                details={
                    "entrypoint_path": str(main_py),
                    "suggested_fixes": [f"Fix file permissions or rewrite {main_py.name} so it can be inspected."],
                },
            )

        missing_flags: list[str] = []
        if "--dry-run" not in source and "dry_run" not in source:
            missing_flags.append("--dry-run")
        if "--quick-eval" not in source and "quick_eval" not in source:
            missing_flags.append("--quick-eval")

        if missing_flags:
            return PreflightResult(
                check_name="main_entrypoint",
                status="warning",
                message=f"{main_py.name} missing flag handling: {missing_flags}",
                details={
                    "entrypoint_path": str(main_py),
                    "missing_flags": missing_flags,
                    "suggested_fixes": [
                        f"Update {main_py.name} to accept {', '.join(missing_flags)} and exit cleanly in those modes."
                    ],
                },
            )

        return PreflightResult(
            check_name="main_entrypoint",
            status="passed",
            message=f"{main_py.name} entrypoint OK",
            details={"entrypoint_path": str(main_py)},
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
