"""Code runner helpers: timeout fix, venv setup, legacy runner, subprocess execution."""
from __future__ import annotations

import asyncio
import logging
import os
import re as _re
import subprocess
from pathlib import Path
from typing import Any

from nanoresearch.agents.project_runner import RUNNER_SCRIPT_NAME, ensure_project_runner
from nanoresearch.agents.repair_journal import (
    append_snapshot_journal,
    capture_repair_snapshot,
    rollback_snapshot,
)
from nanoresearch.agents.runtime_env import RuntimeEnvironmentManager

from . import (
    _decode_bytes,
    DRY_RUN_TIMEOUT_SECONDS,
    SUBPROCESS_OUTPUT_LIMIT,
)

logger = logging.getLogger(__name__)


class _CodeRunnerHelpersMixin:
    """Mixin — timeout fix, venv setup, legacy runner, subprocess execution."""

    async def _fix_timeout(self, code_dir: Path) -> list[str]:
        """When quick-eval times out, apply deterministic speed-ups to main.py.

        Instead of sending a vague "timeout" to the LLM for batch-fix, we apply
        targeted edits that reduce computation: fewer epochs, smaller subset,
        num_workers=0, num_runs=1.
        """
        self._remember_mutation_snapshot_entry(None)
        main_py = code_dir / "main.py"
        if not main_py.exists():
            return []
        content = main_py.read_text(encoding="utf-8", errors="replace")
        original = content

        # 1. Reduce epochs to 2 (replace any epochs = N where N > 2)
        # Handles: epochs = 5, epochs: 5 (YAML), training_cfg["epochs"] = max(3, min(5, ...))
        content = _re.sub(
            r'(\bepochs\b\s*[=:]\s*(?:max\(\d+,\s*min\(\d+,\s*(?:int\()?)?)(\d+)',
            lambda m: m.group(1) + ('2' if int(m.group(2)) > 2 else m.group(2)),
            content,
        )

        # 2. Reduce data subset size
        content = _re.sub(
            r'(subset_size\s*[=:]\s*)(\d+)',
            lambda m: m.group(1) + ('200' if int(m.group(2)) > 200 else m.group(2)),
            content,
        )
        content = _re.sub(
            r'(quick_eval_train_size["\']?\s*[,:]\s*)(\d+)',
            lambda m: m.group(1) + ('200' if int(m.group(2)) > 200 else m.group(2)),
            content,
        )

        # 3. Force num_runs = 1
        content = _re.sub(
            r'(num_runs\s*[=:]\s*)(\d+)',
            lambda m: m.group(1) + '1',
            content,
        )

        # 4. Force num_workers = 0 (avoid multiprocessing overhead on Windows)
        content = _re.sub(
            r'(num_workers\s*[=:]\s*)(\d+)',
            lambda m: m.group(1) + '0',
            content,
        )

        if content != original:
            snapshot = capture_repair_snapshot(
                self.workspace.path,
                main_py,
                namespace="timeout_fix",
                root_dir=self.workspace.path,
                operation="rewrite",
            )
            main_py.write_text(content, encoding="utf-8")
            if not self._check_syntax(main_py):
                self.log("Timeout fix introduced invalid syntax in main.py, rolling back")
                rollback_snapshot(self.workspace.path, main_py, snapshot)
                snapshot["rolled_back"] = True
                snapshot["rollback_reason"] = "syntax_error"
                entry = append_snapshot_journal(
                    self.workspace.path,
                    agent=self.__class__.__name__,
                    mutation_kind="timeout_fix",
                    scope="legacy_timeout_fix",
                    snapshots=[snapshot],
                    metadata={"modified_files": []},
                )
                self._remember_mutation_snapshot_entry(entry)
                return []

            entry = append_snapshot_journal(
                self.workspace.path,
                agent=self.__class__.__name__,
                mutation_kind="timeout_fix",
                scope="legacy_timeout_fix",
                snapshots=[snapshot],
                metadata={"modified_files": ["main.py"]},
            )
            self._remember_mutation_snapshot_entry(entry)
            self.log("Timeout fix: reduced epochs/subset/workers in main.py")
            return ["main.py"]

        # If main.py regex didn't match anything, also try config/default.yaml
        config_yaml = code_dir / "config" / "default.yaml"
        if config_yaml.exists():
            cfg_content = config_yaml.read_text(encoding="utf-8", errors="replace")
            cfg_original = cfg_content
            cfg_content = _re.sub(
                r'(\bepochs\s*:\s*)(\d+)',
                lambda m: m.group(1) + ('2' if int(m.group(2)) > 2 else m.group(2)),
                cfg_content,
            )
            cfg_content = _re.sub(r'(num_workers\s*:\s*)(\d+)', r'\g<1>0', cfg_content)
            cfg_content = _re.sub(r'(num_runs\s*:\s*)(\d+)', r'\g<1>1', cfg_content)
            if cfg_content != cfg_original:
                snapshot = capture_repair_snapshot(
                    self.workspace.path,
                    config_yaml,
                    namespace="timeout_fix",
                    root_dir=self.workspace.path,
                    operation="rewrite",
                )
                config_yaml.write_text(cfg_content, encoding="utf-8")
                entry = append_snapshot_journal(
                    self.workspace.path,
                    agent=self.__class__.__name__,
                    mutation_kind="timeout_fix",
                    scope="legacy_timeout_fix",
                    snapshots=[snapshot],
                    metadata={"modified_files": ["config/default.yaml"]},
                )
                self._remember_mutation_snapshot_entry(entry)
                self.log("Timeout fix: reduced epochs/workers/runs in config/default.yaml")
                return ["config/default.yaml"]

        return []

    async def _setup_venv(self, code_dir: Path) -> str:
        """Prepare an ISOLATED Python environment for experiment execution.

        Always creates a fresh venv (or auto-repair conda env) so the user's
        own environment is never polluted.  The configured experiment_conda_env
        is intentionally skipped — experiments must be reproducible in isolation.

        Returns the path to the Python executable.
        """
        # Build a deterministic label so conda env name is idempotent across
        # resumes of the same session — avoids creating duplicate envs.
        session_label = ""
        if hasattr(self, "workspace") and self.workspace:
            m = self.workspace.manifest
            sid = m.session_id[:8]
            slug = m.topic[:20].replace(" ", "_") if m.topic else ""
            session_label = f"{slug}_{sid}" if slug else sid
        runtime = RuntimeEnvironmentManager(self.config, self.log, session_label=session_label)
        env_info = await runtime.prepare(code_dir)
        python = env_info.get("python")
        if not python:
            raise RuntimeError(
                "RuntimeEnvironmentManager.prepare() returned no 'python' key. "
                "Refusing to fall back to system Python."
            )
        return str(python)

    @staticmethod
    def _find_conda_python(env_name: str) -> str | None:
        """Find the Python executable for a named conda env."""
        return RuntimeEnvironmentManager.find_conda_python(env_name)

    async def _install_missing_requirements(self, python: str, code_dir: Path) -> None:
        """pip install requirements.txt if it exists (skips already-installed)."""
        runtime = RuntimeEnvironmentManager(self.config, self.log)
        await runtime.install_requirements(python, code_dir)

    @staticmethod
    def _find_legacy_entry_script(code_dir: Path) -> Path | None:
        """Return the first supported legacy experiment entry script."""
        for candidate in ("main.py", "train.py", "run.py"):
            script_path = code_dir / candidate
            if script_path.exists():
                return script_path
        return None

    def _ensure_legacy_runner(self, code_dir: Path) -> dict[str, Any] | None:
        """Materialize deterministic runner assets for legacy experiment projects."""
        entry_script = self._find_legacy_entry_script(code_dir)
        if entry_script is None:
            return None
        return ensure_project_runner(code_dir, f"python {entry_script.name}")

    def _build_legacy_runner_command(self, code_dir: Path, *, mode: str) -> str | None:
        """Build a runner-backed shell command for legacy experiment execution."""
        runner_assets = self._ensure_legacy_runner(code_dir)
        if runner_assets is None:
            return None

        command = str(runner_assets.get("runner_command") or f"python {RUNNER_SCRIPT_NAME}").strip()
        if mode == "dry-run":
            return f"{command} --dry-run"
        if mode == "quick-eval":
            return f"{command} --quick-eval"
        return command

    def _build_legacy_subprocess_command(
        self,
        code_dir: Path,
        python: str | None,
        *,
        mode: str,
    ) -> list[str] | None:
        """Build the concrete argv for a legacy experiment subprocess."""
        if self._build_legacy_runner_command(code_dir, mode=mode) is None:
            return None

        if not python:
            raise RuntimeError(
                "No Python executable provided to _build_legacy_subprocess_command. "
                "Ensure _setup_venv() was called first."
            )
        normalized_python = python
        suffix: list[str] = []
        if mode == "dry-run":
            suffix = ["--dry-run"]
        elif mode == "quick-eval":
            suffix = ["--quick-eval"]
        return [normalized_python, RUNNER_SCRIPT_NAME, *suffix]

    async def _run_main_py(self, code_dir: Path, python: str | None = None) -> dict:
        """Run the legacy experiment entrypoint in dry-run mode with timeout."""
        if not python:
            raise RuntimeError(
                "No Python executable provided to _run_main_py. "
                "Ensure _setup_venv() was called first."
            )
        command = self._build_legacy_subprocess_command(code_dir, python, mode="dry-run")
        if command is None:
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": "No runnable entry script found (expected one of main.py/train.py/run.py)",
            }
        loop = asyncio.get_running_loop()
        try:
            proc_result = await loop.run_in_executor(
                None,
                lambda: self._run_with_tree_kill(
                    command,
                    cwd=str(code_dir),
                    timeout=DRY_RUN_TIMEOUT_SECONDS,
                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                ),
            )
            return proc_result
        except Exception as e:
            return {"returncode": -1, "stdout": "", "stderr": str(e)}

    @staticmethod
    def _run_with_tree_kill(
        command: list[str],
        *,
        cwd: str | None = None,
        timeout: int = 60,
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Run subprocess with proper process-tree cleanup on timeout."""
        from nanoresearch.agents.execution.cluster_runner import _kill_process_tree

        proc = subprocess.Popen(
            command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_process_tree(proc.pid)
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            try:
                proc.communicate(timeout=5)
            except (subprocess.TimeoutExpired, OSError):
                pass
            return {"returncode": -1, "stdout": "", "stderr": f"Timeout after {timeout}s"}
        return {
            "returncode": proc.returncode,
            "stdout": _decode_bytes(stdout, SUBPROCESS_OUTPUT_LIMIT),
            "stderr": _decode_bytes(stderr, SUBPROCESS_OUTPUT_LIMIT),
        }
