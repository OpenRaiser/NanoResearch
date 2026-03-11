"""Code execution: cluster, local subprocess, batch fix, env setup."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re as _re
import subprocess
import sys
from pathlib import Path
from typing import Any

from nanoresearch.agents.cluster_executor import ClusterExecutor
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
    STDERR_SNIPPET_LIMIT,
    LLM_CONTEXT_TRUNCATION,
)

logger = logging.getLogger(__name__)


class _CodeRunnerMixin:
    """Mixin — code execution, batch fix, environment setup."""

    async def _run_on_cluster(
        self,
        cluster: "ClusterExecutor",
        code_dir: Path,
        round_num: int,
        cluster_code_path: str,
    ) -> tuple[dict, dict]:
        """Run experiment on SLURM cluster (local or remote).

        Returns (execution_dict, quick_eval_dict) in the same format as
        the local execution path.
        """
        session_id = self.workspace.path.name

        try:
            runner_command = self._build_legacy_runner_command(
                code_dir,
                mode="quick-eval",
            )
            if runner_command is None:
                return (
                    {
                        "status": "skipped",
                        "cluster_code_path": cluster_code_path,
                        "stderr": "No runnable entry script found (expected one of main.py/train.py/run.py)",
                    },
                    {"status": "skipped", "metrics": {}},
                )

            # Step 1: Prepare code on cluster
            if not cluster_code_path:
                self.log("Preparing code on cluster...")
                cluster_code_path = await cluster.prepare_code(code_dir, session_id)

                # Step 2: Create conda env + install deps (first round only)
                env_result = await cluster.setup_env(cluster_code_path)
                if not env_result["ok"]:
                    return (
                        {
                            "status": "failed",
                            "cluster_code_path": cluster_code_path,
                            "stderr": f"Environment setup failed:\n{env_result['output'][-2000:]}",
                        },
                        {"status": "skipped", "metrics": {}},
                    )
            else:
                # Re-sync code after LLM modifications
                self.log("Re-syncing code to cluster...")
                await cluster.reupload_code(code_dir, cluster_code_path)

            # Step 3: Submit SLURM job
            script_cmd = runner_command
            job_id = await cluster.submit_job(cluster_code_path, script_cmd)

            # Step 4: Wait for completion
            job_status = await cluster.wait_for_job(job_id)
            state = job_status.get("state", "UNKNOWN")

            # Step 5: Collect results or error logs
            if state == "COMPLETED":
                downloaded = await cluster.download_results(
                    cluster_code_path, self.workspace.path
                )
                if downloaded:
                    metrics = self._parse_metrics_json(code_dir)
                    if metrics:
                        self.log("Cluster experiment succeeded — real results collected!")
                        return (
                            {
                                "status": "success",
                                "cluster_code_path": cluster_code_path,
                                "job_id": job_id,
                                "stdout": f"Job {job_id} completed",
                                "stderr": "",
                            },
                            {"status": "success", "metrics": metrics},
                        )

                # Job completed but metrics.json missing/invalid
                log_text = await cluster.get_job_log(cluster_code_path, job_id)
                return (
                    {
                        "status": "failed",
                        "cluster_code_path": cluster_code_path,
                        "job_id": job_id,
                        "stdout": f"Job {job_id} rc=0 but metrics.json missing/invalid",
                        "stderr": log_text[-STDERR_SNIPPET_LIMIT:] if log_text else "",
                    },
                    {"status": "partial", "metrics": {}, "stderr": log_text},
                )
            else:
                # Job failed
                log_text = await cluster.get_job_log(cluster_code_path, job_id)
                self.log(f"Cluster job {job_id} failed ({state})")
                return (
                    {
                        "status": "failed",
                        "cluster_code_path": cluster_code_path,
                        "job_id": job_id,
                        "stdout": "",
                        "stderr": log_text[-STDERR_SNIPPET_LIMIT:] if log_text else f"Job {state}",
                    },
                    {"status": "failed", "metrics": {}, "stderr": log_text},
                )

        except Exception as e:
            self.log(f"Cluster execution error: {e}")
            return (
                {
                    "status": "failed",
                    "cluster_code_path": cluster_code_path,
                    "stderr": str(e),
                },
                {"status": "failed", "metrics": {}},
            )

    async def _execute_code_with_venv(
        self, generated_files: list[str], blueprint_summary: str
    ) -> tuple[dict, str]:
        """Run _execute_code and also return the venv python path for reuse."""
        code_dir = self.workspace.path / "code"
        entry_script = self._find_legacy_entry_script(code_dir)
        if entry_script is None:
            return (
                {
                    "status": "skipped",
                    "reason": "No runnable entry script found (expected one of main.py/train.py/run.py)",
                    "stdout": "",
                    "stderr": "",
                },
                "",  # no python path — caller must call _setup_venv before using
            )

        venv_python = await self._setup_venv(code_dir)
        result = await self._execute_code(
            generated_files, blueprint_summary,
            _code_dir=code_dir, _main_py=entry_script, _venv_python=venv_python,
        )
        return result, venv_python

    async def _execute_code(
        self,
        generated_files: list[str],
        blueprint_summary: str,
        *,
        _code_dir: Path | None = None,
        _main_py: Path | None = None,
        _venv_python: str | None = None,
    ) -> dict:
        """Execute main.py --dry-run with up to 5 batch-fix cycles.

        Each cycle: run → collect all errors → fix ALL affected files in one
        LLM call → run again.  This is much more efficient than fixing one
        bug at a time.
        """
        code_dir = _code_dir or (self.workspace.path / "code")
        main_py = _main_py or self._find_legacy_entry_script(code_dir)

        if main_py is None or not main_py.exists():
            return {
                "status": "skipped",
                "reason": "No runnable entry script found (expected one of main.py/train.py/run.py)",
                "stdout": "",
                "stderr": "",
            }

        venv_python = _venv_python or await self._setup_venv(code_dir)

        max_fix_cycles = 5
        last_result: dict = {}
        fix_history: list[dict] = []  # Track previous fixes to avoid repeating

        for cycle in range(1, max_fix_cycles + 1):
            result = await self._run_main_py(code_dir, venv_python)
            last_result = result
            if result["returncode"] == 0:
                status = "success" if cycle == 1 else "fixed"
                return {"status": status, "attempts": cycle, **result}

            self.log(f"Code execution failed (attempt {cycle}): {result['stderr'][:200]}")

            if cycle >= max_fix_cycles:
                break

            # Batch fix: identify ALL affected files and fix them in one call
            stderr_text = result["stderr"]
            try:
                modified = await self._batch_fix_errors(
                    code_dir, stderr_text, blueprint_summary,
                    mode="dry-run",
                    previous_fixes=fix_history,
                )
                fix_history.append({"error_msg": stderr_text[:300], "cycle": cycle})
                if not modified:
                    self.log("Dry-run: no files modified by batch fix, stopping")
                    break
            except Exception as e:
                self.log(f"Batch fix error in cycle {cycle}: {e}")
                break

        return {"status": "failed", "attempts": cycle, **last_result}

    async def _batch_fix_errors(
        self,
        code_dir: Path,
        stderr: str,
        blueprint_summary: str,
        mode: str = "dry-run",
        previous_fixes: list[dict] | None = None,
        extra_context: str = "",
    ) -> list[str]:
        """Parse traceback, fix each affected file with a targeted LLM call.

        Surgical approach: for each file in the traceback, send ONLY that file
        + the error to the LLM → get a search-replace patch → apply.
        Uses 4-layer patch matching and syntax validation with rollback.

        Returns list of modified file paths.
        """
        import re as _re

        # 1. Parse traceback to find affected files with line numbers
        code_dir_str = str(code_dir.resolve()).replace("\\", "/")
        # Match: File "path", line N, in func
        tb_entries = _re.findall(
            r'File "([^"]+)",\s*line\s+(\d+)', stderr
        )

        # Deduplicate and filter to project files only (deepest frame first)
        affected: list[tuple[Path, int]] = []
        seen_files: set[str] = set()
        for fpath, lineno in reversed(tb_entries):
            f_norm = fpath.replace("\\", "/")
            resolved = Path(fpath).resolve()
            resolved_norm = str(resolved).replace("\\", "/")
            if code_dir_str not in resolved_norm:
                continue
            try:
                rel = str(resolved.relative_to(code_dir.resolve())).replace("\\", "/")
            except ValueError:
                continue
            if rel not in seen_files and resolved.exists():
                affected.append((resolved, int(lineno)))
                seen_files.add(rel)

        # If no project files found, default to main.py
        if not affected:
            main_py = code_dir / "main.py"
            if main_py.exists():
                affected = [(main_py, 0)]

        if not affected:
            return []

        # 2. Extract the final error message
        error_lines = stderr.strip().split("\n")
        error_msg = ""
        for line in reversed(error_lines):
            line = line.strip()
            if line and not line.startswith("File ") and not line.startswith("Traceback"):
                error_msg = line
                break

        # 3. Gather context: config files, requirements, imports
        context_files: list[str] = []
        for pattern in ("config/*.yaml", "config/*.yml", "config/*.json",
                        "*.yaml", "*.yml", "requirements.txt"):
            for cf in code_dir.glob(pattern):
                if cf.is_file():
                    try:
                        rel = str(cf.relative_to(code_dir)).replace("\\", "/")
                        ctx = cf.read_text(encoding="utf-8", errors="replace")[:1500]
                        context_files.append(f"--- {rel} ---\n{ctx}")
                    except OSError:
                        pass
        config_context = "\n\n".join(context_files) if context_files else "(no config files)"

        # Also list all project files for reference
        all_files = []
        for f in sorted(code_dir.rglob("*")):
            if f.is_file() and "__pycache__" not in str(f):
                all_files.append(str(f.relative_to(code_dir)).replace("\\", "/"))
        file_list = "\n".join(f"  {f}" for f in all_files)

        # 4. Fix each affected file with a targeted LLM call
        flag = "--quick-eval" if mode == "quick-eval" else "--dry-run"
        modified: list[str] = []
        code_gen_config = self.config.for_stage("code_gen")

        for target_file, error_line in affected:
            rel_path = str(target_file.relative_to(code_dir)).replace("\\", "/")
            try:
                content = target_file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            # Show context around the error line (±15 lines)
            lines = content.split("\n")
            if error_line > 0:
                start = max(0, error_line - 16)
                end = min(len(lines), error_line + 15)
                context_snippet = "\n".join(
                    f"{'>>>' if i+1 == error_line else '   '} {i+1:4d} | {l}"
                    for i, l in enumerate(lines[start:end], start=start)
                )
            else:
                context_snippet = content[:2000]

            # Build previous fix history to avoid repeating failed fixes
            fix_history = ""
            if previous_fixes:
                fix_history = (
                    "\n\nPrevious fix attempts that did NOT resolve the problem:\n"
                    + "\n".join(
                        (
                            f"  Round {i+1}: "
                            f"{fx.get('diagnosis', fx.get('error_msg', ''))[:200]}"
                            + (
                                f" | repeated={fx.get('repeat_count')}"
                                if fx.get("repeat_count", 1) > 1
                                else ""
                            )
                            + (
                                f" | files={fx.get('fixed_files', [])}"
                                if fx.get("fixed_files")
                                else ""
                            )
                        )
                        for i, fx in enumerate(previous_fixes)
                    )
                    + "\nDo NOT repeat the same fixes. Try a different approach.\n"
                )

            extra_context_text = (
                f"Additional execution context:\n{extra_context}\n\n"
                if extra_context.strip()
                else ""
            )
            fix_prompt = (
                f"`python main.py {flag}` failed.\n\n"
                f"Error: {error_msg}\n\n"
                f"Full traceback (last 40 lines):\n```\n"
                f"{chr(10).join(error_lines[-40:])}\n```\n\n"
                f"{extra_context_text}"
                f"File: {rel_path} (error around line {error_line}):\n```python\n"
                f"{context_snippet}\n```\n\n"
                f"Full file ({len(lines)} lines):\n```python\n{content[:4000]}\n```\n\n"
                f"== Config / Data Files (for reference) ==\n{config_context}\n\n"
                f"== Project Files ==\n{file_list}\n"
                f"{fix_history}\n"
                f"Output a JSON array of search-replace edits:\n"
                f'[{{"old": "exact text to find", "new": "replacement text"}}]\n\n'
                f"Rules:\n"
                f"- 'old' must be an EXACT substring of the file (including indentation)\n"
                f"- Multiple edits are fine — fix ALL issues in this file\n"
                f"- If config is YAML, use yaml.safe_load(), NOT json.load()\n"
                f"- Ensure imports match actual module structure\n"
                f"- Output ONLY valid JSON array, no markdown"
            )

            try:
                raw = await self._dispatcher.generate(
                    code_gen_config,
                    f"You are a Python debugging expert. Fix the bug in {rel_path} using precise search-replace edits.",
                    fix_prompt,
                )
                edits = self._parse_llm_json_payload(raw)
                if not isinstance(edits, list):
                    edits = [edits]

                # Save backup for syntax rollback
                backup_content = content
                applied = 0
                for edit in edits:
                    if not isinstance(edit, dict):
                        continue
                    old = edit.get("old", "")
                    new = edit.get("new", "")
                    if not old:
                        continue
                    content, matched, match_strategy = self._apply_search_replace_edit(
                        content,
                        old,
                        new,
                    )
                    if matched:
                        applied += 1
                        self.log(f"  Patch matched in {rel_path} via {match_strategy}")

                if applied > 0:
                    # Syntax validation + rollback (borrowed from Deep Pipeline DebugAgent)
                    target_file.write_text(content, encoding="utf-8")
                    if target_file.suffix == ".py" and not self._check_syntax(target_file):
                        self.log(f"  Patch introduced syntax error in {rel_path}, rolling back")
                        target_file.write_text(backup_content, encoding="utf-8")
                    else:
                        modified.append(rel_path)
                        self.log(f"  Fixed {rel_path}: {applied} edit(s) applied")
                else:
                    self.log(f"  No edits matched in {rel_path}")

            except json.JSONDecodeError:
                # Fallback: LLM might return the full fixed file
                if raw and len(raw) > 50:
                    target_file.write_text(raw, encoding="utf-8")
                    if target_file.suffix == ".py" and not self._check_syntax(target_file):
                        self.log(f"  Fallback rewrite has syntax error in {rel_path}, rolling back")
                        target_file.write_text(content, encoding="utf-8")
                    else:
                        modified.append(rel_path)
                        self.log(f"  Rewrote {rel_path} (fallback)")
            except Exception as e:
                self.log(f"  Fix failed for {rel_path}: {e}")

        if modified:
            self.log(f"Batch fix: modified {len(modified)} files")
        else:
            self.log("Batch fix: no files were modified")

        return modified

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
        import re as _re
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

