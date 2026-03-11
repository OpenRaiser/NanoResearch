"""Cluster execution: SLURM job management and subprocess helpers."""
from __future__ import annotations

import asyncio
import csv
import gzip
import json
import logging
import os
import platform
import re
import shlex
import shutil
import signal
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_IS_WINDOWS = platform.system() == "Windows"


def _kill_process_tree(pid: int) -> None:
    """Kill a process and all its descendants.

    On Windows the venv ``python.exe`` is a thin wrapper that spawns the real
    interpreter as a child.  A plain ``proc.kill()`` only terminates the
    wrapper, leaving the child alive.  ``taskkill /T /F`` kills the entire
    tree.  On Unix we recursively kill child processes via ``pgrep -P``.

    NOTE: We intentionally avoid ``os.killpg`` because child processes
    typically inherit the parent's process group — killing the group would
    terminate the pipeline itself.
    """
    try:
        if _IS_WINDOWS:
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
        else:
            # Recursively kill children first (depth-first), then the target.
            try:
                result = subprocess.run(
                    ["pgrep", "-P", str(pid)],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                for child in result.stdout.strip().split():
                    if child.strip():
                        _kill_process_tree(int(child))
            except Exception:  # noqa: BLE001
                pass
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
    except Exception:  # noqa: BLE001 — best effort
        pass


from nanoresearch.agents.repair_journal import (
    REPAIR_SNAPSHOT_JOURNAL_PATH,
)

POLL_INTERVAL = 30
MAX_WAIT_TIME = 7 * 24 * 3600

class _ClusterRunnerMixin:
    """Mixin — see module docstring."""

    async def _find_existing_job(self, code_dir: Path) -> tuple[str, str] | None:
        """Check if a previous SLURM job exists (from a crashed run).

        Returns (job_id, status) if found, None otherwise.
        """
        tracker = code_dir / "logs" / "active_job_id.txt"
        if not tracker.exists():
            return None

        job_id = tracker.read_text().strip()
        if not job_id or not job_id.isdigit():
            return None

        status = await self._get_job_status(job_id)
        if status in ("RUNNING", "PENDING", "COMPLETED"):
            return (job_id, status)

        return None  # FAILED/CANCELLED/UNKNOWN — need fresh submit

    async def _submit_job(self, slurm_script: str) -> str:
        """Submit a SLURM batch job and return the job ID."""
        if not Path(slurm_script).exists():
            raise RuntimeError(f"SLURM script not found: {slurm_script}")

        result = await self._run_shell(f"sbatch {slurm_script}")
        stdout = result.get("stdout", "")
        stderr = result.get("stderr", "")

        # Parse job ID from "Submitted batch job 12345"
        match = re.search(r"Submitted batch job (\d+)", stdout)
        if not match:
            raise RuntimeError(
                f"Failed to submit SLURM job. stdout: {stdout}, stderr: {stderr}"
            )

        job_id = match.group(1)

        # Save job ID for resume tracking
        tracker_path = Path(slurm_script).parent / "logs" / "active_job_id.txt"
        tracker_path.parent.mkdir(parents=True, exist_ok=True)
        tracker_path.write_text(job_id)

        return job_id

    async def _monitor_job(self, job_id: str, code_dir: Path) -> str:
        """Poll SLURM until job completes. Returns final status."""
        start_time = time.time()
        last_log_lines = 0

        while time.time() - start_time < MAX_WAIT_TIME:
            status = await self._get_job_status(job_id)

            # Stream training log if available
            log_files = list(code_dir.glob("logs/slurm_*.out"))
            if log_files:
                try:
                    content = log_files[-1].read_text(errors="replace")
                    lines = content.strip().split("\n")
                    if len(lines) > last_log_lines:
                        new_lines = lines[last_log_lines:]
                        for line in new_lines[-5:]:  # show last 5 new lines
                            self.log(f"[TRAIN] {line.strip()}")
                        last_log_lines = len(lines)
                except Exception:
                    pass

            if status in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "PREEMPTED", "OUT_OF_MEMORY"):
                return status

            if status == "PENDING":
                elapsed = int(time.time() - start_time)
                self.log(f"Job {job_id} pending... ({elapsed}s elapsed)")
            elif status == "RUNNING":
                elapsed = int(time.time() - start_time)
                self.log(f"Job {job_id} running... ({elapsed}s elapsed)")

            await asyncio.sleep(POLL_INTERVAL)

        # Timeout — cancel the job
        self.log(f"Job {job_id} exceeded max wait time ({MAX_WAIT_TIME}s), cancelling")
        await self._run_shell(f"scancel {job_id}")
        return "TIMEOUT"

    async def _get_job_status(self, job_id: str) -> str:
        """Query SLURM for job status."""
        result = await self._run_shell(
            f"squeue -j {job_id} -h -o '%T' 2>/dev/null || "
            f"sacct -j {job_id} -n -o State -X 2>/dev/null"
        )
        stdout = result.get("stdout", "").strip()

        if not stdout:
            # Job not in queue and not in accounting — might have just finished
            result2 = await self._run_shell(
                f"sacct -j {job_id} -n -o State -X"
            )
            stdout = result2.get("stdout", "").strip()

        # Parse status
        status = stdout.split("\n")[0].strip().upper() if stdout else "UNKNOWN"
        # Clean up status (sacct sometimes adds '+')
        status = status.rstrip("+").strip()

        return status


    async def _run_shell(self, cmd: str, timeout: int = 60) -> dict:
        """Run a shell command asynchronously with proxy environment."""
        env = self._build_proxy_env()
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            _kill_process_tree(proc.pid)
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            return {"returncode": -1, "stdout": "", "stderr": "Command timed out"}
        return {
            "returncode": proc.returncode or 0,
            "stdout": stdout.decode(errors="replace"),
            "stderr": stderr.decode(errors="replace"),
        }

    def _build_proxy_env(self) -> dict[str, str]:
        env = {**os.environ}
        proxy_url = env.get("https_proxy") or env.get("HTTPS_PROXY", "")
        if not proxy_url:
            import re as _re

            bashrc = Path.home() / ".bashrc"
            if bashrc.exists():
                content = bashrc.read_text(errors="replace")
                match = _re.search(r"https_proxy=(http://[^\s;'\"]+)", content)
                if match:
                    proxy_url = match.group(1)
        if proxy_url:
            env.update(
                {
                    "http_proxy": proxy_url,
                    "https_proxy": proxy_url,
                    "HTTP_PROXY": proxy_url,
                    "HTTPS_PROXY": proxy_url,
                }
            )
        env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
        return env

    async def _run_subprocess(
        self,
        command: list[str],
        *,
        cwd: Path | None = None,
        timeout: int = 60,
    ) -> dict[str, Any]:
        env = self._build_proxy_env()
        try:
            proc = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(cwd) if cwd is not None else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
        except PermissionError:
            return await asyncio.to_thread(
                self._run_subprocess_sync,
                command,
                cwd=cwd,
                timeout=timeout,
                env=env,
            )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            _kill_process_tree(proc.pid)
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            return {"returncode": -1, "stdout": "", "stderr": "Command timed out"}
        return {
            "returncode": proc.returncode or 0,
            "stdout": stdout.decode(errors="replace"),
            "stderr": stderr.decode(errors="replace"),
        }

    @staticmethod
    def _run_subprocess_sync(
        command: list[str],
        *,
        cwd: Path | None = None,
        timeout: int = 60,
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        proc = subprocess.Popen(
            command,
            cwd=str(cwd) if cwd is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
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
                proc.communicate(timeout=5)  # reap zombie
            except (subprocess.TimeoutExpired, OSError):
                pass
            return {"returncode": -1, "stdout": "", "stderr": "Command timed out"}
        return {
            "returncode": proc.returncode or 0,
            "stdout": stdout.decode(errors="replace"),
            "stderr": stderr.decode(errors="replace"),
        }

    async def close(self) -> None:
        pass
