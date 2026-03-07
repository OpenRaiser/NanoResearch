"""Execution agent — submits SLURM jobs, monitors progress, debugs failures, collects results."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from nanoresearch.agents.base import BaseResearchAgent
from nanoresearch.agents.debug import DebugAgent, MAX_DEBUG_ROUNDS
from nanoresearch.schemas.manifest import PipelineStage

logger = logging.getLogger(__name__)

# Poll interval and max wait time for SLURM jobs
POLL_INTERVAL = 30  # seconds
MAX_WAIT_TIME = 7 * 24 * 3600  # 7 days — real training can run for days


class ExecutionAgent(BaseResearchAgent):
    """Submits SLURM training jobs, monitors them, debugs failures, and collects results."""

    stage = PipelineStage.EXPERIMENT  # reuse stage config

    async def run(self, **inputs: Any) -> dict[str, Any]:
        coding_output: dict = inputs.get("coding_output", {})

        code_dir = Path(coding_output.get("code_dir", ""))
        slurm_script = coding_output.get("slurm_script", "")

        if not code_dir.exists():
            raise RuntimeError(f"Code directory not found: {code_dir}")

        self.log(f"Starting execution in: {code_dir}")

        # Create logs directory
        (code_dir / "logs").mkdir(exist_ok=True)
        (code_dir / "results").mkdir(exist_ok=True)

        # Pre-flight: fix common SLURM issues before first submission
        debug_agent = DebugAgent(self.workspace, self.config)
        preflight_fixed = debug_agent._fix_common_slurm_issues(code_dir)
        if preflight_fixed:
            self.log("Pre-flight: fixed common SLURM script issues")

        # Pre-flight: local syntax/import check before wasting SLURM queue time
        local_ok, local_err = await self._local_preflight(code_dir)
        if not local_ok:
            self.log(f"Pre-flight import check failed, fixing before submission")
            # Run a mini debug loop locally (no SLURM submission)
            for pre_round in range(MAX_DEBUG_ROUNDS):
                debug_result = await debug_agent.run(
                    code_dir=str(code_dir),
                    stdout_log="",
                    stderr_log=local_err,
                    job_status="IMPORT_ERROR",
                    debug_round=pre_round + 1,
                    previous_fixes=[],
                )
                if not debug_result.get("needs_resubmit", False):
                    break
                local_ok, local_err = await self._local_preflight(code_dir)
                if local_ok:
                    self.log(f"Pre-flight fixed after {pre_round + 1} round(s)")
                    break

        # Debug loop: submit → monitor → if failed, debug & retry
        previous_fixes: list[dict] = []
        final_result = None

        for debug_round in range(MAX_DEBUG_ROUNDS + 1):
            # On first round, check for existing job from a previous run (resume)
            existing = await self._find_existing_job(code_dir) if debug_round == 0 else None
            if existing:
                job_id, existing_status = existing
                self.log(f"Found existing SLURM job {job_id} (status: {existing_status})")
                if existing_status == "COMPLETED":
                    final_status = "COMPLETED"
                else:  # RUNNING or PENDING
                    final_status = await self._monitor_job(job_id, code_dir)
                    self.log(f"Existing job {job_id} finished: {final_status}")
            else:
                # Submit new SLURM job
                job_id = await self._submit_job(slurm_script)
                self.log(f"Submitted SLURM job: {job_id}")
                # Monitor job until completion
                final_status = await self._monitor_job(job_id, code_dir)
                self.log(f"Job {job_id} finished with status: {final_status}")

            # Collect results
            results = await self._collect_results(code_dir, job_id, final_status)
            self.log(f"Collected results: {list(results.keys())}")

            final_result = {
                "job_id": job_id,
                "final_status": final_status,
                "code_dir": str(code_dir),
                "debug_rounds": debug_round,
                **results,
            }

            # If job succeeded or we've exhausted debug rounds, stop
            if final_status == "COMPLETED":
                # Verify training actually produced results (not just exit code 0)
                has_metrics = bool(
                    results.get("metrics")
                    or results.get("parsed_metrics")
                    or results.get("training_log")
                    or results.get("training_log_csv")
                    or results.get("checkpoints")
                )
                if has_metrics:
                    self.log(f"Job completed successfully after {debug_round} debug round(s)")
                    break
                else:
                    # Check stdout/stderr for crash indicators
                    combined_log = results.get("stdout_log", "") + results.get("stderr_log", "")
                    crash_indicators = [
                        "RuntimeError", "Error(s) in loading", "Traceback",
                        "CUDA out of memory", "OOM", "Killed",
                        "Exception", "FileNotFoundError", "ModuleNotFoundError",
                    ]
                    has_crash = any(ind in combined_log for ind in crash_indicators)
                    if has_crash:
                        self.log(
                            "Job exited with code 0 but logs contain errors and no metrics produced. "
                            "Treating as FAILED."
                        )
                        final_status = "FAILED"
                        final_result["final_status"] = "FAILED"
                        # Fall through to debug loop
                    else:
                        self.log(f"Job completed after {debug_round} debug round(s) (no metrics found)")
                        break

            if debug_round >= MAX_DEBUG_ROUNDS:
                self.log(f"Max debug rounds ({MAX_DEBUG_ROUNDS}) reached, giving up")
                break

            # Job failed — enter debug loop
            self.log(f"Job failed, entering debug round {debug_round + 1}/{MAX_DEBUG_ROUNDS}")

            try:
                debug_result = await debug_agent.run(
                    code_dir=str(code_dir),
                    stdout_log=results.get("stdout_log", ""),
                    stderr_log=results.get("stderr_log", ""),
                    job_status=final_status,
                    debug_round=debug_round + 1,
                    previous_fixes=previous_fixes,
                )

                if not debug_result.get("needs_resubmit", False):
                    self.log("Debug agent determined no fix is possible, stopping")
                    break

                previous_fixes.append({
                    "diagnosis": debug_result.get("diagnosis", ""),
                    "patches": debug_result.get("patches", []),
                    "fixed_files": debug_result.get("fixed_files", []),
                })

                self.log(f"Debug round {debug_round + 1}: fixed {debug_result.get('fixed_files', [])}, resubmitting...")

            except Exception as e:
                self.log(f"Debug agent failed: {e}")
                break

        await debug_agent.close()

        self.workspace.write_json("plans/execution_output.json", final_result)
        return final_result

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

            if status in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"):
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

    async def _collect_results(
        self, code_dir: Path, job_id: str, status: str
    ) -> dict:
        """Collect training results from output files."""
        results: dict[str, Any] = {
            "metrics": {},
            "training_log": [],
            "stdout_log": "",
            "stderr_log": "",
        }

        # Read SLURM stdout/stderr (use the latest log files for this job)
        for log_file in sorted(code_dir.glob("logs/slurm_*.out")):
            if job_id in log_file.name:
                results["stdout_log"] = log_file.read_text(errors="replace")[-10000:]
                break
        else:
            # Fallback: read any .out file
            for log_file in code_dir.glob("logs/slurm_*.out"):
                results["stdout_log"] = log_file.read_text(errors="replace")[-10000:]

        for log_file in sorted(code_dir.glob("logs/slurm_*.err")):
            if job_id in log_file.name:
                results["stderr_log"] = log_file.read_text(errors="replace")[-5000:]
                break
        else:
            for log_file in code_dir.glob("logs/slurm_*.err"):
                results["stderr_log"] = log_file.read_text(errors="replace")[-5000:]

        # Read metrics.json if produced
        metrics_path = code_dir / "results" / "metrics.json"
        if metrics_path.exists():
            try:
                results["metrics"] = json.loads(metrics_path.read_text())
            except json.JSONDecodeError:
                results["metrics"] = {"raw": metrics_path.read_text()[:5000]}

        # Read training_log.csv if produced
        log_csv = code_dir / "results" / "training_log.csv"
        if log_csv.exists():
            results["training_log_csv"] = log_csv.read_text(errors="replace")[:10000]

        # Look for any results files
        for results_file in (code_dir / "results").glob("*"):
            if results_file.is_file() and results_file.name not in ("metrics.json", "training_log.csv"):
                try:
                    content = results_file.read_text(errors="replace")[:5000]
                    results[f"result_file_{results_file.name}"] = content
                except Exception:
                    pass

        # Check for checkpoints
        checkpoints = list((code_dir / "checkpoints").glob("*.pt")) if (code_dir / "checkpoints").exists() else []
        results["checkpoints"] = [str(p) for p in checkpoints]

        # Parse metrics from stdout if metrics.json missing
        if not results["metrics"] and results["stdout_log"]:
            results["parsed_metrics"] = self._parse_metrics_from_log(results["stdout_log"])

        return results

    def _parse_metrics_from_log(self, log_text: str) -> dict:
        """Try to extract metrics from training log output."""
        metrics: dict[str, Any] = {}
        lines = log_text.split("\n")

        # Common patterns in training logs
        patterns = [
            # "Epoch 10: loss=0.123, accuracy=0.95"
            r"[Ee]poch\s+(\d+).*?loss[=:\s]+([0-9.e-]+)",
            # "Test accuracy: 0.95"
            r"[Tt]est\s+(?:accuracy|acc)[=:\s]+([0-9.e-]+)",
            # "Best metric: 0.95"
            r"[Bb]est\s+\w+[=:\s]+([0-9.e-]+)",
            # "AUC: 0.95" / "F1: 0.85"
            r"(AUC|F1|RMSE|MAE|accuracy|precision|recall)[=:\s]+([0-9.e-]+)",
        ]

        epochs = []
        for line in lines:
            for pattern in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    groups = match.groups()
                    if len(groups) == 2:
                        metrics[groups[0]] = groups[1]

            # Track epoch losses
            epoch_match = re.search(
                r"[Ee]poch\s+(\d+).*?loss[=:\s]+([0-9.e-]+)", line
            )
            if epoch_match:
                epochs.append({
                    "epoch": int(epoch_match.group(1)),
                    "loss": float(epoch_match.group(2)),
                })

        if epochs:
            metrics["epoch_losses"] = epochs
            metrics["final_loss"] = epochs[-1]["loss"]

        return metrics

    async def _local_preflight(self, code_dir: Path) -> tuple[bool, str]:
        """Run local checks before submitting to SLURM.

        Tests:
        1. Python syntax check (py_compile) on all .py files
        2. Import check — try importing the entry point module
        3. Verify all cross-file imports resolve

        Returns (ok, error_message).
        """
        errors = []

        # 1. Syntax check all .py files
        for py_file in sorted(code_dir.glob("*.py")):
            result = await self._run_shell(
                f"python -c \"import py_compile; py_compile.compile('{py_file}', doraise=True)\"",
                timeout=10,
            )
            if result["returncode"] != 0:
                errors.append(f"Syntax error in {py_file.name}:\n{result['stderr']}")

        if errors:
            return False, "\n".join(errors)

        # 2. Try importing the main modules to catch import errors
        # (run in the code directory so local imports work)
        py_modules = [f.stem for f in code_dir.glob("*.py")]
        for module in py_modules:
            result = await self._run_shell(
                f"cd {code_dir} && python -c \"import {module}\" 2>&1",
                timeout=30,
            )
            if result["returncode"] != 0:
                err_text = result["stdout"] + result["stderr"]
                # Ignore errors from missing heavy dependencies (torch, etc.)
                # — those will be installed on the cluster node
                if any(pkg in err_text for pkg in [
                    "No module named 'torch'",
                    "No module named 'torchvision'",
                    "No module named 'torchaudio'",
                    "No module named 'timm'",
                    "No module named 'transformers'",
                    "No module named 'torch_geometric'",
                    "No module named 'torch_scatter'",
                    "No module named 'torch_sparse'",
                    "No module named 'esm'",
                    "No module named 'dgl'",
                    "No module named 'accelerate'",
                    "No module named 'datasets'",
                    "No module named 'einops'",
                    "No module named 'wandb'",
                    "No module named 'scipy'",
                    "No module named 'sklearn'",
                    "No module named 'cv2'",
                    "No module named 'PIL'",
                    "CUDA",
                ]):
                    continue
                errors.append(f"Import error in {module}.py:\n{err_text}")

        if errors:
            return False, "\n".join(errors)

        return True, ""

    async def _run_shell(self, cmd: str, timeout: int = 60) -> dict:
        """Run a shell command asynchronously with proxy environment."""
        env = {**__import__('os').environ}
        proxy_url = env.get("https_proxy") or env.get("HTTPS_PROXY", "")
        if not proxy_url:
            import re as _re
            bashrc = Path.home() / ".bashrc"
            if bashrc.exists():
                content = bashrc.read_text(errors="replace")
                m = _re.search(r"https_proxy=(http://[^\s;'\"]+)", content)
                if m:
                    proxy_url = m.group(1)
        if proxy_url:
            env.update({
                "http_proxy": proxy_url, "https_proxy": proxy_url,
                "HTTP_PROXY": proxy_url, "HTTPS_PROXY": proxy_url,
            })
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return {"returncode": -1, "stdout": "", "stderr": "Command timed out"}
        return {
            "returncode": proc.returncode or 0,
            "stdout": stdout.decode(errors="replace"),
            "stderr": stderr.decode(errors="replace"),
        }

    async def close(self) -> None:
        pass
