"""Cluster executor — run experiments on a SLURM cluster.

Two modes:
  - LOCAL mode (local=true): run sbatch/squeue directly on the current machine.
    Use when NanoResearch is running ON the cluster login/bastion node.
  - REMOTE mode (local=false): run commands via SSH/SCP through a bastion.
    Use when NanoResearch is running on a different machine (e.g., laptop).

Usage:
    executor = ClusterExecutor(cluster_config, logger_fn)
    code_path = await executor.prepare_code(local_code_dir, session_id)
    await executor.setup_env(code_path)
    job_id = await executor.submit_job(code_path, "python main.py --quick-eval")
    status = await executor.wait_for_job(job_id)
    if status["state"] == "COMPLETED":
        metrics = executor.read_local_metrics(code_path)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

DEFAULT_POLL_INTERVAL = 30
DEFAULT_MAX_WAIT = 14400  # 4 hours
CMD_TIMEOUT = 120
SCP_TIMEOUT = 600
ENV_SETUP_TIMEOUT = 900  # 15 min for pip install


class ClusterExecutor:
    """Execute experiments on a SLURM cluster (local or remote)."""

    def __init__(self, config: dict, log_fn: Callable[[str], None] | None = None):
        self.local_mode = config.get("local", False)
        self.host = config.get("host", "")
        self.user = config.get("user", "")
        self.bastion = config.get("bastion")
        self.partition = config.get("partition", "raise")
        self.gpus = config.get("gpus", 4)
        self.quota_type = config.get("quota_type", "reserved")
        self.conda_env = config.get("conda_env", "nano_exp")
        self.python_version = config.get("python_version", "3.10")
        self.container = config.get("container")
        self.base_path = config.get("code_path", "")
        self.time_limit = config.get("time_limit", "24:00:00")
        self.poll_interval = config.get("poll_interval", DEFAULT_POLL_INTERVAL)
        self.max_wait = config.get("max_wait", DEFAULT_MAX_WAIT)
        self._log_fn = log_fn or (lambda msg: logger.info(msg))

    def log(self, msg: str) -> None:
        self._log_fn(f"[Cluster] {msg}")

    # ------------------------------------------------------------------
    # Shell execution
    # ------------------------------------------------------------------

    async def _run_cmd(self, cmd: str, timeout: int = CMD_TIMEOUT) -> dict:
        """Run a shell command — locally or via SSH depending on mode."""
        if self.local_mode:
            return await self._run_local_shell(cmd, timeout)
        else:
            return await self._run_ssh(cmd, timeout)

    async def _run_local_shell(self, cmd: str, timeout: int = CMD_TIMEOUT) -> dict:
        """Run a command locally via bash."""
        self.log(f"$ {cmd[:120]}...")
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    ["bash", "-c", cmd],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                ),
            )
            return {
                "returncode": result.returncode,
                "stdout": result.stdout[:10000],
                "stderr": result.stderr[:10000],
            }
        except subprocess.TimeoutExpired:
            return {"returncode": -1, "stdout": "", "stderr": f"Timeout after {timeout}s"}
        except Exception as e:
            return {"returncode": -1, "stdout": "", "stderr": str(e)}

    async def _run_ssh(self, cmd: str, timeout: int = CMD_TIMEOUT) -> dict:
        """Run a command on the remote host via SSH."""
        ssh_cmd = ["ssh"]
        if self.bastion:
            ssh_cmd.extend(["-J", self.bastion])
        ssh_cmd.extend([
            "-o", "StrictHostKeyChecking=no",
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=15",
            f"{self.user}@{self.host}",
            cmd,
        ])
        self.log(f"ssh$ {cmd[:120]}...")
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    ssh_cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                ),
            )
            return {
                "returncode": result.returncode,
                "stdout": result.stdout[:10000],
                "stderr": result.stderr[:10000],
            }
        except subprocess.TimeoutExpired:
            return {"returncode": -1, "stdout": "", "stderr": f"Timeout after {timeout}s"}
        except Exception as e:
            return {"returncode": -1, "stdout": "", "stderr": str(e)}

    async def _scp_upload(self, local: str, remote: str, timeout: int = SCP_TIMEOUT) -> dict:
        """SCP upload (remote mode only)."""
        cmd = ["scp", "-r"]
        if self.bastion:
            cmd.extend(["-o", f"ProxyJump={self.bastion}"])
        cmd.extend(["-o", "StrictHostKeyChecking=no", local, f"{self.user}@{self.host}:{remote}"])
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(cmd, capture_output=True, text=True, timeout=timeout),
            )
            return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
        except Exception as e:
            return {"returncode": -1, "stdout": "", "stderr": str(e)}

    async def _scp_download(self, remote: str, local: str, timeout: int = SCP_TIMEOUT) -> dict:
        """SCP download (remote mode only)."""
        cmd = ["scp", "-r"]
        if self.bastion:
            cmd.extend(["-o", f"ProxyJump={self.bastion}"])
        cmd.extend(["-o", "StrictHostKeyChecking=no", f"{self.user}@{self.host}:{remote}", local])
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(cmd, capture_output=True, text=True, timeout=timeout),
            )
            return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
        except Exception as e:
            return {"returncode": -1, "stdout": "", "stderr": str(e)}

    # ------------------------------------------------------------------
    # High-level operations
    # ------------------------------------------------------------------

    async def check_connectivity(self) -> bool:
        """Test that we can reach the cluster and sbatch is available."""
        if self.local_mode:
            result = await self._run_local_shell("which sbatch && echo OK", timeout=10)
        else:
            result = await self._run_ssh("which sbatch && echo OK", timeout=30)
        ok = result["returncode"] == 0 and "OK" in result["stdout"]
        if ok:
            self.log("Cluster connectivity OK (sbatch found)")
        else:
            self.log(f"Cluster check FAILED: {result['stderr'][:200]}")
        return ok

    async def prepare_code(self, local_code_dir: Path, session_id: str) -> str:
        """Prepare code on the cluster. Returns the code path on the cluster.

        LOCAL mode: code is already on disk — just return the path (or copy
        to base_path if configured).
        REMOTE mode: SCP upload to remote base_path.
        """
        if self.local_mode:
            if self.base_path:
                # Copy code to the designated cluster path
                dest = Path(self.base_path) / session_id / "code"
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(local_code_dir, dest)
                # Ensure logs/results dirs
                (dest / "logs").mkdir(exist_ok=True)
                (dest / "results").mkdir(exist_ok=True)
                self.log(f"Code copied to {dest}")
                return str(dest)
            else:
                # Use code in-place
                (local_code_dir / "logs").mkdir(exist_ok=True)
                (local_code_dir / "results").mkdir(exist_ok=True)
                return str(local_code_dir)
        else:
            # Remote mode: SCP upload
            remote_dir = f"{self.base_path}/{session_id}"
            await self._run_ssh(f"mkdir -p {remote_dir}")
            result = await self._scp_upload(str(local_code_dir), f"{remote_dir}/code")
            if result["returncode"] != 0:
                raise RuntimeError(f"SCP upload failed: {result['stderr']}")
            await self._run_ssh(f"mkdir -p {remote_dir}/code/logs {remote_dir}/code/results")
            self.log(f"Code uploaded to {remote_dir}/code")
            return f"{remote_dir}/code"

    async def reupload_code(self, local_code_dir: Path, cluster_code_path: str) -> None:
        """Re-sync code after LLM modifications."""
        if self.local_mode:
            if str(local_code_dir) != cluster_code_path:
                # Different path — re-copy
                dest = Path(cluster_code_path)
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(local_code_dir, dest)
                (dest / "logs").mkdir(exist_ok=True)
                (dest / "results").mkdir(exist_ok=True)
            # else: same path, already in-place
        else:
            # Remote: re-upload
            parent = str(Path(cluster_code_path).parent)
            result = await self._scp_upload(str(local_code_dir), f"{parent}/code")
            if result["returncode"] != 0:
                self.log(f"Re-upload warning: {result['stderr'][:200]}")

    async def setup_env(self, cluster_code_path: str) -> dict:
        """Create conda env (if needed) and install requirements.txt.

        Returns {"ok": bool, "output": str}.
        """
        self.log(f"Setting up conda env '{self.conda_env}'...")

        # Detect conda
        detect = (
            "CONDA_SH=$HOME/anaconda3/etc/profile.d/conda.sh; "
            "[ ! -f $CONDA_SH ] && CONDA_SH=$HOME/miniconda3/etc/profile.d/conda.sh; "
            "[ ! -f $CONDA_SH ] && CONDA_SH=$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh; "
            "echo $CONDA_SH"
        )
        detect_result = await self._run_cmd(detect, timeout=15)
        conda_sh = detect_result["stdout"].strip()
        if not conda_sh or "No such" in conda_sh:
            conda_sh = "~/anaconda3/etc/profile.d/conda.sh"
        self._conda_sh = conda_sh
        self.log(f"Using conda: {conda_sh}")

        # Create env if it doesn't exist
        check_env = (
            f"source {conda_sh} 2>/dev/null && "
            f"conda env list | grep -w {self.conda_env} && echo ENV_EXISTS || echo ENV_MISSING"
        )
        check_result = await self._run_cmd(check_env, timeout=30)

        if "ENV_MISSING" in check_result["stdout"]:
            self.log(f"Creating conda env '{self.conda_env}' (python={self.python_version})...")
            create_cmd = (
                f"source {conda_sh} && "
                f"conda create -n {self.conda_env} python={self.python_version} -y 2>&1 | tail -10"
            )
            create_result = await self._run_cmd(create_cmd, timeout=300)
            if create_result["returncode"] != 0:
                self.log(f"Conda create failed: {create_result['stderr'][:300]}")
                return {"ok": False, "output": create_result["stderr"]}
            self.log("Conda env created")
        else:
            self.log(f"Conda env '{self.conda_env}' already exists")

        # Check if requirements.txt exists
        req_path = f"{cluster_code_path}/requirements.txt"
        check_req = f"test -f {req_path} && echo EXISTS || echo MISSING"
        req_result = await self._run_cmd(check_req, timeout=10)
        if "MISSING" in req_result["stdout"]:
            self.log("No requirements.txt, skipping pip install")
            return {"ok": True, "output": "No requirements.txt"}

        # Enable proxy if available (PJLab-specific), then pip install
        install_cmd = (
            f"source {conda_sh} && "
            f"conda activate {self.conda_env} && "
            f"type proxy_on &>/dev/null && proxy_on; "
            f"pip install -r {req_path} 2>&1 | tail -40"
        )
        self.log("Installing requirements (this may take a while)...")
        result = await self._run_cmd(install_cmd, timeout=ENV_SETUP_TIMEOUT)
        ok = result["returncode"] == 0
        output = result["stdout"] + "\n" + result["stderr"]

        if ok:
            self.log("Requirements installed successfully")
        else:
            self.log(f"pip install failed (rc={result['returncode']})")
            # Try to show the actual error
            self.log(output[-500:])

        return {"ok": ok, "output": output}

    def _generate_sbatch_script(self, cluster_code_path: str, script_cmd: str) -> str:
        """Generate sbatch script content."""
        conda_sh = getattr(self, "_conda_sh", "~/anaconda3/etc/profile.d/conda.sh")

        if self.container:
            run_cmd = (
                f"apptainer exec --nv -B /mnt:/mnt {self.container} "
                f"bash -c 'source {conda_sh} && conda activate {self.conda_env} && "
                f"cd {cluster_code_path} && {script_cmd}'"
            )
        else:
            run_cmd = (
                f"source {conda_sh} && "
                f"conda activate {self.conda_env} && "
                f"cd {cluster_code_path} && "
                f"{script_cmd}"
            )

        cpus = max(self.gpus * 8, 4)
        return f"""#!/bin/bash
#SBATCH --partition={self.partition}
#SBATCH --gres=gpu:{self.gpus}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --quotatype={self.quota_type}
#SBATCH --job-name=nano_exp
#SBATCH --output={cluster_code_path}/logs/%j.log
#SBATCH --error={cluster_code_path}/logs/%j.err
#SBATCH --time={self.time_limit}

echo "=== Job $SLURM_JOB_ID on $SLURM_NODELIST | {self.gpus} GPUs | $(date) ==="
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo "Working dir: {cluster_code_path}"

{run_cmd}

EXIT_CODE=$?
echo "=== Done: exit $EXIT_CODE at $(date) ==="
exit $EXIT_CODE
"""

    async def submit_job(self, cluster_code_path: str, script_cmd: str) -> str:
        """Generate sbatch script, write it, submit. Returns job ID."""
        sbatch_content = self._generate_sbatch_script(cluster_code_path, script_cmd)
        sbatch_path = f"{cluster_code_path}/job.sh"

        # Write sbatch script
        if self.local_mode:
            Path(sbatch_path).write_text(sbatch_content, encoding="utf-8")
            Path(sbatch_path).chmod(0o755)
        else:
            write_cmd = f"cat > {sbatch_path} << 'NANO_SBATCH_EOF'\n{sbatch_content}\nNANO_SBATCH_EOF"
            await self._run_cmd(write_cmd, timeout=15)
            await self._run_cmd(f"chmod +x {sbatch_path}", timeout=5)

        # Submit
        self.log(f"Submitting: sbatch {sbatch_path}")
        result = await self._run_cmd(f"sbatch {sbatch_path}", timeout=30)
        if result["returncode"] != 0:
            raise RuntimeError(
                f"sbatch failed (rc={result['returncode']}): {result['stderr']}"
            )

        # Parse job ID from "Submitted batch job 12345"
        match = re.search(r"(\d+)", result["stdout"])
        if not match:
            raise RuntimeError(f"Could not parse job ID from: {result['stdout']}")

        job_id = match.group(1)
        self.log(f"Job submitted: {job_id}")
        return job_id

    async def wait_for_job(self, job_id: str) -> dict:
        """Poll squeue until job completes."""
        self.log(f"Waiting for job {job_id} (poll={self.poll_interval}s, max={self.max_wait}s)...")
        start = time.time()
        last_status = ""

        while time.time() - start < self.max_wait:
            result = await self._run_cmd(
                f"squeue -j {job_id} -h -o '%T' 2>/dev/null",
                timeout=15,
            )
            status = result["stdout"].strip().strip("'\"")

            if not status:
                # Not in queue anymore — finished
                break

            if status != last_status:
                elapsed = int(time.time() - start)
                self.log(f"Job {job_id}: {status} ({elapsed}s)")
                last_status = status

            if status in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL",
                          "OUT_OF_MEMORY", "PREEMPTED"):
                break

            await asyncio.sleep(self.poll_interval)
        else:
            self.log(f"Job {job_id}: wait timed out after {self.max_wait}s")
            return {
                "job_id": job_id,
                "state": "WAIT_TIMEOUT",
                "exit_code": "?",
                "elapsed": int(time.time() - start),
            }

        # Get final status from sacct
        sacct_result = await self._run_cmd(
            f"sacct -j {job_id} --format=JobID,State,ExitCode,Elapsed -P -n 2>/dev/null | head -5",
            timeout=15,
        )

        state = "UNKNOWN"
        exit_code = "?"
        elapsed_str = "?"
        for line in sacct_result["stdout"].strip().split("\n"):
            parts = line.split("|")
            if len(parts) >= 3 and parts[0].strip() == job_id:
                state = parts[1].strip()
                exit_code = parts[2].strip()
                if len(parts) >= 4:
                    elapsed_str = parts[3].strip()
                break

        if state == "UNKNOWN":
            # Fallback: if sacct not available, assume completed if not in squeue
            state = "COMPLETED"

        total = int(time.time() - start)
        self.log(f"Job {job_id}: {state} (exit={exit_code}, slurm_elapsed={elapsed_str}, wait={total}s)")
        return {
            "job_id": job_id,
            "state": state,
            "exit_code": exit_code,
            "elapsed_slurm": elapsed_str,
            "elapsed_wait": total,
        }

    async def get_job_log(self, cluster_code_path: str, job_id: str, tail: int = 300) -> str:
        """Read job stdout + stderr logs."""
        cmd = (
            f"echo '=== STDOUT ===' && tail -{tail} {cluster_code_path}/logs/{job_id}.log 2>/dev/null; "
            f"echo '\\n=== STDERR ===' && tail -{tail} {cluster_code_path}/logs/{job_id}.err 2>/dev/null"
        )
        result = await self._run_cmd(cmd, timeout=30)
        log_text = result["stdout"]

        # If no job-specific log found, try to find any recent log
        if not log_text.strip() or log_text.strip() in ("=== STDOUT ===\n\n=== STDERR ===",):
            fallback = (
                f"ls -t {cluster_code_path}/logs/*.log 2>/dev/null | head -1 | "
                f"xargs -I{{}} tail -{tail} {{}}"
            )
            fb_result = await self._run_cmd(fallback, timeout=15)
            if fb_result["stdout"].strip():
                log_text = fb_result["stdout"]

        return log_text

    async def download_results(self, cluster_code_path: str, local_workspace: Path) -> bool:
        """Copy results from cluster to local workspace.

        LOCAL mode: just copy (or it's already in-place).
        REMOTE mode: SCP download.
        """
        if self.local_mode:
            src = Path(cluster_code_path) / "results" / "metrics.json"
            dst = local_workspace / "code" / "results"
            dst.mkdir(parents=True, exist_ok=True)
            if src.exists():
                shutil.copy2(src, dst / "metrics.json")
                self.log("Results copied locally")
                return True
            else:
                self.log("metrics.json not found on cluster")
                return False
        else:
            remote = f"{cluster_code_path}/results/metrics.json"
            local = str(local_workspace / "code" / "results" / "metrics.json")
            (local_workspace / "code" / "results").mkdir(parents=True, exist_ok=True)
            result = await self._scp_download(remote, local)
            return result["returncode"] == 0

    async def cancel_job(self, job_id: str) -> None:
        """Cancel a running SLURM job."""
        await self._run_cmd(f"scancel {job_id}", timeout=15)
        self.log(f"Job {job_id} cancelled")

    async def check_resources(self) -> str:
        """Quick view of cluster GPU availability."""
        result = await self._run_cmd(
            f"svp list -p {self.partition} 2>/dev/null || "
            f"sinfo -p {self.partition} -o '%n %G %t' 2>/dev/null | head -20",
            timeout=15,
        )
        return result["stdout"]
