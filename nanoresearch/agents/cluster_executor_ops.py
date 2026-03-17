"""Cluster executor ops mixin -- SSH/SCP, SLURM submit/wait, file transfer."""

from __future__ import annotations

import asyncio
import logging
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from nanoresearch.agents.constants import CMD_TIMEOUT, SCP_TIMEOUT

logger = logging.getLogger(__name__)

ARTIFACT_DIRS = ("results", "checkpoints", "logs")


class _ClusterExecutorOpsMixin:
    """Mixin: shell execution, code transfer, SLURM operations."""

    # ── shell execution ──

    async def _run_cmd(self, cmd: str, timeout: int = CMD_TIMEOUT) -> dict:
        if self.local_mode:
            return await self._run_local_shell(cmd, timeout)
        else:
            return await self._run_ssh(cmd, timeout)

    async def _run_local_shell(self, cmd: str, timeout: int = CMD_TIMEOUT) -> dict:
        self.log(f"$ {cmd[:120]}...")
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    ["bash", "-c", cmd],
                    capture_output=True, text=True, timeout=timeout,
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
                    ssh_cmd, capture_output=True, text=True, timeout=timeout,
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

    # ── high-level operations ──

    async def check_connectivity(self) -> bool:
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
        if self.local_mode:
            if self.base_path:
                dest = Path(self.base_path) / session_id / "code"
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(local_code_dir, dest)
                self._ensure_local_artifact_dirs(dest)
                self._cache_manifest_snapshot(str(dest), local_code_dir)
                self.log(f"Code copied to {dest}")
                return str(dest)
            else:
                self._ensure_local_artifact_dirs(local_code_dir)
                self._cache_manifest_snapshot(str(local_code_dir), local_code_dir)
                return str(local_code_dir)
        else:
            remote_dir = f"{self.base_path}/{session_id}"
            await self._run_ssh(f"mkdir -p {remote_dir}")
            result = await self._scp_upload(str(local_code_dir), f"{remote_dir}/code")
            if result["returncode"] != 0:
                raise RuntimeError(f"SCP upload failed: {result['stderr']}")
            await self._run_ssh(
                f"mkdir -p {remote_dir}/code/results {remote_dir}/code/checkpoints {remote_dir}/code/logs"
            )
            self._cache_manifest_snapshot(f"{remote_dir}/code", local_code_dir)
            self.log(f"Code uploaded to {remote_dir}/code")
            return f"{remote_dir}/code"

    async def reupload_code(self, local_code_dir: Path, cluster_code_path: str) -> None:
        if self.local_mode:
            if str(local_code_dir) != cluster_code_path:
                dest = Path(cluster_code_path)
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(local_code_dir, dest)
                self._ensure_local_artifact_dirs(dest)
                self._cache_manifest_snapshot(cluster_code_path, local_code_dir)
            else:
                self._ensure_local_artifact_dirs(local_code_dir)
                self._cache_manifest_snapshot(cluster_code_path, local_code_dir)
        else:
            parent = str(Path(cluster_code_path).parent)
            result = await self._scp_upload(str(local_code_dir), f"{parent}/code")
            if result["returncode"] != 0:
                self.log(f"Re-upload warning: {result['stderr'][:200]}")
            await self._run_ssh(
                f"mkdir -p {cluster_code_path}/results {cluster_code_path}/checkpoints {cluster_code_path}/logs"
            )
            self._cache_manifest_snapshot(cluster_code_path, local_code_dir)

    def _generate_sbatch_script(self, cluster_code_path: str, script_cmd: str) -> str:
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
        launch_contract = self._validate_launch_contract(cluster_code_path, script_cmd)
        if launch_contract.get("status") == "failed":
            repair = await self._repair_launch_contract(cluster_code_path, script_cmd)
            if repair.get("status") == "applied":
                repaired_cmd = str(repair.get("command_string") or "").strip()
                if repaired_cmd:
                    script_cmd = repaired_cmd
                launch_contract = self._validate_launch_contract(cluster_code_path, script_cmd)
                if repair.get("actions"):
                    self.log(f"Applied launch-contract repair: {repair['actions']}")
            if launch_contract.get("status") == "failed":
                failure_text = "; ".join(launch_contract.get("failures", [])[:3]) or "unknown launch target failure"
                raise RuntimeError(f"Launch contract failed: {failure_text}")

        sbatch_content = self._generate_sbatch_script(cluster_code_path, script_cmd)
        sbatch_path = f"{cluster_code_path}/job.sh"

        if self.local_mode:
            Path(sbatch_path).write_text(sbatch_content, encoding="utf-8")
            Path(sbatch_path).chmod(0o755)
        else:
            write_cmd = f"cat > {sbatch_path} << 'NANO_SBATCH_EOF'\n{sbatch_content}\nNANO_SBATCH_EOF"
            await self._run_cmd(write_cmd, timeout=15)
            await self._run_cmd(f"chmod +x {sbatch_path}", timeout=5)

        self.log(f"Submitting: sbatch {sbatch_path}")
        result = await self._run_cmd(f"sbatch {sbatch_path}", timeout=30)
        if result["returncode"] != 0:
            raise RuntimeError(f"sbatch failed (rc={result['returncode']}): {result['stderr']}")

        match = re.search(r"(\d+)", result["stdout"])
        if not match:
            raise RuntimeError(f"Could not parse job ID from: {result['stdout']}")

        job_id = match.group(1)
        self.log(f"Job submitted: {job_id}")
        return job_id

    async def wait_for_job(self, job_id: str) -> dict:
        self.log(f"Waiting for job {job_id} (poll={self.poll_interval}s, max={self.max_wait}s)...")
        start = time.time()
        last_status = ""

        while time.time() - start < self.max_wait:
            result = await self._run_cmd(
                f"squeue -j {job_id} -h -o '%T' 2>/dev/null", timeout=15,
            )
            status = result["stdout"].strip().strip("'\"")
            if not status:
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
            return {"job_id": job_id, "state": "WAIT_TIMEOUT", "exit_code": "?", "elapsed": int(time.time() - start)}

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
            state = "COMPLETED"
        total = int(time.time() - start)
        self.log(f"Job {job_id}: {state} (exit={exit_code}, slurm_elapsed={elapsed_str}, wait={total}s)")
        return {"job_id": job_id, "state": state, "exit_code": exit_code, "elapsed_slurm": elapsed_str, "elapsed_wait": total}

    async def get_job_log(self, cluster_code_path: str, job_id: str, tail: int = 300) -> str:
        cmd = (
            f"echo '=== STDOUT ===' && tail -{tail} {cluster_code_path}/logs/{job_id}.log 2>/dev/null; "
            f"echo '\\n=== STDERR ===' && tail -{tail} {cluster_code_path}/logs/{job_id}.err 2>/dev/null"
        )
        result = await self._run_cmd(cmd, timeout=30)
        log_text = result["stdout"]
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
        if self.local_mode:
            source_root = Path(cluster_code_path)
            target_root = local_workspace / "code"
            copied_any = False
            self._ensure_local_artifact_dirs(target_root)
            for name in ARTIFACT_DIRS:
                src_dir = source_root / name
                dst_dir = target_root / name
                if not src_dir.exists():
                    continue
                if src_dir.resolve() == dst_dir.resolve():
                    copied_any = True
                    continue
                if dst_dir.exists():
                    shutil.rmtree(dst_dir)
                shutil.copytree(src_dir, dst_dir)
                copied_any = True
            if copied_any:
                self.log("Results copied locally")
                return True
            self.log("No cluster artifacts found to copy")
            return False
        else:
            target_root = local_workspace / "code"
            self._ensure_local_artifact_dirs(target_root)
            copied_any = False
            for name in ARTIFACT_DIRS:
                remote = f"{cluster_code_path}/{name}"
                local = str(target_root / name)
                result = await self._scp_download(remote, local)
                if result["returncode"] == 0:
                    copied_any = True
                else:
                    self.log(f"Artifact sync warning for {name}: {result['stderr'][:200]}")
            return copied_any

    async def cancel_job(self, job_id: str) -> None:
        await self._run_cmd(f"scancel {job_id}", timeout=15)
        self.log(f"Job {job_id} cancelled")

    async def check_resources(self) -> str:
        result = await self._run_cmd(
            f"svp list -p {self.partition} 2>/dev/null || "
            f"sinfo -p {self.partition} -o '%n %G %t' 2>/dev/null | head -20",
            timeout=15,
        )
        return result["stdout"]
