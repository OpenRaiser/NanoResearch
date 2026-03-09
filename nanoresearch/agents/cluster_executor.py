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
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable

from nanoresearch.agents.project_runner import repair_launch_contract, validate_launch_contract
from nanoresearch.agents.runtime_env import ProjectManifestSnapshot, RuntimeEnvironmentManager

logger = logging.getLogger(__name__)

DEFAULT_POLL_INTERVAL = 30
DEFAULT_MAX_WAIT = 14400  # 4 hours
CMD_TIMEOUT = 120
SCP_TIMEOUT = 600
ENV_SETUP_TIMEOUT = 900  # 15 min for pip install
CLUSTER_ENV_VALIDATION_TIMEOUT = 60
ARTIFACT_DIRS = ("results", "checkpoints", "logs")
PIP_MANIFESTS = ("requirements.txt", "pyproject.toml", "setup.py", "setup.cfg")
ENVIRONMENT_MANIFESTS = ("environment.yml", "environment.yaml")
MAX_CLUSTER_IMPORT_PROBES = 5
MAX_CLUSTER_VALIDATION_REPAIR_PACKAGES = 3


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
        self._manifest_snapshots: dict[str, ProjectManifestSnapshot] = {}
        self._manifest_declared_dependencies: dict[str, tuple[str, ...]] = {}
        self._manifest_repair_specs: dict[str, dict[str, str]] = {}
        self._local_code_dirs: dict[str, str] = {}

    def log(self, msg: str) -> None:
        self._log_fn(f"[Cluster] {msg}")

    @staticmethod
    def _ensure_local_artifact_dirs(base_dir: Path) -> None:
        for name in ARTIFACT_DIRS:
            (base_dir / name).mkdir(parents=True, exist_ok=True)

    def _cache_manifest_snapshot(self, cluster_code_path: str, local_code_dir: Path) -> None:
        snapshot = RuntimeEnvironmentManager.inspect_project_manifests(local_code_dir)
        self._manifest_snapshots[cluster_code_path] = snapshot
        self._manifest_declared_dependencies[cluster_code_path] = tuple(
            RuntimeEnvironmentManager.collect_declared_dependency_names(local_code_dir)
        )
        self._manifest_repair_specs[cluster_code_path] = RuntimeEnvironmentManager.collect_repairable_dependency_specs(
            local_code_dir
        )
        self._local_code_dirs[cluster_code_path] = str(local_code_dir)

    def _launch_contract_code_dir(self, cluster_code_path: str) -> Path | None:
        if self.local_mode:
            cluster_dir = Path(cluster_code_path)
            if cluster_dir.exists():
                return cluster_dir
        local_source = self._local_code_dirs.get(cluster_code_path, "")
        if local_source:
            source_dir = Path(local_source)
            if source_dir.exists():
                return source_dir
        return None

    def _validate_launch_contract(
        self,
        cluster_code_path: str,
        script_cmd: str,
    ) -> dict:
        code_dir = self._launch_contract_code_dir(cluster_code_path)
        if code_dir is None:
            return {
                "status": "skipped",
                "command": [],
                "target_kind": "unknown",
                "target": "",
                "resolved_target": "",
                "runner_target": {},
                "artifact_dirs": {},
                "created_dirs": [],
                "warnings": ["Launch contract skipped because no local project mirror is available"],
                "failures": [],
            }
        return validate_launch_contract(script_cmd, code_dir)

    async def _repair_launch_contract(
        self,
        cluster_code_path: str,
        script_cmd: str,
    ) -> dict:
        code_dir = self._launch_contract_code_dir(cluster_code_path)
        if code_dir is None:
            return {
                "status": "skipped",
                "command": [],
                "command_string": script_cmd,
                "actions": [],
                "files_modified": [],
                "initial_contract": {},
                "final_contract": {},
            }

        repair = repair_launch_contract(script_cmd, code_dir)
        if (
            repair.get("status") == "applied"
            and not self.local_mode
            and repair.get("files_modified")
            and cluster_code_path in self._local_code_dirs
        ):
            await self.reupload_code(Path(self._local_code_dirs[cluster_code_path]), cluster_code_path)
        return repair

    def _activate_prefix(self, conda_sh: str, *, pipefail: bool = False) -> str:
        prefix = "set -o pipefail; " if pipefail else ""
        return (
            prefix
            + f"source {conda_sh} && "
            + f"conda activate {self.conda_env} && "
            + "type proxy_on &>/dev/null && proxy_on; "
        )

    @staticmethod
    def _parse_json_tail(stdout: str) -> dict:
        text = str(stdout or "").strip()
        if not text:
            return {}
        try:
            return json.loads(text.splitlines()[-1])
        except json.JSONDecodeError:
            return {}

    def _select_cluster_import_probe_targets(
        self,
        cluster_code_path: str,
        *,
        install_kind: str,
    ) -> tuple[list[dict[str, str]], str]:
        if install_kind not in {"requirements", "environment"}:
            return [], "install_source_not_probe_safe"

        declared_dependencies = list(self._manifest_declared_dependencies.get(cluster_code_path, ()))
        if not declared_dependencies:
            return [], "no_cached_declared_dependencies"

        targets: list[dict[str, str]] = []
        for package_name in declared_dependencies:
            candidates = RuntimeEnvironmentManager._package_import_candidates(package_name)
            if not candidates:
                continue
            targets.append({"package": package_name, "module": candidates[0]})
            if len(targets) >= MAX_CLUSTER_IMPORT_PROBES:
                break

        if not targets:
            return [], "no_probeable_dependencies"
        return targets, ""

    @staticmethod
    def _extract_failed_import_packages(validation: dict | None) -> list[str]:
        if not isinstance(validation, dict):
            return []
        import_probe = validation.get("import_probe")
        if not isinstance(import_probe, dict):
            return []

        packages: list[str] = []
        for failure in import_probe.get("failures", []) or []:
            if not isinstance(failure, dict):
                continue
            package_name = str(failure.get("package") or "").strip()
            if package_name and package_name not in packages:
                packages.append(package_name)
        return packages

    @staticmethod
    def _format_runtime_validation_summary(
        validation: dict[str, object],
        repair: dict[str, object] | None = None,
    ) -> str:
        lines = [f"[runtime_validation] status={validation.get('status', '')}"]
        python_smoke = validation.get("python_smoke")
        if isinstance(python_smoke, dict):
            lines.append(
                f"python={python_smoke.get('status', '')} "
                f"{python_smoke.get('executable', '')} "
                f"{python_smoke.get('version', '')}".strip()
            )
        pip_probe = validation.get("pip_probe")
        if isinstance(pip_probe, dict):
            lines.append(f"pip={pip_probe.get('status', '')} {pip_probe.get('version', '')}".strip())
        import_probe = validation.get("import_probe")
        if isinstance(import_probe, dict):
            lines.append(
                f"imports={import_probe.get('status', '')} "
                f"skipped={import_probe.get('skipped_reason', '')}".strip()
            )
            failures = import_probe.get("failures", []) or []
            if failures:
                lines.append(f"failed_imports={json.dumps(failures, ensure_ascii=False)}")
        if isinstance(repair, dict):
            lines.append(f"[runtime_validation_repair] status={repair.get('status', '')}")
            actions = repair.get("actions", []) or []
            if actions:
                lines.append(f"repair_actions={json.dumps(actions, ensure_ascii=False)}")
        return "\n".join(line for line in lines if line)

    async def _validate_cluster_env(
        self,
        cluster_code_path: str,
        *,
        conda_sh: str,
        install_kind: str,
    ) -> dict:
        activate_prefix = self._activate_prefix(conda_sh, pipefail=True)

        python_script = (
            "import json, sys; "
            "print(json.dumps({'executable': sys.executable, 'version': sys.version.split()[0]}))"
        )
        python_result = await self._run_cmd(
            f"{activate_prefix}python -c {shlex.quote(python_script)}",
            timeout=CLUSTER_ENV_VALIDATION_TIMEOUT,
        )
        python_payload = self._parse_json_tail(python_result.get("stdout", ""))
        python_smoke = {
            "status": "passed" if python_result.get("returncode") == 0 else "failed",
            "returncode": python_result.get("returncode"),
            "stderr": str(python_result.get("stderr") or "")[:300],
            "executable": str(python_payload.get("executable") or ""),
            "version": str(python_payload.get("version") or ""),
        }
        if python_smoke["status"] != "passed":
            return {
                "status": "failed",
                "python_smoke": python_smoke,
                "pip_probe": {"status": "skipped"},
                "import_probe": {"status": "skipped"},
            }

        pip_result = await self._run_cmd(
            f"{activate_prefix}python -m pip --version",
            timeout=CLUSTER_ENV_VALIDATION_TIMEOUT,
        )
        pip_probe = {
            "status": "passed" if pip_result.get("returncode") == 0 else "failed",
            "returncode": pip_result.get("returncode"),
            "version": str(pip_result.get("stdout") or "").strip()[:200],
            "stderr": str(pip_result.get("stderr") or "")[:300],
        }
        if pip_probe["status"] != "passed":
            return {
                "status": "failed",
                "python_smoke": python_smoke,
                "pip_probe": pip_probe,
                "import_probe": {"status": "skipped"},
            }

        probe_targets, skipped_reason = self._select_cluster_import_probe_targets(
            cluster_code_path,
            install_kind=install_kind,
        )
        if not probe_targets:
            import_probe = {
                "status": "skipped",
                "targets": [],
                "failures": [],
                "skipped_reason": skipped_reason,
            }
            return {
                "status": "ready",
                "python_smoke": python_smoke,
                "pip_probe": pip_probe,
                "import_probe": import_probe,
            }

        import_script = "\n".join(
            [
                "import importlib",
                "import json",
                f"targets = {json.dumps(probe_targets, ensure_ascii=False)}",
                "results = []",
                "for item in targets:",
                "    package = item['package']",
                "    module = item['module']",
                "    try:",
                "        importlib.import_module(module)",
                "        results.append({'package': package, 'module': module, 'status': 'passed'})",
                "    except Exception as exc:",
                "        results.append({",
                "            'package': package,",
                "            'module': module,",
                "            'status': 'failed',",
                "            'error': f'{exc.__class__.__name__}: {exc}',",
                "        })",
                "print(json.dumps({'results': results}, ensure_ascii=False))",
            ]
        )
        import_result = await self._run_cmd(
            f"{activate_prefix}python -c {shlex.quote(import_script)}",
            timeout=CLUSTER_ENV_VALIDATION_TIMEOUT,
        )
        parsed_payload = self._parse_json_tail(import_result.get("stdout", ""))
        parsed_results = parsed_payload.get("results", []) if isinstance(parsed_payload, dict) else []
        if not isinstance(parsed_results, list):
            parsed_results = []
        failures = [item for item in parsed_results if isinstance(item, dict) and item.get("status") != "passed"]
        import_status = "passed"
        if import_result.get("returncode") != 0 and not failures:
            import_status = "failed"
            failures = [
                {
                    "package": "",
                    "module": "",
                    "error": str(import_result.get("stderr") or "")[:300],
                }
            ]
        elif failures:
            import_status = "partial"

        return {
            "status": "ready" if import_status == "passed" else ("failed" if import_status == "failed" else "partial"),
            "python_smoke": python_smoke,
            "pip_probe": pip_probe,
            "import_probe": {
                "status": import_status,
                "targets": list(probe_targets),
                "results": list(parsed_results),
                "failures": list(failures),
                "stderr": str(import_result.get("stderr") or "")[:300],
                "skipped_reason": "",
            },
        }

    async def _repair_cluster_validation(
        self,
        cluster_code_path: str,
        *,
        conda_sh: str,
        install_kind: str,
        validation: dict,
    ) -> dict:
        failed_packages = self._extract_failed_import_packages(validation)
        if not failed_packages:
            return {"validation": validation, "repair": {"status": "skipped", "actions": []}}

        spec_index = self._manifest_repair_specs.get(cluster_code_path, {})
        repair_specs: list[str] = []
        unresolved: list[str] = []
        for package_name in failed_packages:
            spec = spec_index.get(package_name)
            if spec and spec not in repair_specs:
                repair_specs.append(spec)
            else:
                unresolved.append(package_name)
            if len(repair_specs) >= MAX_CLUSTER_VALIDATION_REPAIR_PACKAGES:
                break

        repair_actions: list[dict] = []
        current_validation = validation
        if repair_specs:
            install_cmd = (
                f"{self._activate_prefix(conda_sh, pipefail=True)}"
                f"pip install {' '.join(shlex.quote(spec) for spec in repair_specs)} 2>&1 | tail -40"
            )
            result = await self._run_cmd(install_cmd, timeout=ENV_SETUP_TIMEOUT)
            action = {
                "kind": "import_repair_install",
                "status": "installed" if result.get("returncode") == 0 else "failed",
                "specs": list(repair_specs),
                "returncode": result.get("returncode"),
                "stderr": str(result.get("stderr") or "")[:300],
            }
            repair_actions.append(action)
            if result.get("returncode") == 0:
                current_validation = await self._validate_cluster_env(
                    cluster_code_path,
                    conda_sh=conda_sh,
                    install_kind=install_kind,
                )
        elif unresolved:
            repair_actions.append(
                {
                    "kind": "import_repair_skipped",
                    "status": "skipped",
                    "packages": list(unresolved),
                }
            )

        final_status = str(current_validation.get("status") or "").strip()
        if not repair_actions:
            repair_status = "skipped"
        elif final_status == "ready":
            repair_status = "applied"
        elif any(action.get("status") == "failed" for action in repair_actions):
            repair_status = "failed"
        else:
            repair_status = "partial"
        return {
            "validation": current_validation,
            "repair": {
                "status": repair_status,
                "actions": repair_actions,
            },
        }

    @staticmethod
    def _probe_manifest_names(stdout: str) -> set[str]:
        return {line.strip() for line in stdout.splitlines() if line.strip()}

    def _resolve_manifest_policy(
        self,
        cluster_code_path: str,
        probe_stdout: str,
    ) -> tuple[str, str, str, str, str]:
        found = self._probe_manifest_names(probe_stdout)
        snapshot = self._manifest_snapshots.get(cluster_code_path)
        if snapshot is not None:
            expected = {
                name
                for name in (snapshot.environment_source, snapshot.install_source)
                if name
            }
            if not expected or expected.issubset(found):
                manifest_name = snapshot.install_source or snapshot.environment_source
                manifest_kind = "conda" if snapshot.install_kind in {"", "environment"} else "pip"
                environment_name = snapshot.environment_source
                self.log(
                    "Using cached local manifest policy for cluster env setup: "
                    f"install={snapshot.install_source or 'none'}, env={snapshot.environment_source or 'none'}"
                )
                return (
                    manifest_name,
                    manifest_kind if manifest_name else "",
                    environment_name,
                    "cached_local_manifest",
                    snapshot.install_kind,
                )
            self.log(
                "Cluster manifest probe does not match cached local policy; "
                "falling back to remote probe selection"
            )

        manifest_name, manifest_kind = self._select_manifest_from_probe(probe_stdout)
        environment_name = next(
            (name for name in ENVIRONMENT_MANIFESTS if name in found),
            "",
        )
        return manifest_name, manifest_kind, environment_name, "remote_probe", ""

    @staticmethod
    def _manifest_probe_command(cluster_code_path: str) -> str:
        quoted_dir = shlex.quote(cluster_code_path)
        checks = [
            *PIP_MANIFESTS,
            *ENVIRONMENT_MANIFESTS,
        ]
        probe_lines = [
            f'if [ -f {quoted_dir}/{name} ]; then echo "{name}"; fi'
            for name in checks
        ]
        return " ".join(probe_lines) or "true"

    @staticmethod
    def _select_manifest_from_probe(stdout: str) -> tuple[str, str]:
        found = {line.strip() for line in stdout.splitlines() if line.strip()}
        for name in PIP_MANIFESTS:
            if name in found:
                return name, "pip"
        for name in ENVIRONMENT_MANIFESTS:
            if name in found:
                return name, "conda"
        return "", ""

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
                self._ensure_local_artifact_dirs(dest)
                self._cache_manifest_snapshot(str(dest), local_code_dir)
                self.log(f"Code copied to {dest}")
                return str(dest)
            else:
                # Use code in-place
                self._ensure_local_artifact_dirs(local_code_dir)
                self._cache_manifest_snapshot(str(local_code_dir), local_code_dir)
                return str(local_code_dir)
        else:
            # Remote mode: SCP upload
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
        """Re-sync code after LLM modifications."""
        if self.local_mode:
            if str(local_code_dir) != cluster_code_path:
                # Different path — re-copy
                dest = Path(cluster_code_path)
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(local_code_dir, dest)
                self._ensure_local_artifact_dirs(dest)
                self._cache_manifest_snapshot(cluster_code_path, local_code_dir)
            # else: same path, already in-place
            else:
                self._ensure_local_artifact_dirs(local_code_dir)
                self._cache_manifest_snapshot(cluster_code_path, local_code_dir)
        else:
            # Remote: re-upload
            parent = str(Path(cluster_code_path).parent)
            result = await self._scp_upload(str(local_code_dir), f"{parent}/code")
            if result["returncode"] != 0:
                self.log(f"Re-upload warning: {result['stderr'][:200]}")
            await self._run_ssh(
                f"mkdir -p {cluster_code_path}/results {cluster_code_path}/checkpoints {cluster_code_path}/logs"
            )
            self._cache_manifest_snapshot(cluster_code_path, local_code_dir)

    async def setup_env(self, cluster_code_path: str) -> dict:
        """Create/update the cluster conda env and install project dependencies."""
        self.log(f"Setting up conda env '{self.conda_env}'...")
        quoted_code_path = shlex.quote(cluster_code_path)

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

        manifest_probe = await self._run_cmd(
            self._manifest_probe_command(cluster_code_path),
            timeout=10,
        )
        manifest_name, manifest_kind, environment_name, policy_source, install_kind = self._resolve_manifest_policy(
            cluster_code_path,
            manifest_probe.get("stdout", ""),
        )
        environment_path = (
            f"{cluster_code_path}/{environment_name}" if environment_name else ""
        )

        check_env = (
            f"source {conda_sh} 2>/dev/null && "
            f"conda env list | grep -w {self.conda_env} && echo ENV_EXISTS || echo ENV_MISSING"
        )
        check_result = await self._run_cmd(check_env, timeout=30)
        env_missing = "ENV_MISSING" in check_result.get("stdout", "")

        env_cmd = ""
        env_strategy = "existing"
        if env_missing:
            if environment_path:
                self.log(
                    f"Creating conda env '{self.conda_env}' from {environment_name}..."
                )
                env_cmd = (
                    "set -o pipefail; "
                    f"source {conda_sh} && "
                    f"conda env create -n {self.conda_env} -f {shlex.quote(environment_path)} "
                    f"2>&1 | tail -40"
                )
                env_strategy = "conda_env_create"
            else:
                self.log(
                    f"Creating conda env '{self.conda_env}' (python={self.python_version})..."
                )
                env_cmd = (
                    "set -o pipefail; "
                    f"source {conda_sh} && "
                    f"conda create -n {self.conda_env} python={self.python_version} -y "
                    f"2>&1 | tail -40"
                )
                env_strategy = "conda_create"
        elif environment_path:
            self.log(
                f"Updating conda env '{self.conda_env}' from {environment_name}..."
            )
            env_cmd = (
                "set -o pipefail; "
                f"source {conda_sh} && "
                f"conda env update -n {self.conda_env} -f {shlex.quote(environment_path)} --prune "
                f"2>&1 | tail -40"
            )
            env_strategy = "conda_env_update"
        else:
            self.log(f"Conda env '{self.conda_env}' already exists")

        env_output = ""
        if env_cmd:
            env_result = await self._run_cmd(env_cmd, timeout=ENV_SETUP_TIMEOUT)
            env_output = (env_result.get("stdout", "") + "\n" + env_result.get("stderr", "")).strip()
            if env_result.get("returncode") != 0:
                self.log(f"Conda env preparation failed (rc={env_result['returncode']})")
                self.log(env_output[-500:])
                return {
                    "ok": False,
                    "output": env_output,
                    "manifest": environment_path,
                    "source": environment_name,
                    "strategy": env_strategy,
                    "policy_source": policy_source,
                }

        async def finalize_success(
            *,
            output: str,
            manifest: str,
            source: str,
            strategy: str,
        ) -> dict:
            runtime_validation = await self._validate_cluster_env(
                cluster_code_path,
                conda_sh=conda_sh,
                install_kind=install_kind,
            )
            repair_result = await self._repair_cluster_validation(
                cluster_code_path,
                conda_sh=conda_sh,
                install_kind=install_kind,
                validation=runtime_validation,
            )
            runtime_validation = repair_result["validation"]
            runtime_validation_repair = repair_result["repair"]
            summary = self._format_runtime_validation_summary(
                runtime_validation,
                runtime_validation_repair,
            )
            combined_output = "\n".join(part for part in [output.strip(), summary] if part).strip()
            return {
                "ok": runtime_validation.get("status") == "ready",
                "output": combined_output,
                "manifest": manifest,
                "source": source,
                "strategy": strategy,
                "policy_source": policy_source,
                "runtime_validation": runtime_validation,
                "runtime_validation_repair": runtime_validation_repair,
            }

        if not manifest_name:
            self.log("No dependency manifest found, skipping dependency install")
            return await finalize_success(
                output=env_output or "No dependency manifest",
                manifest="",
                source="",
                strategy=env_strategy,
            )

        if manifest_kind == "conda" or install_kind == "environment":
            return await finalize_success(
                output=env_output or f"Applied {manifest_name}",
                manifest=environment_path,
                source=manifest_name,
                strategy=env_strategy,
            )

        manifest_path = f"{cluster_code_path}/{manifest_name}"
        activate_prefix = (
            "set -o pipefail; "
            f"source {conda_sh} && "
            f"conda activate {self.conda_env} && "
            f"type proxy_on &>/dev/null && proxy_on; "
        )

        install_attempts: list[tuple[str, str]] = []
        if manifest_name == "requirements.txt":
            install_attempts.append(
                (
                    "requirements",
                    f"{activate_prefix}pip install -r {shlex.quote(manifest_path)} 2>&1 | tail -40",
                )
            )
        else:
            install_attempts.append(
                (
                    "editable",
                    f"{activate_prefix}pip install -e {quoted_code_path} 2>&1 | tail -40",
                )
            )
            install_attempts.append(
                (
                    "package",
                    f"{activate_prefix}pip install {quoted_code_path} 2>&1 | tail -40",
                )
            )

        last_output = env_output
        for strategy_name, install_cmd in install_attempts:
            self.log(
                f"Installing dependencies from {manifest_name} ({strategy_name})..."
            )
            result = await self._run_cmd(install_cmd, timeout=ENV_SETUP_TIMEOUT)
            last_output = (
                (env_output + "\n") if env_output else ""
            ) + result.get("stdout", "") + "\n" + result.get("stderr", "")
            if result.get("returncode") == 0:
                self.log("Dependency install completed successfully")
                return await finalize_success(
                    output=last_output.strip(),
                    manifest=manifest_path,
                    source=manifest_name,
                    strategy=strategy_name,
                )
            self.log(
                f"Dependency install failed via {manifest_name} ({strategy_name}), rc={result['returncode']}"
            )
            self.log(last_output[-500:])

        return {
            "ok": False,
            "output": last_output.strip(),
            "manifest": manifest_path,
            "source": manifest_name,
            "strategy": install_attempts[-1][0],
            "policy_source": policy_source,
            "runtime_validation": {"status": "skipped"},
            "runtime_validation_repair": {"status": "skipped", "actions": []},
        }

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
