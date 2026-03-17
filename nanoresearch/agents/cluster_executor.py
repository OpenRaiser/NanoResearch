"""Cluster executor -- run experiments on a SLURM cluster.

Two modes:
  - LOCAL mode (local=true): run sbatch/squeue directly on the current machine.
  - REMOTE mode (local=false): run commands via SSH/SCP through a bastion.

Split into 3 modules:
    cluster_executor.py      -- ClusterExecutor facade + __init__ + helpers
    cluster_executor_env.py  -- _ClusterExecutorEnvMixin (env validation/setup)
    cluster_executor_ops.py  -- _ClusterExecutorOpsMixin (shell/SCP/SLURM ops)
"""

from __future__ import annotations

import json
import logging
import shlex
from pathlib import Path
from typing import Callable

from nanoresearch.agents.project_runner import repair_launch_contract, validate_launch_contract
from nanoresearch.agents.runtime_env import ProjectManifestSnapshot, RuntimeEnvironmentManager

logger = logging.getLogger(__name__)

from nanoresearch.agents.constants import (
    CLUSTER_ENV_VALIDATION_TIMEOUT,
    CLUSTER_MAX_WAIT,
    CLUSTER_POLL_INTERVAL,
    CMD_TIMEOUT,
    ENV_SETUP_TIMEOUT,
    SCP_TIMEOUT,
)

DEFAULT_POLL_INTERVAL = CLUSTER_POLL_INTERVAL  # backward compat alias
DEFAULT_MAX_WAIT = CLUSTER_MAX_WAIT
ARTIFACT_DIRS = ("results", "checkpoints", "logs")
PIP_MANIFESTS = ("requirements.txt", "pyproject.toml", "setup.py", "setup.cfg")
ENVIRONMENT_MANIFESTS = ("environment.yml", "environment.yaml")
MAX_CLUSTER_IMPORT_PROBES = 5
MAX_CLUSTER_VALIDATION_REPAIR_PACKAGES = 3

from nanoresearch.agents.cluster_executor_env import _ClusterExecutorEnvMixin   # noqa: E402
from nanoresearch.agents.cluster_executor_ops import _ClusterExecutorOpsMixin   # noqa: E402


class ClusterExecutor(_ClusterExecutorEnvMixin, _ClusterExecutorOpsMixin):
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

    def _validate_launch_contract(self, cluster_code_path: str, script_cmd: str) -> dict:
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

    async def _repair_launch_contract(self, cluster_code_path: str, script_cmd: str) -> dict:
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
