"""Conda environment management mixin."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ._constants import (
    _CUDA_DRIVER_TO_CONDA_CUDA,
    _TORCH_FAMILY_PACKAGES,
)
import nanoresearch.agents.runtime_env as _runtime_env_mod
from ._types import ExperimentExecutionPolicy

logger = logging.getLogger(__name__)


class _CondaMixin:
    """Mixin — conda environment creation and management."""

    async def _create_per_session_conda_env(
        self,
        code_dir: Path,
        execution_policy: "ExperimentExecutionPolicy",
    ) -> dict[str, Any]:
        """Create (or reuse) a per-session conda env and return env_info dict.

        Steps:
        1. Check if env already exists (resume idempotency).
        2. Create bare env with Python 3.11.
        3. If GPU detected → ``_install_torch_conda()``.
        4. ``install_requirements()`` for remaining pip deps.
        5. ``validate_runtime()``.
        """
        env_name = self._per_session_env_name()
        cmd = _runtime_env_mod._find_conda() or "conda"
        requirements_path = code_dir / "requirements.txt"
        environment_file = self._find_environment_file(code_dir)

        # 1. Check if already exists
        freshly_created = False
        conda_python = self.find_conda_python(env_name)
        if conda_python:
            self._log(f"Reusing existing conda env '{env_name}': {conda_python}")
        else:
            # 2. Create bare env
            self._log(f"Creating per-session conda env '{env_name}' via {cmd} ...")
            loop = asyncio.get_running_loop()
            try:
                proc = await loop.run_in_executor(
                    None,
                    lambda: subprocess.run(
                        [cmd, "create", "-y", "-n", env_name, "python=3.11"],
                        capture_output=True, text=True, timeout=600,
                    ),
                )
                if proc.returncode != 0:
                    stderr = (proc.stderr or "").strip()[:500]
                    self._log(f"Conda env creation failed: {stderr}")
                    return {}  # signal failure — caller falls back to venv
            except Exception as exc:
                self._log(f"Conda env creation error: {exc}")
                return {}

            conda_python = self.find_conda_python(env_name)
            if not conda_python:
                self._log(f"Could not locate Python in new conda env '{env_name}'")
                return {}
            freshly_created = True
            self._log(f"Conda env '{env_name}' created (python: {conda_python})")

        # 3. GPU-aware torch installation via conda
        gpu_info = _runtime_env_mod._detect_gpu_cuda()
        if gpu_info:
            await self._install_torch_conda(env_name, cmd, gpu_info)

        # 4. Install remaining deps via pip
        install_info = await self.install_requirements(conda_python, code_dir)

        # 5. Verify torch CUDA if GPU present
        if gpu_info:
            await self._verify_torch_cuda(conda_python, code_dir, gpu_info)

        # 6. Validate runtime
        runtime_validation = await self.validate_runtime(
            conda_python, code_dir, execution_policy=execution_policy,
        )

        return {
            "kind": "conda",
            "python": conda_python,
            "env_name": env_name,
            "created": freshly_created,
            "per_session": True,
            "requirements_path": str(requirements_path) if requirements_path.exists() else "",
            "environment_file": str(environment_file) if environment_file else "",
            "dependency_install": install_info,
            "runtime_validation": runtime_validation,
            "runtime_validation_repair": {"status": "skipped", "actions": []},
            "execution_policy": execution_policy.to_dict(),
        }

    async def _install_torch_conda(
        self,
        env_name: str,
        cmd: str,
        gpu_info: dict[str, Any],
    ) -> bool:
        """Install PyTorch with CUDA via conda into a named env.

        Uses ``conda install pytorch torchvision torchaudio pytorch-cuda=XX.X
        -c pytorch -c nvidia``.  Returns True on success.
        """
        cuda_ver = gpu_info.get("cuda_version")
        if not cuda_ver:
            return False

        # Find best conda pytorch-cuda version
        conda_cuda = ""
        for min_ver, tag in _CUDA_DRIVER_TO_CONDA_CUDA:
            if cuda_ver >= min_ver:
                conda_cuda = tag
                break
        if not conda_cuda:
            self._log(f"CUDA {cuda_ver} too old for conda pytorch-cuda packages")
            return False

        self._log(
            f"Installing PyTorch with CUDA via {cmd}: "
            f"GPU={gpu_info['gpu_name']}, pytorch-cuda={conda_cuda}"
        )

        loop = asyncio.get_running_loop()
        try:
            proc = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    [
                        cmd, "install", "-y", "-n", env_name,
                        "pytorch", "torchvision", "torchaudio",
                        f"pytorch-cuda={conda_cuda}",
                        "-c", "pytorch", "-c", "nvidia",
                    ],
                    capture_output=True, text=True, timeout=1800,
                ),
            )
            if proc.returncode == 0:
                self._log("PyTorch CUDA conda install OK")
                return True
            stderr = (proc.stderr or "").strip()[:500]
            self._log(f"PyTorch CUDA conda install failed: {stderr}")
        except Exception as exc:
            self._log(f"PyTorch CUDA conda install error: {exc}")

        return False

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    @staticmethod
    def list_nanoresearch_conda_envs() -> list[dict[str, str]]:
        """List all ``nanoresearch_*`` conda environments.

        Returns list of ``{"name": str, "path": str}``.
        """
        cmd = _runtime_env_mod._find_conda()
        if cmd is None:
            return []
        try:
            proc = subprocess.run(
                [cmd, "env", "list", "--json"],
                capture_output=True, text=True, timeout=30,
            )
            if proc.returncode != 0:
                return []
            import json as _json
            data = _json.loads(proc.stdout)
            envs = []
            for env_path in data.get("envs", []):
                name = Path(env_path).name
                if name.startswith("nanoresearch_"):
                    envs.append({"name": name, "path": env_path})
            return envs
        except Exception:
            return []

    @staticmethod
    def remove_conda_env(env_name: str) -> bool:
        """Remove a conda environment by name. Returns True on success."""
        cmd = _runtime_env_mod._find_conda() or "conda"
        try:
            proc = subprocess.run(
                [cmd, "env", "remove", "-y", "-n", env_name],
                capture_output=True, text=True, timeout=120,
            )
            return proc.returncode == 0
        except Exception:
            return False

    @staticmethod
    def find_conda_python(env_name: str) -> str | None:
        """Find the Python executable for a named conda env."""
        conda_cmd = _runtime_env_mod._find_conda() or "conda"
        try:
            result = subprocess.run(
                [conda_cmd, "run", "-n", env_name, "python", "-c", "import sys; print(sys.executable)"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                path = result.stdout.strip().split("\n")[-1].strip()
                if path and Path(path).exists():
                    return path
        except Exception:
            pass

        # Fallback: query conda for env directories, then look for python
        is_windows = platform.system() == "Windows"
        python_bin = "python.exe" if is_windows else "bin/python"
        envs_dirs: list[Path] = []
        try:
            info_result = subprocess.run(
                [conda_cmd, "info", "--json"],
                capture_output=True, text=True, timeout=15,
            )
            if info_result.returncode == 0:
                info_data = json.loads(info_result.stdout)
                # envs_dirs contains directories that hold conda envs
                # e.g. ["/home/user/anaconda3/envs", "/data/conda_envs"]
                for ed in info_data.get("envs_dirs", []):
                    p = Path(ed)
                    if p not in envs_dirs:
                        envs_dirs.append(p)
                # Also derive from root_prefix
                root_prefix = info_data.get("root_prefix", "")
                if root_prefix:
                    default_envs = Path(root_prefix) / "envs"
                    if default_envs not in envs_dirs:
                        envs_dirs.append(default_envs)
        except Exception:
            pass

        # Static fallbacks (only if dynamic query failed)
        if not envs_dirs:
            envs_dirs = [
                Path.home() / "anaconda3" / "envs",
                Path.home() / "miniconda3" / "envs",
                Path("D:/anaconda") / "envs",
                Path("C:/anaconda3") / "envs",
            ]

        for envs_dir in envs_dirs:
            python_path = envs_dir / env_name / python_bin
            if python_path.exists():
                return str(python_path)
        return None

    async def create_conda_env(self, env_name: str, code_dir: Path) -> bool:
        """Create a conda environment when requested and missing."""
        env_file = self._find_environment_file(code_dir)
        conda_cmd = _runtime_env_mod._find_conda() or "conda"
        self._log(f"Creating conda env '{env_name}' via {conda_cmd} ...")
        loop = asyncio.get_running_loop()
        try:
            if env_file is not None:
                proc_result = await loop.run_in_executor(
                    None,
                    lambda: subprocess.run(
                        [conda_cmd, "env", "create", "-n", env_name, "-f", str(env_file)],
                        cwd=str(code_dir),
                        capture_output=True,
                        text=True,
                        timeout=1800,
                    ),
                )
            else:
                proc_result = await loop.run_in_executor(
                    None,
                    lambda: subprocess.run(
                        [conda_cmd, "create", "-y", "-n", env_name, "python=3.10"],
                        cwd=str(code_dir),
                        capture_output=True,
                        text=True,
                        timeout=1200,
                    ),
                )
            if proc_result.returncode == 0:
                self._log(f"Conda env '{env_name}' created")
                return True
            stderr = (proc_result.stderr or "").strip()
            self._log(f"Failed to create conda env '{env_name}': {stderr[:500]}")
        except Exception as exc:
            self._log(f"Conda env creation error: {exc}")
        return False
