"""Installation, venv recreation, and torch CUDA mixin."""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import venv
from pathlib import Path
from typing import Any

from ._constants import (
    _CUDA_DRIVER_TO_TORCH_TAG,
    _TORCH_FAMILY_PACKAGES,
    PACKAGE_IMPORT_ALIASES,
)
from ._discovery import _split_torch_requirements
import nanoresearch.agents.runtime_env as _runtime_env_mod
from ._types import DependencyInstallPlan

logger = logging.getLogger(__name__)


class _InstallMixin:
    """Mixin — dependency installation, venv, torch CUDA."""

    async def install_dependency_specs(
        self,
        python: str,
        code_dir: Path,
        specs: list[str],
        *,
        source: str = "runtime_validation",
    ) -> dict[str, Any]:
        filtered_specs = [str(spec).strip() for spec in specs if str(spec).strip()]
        if not filtered_specs:
            return {"status": "skipped", "source": source, "specs": []}

        self._log(f"Installing targeted dependency specs via {source}: {filtered_specs}")
        loop = asyncio.get_running_loop()
        try:
            proc_result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    [python, "-m", "pip", "install", *filtered_specs, "--quiet"],
                    cwd=str(code_dir),
                    capture_output=True,
                    text=True,
                    timeout=600,
                ),
            )
        except Exception as exc:
            self._log(f"Targeted dependency install failed via {source}: {exc}")
            return {
                "status": "error",
                "source": source,
                "specs": filtered_specs,
                "error": str(exc),
            }

        if proc_result.returncode == 0:
            self._log(f"Targeted dependency install OK via {source}")
            return {
                "status": "installed",
                "source": source,
                "specs": filtered_specs,
            }

        stderr = (proc_result.stderr or "").strip()
        self._log(f"Targeted dependency install rc={proc_result.returncode} via {source}: {stderr[:500]}")
        return {
            "status": "failed",
            "source": source,
            "specs": filtered_specs,
            "returncode": proc_result.returncode,
            "stderr": stderr[:500],
        }

    async def _recreate_venv(self, env_dir: Path) -> dict[str, Any]:
        self._log(f"Recreating invalid venv at {env_dir}")
        loop = asyncio.get_running_loop()
        shutil.rmtree(env_dir, ignore_errors=True)
        try:
            await loop.run_in_executor(
                None,
                lambda: venv.create(str(env_dir), with_pip=True),
            )
        except Exception as exc:
            self._log(f"Venv recreation failed at {env_dir}: {exc}")
            return {
                "status": "failed",
                "error": str(exc),
            }

        is_windows = platform.system() == "Windows"
        python_path = env_dir / ("Scripts/python.exe" if is_windows else "bin/python")
        return {
            "status": "applied",
            "python": str(python_path),
        }

    async def install_requirements(self, python: str, code_dir: Path) -> dict[str, Any]:
        """Install dependencies from the best available project manifest.

        If the project requires PyTorch-family packages and an NVIDIA GPU is
        detected, installs them from the official PyTorch CUDA wheel index
        *before* installing the rest of the requirements.  This avoids the
        common Windows issue where the default PyPI torch wheel doesn't match
        the system's CUDA/driver configuration and fails with DLL load errors.
        """
        install_plan = self._select_install_plan(code_dir)
        if install_plan is None:
            self._log("No dependency manifest found, skipping pip install")
            return {"status": "skipped", "source": "", "manifest": ""}

        # ── Pre-install: CUDA-aware PyTorch installation ──
        gpu_info: dict[str, Any] | None = None
        torch_pre_installed = False
        try:
            gpu_info = _runtime_env_mod._detect_gpu_cuda()
        except Exception:
            pass  # GPU detection is best-effort; fall through to normal install
        if gpu_info is not None:
            torch_pre_installed = await self._preinstall_torch_cuda(
                python, code_dir, install_plan, gpu_info,
            )

        self._log(f"Installing dependencies from {install_plan.source} ...")
        loop = asyncio.get_running_loop()
        attempts = [
            ("primary", install_plan.args),
            ("fallback", install_plan.fallback_args or []),
        ]
        last_failure: dict[str, Any] | None = None
        for strategy, install_args in attempts:
            if not install_args:
                continue
            try:
                proc_result = await loop.run_in_executor(
                    None,
                    lambda: subprocess.run(
                        [python, "-m", "pip", "install", *install_args, "--quiet"],
                        cwd=str(code_dir),
                        capture_output=True,
                        text=True,
                        timeout=1800,  # 30 min — torch+transformers+datasets can be large
                    ),
                )
            except Exception as exc:
                self._log(f"pip install error via {install_plan.source} ({strategy}): {exc}")
                last_failure = {
                    "status": "error",
                    "source": install_plan.source,
                    "manifest": install_plan.manifest_path,
                    "strategy": strategy,
                    "error": str(exc),
                }
                continue

            if proc_result.returncode == 0:
                self._log(f"Dependency install OK via {install_plan.source} ({strategy})")
                result: dict[str, Any] = {
                    "status": "installed",
                    "source": install_plan.source,
                    "manifest": install_plan.manifest_path,
                    "strategy": strategy,
                }
                if torch_pre_installed and gpu_info:
                    result["torch_cuda"] = {
                        "gpu": gpu_info["gpu_name"],
                        "cuda_version": gpu_info["cuda_version_str"],
                        "torch_tag": gpu_info["torch_tag"],
                        "index_url": gpu_info["torch_index_url"],
                    }
                # ── Post-install: verify torch CUDA actually works ──
                if torch_pre_installed:
                    await self._verify_torch_cuda(python, code_dir, gpu_info)
                return result

            stderr = (proc_result.stderr or "").strip()
            self._log(
                f"pip install returned rc={proc_result.returncode} via "
                f"{install_plan.source} ({strategy}): {stderr[:500]}"
            )
            last_failure = {
                "status": "failed",
                "source": install_plan.source,
                "manifest": install_plan.manifest_path,
                "strategy": strategy,
                "returncode": proc_result.returncode,
                "stderr": stderr[:500],
            }

        return last_failure or {
            "status": "skipped",
            "source": install_plan.source,
            "manifest": install_plan.manifest_path,
        }

    async def _preinstall_torch_cuda(
        self,
        python: str,
        code_dir: Path,
        install_plan: DependencyInstallPlan,
        gpu_info: dict[str, Any],
    ) -> bool:
        """Install PyTorch-family packages from the official CUDA wheel index.

        Parses the project manifest to find torch-family dependencies, then
        installs them from ``https://download.pytorch.org/whl/<cuda_tag>``
        *before* the main pip install runs.  Returns True if any torch packages
        were pre-installed.
        """
        # Read requirements to find torch-family packages
        torch_specs: list[str] = []
        manifest_path = Path(install_plan.manifest_path)
        if manifest_path.suffix == ".txt" and manifest_path.exists():
            lines = manifest_path.read_text(encoding="utf-8").splitlines()
            torch_specs, _ = _split_torch_requirements(lines)
        elif install_plan.source in ("pyproject.toml", "setup.py", "setup.cfg"):
            # For project installs, just pre-install torch with CUDA
            torch_specs = ["torch"]
        else:
            # environment.yml — check pip deps
            env_file = Path(install_plan.manifest_path)
            if env_file.exists():
                try:
                    import yaml
                    data = yaml.safe_load(env_file.read_text(encoding="utf-8"))
                    for dep_block in (data or {}).get("dependencies", []):
                        if isinstance(dep_block, dict) and "pip" in dep_block:
                            pip_deps = dep_block["pip"]
                            torch_specs, _ = _split_torch_requirements(pip_deps)
                except Exception:
                    pass
            if not torch_specs:
                # Fallback: check if torch is likely needed
                torch_specs = ["torch"]

        if not torch_specs:
            return False

        index_url = gpu_info["torch_index_url"]
        self._log(
            f"Pre-installing PyTorch with CUDA support: "
            f"GPU={gpu_info['gpu_name']}, "
            f"CUDA={gpu_info['cuda_version_str']}, "
            f"index={index_url}"
        )
        self._log(f"  torch packages: {torch_specs}")

        loop = asyncio.get_running_loop()
        try:
            proc = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    [
                        python, "-m", "pip", "install",
                        *torch_specs,
                        "--index-url", index_url,
                        "--quiet",
                    ],
                    cwd=str(code_dir),
                    capture_output=True,
                    text=True,
                    timeout=1800,
                ),
            )
            if proc.returncode == 0:
                self._log("PyTorch CUDA pre-install OK")
                return True
            else:
                stderr = (proc.stderr or "").strip()[:500]
                self._log(f"PyTorch CUDA pre-install failed (rc={proc.returncode}): {stderr}")
                # Try without version pin (just 'torch')
                if torch_specs != ["torch"]:
                    self._log("Retrying with unversioned 'torch' ...")
                    proc2 = await loop.run_in_executor(
                        None,
                        lambda: subprocess.run(
                            [
                                python, "-m", "pip", "install",
                                "torch", "torchvision", "torchaudio",
                                "--index-url", index_url,
                                "--quiet",
                            ],
                            cwd=str(code_dir),
                            capture_output=True,
                            text=True,
                            timeout=1800,
                        ),
                    )
                    if proc2.returncode == 0:
                        self._log("PyTorch CUDA pre-install OK (unversioned)")
                        return True
                    stderr2 = (proc2.stderr or "").strip()[:500]
                    self._log(f"PyTorch CUDA pre-install retry failed: {stderr2}")
        except Exception as exc:
            self._log(f"PyTorch CUDA pre-install error: {exc}")

        return False

    async def _verify_torch_cuda(
        self,
        python: str,
        code_dir: Path,
        gpu_info: dict[str, Any] | None,
    ) -> None:
        """Verify that torch imports and CUDA is available post-install."""
        loop = asyncio.get_running_loop()
        check_script = (
            "import torch; "
            "print(f'torch={torch.__version__} cuda={torch.cuda.is_available()} '  "
            "f'device_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}')"
        )
        try:
            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    [python, "-c", check_script],
                    capture_output=True, text=True, timeout=30,
                    cwd=str(code_dir),
                ),
            )
            output = (result.stdout or "").strip()
            if result.returncode == 0:
                self._log(f"Torch CUDA verification: {output}")
            else:
                stderr = (result.stderr or "").strip()[:300]
                self._log(f"Torch CUDA verification FAILED: {stderr}")
        except Exception as exc:
            self._log(f"Torch CUDA verification error: {exc}")

    async def _auto_repair_env(
        self,
        code_dir: Path,
        failed_venv_dir: Path,
        execution_policy: "ExecutionPolicy",
        requirements_path: Path,
        environment_file: Path | None,
    ) -> dict[str, Any] | None:
        """Try to repair environment creation after venv failure.

        Strategies (tried in order):
        1. If conda is available, create a fresh conda env
        2. If venv dir is corrupted, remove and retry

        Returns env_info dict on success, None if all strategies failed.
        """
        # Strategy 1: Try conda — reuse existing or create new
        # Name is deterministic per session, so resume won't create duplicates.
        if _runtime_env_mod._find_conda() is not None:
            auto_env_name = self._per_session_env_name()

            # Check if this env already exists (idempotent on resume)
            conda_python = self.find_conda_python(auto_env_name)
            if conda_python:
                self._log(f"Auto-repair: reusing existing conda env '{auto_env_name}'")
            else:
                self._log(f"Auto-repair: creating conda env '{auto_env_name}'")
                conda_ok = await self.create_conda_env(auto_env_name, code_dir)
                if conda_ok:
                    conda_python = self.find_conda_python(auto_env_name)

            if conda_python:
                self._log(f"Auto-repair SUCCESS: using conda env '{auto_env_name}'")
                install_info = await self.install_requirements(conda_python, code_dir)
                runtime_validation = await self.validate_runtime(
                    conda_python,
                    code_dir,
                    execution_policy=execution_policy,
                )
                return {
                    "kind": "conda",
                    "python": conda_python,
                    "env_name": auto_env_name,
                    "created": True,
                    "auto_repaired": True,
                    "requirements_path": str(requirements_path) if requirements_path.exists() else "",
                    "environment_file": str(environment_file) if environment_file else "",
                    "dependency_install": install_info,
                    "runtime_validation": runtime_validation,
                    "runtime_validation_repair": {"status": "skipped", "actions": []},
                    "execution_policy": execution_policy.to_dict(),
                }
            self._log("Auto-repair: conda create also failed")

        # Strategy 2: If venv dir exists but is corrupted, remove and retry
        if failed_venv_dir.exists():
            self._log("Auto-repair: removing corrupted venv and retrying")
            try:
                shutil.rmtree(str(failed_venv_dir), ignore_errors=True)
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None,
                    lambda: venv.create(str(failed_venv_dir), with_pip=True),
                )
                is_windows = platform.system() == "Windows"
                python_path = failed_venv_dir / (
                    "Scripts/python.exe" if is_windows else "bin/python"
                )
                if python_path.exists():
                    self._log("Auto-repair SUCCESS: venv recreated after cleanup")
                    install_info = await self.install_requirements(str(python_path), code_dir)
                    runtime_validation = await self.validate_runtime(
                        str(python_path),
                        code_dir,
                        execution_policy=execution_policy,
                    )
                    return {
                        "kind": "venv",
                        "python": str(python_path),
                        "env_path": str(failed_venv_dir),
                        "created": True,
                        "auto_repaired": True,
                        "requirements_path": str(requirements_path) if requirements_path.exists() else "",
                        "environment_file": str(environment_file) if environment_file else "",
                        "dependency_install": install_info,
                        "runtime_validation": runtime_validation,
                        "runtime_validation_repair": {"status": "skipped", "actions": []},
                        "execution_policy": execution_policy.to_dict(),
                    }
            except Exception as retry_exc:
                self._log(f"Auto-repair: venv retry also failed: {retry_exc}")

        return None

    @staticmethod
    def _diagnose_env_failure(venv_dir: Path, exc: Exception) -> str:
        """Produce a human-readable diagnosis for environment creation failure."""
        reasons = []
        err_str = str(exc).lower()

        # Check disk space
        try:
            import shutil as _shutil
            usage = _shutil.disk_usage(str(venv_dir.parent))
            free_gb = usage.free / (1024 ** 3)
            if free_gb < 1.0:
                reasons.append(f"Low disk space: {free_gb:.1f} GB free")
        except Exception:
            pass

        # Check permissions
        parent = venv_dir.parent
        if parent.exists() and not os.access(str(parent), os.W_OK):
            reasons.append(f"No write permission to {parent}")

        # Check python3-venv package
        if "ensurepip" in err_str or "no module named" in err_str:
            reasons.append(
                "python3-venv package likely missing (install via: "
                "sudo apt install python3-venv)"
            )

        # Check if venv module is broken
        if "permission denied" in err_str:
            reasons.append("Permission denied — check file system permissions")

        if not reasons:
            reasons.append(f"Unknown cause: {str(exc)[:200]}")

        return "; ".join(reasons)

