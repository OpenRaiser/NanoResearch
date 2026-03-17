"""Shared runtime environment helpers for experiment execution."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import venv
from pathlib import Path
from typing import Any, Callable

from nanoresearch.config import ResearchConfig

from ._constants import (  # noqa: F401 — re-exported
    PACKAGE_IMPORT_ALIASES,
    MAX_RUNTIME_IMPORT_PROBES,
    MAX_RUNTIME_VALIDATION_REPAIR_PACKAGES,
    _TORCH_FAMILY_PACKAGES,
    _CUDA_DRIVER_TO_TORCH_TAG,
    _CUDA_DRIVER_TO_CONDA_CUDA,
)
from ._gpu_detect import (  # noqa: F401
    _find_conda,
    _detect_gpu_cuda,
    _probe_python_info,
)
from ._discovery import (  # noqa: F401
    discover_environments,
    _split_torch_requirements,
    _canonicalize_dependency_name,
)
from ._types import (  # noqa: F401
    DependencyInstallPlan,
    ProjectManifestSnapshot,
    ExperimentExecutionPolicy,
)
from ._conda import _CondaMixin
from ._install import _InstallMixin
from ._validation import _ValidationMixin
from ._manifests import _ManifestsMixin

__all__ = [
    "RuntimeEnvironmentManager",
    "ExperimentExecutionPolicy",
    "ProjectManifestSnapshot",
    "DependencyInstallPlan",
    "discover_environments",
]

logger = logging.getLogger(__name__)


class RuntimeEnvironmentManager(
    _CondaMixin,
    _InstallMixin,
    _ValidationMixin,
    _ManifestsMixin,
):
    """Prepare Python runtimes for local experiment execution."""

    def __init__(
        self,
        config: ResearchConfig,
        log_fn: Callable[[str], None] | None = None,
        session_label: str = "",
    ) -> None:
        self.config = config
        self._log = log_fn or (lambda _message: None)
        self._session_label = session_label

    # ------------------------------------------------------------------
    # User-specified environment resolution
    # ------------------------------------------------------------------

    def _resolve_user_python(self, spec: str) -> str | None:
        """Resolve a user-supplied environment spec to a Python executable.

        Accepts three formats:
        1. Direct path to python executable  (file exists & runnable)
        2. Path to an env directory           (look for Scripts/python.exe or bin/python)
        3. Conda environment name             (no path separators → conda lookup)

        Returns the absolute path to the python executable, or None if
        resolution fails.
        """
        spec = spec.strip()
        if not spec:
            return None

        p = Path(spec)

        # --- Case 1: direct python executable path ---
        if p.is_file():
            return str(p.resolve())

        # --- Case 2: directory → find python inside ---
        if p.is_dir():
            is_win = platform.system() == "Windows"
            candidates = (
                [p / "Scripts" / "python.exe", p / "python.exe"]
                if is_win
                else [p / "bin" / "python", p / "bin" / "python3"]
            )
            for c in candidates:
                if c.is_file():
                    return str(c.resolve())
            self._log(f"Directory '{spec}' exists but no python found inside")
            return None

        # --- Case 3: no path separators → treat as conda env name ---
        if "/" not in spec and "\\" not in spec:
            conda_python = self.find_conda_python(spec)
            if conda_python:
                return conda_python
            self._log(f"Conda env '{spec}' not found")
            return None

        # Path-like but doesn't exist
        self._log(f"experiment_python path not found: {spec}")
        return None

    def _validate_user_python(self, python: str) -> bool:
        """Quick smoke-test: can this python execute?"""
        try:
            proc = subprocess.run(
                [python, "-c", "import sys; print(sys.version)"],
                capture_output=True, text=True, timeout=15,
            )
            if proc.returncode == 0:
                version = (proc.stdout or "").strip().split("\n")[0]
                self._log(f"User python validated: {version}")
                return True
            self._log(f"User python failed smoke test: {proc.stderr}")
        except Exception as exc:
            self._log(f"User python not executable: {exc}")
        return False

    # ------------------------------------------------------------------
    # Interactive environment selection
    # ------------------------------------------------------------------

    def _interactive_env_select(self) -> str | None:
        """Discover environments and prompt the user to pick one.

        Returns the selected python path, or None if user skips / no envs.
        Also saves the choice to config.json for future runs.
        """
        self._log("No experiment_python configured — scanning environments...")
        envs = discover_environments()
        if not envs:
            self._log("No environments discovered, will auto-create")
            return None

        # Print table to terminal
        print("\n" + "=" * 60)
        print("  Available Python Environments")
        print("=" * 60)
        for i, env in enumerate(envs, 1):
            pkgs = ", ".join(env["packages"]) if env["packages"] else ""
            pkg_str = f"  [{pkgs}]" if pkgs else ""
            print(f"  [{i:>2}]  {env['name']:<30s}  "
                  f"Python {env['version']}{pkg_str}")
        print(f"  [ 0]  Skip (auto-create environment)")
        print("=" * 60)

        try:
            raw = input(f"Select environment [0-{len(envs)}]: ").strip()
            choice = int(raw) if raw else 0
        except (ValueError, EOFError, KeyboardInterrupt):
            print()
            return None

        if choice < 1 or choice > len(envs):
            return None

        selected = envs[choice - 1]
        python_path = selected["python"]

        # Save to config.json so future runs reuse this choice
        self._save_python_to_config(python_path)
        self._log(
            f"Selected: {selected['name']} → {python_path}  "
            f"(saved to config.json)"
        )
        return python_path

    def _save_python_to_config(self, python_path: str) -> None:
        """Persist experiment_python to ~/.nanobot/config.json."""
        cfg_path = Path.home() / ".nanobot" / "config.json"
        cfg_data: dict[str, Any] = {}
        if cfg_path.exists():
            try:
                cfg_data = json.loads(cfg_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        research = cfg_data.setdefault("research", {})
        research["experiment_python"] = python_path
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(
            json.dumps(cfg_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Backend resolution
    # ------------------------------------------------------------------

    def _resolve_backend(self) -> tuple[str, bool]:
        """Decide between 'conda' and 'venv' based on config + system state.

        Returns ``(backend, forced)`` where *forced* is True when the user
        explicitly set ``environment_backend`` to ``"conda"`` or ``"venv"``
        (as opposed to ``"auto"`` detection).
        """
        backend = (self.config.environment_backend or "auto").strip().lower()

        if backend == "venv":
            return "venv", True

        if backend == "conda":
            cmd = _find_conda()
            if cmd is None:
                raise RuntimeError(
                    "environment_backend='conda' but conda "
                    "is not installed.\n"
                    "Install Miniconda: https://docs.conda.io/en/latest/miniconda.html\n"
                    "Or set environment_backend='venv' in config.json."
                )
            return "conda", True

        # "auto" — prefer conda when available
        if _find_conda() is not None:
            self._log("Auto-detected conda — using conda backend")
            return "conda", False
        return "venv", False

    def _per_session_env_name(self) -> str:
        """Deterministic conda env name for the current session.

        Uses ``nanoresearch_{sanitized_label}`` so that resume reuses the
        same env (idempotent).
        """
        import hashlib as _hl

        label = re.sub(r'[^A-Za-z0-9_-]', '_', self._session_label)[:30].strip('_')
        if not label:
            label = (
                _hl.md5(self._session_label.encode()).hexdigest()[:10]
                if self._session_label else "default"
            )
        return f"nanoresearch_{label}"

    # ------------------------------------------------------------------
    # Per-session conda environment
    # ------------------------------------------------------------------

    async def prepare(self, code_dir: Path, *, force_isolated: bool = False) -> dict[str, Any]:
        requirements_path = code_dir / "requirements.txt"
        environment_file = self._find_environment_file(code_dir)
        execution_policy = self.build_execution_policy(code_dir)

        # ----- Priority 0: user-managed environment (experiment_python) ---
        user_spec = (self.config.experiment_python or "").strip()
        if user_spec and not force_isolated:
            python = self._resolve_user_python(user_spec)
            if python and self._validate_user_python(python):
                self._log(f"Using user-specified environment: {python}")
                return {
                    "kind": "user",
                    "python": python,
                    "created": False,
                    "requirements_path": str(requirements_path) if requirements_path.exists() else "",
                    "environment_file": str(environment_file) if environment_file else "",
                    "dependency_install": {"status": "skipped", "reason": "user-managed environment"},
                    "runtime_validation": {"status": "ready", "source": "user-managed"},
                    "runtime_validation_repair": {"status": "skipped", "actions": []},
                    "execution_policy": execution_policy.to_dict(),
                }
            if python:
                self._log(f"User python '{user_spec}' resolved to '{python}' but failed validation, falling back")
            else:
                self._log(f"Could not resolve experiment_python='{user_spec}', falling back")

        # ----- Priority 0.5: interactive env selection (TTY only) ---------
        if not user_spec and not force_isolated and sys.stdin.isatty():
            selected = self._interactive_env_select()
            if selected:
                self._log(f"Using interactively selected environment: {selected}")
                return {
                    "kind": "user",
                    "python": selected,
                    "created": False,
                    "requirements_path": str(requirements_path) if requirements_path.exists() else "",
                    "environment_file": str(environment_file) if environment_file else "",
                    "dependency_install": {"status": "skipped", "reason": "user-managed environment"},
                    "runtime_validation": {"status": "ready", "source": "user-selected"},
                    "runtime_validation_repair": {"status": "skipped", "actions": []},
                    "execution_policy": execution_policy.to_dict(),
                }

        # ----- Priority 1: explicit named conda env from config ----------
        conda_env = self.config.experiment_conda_env.strip()
        if conda_env and not force_isolated:
            conda_python = self.find_conda_python(conda_env)
            if conda_python:
                self._log(f"Using existing conda env '{conda_env}': {conda_python}")
                install_info = await self.install_requirements(conda_python, code_dir)
                runtime_validation = await self.validate_runtime(
                    conda_python,
                    code_dir,
                    execution_policy=execution_policy,
                )
                return {
                    "kind": "conda",
                    "python": conda_python,
                    "env_name": conda_env,
                    "requirements_path": str(requirements_path) if requirements_path.exists() else "",
                    "environment_file": str(environment_file) if environment_file else "",
                    "dependency_install": install_info,
                    "runtime_validation": runtime_validation,
                    "runtime_validation_repair": {"status": "skipped", "actions": []},
                    "execution_policy": execution_policy.to_dict(),
                }
            if self.config.auto_create_env and _find_conda() is not None:
                created = await self.create_conda_env(conda_env, code_dir)
                if created:
                    conda_python = self.find_conda_python(conda_env)
                    if conda_python:
                        install_info = await self.install_requirements(conda_python, code_dir)
                        runtime_validation = await self.validate_runtime(
                            conda_python, code_dir, execution_policy=execution_policy,
                        )
                        return {
                            "kind": "conda",
                            "python": conda_python,
                            "env_name": conda_env,
                            "created": True,
                            "requirements_path": str(requirements_path) if requirements_path.exists() else "",
                            "environment_file": str(environment_file) if environment_file else "",
                            "dependency_install": install_info,
                            "runtime_validation": runtime_validation,
                            "runtime_validation_repair": {"status": "skipped", "actions": []},
                            "execution_policy": execution_policy.to_dict(),
                        }
            self._log(f"Conda env '{conda_env}' not found, falling back to venv")

        # ----- Priority 2: auto / forced backend selection ---------------
        if not force_isolated:
            backend, backend_forced = self._resolve_backend()
        else:
            backend, backend_forced = "venv", False

        if backend == "conda":
            env_info = await self._create_per_session_conda_env(code_dir, execution_policy)
            if env_info:
                return env_info
            if backend_forced:
                raise RuntimeError(
                    "environment_backend='conda' but conda env creation failed.\n"
                    "Check that conda is working correctly, or set "
                    "environment_backend='auto' to allow venv fallback."
                )
            # Auto-detected conda failed — graceful degradation to venv
            self._log("Per-session conda env failed, falling back to venv")

        # ----- venv path (default / fallback) ----------------------------
        # Never fall back to sys.executable to avoid polluting the CLI
        # Python with experiment dependencies.
        venv_dir = code_dir / ".venv"
        is_windows = platform.system() == "Windows"
        python_path = venv_dir / ("Scripts/python.exe" if is_windows else "bin/python")
        created = False
        recreated = False

        if not python_path.exists():
            self._log(f"Creating isolated venv at {venv_dir} ...")
            loop = asyncio.get_running_loop()
            try:
                await loop.run_in_executor(
                    None,
                    lambda: venv.create(str(venv_dir), with_pip=True),
                )
                created = True
                self._log(f"Venv created (python: {python_path})")
            except (OSError, subprocess.CalledProcessError) as venv_exc:
                # ── Auto-repair: venv failed, try conda create ──
                self._log(
                    f"Venv creation failed: {venv_exc}. "
                    "Attempting auto-repair via conda..."
                )
                repaired = await self._auto_repair_env(
                    code_dir, venv_dir, execution_policy,
                    requirements_path, environment_file,
                )
                if repaired is not None:
                    return repaired
                # All repair strategies exhausted
                diag = self._diagnose_env_failure(venv_dir, venv_exc)
                raise RuntimeError(
                    f"Environment creation failed and auto-repair exhausted.\n"
                    f"Diagnosis: {diag}\n"
                    f"Original venv error: {venv_exc}\n"
                    "Solutions:\n"
                    "  1. Set 'experiment_conda_env' to a valid conda env name in config.json\n"
                    "  2. Install python3-venv: sudo apt install python3-venv\n"
                    "  3. Ensure sufficient disk space and write permissions"
                ) from venv_exc
        else:
            self._log(f"Reusing existing venv at {venv_dir}")

        install_info = await self.install_requirements(str(python_path), code_dir)
        runtime_validation = await self.validate_runtime(
            str(python_path),
            code_dir,
            execution_policy=execution_policy,
        )
        validation_repair = await self._repair_runtime_validation(
            kind="venv",
            python=str(python_path),
            code_dir=code_dir,
            execution_policy=execution_policy,
            validation=runtime_validation,
            env_dir=venv_dir,
            created=created,
        )
        runtime_validation = validation_repair["validation"]
        python_path = Path(str(validation_repair["python"]))
        install_info = validation_repair.get("dependency_install", install_info)
        recreated = bool(validation_repair.get("recreated", False))
        return {
            "kind": "venv",
            "python": str(python_path),
            "env_path": str(venv_dir),
            "created": created or recreated,
            "recreated": recreated,
            "requirements_path": str(requirements_path) if requirements_path.exists() else "",
            "environment_file": str(environment_file) if environment_file else "",
            "dependency_install": install_info,
            "runtime_validation": runtime_validation,
            "runtime_validation_repair": validation_repair["repair"],
            "execution_policy": execution_policy.to_dict(),
        }
