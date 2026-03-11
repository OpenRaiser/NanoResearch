"""Shared runtime environment helpers for experiment execution."""

from __future__ import annotations

import asyncio
import configparser
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import venv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

from nanoresearch.config import ResearchConfig

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    tomllib = None


PACKAGE_IMPORT_ALIASES = {
    "pyyaml": "yaml",
    "opencv-python": "cv2",
    "pillow": "PIL",
    "scikit-learn": "sklearn",
    "biopython": "Bio",
    "python-dateutil": "dateutil",
    "beautifulsoup4": "bs4",
}
MAX_RUNTIME_IMPORT_PROBES = 50
MAX_RUNTIME_VALIDATION_REPAIR_PACKAGES = 50

# PyTorch-family packages that need special index URL handling for CUDA support
_TORCH_FAMILY_PACKAGES = {"torch", "torchvision", "torchaudio", "torchtext"}

# CUDA driver version → best available PyTorch CUDA wheel tag
# nvidia-smi reports the max CUDA version the driver supports.
# PyTorch ships wheels for specific CUDA toolkit versions (cuXYZ).
# We map driver-reported CUDA version to the newest compatible wheel tag.
_CUDA_DRIVER_TO_TORCH_TAG: list[tuple[tuple[int, int], str]] = [
    # (min_cuda_version, torch_index_tag)
    # Ordered newest → oldest so we pick the best match.
    ((12, 8), "cu128"),
    ((12, 6), "cu126"),
    ((12, 4), "cu124"),
    ((12, 1), "cu121"),
    ((11, 8), "cu118"),
]

# CUDA driver version → conda pytorch-cuda metapackage version (e.g. "12.4")
_CUDA_DRIVER_TO_CONDA_CUDA: list[tuple[tuple[int, int], str]] = [
    ((12, 8), "12.8"),
    ((12, 6), "12.6"),
    ((12, 4), "12.4"),
    ((12, 1), "12.1"),
    ((11, 8), "11.8"),
]


def _find_conda() -> str | None:
    """Return ``"conda"`` if conda is installed, else ``None``."""
    return "conda" if shutil.which("conda") else None


def _detect_gpu_cuda() -> dict[str, Any] | None:
    """Detect NVIDIA GPU and CUDA driver version via nvidia-smi.

    Returns a dict with keys: gpu_name, driver_version, cuda_version (tuple),
    cuda_version_str, torch_index_url.  Returns None if no NVIDIA GPU is found.
    """
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None

    try:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=name,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None
        gpu_line = (result.stdout or "").strip().splitlines()[0]
        parts = [p.strip() for p in gpu_line.split(",")]
        gpu_name = parts[0] if parts else "Unknown"
        driver_version = parts[1] if len(parts) > 1 else ""
    except Exception:
        return None

    # Parse CUDA version from nvidia-smi header output
    cuda_version: tuple[int, int] | None = None
    try:
        result2 = subprocess.run(
            [nvidia_smi],
            capture_output=True, text=True, timeout=10,
        )
        # Look for "CUDA Version: X.Y" in the table header
        m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", result2.stdout or "")
        if m:
            cuda_version = (int(m.group(1)), int(m.group(2)))
    except Exception:
        pass

    if cuda_version is None:
        return None

    # Find best matching torch CUDA wheel tag
    torch_tag = ""
    for min_ver, tag in _CUDA_DRIVER_TO_TORCH_TAG:
        if cuda_version >= min_ver:
            torch_tag = tag
            break

    if not torch_tag:
        # CUDA version too old for any known PyTorch CUDA build
        logger.warning(
            "GPU detected (%s, CUDA %s.%s) but CUDA version is too old for "
            "any known PyTorch CUDA wheel. Falling back to CPU-only torch.",
            gpu_name, cuda_version[0], cuda_version[1],
        )
        return None

    return {
        "gpu_name": gpu_name,
        "driver_version": driver_version,
        "cuda_version": cuda_version,
        "cuda_version_str": f"{cuda_version[0]}.{cuda_version[1]}",
        "torch_tag": torch_tag,
        "torch_index_url": f"https://download.pytorch.org/whl/{torch_tag}",
    }


def _split_torch_requirements(requirements: list[str]) -> tuple[list[str], list[str]]:
    """Split a list of requirement specifiers into torch-family and non-torch.

    Returns (torch_specs, other_specs).  torch_specs are raw specifier strings
    like 'torch>=2.0' that should be installed from the CUDA index URL.
    """
    torch_specs: list[str] = []
    other_specs: list[str] = []
    for line in requirements:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        canonical = _canonicalize_dependency_name(stripped)
        if canonical and canonical in _TORCH_FAMILY_PACKAGES:
            torch_specs.append(stripped)
        else:
            other_specs.append(stripped)
    return torch_specs, other_specs


def _canonicalize_dependency_name(raw_value: str) -> str | None:
    """Normalize a dependency specifier to its package name."""
    candidate = str(raw_value or "").strip()
    if not candidate:
        return None

    if candidate.startswith(("-r ", "--requirement ", "-c ", "--constraint ")):
        return None
    if candidate.startswith(("-f ", "--find-links ", "--index-url ", "--extra-index-url ")):
        return None
    if candidate.startswith("--"):
        return None

    if candidate.startswith(("-e ", "--editable ")):
        _, _, editable_target = candidate.partition(" ")
        candidate = editable_target.strip()

    egg_marker = "#egg="
    if egg_marker in candidate:
        candidate = candidate.split(egg_marker, 1)[1].strip()
    elif "#" in candidate:
        candidate = candidate.split("#", 1)[0].strip()

    if " @" in candidate:
        candidate = candidate.split(" @", 1)[0].strip()
    candidate = candidate.split(";", 1)[0].strip()
    if not candidate or candidate in {".", ".."}:
        return None
    if candidate.startswith((".", "/", "\\")):
        return None

    match = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", candidate)
    if not match:
        return None

    normalized = re.sub(r"[-_.]+", "-", match.group(0)).lower()
    return normalized or None


@dataclass(frozen=True)
class DependencyInstallPlan:
    """Concrete pip-install strategy for a generated experiment project."""

    source: str
    args: list[str]
    manifest_path: str
    fallback_args: list[str] | None = None


@dataclass(frozen=True)
class ProjectManifestSnapshot:
    """Local manifest inspection snapshot shared across execution backends."""

    manifest_source: str
    manifest_path: str
    environment_source: str
    environment_path: str
    install_source: str
    install_manifest_path: str
    install_kind: str

    def to_dict(self) -> dict[str, str]:
        return {
            "manifest_source": self.manifest_source,
            "manifest_path": self.manifest_path,
            "environment_source": self.environment_source,
            "environment_path": self.environment_path,
            "install_source": self.install_source,
            "install_manifest_path": self.install_manifest_path,
            "install_kind": self.install_kind,
        }


@dataclass(frozen=True)
class ExperimentExecutionPolicy:
    """Central execution-time remediation policy for local experiment runs."""

    install_plan: DependencyInstallPlan | None
    manifest_source: str
    manifest_path: str
    declared_dependencies: frozenset[str]
    runtime_auto_install_enabled: bool
    runtime_auto_install_allowlist: frozenset[str]
    max_runtime_auto_installs: int
    max_nltk_downloads: int

    @staticmethod
    def _count_actions(
        fix_history: list[dict[str, Any]] | None,
        *,
        prefix: str,
    ) -> int:
        if not fix_history:
            return 0

        total = 0
        for entry in fix_history:
            for item in entry.get("fixed_files", []):
                if isinstance(item, str) and item.startswith(prefix):
                    total += 1
        return total

    def remaining_runtime_auto_installs(
        self,
        fix_history: list[dict[str, Any]] | None = None,
    ) -> int:
        used = self._count_actions(fix_history, prefix="pip:")
        return max(0, int(self.max_runtime_auto_installs) - used)

    def remaining_nltk_downloads(
        self,
        fix_history: list[dict[str, Any]] | None = None,
    ) -> int:
        used = self._count_actions(fix_history, prefix="nltk:")
        return max(0, int(self.max_nltk_downloads) - used)

    def allows_runtime_package(
        self,
        package_name: str,
        *,
        module_name: str = "",
        aliases: dict[str, str] | None = None,
    ) -> bool:
        if not self.runtime_auto_install_enabled:
            return False

        normalized_package = _canonicalize_dependency_name(package_name)
        if not normalized_package:
            return False
        if normalized_package in self.declared_dependencies:
            return True
        if normalized_package in self.runtime_auto_install_allowlist:
            return True

        normalized_module = _canonicalize_dependency_name(module_name)
        if not normalized_module or not aliases:
            return False
        alias_target = aliases.get(normalized_module)
        if not alias_target:
            return False
        return _canonicalize_dependency_name(alias_target) == normalized_package

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_source": self.manifest_source,
            "manifest_path": self.manifest_path,
            "declared_dependencies": sorted(self.declared_dependencies),
            "runtime_auto_install_enabled": self.runtime_auto_install_enabled,
            "runtime_auto_install_allowlist": sorted(self.runtime_auto_install_allowlist),
            "max_runtime_auto_installs": self.max_runtime_auto_installs,
            "max_nltk_downloads": self.max_nltk_downloads,
        }


class RuntimeEnvironmentManager:
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
        cmd = _find_conda() or "conda"
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
        gpu_info = _detect_gpu_cuda()
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
        cmd = _find_conda()
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
        cmd = _find_conda() or "conda"
        try:
            proc = subprocess.run(
                [cmd, "env", "remove", "-y", "-n", env_name],
                capture_output=True, text=True, timeout=120,
            )
            return proc.returncode == 0
        except Exception:
            return False

    async def prepare(self, code_dir: Path, *, force_isolated: bool = False) -> dict[str, Any]:
        requirements_path = code_dir / "requirements.txt"
        environment_file = self._find_environment_file(code_dir)
        execution_policy = self.build_execution_policy(code_dir)

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

    @staticmethod
    def _package_import_candidates(package_name: str) -> list[str]:
        normalized = _canonicalize_dependency_name(package_name)
        if not normalized:
            return []

        candidates: list[str] = []
        alias = PACKAGE_IMPORT_ALIASES.get(normalized)
        if alias:
            candidates.append(alias)

        direct_candidate = normalized.replace("-", "_")
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", direct_candidate):
            candidates.append(direct_candidate)

        seen: set[str] = set()
        deduped: list[str] = []
        for candidate in candidates:
            if candidate not in seen:
                deduped.append(candidate)
                seen.add(candidate)
        return deduped

    @staticmethod
    def _validation_status(details: dict[str, Any] | None, key: str) -> str:
        if not isinstance(details, dict):
            return ""
        probe = details.get(key)
        if not isinstance(probe, dict):
            return ""
        return str(probe.get("status") or "").strip()

    @classmethod
    def _validation_requires_venv_rebuild(cls, validation: dict[str, Any] | None) -> bool:
        if not isinstance(validation, dict):
            return False
        return (
            cls._validation_status(validation, "python_smoke") == "failed"
            or cls._validation_status(validation, "pip_probe") == "failed"
        )

    @staticmethod
    def _failed_import_packages(validation: dict[str, Any] | None) -> list[str]:
        if not isinstance(validation, dict):
            return []
        import_probe = validation.get("import_probe")
        if not isinstance(import_probe, dict):
            return []

        packages: list[str] = []
        for failure in import_probe.get("failures", []) or []:
            if not isinstance(failure, dict):
                continue
            package_name = _canonicalize_dependency_name(str(failure.get("package") or ""))
            if package_name and package_name not in packages:
                packages.append(package_name)
        return packages

    @staticmethod
    def _extract_requirement_dependency_specs(requirements_file: Path) -> list[str]:
        try:
            lines = requirements_file.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []

        specs: list[str] = []
        for raw_line in lines:
            candidate = raw_line.strip()
            if not candidate or candidate.startswith("#"):
                continue
            if "#egg=" not in candidate and "#" in candidate:
                candidate = candidate.split("#", 1)[0].strip()
            if not candidate:
                continue
            if candidate.startswith(("-r ", "--requirement ", "-c ", "--constraint ")):
                continue
            if candidate.startswith(("-f ", "--find-links ", "--index-url ", "--extra-index-url ")):
                continue
            if candidate.startswith("--"):
                continue
            if candidate.startswith(("-e ", "--editable ")) or "://" in candidate:
                continue
            if _canonicalize_dependency_name(candidate):
                specs.append(candidate)
        return specs

    @classmethod
    def _repairable_dependency_spec_index(
        cls,
        code_dir: Path,
        execution_policy: ExperimentExecutionPolicy | None,
    ) -> dict[str, str]:
        if execution_policy is None or execution_policy.install_plan is None:
            return {}

        source = execution_policy.install_plan.source
        if source not in {"requirements.txt", "environment.yml", "environment.yaml"}:
            return {}

        return cls.collect_repairable_dependency_specs(code_dir)

    @classmethod
    def collect_declared_dependency_names(cls, code_dir: Path) -> list[str]:
        return sorted(cls._collect_declared_dependency_names(code_dir))

    @classmethod
    def collect_repairable_dependency_specs(cls, code_dir: Path) -> dict[str, str]:
        install_plan = cls._select_install_plan(code_dir)
        if install_plan is None:
            return {}

        if install_plan.source == "requirements.txt":
            manifest_file = code_dir / "requirements.txt"
            specs = cls._extract_requirement_dependency_specs(manifest_file)
        elif install_plan.source in {"environment.yml", "environment.yaml"}:
            environment_file = cls._find_environment_file(code_dir)
            specs = cls._extract_pip_dependencies(environment_file) if environment_file is not None else []
        else:
            return {}

        index: dict[str, str] = {}
        for spec in specs:
            normalized = _canonicalize_dependency_name(spec)
            if normalized and normalized not in index:
                index[normalized] = spec
        return index

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

    async def _repair_runtime_validation(
        self,
        *,
        kind: str,
        python: str,
        code_dir: Path,
        execution_policy: ExperimentExecutionPolicy,
        validation: dict[str, Any],
        env_dir: Path | None = None,
        created: bool = False,
    ) -> dict[str, Any]:
        repair_actions: list[dict[str, Any]] = []
        current_python = str(python)
        current_validation = dict(validation)
        current_install_info: dict[str, Any] | None = None
        recreated = False

        if (
            kind == "venv"
            and env_dir is not None
            and not created
            and self._validation_requires_venv_rebuild(current_validation)
        ):
            recreate_result = await self._recreate_venv(env_dir)
            repair_actions.append(
                {
                    "kind": "recreate_venv",
                    **recreate_result,
                }
            )
            if recreate_result.get("status") == "applied":
                current_python = str(recreate_result.get("python") or current_python)
                recreated = True
                current_install_info = await self.install_requirements(current_python, code_dir)
                repair_actions.append(
                    {
                        "kind": "reinstall_manifest",
                        **current_install_info,
                    }
                )
                current_validation = await self.validate_runtime(
                    current_python,
                    code_dir,
                    execution_policy=execution_policy,
                )

        failed_imports = self._failed_import_packages(current_validation)
        if failed_imports:
            spec_index = self._repairable_dependency_spec_index(code_dir, execution_policy)
            repair_specs: list[str] = []
            unresolved: list[str] = []
            for package_name in failed_imports:
                spec = spec_index.get(package_name)
                if spec and spec not in repair_specs:
                    repair_specs.append(spec)
                else:
                    unresolved.append(package_name)
                if len(repair_specs) >= MAX_RUNTIME_VALIDATION_REPAIR_PACKAGES:
                    break

            if repair_specs:
                targeted_install = await self.install_dependency_specs(
                    current_python,
                    code_dir,
                    repair_specs,
                    source="runtime_validation_import_repair",
                )
                repair_actions.append(
                    {
                        "kind": "import_repair_install",
                        **targeted_install,
                    }
                )
                if targeted_install.get("status") == "installed":
                    current_validation = await self.validate_runtime(
                        current_python,
                        code_dir,
                        execution_policy=execution_policy,
                    )
            elif unresolved:
                repair_actions.append(
                    {
                        "kind": "import_repair_skipped",
                        "status": "skipped",
                        "packages": unresolved,
                    }
                )

        if not repair_actions:
            repair_summary = {
                "status": "skipped",
                "actions": [],
            }
        else:
            final_status = str(current_validation.get("status") or "").strip()
            if final_status == "ready":
                summary_status = "applied"
            elif any(action.get("status") == "failed" for action in repair_actions):
                summary_status = "failed"
            else:
                summary_status = "partial"
            repair_summary = {
                "status": summary_status,
                "actions": repair_actions,
            }

        result: dict[str, Any] = {
            "python": current_python,
            "validation": current_validation,
            "repair": repair_summary,
            "recreated": recreated,
        }
        if current_install_info is not None:
            result["dependency_install"] = current_install_info
        return result

    def _select_import_probe_targets(
        self,
        execution_policy: ExperimentExecutionPolicy | None,
    ) -> tuple[list[dict[str, str]], str]:
        if execution_policy is None:
            return [], "no_execution_policy"

        install_plan = execution_policy.install_plan
        if install_plan is None:
            return [], "no_install_plan"
        if install_plan.source not in {"requirements.txt", "environment.yml", "environment.yaml"}:
            return [], "install_source_not_probe_safe"

        targets: list[dict[str, str]] = []
        for package_name in sorted(execution_policy.declared_dependencies):
            import_candidates = self._package_import_candidates(package_name)
            if not import_candidates:
                continue
            targets.append({"package": package_name, "module": import_candidates[0]})
            if len(targets) >= MAX_RUNTIME_IMPORT_PROBES:
                break

        if not targets:
            return [], "no_probeable_dependencies"
        return targets, ""

    async def validate_runtime(
        self,
        python: str,
        code_dir: Path,
        *,
        execution_policy: ExperimentExecutionPolicy | None = None,
    ) -> dict[str, Any]:
        """Validate that the selected runtime can execute and import key dependencies."""
        loop = asyncio.get_running_loop()

        try:
            smoke_result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    [
                        python,
                        "-c",
                        (
                            "import json, sys; "
                            "print(json.dumps({'executable': sys.executable, 'version': sys.version.split()[0]}))"
                        ),
                    ],
                    cwd=str(code_dir),
                    capture_output=True,
                    text=True,
                    timeout=30,
                ),
            )
        except Exception as exc:
            self._log(f"Runtime smoke probe failed to start for {python}: {exc}")
            return {
                "status": "failed",
                "python_smoke": {
                    "status": "failed",
                    "error": str(exc),
                },
                "pip_probe": {"status": "skipped"},
                "import_probe": {"status": "skipped"},
            }

        smoke_stdout = (smoke_result.stdout or "").strip()
        smoke_stderr = (smoke_result.stderr or "").strip()
        smoke_payload: dict[str, Any] = {}
        if smoke_result.returncode == 0 and smoke_stdout:
            try:
                smoke_payload = json.loads(smoke_stdout.splitlines()[-1])
            except json.JSONDecodeError:
                smoke_payload = {}

        python_smoke = {
            "status": "passed" if smoke_result.returncode == 0 else "failed",
            "returncode": smoke_result.returncode,
            "stderr": smoke_stderr[:300],
            "executable": str(smoke_payload.get("executable") or python),
            "version": str(smoke_payload.get("version") or ""),
        }
        if smoke_result.returncode != 0:
            self._log(
                f"Runtime validation failed for {python}: rc={smoke_result.returncode}, stderr={smoke_stderr[:200]}"
            )
            return {
                "status": "failed",
                "python_smoke": python_smoke,
                "pip_probe": {"status": "skipped"},
                "import_probe": {"status": "skipped"},
            }

        try:
            pip_result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    [python, "-m", "pip", "--version"],
                    cwd=str(code_dir),
                    capture_output=True,
                    text=True,
                    timeout=30,
                ),
            )
        except Exception as exc:
            pip_probe = {
                "status": "failed",
                "error": str(exc),
            }
        else:
            pip_probe = {
                "status": "passed" if pip_result.returncode == 0 else "failed",
                "returncode": pip_result.returncode,
                "version": (pip_result.stdout or "").strip()[:200],
                "stderr": (pip_result.stderr or "").strip()[:300],
            }

        probe_targets, skipped_reason = self._select_import_probe_targets(execution_policy)
        if not probe_targets:
            import_probe = {
                "status": "skipped",
                "targets": [],
                "failures": [],
                "skipped_reason": skipped_reason,
            }
        else:
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
            try:
                import_result = await loop.run_in_executor(
                    None,
                    lambda: subprocess.run(
                        [python, "-c", import_script],
                        cwd=str(code_dir),
                        capture_output=True,
                        text=True,
                        timeout=60,
                    ),
                )
            except Exception as exc:
                import_probe = {
                    "status": "failed",
                    "targets": list(probe_targets),
                    "failures": [{"package": "", "module": "", "error": str(exc)}],
                }
            else:
                import_stdout = (import_result.stdout or "").strip()
                parsed_results: list[dict[str, Any]] = []
                if import_stdout:
                    try:
                        parsed_payload = json.loads(import_stdout.splitlines()[-1])
                    except json.JSONDecodeError:
                        parsed_payload = {}
                    results_value = parsed_payload.get("results") if isinstance(parsed_payload, dict) else None
                    if isinstance(results_value, list):
                        parsed_results = [item for item in results_value if isinstance(item, dict)]
                failures = [item for item in parsed_results if item.get("status") != "passed"]
                if import_result.returncode != 0 and not failures:
                    failures = [
                        {
                            "package": "",
                            "module": "",
                            "error": (import_result.stderr or "").strip()[:300],
                        }
                    ]
                import_probe = {
                    "status": "passed" if not failures and import_result.returncode == 0 else "partial",
                    "targets": list(probe_targets),
                    "results": parsed_results,
                    "failures": failures,
                    "stderr": (import_result.stderr or "").strip()[:300],
                }

        overall_status = "ready"
        if pip_probe.get("status") != "passed":
            overall_status = "partial"
        if import_probe.get("status") in {"partial", "failed"}:
            overall_status = "partial"

        validation = {
            "status": overall_status,
            "python_smoke": python_smoke,
            "pip_probe": pip_probe,
            "import_probe": import_probe,
        }
        self._log(
            "Runtime validation "
            f"{overall_status} for {python_smoke.get('executable', python)} "
            f"(pip={pip_probe.get('status')}, imports={import_probe.get('status')})"
        )
        return validation

    def build_execution_policy(self, code_dir: Path) -> ExperimentExecutionPolicy:
        """Build a centralized execution/remediation policy for a project."""
        allowlist = {
            normalized
            for item in self.config.runtime_auto_install_allowlist
            if (normalized := _canonicalize_dependency_name(item))
        }
        manifest_snapshot = self.inspect_project_manifests(code_dir)
        return ExperimentExecutionPolicy(
            install_plan=self._select_install_plan(code_dir),
            manifest_source=manifest_snapshot.manifest_source,
            manifest_path=manifest_snapshot.manifest_path,
            declared_dependencies=frozenset(self._collect_declared_dependency_names(code_dir)),
            runtime_auto_install_enabled=bool(self.config.runtime_auto_install_enabled),
            runtime_auto_install_allowlist=frozenset(allowlist),
            max_runtime_auto_installs=max(0, int(self.config.runtime_auto_install_max_packages)),
            max_nltk_downloads=max(0, int(self.config.runtime_auto_install_max_nltk_downloads)),
        )

    @staticmethod
    def find_conda_python(env_name: str) -> str | None:
        """Find the Python executable for a named conda env."""
        conda_cmd = _find_conda() or "conda"
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
            gpu_info = _detect_gpu_cuda()
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
        if _find_conda() is not None:
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

    async def create_conda_env(self, env_name: str, code_dir: Path) -> bool:
        """Create a conda environment when requested and missing."""
        env_file = self._find_environment_file(code_dir)
        conda_cmd = _find_conda() or "conda"
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

    @staticmethod
    def _find_environment_file(code_dir: Path) -> Path | None:
        """Find the first supported Conda environment manifest."""
        for name in ("environment.yml", "environment.yaml"):
            candidate = code_dir / name
            if candidate.exists():
                return candidate
        return None

    @classmethod
    def _select_install_plan(cls, code_dir: Path) -> DependencyInstallPlan | None:
        """Choose the best available pip install strategy for a project."""
        requirements_path = code_dir / "requirements.txt"
        if requirements_path.exists():
            return DependencyInstallPlan(
                source="requirements.txt",
                args=["-r", str(requirements_path)],
                manifest_path=str(requirements_path),
            )

        environment_file = cls._find_environment_file(code_dir)
        if environment_file is not None:
            pip_dependencies = cls._extract_pip_dependencies(environment_file)
            if pip_dependencies:
                return DependencyInstallPlan(
                    source=environment_file.name,
                    args=pip_dependencies,
                    manifest_path=str(environment_file),
                )

        pyproject_file = code_dir / "pyproject.toml"
        if pyproject_file.exists():
            return DependencyInstallPlan(
                source="pyproject.toml",
                args=["-e", "."],
                fallback_args=["."],
                manifest_path=str(pyproject_file),
            )

        setup_py = code_dir / "setup.py"
        if setup_py.exists():
            return DependencyInstallPlan(
                source="setup.py",
                args=["-e", "."],
                fallback_args=["."],
                manifest_path=str(setup_py),
            )

        setup_cfg = code_dir / "setup.cfg"
        if setup_cfg.exists():
            return DependencyInstallPlan(
                source="setup.cfg",
                args=["-e", "."],
                fallback_args=["."],
                manifest_path=str(setup_cfg),
            )

        return None

    @classmethod
    def _detect_manifest_reference(cls, code_dir: Path) -> tuple[str, str]:
        requirements_path = code_dir / "requirements.txt"
        if requirements_path.exists():
            return "requirements.txt", str(requirements_path)

        environment_file = cls._find_environment_file(code_dir)
        if environment_file is not None:
            return environment_file.name, str(environment_file)

        for name in ("pyproject.toml", "setup.py", "setup.cfg"):
            candidate = code_dir / name
            if candidate.exists():
                return name, str(candidate)

        return "", ""

    @classmethod
    def inspect_project_manifests(cls, code_dir: Path) -> ProjectManifestSnapshot:
        """Inspect local project manifests in the same priority order as local execution."""
        manifest_source, manifest_path = cls._detect_manifest_reference(code_dir)
        environment_file = cls._find_environment_file(code_dir)
        install_plan = cls._select_install_plan(code_dir)

        environment_source = environment_file.name if environment_file is not None else ""
        environment_path = str(environment_file) if environment_file is not None else ""

        install_source = ""
        install_manifest_path = ""
        install_kind = ""
        if install_plan is not None:
            install_source = install_plan.source
            install_manifest_path = install_plan.manifest_path
            if install_plan.source == "requirements.txt":
                install_kind = "requirements"
            elif install_plan.source in {"environment.yml", "environment.yaml"}:
                install_kind = "environment"
            elif install_plan.args[:2] == ["-e", "."]:
                install_kind = "editable"
            else:
                install_kind = "package"

        return ProjectManifestSnapshot(
            manifest_source=manifest_source,
            manifest_path=manifest_path,
            environment_source=environment_source,
            environment_path=environment_path,
            install_source=install_source,
            install_manifest_path=install_manifest_path,
            install_kind=install_kind,
        )

    @classmethod
    def _collect_declared_dependency_names(cls, code_dir: Path) -> set[str]:
        declared: set[str] = set()

        requirements_path = code_dir / "requirements.txt"
        if requirements_path.exists():
            declared.update(cls._extract_requirement_dependency_names(requirements_path))

        environment_file = cls._find_environment_file(code_dir)
        if environment_file is not None:
            for dependency in cls._extract_pip_dependencies(environment_file):
                normalized = _canonicalize_dependency_name(dependency)
                if normalized:
                    declared.add(normalized)

        pyproject_file = code_dir / "pyproject.toml"
        if pyproject_file.exists():
            declared.update(cls._extract_pyproject_dependency_names(pyproject_file))

        setup_cfg = code_dir / "setup.cfg"
        if setup_cfg.exists():
            declared.update(cls._extract_setup_cfg_dependency_names(setup_cfg))

        return declared

    @classmethod
    def _extract_requirement_dependency_names(cls, requirements_file: Path) -> list[str]:
        try:
            lines = requirements_file.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        return cls._normalize_dependency_specs(lines)

    @staticmethod
    def _extract_pip_dependencies(environment_file: Path) -> list[str]:
        """Extract pip-installable dependencies from environment.yml."""
        if not environment_file.exists():
            return []

        try:
            lines = environment_file.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []

        dependencies: list[str] = []
        in_dependencies_block = False
        dependencies_indent = 0
        in_pip_block = False
        pip_indent = 0
        for raw_line in lines:
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            indent = len(raw_line) - len(raw_line.lstrip())
            if stripped == "dependencies:":
                in_dependencies_block = True
                dependencies_indent = indent
                in_pip_block = False
                continue
            if in_dependencies_block and indent <= dependencies_indent and not stripped.startswith("- "):
                in_dependencies_block = False
                in_pip_block = False

            if not in_dependencies_block:
                continue

            if stripped in {"- pip:", "pip:"}:
                in_pip_block = True
                pip_indent = indent
                continue

            if in_pip_block:
                if indent <= pip_indent:
                    in_pip_block = False
                elif stripped.startswith("- "):
                    dependency = stripped[2:].strip()
                    if dependency:
                        dependencies.append(dependency)
                    continue

            if not in_pip_block and stripped.startswith("- "):
                dependency = stripped[2:].strip()
                has_conda_single_equals = bool(
                    re.search(r"(^|[^<>=!~])=([^=]|$)", dependency)
                )
                if (
                    dependency
                    and not dependency.startswith(("python", "pip"))
                    and not has_conda_single_equals
                ):
                    dependencies.append(dependency)

        return dependencies

    @classmethod
    def _extract_pyproject_dependency_names(cls, pyproject_file: Path) -> list[str]:
        if tomllib is None:
            return []

        try:
            parsed = tomllib.loads(pyproject_file.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            return []

        dependencies: list[str] = []
        project = parsed.get("project", {})
        if isinstance(project, dict):
            declared = project.get("dependencies", [])
            if isinstance(declared, list):
                dependencies.extend(str(item) for item in declared)
            optional = project.get("optional-dependencies", {})
            if isinstance(optional, dict):
                for values in optional.values():
                    if isinstance(values, list):
                        dependencies.extend(str(item) for item in values)

        return cls._normalize_dependency_specs(dependencies)

    @classmethod
    def _extract_setup_cfg_dependency_names(cls, setup_cfg_file: Path) -> list[str]:
        parser = configparser.ConfigParser()
        try:
            parser.read(setup_cfg_file, encoding="utf-8")
        except (configparser.Error, OSError):
            return []

        dependency_specs: list[str] = []
        if parser.has_section("options") and parser.has_option("options", "install_requires"):
            dependency_specs.extend(
                line.strip()
                for line in parser.get("options", "install_requires").splitlines()
            )

        if parser.has_section("options.extras_require"):
            for _, value in parser.items("options.extras_require"):
                dependency_specs.extend(line.strip() for line in value.splitlines())

        return cls._normalize_dependency_specs(dependency_specs)

    @staticmethod
    def _normalize_dependency_specs(specs: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for raw_spec in specs:
            dependency_name = _canonicalize_dependency_name(raw_spec)
            if not dependency_name or dependency_name in seen:
                continue
            seen.add(dependency_name)
            normalized.append(dependency_name)
        return normalized
