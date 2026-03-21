"""Environment discovery and dependency name utilities."""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ._constants import _TORCH_FAMILY_PACKAGES
from ._gpu_detect import _find_conda, _probe_python_info

logger = logging.getLogger(__name__)

def discover_environments() -> list[dict[str, Any]]:
    """Scan the system for all available Python environments.

    Returns a list of dicts, each with:
      - name:     human-readable label  (e.g. "conda: shixun")
      - python:   absolute path to python executable
      - source:   "conda" | "system" | "pyenv"
      - version:  Python version string
      - packages: list of detected key ML packages
    """
    seen_paths: set[str] = set()
    envs: list[dict[str, Any]] = []

    def _add(name: str, python: str, source: str) -> None:
        try:
            resolved = str(Path(python).resolve())
        except OSError:
            return
        if resolved in seen_paths:
            return
        info = _probe_python_info(python)
        if info is None:
            return
        seen_paths.add(resolved)
        envs.append({
            "name": name,
            "python": resolved,
            "source": source,
            "version": info.get("version", "?"),
            "packages": info.get("packages", []),
        })

    # 1. Conda environments
    conda_cmd = _find_conda()
    if conda_cmd:
        try:
            proc = subprocess.run(
                [conda_cmd, "env", "list", "--json"],
                capture_output=True, text=True, timeout=30,
            )
            if proc.returncode == 0:
                data = json.loads(proc.stdout)
                for env_path in data.get("envs", []):
                    ep = Path(env_path)
                    env_name = ep.name
                    is_win = platform.system() == "Windows"
                    py = ep / ("python.exe" if is_win else "bin/python")
                    if py.is_file():
                        label = f"conda: {env_name}"
                        if env_name == "base":
                            label = "conda: base"
                        _add(label, str(py), "conda")
        except Exception:
            pass

    # 2. System / PATH pythons
    is_win = platform.system() == "Windows"
    python_names = (
        ["python", "python3", "py"] if is_win else ["python", "python3"]
    )
    for pname in python_names:
        found = shutil.which(pname)
        if found:
            _add(f"system: {pname}", found, "system")

    # 3. pyenv versions
    pyenv_root = os.environ.get("PYENV_ROOT") or os.path.expanduser("~/.pyenv")
    versions_dir = Path(pyenv_root) / "versions"
    if versions_dir.is_dir():
        for vdir in sorted(versions_dir.iterdir()):
            py = vdir / ("Scripts/python.exe" if is_win else "bin/python")
            if py.is_file():
                _add(f"pyenv: {vdir.name}", str(py), "pyenv")

    # 4. Common Windows Python locations
    if is_win:
        for base_env in ["LOCALAPPDATA", "APPDATA"]:
            base = Path(os.environ.get(base_env, ""))
            if base.is_dir():
                for candidate in base.glob("Python3*/python.exe"):
                    _add(
                        f"system: {candidate.parent.name}",
                        str(candidate), "system",
                    )

    return envs


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


