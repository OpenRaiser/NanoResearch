"""GPU detection and Python probing utilities."""

from __future__ import annotations

import json
import logging
import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ._constants import _CUDA_DRIVER_TO_TORCH_TAG

logger = logging.getLogger(__name__)

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


def _probe_python_info(python_path: str) -> dict[str, Any] | None:
    """Get version and key package info from a Python executable."""
    try:
        proc = subprocess.run(
            [python_path, "-c",
             "import sys, json; "
             "pkgs = {}; "
             "[pkgs.update({m: True}) for m in "
             "['torch','numpy','transformers','tensorflow'] "
             "if __import__('importlib').util.find_spec(m)]; "
             "print(json.dumps({'version': sys.version.split()[0], "
             "'prefix': sys.prefix, 'packages': list(pkgs.keys())}))"],
            capture_output=True, text=True, timeout=15,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return json.loads(proc.stdout.strip())
    except Exception:
        pass
    try:
        proc = subprocess.run(
            [python_path, "--version"],
            capture_output=True, text=True, timeout=10,
        )
        if proc.returncode == 0:
            ver = (proc.stdout + proc.stderr).strip().split()[-1]
            return {"version": ver, "prefix": "", "packages": []}
    except Exception:
        pass
    return None


