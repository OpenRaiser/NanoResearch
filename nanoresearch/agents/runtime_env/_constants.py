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

