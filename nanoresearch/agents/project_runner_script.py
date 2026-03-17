"""Runner script template builder."""

from __future__ import annotations

from pathlib import Path

_RUNNER_SCRIPT_TEMPLATE = Path(__file__).with_name("_runner_script_template.py.txt")


def _build_runner_script() -> str:
    return _RUNNER_SCRIPT_TEMPLATE.read_text(encoding="utf-8")
