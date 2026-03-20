"""Command and option parsing utilities for repair."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

from .repair import QUICK_EVAL_AUTO_OPTIONS

logger = logging.getLogger(__name__)


class _RepairCommandsMixin:
    """Mixin — command option extraction and path utilities."""

    @staticmethod
    def _extract_missing_required_options(error_text: str) -> list[str]:
        options: list[str] = []
        patterns = [
            r"""the following arguments are required:\s*([^\n\r]+)""",
            r"""Missing option ['"]?(--[A-Za-z0-9][A-Za-z0-9_-]*)['"]?""",
            r"""argument\s+(--[A-Za-z0-9][A-Za-z0-9_-]*)\s*:\s*expected one argument""",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, error_text, re.IGNORECASE):
                payload = str(match.group(1)).strip()
                if not payload:
                    continue
                if payload.startswith("--"):
                    extracted = re.findall(r"--[A-Za-z0-9][A-Za-z0-9_-]*", payload)
                    if extracted:
                        for option in extracted:
                            if option not in options:
                                options.append(option)
                        continue
                    if payload not in options:
                        options.append(payload)
                    continue
                for option in re.findall(r"--[A-Za-z0-9][A-Za-z0-9_-]*", payload):
                    if option not in options:
                        options.append(option)
        return options

    @staticmethod
    def _extract_unrecognized_options(error_text: str) -> list[str]:
        options: list[str] = []
        patterns = [
            r"""unrecognized arguments:\s*([^\n\r]+)""",
            r"""No such option:\s*(--[A-Za-z0-9][A-Za-z0-9_-]*)""",
            r"""no such option:\s*(--[A-Za-z0-9][A-Za-z0-9_-]*)""",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, error_text, re.IGNORECASE):
                payload = str(match.group(1)).strip()
                if not payload:
                    continue
                extracted = re.findall(r"--[A-Za-z0-9][A-Za-z0-9_-]*", payload)
                if extracted:
                    for option in extracted:
                        if option not in options:
                            options.append(option)
                    continue
                if payload.startswith("--") and payload not in options:
                    options.append(payload)
        return options

    @staticmethod
    def _strip_command_option(tokens: list[str], option: str) -> list[str]:
        updated: list[str] = []
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if token == option:
                index += 1
                if index < len(tokens) and not tokens[index].startswith("--"):
                    index += 1
                continue
            if token.startswith(f"{option}="):
                index += 1
                continue
            updated.append(token)
            index += 1
        return updated

    @staticmethod
    def _command_option_present(tokens: list[str], options: list[str]) -> tuple[str, int, str]:
        for option in options:
            for index, token in enumerate(tokens):
                if token == option:
                    value = tokens[index + 1] if index + 1 < len(tokens) else ""
                    return option, index, value
                if token.startswith(f"{option}="):
                    return option, index, token.split("=", 1)[1]
        return "", -1, ""

    @staticmethod
    def _path_variants(code_dir: Path, path_value: str) -> set[str]:
        normalized = str(path_value or "").strip()
        if not normalized:
            return set()

        variants: set[str] = {os.path.normcase(os.path.normpath(normalized))}
        candidate = Path(normalized)
        if not candidate.is_absolute():
            candidate = code_dir / candidate
        variants.add(os.path.normcase(os.path.normpath(str(candidate))))
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        variants.add(os.path.normcase(os.path.normpath(str(resolved))))
        return variants

    @classmethod
    def _option_value_matches_missing_target(
        cls,
        code_dir: Path,
        option_value: str,
        missing_targets: list[str],
    ) -> bool:
        option_variants = cls._path_variants(code_dir, option_value)
        if not option_variants:
            return False
        for target in missing_targets:
            target_variants = cls._path_variants(code_dir, target)
            if option_variants & target_variants:
                return True
        return False

    @staticmethod
    def _command_entry_script(tokens: list[str], code_dir: Path) -> Path | None:
        for token in tokens:
            normalized = str(token or "").strip()
            if not normalized:
                continue
            if normalized in {"-m", "-c"}:
                return None
            if normalized.endswith(".py"):
                candidate = Path(normalized)
                return candidate if candidate.is_absolute() else code_dir / candidate
        return None

    @staticmethod
    def _entry_script_supports_flag(entry_script: Path | None, flag: str) -> bool:
        if not entry_script or not entry_script.exists():
            return False
        try:
            content = entry_script.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False
        normalized_flag = flag.lstrip("-").replace("-", "_")
        return flag in content or normalized_flag in content

    @classmethod
    def _resume_failure_signals(cls, error_text: str) -> list[str]:
        lower = str(error_text or "").lower()
        signals: list[str] = []
        signal_map = {
            "timed out": "timeout",
            "timeout": "timeout",
            "keyboardinterrupt": "keyboard_interrupt",
            "interrupted": "interrupted",
            "sigterm": "sigterm",
            "terminated": "terminated",
            "preempt": "preempted",
            "cancelled": "cancelled",
            "node_fail": "node_fail",
            "node fail": "node_fail",
        }
        for token, label in signal_map.items():
            if token in lower and label not in signals:
                signals.append(label)
        return signals

    @staticmethod
    def _choose_single_path(candidates: list[Path]) -> Path | None:
        unique: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                resolved = candidate.resolve()
            except OSError:
                resolved = candidate
            key = str(resolved)
            if key in seen:
                continue
            seen.add(key)
            unique.append(candidate)
        if len(unique) == 1:
            return unique[0]
        if len(unique) > 1:
            # Heuristic: prefer most recently modified file (same logic as
            # _choose_latest_path) so callers don't silently get None.
            logger.debug(
                "_choose_single_path: %d candidates, picking most recent: %s",
                len(unique), [str(p) for p in unique],
            )
            def _mtime(p: Path) -> float:
                try:
                    return p.stat().st_mtime
                except OSError:
                    return -1.0
            return max(unique, key=_mtime)
        return None

    @staticmethod
    def _choose_latest_path(candidates: list[Path]) -> Path | None:
        unique: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                resolved = candidate.resolve()
            except OSError:
                resolved = candidate
            key = str(resolved)
            if key in seen:
                continue
            seen.add(key)
            unique.append(candidate)
        if not unique:
            return None

        def sort_key(path: Path) -> tuple[float, str]:
            try:
                mtime = path.stat().st_mtime
            except OSError:
                mtime = -1.0
            try:
                resolved = path.resolve()
            except OSError:
                resolved = path
            return mtime, str(resolved)

        return sorted(unique, key=sort_key, reverse=True)[0]

