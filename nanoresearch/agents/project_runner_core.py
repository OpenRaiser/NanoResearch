"""Utilities for deterministic experiment runner assets."""

from __future__ import annotations

import json
import platform
import re
import shlex
import shutil
from pathlib import Path
from typing import Any

from nanoresearch.agents.project_runner_script import _build_runner_script


RUNNER_SCRIPT_NAME = "nanoresearch_runner.py"
RUNNER_CONFIG_NAME = "nanoresearch_runner.json"
ARTIFACT_DIR_NAMES = ("results", "checkpoints", "logs")
ENTRYPOINT_REL_PATHS = (
    "main.py",
    "train.py",
    "run.py",
    "run_train.py",
    "experiment.py",
    "scripts/train.py",
    "scripts/run.py",
    "src/main.py",
)
_ENV_ASSIGNMENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=.*")
_SCRIPT_SUFFIXES = {".py", ".sh", ".bash", ".ps1", ".bat", ".cmd"}


def _strip_wrapping_quotes(token: str) -> str:
    normalized = str(token or "").strip()
    if len(normalized) >= 2 and normalized[0] == normalized[-1] and normalized[0] in {'"', "'"}:
        return normalized[1:-1]
    return normalized


def is_python_launcher_token(token: str) -> bool:
    """Return True when a command token refers to a Python launcher."""
    normalized = _strip_wrapping_quotes(token)
    if not normalized:
        return False
    name = Path(normalized).name.lower()
    return bool(re.fullmatch(r"(python(?:\d+(?:\.\d+)*)?|py)(?:\.exe)?", name))


def _split_command(command: str) -> list[str]:
    raw = str(command or "").strip()
    if not raw:
        return []
    try:
        return shlex.split(raw, posix=platform.system() != "Windows")
    except ValueError:
        return raw.split()


def _truncate_shell_chain(tokens: list[str]) -> list[str]:
    chain_tokens = {"&&", "||", ";", "|", "&"}
    truncated: list[str] = []
    for token in tokens:
        if token in chain_tokens:
            break
        truncated.append(token)
    return truncated


def _extract_env_assignments(tokens: list[str]) -> tuple[dict[str, str], list[str]]:
    env_vars: dict[str, str] = {}
    remaining = list(tokens)
    if remaining and remaining[0] == "env":
        remaining = remaining[1:]

    while remaining and _ENV_ASSIGNMENT_RE.fullmatch(remaining[0]):
        key, value = remaining.pop(0).split("=", 1)
        env_vars[key] = value
    return env_vars, remaining


def _unwrap_shell_wrapper(tokens: list[str]) -> tuple[dict[str, str], list[str]]:
    current = _truncate_shell_chain(tokens)
    env_vars, current = _extract_env_assignments(current)
    if not current:
        return env_vars, current

    first = Path(str(current[0]).strip("\"'")).name.lower()
    if first == "cmd" and len(current) >= 3 and current[1].lower() == "/c":
        nested_env, nested_tokens = _unwrap_shell_wrapper(current[2:])
        return {**env_vars, **nested_env}, nested_tokens

    if first in {"bash", "sh"} and len(current) >= 3 and current[1] in {"-c", "-lc"}:
        nested_env, nested_tokens = _unwrap_shell_wrapper(_split_command(current[2]))
        return {**env_vars, **nested_env}, nested_tokens

    if first in {"powershell", "powershell.exe", "pwsh", "pwsh.exe"} and len(current) >= 3:
        if current[1].lower() in {"-command", "-c"}:
            nested_env, nested_tokens = _unwrap_shell_wrapper(_split_command(current[2]))
            return {**env_vars, **nested_env}, nested_tokens

    return env_vars, current


def normalize_target_spec(train_command: str, code_dir: Path) -> tuple[list[str], dict[str, str]]:
    """Normalize a model-generated train command into runnable tokens + env vars."""
    env_vars, tokens = _unwrap_shell_wrapper(_split_command(train_command))
    if tokens and is_python_launcher_token(tokens[0]):
        tokens = tokens[1:]

    normalized = []
    for token in tokens:
        cleaned = _strip_wrapping_quotes(token)
        if not cleaned or cleaned in {"--dry-run", "--quick-eval"}:
            continue
        normalized.append(cleaned)
    if normalized:
        return normalized, env_vars

    for candidate in ("main.py", "train.py", "run.py"):
        if (code_dir / candidate).exists():
            return [candidate], env_vars
    return ["main.py"], env_vars


def normalize_target_command(train_command: str, code_dir: Path) -> list[str]:
    """Normalize a model-generated train command into runner target tokens."""
    normalized, _env_vars = normalize_target_spec(train_command, code_dir)
    return normalized


def _coerce_command_tokens(command: str | list[str]) -> list[str]:
    if isinstance(command, list):
        normalized: list[str] = []
        for token in command:
            cleaned = _strip_wrapping_quotes(str(token))
            if cleaned:
                normalized.append(cleaned)
        return normalized
    return [_strip_wrapping_quotes(token) for token in _split_command(str(command or ""))]


def _path_within_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    except OSError:
        return False
    return True


def _resolve_command_path(token: str, code_dir: Path) -> Path:
    candidate = Path(str(token).strip().strip("\"'"))
    if candidate.is_absolute():
        return candidate
    return code_dir / candidate


def _is_path_like_token(token: str) -> bool:
    normalized = str(token or "").strip().strip("\"'")
    if not normalized:
        return False
    if any(separator in normalized for separator in ("/", "\\")):
        return True
    if normalized.startswith("."):
        return True
    return Path(normalized).suffix.lower() in _SCRIPT_SUFFIXES


def _runner_target_spec(code_dir: Path) -> tuple[list[str] | None, str]:
    config_path = code_dir / RUNNER_CONFIG_NAME
    if not config_path.exists():
        return None, f"Runner config not found: {RUNNER_CONFIG_NAME}"
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"Runner config unreadable: {exc}"
    target_tokens = payload.get("target_command")
    if not isinstance(target_tokens, list):
        return None, "Runner config missing target_command list"
    normalized = [str(token).strip() for token in target_tokens if str(token).strip()]
    if not normalized:
        return None, "Runner config target_command is empty"
    return normalized, ""


def _runner_target_env(code_dir: Path) -> dict[str, str]:
    config_path = code_dir / RUNNER_CONFIG_NAME
    if not config_path.exists():
        return {}
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    env_data = payload.get("target_env")
    if not isinstance(env_data, dict):
        return {}
    return {
        str(key): str(value)
        for key, value in env_data.items()
        if str(key).strip()
    }


def _unique_workspace_paths(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _workspace_python_files(code_dir: Path, *, max_depth: int = 4) -> list[Path]:
    files: list[Path] = []
    for path in code_dir.rglob("*.py"):
        try:
            rel_parts = path.relative_to(code_dir).parts
        except ValueError:
            continue
        if any(part.startswith(".") or part in {"__pycache__", ".venv", "venv"} for part in rel_parts):
            continue
        if len(rel_parts) > max_depth:
            continue
        files.append(path)
    return _unique_workspace_paths(files)


def _workspace_entrypoint_candidates(code_dir: Path) -> list[Path]:
    return _unique_workspace_paths(
        [code_dir / rel_path for rel_path in ENTRYPOINT_REL_PATHS if (code_dir / rel_path).exists()]
    )


def _relative_command_path(path: Path, code_dir: Path) -> str:
    try:
        return path.relative_to(code_dir).as_posix()
    except ValueError:
        return str(path)


def _shell_join_command(tokens: list[str]) -> str:
    if not tokens:
        return ""
    return shlex.join([str(token) for token in tokens if str(token).strip()])


def _write_runner_assets(
    code_dir: Path,
    target_command: list[str],
    *,
    target_env: dict[str, str] | None = None,
) -> list[str]:
    runner_script = code_dir / RUNNER_SCRIPT_NAME
    runner_config = code_dir / RUNNER_CONFIG_NAME
    modified: list[str] = []

    if not runner_script.exists():
        runner_script.write_text(_build_runner_script(), encoding="utf-8")
        modified.append(str(runner_script))

    runner_config.write_text(
        json.dumps(
            {
                "target_command": list(target_command),
                "target_env": {str(key): str(value) for key, value in (target_env or {}).items()},
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    modified.append(str(runner_config))
    return modified


def _repair_target_candidate(code_dir: Path, target_hint: str) -> Path | None:
    normalized_hint = str(target_hint or "").strip()
    hint_name = Path(normalized_hint).name.lower() if normalized_hint else ""

    if hint_name:
        basename_matches = [
            path
            for path in _workspace_python_files(code_dir)
            if path.name.lower() == hint_name
        ]
        unique_matches = _unique_workspace_paths(basename_matches)
        if len(unique_matches) == 1:
            return unique_matches[0]

    entrypoints = _workspace_entrypoint_candidates(code_dir)
    if hint_name:
        exact_entrypoints = [
            path
            for path in entrypoints
            if path.name.lower() == hint_name
        ]
        if len(exact_entrypoints) == 1:
            return exact_entrypoints[0]

    if len(entrypoints) == 1:
        return entrypoints[0]

    return None
