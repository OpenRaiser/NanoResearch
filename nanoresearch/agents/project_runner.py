"""Utilities for deterministic experiment runner assets."""

from __future__ import annotations

import json
import platform
import re
import shlex
import shutil
from pathlib import Path
from typing import Any


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


def _validate_command_target(
    tokens: list[str],
    code_dir: Path,
    *,
    allow_runner_recursion: bool = True,
) -> dict[str, Any]:
    if not tokens:
        return {
            "status": "failed",
            "target_kind": "unknown",
            "target": "",
            "failures": ["Launch command is empty"],
            "warnings": [],
        }

    normalized = [str(token).strip() for token in tokens if str(token).strip()]
    if normalized and is_python_launcher_token(normalized[0]):
        normalized = normalized[1:]

    if not normalized:
        return {
            "status": "failed",
            "target_kind": "unknown",
            "target": "",
            "failures": ["Launch command does not specify a runnable target"],
            "warnings": [],
        }

    first = normalized[0]
    if first in {"-m", "-c"}:
        if len(normalized) < 2 or not str(normalized[1]).strip():
            return {
                "status": "failed",
                "target_kind": "unknown",
                "target": "",
                "failures": [f"Python launcher flag {first} is missing its target value"],
                "warnings": [],
            }
        target_kind = "module" if first == "-m" else "inline_code"
        warnings: list[str] = []
        if first == "-c":
            warnings.append("Inline Python code target cannot be path-validated")
        return {
            "status": "ready",
            "target_kind": target_kind,
            "target": normalized[1],
            "failures": [],
            "warnings": warnings,
        }

    script_token = next((token for token in normalized if token.endswith(".py")), "")
    if script_token:
        resolved_path = _resolve_command_path(script_token, code_dir)
        if not _path_within_root(resolved_path, code_dir):
            return {
                "status": "failed",
                "target_kind": "script",
                "target": script_token,
                "resolved_target": str(resolved_path),
                "failures": [
                    f"Launch target points outside workspace: {script_token}",
                ],
                "warnings": [],
            }
        if not resolved_path.exists():
            return {
                "status": "failed",
                "target_kind": "script",
                "target": script_token,
                "resolved_target": str(resolved_path),
                "failures": [
                    f"Launch target not found: {script_token}",
                ],
                "warnings": [],
            }
        if allow_runner_recursion and resolved_path.name == RUNNER_SCRIPT_NAME:
            runner_tokens, runner_error = _runner_target_spec(code_dir)
            if runner_tokens is None:
                return {
                    "status": "failed",
                    "target_kind": "runner",
                    "target": script_token,
                    "resolved_target": str(resolved_path),
                    "failures": [runner_error],
                    "warnings": [],
                }
            runner_target = _validate_command_target(
                runner_tokens,
                code_dir,
                allow_runner_recursion=False,
            )
            if runner_target.get("status") == "failed":
                nested_failures = [
                    f"Runner target invalid: {failure}"
                    for failure in runner_target.get("failures", [])
                ]
                return {
                    "status": "failed",
                    "target_kind": "runner",
                    "target": script_token,
                    "resolved_target": str(resolved_path),
                    "runner_target": runner_target,
                    "failures": nested_failures,
                    "warnings": list(runner_target.get("warnings", [])),
                }
            return {
                "status": "ready",
                "target_kind": "runner",
                "target": script_token,
                "resolved_target": str(resolved_path),
                "runner_target": runner_target,
                "failures": [],
                "warnings": list(runner_target.get("warnings", [])),
            }
        return {
            "status": "ready",
            "target_kind": "script",
            "target": script_token,
            "resolved_target": str(resolved_path),
            "failures": [],
            "warnings": [],
        }

    if _is_path_like_token(first):
        resolved_path = _resolve_command_path(first, code_dir)
        if not _path_within_root(resolved_path, code_dir):
            return {
                "status": "failed",
                "target_kind": "path",
                "target": first,
                "resolved_target": str(resolved_path),
                "failures": [
                    f"Launch target points outside workspace: {first}",
                ],
                "warnings": [],
            }
        if not resolved_path.exists():
            return {
                "status": "failed",
                "target_kind": "path",
                "target": first,
                "resolved_target": str(resolved_path),
                "failures": [
                    f"Launch target not found: {first}",
                ],
                "warnings": [],
            }
        return {
            "status": "ready",
            "target_kind": "path",
            "target": first,
            "resolved_target": str(resolved_path),
            "failures": [],
            "warnings": [],
        }

    if shutil.which(first):
        return {
            "status": "ready",
            "target_kind": "external_executable",
            "target": first,
            "failures": [],
            "warnings": [],
        }

    return {
        "status": "ready",
        "target_kind": "external_executable",
        "target": first,
        "failures": [],
        "warnings": [f"External executable {first!r} was not path-validated"],
    }


def repair_launch_contract(
    command: str | list[str],
    code_dir: Path,
) -> dict[str, Any]:
    command_tokens = _coerce_command_tokens(command)
    initial_contract = validate_launch_contract(command_tokens, code_dir)
    result: dict[str, Any] = {
        "status": "skipped",
        "command": list(command_tokens),
        "command_string": _shell_join_command(command_tokens),
        "actions": [],
        "files_modified": [],
        "initial_contract": initial_contract,
        "final_contract": initial_contract,
    }
    if initial_contract.get("status") != "failed":
        return result

    actions: list[dict[str, Any]] = []
    files_modified: list[str] = []
    repaired_command = list(command_tokens)
    first_token = repaired_command[0] if repaired_command else ""
    launcher_token = first_token if is_python_launcher_token(first_token) else "python"
    target_hint = str(initial_contract.get("target") or "").strip()
    runner_target = initial_contract.get("runner_target")
    if isinstance(runner_target, dict):
        target_hint = str(runner_target.get("target") or target_hint).strip()
    candidate = _repair_target_candidate(code_dir, target_hint)

    if candidate is not None:
        candidate_rel = _relative_command_path(candidate, code_dir)
        if initial_contract.get("target_kind") == "runner":
            target_env = _runner_target_env(code_dir)
            files_modified.extend(
                _write_runner_assets(
                    code_dir,
                    [candidate_rel],
                    target_env=target_env,
                )
            )
            actions.append(
                {
                    "kind": "runner_target_refresh",
                    "target": candidate_rel,
                }
            )
        else:
            passthrough: list[str] = []
            if repaired_command:
                if is_python_launcher_token(repaired_command[0]):
                    passthrough = repaired_command[2:]
                else:
                    passthrough = repaired_command[1:]
            files_modified.extend(_write_runner_assets(code_dir, [candidate_rel]))
            repaired_command = [launcher_token, RUNNER_SCRIPT_NAME, *passthrough]
            actions.append(
                {
                    "kind": "command_target_redirect",
                    "target": candidate_rel,
                    "strategy": "deterministic_runner",
                }
            )

    final_contract = validate_launch_contract(repaired_command, code_dir)
    result.update(
        {
            "status": "applied" if actions and final_contract.get("status") != "failed" else "failed",
            "command": repaired_command,
            "command_string": _shell_join_command(repaired_command),
            "actions": actions,
            "files_modified": files_modified,
            "final_contract": final_contract,
        }
    )
    if not actions:
        result["status"] = "failed"
    return result


def _ensure_writable_dir(path: Path) -> tuple[bool, str]:
    probe_path = path / ".nanoresearch_write_probe"
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe_path.write_text("ok", encoding="utf-8")
        probe_path.unlink(missing_ok=True)
    except OSError as exc:
        return False, str(exc)
    return True, ""


def validate_launch_contract(
    command: str | list[str],
    code_dir: Path,
    *,
    create_artifact_dirs: bool = True,
) -> dict[str, Any]:
    command_tokens = _coerce_command_tokens(command)
    target_validation = _validate_command_target(command_tokens, code_dir)
    failures = list(target_validation.get("failures", []))
    warnings = list(target_validation.get("warnings", []))
    created_dirs: list[str] = []
    artifact_dirs: dict[str, Any] = {}

    for dirname in ARTIFACT_DIR_NAMES:
        path = code_dir / dirname
        created = False
        if create_artifact_dirs and not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            created = True
            created_dirs.append(str(path))
        writable, error = _ensure_writable_dir(path)
        artifact_dirs[dirname] = {
            "path": str(path),
            "exists": path.exists(),
            "writable": writable,
            "created": created,
        }
        if error:
            artifact_dirs[dirname]["error"] = error
            failures.append(f"Artifact directory is not writable: {dirname} ({error})")

    if failures:
        status = "failed"
    elif created_dirs:
        status = "repaired"
    else:
        status = "ready"

    return {
        "status": status,
        "command": command_tokens,
        "target_kind": target_validation.get("target_kind", "unknown"),
        "target": target_validation.get("target", ""),
        "resolved_target": target_validation.get("resolved_target", ""),
        "runner_target": target_validation.get("runner_target", {}),
        "artifact_dirs": artifact_dirs,
        "created_dirs": created_dirs,
        "warnings": warnings,
        "failures": failures,
    }


def ensure_project_runner(code_dir: Path, train_command: str) -> dict[str, Any]:
    """Write deterministic runner assets for a generated experiment project."""
    target_command, target_env = normalize_target_spec(train_command, code_dir)
    runner_script = code_dir / RUNNER_SCRIPT_NAME
    runner_config = code_dir / RUNNER_CONFIG_NAME
    runner_config.write_text(
        json.dumps({"target_command": target_command, "target_env": target_env}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    runner_script.write_text(_build_runner_script(), encoding="utf-8")
    return {
        "runner_script": str(runner_script),
        "runner_config": str(runner_config),
        "runner_command": f"python {RUNNER_SCRIPT_NAME}",
        "target_command": target_command,
        "target_env": target_env,
    }


def refresh_project_runner_script(code_dir: Path) -> list[str]:
    """Refresh the deterministic runner script in-place without touching runner config."""
    runner_script = code_dir / RUNNER_SCRIPT_NAME
    desired = _build_runner_script()
    try:
        current = runner_script.read_text(encoding="utf-8") if runner_script.exists() else ""
    except OSError:
        current = ""
    if current == desired:
        return []
    runner_script.write_text(desired, encoding="utf-8")
    return [str(runner_script)]


def _build_runner_script() -> str:
    return r'''"""Deterministic wrapper around model-generated training entrypoints."""

from __future__ import annotations

import argparse
import ast
import json
import os
import py_compile
import re
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ImportError:  # pragma: no cover - Python < 3.11
    tomllib = None


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "nanoresearch_runner.json"
AUTO_CONFIG_DIR = PROJECT_ROOT / ".nanoresearch_autofix"


def _load_runner_config() -> dict[str, object]:
    if not CONFIG_PATH.exists():
        return {}
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _load_target_spec(config_data: dict[str, object] | None = None) -> tuple[list[str], dict[str, str]]:
    data = config_data if isinstance(config_data, dict) else _load_runner_config()
    target = data.get("target_command")
    env_data = data.get("target_env")
    target_env = (
        {str(key): str(value) for key, value in env_data.items()}
        if isinstance(env_data, dict)
        else {}
    )
    if isinstance(target, list) and all(isinstance(token, str) and token.strip() for token in target):
        return target, target_env
    return ["main.py"], target_env


def _load_quick_eval_blocked_options(config_data: dict[str, object] | None = None) -> set[str]:
    data = config_data if isinstance(config_data, dict) else _load_runner_config()
    raw = data.get("quick_eval_blocked_options")
    if not isinstance(raw, list):
        return set()
    blocked: set[str] = set()
    for option in raw:
        normalized = str(option or "").strip()
        if normalized.startswith("--"):
            blocked.add(normalized)
    return blocked


def _entry_script_path(tokens: list[str]) -> Path | None:
    if not tokens:
        return None
    for token in tokens:
        if token in {"-m", "-c"}:
            return None
        if token.endswith(".py"):
            candidate = Path(token)
            return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    return None


def _is_python_launcher(token: str) -> bool:
    normalized = str(token or "").strip().strip("'").strip('"')
    if not normalized:
        return False
    name = Path(normalized).name.lower()
    return bool(re.fullmatch(r"(python(?:\d+(?:\.\d+)*)?|py)(?:\.exe)?", name))


def _detect_flag_form(entry_script: Path | None, flag: str) -> str | None:
    """Return the actual CLI flag form found in *entry_script*'s argparse, or *None*.

    Prefers the exact form declared in ``add_argument()``.  Falls back to raw
    string presence with ``--`` prefix so non-argparse scripts still work.
    This avoids the classic ``--dry-run`` vs ``--dry_run`` mismatch.
    """
    if not entry_script or not entry_script.exists():
        return None
    try:
        content = entry_script.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    bare = flag.lstrip("-").replace("-", "_")
    hyphen_form = "--" + bare.replace("_", "-")
    underscore_form = "--" + bare
    # Priority 1: match argparse add_argument() definition
    for form in (hyphen_form, underscore_form):
        if re.search(rf'add_argument\s*\(\s*["\']' + re.escape(form) + r'["\']', content):
            return form
    # Priority 2: raw presence with -- prefix
    if hyphen_form in content:
        return hyphen_form
    if underscore_form in content:
        return underscore_form
    return None


def _supports_flag(entry_script: Path | None, flag: str) -> bool:
    return _detect_flag_form(entry_script, flag) is not None


def _entry_script_source(entry_script: Path | None) -> str:
    if not entry_script or not entry_script.exists():
        return ""
    try:
        return entry_script.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _argument_blocks(entry_script: Path | None) -> list[str]:
    content = _entry_script_source(entry_script)
    if not content:
        return []
    return re.findall(r"add_argument\((.*?)\)", content, flags=re.DOTALL)


def _option_required(entry_script: Path | None, options: list[str]) -> bool:
    option_set = set(options)
    for block in _argument_blocks(entry_script):
        if not any(option in block for option in option_set):
            continue
        if re.search(r"required\s*=\s*True", block):
            return True
    return False


def _option_present(tokens: list[str], options: list[str]) -> tuple[str, int, str]:
    for option in options:
        for index, token in enumerate(tokens):
            if token == option:
                value = tokens[index + 1] if index + 1 < len(tokens) else ""
                return option, index, value
            if token.startswith(f"{option}="):
                return option, index, token.split("=", 1)[1]
    return "", -1, ""


def _resolved_path(path_value: str) -> Path:
    candidate = Path(path_value)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _path_exists(path_value: str) -> bool:
    normalized = str(path_value or "").strip()
    if not normalized:
        return False
    return _resolved_path(normalized).exists()


def _unique_existing_paths(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        key = str(resolved)
        if not path.exists() or key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _project_files(max_depth: int = 3) -> list[Path]:
    files: list[Path] = []
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file():
            continue
        try:
            rel_parts = path.relative_to(PROJECT_ROOT).parts
        except ValueError:
            continue
        if any(part.startswith(".") or part == "__pycache__" for part in rel_parts):
            continue
        if len(rel_parts) > max_depth:
            continue
        files.append(path)
    return files


def _existing_config_files() -> list[Path]:
    preferred = _unique_existing_paths(
        [
            PROJECT_ROOT / "config.py",
            PROJECT_ROOT / "config.yaml",
            PROJECT_ROOT / "config.yml",
            PROJECT_ROOT / "config.json",
            PROJECT_ROOT / "config.toml",
            PROJECT_ROOT / "config" / "default.yaml",
            PROJECT_ROOT / "config" / "default.yml",
            PROJECT_ROOT / "config" / "default.json",
            PROJECT_ROOT / "config" / "default.toml",
            PROJECT_ROOT / "configs" / "default.yaml",
            PROJECT_ROOT / "configs" / "default.yml",
            PROJECT_ROOT / "configs" / "default.json",
            PROJECT_ROOT / "configs" / "default.toml",
        ]
    )
    if preferred:
        return preferred
    return _unique_existing_paths(
        [
            path
            for path in _project_files()
            if path.suffix.lower() in {".py", ".yaml", ".yml", ".json", ".toml"}
            and "config" in path.stem.lower()
        ]
    )


def _expected_config_format(entry_script: Path | None) -> str:
    source = _entry_script_source(entry_script).lower()
    if not source:
        return "unknown"
    if any(token in source for token in ("yaml.safe_load", "yaml.load", "omegaconf", "hydra", ".yaml", ".yml")):
        return "yaml"
    if any(token in source for token in ("json.load", "json.loads", ".json")):
        return "json"
    if any(token in source for token in ("tomllib.load", "toml.load", ".toml")):
        return "toml"
    if any(token in source for token in ("import config", "from config import")):
        return "python"
    return "unknown"


def _parse_python_config(path: Path) -> dict[str, object] | None:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError):
        return None

    mapping: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        key = target.id
        if key.startswith("_"):
            continue
        try:
            value = ast.literal_eval(node.value)
        except Exception:
            continue
        if isinstance(value, dict) and key.lower() in {"config", "cfg", "default_config"}:
            for sub_key, sub_value in value.items():
                if isinstance(sub_key, str):
                    mapping[sub_key.lower()] = sub_value
            continue
        mapping[key.lower()] = value
    return mapping or None


def _parse_yaml_like_text(text: str) -> dict[str, object] | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed

    simple_mapping: dict[str, object] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value or value.startswith(("[", "{")):
            continue
        if value.lower() in {"true", "false"}:
            simple_mapping[key] = value.lower() == "true"
            continue
        if re.fullmatch(r"-?\d+", value):
            simple_mapping[key] = int(value)
            continue
        if re.fullmatch(r"-?\d+\.\d+", value):
            simple_mapping[key] = float(value)
            continue
        simple_mapping[key] = value.strip("\"'")
    return simple_mapping or None


def _load_config_mapping(path: Path) -> dict[str, object] | None:
    suffix = path.suffix.lower()
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    if suffix == ".py":
        return _parse_python_config(path)
    if suffix == ".json":
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    if suffix in {".yaml", ".yml"}:
        return _parse_yaml_like_text(text)
    if suffix == ".toml":
        if tomllib is None:
            return None
        try:
            parsed = tomllib.loads(text)
        except Exception:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _normalize_config_mapping(mapping: dict[str, object]) -> dict[str, object]:
    normalized = dict(mapping)
    if not any(key in normalized for key in ("random_seed", "seed", "rand_seed", "manual_seed")):
        normalized["random_seed"] = 42
    return normalized


def _toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    return json.dumps(value, ensure_ascii=False)


def _dump_toml(mapping: dict[str, object], prefix: str = "") -> list[str]:
    lines: list[str] = []
    nested: list[tuple[str, dict[str, object]]] = []
    for key, value in mapping.items():
        if isinstance(value, dict):
            nested.append((key, value))
            continue
        lines.append(f"{key} = {_toml_value(value)}")
    for key, value in nested:
        section = f"{prefix}.{key}" if prefix else key
        if lines:
            lines.append("")
        lines.append(f"[{section}]")
        lines.extend(_dump_toml(value, section))
    return lines


def _materialize_config(mapping: dict[str, object], fmt: str) -> Path | None:
    AUTO_CONFIG_DIR.mkdir(exist_ok=True)
    target = AUTO_CONFIG_DIR / {
        "yaml": "config_auto.yaml",
        "json": "config_auto.json",
        "toml": "config_auto.toml",
    }.get(fmt, "config_auto.json")
    normalized = _normalize_config_mapping(mapping)
    try:
        if fmt == "toml":
            target.write_text("\n".join(_dump_toml(normalized)) + "\n", encoding="utf-8")
        else:
            target.write_text(json.dumps(normalized, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError:
        return None
    return target


def _config_candidate(entry_script: Path | None) -> str | None:
    expected_format = _expected_config_format(entry_script)
    config_files = _existing_config_files()

    if expected_format in {"yaml", "json", "toml", "python"}:
        matching = [
            path for path in config_files
            if (
                expected_format == "python" and path.suffix.lower() == ".py"
            ) or (
                expected_format == "yaml" and path.suffix.lower() in {".yaml", ".yml"}
            ) or (
                expected_format == "json" and path.suffix.lower() == ".json"
            ) or (
                expected_format == "toml" and path.suffix.lower() == ".toml"
            )
        ]
        matching = _unique_existing_paths(matching)
        if len(matching) == 1:
            return str(matching[0].resolve())

    if expected_format in {"yaml", "json", "toml"}:
        source_candidates: list[Path] = []
        for suffix in (".py", ".json", ".yaml", ".yml", ".toml"):
            source_candidates.extend([path for path in config_files if path.suffix.lower() == suffix])
        for source_path in _unique_existing_paths(source_candidates):
            mapping = _load_config_mapping(source_path)
            if not isinstance(mapping, dict):
                continue
            materialized = _materialize_config(mapping, expected_format)
            if materialized is not None:
                return str(materialized.resolve())

    unique_files = _unique_existing_paths(config_files)
    if len(unique_files) == 1:
        return str(unique_files[0].resolve())
    return None


def _data_roots() -> list[Path]:
    return _unique_existing_paths(
        [
            PROJECT_ROOT / "data",
            PROJECT_ROOT / "datasets",
        ]
    )


def _model_roots() -> list[Path]:
    return _unique_existing_paths(
        [
            PROJECT_ROOT / "models",
            PROJECT_ROOT / "checkpoints",
        ]
    )


def _resource_files(roots: list[Path], *, max_depth: int = 3) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            try:
                rel_parts = path.relative_to(root).parts
            except ValueError:
                continue
            if len(rel_parts) > max_depth:
                continue
            files.append(path)
    return _unique_existing_paths(files)


def _resource_directories(roots: list[Path], *, max_depth: int = 3) -> list[Path]:
    directories: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        directories.append(root)
        for path in root.rglob("*"):
            if not path.is_dir():
                continue
            try:
                rel_parts = path.relative_to(root).parts
            except ValueError:
                continue
            if len(rel_parts) > max_depth:
                continue
            directories.append(path)
    return _unique_existing_paths(directories)


def _normalized_name(path_value: str) -> str:
    name = Path(path_value).name.lower()
    for suffix in (".tar.gz", ".tar.bz2", ".tar.xz"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    else:
        for suffix in (".gz", ".bz2", ".xz", ".zip"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
    while True:
        stem, ext = os.path.splitext(name)
        if ext.lower() in {
            ".csv", ".tsv", ".txt", ".json", ".jsonl", ".pkl", ".pickle",
            ".npy", ".npz", ".pt", ".pth", ".bin", ".ckpt", ".h5", ".hdf5",
            ".parquet", ".fa", ".fasta",
        }:
            name = stem
            continue
        break
    return name


def _keyword_file_candidate(roots: list[Path], keywords: list[str]) -> str | None:
    return _keyword_path_candidate(roots, keywords, files_only=True)


def _latest_path_candidate(paths: list[Path]) -> str | None:
    unique = _unique_existing_paths(paths)
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

    return str(sorted(unique, key=sort_key, reverse=True)[0].resolve())


def _keyword_path_candidate(
    roots: list[Path],
    keywords: list[str],
    *,
    files_only: bool = False,
    dirs_only: bool = False,
    allow_latest: bool = False,
) -> str | None:
    normalized_keywords = [keyword.lower() for keyword in keywords if keyword]
    if not normalized_keywords:
        return None
    if dirs_only:
        candidates = _resource_directories(roots)
    elif files_only:
        candidates = _resource_files(roots)
    else:
        candidates = [*_resource_directories(roots), *_resource_files(roots)]

    scored: list[tuple[int, Path]] = []
    for path in candidates:
        haystacks = {
            _normalized_name(str(path)),
            path.name.lower(),
        }
        parent = path.parent
        if parent != path:
            haystacks.add(parent.name.lower())
            haystacks.add(_normalized_name(str(parent)))
        score = sum(1 for keyword in normalized_keywords if any(keyword in value for value in haystacks))
        if score > 0:
            scored.append((score, path))
    if not scored:
        return None
    best_score = max(score for score, _path in scored)
    matches = [path for score, path in scored if score == best_score]
    unique = _unique_existing_paths(matches)
    if len(unique) == 1:
        return str(unique[0].resolve())
    if allow_latest and unique:
        return _latest_path_candidate(unique)
    return None


def _single_path_candidate(roots: list[Path], *, files_only: bool = False) -> str | None:
    candidates: list[Path] = []
    for root in roots:
        if root.exists():
            if not files_only:
                candidates.append(root)
            candidates.extend(_resource_files([root], max_depth=2))
    unique = _unique_existing_paths(candidates)
    if len(unique) == 1:
        return str(unique[0].resolve())
    return None


def _option_candidate(options: list[str], entry_script: Path | None = None) -> str | None:
    option_set = set(options)

    if option_set & {"--config", "--config-path", "--cfg"}:
        return _config_candidate(entry_script)
    if option_set & {"--data-dir", "--data-root", "--dataset-dir", "--dataset-root"}:
        roots = _data_roots()
        if len(roots) == 1:
            return str(roots[0].resolve())
        return None
    if option_set & {"--model-dir", "--model-root", "--checkpoint-dir", "--log-dir", "--output-dir"}:
        if option_set & {"--output-dir"}:
            return str((PROJECT_ROOT / "results").resolve())
        if option_set & {"--log-dir"}:
            return str((PROJECT_ROOT / "logs").resolve())
        if option_set & {"--checkpoint-dir"}:
            return str((PROJECT_ROOT / "checkpoints").resolve())
        roots = _model_roots()
        if len(roots) == 1:
            return str(roots[0].resolve())
        return None
    if option_set & {"--train-file", "--train-data", "--train-path"}:
        return _keyword_file_candidate(_data_roots(), ["train"])
    if option_set & {
        "--val-file",
        "--valid-file",
        "--validation-file",
        "--val-data",
        "--valid-data",
        "--validation-data",
        "--val-path",
        "--valid-path",
        "--dev-file",
        "--dev-data",
        "--dev-path",
    }:
        return _keyword_path_candidate(_data_roots(), ["val", "valid", "validation", "dev"], files_only=True)
    if option_set & {"--test-file", "--test-data", "--test-path"}:
        return _keyword_file_candidate(_data_roots(), ["test"])
    if option_set & {"--labels-path", "--label-file", "--labels-file", "--label-path"}:
        return _keyword_path_candidate(_data_roots(), ["label", "labels"], files_only=True)
    if option_set & {"--annotations", "--annotation-file", "--annotation-path", "--annotations-file"}:
        return _keyword_path_candidate(_data_roots(), ["annot", "annotation", "annotations", "anno"], files_only=True)
    if option_set & {"--split-file", "--splits-file", "--split-path", "--fold-file", "--folds-file"}:
        return _keyword_path_candidate(_data_roots(), ["split", "splits", "fold", "folds"], files_only=True)
    if option_set & {"--metadata-path", "--meta-path", "--metadata-file", "--meta-file"}:
        return _keyword_path_candidate(_data_roots(), ["meta", "metadata"], files_only=True)
    if option_set & {"--image-dir", "--images-dir", "--image-root", "--images-root"}:
        return _keyword_path_candidate(_data_roots(), ["image", "images", "img"], dirs_only=True)
    if option_set & {"--label-dir", "--labels-dir", "--label-root", "--labels-root"}:
        return _keyword_path_candidate(_data_roots(), ["label", "labels", "mask", "masks"], dirs_only=True)
    if option_set & {"--data-path", "--dataset-path", "--input-path", "--input-file", "--dataset-file"}:
        file_candidate = _single_path_candidate(_data_roots(), files_only=True)
        if file_candidate:
            return file_candidate
        roots = _data_roots()
        if len(roots) == 1:
            return str(roots[0].resolve())
        return None
    if option_set & {"--model-path", "--pretrained-model"}:
        return _single_path_candidate(_model_roots())
    if option_set & {"--checkpoint", "--ckpt", "--checkpoint-path", "--resume", "--resume-from", "--resume-path"}:
        checkpoint_files = [
            path
            for path in _resource_files(_model_roots())
            if path.suffix.lower() in {".pt", ".pth", ".ckpt", ".bin", ".safetensors"}
        ]
        preferred = [
            path
            for path in checkpoint_files
            if any(token in str(path).lower() for token in ("checkpoint", "checkpoints", "ckpt"))
        ]
        return (
            _keyword_path_candidate(_model_roots(), ["checkpoint", "ckpt"], files_only=True, allow_latest=True)
            or _latest_path_candidate(preferred)
            or _latest_path_candidate(checkpoint_files)
            or _single_path_candidate(_model_roots())
        )
    if option_set & {"--tokenizer-path", "--tokenizer-name-or-path"}:
        return _keyword_file_candidate(_model_roots(), ["tokenizer"]) or _single_path_candidate(_model_roots())
    return None


def _apply_option_value(tokens: list[str], option: str, index: int, value: str) -> list[str]:
    updated = list(tokens)
    if index < 0:
        updated.extend([option, value])
        return updated
    if updated[index].startswith(f"{option}="):
        updated[index] = f"{option}={value}"
        return updated
    if index + 1 < len(updated):
        updated[index + 1] = value
    else:
        updated.append(value)
    return updated


def _adapt_target_tokens(target_tokens: list[str], entry_script: Path | None) -> list[str]:
    tokens = list(target_tokens)
    option_groups = [
        ["--config", "--config-path", "--cfg"],
        ["--data-dir", "--data-root", "--dataset-dir", "--dataset-root"],
        ["--data-path", "--dataset-path", "--input-path", "--input-file", "--dataset-file"],
        ["--train-file", "--train-data", "--train-path"],
        [
            "--val-file",
            "--valid-file",
            "--validation-file",
            "--val-data",
            "--valid-data",
            "--validation-data",
            "--val-path",
            "--valid-path",
            "--dev-file",
            "--dev-data",
            "--dev-path",
        ],
        ["--test-file", "--test-data", "--test-path"],
        ["--labels-path", "--label-file", "--labels-file", "--label-path"],
        ["--annotations", "--annotation-file", "--annotation-path", "--annotations-file"],
        ["--split-file", "--splits-file", "--split-path", "--fold-file", "--folds-file"],
        ["--metadata-path", "--meta-path", "--metadata-file", "--meta-file"],
        ["--image-dir", "--images-dir", "--image-root", "--images-root"],
        ["--label-dir", "--labels-dir", "--label-root", "--labels-root"],
        ["--model-dir", "--model-root"],
        ["--model-path", "--pretrained-model"],
        ["--tokenizer-path", "--tokenizer-name-or-path"],
        ["--checkpoint", "--ckpt", "--checkpoint-path"],
        ["--resume", "--resume-from", "--resume-path"],
        ["--output-dir"],
        ["--checkpoint-dir"],
        ["--log-dir"],
    ]
    for options in option_groups:
        supported = [option for option in options if _supports_flag(entry_script, option)]
        if not supported:
            continue
        option, index, current_value = _option_present(tokens, supported)
        candidate = _option_candidate(supported, entry_script)
        if option:
            if current_value and _path_exists(current_value):
                continue
            if candidate:
                tokens = _apply_option_value(tokens, option, index, candidate)
            continue
        if not _option_required(entry_script, supported):
            continue
        if candidate:
            tokens = _apply_option_value(tokens, supported[0], -1, candidate)
    return tokens


def _token_present(tokens: list[str], option: str) -> bool:
    return any(token == option or token.startswith(f"{option}=") for token in tokens)


def _build_quick_eval_tokens(
    target_tokens: list[str],
    entry_script: Path | None,
    passthrough: list[str],
    blocked_options: set[str] | None = None,
) -> list[str]:
    tokens = list(target_tokens)
    blocked = set(blocked_options or set())
    actual_qe = _detect_flag_form(entry_script, "--quick-eval")
    if (
        actual_qe
        and "--quick-eval" not in blocked
        and "--quick_eval" not in blocked
        and not _token_present(tokens, "--quick-eval")
        and not _token_present(tokens, "--quick_eval")
    ):
        tokens.append(actual_qe)

    # Canonical speedup options — only one entry per logical flag.
    # _detect_flag_form resolves hyphen / underscore automatically.
    speedups = [
        ("--epochs", "1"),
        ("--num-epochs", "1"),
        ("--max-steps", "2"),
        ("--steps", "2"),
        ("--batch-size", "8"),
        ("--num-workers", "0"),
        ("--workers", "0"),
        ("--subset-size", "64"),
        ("--train-size", "64"),
        ("--quick-eval-train-size", "64"),
        ("--limit-train-batches", "2"),
        ("--limit-val-batches", "1"),
    ]
    seen_bare: set[str] = set()
    for option, value in speedups:
        bare = option.lstrip("-").replace("-", "_")
        if bare in seen_bare:
            continue
        actual = _detect_flag_form(entry_script, option)
        if (
            actual
            and option not in blocked
            and not _token_present(tokens, actual)
            and not _token_present(passthrough, actual)
            # Also check the other form to avoid duplicates
            and not _token_present(tokens, option)
            and not _token_present(passthrough, option)
        ):
            tokens.extend([actual, value])
            seen_bare.add(bare)
    return [*tokens, *passthrough]


def _run_target(tokens: list[str], mode: str, target_env: dict[str, str] | None = None) -> int:
    env = {**os.environ}
    if target_env:
        env.update(target_env)
    env["NANORESEARCH_EXECUTION_MODE"] = mode
    if mode == "quick-eval":
        env["NANORESEARCH_QUICK_EVAL"] = "1"
        env.setdefault("WANDB_MODE", "disabled")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
    command = _materialize_command(tokens)
    completed = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        env=env,
        check=False,
    )
    return int(completed.returncode or 0)


def _materialize_command(tokens: list[str]) -> list[str]:
    if not tokens:
        return [sys.executable, "main.py"]
    first = tokens[0]
    if _is_python_launcher(first):
        return [sys.executable, *tokens[1:]]
    if first in {"-m", "-c"} or first.endswith(".py"):
        return [sys.executable, *tokens]
    return list(tokens)


def _compile_project() -> None:
    for py_file in PROJECT_ROOT.rglob("*.py"):
        if any(part in {".venv", "__pycache__"} for part in py_file.parts):
            continue
        py_compile.compile(str(py_file), doraise=True)


def _ensure_output_dirs() -> None:
    for dirname in ("results", "checkpoints", "logs"):
        (PROJECT_ROOT / dirname).mkdir(exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="NanoResearch deterministic runner")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quick-eval", action="store_true")
    args, passthrough = parser.parse_known_args()

    if args.dry_run and args.quick_eval:
        print("Only one of --dry-run/--quick-eval may be used.", file=sys.stderr)
        return 2

    _ensure_output_dirs()
    config_data = _load_runner_config()
    target_tokens, target_env = _load_target_spec(config_data)
    quick_eval_blocked_options = _load_quick_eval_blocked_options(config_data)
    entry_script = _entry_script_path(target_tokens)
    target_tokens = _adapt_target_tokens(target_tokens, entry_script)

    if args.dry_run:
        actual_dry = _detect_flag_form(entry_script, "--dry-run")
        if actual_dry:
            return _run_target([*target_tokens, actual_dry, *passthrough], "dry-run", target_env)
        _compile_project()
        print("NanoResearch runner fallback dry-run: syntax check passed.")
        return 0

    if args.quick_eval:
        quick_tokens = _build_quick_eval_tokens(
            target_tokens,
            entry_script,
            passthrough,
            quick_eval_blocked_options,
        )
        return _run_target(quick_tokens, "quick-eval", target_env)

    return _run_target([*target_tokens, *passthrough], "train", target_env)


if __name__ == "__main__":
    raise SystemExit(main())
'''
