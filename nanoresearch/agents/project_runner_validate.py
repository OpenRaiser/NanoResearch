"""Project runner validation and repair."""

from __future__ import annotations

import json
import platform
import re
import shlex
import shutil
from pathlib import Path
from typing import Any

from .project_runner_core import (
    RUNNER_SCRIPT_NAME,
    RUNNER_CONFIG_NAME,
    ARTIFACT_DIR_NAMES,
    ENTRYPOINT_REL_PATHS,
    _strip_wrapping_quotes,
    is_python_launcher_token,
    _split_command,
    _truncate_shell_chain,
    _extract_env_assignments,
    _unwrap_shell_wrapper,
    normalize_target_spec,
    normalize_target_command,
    _coerce_command_tokens,
    _path_within_root,
    _resolve_command_path,
    _is_path_like_token,
    _runner_target_spec,
    _runner_target_env,
    _unique_workspace_paths,
    _workspace_python_files,
    _workspace_entrypoint_candidates,
    _relative_command_path,
    _shell_join_command,
    _write_runner_assets,
    _repair_target_candidate,
)
from .project_runner_script import _build_runner_script

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


