"""Universal experiment tools — filesystem, shell, and SLURM operations.

Provides a ToolRegistry that works identically on local machines and SLURM
clusters.  When SLURM is available the LLM can submit batch jobs; otherwise
it falls back to direct subprocess execution.

Tools registered:
  read_file          — read any file (text or binary-as-hex)
  write_file         — create / overwrite a file
  list_dir           — ls with sizes and types
  run_command        — arbitrary shell command (with timeout + safety)
  search_files       — glob pattern search
  grep_content       — search file contents by regex
  probe_environment  — one-shot GPU/Python/CUDA/pip/OS diagnostic
  check_process      — inspect running processes and GPU utilization
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import platform
import re
import shutil
from pathlib import Path
from typing import Any

from nanoresearch.agents.tools import ToolDefinition, ToolRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Safety limits
# ---------------------------------------------------------------------------
_MAX_READ_SIZE = 200_000          # 200 KB text cap
_MAX_WRITE_SIZE = 500_000         # 500 KB write cap
_CMD_TIMEOUT_DEFAULT = 120        # 2 min default
_CMD_TIMEOUT_MAX = 1800           # 30 min ceiling
_MAX_LIST_ENTRIES = 200           # max items from list_dir
_MAX_GREP_RESULTS = 50            # max matches from grep
_BLOCKED_COMMANDS = re.compile(
    r"(\brm\s+-rf\s+[/~.]|\brm\s+-r\s+[/~.]|\bmkfs\b|\bdd\s+if=|\bshutdown\b|\breboot\b|"
    r"\bchmod\s+777\s+[/~]|:\s*\(\)\s*\{[^}]*\|\s*:\s*&|"
    r"\bcurl\b[^;|]*\|\s*(?:ba)?sh\b|\bwget\b[^;|]*\|\s*(?:ba)?sh\b)"
)

# ---------------------------------------------------------------------------
# Environment isolation for ReAct mode
# ---------------------------------------------------------------------------
# conda activate / run -n / run --name (supports both -n env and --name=env)
_CONDA_ACTIVATE_RE = re.compile(
    r"conda\s+(?:activate|run\s+(?:-n|--name)(?:\s*=|\s+))\s*['\"]?(\w[^\s'\"]*)['\"]?",
    re.IGNORECASE,
)
# source activate (bash-specific)
_SOURCE_ACTIVATE_RE = re.compile(
    r"(?:source|\.)\s+activate\s+['\"]?(\w[^\s'\"]*)['\"]?", re.IGNORECASE
)
# conda install/create -n <env> (writing to other envs; supports -n=env)
_CONDA_WRITE_RE = re.compile(
    r"conda\s+(?:install|create|env\s+create)\b[^|;]*(?:-n|--name)(?:\s*=|\s+)\s*['\"]?(\w[^\s'\"]*)['\"]?",
    re.IGNORECASE,
)
# Direct path to env Python: handle both / and \ (Windows)
_CONDA_ENV_PYTHON_RE = re.compile(
    r"(?:anaconda|miniconda|miniforge)\d?[/\\]envs[/\\]([^/\\\s'\"]+)[/\\]",
    re.IGNORECASE,
)
# Valid per-session env name: nanoresearch_ + alphanum/underscore/hyphen only
_PER_SESSION_RE = re.compile(r"^nanoresearch_[a-zA-Z0-9_-]+$")


def _check_env_isolation(
    command: str, allowed_envs: frozenset[str] | None
) -> str | None:
    """Return an error message if command tries to use a disallowed conda env.

    Returns None if the command is safe.
    ``allowed_envs=None`` disables isolation entirely.
    ``allowed_envs=frozenset()`` blocks ALL conda envs except nanoresearch_*.
    """
    if allowed_envs is None:
        return None  # no restriction

    # Normalize Windows backslashes so path regex works reliably
    cmd_normalized = command.replace("\\", "/")

    def _is_allowed(name: str) -> bool:
        # Strip quotes that regex might have captured
        name = name.strip("'\"")
        if name in allowed_envs:
            return True
        # Per-session auto-created envs: strict naming validation
        if _PER_SESSION_RE.match(name):
            return True
        return False

    def _fmt_err(kind: str, env_name: str) -> str:
        safe_name = re.sub(r"[^\w./-]", "_", env_name)[:60]
        return (
            f"Environment isolation: {kind} '{safe_name}'. "
            f"Allowed envs: {sorted(allowed_envs)}. "
            f"Use the allowed env or create a new per-session env (nanoresearch_<name>)."
        )

    # Check conda activate / conda run -n <env>
    for m in _CONDA_ACTIVATE_RE.finditer(command):
        env_name = m.group(1)
        if not _is_allowed(env_name):
            return _fmt_err("cannot activate conda env", env_name)

    # Check source activate <env> (bash)
    for m in _SOURCE_ACTIVATE_RE.finditer(command):
        env_name = m.group(1)
        if not _is_allowed(env_name):
            return _fmt_err("cannot source-activate conda env", env_name)

    # Check conda install/create -n <env> (writing to another env)
    for m in _CONDA_WRITE_RE.finditer(command):
        env_name = m.group(1)
        if not _is_allowed(env_name):
            return _fmt_err("cannot install/create into conda env", env_name)

    # Check direct path to env Python (both / and \ via normalization)
    for m in _CONDA_ENV_PYTHON_RE.finditer(cmd_normalized):
        env_name = m.group(1)
        if not _is_allowed(env_name):
            return _fmt_err("cannot invoke Python from conda env", env_name)

    return None


# ---------------------------------------------------------------------------
# Tool handler functions
# ---------------------------------------------------------------------------

def _resolve(path: str, base: Path | None) -> Path:
    """Resolve a path: if relative and base is set, resolve against base."""
    p = Path(path).expanduser()
    if not p.is_absolute() and base is not None:
        p = base / p
    return p


async def _read_file(path: str, _base: Path | None = None) -> dict[str, Any]:
    """Read a file and return its contents."""
    p = _resolve(path, _base)
    if not p.exists():
        return {"error": f"File not found: {path}"}
    if not p.is_file():
        return {"error": f"Not a file (maybe a directory?): {path}"}
    size = p.stat().st_size
    if size > _MAX_READ_SIZE:
        # Read head + tail for large files
        text = p.read_text(encoding="utf-8", errors="replace")
        head = text[:80_000]
        tail = text[-40_000:]
        return {
            "content": f"{head}\n\n... [{size} bytes total, middle truncated] ...\n\n{tail}",
            "size": size,
            "truncated": True,
        }
    content = p.read_text(encoding="utf-8", errors="replace")
    return {"content": content, "size": size}


async def _write_file(path: str, content: str, _base: Path | None = None) -> dict[str, Any]:
    """Write content to a file. Creates parent directories if needed."""
    if len(content) > _MAX_WRITE_SIZE:
        return {"error": f"Content too large ({len(content)} chars, max {_MAX_WRITE_SIZE})"}
    p = _resolve(path, _base)
    # Safety: restrict writes to within the work directory (if set)
    if _base is not None:
        try:
            p.resolve().relative_to(_base.resolve())
        except ValueError:
            return {"error": f"Path traversal blocked: {path} is outside work directory"}
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return {"status": "ok", "path": str(p), "size": len(content)}
    except OSError as e:
        return {"error": str(e)}


async def _list_dir(path: str, _base: Path | None = None) -> dict[str, Any]:
    """List directory contents with type and size info."""
    p = _resolve(path, _base)
    if not p.exists():
        return {"error": f"Directory not found: {path}"}
    if not p.is_dir():
        return {"error": f"Not a directory: {path}"}
    entries = []
    try:
        for item in sorted(p.iterdir()):
            if item.name.startswith(".") and item.name not in (".env",):
                continue  # skip hidden files by default
            if len(entries) >= _MAX_LIST_ENTRIES:
                entries.append("... (truncated)")
                break
            try:
                if item.is_dir():
                    n_children = sum(1 for _ in item.iterdir())
                    entries.append(f"[DIR]  {item.name}/  ({n_children} items)")
                else:
                    size = item.stat().st_size
                    if size >= 1_000_000:
                        size_str = f"{size / 1_000_000:.1f}MB"
                    elif size >= 1000:
                        size_str = f"{size / 1000:.1f}KB"
                    else:
                        size_str = f"{size}B"
                    entries.append(f"[FILE] {item.name}  ({size_str})")
            except OSError:
                entries.append(f"[?]    {item.name}")
    except PermissionError as e:
        return {"error": str(e)}
    return {"path": str(p), "entries": entries, "count": len(entries)}


async def _run_command(
    command: str,
    timeout: int = _CMD_TIMEOUT_DEFAULT,
    workdir: str = "",
    _base: Path | None = None,
    _allowed_envs: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Run a shell command and return stdout/stderr."""
    # Safety: block obviously destructive commands
    if _BLOCKED_COMMANDS.search(command):
        return {"error": f"Command blocked by safety filter: {command[:100]}"}

    # Environment isolation: block usage of non-allowed conda envs
    env_err = _check_env_isolation(command, _allowed_envs)
    if env_err:
        logger.warning("Environment isolation blocked: %s | cmd: %s", env_err, command[:200])
        return {"error": env_err}

    timeout = min(timeout, _CMD_TIMEOUT_MAX)
    # Resolve workdir: explicit workdir > _base > None (inherit cwd)
    if workdir:
        cwd = workdir
    elif _base is not None:
        cwd = str(_base)
    else:
        cwd = None

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": f"Command timed out after {timeout}s",
            }
        stdout_str = stdout.decode(errors="replace")
        stderr_str = stderr.decode(errors="replace")
        # Truncate large outputs
        if len(stdout_str) > 50_000:
            stdout_str = stdout_str[:30_000] + f"\n\n... [{len(stdout_str)} chars, truncated] ...\n\n" + stdout_str[-10_000:]
        if len(stderr_str) > 20_000:
            stderr_str = stderr_str[:12_000] + f"\n\n... [{len(stderr_str)} chars, truncated] ...\n\n" + stderr_str[-5_000:]
        return {
            "returncode": proc.returncode,
            "stdout": stdout_str,
            "stderr": stderr_str,
        }
    except Exception as e:
        return {"error": str(e)}


async def _search_files(pattern: str, path: str = ".", _base: Path | None = None) -> dict[str, Any]:
    """Search for files matching a glob pattern."""
    p = _resolve(path, _base) if path != "." else (_base or Path("."))
    if not p.exists():
        return {"error": f"Path not found: {path}"}
    matches = []
    for f in p.rglob(pattern):
        if "__pycache__" in str(f) or ".git" in str(f):
            continue
        if len(matches) >= _MAX_LIST_ENTRIES:
            matches.append("... (truncated)")
            break
        try:
            rel = str(f.relative_to(p))
        except ValueError:
            rel = str(f)
        matches.append(rel)
    return {"pattern": pattern, "base": str(p), "matches": matches, "count": len(matches)}


async def _grep_content(
    pattern: str,
    path: str = ".",
    file_glob: str = "*.py",
    _base: Path | None = None,
) -> dict[str, Any]:
    """Search file contents by regex pattern."""
    p = _resolve(path, _base) if path != "." else (_base or Path("."))
    if not p.exists():
        return {"error": f"Path not found: {path}"}
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return {"error": f"Invalid regex: {e}"}
    results = []
    for f in p.rglob(file_glob):
        if "__pycache__" in str(f) or ".git" in str(f):
            continue
        if not f.is_file() or f.stat().st_size > 1_000_000:
            continue
        try:
            content = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for i, line in enumerate(content.split("\n"), 1):
            if regex.search(line):
                try:
                    rel = str(f.relative_to(p))
                except ValueError:
                    rel = str(f)
                results.append(f"{rel}:{i}: {line.rstrip()[:200]}")
                if len(results) >= _MAX_GREP_RESULTS:
                    break
        if len(results) >= _MAX_GREP_RESULTS:
            results.append("... (max results reached)")
            break
    return {"pattern": pattern, "matches": results, "count": len(results)}


# ---------------------------------------------------------------------------
# Infrastructure diagnostic tools and registry builder moved to
# experiment_tool_handlers.py. Re-export build_experiment_tools for
# backward compatibility.
# ---------------------------------------------------------------------------
from .experiment_tool_handlers import (  # noqa: F401
    build_experiment_tools,
    _probe_environment,
    _check_process,
)
