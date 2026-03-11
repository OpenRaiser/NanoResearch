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
) -> dict[str, Any]:
    """Run a shell command and return stdout/stderr."""
    # Safety: block obviously destructive commands
    if _BLOCKED_COMMANDS.search(command):
        return {"error": f"Command blocked by safety filter: {command[:100]}"}

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
# Infrastructure diagnostic tools
# ---------------------------------------------------------------------------

async def _probe_environment(_base: Path | None = None) -> dict[str, Any]:
    """One-shot environment diagnostic: GPU, Python, CUDA, pip packages, OS.

    This runs several diagnostic commands and returns structured results so the
    agent can reason about environment issues (wrong torch build, CUDA mismatch,
    missing packages) without needing multiple round-trips.
    """

    # Force UTF-8 for subprocess output on Windows
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    async def _run(cmd: str, timeout: int = 15) -> str:
        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(_base) if _base else None,
                env=env,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            out = stdout.decode("utf-8", errors="replace").strip()
            if not out and stderr:
                out = stderr.decode("utf-8", errors="replace").strip()
            return out
        except asyncio.TimeoutError:
            return "(timed out)"
        except Exception as e:
            return f"(error: {e})"

    result: dict[str, Any] = {"os": {}, "python": {}, "gpu": {}, "packages": {}}

    # --- OS info ---
    result["os"]["platform"] = platform.platform()
    result["os"]["system"] = platform.system()
    result["os"]["machine"] = platform.machine()
    if platform.system() == "Linux":
        result["os"]["glibc"] = await _run("ldd --version 2>&1 | head -1")

    # --- Python info ---
    # Use the venv python if we have a base directory with a venv
    python_cmd = "python"
    if _base:
        venv_python = _base / ".venv" / ("Scripts" if os.name == "nt" else "bin") / "python"
        if venv_python.exists():
            python_cmd = str(venv_python)

    result["python"]["executable"] = await _run(f"{python_cmd} -c \"import sys; print(sys.executable)\"")
    result["python"]["version"] = await _run(f"{python_cmd} --version")

    # --- GPU / CUDA info ---
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        result["gpu"]["nvidia_smi"] = await _run(
            "nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,driver_version,temperature.gpu "
            "--format=csv,noheader,nounits"
        )
        result["gpu"]["cuda_version"] = await _run(
            "nvidia-smi 2>&1 | head -3"
        )
    else:
        result["gpu"]["nvidia_smi"] = "(nvidia-smi not found — no GPU or driver not installed)"

    # --- torch status ---
    torch_script = (
        "import torch; "
        "print('version=' + torch.__version__); "
        "print('cuda_available=' + str(torch.cuda.is_available())); "
        "print('cuda_version=' + str(torch.version.cuda)); "
        "print('device_count=' + str(torch.cuda.device_count())); "
        "print('device_name=' + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'))"
    )
    torch_check = await _run(f'{python_cmd} -c "{torch_script}"', timeout=30)
    result["packages"]["torch"] = torch_check

    # --- Key ML packages ---
    pip_list = await _run(
        f'{python_cmd} -m pip list --format=columns 2>{"NUL" if os.name == "nt" else "/dev/null"}',
        timeout=30,
    )
    # Truncate to first 60 lines
    pip_lines = pip_list.split("\n")[:60]
    result["packages"]["pip_list_head"] = "\n".join(pip_lines)

    # --- Conda info ---
    conda = shutil.which("conda")
    if conda:
        result["packages"]["conda_envs"] = await _run("conda info --envs 2>/dev/null | head -20")

    # --- Disk space ---
    if _base:
        if os.name == "nt":
            result["os"]["disk"] = await _run(f'powershell -Command "(Get-PSDrive -Name C).Free / 1GB"')
        else:
            result["os"]["disk"] = await _run(f"df -h {_base} 2>/dev/null | tail -1")

    return result


async def _check_process(pattern: str = "", _base: Path | None = None) -> dict[str, Any]:
    """Check running processes and GPU utilization.

    Useful for diagnosing: is training still alive? is GPU being used?
    is there a zombie process hogging memory?

    Parameters
    ----------
    pattern : str
        Optional grep pattern to filter processes (e.g. 'python', 'train').
        If empty, shows all python processes + GPU processes.
    """

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    async def _run(cmd: str, timeout: int = 15) -> str:
        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(_base) if _base else None,
                env=env,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            return stdout.decode("utf-8", errors="replace").strip()
        except asyncio.TimeoutError:
            return "(timed out)"
        except Exception as e:
            return f"(error: {e})"

    result: dict[str, Any] = {}

    # --- GPU processes ---
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        result["gpu_processes"] = await _run("nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader,nounits 2>/dev/null")
        result["gpu_utilization"] = await _run(
            "nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total "
            "--format=csv,noheader,nounits"
        )

    # --- System processes ---
    if os.name == "nt":
        # Windows
        if pattern:
            result["processes"] = await _run(
                f'powershell -Command "Get-Process | Where-Object {{$_.ProcessName -match \'{pattern}\'}} '
                f'| Select-Object Id,ProcessName,CPU,WorkingSet64 | Format-Table -AutoSize | Out-String"'
            )
        else:
            result["processes"] = await _run(
                'powershell -Command "Get-Process python* | Select-Object Id,ProcessName,CPU,WorkingSet64 '
                '| Format-Table -AutoSize | Out-String"'
            )
    else:
        # Linux/Mac
        if pattern:
            result["processes"] = await _run(f"ps aux | grep -i '{pattern}' | grep -v grep | head -30")
        else:
            result["processes"] = await _run("ps aux | grep -i python | grep -v grep | head -30")

    # --- Memory ---
    if os.name == "nt":
        result["memory"] = await _run(
            'powershell -Command "$os = Get-CimInstance Win32_OperatingSystem; '
            '$used = ($os.TotalVisibleMemorySize - $os.FreePhysicalMemory) / 1MB; '
            '$total = $os.TotalVisibleMemorySize / 1MB; '
            'Write-Output \\\"Used: $([math]::Round($used,1)) GB / Total: $([math]::Round($total,1)) GB\\\""'
        )
    else:
        result["memory"] = await _run("free -h 2>/dev/null | head -3")

    return result


# ---------------------------------------------------------------------------
# Registry builder
# ---------------------------------------------------------------------------

def build_experiment_tools(work_dir: Path | None = None) -> ToolRegistry:
    """Create a ToolRegistry with all experiment tools.

    Parameters
    ----------
    work_dir : Path, optional
        Base working directory.  If given, relative paths in tool calls
        are resolved against this directory.
    """
    registry = ToolRegistry()

    # Bind work_dir as _base to each handler so relative paths resolve correctly
    _rf = functools.partial(_read_file, _base=work_dir)
    _wf = functools.partial(_write_file, _base=work_dir)
    _ld = functools.partial(_list_dir, _base=work_dir)
    _rc = functools.partial(_run_command, _base=work_dir)
    _sf = functools.partial(_search_files, _base=work_dir)
    _gc = functools.partial(_grep_content, _base=work_dir)
    _pe = functools.partial(_probe_environment, _base=work_dir)
    _cp = functools.partial(_check_process, _base=work_dir)

    registry.register(ToolDefinition(
        name="read_file",
        description=(
            "Read a file and return its text content. "
            "For large files (>200KB) the middle is truncated. "
            "Use this to inspect source code, configs, logs, results, etc."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative file path",
                },
            },
            "required": ["path"],
        },
        handler=_rf,
    ))

    registry.register(ToolDefinition(
        name="write_file",
        description=(
            "Write text content to a file. Creates parent directories automatically. "
            "Use this to create Python scripts, SLURM batch scripts, config files, etc."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path to write to",
                },
                "content": {
                    "type": "string",
                    "description": "Full file content to write",
                },
            },
            "required": ["path", "content"],
        },
        handler=_wf,
    ))

    registry.register(ToolDefinition(
        name="list_dir",
        description=(
            "List files and subdirectories in a directory. "
            "Shows file sizes and directory item counts. "
            "Use this to explore project structure, find output files, etc."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to list",
                },
            },
            "required": ["path"],
        },
        handler=_ld,
    ))

    registry.register(ToolDefinition(
        name="run_command",
        description=(
            "Run a shell command and return stdout/stderr. "
            "Use for: checking environment (whoami, hostname, nvidia-smi, which python, conda info), "
            "installing packages (pip install, conda install), "
            "running Python scripts (python train.py), "
            "SLURM operations (sbatch, squeue, scancel, sinfo), "
            "git operations, and any other shell commands. "
            "Timeout defaults to 120s, max 1800s."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default 120, max 1800)",
                },
                "workdir": {
                    "type": "string",
                    "description": "Working directory for the command (optional)",
                },
            },
            "required": ["command"],
        },
        handler=_rc,
    ))

    registry.register(ToolDefinition(
        name="search_files",
        description=(
            "Search for files matching a glob pattern recursively. "
            "Example: search_files('*.py', '/home/user/project') "
            "or search_files('*.json', './results')"
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern (e.g., '*.py', 'config/*.yaml')",
                },
                "path": {
                    "type": "string",
                    "description": "Base directory to search in (default: current dir)",
                },
            },
            "required": ["pattern"],
        },
        handler=_sf,
    ))

    registry.register(ToolDefinition(
        name="grep_content",
        description=(
            "Search file contents by regex pattern (like grep -rn). "
            "Returns matching lines with file:line: prefix. "
            "Example: grep_content('import torch', '.', '*.py')"
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for",
                },
                "path": {
                    "type": "string",
                    "description": "Base directory to search in (default: current dir)",
                },
                "file_glob": {
                    "type": "string",
                    "description": "File pattern to search in (default: '*.py')",
                },
            },
            "required": ["pattern"],
        },
        handler=_gc,
    ))

    registry.register(ToolDefinition(
        name="probe_environment",
        description=(
            "One-shot environment diagnostic. Returns GPU info (nvidia-smi), "
            "Python version, torch CUDA status, installed packages (pip list), "
            "conda environments, OS info, and disk space. "
            "Use this FIRST when debugging environment/infrastructure issues "
            "(DLL errors, CUDA mismatch, wrong torch version, import failures). "
            "No parameters needed — it auto-detects everything."
        ),
        parameters={
            "type": "object",
            "properties": {},
        },
        handler=_pe,
    ))

    registry.register(ToolDefinition(
        name="check_process",
        description=(
            "Check running processes and GPU utilization. "
            "Shows: GPU processes (what's using VRAM), system python processes, "
            "GPU utilization %, and system memory usage. "
            "Use this to check if training is still alive, if GPU is being used, "
            "or if zombie processes are hogging resources."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Optional process name filter (e.g. 'python', 'train'). Default: show all python processes.",
                },
            },
        },
        handler=_cp,
    ))

    return registry
