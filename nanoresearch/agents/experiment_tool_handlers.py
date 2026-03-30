"""Infrastructure diagnostic tools and tool registry builder for experiment tools."""
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

from nanoresearch.agents.experiment_tools import (
    _read_file,
    _write_file,
    _list_dir,
    _run_command,
    _search_files,
    _grep_content,
    _MAX_READ_SIZE,
    _MAX_WRITE_SIZE,
    _CMD_TIMEOUT_DEFAULT,
    _CMD_TIMEOUT_MAX,
    _MAX_LIST_ENTRIES,
    _MAX_GREP_RESULTS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Infrastructure diagnostic tools
# ---------------------------------------------------------------------------

async def _probe_environment(_base: Path | None = None) -> dict[str, Any]:
    """One-shot environment diagnostic: GPU, Python, CUDA, pip packages, OS."""

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
    python_cmd = "python"
    if _base:
        venv_python = _base / ".venv" / ("Scripts" if os.name == "nt" else "bin") / "python"
        if venv_python.exists():
            python_cmd = str(venv_python)

    result["python"]["executable"] = await _run(f'"{python_cmd}" -c "import sys; print(sys.executable)"')
    result["python"]["version"] = await _run(f'"{python_cmd}" --version')

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
        result["gpu"]["nvidia_smi"] = "(nvidia-smi not found -- no GPU or driver not installed)"

    # --- torch status ---
    torch_script = (
        "import torch; "
        "print('version=' + torch.__version__); "
        "print('cuda_available=' + str(torch.cuda.is_available())); "
        "print('cuda_version=' + str(torch.version.cuda)); "
        "print('device_count=' + str(torch.cuda.device_count())); "
        "print('device_name=' + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'))"
    )
    torch_check = await _run(f'"{python_cmd}" -c "{torch_script}"', timeout=30)
    result["packages"]["torch"] = torch_check

    # --- Key ML packages ---
    pip_list = await _run(
        f'"{python_cmd}" -m pip list --format=columns 2>{"NUL" if os.name == "nt" else "/dev/null"}',
        timeout=30,
    )
    pip_lines = pip_list.split("\n")[:60]
    result["packages"]["pip_list_head"] = "\n".join(pip_lines)

    # --- Conda info ---
    conda = shutil.which("conda")
    if conda:
        _null = "NUL" if os.name == "nt" else "/dev/null"
        conda_out = await _run(f"conda info --envs 2>{_null}", timeout=30)
        conda_lines = conda_out.split("\n")[:20]
        result["packages"]["conda_envs"] = "\n".join(conda_lines)

    # --- Disk space ---
    if _base:
        if os.name == "nt":
            result["os"]["disk"] = await _run(f'powershell -Command "(Get-PSDrive -Name C).Free / 1GB"')
        else:
            result["os"]["disk"] = await _run(f"df -h {_base} 2>/dev/null | tail -1")

    return result


async def _check_process(pattern: str = "", _base: Path | None = None) -> dict[str, Any]:
    """Check running processes and GPU utilization."""

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

    # Sanitize pattern to prevent shell injection
    pattern = re.sub(r"[^a-zA-Z0-9_. -]", "", pattern)[:100]

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

def build_experiment_tools(
    work_dir: Path | None = None,
    allowed_envs: frozenset[str] | None = None,
) -> ToolRegistry:
    """Create a ToolRegistry with all experiment tools."""
    registry = ToolRegistry()

    _rf = functools.partial(_read_file, _base=work_dir)
    _wf = functools.partial(_write_file, _base=work_dir)
    _ld = functools.partial(_list_dir, _base=work_dir)
    _rc = functools.partial(_run_command, _base=work_dir, _allowed_envs=allowed_envs)
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
                "path": {"type": "string", "description": "Absolute or relative file path"},
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
                "path": {"type": "string", "description": "File path to write to"},
                "content": {"type": "string", "description": "Full file content to write"},
            },
            "required": ["path", "content"],
        },
        handler=_wf,
    ))

    registry.register(ToolDefinition(
        name="list_dir",
        description=(
            "List files and subdirectories in a directory. "
            "Shows file sizes and directory item counts."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path to list"},
            },
            "required": ["path"],
        },
        handler=_ld,
    ))

    registry.register(ToolDefinition(
        name="run_command",
        description=(
            "Run a shell command and return stdout/stderr. "
            "Use for: checking environment, installing packages, "
            "running Python scripts, SLURM operations, git operations. "
            "Timeout defaults to 120s, max 1800s."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
                "timeout": {"type": "integer", "description": "Timeout in seconds (default 120, max 1800)"},
                "workdir": {"type": "string", "description": "Working directory for the command (optional)"},
            },
            "required": ["command"],
        },
        handler=_rc,
    ))

    registry.register(ToolDefinition(
        name="search_files",
        description=(
            "Search for files matching a glob pattern recursively. "
            "Example: search_files('*.py', '/home/user/project')"
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern (e.g., '*.py', 'config/*.yaml')"},
                "path": {"type": "string", "description": "Base directory to search in (default: current dir)"},
            },
            "required": ["pattern"],
        },
        handler=_sf,
    ))

    registry.register(ToolDefinition(
        name="grep_content",
        description=(
            "Search file contents by regex pattern (like grep -rn). "
            "Returns matching lines with file:line: prefix."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern to search for"},
                "path": {"type": "string", "description": "Base directory to search in (default: current dir)"},
                "file_glob": {"type": "string", "description": "File pattern to search in (default: '*.py')"},
            },
            "required": ["pattern"],
        },
        handler=_gc,
    ))

    registry.register(ToolDefinition(
        name="probe_environment",
        description=(
            "One-shot environment diagnostic. Returns GPU info, "
            "Python version, torch CUDA status, installed packages, "
            "conda environments, OS info, and disk space. "
            "Use this FIRST when debugging environment/infrastructure issues."
        ),
        parameters={"type": "object", "properties": {}},
        handler=_pe,
    ))

    registry.register(ToolDefinition(
        name="check_process",
        description=(
            "Check running processes and GPU utilization. "
            "Shows GPU processes, system python processes, "
            "GPU utilization %, and system memory usage."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Optional process name filter (e.g. 'python', 'train')."},
            },
        },
        handler=_cp,
    ))

    # ── Think Tool: structured reflection at decision points ──
    registry.register(ToolDefinition(
        name="think",
        description=(
            "Structured reflection tool. Use at key decision points to reason "
            "through your approach BEFORE acting. Call this when:\n"
            "- Before starting: What do I know? What's the plan?\n"
            "- After results: What did I learn? Is this expected?\n"
            "- Choosing between options: What are the trade-offs?\n"
            "- When stuck: What went wrong? What should I try differently?\n"
            "- Before concluding: Is the evidence sufficient?\n"
            "The thought is recorded but no action is taken."
        ),
        parameters={
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": (
                        "Your structured reflection. Consider: "
                        "(1) Progress — what's done, what remains; "
                        "(2) Evidence quality — would a reviewer accept this?; "
                        "(3) Strategy — continue, adjust, or pivot?; "
                        "(4) Resources — runtime estimate, memory constraints."
                    ),
                },
            },
            "required": ["thought"],
        },
        handler=_think,
    ))

    return registry


async def _think(thought: str, **_kwargs: Any) -> dict[str, Any]:
    """Record a structured thought. No side effects — just confirms receipt."""
    logger.info("[THINK] %s", thought[:200])
    return {"status": "recorded", "thought_length": len(thought)}
