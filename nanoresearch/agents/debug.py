"""Debug agent — diagnoses failed jobs, applies fixes, enables retry.

Implements a Claude-Code-style debug loop:
  1. Read error logs + all source files
  2. Send full context to LLM for diagnosis
  3. LLM returns structured file patches
  4. Apply patches to source files
  5. Verify patches didn't introduce new syntax errors
  6. Return control to ExecutionAgent for re-submission
"""

from __future__ import annotations

import logging
import os
import re as _re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from nanoresearch.agents.base import BaseResearchAgent
from nanoresearch.agents._debug_helpers import _DebugHelpersMixin
from nanoresearch.schemas.manifest import PipelineStage

logger = logging.getLogger(__name__)

MAX_DEBUG_ROUNDS = 20


class DebugAgent(_DebugHelpersMixin, BaseResearchAgent):
    """Reads error context, diagnoses failures, and patches code."""

    stage = PipelineStage.EXPERIMENT  # reuse experiment stage

    @property
    def stage_config(self):
        """Use code_gen model config for debugging (same model that writes code)."""
        return self.config.for_stage("code_gen")

    async def run(self, **inputs: Any) -> dict[str, Any]:
        """Diagnose a failed job and return file patches."""
        code_dir = Path(inputs["code_dir"])
        stdout_log = inputs.get("stdout_log", "")
        stderr_log = inputs.get("stderr_log", "")
        job_status = inputs.get("job_status", "FAILED")
        debug_round = inputs.get("debug_round", 1)
        previous_fixes = inputs.get("previous_fixes", [])

        self.log(f"Debug round {debug_round}/{MAX_DEBUG_ROUNDS}: diagnosing {job_status}")

        # Step 1: Read all source files in the experiment directory
        source_files = self._read_all_sources(code_dir)
        self.log(f"Read {len(source_files)} source files")

        # Step 1b: Check if this is a missing-data error (not a code bug)
        error_type, missing_path = self._classify_error(stdout_log, stderr_log)
        if error_type == "data_missing" and missing_path:
            self.log(f"Detected missing data file: {missing_path}")
            downloaded = await self._download_missing_resource(missing_path)
            if downloaded:
                return {
                    "diagnosis": f"Missing data file: {missing_path} (downloaded)",
                    "patches": [],
                    "fixed_files": [],
                    "needs_resubmit": True,
                    "debug_round": debug_round,
                }

        # Step 1c: Collect environment context for infrastructure-aware diagnosis
        env_context = ""
        if self._looks_like_infra_error(stderr_log, stdout_log) or debug_round >= 3:
            self.log("Collecting environment diagnostics for infrastructure-aware diagnosis")
            try:
                env_context = self._probe_env_sync(code_dir)
            except Exception as exc:
                logger.debug("Environment probe failed: %s", exc)
                env_context = "(environment probe failed)"

        # Step 2: Ask LLM to diagnose and generate patches
        diagnosis, patches = await self._diagnose_and_patch(
            source_files, stdout_log, stderr_log, job_status,
            debug_round, previous_fixes, env_context=env_context,
        )
        self.log(f"Diagnosis: {diagnosis[:200]}")
        self.log(f"Generated {len(patches)} patches")

        # Step 3: Apply patches with rollback on syntax errors
        fixed_files = []
        applied_patches = []
        for patch in patches:
            filepath = code_dir / patch.get("file", "")
            backup = filepath.read_text(errors="replace") if filepath.exists() else None

            success = self._apply_patch(code_dir, patch)
            if success:
                if filepath.suffix == ".py" and not self._check_syntax(filepath):
                    self.log(f"Patch to {patch['file']} introduced syntax error, rolling back")
                    if backup is not None:
                        filepath.write_text(backup)
                    rewrite_ok = await self._rewrite_file(code_dir, patch["file"], source_files, stderr_log)
                    if rewrite_ok:
                        fixed_files.append(patch["file"])
                        applied_patches.append({**patch, "description": f"(rewritten) {patch.get('description', '')}"})
                else:
                    fixed_files.append(patch["file"])
                    applied_patches.append(patch)
                    self.log(f"Patched: {patch['file']} — {patch.get('description', '')}")
            else:
                self.log(f"Patch match failed for {patch['file']}, trying full rewrite")
                rewrite_ok = await self._rewrite_file(code_dir, patch["file"], source_files, stderr_log)
                if rewrite_ok:
                    fixed_files.append(patch["file"])
                    applied_patches.append({**patch, "description": f"(rewritten) {patch.get('description', '')}"})

        # Step 4: Check if SLURM script itself needs fixing
        slurm_fixed = self._fix_common_slurm_issues(code_dir)
        if slurm_fixed:
            fixed_files.append("run_train.slurm")
            self.log("Fixed common SLURM script issues")

        needs_resubmit = True
        if not fixed_files and not patches:
            needs_resubmit = False

        result = {
            "diagnosis": diagnosis,
            "patches": applied_patches,
            "fixed_files": fixed_files,
            "needs_resubmit": needs_resubmit,
            "debug_round": debug_round,
        }

        self.workspace.write_json(f"plans/debug_round_{debug_round}.json", result)
        return result

    # ------------------------------------------------------------------
    # Environment probing (infrastructure-level diagnostics)
    # ------------------------------------------------------------------

    def _probe_env_sync(self, code_dir: Path) -> str:
        """Collect environment info synchronously for inclusion in LLM context."""
        lines: list[str] = []

        def _run(cmd: str, timeout: int = 10) -> str:
            try:
                r = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True,
                    timeout=timeout, cwd=str(code_dir),
                )
                return (r.stdout.strip() or r.stderr.strip())[:500]
            except Exception:
                return "(unavailable)"

        python_cmd = "python"
        if os.name == "nt":
            venv_py = code_dir / ".venv" / "Scripts" / "python.exe"
        else:
            venv_py = code_dir / ".venv" / "bin" / "python"
        if venv_py.exists():
            python_cmd = str(venv_py)

        lines.append(f"Python: {_run(f'{python_cmd} --version')}")
        _sys_exe_cmd = '{} -c "import sys; print(sys.executable)"'.format(python_cmd)
        lines.append(f"Python path: {_run(_sys_exe_cmd)}")

        if shutil.which("nvidia-smi"):
            lines.append(f"GPU: {_run('nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader,nounits')}")
            smi_header = _run("nvidia-smi 2>&1 | head -3")
            m = _re.search(r"CUDA Version:\s*([\d.]+)", smi_header)
            if m:
                lines.append(f"CUDA Driver Version: {m.group(1)}")
        else:
            lines.append("GPU: nvidia-smi not found")

        torch_script = (
            "import torch; "
            "print('torch=' + torch.__version__); "
            "print('cuda_avail=' + str(torch.cuda.is_available())); "
            "print('cuda_ver=' + str(torch.version.cuda)); "
            "print('devices=' + str(torch.cuda.device_count()))"
        )
        torch_info = _run(f'{python_cmd} -c "{torch_script}"')
        lines.append(f"Torch: {torch_info}")

        pkg_check = _run(
            f"{python_cmd} -m pip list --format=columns 2>/dev/null "
            "| grep -iE 'torch|transformers|datasets|numpy|scipy|scikit|tensorflow|jax' "
            "| head -15"
        )
        if pkg_check and pkg_check != "(unavailable)":
            lines.append(f"Key packages:\n{pkg_check}")

        return "\n".join(lines)

    _INFRA_ERROR_PATTERNS = [
        "dll load failed", "winerror", "oserror: libcu", "cuda",
        "glibc", "modulenotfounderror", "no module named",
        "command timed out", "timed out after", "out of memory",
        "oom", "permission denied", "torch.cuda.is_available",
    ]

    def _looks_like_infra_error(self, stderr_log: str, stdout_log: str) -> bool:
        """Heuristic: does the error look like an infrastructure problem?"""
        combined = (stderr_log + stdout_log).lower()
        return any(pat in combined for pat in self._INFRA_ERROR_PATTERNS)

    def _read_all_sources(self, code_dir: Path) -> dict[str, str]:
        """Read all Python and shell files in the experiment directory."""
        sources = {}
        for ext in ("*.py", "*.sh", "*.slurm", "*.txt", "*.cfg", "*.yaml", "*.yml"):
            for f in code_dir.rglob(ext):
                if f.is_file() and f.stat().st_size < 100_000:
                    try:
                        sources[f.name] = f.read_text(errors="replace")
                    except Exception as exc:
                        logger.debug("Failed to read source snapshot %s: %s", f, exc)
        return sources

    async def _diagnose_and_patch(
        self,
        source_files: dict[str, str],
        stdout_log: str,
        stderr_log: str,
        job_status: str,
        debug_round: int,
        previous_fixes: list[dict],
        env_context: str = "",
    ) -> tuple[str, list[dict]]:
        """Send full context to LLM, get diagnosis + structured patches."""
        source_parts: list[str] = []
        for filename, content in sorted(source_files.items()):
            numbered = "\n".join(
                f"{i+1:4d} | {line}"
                for i, line in enumerate(content.split("\n"))
            )
            source_parts.append(f"\n{'='*60}\n# FILE: {filename}\n{'='*60}\n{numbered}\n")
        source_listing = "".join(source_parts)

        stdout_tail = stdout_log[-5000:] if stdout_log else "(empty)"
        stderr_tail = stderr_log[-3000:] if stderr_log else "(empty)"

        fix_history = ""
        if previous_fixes:
            fix_parts: list[str] = ["\n\nPrevious debug attempts that did NOT fix the problem:\n"]
            for i, fix in enumerate(previous_fixes, 1):
                fix_parts.append(f"\nRound {i}: {fix.get('diagnosis', 'N/A')[:300]}\n")
                for p in fix.get("patches", []):
                    fix_parts.append(f"  - Patched {p.get('file', '?')}: {p.get('description', '?')}\n")
            fix_parts.append("\nDo NOT repeat the same fixes. Try a different approach.\n")
            fix_history = "".join(fix_parts)

        system_prompt = self._build_diagnosis_system_prompt()

        env_block = ""
        if env_context:
            env_block = f"\n=== ENVIRONMENT INFO ===\n{env_context}\n"

        user_prompt = f"""Job Status: {job_status}
Debug Round: {debug_round}/{MAX_DEBUG_ROUNDS}
{env_block}
=== STDERR ===
{stderr_tail}

=== STDOUT (last 5000 chars) ===
{stdout_tail}

=== ALL SOURCE FILES ===
{source_listing}
{fix_history}
FIRST classify the error (infrastructure / configuration / code), THEN diagnose the root cause and generate patches. Return JSON only."""

        result = await self.generate_json(system_prompt, user_prompt)

        diagnosis = result.get("diagnosis", "Unknown error")
        patches = result.get("patches", [])

        valid_patches = []
        for p in patches:
            if isinstance(p, dict) and "file" in p and "old" in p and "new" in p:
                valid_patches.append(p)

        return diagnosis, valid_patches

    @staticmethod
    def _build_diagnosis_system_prompt() -> str:
        """Build the system prompt for diagnosis (extracted for readability)."""
        return """You are an expert ML engineer debugging a failed training job.

Your task:
1. FIRST classify the error: is it infrastructure (Layer 1), configuration (Layer 2), or code (Layer 3)?
2. Analyze the error logs, source code, AND environment info to identify the ROOT CAUSE
3. Generate precise file patches to fix the issue

## Error Classification (check in this order):

**Layer 1 — Environment / Infrastructure:**
- DLL load failed / OSError: libcudart / WinError -> Wrong PyTorch build (CPU vs CUDA mismatch)
- ModuleNotFoundError -> Package not installed or wrong python env
- CUDA out of memory -> reduce batch_size/model_size in code
- torch.cuda.is_available()==False when GPU exists -> CPU-only torch was installed

**Layer 2 — Configuration / Timeout:**
- "Command timed out" -> NOT a code bug! Process was still running.
- API rate limit (429) -> add retry/backoff

**Layer 3 — Code bugs (fix only AFTER ruling out Layer 1-2):**
- SyntaxError, NameError, TypeError, ValueError, etc.

Rules:
- Focus on the actual error, not style issues
- Each patch must specify the EXACT old text to replace
- Only patch what's necessary to fix the error
- IMPORTANT: The "old" field must be an EXACT substring of the file content

Return JSON:
{
  "diagnosis": "Clear explanation of the root cause",
  "error_layer": "infrastructure|configuration|code",
  "patches": [
    {
      "file": "filename.py",
      "old": "exact text to find in the file",
      "new": "replacement text with correct indentation",
      "description": "what this patch fixes"
    }
  ]
}"""

    def _apply_patch(self, code_dir: Path, patch: dict) -> bool:
        """Apply a single patch to a file. Returns True if successful."""
        filename = patch["file"]
        old_text = patch["old"]
        new_text = patch["new"]

        filepath = code_dir / filename
        try:
            filepath.resolve().relative_to(code_dir.resolve())
        except ValueError:
            logger.warning(f"Patch target outside code_dir: {filepath}, skipping")
            return False
        if not filepath.exists():
            if not old_text or old_text.strip() == "":
                filepath.parent.mkdir(parents=True, exist_ok=True)
                filepath.write_text(new_text)
                logger.info(f"Created new file: {filepath}")
                return True
            logger.warning(f"Patch target not found: {filepath}, skipping")
            return False

        content = filepath.read_text(errors="replace")

        # Strategy 1: Exact match
        if old_text in content:
            filepath.write_text(content.replace(old_text, new_text, 1))
            return True

        # Strategy 2: Strip trailing whitespace
        def strip_trailing(text: str) -> str:
            return "\n".join(line.rstrip() for line in text.split("\n"))

        content_stripped = strip_trailing(content)
        old_stripped = strip_trailing(old_text)
        if old_stripped in content_stripped:
            filepath.write_text(content_stripped.replace(old_stripped, strip_trailing(new_text), 1))
            return True

        # Strategy 3: Line-by-line fuzzy match
        content_lines = content.split("\n")
        old_lines = old_text.strip().split("\n")
        if len(old_lines) >= 2:
            first_line = old_lines[0].strip()
            last_line = old_lines[-1].strip()
            for i in range(len(content_lines)):
                if first_line and first_line in content_lines[i].strip():
                    for j in range(i + len(old_lines) - 1, min(i + len(old_lines) + 5, len(content_lines))):
                        if last_line and last_line in content_lines[j].strip():
                            new_lines = new_text.rstrip().split("\n")
                            content_lines[i:j+1] = new_lines
                            filepath.write_text("\n".join(content_lines))
                            return True

        # Strategy 4: Single line matching
        if "\n" not in old_text.strip():
            old_line = old_text.strip()
            for i, line in enumerate(content_lines):
                if old_line == line.strip():
                    indent = len(line) - len(line.lstrip())
                    new_lines = new_text.strip().split("\n")
                    new_indented = [" " * indent + nl.strip() if nl.strip() else "" for nl in new_lines]
                    content_lines[i:i+1] = new_indented
                    filepath.write_text("\n".join(content_lines))
                    return True

        logger.warning(f"All patch strategies failed for {filename}")
        return False
