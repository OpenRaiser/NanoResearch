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

import asyncio
import json
import logging
import re as _re
import subprocess
from pathlib import Path
from typing import Any

from nanoresearch.agents.base import BaseResearchAgent
from nanoresearch.schemas.manifest import PipelineStage

logger = logging.getLogger(__name__)

MAX_DEBUG_ROUNDS = 20


class DebugAgent(BaseResearchAgent):
    """Reads error context, diagnoses failures, and patches code."""

    stage = PipelineStage.EXPERIMENT  # reuse experiment stage

    @property
    def stage_config(self):
        """Use code_gen model config for debugging (same model that writes code)."""
        return self.config.for_stage("code_gen")

    async def run(self, **inputs: Any) -> dict[str, Any]:
        """Diagnose a failed job and return file patches.

        Inputs:
            code_dir: str — path to experiment directory
            stdout_log: str — SLURM stdout
            stderr_log: str — SLURM stderr
            job_status: str — SLURM final status (FAILED, etc.)
            debug_round: int — current debug iteration (1-based)
            previous_fixes: list[dict] — patches applied in prior rounds

        Returns:
            diagnosis: str — root cause analysis
            patches: list[dict] — [{file, old, new, description}]
            fixed_files: list[str] — files that were patched
            needs_resubmit: bool — whether to resubmit the job
        """
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
            # Download failed — fall through, but tell LLM to simplify

        # Step 2: Ask LLM to diagnose and generate patches
        diagnosis, patches = await self._diagnose_and_patch(
            source_files, stdout_log, stderr_log, job_status,
            debug_round, previous_fixes,
        )
        self.log(f"Diagnosis: {diagnosis[:200]}")
        self.log(f"Generated {len(patches)} patches")

        # Step 3: Apply patches with rollback on syntax errors
        fixed_files = []
        applied_patches = []
        for patch in patches:
            filepath = code_dir / patch.get("file", "")
            # Save backup before patching
            backup = filepath.read_text(errors="replace") if filepath.exists() else None

            success = self._apply_patch(code_dir, patch)
            if success:
                # Verify the patch didn't introduce syntax errors
                if filepath.suffix == ".py" and not self._check_syntax(filepath):
                    self.log(f"Patch to {patch['file']} introduced syntax error, rolling back")
                    if backup is not None:
                        filepath.write_text(backup)
                    # Try asking LLM to rewrite the entire file instead
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

        # Step 4: Check if SLURM script itself needs fixing (common issues)
        slurm_fixed = self._fix_common_slurm_issues(code_dir)
        if slurm_fixed:
            fixed_files.append("run_train.slurm")
            self.log("Fixed common SLURM script issues")

        # Always try to resubmit — even if this round's patches failed,
        # the error might have shifted and next round can catch it
        needs_resubmit = True
        if not fixed_files and not patches:
            # LLM returned zero patches — truly stuck
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

    def _read_all_sources(self, code_dir: Path) -> dict[str, str]:
        """Read all Python and shell files in the experiment directory."""
        sources = {}
        for ext in ("*.py", "*.sh", "*.slurm", "*.txt", "*.cfg", "*.yaml", "*.yml"):
            for f in code_dir.glob(ext):
                if f.is_file() and f.stat().st_size < 100_000:  # skip huge files
                    try:
                        sources[f.name] = f.read_text(errors="replace")
                    except Exception:
                        pass
        return sources

    async def _diagnose_and_patch(
        self,
        source_files: dict[str, str],
        stdout_log: str,
        stderr_log: str,
        job_status: str,
        debug_round: int,
        previous_fixes: list[dict],
    ) -> tuple[str, list[dict]]:
        """Send full context to LLM, get diagnosis + structured patches."""

        # Build source file listing
        source_listing = ""
        for filename, content in sorted(source_files.items()):
            # Add line numbers for easier reference
            numbered = "\n".join(
                f"{i+1:4d} | {line}"
                for i, line in enumerate(content.split("\n"))
            )
            source_listing += f"\n{'='*60}\n# FILE: {filename}\n{'='*60}\n{numbered}\n"

        # Truncate logs to last N chars
        stdout_tail = stdout_log[-5000:] if stdout_log else "(empty)"
        stderr_tail = stderr_log[-3000:] if stderr_log else "(empty)"

        # Build previous fix history
        fix_history = ""
        if previous_fixes:
            fix_history = "\n\nPrevious debug attempts that did NOT fix the problem:\n"
            for i, fix in enumerate(previous_fixes, 1):
                fix_history += f"\nRound {i}: {fix.get('diagnosis', 'N/A')[:300]}\n"
                for p in fix.get("patches", []):
                    fix_history += f"  - Patched {p.get('file', '?')}: {p.get('description', '?')}\n"
            fix_history += "\nDo NOT repeat the same fixes. Try a different approach.\n"

        system_prompt = """You are an expert ML engineer debugging a failed training job on a SLURM GPU cluster.

Your task:
1. Analyze the error logs and source code to identify the ROOT CAUSE of the failure
2. Generate precise file patches to fix the issue

Rules:
- Focus on the actual error, not style issues
- Each patch must specify the EXACT old text to replace (copy-paste from the source, including indentation)
- Only patch what's necessary to fix the error
- If multiple files have related issues (e.g., mismatched imports), fix all of them
- Common issues: import name mismatches, missing dependencies, wrong file paths, conda/env issues
- If the error is a missing DATA file (FileNotFoundError on .csv, .obo, .gaf, .fasta, etc.),
  do NOT write download code (wget, requests, urllib). Instead, REMOVE or SIMPLIFY the code
  that requires the missing file. Comment it out or use a simpler alternative approach.
- COMMON ML PITFALLS you must know how to fix:
  * "size mismatch for classifier" / "Error(s) in loading state_dict" when using from_pretrained
    with different num_labels → add `ignore_mismatched_sizes=True` to from_pretrained() call
  * Data archives (.tar.gz, .zip) not decompressed → add decompression logic before loading
  * `import X; X.func()` where func doesn't exist in X → add the missing factory function to X
  * Script catches exceptions but exits with code 0 → add `sys.exit(1)` in except blocks
- IMPORTANT: The "old" field must be an EXACT substring of the file content (verbatim, preserving whitespace)
- IMPORTANT: The "new" field must have correct Python indentation
- When adding new functions/classes to a file, include them as a separate patch with "old" being the last few lines of the file

Return JSON:
{
  "diagnosis": "Clear explanation of the root cause",
  "patches": [
    {
      "file": "filename.py",
      "old": "exact text to find in the file (copy from source above)",
      "new": "replacement text with correct indentation",
      "description": "what this patch fixes"
    }
  ]
}"""

        user_prompt = f"""SLURM Job Status: {job_status}
Debug Round: {debug_round}/{MAX_DEBUG_ROUNDS}

=== STDERR ===
{stderr_tail}

=== STDOUT (last 5000 chars) ===
{stdout_tail}

=== ALL SOURCE FILES ===
{source_listing}
{fix_history}
Diagnose the root cause and generate patches to fix it. Return JSON only."""

        result = await self.generate_json(system_prompt, user_prompt)

        diagnosis = result.get("diagnosis", "Unknown error")
        patches = result.get("patches", [])

        # Validate patches
        valid_patches = []
        for p in patches:
            if isinstance(p, dict) and "file" in p and "old" in p and "new" in p:
                valid_patches.append(p)

        return diagnosis, valid_patches

    def _apply_patch(self, code_dir: Path, patch: dict) -> bool:
        """Apply a single patch to a file. Returns True if successful."""
        filename = patch["file"]
        old_text = patch["old"]
        new_text = patch["new"]

        filepath = code_dir / filename
        # Security: ensure filepath is within code_dir
        try:
            filepath.resolve().relative_to(code_dir.resolve())
        except ValueError:
            logger.warning(f"Patch target outside code_dir: {filepath}, skipping")
            return False
        if not filepath.exists():
            # File doesn't exist — create it only if old is empty (new file creation)
            if not old_text or old_text.strip() == "":
                filepath.parent.mkdir(parents=True, exist_ok=True)
                filepath.write_text(new_text)
                logger.info(f"Created new file: {filepath}")
                return True
            # LLM wants to patch a non-existent file with specific old_text — skip
            logger.warning(f"Patch target not found: {filepath}, cannot match old_text, skipping")
            return False

        content = filepath.read_text(errors="replace")

        # Strategy 1: Exact match
        if old_text in content:
            filepath.write_text(content.replace(old_text, new_text, 1))
            return True

        # Strategy 2: Strip trailing whitespace from each line and match
        def strip_trailing(text: str) -> str:
            return "\n".join(line.rstrip() for line in text.split("\n"))

        content_stripped = strip_trailing(content)
        old_stripped = strip_trailing(old_text)
        if old_stripped in content_stripped:
            # Find the position in the stripped version and map back
            filepath.write_text(content_stripped.replace(old_stripped, strip_trailing(new_text), 1))
            return True

        # Strategy 3: Line-by-line fuzzy match (match first and last lines)
        content_lines = content.split("\n")
        old_lines = old_text.strip().split("\n")
        if len(old_lines) >= 2:
            first_line = old_lines[0].strip()
            last_line = old_lines[-1].strip()
            for i in range(len(content_lines)):
                if first_line and first_line in content_lines[i].strip():
                    # Look for last line
                    for j in range(i + len(old_lines) - 1, min(i + len(old_lines) + 5, len(content_lines))):
                        if last_line and last_line in content_lines[j].strip():
                            # Found the span — replace it
                            new_lines = new_text.rstrip().split("\n")
                            content_lines[i:j+1] = new_lines
                            filepath.write_text("\n".join(content_lines))
                            return True

        # Strategy 4: If old_text is very short (single line), try line matching
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

    def _check_syntax(self, filepath: Path) -> bool:
        """Check if a Python file has valid syntax."""
        try:
            result = subprocess.run(
                ["python", "-c", f"import py_compile; py_compile.compile('{filepath}', doraise=True)"],
                capture_output=True, text=True, timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return True  # assume OK if check itself fails

    async def _rewrite_file(
        self, code_dir: Path, filename: str, source_files: dict[str, str], error_log: str
    ) -> bool:
        """When patching fails, ask LLM to rewrite the entire file."""
        filepath = code_dir / filename
        is_new_file = not filepath.exists()
        current_content = ""
        if not is_new_file:
            current_content = filepath.read_text(errors="replace")

        # Gather context from other files (imports they expect from this file)
        cross_refs = ""
        for other_name, other_content in source_files.items():
            if other_name == filename:
                continue
            module = filename.replace(".py", "")
            import_lines = [
                line for line in other_content.split("\n")
                if f"from {module} import" in line or f"import {module}" in line
            ]
            if import_lines:
                cross_refs += f"\n{other_name} imports: {'; '.join(import_lines)}"

        system_prompt = (
            "You are a senior ML engineer. "
            + ("Write" if is_new_file else "Rewrite")
            + " the following Python file to fix all errors. "
            "The file must be COMPLETE and RUNNABLE with correct Python syntax and indentation. "
            "Keep the same functionality and class/function names. "
            "Make sure all names that other files import from this file are defined. "
            "Return ONLY the Python code, no markdown fences, no explanation."
        )

        user_prompt = f"""File: {filename} ({'NEW FILE — does not exist yet' if is_new_file else 'existing file'})
Error: {error_log[:1500]}

Other files import from this file:
{cross_refs}

{'This file needs to be CREATED from scratch.' if is_new_file else f'Current content:{chr(10)}{current_content}'}

{'Write' if is_new_file else 'Rewrite'} this file with correct syntax. Return ONLY Python code."""

        try:
            new_content = await self.generate(system_prompt, user_prompt)

            # Strip markdown fences if present
            new_content = new_content.strip()
            if new_content.startswith("```"):
                lines = new_content.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                new_content = "\n".join(lines)

            # Verify syntax before writing
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_text(new_content)
            if self._check_syntax(filepath):
                self.log(f"{'Created' if is_new_file else 'Rewrote'} {filename} successfully")
                return True
            else:
                # Rewrite also has syntax error — restore original or remove
                if is_new_file:
                    filepath.unlink(missing_ok=True)
                    self.log(f"Created {filename} has syntax errors, removed")
                else:
                    filepath.write_text(current_content)
                    self.log(f"Rewrite of {filename} also has syntax errors, rolled back")
                return False

        except Exception as e:
            self.log(f"Rewrite of {filename} failed: {e}")
            filepath.write_text(current_content)
            return False

    def _fix_common_slurm_issues(self, code_dir: Path) -> bool:
        """Fix known SLURM script issues that LLMs commonly produce."""
        fixed = False

        for slurm_file in list(code_dir.glob("*.slurm")) + list(code_dir.glob("*.sh")):
            content = slurm_file.read_text(errors="replace")
            original = content

            # Fix 1: conda activate without proper init
            if "conda activate" in content and "conda.sh" not in content:
                content = content.replace(
                    "source ~/.bashrc\nconda activate",
                    "source ~/anaconda3/etc/profile.d/conda.sh\nconda activate",
                )
                if "source ~/anaconda3/etc/profile.d/conda.sh" not in content:
                    content = content.replace(
                        "conda activate",
                        "source ~/anaconda3/etc/profile.d/conda.sh\nconda activate",
                        1,
                    )

            # Fix 2: Ensure proxy is present for pip install (read from env, no hardcoded creds)
            if "pip install" in content and "proxy" not in content.lower():
                content = content.replace(
                    "pip install",
                    "# Enable proxy for pip (from environment)\n"
                    'export https_proxy="${HTTPS_PROXY:-}"\n'
                    'export http_proxy="${HTTP_PROXY:-}"\n'
                    "pip install",
                    1,
                )

            if content != original:
                slurm_file.write_text(content)
                fixed = True

        return fixed

    def _classify_error(self, stdout_log: str, stderr_log: str) -> tuple[str, str]:
        """Classify error as ('data_missing', path) or ('code_bug', '')."""
        combined = stderr_log + "\n" + stdout_log
        combined_lower = combined.lower()
        data_missing_patterns = [
            "filenotfounderror",
            "no such file or directory",
            "file not found",
            "path does not exist",
        ]
        for pattern in data_missing_patterns:
            if pattern not in combined_lower:
                continue
            # Try quoted paths first
            for m in _re.finditer(
                r"(?:FileNotFoundError|No such file or directory|file not found)[^\n]*?['\"]([^'\"]+)['\"]",
                combined, _re.IGNORECASE,
            ):
                missing = m.group(1)
                if not missing.endswith((".py", ".pyc", ".so", ".pth")):
                    return "data_missing", missing
            # Try unquoted paths: "FileNotFoundError: ... /path/to/file.ext" or "... path/to/file.ext"
            for m in _re.finditer(
                r"(?:FileNotFoundError|file not found)[^\n]*?(\S+\.(?:csv|tsv|obo|gaf|txt|gz|fasta|fa|pdb|pkl|h5|hdf5|json|xml|dat))\b",
                combined, _re.IGNORECASE,
            ):
                missing = m.group(1).rstrip(")")
                return "data_missing", missing
        return "code_bug", ""

    async def _download_missing_resource(self, missing_path: str) -> bool:
        """Try to download a missing data file.

        Security: validates URL scheme (http/https only) and uses shlex.quote.
        """
        import shlex as _shlex
        system_prompt = (
            "Given a missing file path from an ML experiment, determine its download URL. "
            "Return JSON: {\"url\": \"...\", \"filename\": \"...\"} or {\"cannot_download\": true}."
        )
        user_prompt = f"Missing file: {missing_path}\nReturn JSON only."
        try:
            result = await self.generate_json(system_prompt, user_prompt)
            if result.get("cannot_download"):
                return False
            url = result.get("url", "")
            filename = result.get("filename", "") or Path(missing_path).name
            if not url:
                return False
            # Validate URL: only allow http/https
            if not url.startswith(("http://", "https://")):
                self.log(f"Rejecting non-HTTP URL: {url}")
                return False

            dest = Path(missing_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            proc = await asyncio.create_subprocess_shell(
                f"wget -q -O {_shlex.quote(str(dest))} {_shlex.quote(url)}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.communicate(), timeout=600)
            if dest.exists() and dest.stat().st_size > 0:
                self.log(f"Downloaded missing resource: {filename} -> {dest}")
                return True
        except Exception as e:
            self.log(f"Failed to download missing resource: {e}")
        return False

    async def close(self) -> None:
        pass
