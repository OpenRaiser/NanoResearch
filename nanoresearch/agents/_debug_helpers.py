"""Debug agent helpers — syntax check, file rewrite, SLURM fixes, error classification, download."""

from __future__ import annotations

import asyncio
import logging
import re as _re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class _DebugHelpersMixin:
    """Mixin — syntax check, file rewrite, SLURM fixes, error classification, download."""

    def _check_syntax(self, filepath: Path) -> bool:
        """Check if a Python file has valid syntax."""
        try:
            import py_compile
            py_compile.compile(str(filepath), doraise=True)
            return True
        except py_compile.PyCompileError:
            return False
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

            # Robust fence stripping — handles LLM self-correction and multiple blocks
            from nanoresearch.agents._code_utils import _strip_code_fences
            new_content = _strip_code_fences(new_content)

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
            if is_new_file:
                filepath.unlink(missing_ok=True)
            else:
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
            # Try unquoted paths
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
