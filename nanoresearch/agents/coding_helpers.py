"""Coding agent helper mixin: SLURM scripts, requirements, path validation, import fixes."""
from __future__ import annotations

import json
import logging
import re as _re
from pathlib import Path
from typing import Any

from nanoresearch.agents._code_utils import _strip_code_fences

logger = logging.getLogger(__name__)


class _CodingHelpersMixin:
    """Mixin -- SLURM generation, requirements, path validation, import fix."""

    async def _generate_slurm_script(
        self,
        code_plan: dict,
        blueprint: dict,
        code_dir: Path,
        train_command: str,
    ) -> str:
        """Generate a SLURM batch script for training."""
        compute = blueprint.get("compute_requirements", {})
        try:
            num_gpus = min(int(compute.get("num_gpus", 1)), 4)
        except (ValueError, TypeError):
            num_gpus = 1
        project_name = code_plan.get("project_name", "experiment")

        partition = getattr(self.config, "slurm_partition", "gpu") or "gpu"
        estimated_time = getattr(self.config, "slurm_default_time", "30-00:00:00")
        conda_env = getattr(self.config, "experiment_conda_env", None)

        script = f"""#!/bin/bash
#SBATCH --job-name={project_name[:15]}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:{num_gpus}
#SBATCH --time={estimated_time}
#SBATCH --output={code_dir}/logs/slurm_%j.out
#SBATCH --error={code_dir}/logs/slurm_%j.err

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"
echo "========================================"

# Setup environment -- auto-detect conda location
CONDA_SH="$HOME/anaconda3/etc/profile.d/conda.sh"
[ ! -f "$CONDA_SH" ] && CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
[ ! -f "$CONDA_SH" ] && CONDA_SH="$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"
source "$CONDA_SH" 2>/dev/null || true
"""
        if conda_env:
            script += f'conda activate {conda_env}\n'
        else:
            script += """if conda env list 2>/dev/null | grep -q "^torch "; then
    conda activate torch
elif conda env list 2>/dev/null | grep -q "^nanoresearch "; then
    conda activate nanoresearch
else
    echo "Warning: no suitable conda env found, using base"
fi
"""
        script += f"""

# Enable proxy for downloading models/data (read from environment)
export https_proxy="${{HTTPS_PROXY:-}}"
export http_proxy="${{HTTP_PROXY:-}}"

# Create output directories
mkdir -p {code_dir}/results
mkdir -p {code_dir}/checkpoints
mkdir -p {code_dir}/logs

# Install requirements
cd {code_dir}
pip install -r requirements.txt --quiet 2>/dev/null || true

# Run training
echo "Starting training..."
{train_command}

EXIT_CODE=$?

echo "========================================"
echo "End: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================"

exit $EXIT_CODE
"""
        return script

    def _format_resource_paths(
        self, resources: list[dict], data_dir: str, models_dir: str
    ) -> str:
        """Format downloaded resource paths for inclusion in prompts."""
        lines = []
        if data_dir:
            lines.append(f"Data directory (for ALL dataset downloads): {data_dir}")
        if models_dir:
            lines.append(f"Models directory: {models_dir}")
        lines.append("")

        available = []
        unavailable = []

        for r in resources:
            status = r.get("status", "unknown")
            path = r.get("path", "N/A")
            name = r.get("name", "unknown")
            rtype = r.get("type", "unknown")
            size = r.get("size_bytes", 0)

            if status in ("downloaded", "full", "config_only"):
                size_str = f" ({size / 1024 / 1024:.1f} MB)" if size else ""
                available.append(f"  - [{rtype}] {name}: {path}{size_str}")
                if r.get("files"):
                    for f in r["files"][:10]:
                        available.append(f"      - {f}")
            else:
                unavailable.append(f"  - [{rtype}] {name}: NOT AVAILABLE ({r.get('error', status)})")

        lines.append("=== AVAILABLE (you may ONLY use these) ===")
        lines.extend(available if available else ["  (none)"])
        lines.append("")
        lines.append("=== NOT AVAILABLE (must be DOWNLOADED at runtime in code -- NEVER use synthetic data) ===")
        lines.extend(unavailable if unavailable else ["  (none)"])

        return "\n".join(lines)

    async def _generate_requirements(self, code_plan: dict) -> str:
        """Generate requirements.txt from code plan."""
        deps = code_plan.get("dependencies", [])
        if not deps:
            deps = ["torch", "numpy", "pandas", "scikit-learn", "matplotlib", "tqdm"]

        def _pkg_base(spec: str) -> str:
            return _re.split(r'[>=<!~\[]', spec)[0].strip().lower()

        existing_bases = {_pkg_base(d) for d in deps}
        essential = {"torch", "numpy", "datasets", "requests", "Pillow"}
        for d in essential:
            if d.lower() not in existing_bases:
                deps.append(d)

        seen_bases: set[str] = set()
        deduped: list[str] = []
        for d in deps:
            base = _pkg_base(d)
            if base not in seen_bases:
                seen_bases.add(base)
                deduped.append(d)

        return "\n".join(sorted(deduped)) + "\n"

    def _generate_environment_yaml(self, code_plan: dict) -> str:
        """Generate a lightweight conda environment file from the code plan."""
        deps = code_plan.get("dependencies", [])
        if not deps:
            deps = ["torch", "numpy", "pandas", "scikit-learn", "matplotlib", "tqdm"]

        lines = [
            "name: nanoresearch-auto",
            "channels:",
            "  - conda-forge",
            "  - pytorch",
            "  - defaults",
            "dependencies:",
            "  - python=3.10",
            "  - pip",
            "  - pip:",
        ]
        for dep in sorted(set(deps)):
            lines.append(f"      - {dep}")
        return "\n".join(lines) + "\n"

    def _validate_data_paths(
        self, code_dir: Path, downloaded_resources: list[dict],
        data_dir: str, models_dir: str,
    ) -> list[dict]:
        """Scan generated code for file path references and check they exist."""
        valid_paths: set[str] = set()
        for d in (data_dir, models_dir):
            if d:
                valid_paths.add(d)
        for r in downloaded_resources:
            if r.get("status") in ("downloaded", "full", "config_only"):
                p = r.get("path", "")
                if p:
                    valid_paths.add(p)
                    valid_paths.add(str(Path(p).parent))

        valid_paths.add(str(code_dir))

        path_patterns = [
            r'''open\s*\(\s*[f]?['"](\/[^'"]+)['"]''',
            r'''Path\s*\(\s*[f]?['"](\/[^'"]+)['"]''',
            r'''pd\.read_csv\s*\(\s*[f]?['"](\/[^'"]+)['"]''',
            r'''pd\.read_table\s*\(\s*[f]?['"](\/[^'"]+)['"]''',
            r'''default\s*=\s*['"](\/[^'"]+)['"]''',
            r'''["\'](\/.+?(?:\.csv|\.tsv|\.obo|\.gaf|\.txt|\.gz|\.fasta|\.fa|\.pdb|\.pkl|\.h5|\.hdf5))["\']''',
        ]

        issues = []
        seen: set[tuple[str, str]] = set()
        for py_file in code_dir.rglob("*.py"):
            content = py_file.read_text(errors="replace")
            rel_name = str(py_file.relative_to(code_dir))
            for pattern in path_patterns:
                for match in _re.finditer(pattern, content):
                    ref_path = match.group(1)
                    if not ref_path or not ref_path.startswith("/"):
                        continue
                    key = (rel_name, ref_path)
                    if key in seen:
                        continue
                    seen.add(key)
                    if Path(ref_path).exists():
                        continue
                    if any(ref_path.startswith(vp) for vp in valid_paths if vp):
                        continue
                    issues.append({"file": rel_name, "path": ref_path})

        return issues

    async def _fix_import_mismatches(self, code_dir: Path, code_plan: dict) -> None:
        """Scan all generated files for cross-file import mismatches and fix them via LLM."""
        from nanoresearch.agents.import_checker import ImportChecker

        checker = ImportChecker(code_dir)
        mismatches = checker.check_imports()
        mismatches = [m for m in mismatches if m.get("type") != "syntax_error"]

        if not mismatches:
            self.log("Import consistency check passed")
            return

        self.log(f"Found {len(mismatches)} import mismatches, asking LLM to fix")

        all_sources = {}
        for py_file in code_dir.rglob("*.py"):
            rel_name = str(py_file.relative_to(code_dir))
            all_sources[rel_name] = py_file.read_text(errors="replace")

        source_listing = ""
        for fname, content in sorted(all_sources.items()):
            source_listing += f"\n# FILE: {fname}\n{content}\n"

        system_prompt = (
            "You are fixing cross-file interface mismatches between Python files in a project. "
            "Some files reference names that don't exist in the target module, either via:\n"
            "1. `from X import missing_name` -- name not defined in X\n"
            "2. `import X; X.missing_name()` -- function/class not defined in X\n\n"
            "Fix this by EITHER:\n"
            "- Adding the missing function/class to the target module (preferred if the caller expects specific behavior)\n"
            "- Renaming the call to match what's already defined\n\n"
            "For factory functions like create_model(), build_model() etc. that don't exist, "
            "ADD them to the target module. The factory function should instantiate and return "
            "the appropriate model/class using the existing definitions in that module.\n"
            "Return JSON with patches."
        )

        mismatch_desc = json.dumps(mismatches, indent=2)
        user_prompt = f"""Import mismatches found:
{mismatch_desc}

Source files:
{source_listing[:15000]}

Return JSON:
{{
  "patches": [
    {{
      "file": "filename.py",
      "old": "exact text to replace",
      "new": "replacement text",
      "description": "what this fixes"
    }}
  ]
}}"""

        try:
            result = await self.generate_json(system_prompt, user_prompt)
            patches = result.get("patches", [])

            for patch in patches:
                filepath = code_dir / patch.get("file", "")
                old_text = patch.get("old", "")
                new_text = patch.get("new", "")
                if filepath.exists() and old_text and new_text:
                    content = filepath.read_text(errors="replace")
                    if old_text in content:
                        filepath.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
                        self.log(f"Fixed import mismatch in {patch['file']}: {patch.get('description', '')}")

        except Exception as e:
            self.log(f"Import fix failed (non-fatal): {e}")

    async def close(self) -> None:
        await super().close()
