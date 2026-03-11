"""Coding agent — writes runnable experiment code based on cloned repos and blueprint."""

from __future__ import annotations

import asyncio
import json
import logging
import re as _re
from pathlib import Path
from typing import Any

from nanoresearch.agents.base import BaseResearchAgent
from nanoresearch.agents.project_runner import (
    RUNNER_CONFIG_NAME,
    RUNNER_SCRIPT_NAME,
    ensure_project_runner,
)
from nanoresearch.exceptions import LLMError
from nanoresearch.schemas.manifest import PipelineStage

logger = logging.getLogger(__name__)


class CodingAgent(BaseResearchAgent):
    """Generates runnable training code + SLURM scripts based on cloned repos and experiment plan."""

    stage = PipelineStage.CODING

    @property
    def stage_config(self):
        """Use code_gen model config for writing code."""
        return self.config.for_stage("code_gen")

    @staticmethod
    def _default_code_plan_files() -> list[dict[str, Any]]:
        return [
            {
                "path": "train.py",
                "description": "Main training script with argparse, training loop, evaluation, and support for --dry-run / --quick-eval",
                "is_entrypoint": True,
            },
            {"path": "model.py", "description": "Model architecture definition"},
            {"path": "dataset.py", "description": "Dataset loading and preprocessing"},
            {"path": "evaluate.py", "description": "Evaluation metrics and testing"},
            {"path": "config.py", "description": "Default hyperparameters and configuration"},
        ]

    def _normalize_code_plan(self, code_plan: dict[str, Any] | None) -> dict[str, Any]:
        plan = dict(code_plan) if isinstance(code_plan, dict) else {}

        normalized_files: list[dict[str, Any]] = []
        seen_paths: set[str] = set()
        for raw_spec in plan.get("files", []):
            if not isinstance(raw_spec, dict):
                continue
            path = str(raw_spec.get("path") or "").strip().replace("\\", "/")
            if not path or path in seen_paths:
                continue
            seen_paths.add(path)
            normalized_files.append(
                {
                    "path": path,
                    "description": str(raw_spec.get("description") or "").strip(),
                    "is_entrypoint": bool(raw_spec.get("is_entrypoint", False)),
                }
            )

        for default_spec in self._default_code_plan_files():
            if default_spec["path"] in seen_paths:
                continue
            normalized_files.append(dict(default_spec))
            seen_paths.add(default_spec["path"])

        dependencies: list[str] = []
        seen_dependencies: set[str] = set()
        for raw_dependency in plan.get("dependencies", []):
            dependency = str(raw_dependency or "").strip()
            if not dependency or dependency in seen_dependencies:
                continue
            seen_dependencies.add(dependency)
            dependencies.append(dependency)
        if not dependencies:
            dependencies = [
                "torch>=2.0.0",
                "numpy>=1.24.0",
                "pandas>=1.5.0",
                "scikit-learn>=1.2.0",
            ]

        expected_output_files = [
            str(item).strip()
            for item in plan.get("expected_output_files", [])
            if str(item).strip()
        ]
        if not expected_output_files:
            expected_output_files = [
                "results/metrics.json",
                "results/training_log.csv",
                "checkpoints/best_model.pt",
            ]

        normalized = {
            "project_name": str(plan.get("project_name") or "generated_experiment").strip(),
            "description": str(plan.get("description") or "Runnable experiment project.").strip(),
            "python_version": str(plan.get("python_version") or "3.10").strip(),
            "dependencies": dependencies,
            "files": normalized_files,
            "train_command": str(plan.get("train_command") or "python train.py").strip(),
            "expected_output_files": expected_output_files,
        }
        return normalized

    async def run(self, **inputs: Any) -> dict[str, Any]:
        topic: str = inputs["topic"]
        experiment_blueprint: dict = inputs.get("experiment_blueprint", {})
        setup_output: dict = inputs.get("setup_output", {})

        self.log("Starting coding: generating runnable experiment")

        code_dir = self.workspace.path / "experiment"
        code_dir.mkdir(exist_ok=True)

        # Step 1: Design the experiment code plan
        code_plan = await self._design_code_plan(
            topic, experiment_blueprint, setup_output
        )
        self.log(f"Code plan: {len(code_plan.get('files', []))} files")

        # Step 2: Generate each file (parallel for speed)
        valid_specs = [s for s in code_plan.get("files", []) if isinstance(s, dict) and s.get("path")]
        self.log(f"  Generating {len(valid_specs)} files in parallel")
        contents = await asyncio.gather(*(
            self._generate_file(spec, code_plan, experiment_blueprint, setup_output)
            for spec in valid_specs
        ))

        generated_files = []
        for spec, content in zip(valid_specs, contents):
            filepath = spec["path"]
            full_path = code_dir / filepath
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content, encoding="utf-8")
            generated_files.append(str(filepath))
            self.log(f"Generated: {filepath}")

        # Step 2b: Cross-file import consistency check
        await self._fix_import_mismatches(code_dir, code_plan)

        # Step 2c: Validate generated code only references existing data paths
        path_issues = self._validate_data_paths(
            code_dir,
            setup_output.get("downloaded_resources", []),
            setup_output.get("data_dir", ""),
            setup_output.get("models_dir", ""),
        )
        if path_issues:
            self.log(f"Found {len(path_issues)} invalid path references, re-generating affected files")
            affected_files: dict[str, list[str]] = {}
            for issue in path_issues:
                affected_files.setdefault(issue["file"], []).append(issue["path"])

            for filename, bad_paths in affected_files.items():
                file_spec = next(
                    (f for f in code_plan.get("files", []) if f.get("path") == filename),
                    {"path": filename, "description": ""},
                )
                file_spec = {**file_spec, "_path_errors": bad_paths}
                content = await self._generate_file(
                    file_spec, code_plan, experiment_blueprint, setup_output
                )
                (code_dir / filename).write_text(content, encoding="utf-8")
                self.log(f"Re-generated {filename} to fix invalid paths: {bad_paths}")

        original_train_command = code_plan.get("train_command", "python train.py")
        runner_assets = ensure_project_runner(code_dir, original_train_command)
        generated_files.extend([RUNNER_SCRIPT_NAME, RUNNER_CONFIG_NAME])
        self.log("Generated deterministic execution runner")

        # Step 3: Generate SLURM script
        slurm_script = await self._generate_slurm_script(
            code_plan,
            experiment_blueprint,
            code_dir,
            runner_assets["runner_command"],
        )
        slurm_path = code_dir / "run_train.slurm"
        slurm_path.write_text(slurm_script, encoding="utf-8")
        generated_files.append("run_train.slurm")
        self.log("Generated SLURM script")

        # Step 4: Generate requirements.txt
        requirements = await self._generate_requirements(code_plan)
        (code_dir / "requirements.txt").write_text(requirements, encoding="utf-8")
        generated_files.append("requirements.txt")

        # Step 5: Generate environment.yml for optional conda-based execution
        environment_yaml = self._generate_environment_yaml(code_plan)
        (code_dir / "environment.yml").write_text(environment_yaml, encoding="utf-8")
        generated_files.append("environment.yml")

        result = {
            "code_plan": code_plan,
            "generated_files": generated_files,
            "code_dir": str(code_dir),
            "slurm_script": str(slurm_path),
            "train_command": runner_assets["runner_command"],
            "entry_train_command": original_train_command,
            "runner_script": runner_assets["runner_script"],
            "runner_config": runner_assets["runner_config"],
            "requirements_path": str(code_dir / "requirements.txt"),
            "environment_file": str(code_dir / "environment.yml"),
        }

        self.workspace.write_json("plans/coding_output.json", result)
        return result

    async def _design_code_plan(
        self, topic: str, blueprint: dict, setup: dict
    ) -> dict:
        """Design the code structure based on blueprint and cloned repos."""
        code_analysis = setup.get("code_analysis", {})
        cloned_repos = setup.get("cloned_repos", [])
        downloaded_resources = setup.get("downloaded_resources", [])
        data_dir = setup.get("data_dir", "")
        models_dir = setup.get("models_dir", "")

        # Build resource paths info for the LLM
        resource_paths = self._format_resource_paths(downloaded_resources, data_dir, models_dir)

        system_prompt = (
            "You are a senior ML research engineer designing a runnable experiment project. "
            "Based on the experiment blueprint and analysis of existing codebases, "
            "design a complete, runnable training project. The code must:\n"
            "1. Actually run on a GPU cluster via SLURM or locally\n"
            "2. Use PyTorch and standard ML libraries\n"
            "3. Include proper training loop, evaluation, checkpointing\n"
            "4. Log metrics to a results file (JSON or CSV)\n"
            "5. Support command-line arguments for hyperparameters\n"
            "6. If a dataset is listed as AVAILABLE below, use its exact path\n"
            "7. If a dataset is NOT AVAILABLE, the code MUST download it at runtime "
            "using `datasets.load_dataset()`, `torchvision.datasets`, `urllib`, `requests`, "
            "or `git clone` — whichever is appropriate. NEVER generate synthetic/random/fake data.\n"
            "8. All runtime downloads MUST go to the Data directory path below (use as root/cache_dir)\n"
            "9. All file paths must be ABSOLUTE, never relative like ./data/\n"
            "10. NEVER fall back to synthetic data. If a dataset truly cannot be downloaded, "
            "use a publicly available alternative dataset for the same task from HuggingFace Hub.\n"
            "Return JSON only."
        )

        user_prompt = f"""Topic: {topic}

Experiment Blueprint:
- Method: {json.dumps(blueprint.get('proposed_method', {}), indent=2)[:1500]}
- Datasets: {json.dumps(blueprint.get('datasets', []), indent=2)[:500]}
- Metrics: {json.dumps(blueprint.get('metrics', []), indent=2)[:300]}
- Baselines: {json.dumps(blueprint.get('baselines', []), indent=2)[:500]}

=== ALREADY DOWNLOADED DATA & MODELS (use these exact paths) ===
{resource_paths}

Code Analysis from Cloned Repos:
- Best base repo: {code_analysis.get('best_base_repo', 'N/A')}
- Reusable components: {json.dumps(code_analysis.get('reusable_components', []), indent=2)[:1000]}
- Missing components: {json.dumps(code_analysis.get('missing_components', []), indent=2)[:500]}
- Recommended approach: {code_analysis.get('recommended_approach', 'N/A')[:500]}

Available cloned repos: {json.dumps([r['name'] for r in cloned_repos])}

IMPORTANT — DATA HANDLING:
- For AVAILABLE datasets above: load them from their exact paths.
- For NOT AVAILABLE datasets: your code MUST download them at runtime using
  `datasets.load_dataset()`, `torchvision.datasets`, `urllib`, or `requests`.
  Save ALL downloads to the Data directory listed above (as root/cache_dir).
  Example: `datasets.load_dataset('dataset_name', cache_dir=args.data_dir)`
  Example: `torchvision.datasets.CIFAR10(root=args.data_dir, download=True)`
  NEVER use default cache like `~/.cache/` or `./data/`.
- NEVER generate synthetic/random/fake data as a substitute. If the exact dataset
  is unavailable from any public source, use a real alternative from HuggingFace Hub.
- Use ABSOLUTE paths as argparse defaults, never relative paths like ./data/.
- CROSS-FILE CONSISTENCY: If train.py calls `model.create_model()`, then model.py MUST define `def create_model()`.
  Every module.function() call must correspond to an actual function defined in that module.
  Every `from X import Y` must import a name that exists in X.

Design a runnable project. Return JSON:
{{
  "project_name": "experiment_name",
  "description": "one-line description",
  "python_version": "3.10",
  "dependencies": ["torch", "transformers", ...],
  "files": [
    {{
      "path": "train.py",
      "description": "Main training script with argparse, training loop, evaluation, and support for --dry-run / --quick-eval",
      "is_entrypoint": true
    }},
    {{
      "path": "model.py",
      "description": "Model architecture definition"
    }},
    {{
      "path": "dataset.py",
      "description": "Dataset loading and preprocessing"
    }},
    {{
      "path": "evaluate.py",
      "description": "Evaluation metrics and testing"
    }},
    {{
      "path": "config.py",
      "description": "Default hyperparameters and configuration"
    }}
  ],
  "train_command": "python train.py --config config.py --epochs 10",
  "expected_output_files": ["results/metrics.json", "results/training_log.csv", "checkpoints/best_model.pt"]
}}"""

        try:
            result = await self.generate_json(system_prompt, user_prompt)
        except LLMError as exc:
            self.log(f"Code plan JSON parse failed, retrying with minimal schema: {exc}")
            retry_prompt = (
                user_prompt
                + "\n\nPrevious attempt was not valid JSON."
                + "\nRetry with a MINIMAL schema only."
                + "\nRules:"
                + "\n- Keep `files` to at most 5 entries."
                + "\n- Each file entry may only contain `path`, `description`, `is_entrypoint`."
                + "\n- Do NOT include file contents, interfaces, or long explanations."
                + "\n- Output ONLY a single JSON object."
            )
            result = await self.generate_json(system_prompt, retry_prompt)

        if isinstance(result, list) and len(result) == 1 and isinstance(result[0], dict):
            result = result[0]
        return self._normalize_code_plan(result if isinstance(result, dict) else {})

    async def _generate_file(
        self,
        file_spec: dict,
        code_plan: dict,
        blueprint: dict,
        setup: dict,
    ) -> str:
        """Generate a single source file."""
        code_analysis = setup.get("code_analysis", {})
        cloned_repos = setup.get("cloned_repos", [])
        downloaded_resources = setup.get("downloaded_resources", [])
        data_dir = setup.get("data_dir", "")
        models_dir = setup.get("models_dir", "")

        resource_paths = self._format_resource_paths(downloaded_resources, data_dir, models_dir)

        # Find relevant reference code from cloned repos
        reference_code = ""
        for component in code_analysis.get("reusable_components", [])[:3]:
            source_file = component.get("source_file", "")
            if source_file and Path(source_file).exists():
                try:
                    content = Path(source_file).read_text(errors="replace")[:3000]
                    reference_code += f"\n# Reference from {source_file}:\n{content}\n"
                except Exception as exc:
                    logger.debug("Failed to read reference code from %s: %s", source_file, exc)
        if len(reference_code) > 8000:
            reference_code = reference_code[:8000]

        system_prompt = (
            "You are a senior ML engineer writing production-quality research code. "
            "Write COMPLETE, RUNNABLE Python code. No stubs, no TODOs, no placeholders. "
            "The code must actually work when executed. "
            "Use standard PyTorch patterns. Include proper error handling, "
            "logging, and metric tracking. Save results to JSON/CSV files. "
            "On ANY unhandled error in the main training script, call sys.exit(1) — "
            "never let exceptions be silently caught with exit code 0."
        )

        all_files = [f.get("path", "") for f in code_plan.get("files", [])]

        user_prompt = f"""Write the complete code for: {file_spec.get('path', '')}
Description: {file_spec.get('description', '')}

Project structure: {json.dumps(all_files)}
Method: {json.dumps(blueprint.get('proposed_method', {}), indent=2)[:1000]}
Datasets: {json.dumps(blueprint.get('datasets', []), indent=2)[:500]}
Metrics: {json.dumps(blueprint.get('metrics', []), indent=2)[:300]}
Dependencies: {json.dumps(code_plan.get('dependencies', []))}
Train command: {code_plan.get('train_command', 'python train.py')}

=== ALREADY DOWNLOADED DATA & MODELS (use these exact paths) ===
{resource_paths}

{f'Reference code from existing repos:{reference_code}' if reference_code else ''}

IMPORTANT:
- Write COMPLETE, RUNNABLE code. Every function must be fully implemented.
- The training script must save metrics to results/metrics.json after each epoch.
- The training script must save the best model checkpoint.
- Use argparse for CLI arguments with DEFAULTS pointing to the data/model paths above.
- Include a results/ directory for outputs.
- Log training progress (loss, metrics) at each epoch.
- Handle both training and evaluation in the same script or via flags.
- The entry script MUST support `--dry-run` for a lightweight pipeline sanity check.
- The entry script MUST support `--quick-eval` for a tiny end-to-end experiment that writes `results/metrics.json`.
- In `--quick-eval` mode, force a very small subset / a few epochs so it finishes quickly on a local machine.
- CRITICAL: All class/function names used in imports between files MUST be consistent.
  For example, if train.py does `from dataset import MyDataset`, then dataset.py MUST define `class MyDataset`.
  If train.py does `import model; model.create_model(...)`, then model.py MUST define `def create_model(...)`.
  Double-check every cross-file import AND every module.function() call before writing the code.
- For AVAILABLE datasets: use their exact file paths as argparse defaults.
- For NOT AVAILABLE datasets: write code to DOWNLOAD them at runtime.
  * Prefer `datasets.load_dataset('dataset_name', cache_dir=args.data_dir)` from HuggingFace Hub.
  * Or use `torchvision.datasets.XXX(root=args.data_dir, download=True)`.
  * Or use `urllib.request.urlretrieve()` / `requests.get()` for direct URLs.
  * Or use `subprocess.run(['git', 'clone', url, target_dir])` for GitHub repos.
  * Save ALL downloads to args.data_dir (the Data directory path above).
  * NEVER generate synthetic/random/fake data. NEVER use `torch.randn` as training data.
  * If the exact dataset is unavailable on any public source, find a real alternative
    dataset for the same task on HuggingFace Hub and use that instead.
- Do NOT use relative paths like ./data/ or ./models/. Use the exact absolute paths provided.
- COMMON ML PITFALLS — you MUST handle these:
  * When loading pretrained models with a different number of classes, use `ignore_mismatched_sizes=True` in from_pretrained().
  * If data files are archives (.tar.gz, .zip, .tar), decompress them before use. Add decompression logic in dataset loading.
  * The main training script must exit with non-zero code on failure. Use `sys.exit(1)` in except blocks, never silently swallow errors.
  * When using from_pretrained with num_labels different from the pretrained model, ALWAYS pass ignore_mismatched_sizes=True.
  * If loading HuggingFace models with custom num_labels/num_classes, handle weight mismatch gracefully.

Return ONLY the Python code, no markdown fences."""

        # If this is a re-generation due to invalid paths, add explicit warning
        path_errors = file_spec.get("_path_errors", [])
        if path_errors:
            user_prompt += (
                "\n\nWARNING: A previous version of this file referenced these NON-EXISTENT paths:\n"
                + "\n".join(f"  - {p}" for p in path_errors)
                + "\nYou MUST NOT reference these paths. Use ONLY paths from the AVAILABLE list above."
                "\nIf a needed dataset is not in the AVAILABLE list, remove that functionality from the code."
            )

        content = await self.generate(
            system_prompt, user_prompt,
        )

        # Strip markdown fences if present
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            # Remove first and last fence lines
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines)

        return content

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
            num_gpus = min(int(compute.get("num_gpus", 1)), 4)  # cap at 4
        except (ValueError, TypeError):
            num_gpus = 1
        project_name = code_plan.get("project_name", "experiment")

        # BUG-16 fix: read SLURM partition from config instead of hardcoding
        partition = getattr(self.config, "slurm_partition", "gpu")
        # Use estimated time from blueprint, default to 1 day
        estimated_time = compute.get("estimated_time", "1-00:00:00")
        # BUG-17 fix: read conda env from config instead of hardcoding
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

# Setup environment — auto-detect conda location
CONDA_SH="$HOME/anaconda3/etc/profile.d/conda.sh"
[ ! -f "$CONDA_SH" ] && CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
[ ! -f "$CONDA_SH" ] && CONDA_SH="$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"
source "$CONDA_SH" 2>/dev/null || true
"""
        if conda_env:
            script += f'conda activate {conda_env}\n'
        else:
            # Try common env names
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
        """Format downloaded resource paths for inclusion in prompts.

        Splits resources into AVAILABLE and NOT AVAILABLE sections so the LLM
        knows exactly what it can and cannot use.
        """
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
        lines.append("=== NOT AVAILABLE (must be DOWNLOADED at runtime in code — NEVER use synthetic data) ===")
        lines.extend(unavailable if unavailable else ["  (none)"])

        return "\n".join(lines)

    async def _generate_requirements(self, code_plan: dict) -> str:
        """Generate requirements.txt from code plan."""
        deps = code_plan.get("dependencies", [])
        if not deps:
            deps = ["torch", "numpy", "pandas", "scikit-learn", "matplotlib", "tqdm"]

        # BUG-23 fix: compare base package names (stripping version specifiers)
        # to avoid duplicates like "torch" and "torch>=2.0.0"
        def _pkg_base(spec: str) -> str:
            return _re.split(r'[>=<!~\[]', spec)[0].strip().lower()

        existing_bases = {_pkg_base(d) for d in deps}
        essential = {"torch", "numpy", "datasets", "requests", "Pillow"}
        for d in essential:
            if d.lower() not in existing_bases:
                deps.append(d)

        # Deduplicate by base name (keep first occurrence / version-pinned variant)
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
        # Collect all known-good paths and their parents
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

        # Also include the experiment directory itself (results/, checkpoints/, etc.)
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
            # Use path relative to code_dir for readable diagnostics
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
                    # Skip if path exists on disk
                    if Path(ref_path).exists():
                        continue
                    # Skip if under a known valid directory
                    if any(ref_path.startswith(vp) for vp in valid_paths if vp):
                        continue
                    issues.append({
                        "file": rel_name,
                        "path": ref_path,
                    })

        return issues

    async def _fix_import_mismatches(self, code_dir: Path, code_plan: dict) -> None:
        """Scan all generated files for cross-file import mismatches and fix them via LLM.

        Uses AST-based ImportChecker to detect:
        1. `from X import Y` where Y doesn't exist in X
        2. `import X` + `X.func()` where func doesn't exist in X
        """
        from nanoresearch.agents.import_checker import ImportChecker

        checker = ImportChecker(code_dir)
        mismatches = checker.check_imports()
        # Filter out syntax_error entries (not actionable by LLM fix)
        mismatches = [m for m in mismatches if m.get("type") != "syntax_error"]

        if not mismatches:
            self.log("Import consistency check passed")
            return

        self.log(f"Found {len(mismatches)} import mismatches, asking LLM to fix")

        # Read all source files (recursive to catch subpackages)
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
            "1. `from X import missing_name` — name not defined in X\n"
            "2. `import X; X.missing_name()` — function/class not defined in X\n\n"
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
