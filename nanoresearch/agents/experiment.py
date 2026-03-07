"""Experiment agent — multi-phase code project generation with iterative improvement.

Round 1 (initial):
  Phase 1: Generate project plan (file list + interface contracts) via Codex.
  Phase 2: Generate each file individually via Codex.
  Preflight checks (fail-fast validation).
  Phase 3: --dry-run execution.
  Phase 4: --quick-eval for real experiment results.
  Feedback analysis → decide continue/stop.

Round 2+ (iteration):
  LLM generates hypothesis from feedback.
  LLM modifies specific files (not full regeneration).
  Preflight → dry-run → quick-eval → feedback analysis → continue/stop.

Returns the best round's results.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import platform
import subprocess
import sys
import venv
from pathlib import Path
from typing import Any

from nanoresearch.agents.base import BaseResearchAgent
from nanoresearch.agents.cluster_executor import ClusterExecutor
from nanoresearch.agents.experiment_tools import build_experiment_tools
from nanoresearch.agents.feedback_analyzer import FeedbackAnalyzer
from nanoresearch.agents.preflight import PreflightChecker
from nanoresearch.schemas.iteration import (
    ExperimentHypothesis,
    FeedbackAnalysis,
    IterationState,
    RoundResult,
)
from nanoresearch.schemas.manifest import PipelineStage

logger = logging.getLogger(__name__)

# Configurable limits
MAX_REFERENCE_REPOS = 3
MAX_FILE_TREE_ENTRIES = 30
MAX_README_EXCERPT_LENGTH = 500

# Subprocess / output limits
DRY_RUN_TIMEOUT_SECONDS = 60
SUBPROCESS_OUTPUT_LIMIT = 5000
LLM_CONTEXT_TRUNCATION = 4000
STDERR_SNIPPET_LIMIT = 2000


def _decode_bytes(data: bytes | str, limit: int = 0) -> str:
    """Decode subprocess output bytes to str safely on Windows (GBK fallback)."""
    if isinstance(data, str):
        return data[:limit] if limit else data
    try:
        text = data.decode("utf-8", errors="replace")
    except Exception:
        text = data.decode("latin-1", errors="replace")
    return text[:limit] if limit else text


PROJECT_PLAN_SYSTEM_PROMPT = """You are an ML project architect. Given an experiment blueprint, design a complete, runnable Python project structure.

Output a JSON object with:
{
  "project_name": "short_snake_case_name",
  "description": "One-line project description",
  "python_version": ">=3.9",
  "dependencies": ["torch>=2.0", "numpy", ...],
  "files": [
    {
      "path": "src/model.py",
      "description": "Model architecture",
      "interfaces": [
        "class ProposedModel(nn.Module): __init__(self, config: dict), forward(self, x: Tensor) -> Tensor",
        "def build_model(config: dict) -> ProposedModel"
      ],
      "depends_on": ["config/default.yaml"]
    },
    ...
  ],
  "interface_contract": "Full interface contract text listing every class/function signature across all files"
}

The project MUST include these files:
- README.md (with usage instructions)
- requirements.txt
- config/default.yaml (hyperparameters and paths, YAML format — use yaml.safe_load() to parse)
- src/__init__.py
- src/model.py (complete model architecture)
- src/dataset.py (data loading and preprocessing)
- src/trainer.py (training loop with logging)
- src/evaluate.py (evaluation metrics and visualization)
- src/utils.py (utility functions)
- scripts/train.sh (training launch script)
- scripts/run_ablation.sh (ablation experiment script)
- main.py (entry point)

IMPORTANT: main.py MUST accept a `--dry-run` flag (via argparse or sys.argv).
When --dry-run is passed, main.py should:
  1. Import all dependencies
  2. Validate configuration / hyperparameters
  3. Initialize model architecture (without loading pretrained weights)
  4. Print "Dry run complete" and exit with code 0
Do NOT perform actual training or data loading during --dry-run.

IMPORTANT: main.py MUST also accept a `--quick-eval` flag for fast evaluation.
When --quick-eval is passed, main.py should:
  1. Use a SCALED-DOWN model: divide layers by 4, hidden_dim by 4 (minimum 1 layer, 16 dim)
  2. Train for only 3-5 epochs
  3. Use a small data subset: first 500-1000 samples, or generate synthetic data if real data unavailable
  4. Run evaluation on a validation/test subset
  5. Run a minimal ablation (at least 2 variants: full model + one ablation)
  6. Write ALL results to `results/metrics.json` with this EXACT format:
     {
       "main_results": [
         {"method_name": "Ours", "dataset": "DATASET_NAME", "is_proposed": true,
          "metrics": [{"metric_name": "METRIC", "value": 85.4, "std": 0.3, "num_runs": 3}]},
         {"method_name": "BASELINE", "dataset": "DATASET_NAME", "is_proposed": false,
          "metrics": [{"metric_name": "METRIC", "value": 82.1, "std": 0.4, "num_runs": 3}]}
       ],
       "ablation_results": [
         {"variant_name": "Full Model", "metrics": [{"metric_name": "METRIC", "value": 85.4}]},
         {"variant_name": "w/o Component", "metrics": [{"metric_name": "METRIC", "value": 83.1}]}
       ],
       "training_log": [
         {"epoch": 1, "train_loss": 2.5, "val_loss": 2.3, "metrics": {"METRIC": 78.2}},
         {"epoch": 2, "train_loss": 1.8, "val_loss": 1.7, "metrics": {"METRIC": 84.5}}
       ]
     }
  7. Use a helper function `save_metrics(results_dict, output_dir="results")` to write the JSON.
  8. All numeric values MUST come from actual computation, NEVER hardcoded or fabricated.

DEVICE REQUIREMENTS (critical):
- config/default.yaml MUST include: device: "auto" (auto-detects CUDA/MPS/CPU)
- main.py device selection: use "cuda" if torch.cuda.is_available(), else "cpu"
- NEVER default to "cpu" — always prefer GPU when available

WINDOWS COMPATIBILITY (critical):
- config/default.yaml MUST set: num_workers: 0
- All DataLoader calls MUST use num_workers=0 (Windows multiprocessing spawn breaks with >0)

QUICK-EVAL PERFORMANCE:
- --quick-eval MUST use num_runs: 1 (not 3) and at most 200 train samples
- The goal is finishing in under 5 minutes on a single GPU

REPRODUCIBILITY REQUIREMENTS (critical):
- config/default.yaml MUST include: random_seed (default 42), num_runs (default 3)
- main.py MUST set random seeds at startup:
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
- src/trainer.py MUST support running experiments num_runs times with different seeds
  and report mean ± std for all metrics
- src/evaluate.py MUST compute per-run metrics and aggregate statistics
- scripts/run_ablation.sh MUST run each ablation variant num_runs times

Output ONLY valid JSON, no markdown formatting."""

FILE_GEN_SYSTEM_PROMPT = """You are an expert ML engineer. Generate a complete, production-quality file for a research project.

You MUST follow:
1. The interface contract exactly (same class names, method signatures, types)
2. Use PyTorch as the ML framework
3. Include proper imports
4. Include docstrings for public classes and functions
5. Handle edge cases and include proper error messages
6. Mark any truly unimplemented stubs with # TODO comments
7. If this file is main.py, it MUST handle a --dry-run flag (via argparse or sys.argv).
   When --dry-run is passed: import deps, validate config, init model, print "Dry run complete", exit 0.
8. If this file is main.py, it MUST also handle a --quick-eval flag.
   When --quick-eval is passed:
   a. Scale down model (layers/4, hidden_dim/4, minimum 1 layer and 16 dim)
   b. Train for 3-5 epochs on a small data subset (500-1000 samples or synthetic data)
   c. Evaluate and run minimal ablation (full model + at least 1 variant)
   d. Write results to results/metrics.json using a save_metrics() helper
   e. All values MUST come from actual computation — NEVER hardcode or fabricate numbers
9. REPRODUCIBILITY: Always set random seeds (random, numpy, torch) at program start.
   Report results as mean ± std from multiple runs. Use deterministic mode for CUDA.
10. In evaluate.py: compute per-class or per-sample breakdown, save top-5 best and worst
    predictions for qualitative analysis. Generate confusion matrix if classification task.
11. Include a save_metrics() utility function (in src/utils.py or main.py) that:
    - Creates the results/ directory if needed
    - Writes results/metrics.json with the exact schema required by --quick-eval
    - Validates that all metric values are real numbers (not NaN or Inf)
12. CRITICAL: Config files use YAML format (config/default.yaml). Use `yaml.safe_load()` to
    parse them, NOT `json.load()`. Add `pyyaml` to requirements.txt if needed.
13. Keep imports consistent: only import from modules that actually exist in the project.
    The project structure is: main.py, src/{model,dataset,trainer,evaluate,utils}.py, config/default.yaml
14. DEVICE: Always default to "cuda" when torch.cuda.is_available(), NOT "cpu".
    Use `torch.device("cuda" if torch.cuda.is_available() else "cpu")`.
    Never write `config.get("device", "cpu")` — always use `config.get("device", "cuda")`.
    In config/default.yaml, set `device: auto` or `device: cuda`.
15. WINDOWS COMPATIBILITY: Set `num_workers: 0` in config/default.yaml and DataLoader calls.
    On Windows, multiprocessing spawn with num_workers > 0 causes FileNotFoundError in child processes.
    Always use `num_workers=0` for DataLoader unless explicitly overridden.
16. QUICK-EVAL SPEED: In --quick-eval mode, use num_runs=1 (not 3) and at most 200 training samples.
    The goal is to finish in under 5 minutes on a single GPU. Keep the model scaled-down.

Generate ONLY the file content, no markdown formatting, no explanation."""


def _is_finite(value: Any) -> bool:
    """Check if a numeric value is a finite real number."""
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    return False  # non-numeric values are not valid metrics


def _all_metrics_finite(metrics: list) -> bool:
    """Check that all metric values in a list are finite numbers."""
    if not isinstance(metrics, list):
        return False
    for m in metrics:
        if not isinstance(m, dict):
            return False
        val = m.get("value")
        if val is not None and not _is_finite(val):
            return False
    return True


def _training_entry_finite(entry: dict) -> bool:
    """Check that numeric fields in a training log entry are finite."""
    for key in ("train_loss", "val_loss"):
        val = entry.get(key)
        if val is not None and not _is_finite(val):
            return False
    metrics = entry.get("metrics", {})
    if not isinstance(metrics, dict):
        return False  # malformed metrics field
    for val in metrics.values():
        if val is not None and not _is_finite(val):
            return False
    return True


class ExperimentAgent(BaseResearchAgent):
    stage = PipelineStage.EXPERIMENT

    async def run(self, **inputs: Any) -> dict[str, Any]:
        blueprint_data: dict = inputs["experiment_blueprint"]
        reference_repos: list[dict] = inputs.get("reference_repos", [])

        # Dispatch to ReAct mode or pipeline mode
        if self.config.experiment_mode == "react":
            return await self._run_react_mode(blueprint_data, reference_repos)

        max_rounds = self.config.experiment_max_rounds
        self.log(f"Starting iterative experiment (max {max_rounds} rounds)")

        title = blueprint_data.get("title", "")
        method = blueprint_data.get("proposed_method", {})
        datasets = blueprint_data.get("datasets", [])
        metrics = blueprint_data.get("metrics", [])
        baselines = blueprint_data.get("baselines", [])
        ablations = blueprint_data.get("ablation_groups", [])

        blueprint_summary = json.dumps({
            "title": title,
            "proposed_method": method,
            "datasets": datasets,
            "metrics": metrics,
            "baselines": baselines,
            "ablation_groups": ablations,
        }, indent=2, ensure_ascii=False)

        repo_context = self._build_repo_context(reference_repos)
        if repo_context:
            self.log(f"Using {len(reference_repos)} reference repos for code grounding")

        analyzer = FeedbackAnalyzer(self.config, self._dispatcher)
        iteration_state = IterationState(max_rounds=max_rounds)
        code_dir = self.workspace.path / "code"
        venv_python: str = sys.executable
        generated_files: list[str] = []
        project_plan: dict = {}

        # --- Cluster mode detection ---
        cluster_cfg = self.config.cluster
        cluster_mode = bool(cluster_cfg and cluster_cfg.get("enabled"))
        cluster: ClusterExecutor | None = None
        cluster_code_path: str = ""
        if cluster_mode:
            cluster = ClusterExecutor(cluster_cfg, log_fn=self.log)
            mode_desc = "LOCAL SLURM" if cluster.local_mode else "REMOTE SSH+SLURM"
            self.log(f"Cluster mode ENABLED ({mode_desc}) — experiments will run on SLURM cluster")
            if not await cluster.check_connectivity():
                self.log("WARNING: Cluster check failed, falling back to local execution")
                cluster_mode = False
                cluster = None

        # Sub-round checkpoint: load previous iteration state if resuming
        iteration_state, start_round = self._load_iteration_checkpoint(iteration_state)

        for round_num in range(start_round, max_rounds + 1):
            self.log(f"=== Iteration Round {round_num}/{max_rounds} ===")
            files_modified: list[str] = []

            if round_num == 1:
                # ---- Round 1: full generation (baseline) ----
                hypothesis = ExperimentHypothesis(
                    round_number=1,
                    hypothesis="Implement baseline experiment per blueprint",
                    planned_changes=["Generate all project files from scratch"],
                    expected_signal="Successful dry-run and quick-eval with baseline metrics",
                    rationale="Initial implementation of the experiment blueprint",
                )

                # Phase 1: Generate project plan
                self.log("Phase 1: Generating project plan")
                project_plan = await self._generate_project_plan(blueprint_summary, repo_context)
                self.workspace.write_json("plans/project_plan.json", project_plan)
                self.log(f"Project plan: {len(project_plan.get('files', []))} files")

                # Phase 2: Generate each file (parallel)
                self.log("Phase 2: Generating files")
                generated_files = []
                interface_contract = project_plan.get("interface_contract", "")

                # Collect valid file specs
                valid_specs = []
                code_root = (self.workspace.path / "code").resolve()
                for file_spec in project_plan.get("files", []):
                    if not isinstance(file_spec, dict) or "path" not in file_spec:
                        logger.warning("Skipping invalid file_spec: %s", file_spec)
                        continue
                    file_path = file_spec["path"]
                    try:
                        (self.workspace.path / "code" / file_path).resolve().relative_to(code_root)
                    except ValueError:
                        logger.warning("Skipping unsafe file path: %s", file_path)
                        continue
                    valid_specs.append(file_spec)

                # Generate all files in parallel
                self.log(f"  Generating {len(valid_specs)} files in parallel")
                contents = await asyncio.gather(*(
                    self._generate_file(
                        spec, interface_contract, blueprint_summary, repo_context
                    )
                    for spec in valid_specs
                ))

                # Write files sequentially (filesystem ops)
                for spec, content in zip(valid_specs, contents):
                    file_path = spec["path"]
                    self.workspace.write_text(f"code/{file_path}", content)
                    generated_files.append(file_path)

                # Phase 2b: Cross-file import consistency check
                import_mismatches = self._check_import_consistency(code_dir)
                if import_mismatches:
                    self.log(f"Found {len(import_mismatches)} import mismatches, fixing...")
                    await self._fix_import_mismatches(code_dir, import_mismatches)

                # Legacy code_skeleton.py
                main_path = code_dir / "main.py"
                if main_path.exists():
                    try:
                        self.workspace.write_text(
                            "plans/code_skeleton.py", main_path.read_text(encoding="utf-8")
                        )
                    except OSError as exc:
                        logger.warning("Failed to copy main.py as code_skeleton.py: %s", exc)

                # Register artifacts
                for fp in generated_files:
                    self.workspace.register_artifact(
                        f"code_{fp.replace('/', '_')}",
                        self.workspace.path / "code" / fp,
                        self.stage,
                    )

                # Verify syntax
                verification = self._verify_code(generated_files)
                self.workspace.write_json("logs/code_verification.json", verification)
                self.log(
                    f"Code verification: {verification['passed']}/{verification['total']} files OK"
                )
            else:
                # ---- Round 2+: iterative improvement ----
                prev_round = iteration_state.rounds[-1]
                prev_analysis = prev_round.analysis
                history_summary = self._build_history_summary(iteration_state.rounds)

                # If previous round failed preflight, inject the error into context
                preflight_error_ctx = ""
                if prev_round.preflight and prev_round.preflight.overall_status == "failed":
                    failures = []
                    for chk in prev_round.preflight.checks:
                        if chk.status == "failed":
                            failures.append(f"- [{chk.check_name}] {chk.message}")
                    preflight_error_ctx = (
                        "\n== PREFLIGHT FAILURES (must fix these first!) ==\n"
                        + "\n".join(failures)
                        + "\n== END PREFLIGHT FAILURES =="
                    )

                hypothesis = await self._generate_iteration_hypothesis(
                    prev_analysis, history_summary, blueprint_summary,
                    preflight_error_ctx=preflight_error_ctx,
                )
                hypothesis.round_number = round_num

                # Early stop if LLM has no new ideas
                if hypothesis.hypothesis == "__NO_NEW_IDEAS__":
                    self.log("LLM exhausted improvement ideas — stopping iteration")
                    iteration_state.termination_reason = "no_new_ideas"
                    break

                self.log(f"Hypothesis: {hypothesis.hypothesis[:100]}")

                files_modified = await self._apply_iteration_changes(
                    hypothesis, code_dir, venv_python
                )
                # Fallback: if search-replace failed to match anything, retry with
                # full-file rewrite for the first planned_changes target
                if not files_modified and hypothesis.planned_changes:
                    self.log("Search-replace matched nothing, retrying with full-file rewrite")
                    files_modified = await self._apply_iteration_changes_fullwrite(
                        hypothesis, code_dir
                    )
                generated_files = files_modified or generated_files
                self.log(f"Modified {len(files_modified)} files")

            # ---- Preflight checks ----
            self.log("Running preflight checks")
            checker = PreflightChecker(code_dir)
            preflight = checker.run_all()
            self.workspace.write_json(
                f"logs/iteration_round_{round_num}_preflight.json",
                preflight.model_dump(),
            )
            self.log(f"Preflight: {preflight.overall_status}")

            if preflight.overall_status == "failed":
                self.log(f"Blocking preflight failures: {preflight.blocking_failures}")
                round_result = RoundResult(
                    round_number=round_num,
                    hypothesis=hypothesis,
                    preflight=preflight,
                    execution_status="skipped",
                    quick_eval_status="skipped",
                    metrics={},
                )
                iteration_state.rounds.append(round_result)
                # Always keep trying when there are rounds left —
                # implementation bugs are fixable with LLM iteration.
                self.log(f"Preflight failed, will retry in next round ({round_num}/{max_rounds})")
                continue

            # ---- Phase 3 & 4: execution ----
            if cluster_mode and cluster:
                # ===== CLUSTER EXECUTION =====
                execution, quick_eval = await self._run_on_cluster(
                    cluster, code_dir, round_num, cluster_code_path,
                )
                # Update cluster_code_path after first prepare
                if execution.get("cluster_code_path"):
                    cluster_code_path = execution["cluster_code_path"]
                execution_status = execution.get("status", "failed")
                self.log(f"Cluster execution: {execution_status}")
                self.log(f"Cluster quick-eval: {quick_eval.get('status', 'skipped')}")
            else:
                # ===== LOCAL EXECUTION =====
                # Phase 3: dry-run
                if round_num == 1:
                    execution, venv_python = await self._execute_code_with_venv(
                        generated_files, blueprint_summary
                    )
                else:
                    execution = await self._execute_code(
                        generated_files, blueprint_summary,
                        _code_dir=code_dir,
                        _main_py=code_dir / "main.py",
                        _venv_python=venv_python,
                    )
                execution_status = execution.get("status", "failed")
                self.log(f"Dry-run: {execution_status}")

                # Phase 4: quick-eval
                quick_eval = {"status": "skipped", "metrics": {}}
                if execution_status in ("success", "fixed"):
                    quick_eval = await self._run_quick_eval(code_dir, venv_python)
                    self.log(f"Quick-eval: {quick_eval['status']}")
                else:
                    self.log("Skipping quick-eval (dry-run did not succeed)")

            self.workspace.write_json(
                f"logs/iteration_round_{round_num}_execution.json", execution
            )
            self.workspace.write_json(
                f"logs/iteration_round_{round_num}_quick_eval.json", quick_eval
            )

            # ---- Feedback analysis ----
            stderr_snippet = quick_eval.get("stderr", "") or execution.get("stderr", "")
            analysis = await analyzer.analyze(
                current_round=round_num,
                metrics=quick_eval.get("metrics", {}),
                previous_rounds=iteration_state.rounds,
                stderr_snippet=str(stderr_snippet)[:STDERR_SNIPPET_LIMIT // 2],
                max_rounds=max_rounds,
            )

            round_result = RoundResult(
                round_number=round_num,
                hypothesis=hypothesis,
                preflight=preflight,
                execution_status=execution_status,
                quick_eval_status=quick_eval.get("status", "skipped"),
                metrics=quick_eval.get("metrics", {}),
                analysis=analysis,
                files_modified=generated_files if round_num == 1 else (
                    files_modified if round_num > 1 else []
                ),
            )
            iteration_state.rounds.append(round_result)

            # Track best round (pick primary metric, handle higher/lower-is-better)
            if analysis.metric_summary:
                primary_key = next(iter(analysis.metric_summary), None)
                primary_value = analysis.metric_summary.get(primary_key) if primary_key else None
                best_value = iteration_state.best_metrics.get(primary_key) if (iteration_state.best_metrics and primary_key) else None
                # Heuristic: loss/error/perplexity → lower is better; else higher is better
                _lower_is_better = primary_key and any(
                    kw in primary_key.lower() for kw in ("loss", "error", "perplexity", "mse", "mae", "cer", "wer")
                )
                if best_value is None or primary_value is None:
                    is_improvement = best_value is None and primary_value is not None
                elif _lower_is_better:
                    is_improvement = primary_value < best_value
                else:
                    is_improvement = primary_value > best_value
                if is_improvement:
                    iteration_state.best_round = round_num
                    iteration_state.best_metrics = analysis.metric_summary

            # Save round state
            self.workspace.write_json(
                f"logs/iteration_round_{round_num}.json",
                round_result.model_dump(),
            )
            # Sub-round checkpoint: save iteration state after each round
            # so crash recovery can resume from the last completed round
            self._save_iteration_checkpoint(iteration_state)

            self.log(
                f"Round {round_num} analysis: attribution={analysis.attribution}, "
                f"should_continue={analysis.should_continue}"
            )

            # ---- Check termination ----
            if not analysis.should_continue:
                iteration_state.final_status = analysis.termination_reason or "completed"
                self.log(f"Stopping iteration: {iteration_state.final_status}")
                break
        else:
            # Exhausted all rounds
            iteration_state.final_status = "max_rounds"

        # ---- Build final result (backwards-compatible) ----
        # Use best round's data for downstream stages
        best_round_data = self._get_best_round(iteration_state)
        self.workspace.write_json("logs/code_execution.json", {"status": best_round_data["execution_status"]})
        self.workspace.write_json("logs/quick_eval_results.json", {
            "status": best_round_data["quick_eval_status"],
            "metrics": best_round_data["metrics"],
        })

        self.log(
            f"Experiment complete: {len(iteration_state.rounds)} rounds, "
            f"best=round {iteration_state.best_round}, "
            f"status={iteration_state.final_status}"
        )

        result = {
            "code_project_plan": project_plan,
            "generated_files": generated_files,
            "file_count": len(generated_files),
            "code_verification": self._verify_code(generated_files),
            "code_execution": {"status": best_round_data["execution_status"]},
            "experiment_results": best_round_data["metrics"],
            "experiment_status": best_round_data["quick_eval_status"],
            "iteration_state": iteration_state.model_dump(),
        }
        self.workspace.write_json("logs/experiment_output.json", result)
        return result

    # ------------------------------------------------------------------
    # ReAct experiment mode
    # ------------------------------------------------------------------

    _REACT_SYSTEM_PROMPT = """\
You are an autonomous ML experiment agent. You have full access to the \
filesystem and shell through your tools. Your job is to implement, run, \
debug, and collect results for the experiment described in the blueprint.

## Your tools
- **read_file(path)** — read any file (relative paths resolve against working dir)
- **write_file(path, content)** — write / create a file
- **list_dir(path)** — list directory contents
- **run_command(command, timeout?, workdir?)** — run any shell command (default timeout 120s, max 1800s)
- **search_files(pattern, path?)** — glob search for files
- **grep_content(pattern, path?, file_glob?)** — search file contents

## Workflow

### Phase 0: Environment discovery (DO THIS FIRST)
Run these commands to understand where you are:
```
whoami && hostname && pwd
ldd --version 2>&1 | head -1   # check glibc version (CRITICAL)
nvidia-smi                      # check GPU availability
which python && python --version
conda info --envs               # list conda environments
sinfo 2>/dev/null               # check if SLURM is available
which apptainer 2>/dev/null || which singularity 2>/dev/null   # container runtime
```

**CRITICAL: Check glibc version FIRST.** Many clusters have old glibc (e.g., 2.17 on CentOS 7).
Modern PyTorch/CUDA requires glibc >= 2.28. If glibc is too old, you MUST use apptainer
containers — direct `pip install torch` will fail with `GLIBC_2.xx not found`.

Based on the results, decide:
- **glibc < 2.28** → you MUST use apptainer container (see Phase 0.5)
- **glibc >= 2.28** → conda/pip works directly, container optional
- **SLURM available** → write SLURM batch scripts and `sbatch`
- **No SLURM** → run training directly with `python`
- **GPU available** → use CUDA. No GPU → CPU with smaller model/data.
{conda_env_hint}
### Phase 0.5: Container setup (when glibc < 2.28)
{container_config}

### Phase 1: Setup environment
**Decision tree** (pick the FIRST that applies):
1. **glibc < 2.28** → you MUST use container from Phase 0.5; ALL python/pip via `apptainer exec --nv`
2. **glibc >= 2.28 + pre-configured conda env with PyTorch** → `conda activate ENV_NAME` (fast)
3. **glibc >= 2.28 + other conda env** → activate + `pip install torch` (medium)
4. **glibc >= 2.28 + nothing usable** → create new conda env (last resort)

After choosing:
- Test that PyTorch imports: `python -c "import torch; print(torch.cuda.is_available())"`
  (if container: `apptainer exec --nv CONTAINER.sif python -c "import torch; ..."`)
- Install experiment-specific packages (`pip install -r requirements.txt`, `timeout=600`)
- Create the experiment directory structure

**IMPORTANT: If container mode, ALL subsequent python/pip commands must be wrapped in
`apptainer exec --nv -B BINDS CONTAINER.sif bash -c "..."`**

### Phase 2: Write experiment code
Based on the blueprint, create all necessary files:
- `main.py` — entry point with argparse (`--quick-eval` for fast run, `--dry-run` for import check)
- `src/model.py` — model architecture
- `src/trainer.py` — training loop
- `src/dataset.py` — data loading
- `src/evaluate.py` — evaluation metrics
- `requirements.txt` — dependencies
- If SLURM: a `.sh` batch script (see template below)

IMPORTANT for `main.py`:
- Add `--quick-eval` flag: use scaled-down model (layers/4, hidden_dim/4), train 3-5 epochs,
  use 500-1000 data samples (or synthetic data), run ablation, save to results/metrics.json
- All metric values MUST come from actual computation, NEVER hardcode
- Set random seed at startup for reproducibility

### Phase 3: Run the experiment
- First do a dry-run to check imports: `python main.py --dry-run`
- Then run quick-eval: `python main.py --quick-eval`
- If using **container + SLURM**: wrap python command in `apptainer exec` inside sbatch script
- If SLURM: `sbatch run.sh`, then poll with `squeue -u $(whoami)` every ~30s and read log files
- If local: run directly or with `nohup` for long jobs
- For SLURM jobs that take time, use `run_command` with `timeout=300` or higher to poll status

### Phase 4: Debug if it fails
- Read error logs carefully
- Fix the bug in the source code
- Re-run the experiment
- Repeat until it succeeds or you've tried 5+ different fixes

### Phase 5: Collect results
- Read `results/metrics.json` (or however results are saved)
- Report final metrics clearly
- The experiment is done when you have real metric numbers

## CRITICAL RULES
1. **Always check your environment first** — don't assume anything
2. **Start simple** — get a basic version running before adding complexity
3. **Read error messages carefully** — fix the actual root cause, not symptoms
4. **Don't give up** — if one approach fails, try another
5. **Write complete files** — never write partial or placeholder code
6. **Save results to `results/metrics.json`** — this is the standard output format:
   ```json
   {{"main_results": [{{"method_name": "...", "dataset": "...", "is_proposed": true, "metrics": [{{"metric_name": "accuracy", "value": 0.95}}]}}]}}
   ```
7. **When using SLURM**: after `sbatch`, use `squeue` to check status, and read the SLURM output file for logs
8. **Be resourceful** — you can install packages, download data, check documentation, etc.
9. **Use `timeout` parameter** for long commands: `pip install` (timeout=600), training (timeout=1800), `apptainer pull` (timeout=1800)
10. **Device**: use CUDA if available, else CPU. NEVER default to CPU when GPU exists.
11. **Windows**: if on Windows, use `num_workers=0` in DataLoader.

## SLURM batch script template (without container)
```bash
#!/bin/bash
#SBATCH --job-name=experiment
#SBATCH --partition={{PARTITION}}
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

source activate YOUR_ENV  # or: conda activate YOUR_ENV
cd $SLURM_SUBMIT_DIR

python main.py --quick-eval
```

## SLURM batch script template (with apptainer container)
```bash
#!/bin/bash
#SBATCH --job-name=experiment
#SBATCH --partition={{PARTITION}}
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

cd $SLURM_SUBMIT_DIR
apptainer exec --nv --writable-tmpfs -B {{BIND_MOUNTS}} {{CONTAINER_SIF}} \
    bash -c "pip install -r requirements.txt -q 2>/dev/null; cd $SLURM_SUBMIT_DIR && python main.py --quick-eval"
```
Notes:
- `--writable-tmpfs` allows pip install inside read-only .sif (writes go to tmpfs)
- For a pre-built .sif with PyTorch already inside, you can omit `--writable-tmpfs` and `pip install`
- Adjust `--partition`, `--gres`, `--time`, bind mounts as needed.

## SLURM configuration
{slurm_config}

## End condition
You are DONE when you have:
1. Successfully run the experiment (training completed without errors)
2. Collected real metric numbers (accuracy, loss, etc.)
3. Saved results to `results/metrics.json`

When finished, output a FINAL SUMMARY with:
- What you did
- Final metrics (exact numbers)
- Path to results file
"""

    async def _run_react_mode(
        self,
        blueprint_data: dict,
        reference_repos: list[dict],
    ) -> dict[str, Any]:
        """Run experiment in ReAct mode — LLM drives everything via tools."""
        self.log("Starting experiment in ReAct mode (LLM-driven)")

        code_dir = self.workspace.path / "code"
        code_dir.mkdir(parents=True, exist_ok=True)

        # Build tools
        tools = build_experiment_tools(work_dir=code_dir)

        # Build SLURM config block for system prompt
        partition = self.config.slurm_partition
        max_gpus = self.config.slurm_max_gpus
        wall_time = self.config.slurm_default_time
        if partition:
            slurm_config = (
                f"- Partition: `{partition}`\n"
                f"- Max GPUs per job: {max_gpus}\n"
                f"- Default wall time: {wall_time}\n"
                f"- Submit with: `sbatch your_script.sh`\n"
                f"- Check status: `squeue -u $(whoami)`\n"
                f"- Cancel job: `scancel <job_id>`\n"
                f"- View logs: read the SLURM output file (usually `slurm-<job_id>.out`)"
            )
        else:
            slurm_config = (
                "Not pre-configured. Run `sinfo` to check if SLURM is available.\n"
                f"If available, use at most {max_gpus} GPUs per job."
            )

        # Build conda env hint for system prompt
        conda_env = self.config.experiment_conda_env
        if conda_env:
            conda_env_hint = (
                f"\n**Pre-configured conda env**: `{conda_env}` — use this env "
                f"(activate with `conda activate {conda_env}`). "
                f"It likely already has PyTorch and common ML packages installed.\n"
            )
        else:
            conda_env_hint = ""

        # Build container config block for system prompt
        container_image = self.config.container_image
        container_path = self.config.container_path
        container_bind = self.config.container_bind

        # Common explanation block
        _why = (
            "**WHY containers?** HPC clusters often have old glibc (e.g., 2.17 on CentOS 7).\n"
            "Modern PyTorch/CUDA needs glibc >= 2.28. Direct `pip install torch` fails with\n"
            "`GLIBC_2.xx not found`. Containers (e.g., Ubuntu 22.04) bundle glibc 2.35 inside.\n"
        )

        # Search paths for existing .sif files
        _search_dirs = "/mnt /opt /shared /data $HOME"

        container_lines = [_why]

        # Step 1: Search
        container_lines.extend([
            "**Step 1: Search for existing .sif files on the cluster**",
            f"```",
            f"find {_search_dirs} -name '*.sif' -maxdepth 4 2>/dev/null | head -20",
            f"```",
        ])
        if container_path:
            container_lines.append(
                f"Pre-configured path to check first: `{container_path}`"
            )

        # Step 2: Try each
        container_lines.extend([
            "",
            "**Step 2: Try each .sif file — test if it has a usable Python + PyTorch**",
            "For EACH .sif file found, test:",
            "```",
            "apptainer exec --nv FOUND.sif python3 -c \"import torch; print(torch.__version__, torch.cuda.is_available())\"",
            "```",
            "- If it prints a torch version with `True` → **use this .sif, you are done!**",
            "- If it fails (no python, no torch, wrong CUDA) → try the next .sif",
            "- If no python3, try `python` instead",
        ])

        # Step 3: Download if none work
        container_lines.extend([
            "",
            "**Step 3: If NO usable .sif found → download a clean base image**",
        ])
        if container_image:
            container_lines.append(
                f"Pre-configured image: `{container_image}`"
            )
            sif_target = container_path or "ubuntu2204.sif"
            container_lines.append(
                f"```\napptainer pull {sif_target} {container_image}\n```"
            )
        else:
            container_lines.extend([
                "Download a clean Ubuntu 22.04 image (small, ~30MB, glibc 2.35):",
                "```",
                "apptainer pull ubuntu2204.sif docker://ubuntu:22.04",
                "```",
                "(use `timeout=1800` — first pull may be slow)",
            ])

        # Step 4: Install Python + deps inside
        container_lines.extend([
            "",
            "**Step 4: Install Python + PyTorch inside the clean container**",
            "Use `--writable-tmpfs` to allow temporary writes inside the read-only .sif:",
            "```",
            "# Test if python3 exists inside",
            "apptainer exec ubuntu2204.sif which python3",
            "",
            "# If no python3, install it (needs --writable-tmpfs or --fakeroot):",
            "apptainer exec --writable-tmpfs ubuntu2204.sif bash -c \\",
            '  "apt-get update -qq && apt-get install -y -qq python3 python3-pip > /dev/null && \\',
            '   pip3 install torch torchvision numpy -q && \\',
            '   python3 -c \\"import torch; print(torch.__version__)\\"" ',
            "```",
            "",
            "**BETTER: Build a reusable .sif with a definition file** (so you don't reinstall every time):",
            "```",
            "# Write a .def file",
            "cat > experiment.def << 'DEFEOF'",
            "Bootstrap: docker",
            "From: ubuntu:22.04",
            "",
            "%post",
            "    apt-get update -qq && apt-get install -y -qq python3 python3-pip git > /dev/null",
            "    pip3 install torch torchvision numpy scipy scikit-learn matplotlib -q",
            "",
            "%environment",
            "    export PATH=/usr/local/bin:/usr/bin:$PATH",
            "DEFEOF",
            "",
            "# Build the .sif (use --fakeroot on HPC clusters without root)",
            "apptainer build --fakeroot experiment.sif experiment.def",
            "```",
            "If `--fakeroot` fails, use `--writable-tmpfs` approach instead.",
        ])

        # Step 5: Usage
        container_lines.extend([
            "",
            "**Step 5: Use the container for ALL commands**",
            f"Bind mounts: `-B {container_bind}`",
            "```",
            "# Run python inside container",
            "apptainer exec --nv -B {bind} experiment.sif python3 main.py --quick-eval".format(
                bind=container_bind
            ),
            "",
            "# Install extra packages at runtime (--writable-tmpfs)",
            "apptainer exec --nv --writable-tmpfs -B {bind} experiment.sif bash -c \\".format(
                bind=container_bind
            ),
            '  "pip3 install -r requirements.txt -q && python3 main.py --quick-eval"',
            "```",
            "IMPORTANT: once in container mode, ALL python/pip must go through `apptainer exec`.",
        ])

        container_config = "\n".join(container_lines)

        system_prompt = self._REACT_SYSTEM_PROMPT.format(
            slurm_config=slurm_config,
            conda_env_hint=conda_env_hint,
            container_config=container_config,
        )

        # Build user prompt with blueprint
        blueprint_summary = json.dumps(blueprint_data, indent=2, ensure_ascii=False)
        if len(blueprint_summary) > 6000:
            blueprint_summary = blueprint_summary[:6000] + "\n... (truncated)"

        repo_context = self._build_repo_context(reference_repos)
        repo_block = f"\n\n## Reference code\n{repo_context}" if repo_context else ""

        user_prompt = f"""## Experiment Blueprint

{blueprint_summary}
{repo_block}

## Working directory
`{code_dir}`

Please start by discovering the environment (Phase 0), then implement and run this experiment.
The goal is to get real experimental results (metric numbers) — not placeholder code."""

        # Run the ReAct loop
        max_rounds = self.config.react_max_rounds
        self.log(f"ReAct loop: max {max_rounds} tool rounds")

        try:
            final_output = await self.generate_with_tools(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                tools=tools,
                max_tool_rounds=max_rounds,
                stage_override=self.config.for_stage("code_gen"),
                reminder_text=(
                    "[REMINDER] You are running an ML experiment. Stay focused:\n"
                    "- If the experiment is still running, check its status (squeue / read log file)\n"
                    "- If it failed, read the error and fix the code\n"
                    "- If it succeeded, collect the results and report metrics\n"
                    "- Do NOT start over from scratch unless absolutely necessary\n"
                    "- Your goal is REAL metric numbers, not placeholder code"
                ),
                reminder_interval=5,
            )
        except Exception as exc:
            logger.error("ReAct experiment failed: %s", exc, exc_info=True)
            final_output = f"ReAct experiment failed with error: {exc}"

        self.log(f"ReAct experiment completed. Output length: {len(final_output)}")
        self.workspace.write_text("logs/react_final_output.md", final_output)

        # Try to collect metrics from results/metrics.json
        metrics = self._parse_metrics_json(code_dir)
        experiment_status = "success" if metrics else "partial"

        result = {
            "code_project_plan": {"mode": "react"},
            "generated_files": [
                str(f.relative_to(code_dir))
                for f in code_dir.rglob("*")
                if f.is_file() and "__pycache__" not in str(f)
            ],
            "file_count": sum(
                1 for f in code_dir.rglob("*")
                if f.is_file() and "__pycache__" not in str(f)
            ),
            "code_execution": {"status": experiment_status},
            "experiment_results": metrics,
            "experiment_status": experiment_status,
            "react_output": final_output[:5000],
        }
        self.workspace.write_json("logs/experiment_output.json", result)
        return result

    # ------------------------------------------------------------------
    # Sub-round checkpoint helpers
    # ------------------------------------------------------------------

    def _save_iteration_checkpoint(self, state: IterationState) -> None:
        """Save iteration state checkpoint for crash recovery."""
        self.workspace.write_json(
            "logs/iteration_checkpoint.json",
            state.model_dump(),
        )

    def _load_iteration_checkpoint(
        self, default_state: IterationState
    ) -> tuple[IterationState, int]:
        """Load iteration checkpoint if available.

        Returns (state, start_round) where start_round is the round to
        resume from (1 if no checkpoint exists).
        """
        try:
            data = self.workspace.read_json("logs/iteration_checkpoint.json")
            if isinstance(data, dict) and data.get("rounds"):
                state = IterationState.model_validate(data)
                completed_rounds = len(state.rounds)
                start_round = completed_rounds + 1
                if start_round <= state.max_rounds:
                    logger.info(
                        "Resuming experiment from round %d (checkpoint has %d completed rounds)",
                        start_round, completed_rounds,
                    )
                    return state, start_round
                else:
                    logger.info(
                        "Checkpoint shows all %d rounds completed, starting fresh",
                        completed_rounds,
                    )
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.warning("Failed to load iteration checkpoint: %s", exc)
        return default_state, 1

    # ------------------------------------------------------------------
    # Iteration helpers
    # ------------------------------------------------------------------

    async def _generate_iteration_hypothesis(
        self,
        analysis: FeedbackAnalysis | None,
        history_summary: str,
        blueprint: str,
        preflight_error_ctx: str = "",
    ) -> ExperimentHypothesis:
        """LLM generates the next iteration hypothesis from feedback."""
        analysis_text = ""
        if analysis:
            analysis_text = (
                f"Attribution: {analysis.attribution}\n"
                f"Recommended action: {analysis.recommended_action}\n"
                f"Metrics: {json.dumps(analysis.metric_summary)}\n"
                f"Training dynamics: convergence={analysis.training_dynamics.convergence_speed}, "
                f"overfitting={analysis.training_dynamics.overfitting_detected}, "
                f"stability={analysis.training_dynamics.loss_stability}\n"
                f"Error categories: {analysis.error_categories}"
            )

        # Collect actual file list from code_dir for the LLM
        code_dir = self.workspace.path / "code"
        actual_files = []
        if code_dir.exists():
            for f in sorted(code_dir.rglob("*")):
                if f.is_file() and "__pycache__" not in str(f) and ".pyc" not in str(f):
                    actual_files.append(str(f.relative_to(code_dir)).replace("\\", "/"))
        file_list = "\n".join(f"  - {f}" for f in actual_files) if actual_files else "  (no files yet)"

        # Build list of previously tried hypotheses to prevent repetition
        prev_hypotheses = []
        if history_summary:
            for line in history_summary.split("\n"):
                if line.strip():
                    prev_hypotheses.append(line.strip())
        prev_hyp_block = "\n".join(prev_hypotheses) if prev_hypotheses else "None"

        prompt = f"""Based on the previous experiment round's feedback, generate a hypothesis for the next improvement iteration.
{preflight_error_ctx}
== Previous Analysis ==
{analysis_text or "No analysis available."}

== History ==
{history_summary or "No previous rounds."}

== PREVIOUSLY TRIED HYPOTHESES (DO NOT REPEAT) ==
{prev_hyp_block}

== Experiment Blueprint ==
{blueprint[:2000]}

== Actual Project Files ==
{file_list}

IMPORTANT RULES:
1. Only reference files that exist in the list above. Do NOT invent new file paths.
2. Use the EXACT paths shown above in your planned_changes.
3. The `--quick-eval` mode HARDCODES a small model and 3-5 epochs regardless of config.
   Changing epochs/batch_size/num_runs in config/default.yaml has NO EFFECT on quick-eval.
   DO NOT suggest increasing epochs or changing hyperparameters in config — it is USELESS.
4. Instead, focus on changes that actually affect quick-eval behavior:
   - Fix bugs in model architecture (src/model.py)
   - Fix bugs in training loop (src/trainer.py)
   - Fix evaluation/metrics collection (src/evaluate.py, src/utils.py)
   - Fix data loading/preprocessing (src/dataset.py)
   - Fix the quick-eval code path in main.py directly
   - Improve model architecture (e.g., add batch norm, better init, residual connections)
5. DO NOT repeat any hypothesis from the list above. Each round must try something DIFFERENT.
   If you cannot think of a genuinely new improvement, set "no_new_ideas": true.

Output a JSON object with:
{{
  "hypothesis": "<what you will change and why>",
  "planned_changes": ["<EXACT_FILE_PATH: specific change>", ...],
  "expected_signal": "<what metric improvement you expect>",
  "rationale": "<reasoning>",
  "no_new_ideas": false
}}"""

        try:
            code_gen_config = self.config.for_stage("code_gen")
            raw = await self._dispatcher.generate(
                code_gen_config,
                "You are an ML experiment iteration planner. Generate a focused hypothesis for the next improvement round. Output ONLY valid JSON.",
                prompt,
                json_mode=True,
            )
            text = raw.strip()
            if text.startswith("```"):
                lines = text.split("\n")[1:]
                if lines and lines[-1].strip().startswith("```"):
                    lines = lines[:-1]
                text = "\n".join(lines)
            data = json.loads(text)

            # If LLM says no new ideas, signal early stop
            if data.get("no_new_ideas"):
                logger.info("LLM reports no new ideas — will signal early stop")
                return ExperimentHypothesis(
                    round_number=0,
                    hypothesis="__NO_NEW_IDEAS__",
                    planned_changes=[],
                    expected_signal="",
                    rationale="LLM exhausted improvement ideas",
                )

            return ExperimentHypothesis(
                round_number=0,  # caller sets this
                hypothesis=data.get("hypothesis", "Iterative improvement"),
                planned_changes=data.get("planned_changes", []),
                expected_signal=data.get("expected_signal", ""),
                rationale=data.get("rationale", ""),
            )
        except Exception as exc:
            logger.warning("Failed to generate hypothesis: %s", exc)
            return ExperimentHypothesis(
                round_number=0,
                hypothesis="Retry with general improvements based on error feedback",
                planned_changes=["Fix errors from previous round"],
                expected_signal="Successful execution",
                rationale="Fallback hypothesis after LLM generation failure",
            )

    async def _apply_iteration_changes(
        self,
        hypothesis: ExperimentHypothesis,
        code_dir: Path,
        venv_python: str,
    ) -> list[str]:
        """LLM modifies specific files using search-replace edits (OpenClaw style).

        Uses precise search-replace blocks instead of full file rewrites to:
        1. Reduce token usage (LLM only outputs the diff, not entire files)
        2. Avoid accidental deletion of unchanged code
        3. Make changes auditable
        """
        # Collect current file contents for context
        file_contents: dict[str, str] = {}
        for py_file in code_dir.rglob("*.py"):
            parts = py_file.relative_to(code_dir).parts
            if any(p.startswith(".") or p == "__pycache__" for p in parts):
                continue
            try:
                rel = str(py_file.relative_to(code_dir)).replace("\\", "/")
                content = py_file.read_text(encoding="utf-8", errors="replace")
                file_contents[rel] = content
            except OSError:
                continue

        # Also include config and other non-py files
        for pattern in ("config/*.yaml", "config/*.yml", "*.txt", "*.sh"):
            for f in code_dir.glob(pattern):
                try:
                    rel = str(f.relative_to(code_dir)).replace("\\", "/")
                    file_contents[rel] = f.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    pass

        files_summary = "\n".join(
            f"--- {path} ---\n{content[:2000]}\n"
            for path, content in file_contents.items()
        )

        prompt = f"""Apply the following changes to the experiment code project using SEARCH-REPLACE edits.

== Hypothesis ==
{hypothesis.hypothesis}

== Planned Changes ==
{json.dumps(hypothesis.planned_changes, indent=2)}

== Rationale ==
{hypothesis.rationale}

== Current Files ==
{files_summary[:15000]}

Output a JSON array of edit operations. Two types are supported:

1. **Search-replace edit** (preferred for modifying existing files):
{{
  "path": "relative/path.py",
  "action": "edit",
  "edits": [
    {{"old": "exact text to find", "new": "replacement text"}}
  ]
}}

2. **Full file write** (only for NEW files that don't exist yet):
{{
  "path": "relative/new_file.py",
  "action": "write",
  "content": "full file content"
}}

IMPORTANT RULES:
- "old" must be an EXACT substring of the current file content (including whitespace/indentation)
- Each "old" string must be unique within its file
- Use search-replace for ALL modifications to existing files
- Only use "write" action for creating brand new files
- Multiple edits per file are fine — they are applied sequentially

Output ONLY valid JSON array."""

        modified_files: list[str] = []
        try:
            code_gen_config = self.config.for_stage("code_gen")
            raw = await self._dispatcher.generate(
                code_gen_config,
                "You are an ML code editor. Apply precise search-replace edits to implement the hypothesis. Output ONLY a JSON array.",
                prompt,
            )
            text = raw.strip()
            if text.startswith("```"):
                lines = text.split("\n")[1:]
                if lines and lines[-1].strip().startswith("```"):
                    lines = lines[:-1]
                text = "\n".join(lines)

            changes = json.loads(text)
            if not isinstance(changes, list):
                changes = [changes]

            for change in changes:
                if not isinstance(change, dict) or "path" not in change:
                    continue
                file_path = change["path"]
                # Security: prevent directory traversal
                try:
                    (code_dir / file_path).resolve().relative_to(code_dir.resolve())
                except ValueError:
                    logger.warning("Skipping unsafe iteration path: %s", file_path)
                    continue

                action = change.get("action", "write")  # backwards compat

                if action == "edit":
                    # Search-replace mode
                    edits = change.get("edits", [])
                    if not edits:
                        continue
                    # Read current content
                    target = code_dir / file_path
                    if not target.exists():
                        logger.warning("Edit target does not exist: %s", file_path)
                        continue
                    try:
                        current = target.read_text(encoding="utf-8", errors="replace")
                    except OSError:
                        continue

                    applied = 0
                    for edit in edits:
                        if not isinstance(edit, dict):
                            continue
                        old = edit.get("old", "")
                        new = edit.get("new", "")
                        if not old:
                            continue
                        if old in current:
                            current = current.replace(old, new, 1)
                            applied += 1
                        else:
                            # Try whitespace-normalized match
                            old_normalized = " ".join(old.split())
                            for line_start in range(len(current)):
                                chunk = current[line_start:line_start + len(old) + 200]
                                if " ".join(chunk[:len(old) + 100].split()).startswith(old_normalized[:60]):
                                    # Find the actual extent
                                    end = current.find("\n", line_start + len(old) - 10)
                                    if end == -1:
                                        end = len(current)
                                    candidate = current[line_start:end]
                                    if " ".join(candidate.split()) == old_normalized:
                                        current = current[:line_start] + new + current[end:]
                                        applied += 1
                                        break
                            else:
                                logger.warning(
                                    "Edit old text not found in %s: %s",
                                    file_path, old[:80],
                                )

                    if applied > 0:
                        self.workspace.write_text(f"code/{file_path}", current)
                        modified_files.append(file_path)
                        self.log(f"  Edited: {file_path} ({applied}/{len(edits)} edits applied)")
                else:
                    # Full write mode (new files or backwards compat)
                    content = change.get("content", "")
                    if not content:
                        continue
                    self.workspace.write_text(f"code/{file_path}", content)
                    modified_files.append(file_path)
                    self.log(f"  Wrote: {file_path}")

        except Exception as exc:
            logger.warning("Failed to apply iteration changes: %s", exc)

        return modified_files

    async def _apply_iteration_changes_fullwrite(
        self,
        hypothesis: ExperimentHypothesis,
        code_dir: Path,
    ) -> list[str]:
        """Fallback: when search-replace fails, ask LLM to rewrite the target file entirely."""
        # Find the primary target file from planned_changes
        target_rel = None
        for change_desc in hypothesis.planned_changes:
            # Extract file path from descriptions like "src/trainer.py: fix ..."
            for part in change_desc.replace(":", " ").split():
                candidate = code_dir / part
                try:
                    # Security: ensure candidate is within code_dir (no path traversal)
                    candidate.resolve().relative_to(code_dir.resolve())
                except ValueError:
                    continue
                if candidate.exists() and candidate.is_file():
                    target_rel = part
                    break
            if target_rel:
                break

        if not target_rel:
            # Default to main.py
            if (code_dir / "main.py").exists():
                target_rel = "main.py"
            else:
                return []

        target = code_dir / target_rel
        try:
            current = target.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []

        # Build file context: head + tail for large files to stay within LLM limits
        total_lines = len(current.splitlines())
        if len(current) <= 12000:
            file_block = current
        else:
            # Show first 8K chars + last 4K chars with a separator
            head = current[:8000]
            tail = current[-4000:]
            file_block = (
                f"{head}\n\n... [{total_lines} lines total, middle section omitted for brevity] ...\n\n{tail}"
            )

        prompt = f"""Rewrite the file `{target_rel}` to implement this change:

== Hypothesis ==
{hypothesis.hypothesis}

== Planned Changes ==
{chr(10).join(hypothesis.planned_changes)}

== Current File ({total_lines} lines) ==
```python
{file_block}
```

Output the COMPLETE new file content. No markdown fences, no explanation — ONLY the Python code.
The output MUST be a complete, runnable file — do NOT omit any functions or classes from the original."""

        try:
            code_gen_config = self.config.for_stage("code_gen")
            raw = await self._dispatcher.generate(
                code_gen_config,
                f"You are an ML code editor. Rewrite {target_rel} to implement the requested change. "
                f"Output ONLY the complete file. Do NOT truncate or omit any part of the original code.",
                prompt,
            )
            new_content = (raw or "").strip()
            # Strip markdown fences
            if new_content.startswith("```"):
                lines = new_content.split("\n")[1:]
                if lines and lines[-1].strip().startswith("```"):
                    lines = lines[:-1]
                new_content = "\n".join(lines)

            # Safety: reject if the rewrite looks truncated (LLM hit max_tokens)
            if new_content and len(new_content) > 50:
                # Truncation heuristic: a valid Python file should end with a
                # complete statement — not mid-line or mid-string.
                _last_line = new_content.rstrip().rsplit("\n", 1)[-1].strip()
                _looks_truncated = (
                    # Ends with open string/paren/bracket
                    _last_line.endswith(("(", "[", "{", ",", "\\", '"""', "'''"))
                    # Or ends mid-expression (no closing quote, has unbalanced quotes)
                    or _last_line.count('"') % 2 == 1
                    or _last_line.count("'") % 2 == 1
                    # Or suspiciously short AND the file was large (likely max_tokens cutoff)
                    or (len(new_content) < len(current) * 0.3 and len(current) > 1000)
                )
                if _looks_truncated:
                    logger.warning(
                        "Full-file rewrite for %s looks truncated (%d vs %d chars, last: %s), skipping",
                        target_rel, len(new_content), len(current), _last_line[-60:],
                    )
                    return []
                target.write_text(new_content, encoding="utf-8")
                self.log(f"  Rewrote {target_rel} (full-file fallback, {len(new_content)} chars)")
                return [target_rel]
        except Exception as exc:
            logger.warning("Full-file rewrite fallback failed for %s: %s", target_rel, exc)

        return []

    @staticmethod
    def _build_history_summary(rounds: list[RoundResult]) -> str:
        """Compress historical rounds into a compact summary (~100 chars each)."""
        if not rounds:
            return ""
        lines = []
        for r in rounds:
            metrics_str = ""
            if r.analysis and r.analysis.metric_summary:
                metrics_str = ", ".join(
                    f"{k}={v:.4f}" for k, v in r.analysis.metric_summary.items()
                )
            hyp_short = r.hypothesis.hypothesis[:80]
            attribution = r.analysis.attribution if r.analysis else "n/a"
            lines.append(
                f"R{r.round_number}: [{r.quick_eval_status}] {hyp_short} "
                f"| metrics: {metrics_str or 'none'} | attr: {attribution}"
            )
        return "\n".join(lines)

    @staticmethod
    def _check_import_consistency(code_dir: Path) -> list[dict]:
        """Scan all generated files for cross-file import mismatches.

        Borrowed from Deep Pipeline's CodingAgent — checks two patterns:
        1. `from X import Y` where Y doesn't exist in X
        2. `import X; X.func()` where func doesn't exist in X

        Returns list of mismatch dicts.
        """
        import re as _re

        definitions: dict[str, list[str]] = {}  # module -> [defined names]
        imports: list[dict] = []
        module_accesses: list[dict] = []
        local_modules = {f.stem for f in code_dir.rglob("*.py")}

        for py_file in code_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            try:
                content = py_file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            module_name = py_file.stem

            # Find class and top-level function definitions
            defs = [m.group(1) for m in _re.finditer(r"^(?:class|def)\s+(\w+)", content, _re.MULTILINE)]
            definitions[module_name] = defs

            # Find cross-file imports: from X import Y, Z
            for m in _re.finditer(r"^from\s+(?:src\.)?(\w+)\s+import\s+(.+)$", content, _re.MULTILINE):
                src_module = m.group(1)
                # Strip inline comments before parsing names
                import_text = m.group(2).split("#")[0]
                imported_names = [n.strip().split(" as ")[0].strip() for n in import_text.split(",")]
                imported_names = [n for n in imported_names if n]  # drop empty after comment strip
                imports.append({"importer": py_file.name, "module": src_module, "names": imported_names})

            # Find `import X` for local modules, then scan for X.attr() calls
            imported_modules: dict[str, str] = {}
            for m in _re.finditer(r"^import\s+(?:src\.)?(\w+)(?:\s+as\s+(\w+))?$", content, _re.MULTILINE):
                real_name = m.group(1)
                alias = m.group(2) or real_name
                if real_name in local_modules:
                    imported_modules[alias] = real_name

            for alias, real_name in imported_modules.items():
                for m in _re.finditer(rf"\b{_re.escape(alias)}\.(\w+)\s*\(", content):
                    attr = m.group(1)
                    if not attr.startswith("_"):
                        module_accesses.append({
                            "importer": py_file.name, "module": real_name, "attr": attr,
                        })

        # Check mismatches
        mismatches = []
        for imp in imports:
            module = imp["module"]
            if module not in definitions:
                continue
            defined = set(definitions[module])
            for name in imp["names"]:
                if name and name not in defined:
                    mismatches.append({
                        "importer": imp["importer"], "module": module,
                        "missing_name": name, "available": sorted(defined),
                    })

        seen_access: set[tuple[str, str, str]] = set()
        for acc in module_accesses:
            module = acc["module"]
            if module not in definitions:
                continue
            attr = acc["attr"]
            key = (acc["importer"], module, attr)
            if key in seen_access:
                continue
            seen_access.add(key)
            defined = set(definitions[module])
            if attr not in defined:
                mismatches.append({
                    "importer": acc["importer"], "module": module,
                    "missing_name": attr, "available": sorted(defined),
                    "usage_pattern": f"import {module}; {module}.{attr}()",
                })

        return mismatches

    async def _fix_import_mismatches(
        self, code_dir: Path, mismatches: list[dict],
    ) -> None:
        """Ask LLM to fix cross-file import mismatches via search-replace patches."""
        # Read all source files
        all_sources = {}
        for py_file in code_dir.rglob("*.py"):
            if "__pycache__" not in str(py_file):
                try:
                    all_sources[py_file.name] = py_file.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    pass

        source_listing = ""
        for fname, content in sorted(all_sources.items()):
            source_listing += f"\n# FILE: {fname}\n{content}\n"

        system_prompt = (
            "You are fixing cross-file interface mismatches between Python files in a project. "
            "Some files reference names that don't exist in the target module. "
            "Fix by EITHER adding the missing function/class to the target module, "
            "OR renaming the call to match what's already defined. "
            "Return JSON with patches."
        )

        mismatch_desc = json.dumps(mismatches[:10], indent=2)  # cap at 10
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
      "new": "replacement text"
    }}
  ]
}}"""

        try:
            result = await self.generate_json(system_prompt, user_prompt)
            patches = result.get("patches", []) if isinstance(result, dict) else []

            fixed = 0
            for patch in patches:
                filepath = code_dir / patch.get("file", "")
                try:
                    filepath.resolve().relative_to(code_dir.resolve())
                except ValueError:
                    continue
                old_text = patch.get("old", "")
                new_text = patch.get("new", "")
                if filepath.exists() and old_text and new_text:
                    content = filepath.read_text(encoding="utf-8", errors="replace")
                    if old_text in content:
                        filepath.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
                        fixed += 1
                        self.log(f"  Fixed import mismatch in {patch['file']}")
            self.log(f"Import fix: {fixed}/{len(patches)} patches applied")
        except Exception as e:
            self.log(f"Import fix failed (non-fatal): {e}")

    @staticmethod
    def _check_syntax(filepath: Path) -> bool:
        """Check if a Python file has valid syntax via py_compile."""
        try:
            result = subprocess.run(
                [sys.executable, "-c",
                 f"import py_compile; py_compile.compile(r'{filepath}', doraise=True)"],
                capture_output=True, text=True, timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return True  # assume OK if check itself fails

    @staticmethod
    def _get_best_round(state: IterationState) -> dict:
        """Return result data from the best round, or the last round as fallback."""
        if not state.rounds:
            return {
                "execution_status": "skipped",
                "quick_eval_status": "skipped",
                "metrics": {},
            }
        # Find best round by index
        best_idx = None
        if state.best_round is not None:
            for i, r in enumerate(state.rounds):
                if r.round_number == state.best_round:
                    best_idx = i
                    break
        # Fallback to last round
        if best_idx is None:
            best_idx = len(state.rounds) - 1

        best = state.rounds[best_idx]
        return {
            "execution_status": best.execution_status,
            "quick_eval_status": best.quick_eval_status,
            "metrics": best.metrics,
        }

    async def _run_quick_eval(
        self, code_dir: Path, venv_python: str, timeout: int | None = None,
    ) -> dict:
        """Execute main.py --quick-eval with up to 5 batch-fix cycles.

        Each cycle: run → collect all errors → fix ALL affected files at once
        → run again.  Much more efficient than fixing one file at a time.

        Returns {"status": "success"/"partial"/"failed"/"timeout", "metrics": {...}}.
        """
        if timeout is None:
            timeout = self.config.quick_eval_timeout

        self.log("Phase 4: Running quick-eval for real experiment results")

        max_fix_cycles = 5
        last_result: dict = {}
        fix_history: list[dict] = []  # Track previous fixes to avoid repeating

        for cycle in range(1, max_fix_cycles + 1):
            result = await self._run_quick_eval_subprocess(code_dir, venv_python, timeout)
            last_result = result
            if result["returncode"] == 0:
                return self._collect_quick_eval_results(code_dir, result, attempt=cycle)

            self.log(
                f"Quick-eval failed (cycle {cycle}/{max_fix_cycles}, "
                f"rc={result['returncode']}): {result['stderr'][:300]}"
            )

            if cycle >= max_fix_cycles:
                break

            # Timeout: special handling — inject speed-up edits, not traceback fixes
            if result["returncode"] == -1 and "imeout" in result.get("stderr", ""):
                try:
                    modified = await self._fix_timeout(code_dir)
                    if not modified:
                        self.log("Quick-eval: timeout fix did not modify any files, stopping")
                        break
                except Exception as e:
                    self.log(f"Quick-eval timeout fix error: {e}")
                    break
                continue

            # Batch fix ALL affected files (with fix history to avoid repeats)
            stderr_text = result.get("stderr", "")
            try:
                modified = await self._batch_fix_errors(
                    code_dir, stderr_text, "",
                    mode="quick-eval",
                    previous_fixes=fix_history,
                )
                fix_history.append({"error_msg": stderr_text[:300], "cycle": cycle})
                if not modified:
                    self.log("Quick-eval: no files modified by batch fix, stopping")
                    break
            except Exception as e:
                self.log(f"Quick-eval batch fix error: {e}")
                break

        return {"status": "failed", "metrics": {}, "attempts": cycle, **last_result}

    async def _run_quick_eval_subprocess(
        self, code_dir: Path, venv_python: str, timeout: int,
    ) -> dict:
        """Run main.py --quick-eval in a subprocess. Returns raw result dict."""
        loop = asyncio.get_running_loop()
        # Record metrics.json mtime before run to detect stale files on timeout
        metrics_path = code_dir / "results" / "metrics.json"
        mtime_before = metrics_path.stat().st_mtime if metrics_path.exists() else None
        try:
            proc_result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    [venv_python, "main.py", "--quick-eval"],
                    cwd=str(code_dir),
                    capture_output=True,
                    text=False,
                    timeout=timeout,
                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                ),
            )
            return {
                "returncode": proc_result.returncode,
                "stdout": _decode_bytes(proc_result.stdout, SUBPROCESS_OUTPUT_LIMIT),
                "stderr": _decode_bytes(proc_result.stderr, SUBPROCESS_OUTPUT_LIMIT),
            }
        except subprocess.TimeoutExpired:
            self.log(f"Quick-eval timed out after {timeout}s")
            # Check if metrics.json was written/updated DURING this run (not stale from previous round)
            if metrics_path.exists():
                mtime_after = metrics_path.stat().st_mtime
                if mtime_before is None or mtime_after > mtime_before:
                    metrics = self._parse_metrics_json(code_dir)
                    if metrics:
                        self.log("Quick-eval timed out BUT metrics.json was updated during run — treating as success")
                        return {"returncode": 0, "stdout": "", "stderr": ""}
            return {"returncode": -1, "stdout": "", "stderr": f"Timeout after {timeout}s"}
        except Exception as e:
            self.log(f"Quick-eval subprocess error: {e}")
            return {"returncode": -1, "stdout": "", "stderr": str(e)}

    def _collect_quick_eval_results(
        self, code_dir: Path, proc_result: dict, attempt: int,
    ) -> dict:
        """Parse metrics.json after a successful quick-eval run."""
        metrics = self._parse_metrics_json(code_dir)
        if metrics:
            self.log("Quick-eval succeeded — real experiment results collected")
            return {
                "status": "success",
                "metrics": metrics,
                "attempts": attempt,
                "stdout": proc_result.get("stdout", "")[:SUBPROCESS_OUTPUT_LIMIT],
                "stderr": proc_result.get("stderr", "")[:SUBPROCESS_OUTPUT_LIMIT],
            }
        else:
            self.log("Quick-eval ran (rc=0) but results/metrics.json missing or invalid")
            return {
                "status": "partial",
                "metrics": {},
                "attempts": attempt,
                "stdout": proc_result.get("stdout", "")[:SUBPROCESS_OUTPUT_LIMIT],
                "stderr": proc_result.get("stderr", "")[:SUBPROCESS_OUTPUT_LIMIT],
            }

    @staticmethod
    def _normalize_metrics_format(data: dict) -> dict:
        """Convert alternative metrics.json formats to the expected schema.

        Handles the common case where generated code writes:
          {"variants": {"full": {"accuracy": {"mean": X, "std": Y}, ...}, ...}}
        instead of the required:
          {"main_results": [...], "ablation_results": [...], "training_log": [...]}
        """
        expected_keys = {"main_results", "ablation_results", "training_log"}
        if expected_keys & set(data.keys()):
            return data  # Already in expected format

        variants = data.get("variants")
        if not isinstance(variants, dict) or not variants:
            # Fallback: top-level keys are variant dicts themselves
            # e.g. {"full_model": {"runs": ..., "aggregate": {...}}, "ablation_no_kd": {...}}
            candidate_variants = {}
            for k, v in data.items():
                if isinstance(v, dict) and ("aggregate" in v or "runs" in v or any(
                    isinstance(sv, dict) and "mean" in sv for sv in v.values()
                )):
                    candidate_variants[k] = v
            if not candidate_variants:
                return data  # Can't convert
            # Unwrap "aggregate" sub-dict if present
            variants = {}
            for k, v in candidate_variants.items():
                if "aggregate" in v and isinstance(v["aggregate"], dict):
                    variants[k] = v["aggregate"]
                else:
                    variants[k] = v

        # Convert variants dict → main_results + ablation_results
        main_results = []
        ablation_results = []
        dataset_name = data.get("dataset", "MNIST")

        for variant_name, metrics_dict in variants.items():
            if not isinstance(metrics_dict, dict):
                continue

            # Build metric list from the flat dict
            metric_list = []
            for mname, mval in metrics_dict.items():
                if any(mname.startswith(p) for p in (
                    "per_class", "confusion_matrix", "qualitative",
                )) or mname in (
                    "run_seed", "num_runs", "num_samples", "variant",
                    "training_time_sec", "parameter_count", "FLOPs_M",
                    "best_val_accuracy", "inference_time_ms",
                ):
                    continue  # Skip verbose breakdowns and metadata for summary
                if isinstance(mval, dict) and "mean" in mval:
                    metric_list.append({
                        "metric_name": mname,
                        "value": mval["mean"],
                        "std": mval.get("std", 0.0),
                        "num_runs": data.get("num_runs", 1),
                    })
                elif isinstance(mval, (int, float)):
                    metric_list.append({
                        "metric_name": mname,
                        "value": mval,
                    })

            if not metric_list:
                continue

            _vn = variant_name.lower().replace(" ", "_").replace("-", "_")
            is_proposed = (
                _vn in ("full", "full_model", "ours", "proposed", "calibrated")
                or _vn.startswith("full_model")
                or "proposed" in _vn or "ours" in _vn
            )
            # If there are only 2 variants, the non-ablation one is proposed
            if not is_proposed and len(variants) == 2 and "ablation" in str(list(variants.keys())).lower():
                if "ablation" not in _vn and "baseline" not in _vn and "w/o" not in _vn:
                    is_proposed = True
            main_results.append({
                "method_name": variant_name,
                "dataset": dataset_name,
                "is_proposed": is_proposed,
                "metrics": metric_list,
            })

            # Also add to ablation
            ablation_results.append({
                "variant_name": variant_name,
                "metrics": metric_list,
            })

        if main_results:
            data["main_results"] = main_results
        if ablation_results:
            data["ablation_results"] = ablation_results

        # Preserve training_log if present
        if "training_log" not in data:
            data["training_log"] = []

        logger.info("Converted variants-format metrics to standard schema "
                     "(%d main_results, %d ablation_results)",
                     len(main_results), len(ablation_results))
        return data

    @staticmethod
    def _parse_metrics_json(code_dir: Path) -> dict:
        """Read and validate results/metrics.json from the code directory.

        Returns the parsed dict if valid, empty dict otherwise.
        Validates:
          - Top-level is a dict with at least one expected key
          - main_results / ablation_results are lists (if present)
          - Metric values are finite real numbers (rejects NaN / Inf)
        """
        metrics_path = code_dir / "results" / "metrics.json"
        if not metrics_path.exists():
            return {}
        try:
            raw = metrics_path.read_text(encoding="utf-8")
            data = json.loads(raw)  # NaN/Inf handled by module-level helpers
            if not isinstance(data, dict):
                logger.warning("metrics.json is not a dict, skipping")
                return {}

            # Try to convert alternative formats to the expected schema
            data = ExperimentAgent._normalize_metrics_format(data)

            # Validate expected top-level keys (at least one should be present)
            expected_keys = {"main_results", "ablation_results", "training_log"}
            if not expected_keys & set(data.keys()):
                logger.warning("metrics.json has no expected keys (%s), skipping",
                               list(data.keys())[:5])
                return {}

            # Validate main_results structure
            main_results = data.get("main_results")
            if main_results is not None:
                if not isinstance(main_results, list):
                    logger.warning("main_results is not a list, dropping it")
                    data.pop("main_results")
                else:
                    # Sanitize: drop entries with NaN/Inf metric values
                    data["main_results"] = [
                        entry for entry in main_results
                        if isinstance(entry, dict)
                        and _all_metrics_finite(entry.get("metrics", []))
                    ]

            # Validate ablation_results structure
            ablation = data.get("ablation_results")
            if ablation is not None:
                if not isinstance(ablation, list):
                    logger.warning("ablation_results is not a list, dropping it")
                    data.pop("ablation_results")
                else:
                    data["ablation_results"] = [
                        entry for entry in ablation
                        if isinstance(entry, dict)
                        and _all_metrics_finite(entry.get("metrics", []))
                    ]

            # Validate training_log structure
            training_log = data.get("training_log")
            if training_log is not None:
                if not isinstance(training_log, list):
                    logger.warning("training_log is not a list, dropping it")
                    data.pop("training_log")
                else:
                    data["training_log"] = [
                        entry for entry in training_log
                        if isinstance(entry, dict)
                        and _training_entry_finite(entry)
                    ]

            # After sanitization, re-check that something non-empty remains
            if not any(data.get(k) for k in expected_keys):
                return {}

            return data
        except (json.JSONDecodeError, OSError, TypeError, AttributeError) as exc:
            logger.warning("Failed to parse metrics.json: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # Cluster execution
    # ------------------------------------------------------------------

    async def _run_on_cluster(
        self,
        cluster: "ClusterExecutor",
        code_dir: Path,
        round_num: int,
        cluster_code_path: str,
    ) -> tuple[dict, dict]:
        """Run experiment on SLURM cluster (local or remote).

        Returns (execution_dict, quick_eval_dict) in the same format as
        the local execution path.
        """
        session_id = self.workspace.path.name

        try:
            # Step 1: Prepare code on cluster
            if not cluster_code_path:
                self.log("Preparing code on cluster...")
                cluster_code_path = await cluster.prepare_code(code_dir, session_id)

                # Step 2: Create conda env + install deps (first round only)
                env_result = await cluster.setup_env(cluster_code_path)
                if not env_result["ok"]:
                    return (
                        {
                            "status": "failed",
                            "cluster_code_path": cluster_code_path,
                            "stderr": f"Environment setup failed:\n{env_result['output'][-2000:]}",
                        },
                        {"status": "skipped", "metrics": {}},
                    )
            else:
                # Re-sync code after LLM modifications
                self.log("Re-syncing code to cluster...")
                await cluster.reupload_code(code_dir, cluster_code_path)

            # Step 3: Submit SLURM job
            script_cmd = "python main.py --quick-eval"
            job_id = await cluster.submit_job(cluster_code_path, script_cmd)

            # Step 4: Wait for completion
            job_status = await cluster.wait_for_job(job_id)
            state = job_status.get("state", "UNKNOWN")

            # Step 5: Collect results or error logs
            if state == "COMPLETED":
                downloaded = await cluster.download_results(
                    cluster_code_path, self.workspace.path
                )
                if downloaded:
                    metrics = self._parse_metrics_json(code_dir)
                    if metrics:
                        self.log("Cluster experiment succeeded — real results collected!")
                        return (
                            {
                                "status": "success",
                                "cluster_code_path": cluster_code_path,
                                "job_id": job_id,
                                "stdout": f"Job {job_id} completed",
                                "stderr": "",
                            },
                            {"status": "success", "metrics": metrics},
                        )

                # Job completed but metrics.json missing/invalid
                log_text = await cluster.get_job_log(cluster_code_path, job_id)
                return (
                    {
                        "status": "failed",
                        "cluster_code_path": cluster_code_path,
                        "job_id": job_id,
                        "stdout": f"Job {job_id} rc=0 but metrics.json missing/invalid",
                        "stderr": log_text[-STDERR_SNIPPET_LIMIT:] if log_text else "",
                    },
                    {"status": "partial", "metrics": {}, "stderr": log_text},
                )
            else:
                # Job failed
                log_text = await cluster.get_job_log(cluster_code_path, job_id)
                self.log(f"Cluster job {job_id} failed ({state})")
                return (
                    {
                        "status": "failed",
                        "cluster_code_path": cluster_code_path,
                        "job_id": job_id,
                        "stdout": "",
                        "stderr": log_text[-STDERR_SNIPPET_LIMIT:] if log_text else f"Job {state}",
                    },
                    {"status": "failed", "metrics": {}, "stderr": log_text},
                )

        except Exception as e:
            self.log(f"Cluster execution error: {e}")
            return (
                {
                    "status": "failed",
                    "cluster_code_path": cluster_code_path,
                    "stderr": str(e),
                },
                {"status": "failed", "metrics": {}},
            )

    async def _execute_code_with_venv(
        self, generated_files: list[str], blueprint_summary: str
    ) -> tuple[dict, str]:
        """Run _execute_code and also return the venv python path for reuse."""
        code_dir = self.workspace.path / "code"
        main_py = code_dir / "main.py"

        if not main_py.exists():
            return (
                {"status": "skipped", "reason": "main.py not found", "stdout": "", "stderr": ""},
                sys.executable,
            )

        venv_python = await self._setup_venv(code_dir)
        result = await self._execute_code(
            generated_files, blueprint_summary,
            _code_dir=code_dir, _main_py=main_py, _venv_python=venv_python,
        )
        return result, venv_python

    async def _execute_code(
        self,
        generated_files: list[str],
        blueprint_summary: str,
        *,
        _code_dir: Path | None = None,
        _main_py: Path | None = None,
        _venv_python: str | None = None,
    ) -> dict:
        """Execute main.py --dry-run with up to 5 batch-fix cycles.

        Each cycle: run → collect all errors → fix ALL affected files in one
        LLM call → run again.  This is much more efficient than fixing one
        bug at a time.
        """
        code_dir = _code_dir or (self.workspace.path / "code")
        main_py = _main_py or (code_dir / "main.py")

        if not main_py.exists():
            return {"status": "skipped", "reason": "main.py not found", "stdout": "", "stderr": ""}

        venv_python = _venv_python or await self._setup_venv(code_dir)

        max_fix_cycles = 5
        last_result: dict = {}
        fix_history: list[dict] = []  # Track previous fixes to avoid repeating

        for cycle in range(1, max_fix_cycles + 1):
            result = await self._run_main_py(code_dir, venv_python)
            last_result = result
            if result["returncode"] == 0:
                status = "success" if cycle == 1 else "fixed"
                return {"status": status, "attempts": cycle, **result}

            self.log(f"Code execution failed (attempt {cycle}): {result['stderr'][:200]}")

            if cycle >= max_fix_cycles:
                break

            # Batch fix: identify ALL affected files and fix them in one call
            stderr_text = result["stderr"]
            try:
                modified = await self._batch_fix_errors(
                    code_dir, stderr_text, blueprint_summary,
                    mode="dry-run",
                    previous_fixes=fix_history,
                )
                fix_history.append({"error_msg": stderr_text[:300], "cycle": cycle})
                if not modified:
                    self.log("Dry-run: no files modified by batch fix, stopping")
                    break
            except Exception as e:
                self.log(f"Batch fix error in cycle {cycle}: {e}")
                break

        return {"status": "failed", "attempts": cycle, **last_result}

    async def _batch_fix_errors(
        self,
        code_dir: Path,
        stderr: str,
        blueprint_summary: str,
        mode: str = "dry-run",
        previous_fixes: list[dict] | None = None,
    ) -> list[str]:
        """Parse traceback, fix each affected file with a targeted LLM call.

        Surgical approach: for each file in the traceback, send ONLY that file
        + the error to the LLM → get a search-replace patch → apply.
        Uses 4-layer patch matching and syntax validation with rollback.

        Returns list of modified file paths.
        """
        import re as _re

        # 1. Parse traceback to find affected files with line numbers
        code_dir_str = str(code_dir.resolve()).replace("\\", "/")
        # Match: File "path", line N, in func
        tb_entries = _re.findall(
            r'File "([^"]+)",\s*line\s+(\d+)', stderr
        )

        # Deduplicate and filter to project files only (deepest frame first)
        affected: list[tuple[Path, int]] = []
        seen_files: set[str] = set()
        for fpath, lineno in reversed(tb_entries):
            f_norm = fpath.replace("\\", "/")
            resolved = Path(fpath).resolve()
            resolved_norm = str(resolved).replace("\\", "/")
            if code_dir_str not in resolved_norm:
                continue
            try:
                rel = str(resolved.relative_to(code_dir.resolve())).replace("\\", "/")
            except ValueError:
                continue
            if rel not in seen_files and resolved.exists():
                affected.append((resolved, int(lineno)))
                seen_files.add(rel)

        # If no project files found, default to main.py
        if not affected:
            main_py = code_dir / "main.py"
            if main_py.exists():
                affected = [(main_py, 0)]

        if not affected:
            return []

        # 2. Extract the final error message
        error_lines = stderr.strip().split("\n")
        error_msg = ""
        for line in reversed(error_lines):
            line = line.strip()
            if line and not line.startswith("File ") and not line.startswith("Traceback"):
                error_msg = line
                break

        # 3. Gather context: config files, requirements, imports
        context_files: list[str] = []
        for pattern in ("config/*.yaml", "config/*.yml", "config/*.json",
                        "*.yaml", "*.yml", "requirements.txt"):
            for cf in code_dir.glob(pattern):
                if cf.is_file():
                    try:
                        rel = str(cf.relative_to(code_dir)).replace("\\", "/")
                        ctx = cf.read_text(encoding="utf-8", errors="replace")[:1500]
                        context_files.append(f"--- {rel} ---\n{ctx}")
                    except OSError:
                        pass
        config_context = "\n\n".join(context_files) if context_files else "(no config files)"

        # Also list all project files for reference
        all_files = []
        for f in sorted(code_dir.rglob("*")):
            if f.is_file() and "__pycache__" not in str(f):
                all_files.append(str(f.relative_to(code_dir)).replace("\\", "/"))
        file_list = "\n".join(f"  {f}" for f in all_files)

        # 4. Fix each affected file with a targeted LLM call
        flag = "--quick-eval" if mode == "quick-eval" else "--dry-run"
        modified: list[str] = []
        code_gen_config = self.config.for_stage("code_gen")

        for target_file, error_line in affected:
            rel_path = str(target_file.relative_to(code_dir)).replace("\\", "/")
            try:
                content = target_file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            # Show context around the error line (±15 lines)
            lines = content.split("\n")
            if error_line > 0:
                start = max(0, error_line - 16)
                end = min(len(lines), error_line + 15)
                context_snippet = "\n".join(
                    f"{'>>>' if i+1 == error_line else '   '} {i+1:4d} | {l}"
                    for i, l in enumerate(lines[start:end], start=start)
                )
            else:
                context_snippet = content[:2000]

            # Build previous fix history to avoid repeating failed fixes
            fix_history = ""
            if previous_fixes:
                fix_history = (
                    "\n\nPrevious fix attempts that did NOT resolve the problem:\n"
                    + "\n".join(
                        f"  Round {i+1}: {fx.get('diagnosis', fx.get('error_msg', ''))[:200]}"
                        for i, fx in enumerate(previous_fixes)
                    )
                    + "\nDo NOT repeat the same fixes. Try a different approach.\n"
                )

            fix_prompt = (
                f"`python main.py {flag}` failed.\n\n"
                f"Error: {error_msg}\n\n"
                f"Full traceback (last 40 lines):\n```\n"
                f"{chr(10).join(error_lines[-40:])}\n```\n\n"
                f"File: {rel_path} (error around line {error_line}):\n```python\n"
                f"{context_snippet}\n```\n\n"
                f"Full file ({len(lines)} lines):\n```python\n{content[:4000]}\n```\n\n"
                f"== Config / Data Files (for reference) ==\n{config_context}\n\n"
                f"== Project Files ==\n{file_list}\n"
                f"{fix_history}\n"
                f"Output a JSON array of search-replace edits:\n"
                f'[{{"old": "exact text to find", "new": "replacement text"}}]\n\n'
                f"Rules:\n"
                f"- 'old' must be an EXACT substring of the file (including indentation)\n"
                f"- Multiple edits are fine — fix ALL issues in this file\n"
                f"- If config is YAML, use yaml.safe_load(), NOT json.load()\n"
                f"- Ensure imports match actual module structure\n"
                f"- Output ONLY valid JSON array, no markdown"
            )

            try:
                raw = await self._dispatcher.generate(
                    code_gen_config,
                    f"You are a Python debugging expert. Fix the bug in {rel_path} using precise search-replace edits.",
                    fix_prompt,
                )
                text = (raw or "").strip()
                if text.startswith("```"):
                    text_lines = text.split("\n")[1:]
                    if text_lines and text_lines[-1].strip().startswith("```"):
                        text_lines = text_lines[:-1]
                    text = "\n".join(text_lines)

                edits = json.loads(text)
                if not isinstance(edits, list):
                    edits = [edits]

                # Save backup for syntax rollback
                backup_content = content
                applied = 0
                for edit in edits:
                    if not isinstance(edit, dict):
                        continue
                    old = edit.get("old", "")
                    new = edit.get("new", "")
                    if not old:
                        continue

                    # 4-layer patch matching (borrowed from Deep Pipeline DebugAgent)
                    matched = False

                    # Layer 1: Exact match
                    if old in content:
                        content = content.replace(old, new, 1)
                        matched = True

                    # Layer 2: Strip trailing whitespace per line
                    if not matched:
                        def _strip_trailing(t: str) -> str:
                            return "\n".join(l.rstrip() for l in t.split("\n"))
                        c_stripped = _strip_trailing(content)
                        o_stripped = _strip_trailing(old)
                        if o_stripped in c_stripped:
                            content = c_stripped.replace(o_stripped, _strip_trailing(new), 1)
                            matched = True

                    # Layer 3: First-line + last-line fuzzy span match
                    if not matched:
                        old_parts = old.strip().split("\n")
                        if len(old_parts) >= 2:
                            first_line = old_parts[0].strip()
                            last_line = old_parts[-1].strip()
                            c_lines = content.split("\n")
                            for i in range(len(c_lines)):
                                if first_line and first_line in c_lines[i].strip():
                                    for j in range(i + len(old_parts) - 1,
                                                   min(i + len(old_parts) + 5, len(c_lines))):
                                        if last_line and last_line in c_lines[j].strip():
                                            new_lines = new.rstrip().split("\n")
                                            c_lines[i:j+1] = new_lines
                                            content = "\n".join(c_lines)
                                            matched = True
                                            break
                                    if matched:
                                        break

                    # Layer 4: Single-line match (strip + compare)
                    if not matched and "\n" not in old.strip():
                        old_line = old.strip()
                        c_lines = content.split("\n")
                        for i, line in enumerate(c_lines):
                            if old_line == line.strip():
                                indent = len(line) - len(line.lstrip())
                                new_parts = new.strip().split("\n")
                                indented = [" " * indent + nl.strip() if nl.strip() else "" for nl in new_parts]
                                c_lines[i:i+1] = indented
                                content = "\n".join(c_lines)
                                matched = True
                                break

                    if matched:
                        applied += 1

                if applied > 0:
                    # Syntax validation + rollback (borrowed from Deep Pipeline DebugAgent)
                    target_file.write_text(content, encoding="utf-8")
                    if target_file.suffix == ".py" and not self._check_syntax(target_file):
                        self.log(f"  Patch introduced syntax error in {rel_path}, rolling back")
                        target_file.write_text(backup_content, encoding="utf-8")
                    else:
                        modified.append(rel_path)
                        self.log(f"  Fixed {rel_path}: {applied} edit(s) applied")
                else:
                    self.log(f"  No edits matched in {rel_path}")

            except json.JSONDecodeError:
                # Fallback: LLM might return the full fixed file
                if text and len(text) > 50:
                    target_file.write_text(text, encoding="utf-8")
                    if target_file.suffix == ".py" and not self._check_syntax(target_file):
                        self.log(f"  Fallback rewrite has syntax error in {rel_path}, rolling back")
                        target_file.write_text(content, encoding="utf-8")
                    else:
                        modified.append(rel_path)
                        self.log(f"  Rewrote {rel_path} (fallback)")
            except Exception as e:
                self.log(f"  Fix failed for {rel_path}: {e}")

        if modified:
            self.log(f"Batch fix: modified {len(modified)} files")
        else:
            self.log("Batch fix: no files were modified")

        return modified

    async def _fix_timeout(self, code_dir: Path) -> list[str]:
        """When quick-eval times out, apply deterministic speed-ups to main.py.

        Instead of sending a vague "timeout" to the LLM for batch-fix, we apply
        targeted edits that reduce computation: fewer epochs, smaller subset,
        num_workers=0, num_runs=1.
        """
        main_py = code_dir / "main.py"
        if not main_py.exists():
            return []
        content = main_py.read_text(encoding="utf-8", errors="replace")
        original = content

        # 1. Reduce epochs to 2 (replace any epochs = N where N > 2)
        import re as _re
        # Handles: epochs = 5, epochs: 5 (YAML), training_cfg["epochs"] = max(3, min(5, ...))
        content = _re.sub(
            r'(\bepochs\b\s*[=:]\s*(?:max\(\d+,\s*min\(\d+,\s*(?:int\()?)?)(\d+)',
            lambda m: m.group(1) + ('2' if int(m.group(2)) > 2 else m.group(2)),
            content,
        )

        # 2. Reduce data subset size
        content = _re.sub(
            r'(subset_size\s*[=:]\s*)(\d+)',
            lambda m: m.group(1) + ('200' if int(m.group(2)) > 200 else m.group(2)),
            content,
        )
        content = _re.sub(
            r'(quick_eval_train_size["\']?\s*[,:]\s*)(\d+)',
            lambda m: m.group(1) + ('200' if int(m.group(2)) > 200 else m.group(2)),
            content,
        )

        # 3. Force num_runs = 1
        content = _re.sub(
            r'(num_runs\s*[=:]\s*)(\d+)',
            lambda m: m.group(1) + '1',
            content,
        )

        # 4. Force num_workers = 0 (avoid multiprocessing overhead on Windows)
        content = _re.sub(
            r'(num_workers\s*[=:]\s*)(\d+)',
            lambda m: m.group(1) + '0',
            content,
        )

        if content != original:
            main_py.write_text(content, encoding="utf-8")
            self.log("Timeout fix: reduced epochs/subset/workers in main.py")
            return ["main.py"]

        # If main.py regex didn't match anything, also try config/default.yaml
        config_yaml = code_dir / "config" / "default.yaml"
        if config_yaml.exists():
            cfg_content = config_yaml.read_text(encoding="utf-8", errors="replace")
            cfg_original = cfg_content
            cfg_content = _re.sub(
                r'(\bepochs\s*:\s*)(\d+)',
                lambda m: m.group(1) + ('2' if int(m.group(2)) > 2 else m.group(2)),
                cfg_content,
            )
            cfg_content = _re.sub(r'(num_workers\s*:\s*)(\d+)', r'\g<1>0', cfg_content)
            cfg_content = _re.sub(r'(num_runs\s*:\s*)(\d+)', r'\g<1>1', cfg_content)
            if cfg_content != cfg_original:
                config_yaml.write_text(cfg_content, encoding="utf-8")
                self.log("Timeout fix: reduced epochs/workers/runs in config/default.yaml")
                return ["config/default.yaml"]

        return []

    async def _setup_venv(self, code_dir: Path) -> str:
        """Prepare Python environment for experiment execution.

        If experiment_conda_env is configured, use that conda env's Python
        directly (skip venv creation — much faster, reuses existing packages).
        Otherwise, create an isolated venv and install requirements.

        Returns the path to the Python executable.
        """
        # --- Option A: Use existing conda env ---
        conda_env = self.config.experiment_conda_env
        if conda_env:
            conda_python = self._find_conda_python(conda_env)
            if conda_python:
                self.log(f"Using existing conda env '{conda_env}': {conda_python}")
                # Install any missing requirements into the conda env
                await self._install_missing_requirements(conda_python, code_dir)
                return conda_python
            else:
                self.log(f"Conda env '{conda_env}' not found, falling back to venv")

        # --- Option B: Create isolated venv ---
        venv_dir = code_dir / ".venv"
        is_windows = platform.system() == "Windows"
        if is_windows:
            venv_python = str(venv_dir / "Scripts" / "python.exe")
        else:
            venv_python = str(venv_dir / "bin" / "python")

        loop = asyncio.get_running_loop()

        if not venv_dir.exists():
            self.log("Creating isolated venv at code/.venv ...")
            try:
                await loop.run_in_executor(
                    None,
                    lambda: venv.create(str(venv_dir), with_pip=True),
                )
                self.log(f"Venv created (python: {venv_python})")
            except (OSError, subprocess.CalledProcessError) as e:
                self.log(f"Venv creation failed: {e}, falling back to system Python")
                return sys.executable
        else:
            self.log("Reusing existing venv at code/.venv")

        await self._install_missing_requirements(venv_python, code_dir)
        return venv_python

    @staticmethod
    def _find_conda_python(env_name: str) -> str | None:
        """Find the Python executable for a named conda env."""
        try:
            result = subprocess.run(
                ["conda", "run", "-n", env_name, "python", "-c",
                 "import sys; print(sys.executable)"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                path = result.stdout.strip().split("\n")[-1].strip()
                if path and Path(path).exists():
                    return path
        except Exception:
            pass

        # Fallback: check common paths
        is_windows = platform.system() == "Windows"
        for base in [Path.home() / "anaconda3", Path.home() / "miniconda3",
                      Path("D:/anaconda"), Path("C:/anaconda3")]:
            if is_windows:
                p = base / "envs" / env_name / "python.exe"
            else:
                p = base / "envs" / env_name / "bin" / "python"
            if p.exists():
                return str(p)
        return None

    async def _install_missing_requirements(self, python: str, code_dir: Path) -> None:
        """pip install requirements.txt if it exists (skips already-installed)."""
        req_file = code_dir / "requirements.txt"
        if not req_file.exists():
            self.log("No requirements.txt found, skipping pip install")
            return

        self.log("Installing requirements.txt (skipping already-installed)...")
        loop = asyncio.get_running_loop()
        try:
            proc_result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    [python, "-m", "pip", "install",
                     "-r", str(req_file), "--quiet"],
                    cwd=str(code_dir),
                    capture_output=True,
                    text=False,
                    timeout=600,
                ),
            )
            if proc_result.returncode == 0:
                self.log("Requirements OK")
            else:
                self.log(
                    f"pip install warnings (rc={proc_result.returncode}): "
                    f"{_decode_bytes(proc_result.stderr, 500)}"
                )
        except Exception as e:
            self.log(f"pip install error: {e}")

    async def _run_main_py(self, code_dir: Path, python: str | None = None) -> dict:
        """Run main.py in a subprocess with timeout."""
        python = python or sys.executable
        loop = asyncio.get_running_loop()
        try:
            proc_result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    [python, "main.py", "--dry-run"],
                    cwd=str(code_dir),
                    capture_output=True,
                    text=False,
                    timeout=DRY_RUN_TIMEOUT_SECONDS,
                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                ),
            )
            return {
                "returncode": proc_result.returncode,
                "stdout": _decode_bytes(proc_result.stdout, SUBPROCESS_OUTPUT_LIMIT),
                "stderr": _decode_bytes(proc_result.stderr, SUBPROCESS_OUTPUT_LIMIT),
            }
        except subprocess.TimeoutExpired:
            return {"returncode": -1, "stdout": "", "stderr": f"Timeout after {DRY_RUN_TIMEOUT_SECONDS}s"}
        except Exception as e:
            return {"returncode": -1, "stdout": "", "stderr": str(e)}

    @staticmethod
    def _build_repo_context(reference_repos: list[dict]) -> str:
        """Build a context string from reference GitHub repos."""
        if not reference_repos:
            return ""

        lines = ["=== REFERENCE GITHUB REPOSITORIES ==="]
        lines.append("Use these real open-source projects as structural references.")
        lines.append("Mirror their project layout, naming conventions, and best practices.\n")

        for repo in reference_repos[:MAX_REFERENCE_REPOS]:
            name = repo.get("full_name", "unknown")
            desc = repo.get("description", "")
            stars = repo.get("stars", 0)
            tree = repo.get("file_tree", [])
            readme = repo.get("readme_excerpt", "")

            lines.append(f"--- {name} ({stars} stars) ---")
            if desc:
                lines.append(f"Description: {desc}")

            if tree:
                lines.append("File structure:")
                for path in tree[:MAX_FILE_TREE_ENTRIES]:
                    lines.append(f"  {path}")
                if len(tree) > MAX_FILE_TREE_ENTRIES:
                    lines.append(f"  ... ({len(tree) - MAX_FILE_TREE_ENTRIES} more files)")

            if readme:
                lines.append(f"README excerpt:\n  {readme[:MAX_README_EXCERPT_LENGTH]}")

            lines.append("")

        lines.append("=== END REFERENCE REPOS ===")
        return "\n".join(lines)

    async def _generate_project_plan(self, blueprint_summary: str, repo_context: str = "") -> dict:
        """Phase 1: Generate the project plan JSON via Codex."""
        repo_section = ""
        if repo_context:
            repo_section = (
                f"\n{repo_context}\n\n"
                "IMPORTANT: Model your project structure after the reference repos above.\n"
                "Use similar directory layouts, naming conventions, and design patterns.\n"
                "The generated code should look like it belongs in one of these real repos.\n"
            )

        prompt = f"""Design a complete Python ML project for this experiment:

{blueprint_summary}
{repo_section}
The project must be a self-contained, runnable research codebase with:
- Full model architecture implementation
- Data loading and preprocessing pipeline
- Training loop with checkpoint saving and logging
- Evaluation with all specified metrics
- Ablation experiment support
- Configuration via YAML
- Shell scripts for launching experiments

Output the project plan as a JSON object."""

        code_gen_config = self.config.for_stage("code_gen")
        raw = await self._dispatcher.generate(
            code_gen_config, PROJECT_PLAN_SYSTEM_PROMPT, prompt, json_mode=True
        )

        # Parse JSON (handle markdown fences)
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove opening fence (e.g. ```json)
            lines = lines[1:]
            # Remove only the last closing fence
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines)

        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            logger.error(
                "Failed to parse project plan JSON. First 500 chars: %s",
                text[:500],
            )
            raise RuntimeError(
                f"Project plan is not valid JSON: {exc}"
            ) from exc

    async def _generate_file(
        self,
        file_spec: dict,
        interface_contract: str,
        blueprint_summary: str,
        repo_context: str = "",
    ) -> str:
        """Phase 2: Generate a single file via Codex."""
        file_path = file_spec["path"]
        description = file_spec.get("description", "")
        interfaces = file_spec.get("interfaces", [])
        depends_on = file_spec.get("depends_on", [])

        repo_section = ""
        if repo_context:
            repo_section = (
                f"\n{repo_context}\n\n"
                "Write code that follows the patterns and conventions of the reference repos above.\n"
            )

        prompt = f"""Generate the file: {file_path}
Description: {description}

This file must implement these interfaces:
{json.dumps(interfaces, indent=2)}

Dependencies (other project files this imports from):
{json.dumps(depends_on, indent=2)}

=== FULL PROJECT INTERFACE CONTRACT ===
{interface_contract}
=== END CONTRACT ===

=== EXPERIMENT BLUEPRINT ===
{blueprint_summary}
=== END BLUEPRINT ===
{repo_section}
Generate the COMPLETE file content. Follow the interface contract exactly."""

        code_gen_config = self.config.for_stage("code_gen")
        content = await self._dispatcher.generate(
            code_gen_config, FILE_GEN_SYSTEM_PROMPT, prompt
        )

        # Clean up: remove markdown fences if present (handles ```python, ```json, etc.)
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            # Remove opening fence line
            lines = lines[1:]
            # Remove only the last closing fence
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines)

        return content

    def _verify_code(self, generated_files: list[str]) -> dict:
        """Verify generated Python files have valid syntax via compile()."""
        results = []
        passed = 0
        total = 0

        for fp in generated_files:
            if not fp.endswith(".py"):
                continue

            total += 1
            file_path = self.workspace.path / "code" / fp
            if not file_path.exists():
                results.append({
                    "file": fp,
                    "status": "missing",
                    "error": "File not found",
                })
                continue

            try:
                source = file_path.read_text(encoding="utf-8")
                compile(source, fp, "exec")
                results.append({"file": fp, "status": "ok", "error": None})
                passed += 1
            except SyntaxError as e:
                results.append({
                    "file": fp,
                    "status": "syntax_error",
                    "error": f"Line {e.lineno}: {e.msg}",
                })

        return {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "files": results,
        }
