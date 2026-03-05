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
- config/default.yaml (hyperparameters and paths)
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
                    resolved = (self.workspace.path / "code" / file_path).resolve()
                    if not str(resolved).startswith(str(code_root)):
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
                prev_analysis = iteration_state.rounds[-1].analysis
                history_summary = self._build_history_summary(iteration_state.rounds)

                hypothesis = await self._generate_iteration_hypothesis(
                    prev_analysis, history_summary, blueprint_summary
                )
                hypothesis.round_number = round_num

                self.log(f"Hypothesis: {hypothesis.hypothesis[:100]}")

                files_modified = await self._apply_iteration_changes(
                    hypothesis, code_dir, venv_python
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
                # On round 1 blocking failure, still try to continue if we have more rounds
                if round_num == 1:
                    continue
                # On round 2+, stop iterations
                iteration_state.final_status = "failed"
                break

            # ---- Phase 3: dry-run ----
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
            self.workspace.write_json(
                f"logs/iteration_round_{round_num}_execution.json", execution
            )
            execution_status = execution.get("status", "failed")
            self.log(f"Dry-run: {execution_status}")

            # ---- Phase 4: quick-eval ----
            quick_eval: dict = {"status": "skipped", "metrics": {}}
            if execution_status in ("success", "fixed"):
                quick_eval = await self._run_quick_eval(code_dir, venv_python)
                self.log(f"Quick-eval: {quick_eval['status']}")
            else:
                self.log("Skipping quick-eval (dry-run did not succeed)")

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

            # Track best round
            if analysis.metric_summary:
                primary_value = next(iter(analysis.metric_summary.values()), None)
                best_value = next(iter(iteration_state.best_metrics.values()), None) if iteration_state.best_metrics else None
                if best_value is None or (primary_value is not None and primary_value > best_value):
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

        prompt = f"""Based on the previous experiment round's feedback, generate a hypothesis for the next improvement iteration.

== Previous Analysis ==
{analysis_text or "No analysis available."}

== History ==
{history_summary or "No previous rounds."}

== Experiment Blueprint ==
{blueprint[:2000]}

Output a JSON object with:
{{
  "hypothesis": "<what you will change and why>",
  "planned_changes": ["<file: specific change>", ...],
  "expected_signal": "<what metric improvement you expect>",
  "rationale": "<reasoning>"
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
        """LLM modifies specific files based on the hypothesis (not full regeneration)."""
        # Collect current file contents for context
        file_contents: dict[str, str] = {}
        for py_file in code_dir.rglob("*.py"):
            parts = py_file.relative_to(code_dir).parts
            if any(p.startswith(".") or p == "__pycache__" for p in parts):
                continue
            try:
                rel = str(py_file.relative_to(code_dir)).replace("\\", "/")
                content = py_file.read_text(encoding="utf-8", errors="replace")
                file_contents[rel] = content[:3000]  # truncate for context
            except OSError:
                continue

        # Also include config/default.yaml
        yaml_path = code_dir / "config" / "default.yaml"
        if yaml_path.exists():
            try:
                file_contents["config/default.yaml"] = yaml_path.read_text(
                    encoding="utf-8", errors="replace"
                )[:2000]
            except OSError:
                pass

        files_summary = "\n".join(
            f"--- {path} ---\n{content[:1500]}\n"
            for path, content in file_contents.items()
        )

        prompt = f"""Apply the following changes to the experiment code project.

== Hypothesis ==
{hypothesis.hypothesis}

== Planned Changes ==
{json.dumps(hypothesis.planned_changes, indent=2)}

== Rationale ==
{hypothesis.rationale}

== Current Files ==
{files_summary[:12000]}

For each file you want to modify, output a JSON array of objects:
[
  {{
    "path": "relative/path.py",
    "content": "FULL file content (not a diff)"
  }},
  ...
]

Only include files that need changes. Output the COMPLETE file content for each.
Output ONLY valid JSON array."""

        modified_files: list[str] = []
        try:
            code_gen_config = self.config.for_stage("code_gen")
            raw = await self._dispatcher.generate(
                code_gen_config,
                "You are an ML code editor. Modify the specified files to implement the hypothesis. Output ONLY a JSON array of {path, content} objects.",
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
                if not isinstance(change, dict) or "path" not in change or "content" not in change:
                    continue
                file_path = change["path"]
                # Security: prevent directory traversal
                resolved = (code_dir / file_path).resolve()
                code_root = code_dir.resolve()
                if not str(resolved).startswith(str(code_root)):
                    logger.warning("Skipping unsafe iteration path: %s", file_path)
                    continue

                content = change["content"]
                self.workspace.write_text(f"code/{file_path}", content)
                modified_files.append(file_path)
                self.log(f"  Modified: {file_path}")

        except Exception as exc:
            logger.warning("Failed to apply iteration changes: %s", exc)

        return modified_files

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
        """Execute main.py --quick-eval and read results/metrics.json.

        If the first attempt fails, try LLM-assisted fix once (same pattern
        as _execute_code for --dry-run).

        Returns {"status": "success"/"partial"/"failed"/"timeout", "metrics": {...}}.
        """
        if timeout is None:
            timeout = self.config.quick_eval_timeout

        self.log("Phase 4: Running quick-eval for real experiment results")

        # --- Attempt 1 ---
        result = await self._run_quick_eval_subprocess(code_dir, venv_python, timeout)
        if result["returncode"] == 0:
            return self._collect_quick_eval_results(code_dir, result, attempt=1)

        self.log(
            f"Quick-eval failed (attempt 1, rc={result['returncode']}): "
            f"{result['stderr'][:300]}"
        )

        # --- Attempt 2: LLM fix ---
        main_py = code_dir / "main.py"
        if not main_py.exists():
            return {"status": "failed", "metrics": {}, "attempts": 1, **result}

        try:
            fix_prompt = (
                f"The following Python code failed when executed with --quick-eval.\n\n"
                f"Error output:\n```\n{result['stderr'][:STDERR_SNIPPET_LIMIT]}\n```\n\n"
                f"Original main.py (first 4000 chars):\n```python\n"
                f"{main_py.read_text(encoding='utf-8')[:LLM_CONTEXT_TRUNCATION]}\n```\n\n"
                f"Fix the code so that `python main.py --quick-eval` runs successfully.\n"
                f"It must:\n"
                f"1. Train a scaled-down model for a few epochs\n"
                f"2. Write results/metrics.json with main_results, ablation_results, training_log\n"
                f"3. All numeric values must come from actual computation\n\n"
                f"Output ONLY the fixed Python code, no markdown fences."
            )

            code_gen_config = self.config.for_stage("code_gen")
            fixed_code = await self._dispatcher.generate(
                code_gen_config,
                "You are a Python debugging expert. Fix the code to make --quick-eval work.",
                fix_prompt,
            )
            fixed_code = fixed_code.strip()
            if fixed_code.startswith("```"):
                lines = fixed_code.split("\n")[1:]
                if lines and lines[-1].strip().startswith("```"):
                    lines = lines[:-1]
                fixed_code = "\n".join(lines)

            main_py.write_text(fixed_code, encoding="utf-8")
            self.log("Quick-eval: LLM fix applied, retrying...")

            result2 = await self._run_quick_eval_subprocess(code_dir, venv_python, timeout)
            if result2["returncode"] == 0:
                return self._collect_quick_eval_results(code_dir, result2, attempt=2)
            else:
                self.log(f"Quick-eval failed after LLM fix (rc={result2['returncode']})")
                return {"status": "failed", "metrics": {}, "attempts": 2, **result2}
        except Exception as e:
            self.log(f"Quick-eval LLM fix error: {e}")
            return {"status": "failed", "metrics": {}, "attempts": 1, "error": str(e), **result}

    async def _run_quick_eval_subprocess(
        self, code_dir: Path, venv_python: str, timeout: int,
    ) -> dict:
        """Run main.py --quick-eval in a subprocess. Returns raw result dict."""
        loop = asyncio.get_running_loop()
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

            # Validate expected top-level keys (at least one should be present)
            expected_keys = {"main_results", "ablation_results", "training_log"}
            if not expected_keys & set(data.keys()):
                logger.warning("metrics.json has no expected keys, skipping")
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
        """Attempt to execute the generated main.py with --dry-run flag.

        Creates an isolated venv, installs dependencies there, and runs
        main.py using the venv's Python.  If execution fails, ask the LLM
        to fix the code and retry once.
        This is non-blocking for the pipeline — failures are logged but
        do not prevent the pipeline from continuing.
        """
        code_dir = _code_dir or (self.workspace.path / "code")
        main_py = _main_py or (code_dir / "main.py")

        if not main_py.exists():
            return {"status": "skipped", "reason": "main.py not found", "stdout": "", "stderr": ""}

        # Create isolated venv and install requirements
        venv_python = _venv_python or await self._setup_venv(code_dir)

        # First attempt
        result = await self._run_main_py(code_dir, venv_python)
        if result["returncode"] == 0:
            return {"status": "success", "attempts": 1, **result}

        self.log(f"Code execution failed (attempt 1): {result['stderr'][:200]}")

        # Try to fix with LLM
        try:
            fix_prompt = f"""The following Python code failed to execute.

Error output:
```
{result['stderr'][:STDERR_SNIPPET_LIMIT]}
```

Original main.py:
```python
{main_py.read_text(encoding='utf-8')[:LLM_CONTEXT_TRUNCATION]}
```

Blueprint:
{blueprint_summary[:2000]}

Fix the code so it can at least import and run without errors.
A simple --dry-run that just validates imports and configuration is acceptable.
Output ONLY the fixed Python code, no markdown fences."""

            code_gen_config = self.config.for_stage("code_gen")
            fixed_code = await self._dispatcher.generate(
                code_gen_config,
                "You are a Python debugging expert. Fix the code to make it runnable.",
                fix_prompt,
            )
            # Clean markdown fences
            fixed_code = fixed_code.strip()
            if fixed_code.startswith("```"):
                lines = fixed_code.split("\n")[1:]
                if lines and lines[-1].strip().startswith("```"):
                    lines = lines[:-1]
                fixed_code = "\n".join(lines)

            # Write fixed code
            main_py.write_text(fixed_code, encoding="utf-8")

            # Second attempt
            result2 = await self._run_main_py(code_dir, venv_python)
            if result2["returncode"] == 0:
                return {"status": "fixed", "attempts": 2, **result2}
            else:
                return {"status": "failed", "attempts": 2, **result2}
        except Exception as e:
            return {
                "status": "failed",
                "attempts": 1,
                "error": f"LLM fix failed: {e}",
                **result,
            }

    async def _setup_venv(self, code_dir: Path) -> str:
        """Create an isolated venv under code/.venv and install requirements.

        Returns the path to the venv's Python executable.
        No hard timeout — large packages (torch, transformers) may need
        a long download depending on network conditions.
        """
        venv_dir = code_dir / ".venv"
        is_windows = platform.system() == "Windows"
        if is_windows:
            venv_python = str(venv_dir / "Scripts" / "python.exe")
        else:
            venv_python = str(venv_dir / "bin" / "python")

        loop = asyncio.get_running_loop()

        # --- 1. Create venv (fast, no network) ---
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

        # --- 2. pip install requirements (no timeout) ---
        req_file = code_dir / "requirements.txt"
        if not req_file.exists():
            self.log("No requirements.txt found, skipping pip install")
            return venv_python

        self.log("Installing requirements.txt into venv (this may take a while) ...")
        try:
            proc_result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    [venv_python, "-m", "pip", "install",
                     "-r", str(req_file), "--quiet"],
                    cwd=str(code_dir),
                    capture_output=True,
                    text=False,
                ),
            )
            if proc_result.returncode == 0:
                self.log("Requirements installed successfully in venv")
            else:
                self.log(
                    f"pip install failed (rc={proc_result.returncode}): "
                    f"{_decode_bytes(proc_result.stderr, 500)}"
                )
        except Exception as e:
            self.log(f"pip install error: {e}")

        return venv_python

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
