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
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from nanoresearch.agents.base import BaseResearchAgent, _fix_json_escapes, _repair_truncated_json
from nanoresearch.agents.cluster_executor import ClusterExecutor
from nanoresearch.agents.experiment_tools import build_experiment_tools
from nanoresearch.agents.feedback_analyzer import FeedbackAnalyzer
from nanoresearch.agents.preflight import PreflightChecker
from nanoresearch.agents.project_runner import RUNNER_SCRIPT_NAME, ensure_project_runner
from nanoresearch.agents.repair_journal import (
    append_snapshot_journal,
    capture_repair_snapshot,
    rollback_snapshot,
)
from nanoresearch.agents.runtime_env import RuntimeEnvironmentManager
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


from nanoresearch.prompts import load_prompt as _load_prompt

PROJECT_PLAN_SYSTEM_PROMPT = _load_prompt("experiment", "project_plan")

FILE_GEN_SYSTEM_PROMPT = _load_prompt("experiment", "file_gen")


def _is_finite(value: Any) -> bool:
    """Check if a numeric value is a finite real number."""
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    return False  # non-numeric values are not valid metrics


def _all_metrics_finite(metrics: list) -> bool:
    """Check that all metric values in a list are finite numbers.

    Returns False only if there are NO valid metrics at all.
    NaN/Inf values are replaced with None in-place so the row is kept
    with its valid data.
    """
    if not isinstance(metrics, list):
        return False
    has_valid = False
    for m in metrics:
        if not isinstance(m, dict):
            continue
        val = m.get("value")
        if val is not None:
            if _is_finite(val):
                has_valid = True
            else:
                m["value"] = None  # replace NaN/Inf, keep the row
    return has_valid


def _training_entry_finite(entry: dict) -> bool:
    """Check that numeric fields in a training log entry are finite.

    Replaces NaN/Inf values with None in-place.  Returns False if the entry
    has no remaining finite numeric data (loss or metrics) after sanitization.
    """
    has_finite = False
    for key in ("train_loss", "val_loss"):
        val = entry.get(key)
        if val is not None:
            if _is_finite(val):
                has_finite = True
            else:
                entry[key] = None  # replace NaN/Inf
    metrics = entry.get("metrics", {})
    if not isinstance(metrics, dict):
        return False  # malformed metrics field
    for mk, mv in list(metrics.items()):
        if mv is not None and not _is_finite(mv):
            metrics[mk] = None  # replace NaN/Inf
        elif isinstance(mv, (int, float)):
            has_finite = True
    return has_finite


_METRIC_NAME_HINTS = frozenset({
    "acc", "loss", "err", "f1", "prec", "recall", "auc", "mse", "mae",
    "rmse", "bleu", "rouge", "cer", "wer", "perp", "fid", "score",
    "iou", "map", "ndcg", "psnr", "ssim", "dice", "top1", "top5",
})


def _has_metric_name_hint(metric_list: list[dict]) -> bool:
    """Check if any extracted metric name matches a common metric substring."""
    for entry in metric_list:
        name = str(entry.get("metric_name", "")).lower().replace("-", "_")
        if any(hint in name for hint in _METRIC_NAME_HINTS):
            return True
    return False


def _metric_entries_from_mapping(mapping: dict, *, num_runs: int | None = None) -> list[dict[str, Any]]:
    """Extract summary metric entries from a flat/nested metrics mapping."""
    metric_list: list[dict[str, Any]] = []
    for mname, mval in mapping.items():
        if any(str(mname).startswith(prefix) for prefix in ("per_class", "confusion_matrix", "qualitative")):
            continue
        if str(mname) in {
            "run_seed", "num_runs", "num_samples", "variant", "training_time_sec",
            "parameter_count", "FLOPs_M", "best_val_accuracy", "inference_time_ms",
            "epoch", "step", "dataset", "method_name", "model_name", "name",
        }:
            continue

        if isinstance(mval, dict) and "mean" in mval and _is_finite(mval.get("mean")):
            entry = {
                "metric_name": str(mname),
                "value": mval["mean"],
                "std": mval.get("std", 0.0),
            }
            if num_runs is not None:
                entry["num_runs"] = num_runs
            metric_list.append(entry)
        elif _is_finite(mval):
            entry = {
                "metric_name": str(mname),
                "value": mval,
            }
            if num_runs is not None:
                entry["num_runs"] = num_runs
            metric_list.append(entry)
    return metric_list



from .react_mode import _ReactModeMixin
from .iteration import _IterationMixin
from .quick_eval import _QuickEvalMixin
from .code_runner import _CodeRunnerMixin
from .code_gen import _CodeGenMixin


class ExperimentAgent(
    _ReactModeMixin,
    _IterationMixin,
    _QuickEvalMixin,
    _CodeRunnerMixin,
    _CodeGenMixin,
    BaseResearchAgent,
):
    stage = PipelineStage.EXPERIMENT

    @staticmethod
    def _strip_json_fence(raw: str) -> str:
        text = str(raw or "").strip()
        if text.startswith("```"):
            lines = text.split("\n")[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines)
        return text

    @staticmethod
    def _json_parse_candidates(text: str) -> list[str]:
        stripped = str(text or "").strip()
        if not stripped:
            return [""]

        candidates = [stripped]
        bracket_positions = [
            index for index in (stripped.find("{"), stripped.find("[")) if index >= 0
        ]
        if bracket_positions:
            first_json_index = min(bracket_positions)
            if first_json_index > 0:
                candidates.append(stripped[first_json_index:])
        return candidates

    @staticmethod
    def _decode_json_value(text: str, *, strict: bool) -> Any:
        decoder = json.JSONDecoder(strict=strict)
        value, _end = decoder.raw_decode(text.lstrip())
        return value

    @classmethod
    def _parse_llm_json_payload(cls, raw: str) -> Any:
        text = cls._strip_json_fence(raw)

        last_error: json.JSONDecodeError | None = None
        for candidate in cls._json_parse_candidates(text):
            try:
                return cls._decode_json_value(candidate, strict=True)
            except json.JSONDecodeError as exc:
                last_error = exc

        fixed = _fix_json_escapes(text)
        for candidate in cls._json_parse_candidates(fixed):
            try:
                return cls._decode_json_value(candidate, strict=False)
            except json.JSONDecodeError as exc:
                last_error = exc

        repaired = _repair_truncated_json(fixed)
        if repaired is not None:
            for candidate in cls._json_parse_candidates(repaired):
                try:
                    return cls._decode_json_value(candidate, strict=False)
                except json.JSONDecodeError as exc:
                    last_error = exc

        if last_error is not None:
            raise last_error
        raise json.JSONDecodeError("Invalid JSON payload", text, 0)

    @staticmethod
    def _line_range_to_offsets(lines: list[str], start: int, end: int) -> tuple[int, int]:
        start_offset = sum(len(line) for line in lines[:start])
        end_offset = sum(len(line) for line in lines[:end])
        return start_offset, end_offset

    @classmethod
    def _find_rstrip_line_span(cls, content: str, old: str) -> tuple[int, int] | None:
        old_lines = old.splitlines()
        if not old_lines:
            return None

        content_lines = content.splitlines(keepends=True)
        if len(old_lines) > len(content_lines):
            return None

        for start in range(len(content_lines) - len(old_lines) + 1):
            if all(
                content_lines[start + index].rstrip() == old_lines[index].rstrip()
                for index in range(len(old_lines))
            ):
                return cls._line_range_to_offsets(
                    content_lines,
                    start,
                    start + len(old_lines),
                )
        return None

    @classmethod
    def _find_anchor_span(
        cls,
        content: str,
        old: str,
        *,
        max_extra_lines: int = 8,
    ) -> tuple[int, int] | None:
        old_lines = [line.strip() for line in old.splitlines() if line.strip()]
        if len(old_lines) < 2:
            return None

        first_line = old_lines[0]
        last_line = old_lines[-1]
        content_lines = content.splitlines(keepends=True)
        if not content_lines:
            return None

        for start in range(len(content_lines)):
            if first_line not in content_lines[start].strip():
                continue
            min_end = start + max(1, len(old_lines) - 1)
            max_end = min(len(content_lines), start + len(old_lines) + max_extra_lines)
            for end in range(min_end, max_end):
                if last_line and last_line in content_lines[end - 1].strip():
                    return cls._line_range_to_offsets(content_lines, start, end)
        return None

    @classmethod
    def _find_definition_block_span(cls, content: str, old: str) -> tuple[int, int] | None:
        first_nonempty = next((line.strip() for line in old.splitlines() if line.strip()), "")
        if not first_nonempty:
            return None

        signature_match = re.match(r"^(async\s+def|def|class)\s+([A-Za-z_]\w*)\b", first_nonempty)
        if not signature_match:
            return None

        keyword = signature_match.group(1)
        name = signature_match.group(2)
        content_lines = content.splitlines(keepends=True)
        definition_pattern = re.compile(r"^(async\s+def|def|class)\s+([A-Za-z_]\w*)\b")

        for start, line in enumerate(content_lines):
            stripped = line.strip()
            match = definition_pattern.match(stripped)
            if not match:
                continue
            if match.group(1) != keyword or match.group(2) != name:
                continue

            indent = len(line) - len(line.lstrip())
            end = start + 1
            while end < len(content_lines):
                next_line = content_lines[end]
                next_stripped = next_line.strip()
                if not next_stripped:
                    end += 1
                    continue

                next_indent = len(next_line) - len(next_line.lstrip())
                if next_indent <= indent:
                    if definition_pattern.match(next_stripped):
                        break
                    if next_stripped.startswith("@"):
                        lookahead = end + 1
                        while lookahead < len(content_lines) and not content_lines[lookahead].strip():
                            lookahead += 1
                        if lookahead < len(content_lines):
                            decorated = content_lines[lookahead].strip()
                            decorated_match = definition_pattern.match(decorated)
                            if decorated_match and (
                                len(content_lines[lookahead]) - len(content_lines[lookahead].lstrip())
                            ) <= indent:
                                break
                end += 1

            return cls._line_range_to_offsets(content_lines, start, end)
        return None

    @classmethod
    def _apply_search_replace_edit(
        cls,
        content: str,
        old: str,
        new: str,
    ) -> tuple[str, bool, str]:
        if not old:
            return content, False, ""

        if old in content:
            return content.replace(old, new, 1), True, "exact"

        rstrip_span = cls._find_rstrip_line_span(content, old)
        if rstrip_span is not None:
            start, end = rstrip_span
            return content[:start] + new + content[end:], True, "rstrip_lines"

        if "\n" not in old.strip():
            old_line = old.strip()
            content_lines = content.splitlines(keepends=True)
            for index, line in enumerate(content_lines):
                if line.strip() != old_line:
                    continue
                indent = len(line) - len(line.lstrip())
                replacement_lines = new.strip().split("\n")
                replacement = "\n".join(
                    (" " * indent + item.strip()) if item.strip() else ""
                    for item in replacement_lines
                )
                if line.endswith("\n") and not replacement.endswith("\n"):
                    replacement += "\n"
                start, end = cls._line_range_to_offsets(content_lines, index, index + 1)
                return content[:start] + replacement + content[end:], True, "single_line_stripped"

        anchor_span = cls._find_anchor_span(content, old)
        if anchor_span is not None:
            start, end = anchor_span
            return content[:start] + new + content[end:], True, "anchor_span"

        definition_span = cls._find_definition_block_span(content, old)
        if definition_span is not None:
            start, end = definition_span
            return content[:start] + new + content[end:], True, "definition_block"

        return content, False, ""

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
                ), return_exceptions=True)

                # Write files sequentially (filesystem ops)
                for spec, content in zip(valid_specs, contents):
                    file_path = spec["path"]
                    if isinstance(content, BaseException):
                        logger.error("Failed to generate %s: %s", file_path, content)
                        continue
                    self.workspace.write_text(f"code/{file_path}", content)
                    generated_files.append(file_path)

                # Phase 2b: Cross-file import consistency check
                import_mismatches = self._check_import_consistency(code_dir)
                if import_mismatches:
                    self.log(f"Found {len(import_mismatches)} import mismatches, fixing...")
                    await self._fix_import_mismatches(code_dir, import_mismatches)

                # Phase 2c: Auto-format generated code (optional, non-blocking)
                await self._format_generated_code(code_dir)

                # Phase 2d: Smoke test (import check, non-blocking)
                self._generate_and_run_smoke_test(code_dir, generated_files)

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
                    iteration_state.final_status = "no_new_ideas"
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
                if preflight.suggested_fixes:
                    self.log(f"Suggested preflight fixes: {preflight.suggested_fixes[:5]}")
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
    # ReAct experiment mode — prompt loaded from YAML
    # ------------------------------------------------------------------

    _REACT_SYSTEM_TEMPLATE = _load_prompt("experiment", "react_system")

    # ------------------------------------------------------------------
    # Code quality helpers (Phase 2c/2d)
    # ------------------------------------------------------------------

    async def _format_generated_code(self, code_dir: Path) -> None:
        """Try to auto-format generated code with black. Silently skips on failure."""
        try:
            subprocess.run(
                [sys.executable, "-m", "black", "--quiet", "--line-length", "100",
                 str(code_dir)],
                capture_output=True, timeout=30,
            )
            self.log("Auto-formatted generated code with black")
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass  # black not available or timed out — skip silently

    def _generate_and_run_smoke_test(
        self, code_dir: Path, file_list: list[str],
    ) -> None:
        """Generate and run a smoke test that imports every generated module.

        Failures are logged as warnings but do not block the pipeline.
        """
        # Collect importable module names (exclude test_ files, subdirs, invalid names)
        modules = []
        for f in file_list:
            if not f.endswith(".py"):
                continue
            p = Path(f)
            if p.name.startswith("test_"):
                continue
            # Only include top-level .py files (no subdirectory nesting)
            if "/" in f or "\\" in f:
                continue
            mod_name = p.stem
            # Validate it's a legal Python identifier
            if not mod_name.isidentifier():
                continue
            modules.append(mod_name)
        if not modules:
            return

        # Generate test_smoke.py
        import_lines = []
        for mod in modules:
            import_lines.append(
                f"    try:\n"
                f"        import {mod}\n"
                f'        print(f"OK: {mod}")\n'
                f"    except Exception as e:\n"
                f'        print(f"FAIL: {mod}: {{e}}")\n'
                f"        failures.append('{mod}')"
            )
        smoke_code = (
            "#!/usr/bin/env python3\n"
            '"""Auto-generated smoke test: verify all modules are importable."""\n'
            "import sys, os\n"
            f"sys.path.insert(0, {str(code_dir)!r})\n"
            "os.chdir(sys.path[0])\n\n"
            "failures = []\n"
            + "\n".join(import_lines)
            + "\n\nif failures:\n"
            '    print(f"SMOKE TEST: {len(failures)} modules failed to import")\n'
            "    sys.exit(1)\n"
            "else:\n"
            '    print("SMOKE TEST: all modules imported OK")\n'
        )

        smoke_path = code_dir / "test_smoke.py"
        smoke_path.write_text(smoke_code, encoding="utf-8")

        # Run smoke test
        try:
            result = subprocess.run(
                [sys.executable, str(smoke_path)],
                capture_output=True, text=True, timeout=30,
                cwd=str(code_dir),
            )
            if result.returncode == 0:
                self.log("Smoke test passed: all modules importable")
            else:
                stdout = (result.stdout or "").strip()
                stderr = (result.stderr or "").strip()
                self.log(f"Smoke test WARNING: {stdout[-200:]}")
                if stderr:
                    logger.warning("Smoke test stderr: %s", stderr[:300])
        except subprocess.TimeoutExpired:
            self.log("Smoke test WARNING: timed out (30s)")
        except (OSError, FileNotFoundError) as e:
            logger.warning("Smoke test failed to run: %s", e)
