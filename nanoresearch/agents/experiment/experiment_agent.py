"""ExperimentAgent main run method and code quality helpers."""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

from nanoresearch.agents.cluster_executor import ClusterExecutor
from nanoresearch.agents.feedback_analyzer import FeedbackAnalyzer
from nanoresearch.evolution.memory import MemoryType
from nanoresearch.agents.preflight import PreflightChecker
from nanoresearch.schemas.iteration import (
    ExperimentHypothesis,
    IterationState,
    RoundResult,
)

from . import STDERR_SNIPPET_LIMIT

logger = logging.getLogger(__name__)


class _ExperimentAgentMixin:
    """Mixin — ExperimentAgent.run() and code quality helpers."""

    async def run(self, **inputs: Any) -> dict[str, Any]:
        blueprint_data: dict = inputs["experiment_blueprint"]
        reference_repos: list[dict] = inputs.get("reference_repos", [])

        # Dispatch to ReAct mode or pipeline mode
        if self.config.experiment_mode == "react":
            return await self._run_react_mode(blueprint_data, reference_repos)

        max_rounds = self.config.experiment_max_rounds
        self.log(f"Starting iterative experiment (max {max_rounds} rounds)")

        title = blueprint_data.get("title", "")
        adaptive_context = self.build_adaptive_context(
            "experiment",
            topic=title,
            blueprint=blueprint_data,
            text=json.dumps(blueprint_data, ensure_ascii=False)[:5000],
            tags=[title, "experiment", self.workspace.manifest.paper_mode.value],
            include_script_recommendations=True,
        )
        retry_error = str(inputs.get("_retry_error", "")).strip()
        if retry_error:
            self.learn_from_trace(
                "experiment",
                "experiment_retry",
                retry_error,
                tags=[title, "retry", "experiment"],
            )
        method = blueprint_data.get("proposed_method", {})
        datasets = blueprint_data.get("datasets", [])
        metrics = blueprint_data.get("metrics", [])
        baselines = blueprint_data.get("baselines", [])
        ablations = blueprint_data.get("ablation_groups", [])

        blueprint_summary = json.dumps({
            "title": title, "proposed_method": method,
            "datasets": datasets, "metrics": metrics,
            "baselines": baselines, "ablation_groups": ablations,
        }, indent=2, ensure_ascii=False)

        repo_context = self._build_repo_context(reference_repos)
        if adaptive_context:
            repo_context = f"{adaptive_context}\n\n{repo_context}" if repo_context else adaptive_context
        if repo_context:
            self.log(f"Using {len(reference_repos)} reference repos for code grounding")

        analyzer = FeedbackAnalyzer(self.config, self._dispatcher)
        iteration_state = IterationState(max_rounds=max_rounds)
        code_dir = self.workspace.path / "code"
        venv_python: str = ""
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
            self.log(f"Cluster mode ENABLED ({mode_desc}) -- experiments will run on SLURM cluster")
            if not await cluster.check_connectivity():
                self.log("WARNING: Cluster check failed, falling back to local execution")
                cluster_mode = False
                cluster = None

        iteration_state, start_round = self._load_iteration_checkpoint(iteration_state)

        for round_num in range(start_round, max_rounds + 1):
            self.log(f"=== Iteration Round {round_num}/{max_rounds} ===")
            files_modified: list[str] = []

            if round_num == 1:
                hypothesis, project_plan, generated_files, venv_python = (
                    await self._run_round_one(
                        blueprint_summary, repo_context, code_dir,
                    )
                )
            else:
                hypothesis, files_modified, generated_files = (
                    await self._run_iteration_round(
                        round_num, iteration_state, blueprint_summary,
                        code_dir, venv_python, generated_files,
                    )
                )

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
                    round_number=round_num, hypothesis=hypothesis,
                    preflight=preflight, execution_status="skipped",
                    quick_eval_status="skipped", metrics={},
                )
                iteration_state.rounds.append(round_result)
                self.log(f"Preflight failed, will retry in next round ({round_num}/{max_rounds})")
                failure_summary = "; ".join(preflight.blocking_failures)
                fix_summary = "; ".join(preflight.suggested_fixes[:5])
                self.learn_from_trace(
                    "experiment",
                    "preflight_failure",
                    f"Round {round_num} preflight failures: {failure_summary} | suggested fixes: {fix_summary}",
                    tags=[title, "preflight", "experiment"],
                    confidence=0.6,
                )
                continue

            # ---- Phase 3 & 4: execution ----
            execution, quick_eval, venv_python = await self._run_execution_phase(
                cluster_mode, cluster, code_dir, round_num, cluster_code_path,
                generated_files, blueprint_summary, venv_python,
            )
            if execution.get("cluster_code_path"):
                cluster_code_path = execution["cluster_code_path"]
            execution_status = execution.get("status", "failed")

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
                round_number=round_num, hypothesis=hypothesis,
                preflight=preflight, execution_status=execution_status,
                quick_eval_status=quick_eval.get("status", "skipped"),
                metrics=quick_eval.get("metrics", {}), analysis=analysis,
                files_modified=generated_files if round_num == 1 else (
                    files_modified if round_num > 1 else []
                ),
            )
            iteration_state.rounds.append(round_result)

            # Track best round
            if analysis.metric_summary:
                primary_key = next(iter(analysis.metric_summary), None)
                primary_value = analysis.metric_summary.get(primary_key) if primary_key else None
                best_value = iteration_state.best_metrics.get(primary_key) if (iteration_state.best_metrics and primary_key) else None
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

            self.workspace.write_json(
                f"logs/iteration_round_{round_num}.json", round_result.model_dump(),
            )
            self._save_iteration_checkpoint(iteration_state)

            self.log(
                f"Round {round_num} analysis: attribution={analysis.attribution}, "
                f"should_continue={analysis.should_continue}"
            )

            if not analysis.should_continue:
                iteration_state.final_status = analysis.termination_reason or "completed"
                self.log(f"Stopping iteration: {iteration_state.final_status}")
                break
        else:
            iteration_state.final_status = "max_rounds"

        # ---- Build final result ----
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
        metrics_summary = str(best_round_data["metrics"])
        self.remember_context(
            MemoryType.PROJECT_CONTEXT,
            f"Experiment status for {title}: final_status={iteration_state.final_status}, best_round={iteration_state.best_round}, metrics={metrics_summary}",
            importance=0.82,
            tags=[title, "experiment", iteration_state.final_status or "completed"],
            source="experiment_output",
            topic=title,
        )
        quick_eval_status = best_round_data["quick_eval_status"]
        execution_status = best_round_data["execution_status"]
        self.remember_context(
            MemoryType.DECISION_HISTORY,
            f"Experiment loop for {title} ended with execution={execution_status} and quick_eval={quick_eval_status} after {len(iteration_state.rounds)} rounds.",
            importance=0.75,
            tags=[title, "iteration", str(quick_eval_status)],
            source="experiment_output",
            topic=title,
        )
        if len(iteration_state.rounds) > 1 and iteration_state.best_round > 1:
            self.learn_from_trace(
                "experiment",
                "retry_success",
                f"Best result for {title} emerged at round {iteration_state.best_round} after iterative retries. Preserve successful fixes from earlier failed rounds.",
                tags=[title, "iteration", "success"],
                confidence=0.65,
            )
        return result

    # ------------------------------------------------------------------
    # Round execution helpers (extracted from run for readability)
    # ------------------------------------------------------------------

    async def _run_round_one(
        self, blueprint_summary: str, repo_context: str, code_dir: Path,
    ) -> tuple[Any, dict, list[str], str]:
        """Execute round 1: full generation (baseline)."""
        import asyncio
        from nanoresearch.schemas.iteration import ExperimentHypothesis

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

        self.log(f"  Generating {len(valid_specs)} files in parallel")
        contents = await asyncio.gather(*(
            self._generate_file(
                spec, interface_contract, blueprint_summary, repo_context
            )
            for spec in valid_specs
        ), return_exceptions=True)

        for spec, content in zip(valid_specs, contents):
            file_path = spec["path"]
            if isinstance(content, BaseException):
                logger.error("Failed to generate %s: %s", file_path, content)
                continue
            self.workspace.write_text(f"code/{file_path}", content)
            generated_files.append(file_path)

        # Phase 2b-2d: consistency, format, smoke test
        import_mismatches = self._check_import_consistency(code_dir)
        if import_mismatches:
            self.log(f"Found {len(import_mismatches)} import mismatches, fixing...")
            await self._fix_import_mismatches(code_dir, import_mismatches)

        await self._format_generated_code(code_dir)
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

        for fp in generated_files:
            self.workspace.register_artifact(
                f"code_{fp.replace('/', '_')}",
                self.workspace.path / "code" / fp, self.stage,
            )

        verification = self._verify_code(generated_files)
        self.workspace.write_json("logs/code_verification.json", verification)
        self.log(f"Code verification: {verification['passed']}/{verification['total']} files OK")

        return hypothesis, project_plan, generated_files, ""

    async def _run_iteration_round(
        self,
        round_num: int,
        iteration_state: Any,
        blueprint_summary: str,
        code_dir: Path,
        venv_python: str,
        generated_files: list[str],
    ) -> tuple[Any, list[str], list[str]]:
        """Execute round 2+: iterative improvement."""
        prev_round = iteration_state.rounds[-1]
        prev_analysis = prev_round.analysis
        history_summary = self._build_history_summary(iteration_state.rounds)

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

        if hypothesis.hypothesis == "__NO_NEW_IDEAS__":
            self.log("LLM exhausted improvement ideas -- stopping iteration")
            iteration_state.final_status = "no_new_ideas"
            return hypothesis, [], generated_files

        self.log(f"Hypothesis: {hypothesis.hypothesis[:100]}")

        files_modified = await self._apply_iteration_changes(
            hypothesis, code_dir, venv_python
        )
        if not files_modified and hypothesis.planned_changes:
            self.log("Search-replace matched nothing, retrying with full-file rewrite")
            files_modified = await self._apply_iteration_changes_fullwrite(
                hypothesis, code_dir
            )
        generated_files = files_modified or generated_files
        self.log(f"Modified {len(files_modified)} files")
        return hypothesis, files_modified, generated_files

    async def _run_execution_phase(
        self,
        cluster_mode: bool,
        cluster: Any,
        code_dir: Path,
        round_num: int,
        cluster_code_path: str,
        generated_files: list[str],
        blueprint_summary: str,
        venv_python: str,
    ) -> tuple[dict, dict, str]:
        """Run phase 3 & 4: execution and quick-eval. Returns (execution, quick_eval, venv_python)."""
        if cluster_mode and cluster:
            execution, quick_eval = await self._run_on_cluster(
                cluster, code_dir, round_num, cluster_code_path,
            )
            execution_status = execution.get("status", "failed")
            self.log(f"Cluster execution: {execution_status}")
            self.log(f"Cluster quick-eval: {quick_eval.get('status', 'skipped')}")
            return execution, quick_eval, venv_python

        # LOCAL EXECUTION
        if round_num == 1 or not venv_python:
            execution, venv_python = await self._execute_code_with_venv(
                generated_files, blueprint_summary
            )
        else:
            execution = await self._execute_code(
                generated_files, blueprint_summary,
                _code_dir=code_dir, _main_py=code_dir / "main.py",
                _venv_python=venv_python,
            )
        execution_status = execution.get("status", "failed")
        self.log(f"Dry-run: {execution_status}")

        quick_eval = {"status": "skipped", "metrics": {}}
        if execution_status in ("success", "fixed"):
            quick_eval = await self._run_quick_eval(code_dir, venv_python)
            self.log(f"Quick-eval: {quick_eval['status']}")
        else:
            self.log("Skipping quick-eval (dry-run did not succeed)")

        return execution, quick_eval, venv_python

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
            pass

    def _generate_and_run_smoke_test(
        self, code_dir: Path, file_list: list[str],
    ) -> None:
        """Generate and run a smoke test that imports every generated module."""
        modules = []
        for f in file_list:
            if not f.endswith(".py"):
                continue
            p = Path(f)
            if p.name.startswith("test_"):
                continue
            if "/" in f or "\\" in f:
                continue
            mod_name = p.stem
            if not mod_name.isidentifier():
                continue
            modules.append(mod_name)
        if not modules:
            return

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
