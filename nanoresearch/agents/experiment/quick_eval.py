"""Quick evaluation: run --quick-eval, collect metrics, normalize format."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

from . import (
    _decode_bytes,
    _all_metrics_finite,
    _has_metric_name_hint,
    _metric_entries_from_mapping,
    _training_entry_finite,
    SUBPROCESS_OUTPUT_LIMIT,
)
from nanoresearch.agents.experiment._quick_eval_helpers import _QuickEvalHelpersMixin

logger = logging.getLogger(__name__)


class _QuickEvalMixin(_QuickEvalHelpersMixin):
    """Mixin — quick-eval execution and metrics parsing."""

    async def _run_quick_eval(
        self, code_dir: Path, venv_python: str, timeout: int | None = None,
    ) -> dict:
        """Execute main.py --quick-eval with up to 5 batch-fix cycles."""
        if timeout is None:
            timeout = self.config.quick_eval_timeout

        self.log("Phase 4: Running quick-eval for real experiment results")

        max_fix_cycles = 5
        last_result: dict = {}
        fix_history: list[dict] = []

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

            if result["returncode"] == -1 and "Timeout" in result.get("stderr", ""):
                try:
                    modified = await self._fix_timeout(code_dir)
                    if not modified:
                        self.log("Quick-eval: timeout fix did not modify any files, stopping")
                        break
                except Exception as e:
                    self.log(f"Quick-eval timeout fix error: {e}")
                    break
                continue

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
        """Run the legacy experiment entrypoint in quick-eval mode."""
        loop = asyncio.get_running_loop()
        metrics_path = code_dir / "results" / "metrics.json"
        mtime_before = metrics_path.stat().st_mtime if metrics_path.exists() else None
        command = self._build_legacy_subprocess_command(
            code_dir, venv_python, mode="quick-eval",
        )
        if command is None:
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": "No runnable entry script found (expected one of main.py/train.py/run.py)",
            }
        try:
            proc_result = await loop.run_in_executor(
                None,
                lambda: self._run_quick_eval_sync(command, code_dir, timeout),
            )
            return {
                "returncode": proc_result["returncode"],
                "stdout": proc_result["stdout"],
                "stderr": proc_result["stderr"],
            }
        except subprocess.TimeoutExpired:
            self.log(f"Quick-eval timed out after {timeout}s")
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

    @staticmethod
    def _run_quick_eval_sync(
        command: list[str], code_dir: Path, timeout: int,
    ) -> dict[str, Any]:
        """Run quick-eval subprocess with process-tree kill on timeout."""
        from nanoresearch.agents.execution.cluster_runner import _kill_process_tree
        proc = subprocess.Popen(
            command, cwd=str(code_dir),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_process_tree(proc.pid)
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            try:
                proc.communicate(timeout=5)
            except (subprocess.TimeoutExpired, OSError):
                pass
            raise
        return {
            "returncode": proc.returncode or 0,
            "stdout": _decode_bytes(stdout, SUBPROCESS_OUTPUT_LIMIT),
            "stderr": _decode_bytes(stderr, SUBPROCESS_OUTPUT_LIMIT),
        }

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
        """Read and validate results/metrics.json from the code directory."""
        from nanoresearch.agents.experiment._quick_eval_helpers import _QuickEvalHelpersMixin

        metrics_path = code_dir / "results" / "metrics.json"
        if not metrics_path.exists():
            return {}
        try:
            raw = metrics_path.read_text(encoding="utf-8")
            data = json.loads(raw)
            if isinstance(data, list):
                if all(isinstance(entry, dict) for entry in data):
                    data = {"training_log": data}
                else:
                    logger.warning("metrics.json is a non-dict list, skipping")
                    return {}
            if not isinstance(data, dict):
                logger.warning("metrics.json is not a dict, skipping")
                return {}

            data = _QuickEvalHelpersMixin._normalize_metrics_format(data)

            expected_keys = {"main_results", "ablation_results", "training_log"}
            if not expected_keys & set(data.keys()):
                logger.warning("metrics.json has no expected keys (%s), skipping",
                               list(data.keys())[:5])
                return {}

            main_results = data.get("main_results")
            if main_results is not None:
                if not isinstance(main_results, list):
                    logger.warning("main_results is not a list, dropping it")
                    data.pop("main_results")
                else:
                    data["main_results"] = [
                        entry for entry in main_results
                        if isinstance(entry, dict)
                        and _all_metrics_finite(entry.get("metrics", []))
                    ]

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

            if not any(data.get(k) for k in expected_keys):
                return {}

            # Degenerate-run detection
            tlog = data.get("training_log", [])
            if len(tlog) >= 3:
                _numeric_vals: list[float] = []
                for entry in tlog:
                    for k, v in entry.items():
                        if k in ("epoch", "step", "lr"):
                            continue
                        if isinstance(v, (int, float)) and v != float("inf"):
                            _numeric_vals.append(abs(v))
                if _numeric_vals and all(v == 0.0 for v in _numeric_vals):
                    logger.warning(
                        "DEGENERATE RUN DETECTED: all %d numeric metric "
                        "values across %d training log entries are exactly 0.0",
                        len(_numeric_vals), len(tlog),
                    )
                    data["_degenerate_run"] = True
                    data["_degenerate_reason"] = (
                        "All training metrics are exactly 0.0 across all "
                        f"{len(tlog)} epochs. Probable cause: silent batch "
                        "errors (key-name mismatch between dataset and model, "
                        "or shape errors caught by broad except clauses)."
                    )

            return data
        except (json.JSONDecodeError, OSError, TypeError, AttributeError) as exc:
            logger.warning("Failed to parse metrics.json: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # Cluster execution
    # ------------------------------------------------------------------
