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

logger = logging.getLogger(__name__)


class _QuickEvalMixin:
    """Mixin — quick-eval execution and metrics parsing."""

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
        """Run the legacy experiment entrypoint in quick-eval mode."""
        loop = asyncio.get_running_loop()
        # Record metrics.json mtime before run to detect stale files on timeout
        metrics_path = code_dir / "results" / "metrics.json"
        mtime_before = metrics_path.stat().st_mtime if metrics_path.exists() else None
        command = self._build_legacy_subprocess_command(
            code_dir,
            venv_python,
            mode="quick-eval",
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
                lambda: subprocess.run(
                    command,
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
            normalized = dict(data)

            if not normalized.get("main_results"):
                summary_candidates = [
                    normalized.get("results"),
                    normalized.get("metrics"),
                    normalized.get("summary"),
                    normalized.get("final_metrics"),
                    normalized.get("best_metrics"),
                    normalized.get("aggregate"),
                ]
                for candidate in summary_candidates:
                    if isinstance(candidate, dict):
                        metric_list = _metric_entries_from_mapping(
                            candidate,
                            num_runs=normalized.get("num_runs") if isinstance(normalized.get("num_runs"), int) else None,
                        )
                        if metric_list:
                            normalized["main_results"] = [
                                {
                                    "method_name": str(
                                        normalized.get("method_name")
                                        or normalized.get("model_name")
                                        or normalized.get("name")
                                        or "Ours"
                                    ),
                                    "dataset": str(normalized.get("dataset") or "UNKNOWN"),
                                    "is_proposed": bool(normalized.get("is_proposed", True)),
                                    "metrics": metric_list,
                                }
                            ]
                            break

            if not normalized.get("main_results"):
                training_log = normalized.get("training_log")
                if isinstance(training_log, list):
                    for entry in reversed(training_log):
                        if not isinstance(entry, dict):
                            continue
                        metrics = entry.get("metrics")
                        if isinstance(metrics, dict):
                            metric_list = _metric_entries_from_mapping(metrics)
                            if metric_list:
                                normalized["main_results"] = [
                                    {
                                        "method_name": str(
                                            normalized.get("method_name")
                                            or normalized.get("model_name")
                                            or normalized.get("name")
                                            or "Ours"
                                        ),
                                        "dataset": str(normalized.get("dataset") or "UNKNOWN"),
                                        "is_proposed": bool(normalized.get("is_proposed", True)),
                                        "metrics": metric_list,
                                    }
                                ]
                                break

            if not normalized.get("ablation_results"):
                ablation_candidates = (
                    normalized.get("ablations"),
                    normalized.get("ablation"),
                    normalized.get("ablation_study"),
                )
                for candidate in ablation_candidates:
                    if isinstance(candidate, list):
                        ablation_results = []
                        for item in candidate:
                            if not isinstance(item, dict):
                                continue
                            metric_source = item.get("metrics") if isinstance(item.get("metrics"), dict) else item
                            if not isinstance(metric_source, dict):
                                continue
                            metric_list = _metric_entries_from_mapping(metric_source)
                            if metric_list:
                                ablation_results.append(
                                    {
                                        "variant_name": str(
                                            item.get("variant_name")
                                            or item.get("name")
                                            or item.get("method_name")
                                            or f"variant_{len(ablation_results) + 1}"
                                        ),
                                        "metrics": metric_list,
                                    }
                                )
                        if ablation_results:
                            normalized["ablation_results"] = ablation_results
                            break
                    elif isinstance(candidate, dict):
                        ablation_results = []
                        for variant_name, metric_source in candidate.items():
                            if not isinstance(metric_source, dict):
                                continue
                            metric_list = _metric_entries_from_mapping(metric_source)
                            if metric_list:
                                ablation_results.append(
                                    {
                                        "variant_name": str(variant_name),
                                        "metrics": metric_list,
                                    }
                                )
                        if ablation_results:
                            normalized["ablation_results"] = ablation_results
                            break

            return normalized

        variants = data.get("variants")
        # Handle array-format variants: [{"name": "full", ...}, ...]
        if isinstance(variants, list):
            converted: dict = {}
            for item in variants:
                if isinstance(item, dict):
                    name = item.pop("name", item.pop("variant_name", f"variant_{len(converted)}"))
                    converted[str(name)] = item
            variants = converted if converted else None
            logger.debug("Converted list-format variants to dict (%d entries)", len(converted))
        if not isinstance(variants, dict) or not variants:
            summary_candidates = [
                data.get("results"),
                data.get("metrics"),
                data.get("summary"),
                data.get("final_metrics"),
                data.get("best_metrics"),
                data.get("aggregate"),
            ]
            for candidate in summary_candidates:
                if isinstance(candidate, dict):
                    metric_list = _metric_entries_from_mapping(
                        candidate,
                        num_runs=data.get("num_runs") if isinstance(data.get("num_runs"), int) else None,
                    )
                    if metric_list:
                        return {
                            "main_results": [
                                {
                                    "method_name": str(
                                        data.get("method_name")
                                        or data.get("model_name")
                                        or data.get("name")
                                        or "Ours"
                                    ),
                                    "dataset": str(data.get("dataset") or "UNKNOWN"),
                                    "is_proposed": bool(data.get("is_proposed", True)),
                                    "metrics": metric_list,
                                }
                            ],
                            "ablation_results": [],
                            "training_log": data.get("training_log", []) if isinstance(data.get("training_log"), list) else [],
                        }

            top_level_metric_list = _metric_entries_from_mapping(
                data,
                num_runs=data.get("num_runs") if isinstance(data.get("num_runs"), int) else None,
            )
            # For the raw top-level fallback (most aggressive path), require
            # at least one extracted key to match a known metric name pattern
            # to avoid treating arbitrary dicts as metrics files.
            if top_level_metric_list and _has_metric_name_hint(top_level_metric_list):
                return {
                    "main_results": [
                        {
                            "method_name": str(
                                data.get("method_name")
                                or data.get("model_name")
                                or data.get("name")
                                or "Ours"
                            ),
                            "dataset": str(data.get("dataset") or "UNKNOWN"),
                            "is_proposed": bool(data.get("is_proposed", True)),
                            "metrics": top_level_metric_list,
                        }
                    ],
                    "ablation_results": [],
                    "training_log": data.get("training_log", []) if isinstance(data.get("training_log"), list) else [],
                }

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
            if isinstance(data, list):
                if all(isinstance(entry, dict) for entry in data):
                    data = {"training_log": data}
                else:
                    logger.warning("metrics.json is a non-dict list, skipping")
                    return {}
            if not isinstance(data, dict):
                logger.warning("metrics.json is not a dict, skipping")
                return {}

            # Try to convert alternative formats to the expected schema
            data = _QuickEvalMixin._normalize_metrics_format(data)

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

