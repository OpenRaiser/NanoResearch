"""Result collection: metrics parsing, contract evaluation, and artifact gathering."""
from __future__ import annotations

import csv
import json
import logging
import re
from pathlib import Path
from typing import Any

from nanoresearch.agents.experiment import ExperimentAgent
from nanoresearch.agents.repair_journal import (
    capture_repair_snapshot,
    rollback_snapshot,
)

logger = logging.getLogger(__name__)

RESULT_CONTRACT_CRASH_INDICATORS = (
    "RuntimeError",
    "Error(s) in loading",
    "Traceback",
    "CUDA out of memory",
    "OOM",
    "Killed",
    "Exception",
    "FileNotFoundError",
    "ModuleNotFoundError",
)


class _ResultCollectorMixin:

    def _augment_quick_eval_metrics_from_logs(
        self,
        code_dir: Path,
        quick_eval: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        if quick_eval.get("metrics"):
            return quick_eval
        artifact_results = self._collect_result_artifacts(code_dir)
        artifact_metrics = artifact_results.get("metrics")
        if isinstance(artifact_metrics, dict) and any(
            key in artifact_metrics for key in ("main_results", "ablation_results", "training_log")
        ):
            augmented = {
                **quick_eval,
                "metrics": artifact_metrics,
            }
            recovered_from = str(artifact_results.get("recovered_from") or "").strip()
            if recovered_from:
                augmented["recovered_from"] = recovered_from
            if artifact_results.get("metrics_artifact_materialized"):
                augmented["metrics_artifact_materialized"] = True
                augmented["metrics_artifact_path"] = artifact_results.get("metrics_artifact_path", "")
            return augmented
        recovered = self._recover_metrics_contract_from_logs(result)
        if not recovered:
            return quick_eval
        return {
            **quick_eval,
            "metrics": recovered,
            "recovered_from": "execution_log",
        }

    def _recover_metrics_contract_from_logs(self, result: dict[str, Any]) -> dict[str, Any]:
        log_text = "\n".join(
            part for part in [
                str(result.get("stdout") or "").strip(),
                str(result.get("stderr") or "").strip(),
            ]
            if part
        )
        parsed = self._parse_metrics_from_log(log_text)
        return self._wrap_log_metrics_for_contract(parsed)

    @staticmethod
    def _wrap_log_metrics_for_contract(parsed_metrics: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(parsed_metrics, dict) or not parsed_metrics:
            return {}

        metric_entries: list[dict[str, Any]] = []
        training_log: list[dict[str, Any]] = []
        epoch_losses = parsed_metrics.get("epoch_losses")
        if isinstance(epoch_losses, list):
            for entry in epoch_losses:
                if not isinstance(entry, dict):
                    continue
                epoch = entry.get("epoch")
                loss = entry.get("loss")
                if epoch is None or loss is None:
                    continue
                try:
                    training_log.append({
                        "epoch": int(epoch),
                        "train_loss": float(loss),
                        "metrics": {},
                    })
                except (TypeError, ValueError):
                    continue

        for key, value in parsed_metrics.items():
            if key == "epoch_losses":
                continue
            if isinstance(key, str) and key.isdigit():
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            metric_entries.append({"metric_name": str(key), "value": numeric_value})

        if not metric_entries:
            return {}

        return {
            "main_results": [
                {
                    "method_name": "QuickEvalLog",
                    "dataset": "UNKNOWN",
                    "is_proposed": True,
                    "metrics": metric_entries,
                }
            ],
            "ablation_results": [],
            "training_log": training_log,
        }

    def _materialize_recovered_metrics_artifact(
        self,
        code_dir: Path,
        recovered_metrics: dict[str, Any],
        *,
        source: str,
        scope: str = "",
    ) -> dict[str, Any]:
        self._remember_mutation_snapshot_entry(None)
        artifact_path = "results/metrics.json"
        if not isinstance(recovered_metrics, dict) or not recovered_metrics:
            self._record_snapshot_batch(
                mutation_kind="metrics_artifact_recovery",
                scope=scope or "metrics_artifact_recovery",
                snapshots=[],
                metadata={"modified_files": [], "source": source, "reason": "no_metrics"},
            )
            return {"written": False, "artifact_path": artifact_path, "metrics": {}}

        existing_metrics = ExperimentAgent._parse_metrics_json(code_dir)
        if existing_metrics:
            self._record_snapshot_batch(
                mutation_kind="metrics_artifact_recovery",
                scope=scope or "metrics_artifact_recovery",
                snapshots=[],
                metadata={
                    "modified_files": [],
                    "source": source,
                    "reason": "artifact_already_valid",
                },
            )
            return {"written": False, "artifact_path": artifact_path, "metrics": existing_metrics}

        metrics_path = code_dir / artifact_path
        try:
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            self._record_snapshot_batch(
                mutation_kind="metrics_artifact_recovery",
                scope=scope or "metrics_artifact_recovery",
                snapshots=[],
                metadata={
                    "modified_files": [],
                    "source": source,
                    "reason": "mkdir_failed",
                },
            )
            return {"written": False, "artifact_path": artifact_path, "metrics": recovered_metrics}

        snapshot = capture_repair_snapshot(
            self.workspace.path,
            metrics_path,
            namespace="metrics_artifact_recovery",
            root_dir=self.workspace.path,
            operation="rewrite" if metrics_path.exists() else "create",
        )
        existing_meta = (
            dict(recovered_metrics.get("_nanoresearch_meta"))
            if isinstance(recovered_metrics.get("_nanoresearch_meta"), dict)
            else {}
        )
        payload = {
            "main_results": list(recovered_metrics.get("main_results") or []),
            "ablation_results": list(recovered_metrics.get("ablation_results") or []),
            "training_log": list(recovered_metrics.get("training_log") or []),
            "_nanoresearch_meta": {
                **existing_meta,
                "recovered_from": source,
                "materialized_by": self.__class__.__name__,
            },
        }
        try:
            metrics_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            rollback_snapshot(self.workspace.path, metrics_path, snapshot)
            snapshot["rolled_back"] = True
            snapshot["rollback_reason"] = "write_error"
            self._record_snapshot_batch(
                mutation_kind="metrics_artifact_recovery",
                scope=scope or "metrics_artifact_recovery",
                snapshots=[snapshot],
                metadata={"modified_files": [], "source": source, "reason": "write_error"},
            )
            return {"written": False, "artifact_path": artifact_path, "metrics": recovered_metrics}

        validated_metrics = ExperimentAgent._parse_metrics_json(code_dir)
        if not validated_metrics:
            rollback_snapshot(self.workspace.path, metrics_path, snapshot)
            snapshot["rolled_back"] = True
            snapshot["rollback_reason"] = "validation_failed"
            self._record_snapshot_batch(
                mutation_kind="metrics_artifact_recovery",
                scope=scope or "metrics_artifact_recovery",
                snapshots=[snapshot],
                metadata={
                    "modified_files": [],
                    "source": source,
                    "reason": "validation_failed",
                },
            )
            return {"written": False, "artifact_path": artifact_path, "metrics": recovered_metrics}

        self._record_snapshot_batch(
            mutation_kind="metrics_artifact_recovery",
            scope=scope or "metrics_artifact_recovery",
            snapshots=[snapshot],
            metadata={
                "modified_files": [artifact_path],
                "source": source,
                "training_log_entries": len(validated_metrics.get("training_log") or []),
            },
        )
        return {"written": True, "artifact_path": artifact_path, "metrics": validated_metrics}

    @staticmethod
    def _csv_column_candidates(*names: str) -> tuple[str, ...]:
        candidates: list[str] = []
        for name in names:
            base = str(name or "").strip()
            if not base:
                continue
            lowered = base.lower()
            normalized = lowered.replace(" ", "_").replace("-", "_").replace("/", "_")
            candidates.extend([base, lowered, normalized])
        return tuple(dict.fromkeys(candidates))

    @classmethod
    def _row_numeric_value(
        cls,
        row: dict[str, Any],
        candidates: tuple[str, ...],
    ) -> float | None:
        values: dict[str, Any] = {}
        for key, value in row.items():
            key_text = str(key or "").strip()
            if not key_text:
                continue
            values[key_text] = value
            values[key_text.lower()] = value
            values[key_text.lower().replace(" ", "_").replace("-", "_").replace("/", "_")] = value
        for candidate in candidates:
            raw_value = values.get(candidate)
            if raw_value is None:
                continue
            text = str(raw_value).strip()
            if not text:
                continue
            try:
                return float(text)
            except (TypeError, ValueError):
                continue
        return None

    @classmethod
    def _parse_training_log_csv(cls, csv_path: Path) -> list[dict[str, Any]]:
        if not csv_path.is_file():
            return []

        excluded_metric_tokens = (
            "epoch",
            "step",
            "iter",
            "iteration",
            "batch",
            "loss",
            "lr",
            "learning_rate",
            "time",
            "second",
            "throughput",
            "speed",
            "memory",
            "seed",
            "sample",
            "grad",
        )
        training_log: list[dict[str, Any]] = []
        try:
            with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    return []
                for index, row in enumerate(reader, start=1):
                    if not isinstance(row, dict):
                        continue
                    epoch_value = cls._row_numeric_value(
                        row,
                        cls._csv_column_candidates("epoch", "step", "global_step", "iteration", "iter"),
                    )
                    train_loss = cls._row_numeric_value(
                        row,
                        cls._csv_column_candidates(
                            "train_loss",
                            "loss",
                            "training_loss",
                            "train/loss",
                            "train-loss",
                        ),
                    )
                    val_loss = cls._row_numeric_value(
                        row,
                        cls._csv_column_candidates(
                            "val_loss",
                            "validation_loss",
                            "valid_loss",
                            "dev_loss",
                            "eval_loss",
                            "val/loss",
                            "validation/loss",
                        ),
                    )
                    metrics: dict[str, float] = {}
                    for key, raw_value in row.items():
                        key_text = str(key or "").strip()
                        if not key_text:
                            continue
                        value_text = str(raw_value or "").strip()
                        if not value_text:
                            continue
                        normalized = key_text.lower().replace(" ", "_").replace("-", "_").replace("/", "_")
                        if any(token in normalized for token in excluded_metric_tokens):
                            continue
                        try:
                            metrics[key_text] = float(value_text)
                        except (TypeError, ValueError):
                            continue

                    entry: dict[str, Any] = {
                        "epoch": int(epoch_value) if epoch_value is not None else index,
                        "metrics": metrics,
                    }
                    if train_loss is not None:
                        entry["train_loss"] = train_loss
                    if val_loss is not None:
                        entry["val_loss"] = val_loss
                    if metrics or train_loss is not None or val_loss is not None:
                        training_log.append(entry)
        except (OSError, csv.Error):
            return []
        return training_log

    async def _collect_results(
        self, code_dir: Path, job_id: str, status: str
    ) -> dict:
        """Collect training results from output files."""
        results: dict[str, Any] = {
            **self._collect_result_artifacts(code_dir),
            "stdout_log": "",
            "stderr_log": "",
        }

        def _read_log(patterns: list[str], limit: int) -> str:
            candidates: list[Path] = []
            for pattern in patterns:
                candidates.extend(sorted((code_dir / "logs").glob(pattern)))

            for log_file in candidates:
                if job_id in log_file.name:
                    return log_file.read_text(errors="replace")[-limit:]

            for log_file in candidates:
                return log_file.read_text(errors="replace")[-limit:]
            return ""

        results["stdout_log"] = _read_log(
            ["slurm_*.out", f"{job_id}.log", "*.out", "*.log"],
            10000,
        )
        results["stderr_log"] = _read_log(
            ["slurm_*.err", f"{job_id}.err", "*.err"],
            5000,
        )

        if not results["metrics"]:
            log_text = "\n".join(
                part for part in [results["stdout_log"], results["stderr_log"]] if part
            )
            if log_text:
                parsed_metrics = self._parse_metrics_from_log(log_text)
                if parsed_metrics:
                    results["parsed_metrics"] = parsed_metrics
                    recovered_metrics = self._wrap_log_metrics_for_contract(parsed_metrics)
                    if recovered_metrics:
                        materialized = self._materialize_recovered_metrics_artifact(
                            code_dir,
                            recovered_metrics,
                            source="slurm_logs",
                            scope="cluster_collect",
                        )
                        results["metrics"] = materialized.get("metrics") or recovered_metrics
                        results["recovered_from"] = "slurm_logs"
                        results["training_log"] = list(results["metrics"].get("training_log") or [])
                        if materialized.get("written"):
                            results["metrics_artifact_materialized"] = True
                            results["metrics_artifact_path"] = materialized.get("artifact_path", "")

        return results

    def _collect_result_artifacts(self, code_dir: Path) -> dict[str, Any]:
        results: dict[str, Any] = {
            "metrics": {},
            "training_log": [],
        }

        metrics_path = code_dir / "results" / "metrics.json"
        if metrics_path.exists():
            parsed_metrics = ExperimentAgent._parse_metrics_json(code_dir)
            if parsed_metrics:
                results["metrics"] = parsed_metrics
                results["training_log"] = list(parsed_metrics.get("training_log") or [])
                meta = parsed_metrics.get("_nanoresearch_meta")
                if isinstance(meta, dict) and str(meta.get("recovered_from") or "").strip():
                    results["recovered_from"] = str(meta.get("recovered_from") or "").strip()
            else:
                try:
                    results["metrics"] = {"raw": metrics_path.read_text()[:5000]}
                except OSError:
                    results["metrics"] = {}

        log_csv = code_dir / "results" / "training_log.csv"
        if log_csv.exists():
            try:
                with log_csv.open("r", encoding="utf-8", errors="replace") as handle:
                    results["training_log_csv"] = handle.read(10000)
            except OSError:
                results["training_log_csv"] = ""
            parsed_training_log = self._parse_training_log_csv(log_csv)
            if parsed_training_log and not results["training_log"]:
                results["training_log"] = parsed_training_log
            if parsed_training_log and not (
                isinstance(results.get("metrics"), dict)
                and any(
                    key in results["metrics"] for key in ("main_results", "ablation_results", "training_log")
                )
            ):
                materialized = self._materialize_recovered_metrics_artifact(
                    code_dir,
                    {
                        "main_results": [],
                        "ablation_results": [],
                        "training_log": parsed_training_log,
                    },
                    source="training_log_csv",
                    scope="result_artifacts",
                )
                results["metrics"] = materialized.get("metrics") or results.get("metrics", {})
                if results["metrics"]:
                    results["recovered_from"] = "training_log_csv"
                    results["training_log"] = list(results["metrics"].get("training_log") or parsed_training_log)
                if materialized.get("written"):
                    results["metrics_artifact_materialized"] = True
                    results["metrics_artifact_path"] = materialized.get("artifact_path", "")

        for results_file in (code_dir / "results").glob("*"):
            if results_file.is_file() and results_file.name not in ("metrics.json", "training_log.csv"):
                try:
                    content = results_file.read_text(errors="replace")[:5000]
                    results[f"result_file_{results_file.name}"] = content
                except Exception:
                    pass

        checkpoints = (
            list((code_dir / "checkpoints").glob("*.pt"))
            if (code_dir / "checkpoints").exists()
            else []
        )
        results["checkpoints"] = [str(p) for p in checkpoints]
        return results

    def _collect_local_results(
        self,
        code_dir: Path,
        run_result: dict[str, Any],
    ) -> dict[str, Any]:
        results: dict[str, Any] = {
            **self._collect_result_artifacts(code_dir),
            "stdout_log": str(run_result.get("stdout", ""))[-10000:],
            "stderr_log": str(run_result.get("stderr", ""))[-5000:],
        }
        if not results["metrics"] and results["stdout_log"]:
            results["parsed_metrics"] = self._parse_metrics_from_log(results["stdout_log"])
        return results

    @staticmethod
    def _metrics_satisfy_contract(metrics: dict[str, Any] | None) -> bool:
        if not isinstance(metrics, dict):
            return False
        main_results = metrics.get("main_results")
        if not isinstance(main_results, list):
            return False
        for item in main_results:
            if not isinstance(item, dict):
                continue
            metric_entries = item.get("metrics")
            if not isinstance(metric_entries, list):
                continue
            for metric in metric_entries:
                if not isinstance(metric, dict):
                    continue
                if str(metric.get("metric_name", "")).strip() and metric.get("value") is not None:
                    return True
        return False

    @staticmethod
    def _result_file_names(results: dict[str, Any]) -> list[str]:
        return sorted(
            key.removeprefix("result_file_")
            for key, value in results.items()
            if key.startswith("result_file_") and str(value or "").strip()
        )

    @classmethod
    def _detect_contract_failure_signals(
        cls,
        stdout_log: str,
        stderr_log: str,
    ) -> list[str]:
        combined = f"{stdout_log}\n{stderr_log}".lower()
        found: list[str] = []
        for indicator in RESULT_CONTRACT_CRASH_INDICATORS:
            if indicator.lower() in combined and indicator not in found:
                found.append(indicator)
        return found

    @classmethod
    def _evaluate_experiment_contract(
        cls,
        result_payload: dict[str, Any],
        *,
        execution_backend: str,
        execution_status: str,
        quick_eval_status: str,
        final_status: str,
    ) -> dict[str, Any]:
        metrics = result_payload.get("metrics") if isinstance(result_payload.get("metrics"), dict) else {}
        parsed_metrics = result_payload.get("parsed_metrics") if isinstance(result_payload.get("parsed_metrics"), dict) else {}
        training_log = result_payload.get("training_log") if isinstance(result_payload.get("training_log"), list) else []
        training_log_csv = str(result_payload.get("training_log_csv") or "").strip()
        checkpoints = result_payload.get("checkpoints") if isinstance(result_payload.get("checkpoints"), list) else []
        result_files = cls._result_file_names(result_payload)
        recovered_from = str(result_payload.get("recovered_from") or "").strip()
        stdout_log = str(result_payload.get("stdout_log") or "")
        stderr_log = str(result_payload.get("stderr_log") or "")

        has_structured_metrics = cls._metrics_satisfy_contract(metrics)
        has_parsed_metrics = bool(parsed_metrics)
        has_training_trace = bool(training_log) or bool(training_log_csv)
        has_checkpoints = bool(checkpoints)
        has_result_files = bool(result_files)
        failure_signals = cls._detect_contract_failure_signals(stdout_log, stderr_log)
        run_completed = final_status == "COMPLETED" or quick_eval_status in {"success", "partial"}
        has_recovered_artifact_support = (
            has_structured_metrics
            and bool(recovered_from)
            and run_completed
            and not failure_signals
            and (has_checkpoints or has_result_files)
            and (has_parsed_metrics or has_training_trace)
        )

        satisfied_signals: list[str] = []
        if has_structured_metrics and not recovered_from:
            satisfied_signals.append("structured_metrics_artifact")
        elif has_structured_metrics and recovered_from:
            satisfied_signals.append("structured_metrics_recovered")
        if has_parsed_metrics:
            satisfied_signals.append("parsed_metrics")
        if has_training_trace:
            satisfied_signals.append("training_log")
        if has_checkpoints:
            satisfied_signals.append("checkpoints")
        if has_result_files:
            satisfied_signals.append("result_files")

        success_path = ""
        status = "failed"
        if has_structured_metrics and not recovered_from and run_completed:
            status = "success"
            success_path = "structured_metrics_artifact"
        elif has_recovered_artifact_support:
            status = "success"
            success_path = "structured_metrics_recovered"
        elif has_structured_metrics and recovered_from and run_completed:
            status = "partial"
            success_path = "structured_metrics_recovered"
        elif has_parsed_metrics and (has_training_trace or has_checkpoints or has_result_files) and run_completed:
            status = "partial"
            success_path = "parsed_metrics_with_artifacts"
        elif has_training_trace and has_checkpoints and final_status == "COMPLETED":
            status = "partial"
            success_path = "training_log_with_checkpoints"
        elif has_result_files and (has_training_trace or has_checkpoints) and final_status == "COMPLETED":
            status = "partial"
            success_path = "aux_results_with_artifacts"
        elif execution_backend == "cluster" and has_parsed_metrics and final_status == "COMPLETED":
            status = "partial"
            success_path = "parsed_metrics_only"

        missing_signals: list[str] = []
        if status == "failed":
            if not (has_structured_metrics or has_parsed_metrics):
                missing_signals.append("metrics_signal")
            if not (has_training_trace or has_checkpoints or has_result_files or has_structured_metrics):
                missing_signals.append("artifact_signal")
            if failure_signals:
                missing_signals.append("crash_free_logs")

        return {
            "version": "v1",
            "status": status,
            "execution_backend": execution_backend,
            "execution_status": execution_status,
            "quick_eval_status": quick_eval_status,
            "final_status": final_status,
            "recovered_from": recovered_from,
            "success_path": success_path,
            "satisfied_signals": satisfied_signals,
            "missing_signals": missing_signals,
            "failure_signals": failure_signals,
            "artifact_inventory": {
                "structured_metrics": has_structured_metrics,
                "parsed_metrics": has_parsed_metrics,
                "training_log_entries": len(training_log),
                "training_log_csv": bool(training_log_csv),
                "checkpoint_count": len(checkpoints),
                "result_files": result_files,
            },
        }

    def _parse_metrics_from_log(self, log_text: str) -> dict:
        """Try to extract metrics from training log output."""
        metrics: dict[str, Any] = {}
        lines = log_text.split("\n")

        # Common patterns in training logs
        patterns = [
            # "Epoch 10: loss=0.123, accuracy=0.95"
            r"[Ee]poch\s+(\d+).*?loss[=:\s]+([0-9.e-]+)",
            # "Test accuracy: 0.95"
            r"[Tt]est\s+(accuracy|acc)[=:\s]+([0-9.e-]+)",
            # "Best metric: 0.95"
            r"[Bb]est\s+(\w+)[=:\s]+([0-9.e-]+)",
            # "AUC: 0.95" / "F1: 0.85"
            r"(AUC|F1|RMSE|MAE|accuracy|precision|recall)[=:\s]+([0-9.e-]+)",
        ]

        epochs = []
        for line in lines:
            for pattern in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    groups = match.groups()
                    if len(groups) == 2:
                        metrics[groups[0]] = groups[1]

            # Track epoch losses
            epoch_match = re.search(
                r"[Ee]poch\s+(\d+).*?loss[=:\s]+([0-9.e-]+)", line
            )
            if epoch_match:
                epochs.append({
                    "epoch": int(epoch_match.group(1)),
                    "loss": float(epoch_match.group(2)),
                })

        if epochs:
            metrics["epoch_losses"] = epochs
            metrics["final_loss"] = epochs[-1]["loss"]

        return metrics
