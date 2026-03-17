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
from nanoresearch.agents.execution._result_collector_helpers import (
    _ResultCollectorHelpersMixin,
    RESULT_CONTRACT_CRASH_INDICATORS,
)

logger = logging.getLogger(__name__)


class _ResultCollectorMixin(_ResultCollectorHelpersMixin):

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
            "epoch", "step", "iter", "iteration", "batch", "loss",
            "lr", "learning_rate", "time", "second", "throughput",
            "speed", "memory", "seed", "sample", "grad",
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
                            "train_loss", "loss", "training_loss", "train/loss", "train-loss",
                        ),
                    )
                    val_loss = cls._row_numeric_value(
                        row,
                        cls._csv_column_candidates(
                            "val_loss", "validation_loss", "valid_loss",
                            "dev_loss", "eval_loss", "val/loss", "validation/loss",
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
                # BUG-34 fix: use word-boundary match instead of substring
                # to avoid "123" matching "slurm_1234.out".
                if re.search(rf'(?:^|[_\-])({re.escape(job_id)})(?:$|[_\-.])', log_file.name):
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
