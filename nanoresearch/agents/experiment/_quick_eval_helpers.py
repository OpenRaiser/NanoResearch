"""Quick eval helpers: metrics normalization and parsing."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from . import (
    _all_metrics_finite,
    _has_metric_name_hint,
    _metric_entries_from_mapping,
    _training_entry_finite,
)

logger = logging.getLogger(__name__)


class _QuickEvalHelpersMixin:
    """Mixin — metrics normalization and parsing for quick-eval."""

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
        # Handle array-format variants
        if isinstance(variants, list):
            converted: dict = {}
            for item in variants:
                if isinstance(item, dict):
                    item_copy = dict(item)
                    name = item_copy.pop("name", item_copy.pop("variant_name", f"variant_{len(converted)}"))
                    converted[str(name)] = item_copy
            variants = converted if converted else None
            logger.debug("Converted list-format variants to dict (%d entries)", len(converted))
        if not isinstance(variants, dict) or not variants:
            summary_candidates = [
                data.get("results"), data.get("metrics"), data.get("summary"),
                data.get("final_metrics"), data.get("best_metrics"), data.get("aggregate"),
            ]
            for candidate in summary_candidates:
                if isinstance(candidate, dict):
                    metric_list = _metric_entries_from_mapping(
                        candidate,
                        num_runs=data.get("num_runs") if isinstance(data.get("num_runs"), int) else None,
                    )
                    if metric_list:
                        return {
                            "main_results": [{
                                "method_name": str(data.get("method_name") or data.get("model_name") or data.get("name") or "Ours"),
                                "dataset": str(data.get("dataset") or "UNKNOWN"),
                                "is_proposed": bool(data.get("is_proposed", True)),
                                "metrics": metric_list,
                            }],
                            "ablation_results": [],
                            "training_log": data.get("training_log", []) if isinstance(data.get("training_log"), list) else [],
                        }

            top_level_metric_list = _metric_entries_from_mapping(
                data,
                num_runs=data.get("num_runs") if isinstance(data.get("num_runs"), int) else None,
            )
            if top_level_metric_list and _has_metric_name_hint(top_level_metric_list):
                return {
                    "main_results": [{
                        "method_name": str(data.get("method_name") or data.get("model_name") or data.get("name") or "Ours"),
                        "dataset": str(data.get("dataset") or "UNKNOWN"),
                        "is_proposed": bool(data.get("is_proposed", True)),
                        "metrics": top_level_metric_list,
                    }],
                    "ablation_results": [],
                    "training_log": data.get("training_log", []) if isinstance(data.get("training_log"), list) else [],
                }

            # Fallback: top-level keys are variant dicts themselves
            candidate_variants = {}
            for k, v in data.items():
                if isinstance(v, dict) and ("aggregate" in v or "runs" in v or any(
                    isinstance(sv, dict) and "mean" in sv for sv in v.values()
                )):
                    candidate_variants[k] = v
            if not candidate_variants:
                return data
            variants = {}
            for k, v in candidate_variants.items():
                if "aggregate" in v and isinstance(v["aggregate"], dict):
                    variants[k] = v["aggregate"]
                else:
                    variants[k] = v

        # Convert variants dict -> main_results + ablation_results
        main_results = []
        ablation_results = []
        dataset_name = data.get("dataset", "MNIST")

        for variant_name, metrics_dict in variants.items():
            if not isinstance(metrics_dict, dict):
                continue
            metric_list = []
            for mname, mval in metrics_dict.items():
                if any(mname.startswith(p) for p in (
                    "per_class", "confusion_matrix", "qualitative",
                )) or mname in (
                    "run_seed", "num_runs", "num_samples", "variant",
                    "training_time_sec", "parameter_count", "FLOPs_M",
                    "best_val_accuracy", "inference_time_ms",
                ):
                    continue
                if isinstance(mval, dict) and "mean" in mval:
                    metric_list.append({
                        "metric_name": mname,
                        "value": mval["mean"],
                        "std": mval.get("std", 0.0),
                        "num_runs": data.get("num_runs", 1),
                    })
                elif isinstance(mval, (int, float)):
                    metric_list.append({"metric_name": mname, "value": mval})

            if not metric_list:
                continue
            _vn = variant_name.lower().replace(" ", "_").replace("-", "_")
            is_proposed = (
                _vn in ("full", "full_model", "ours", "proposed", "calibrated")
                or _vn.startswith("full_model")
                or "proposed" in _vn or "ours" in _vn
            )
            if not is_proposed and len(variants) == 2 and "ablation" in str(list(variants.keys())).lower():
                if "ablation" not in _vn and "baseline" not in _vn and "w/o" not in _vn:
                    is_proposed = True
            main_results.append({
                "method_name": variant_name,
                "dataset": dataset_name,
                "is_proposed": is_proposed,
                "metrics": metric_list,
            })
            ablation_results.append({
                "variant_name": variant_name,
                "metrics": metric_list,
            })

        if main_results:
            data["main_results"] = main_results
        if ablation_results:
            data["ablation_results"] = ablation_results
        if "training_log" not in data:
            data["training_log"] = []

        logger.info("Converted variants-format metrics to standard schema "
                     "(%d main_results, %d ablation_results)",
                     len(main_results), len(ablation_results))
        return data
