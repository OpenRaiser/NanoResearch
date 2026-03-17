"""Smoke execution helpers — data generation, blueprints, metrics collection."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from nanoresearch.config import ExecutionProfile, ResearchConfig
from nanoresearch.agents.runtime_env import RuntimeEnvironmentManager
from nanoresearch.pipeline.workspace import Workspace


def _extract_structured_metrics(payload: dict[str, Any]) -> dict[str, float]:
    main_results = payload.get("main_results")
    if not isinstance(main_results, list):
        return {}

    selected: dict[str, Any] | None = None
    for candidate in main_results:
        if not isinstance(candidate, dict):
            continue
        if candidate.get("is_proposed") is True:
            selected = candidate
            break
        if selected is None:
            selected = candidate

    if not isinstance(selected, dict):
        return {}

    metrics_block = selected.get("metrics")
    if not isinstance(metrics_block, list):
        return {}

    metrics: dict[str, float] = {}
    for item in metrics_block:
        if not isinstance(item, dict):
            continue
        name = str(item.get("metric_name") or "").strip()
        value = item.get("value")
        if not name or isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            continue
        metrics[name] = numeric_value
    return metrics


def _collect_scalar_metrics(
    workspace: Workspace,
    execution_output: dict[str, Any] | None = None,
) -> dict[str, float]:
    metrics_path = workspace.path / "experiment" / "results" / "metrics.json"
    if metrics_path.is_file():
        try:
            raw_metrics = workspace.read_json("experiment/results/metrics.json")
        except (FileNotFoundError, RuntimeError):
            raw_metrics = None

        if isinstance(raw_metrics, dict):
            metrics: dict[str, float] = {}
            for key, value in raw_metrics.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                numeric_value = float(value)
                if not math.isfinite(numeric_value):
                    continue
                metrics[str(key)] = numeric_value
            if metrics:
                return metrics

            structured_metrics = _extract_structured_metrics(raw_metrics)
            if structured_metrics:
                return structured_metrics

        if isinstance(raw_metrics, list) and raw_metrics:
            metrics = _extract_metrics_from_training_log(raw_metrics)
            if metrics:
                return metrics

    if not isinstance(execution_output, dict):
        return {}

    for candidate_key in ("parsed_metrics", "best_metrics"):
        candidate = execution_output.get(candidate_key)
        if not isinstance(candidate, dict):
            continue
        metrics = _coerce_scalar_dict(candidate)
        if metrics:
            return metrics
    return {}


def _extract_metrics_from_training_log(log: list[Any]) -> dict[str, float]:
    """Extract scalar metrics from an epoch-level training-log array."""
    summary: dict[str, Any] | None = None
    last_epoch: dict[str, Any] | None = None
    for entry in log:
        if not isinstance(entry, dict):
            continue
        if entry.get("summary") is True:
            summary = entry
        else:
            last_epoch = entry

    if not summary and not last_epoch:
        return {}
    merged: dict[str, Any] = {}
    if last_epoch:
        merged.update(last_epoch)
    if summary:
        merged.update(summary)
    return _coerce_scalar_dict(merged)


def _coerce_scalar_dict(source: dict[str, Any]) -> dict[str, float]:
    """Return finite numeric values from *source*, coercing string floats."""
    metrics: dict[str, float] = {}
    skip_keys = {"epoch", "summary", "epoch_time_sec"}
    for key, value in source.items():
        if key in skip_keys:
            continue
        if isinstance(value, str):
            try:
                value = float(value)
            except (ValueError, TypeError):
                continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            continue
        metrics[str(key)] = numeric_value
    return metrics


def _derive_experiment_status(execution_output: dict[str, Any]) -> str:
    for key in ("status", "execution_status", "experiment_status"):
        value = str(execution_output.get(key) or "").strip()
        if value:
            return value
    final_status = str(execution_output.get("final_status") or "").strip()
    if final_status == "COMPLETED":
        return "success"
    return final_status


async def _revalidate_runtime_after_execution(
    workspace: Workspace,
    config: ResearchConfig,
    execution_output: dict[str, Any],
) -> dict[str, Any]:
    runtime_env = execution_output.get("runtime_env", {})
    if not isinstance(runtime_env, dict):
        return {}

    python_path = str(runtime_env.get("python") or "").strip()
    code_dir = workspace.path / "experiment"
    if not python_path or not code_dir.is_dir():
        return {}

    manager = RuntimeEnvironmentManager(config)
    execution_policy = manager.build_execution_policy(code_dir)
    try:
        validation = await manager.validate_runtime(
            python_path,
            code_dir,
            execution_policy=execution_policy,
        )
    except Exception as exc:
        payload = {
            "status": "failed",
            "python": python_path,
            "error": f"{type(exc).__name__}: {exc}",
        }
        workspace.write_json("logs/runtime_validation_recheck.json", payload)
        return payload

    payload = {
        "status": validation.get("status", ""),
        "python": python_path,
        "execution_policy": execution_policy.to_dict(),
        "validation": validation,
    }
    workspace.write_json("logs/runtime_validation_recheck.json", payload)
    return payload


def _summarize_coding_output(coding_output: dict[str, Any], workspace: Workspace) -> dict[str, Any]:
    generated_files = coding_output.get("generated_files", [])
    file_count = len(generated_files) if isinstance(generated_files, list) else 0
    return {
        "code_dir": str(workspace.path / "experiment"),
        "generated_file_count": file_count,
        "train_command": coding_output.get("train_command", ""),
        "runner_command": coding_output.get("runner_command", ""),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a real end-to-end smoke test for NanoResearch execution automation.",
    )
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to NanoResearch config.json.")
    parser.add_argument("--repo-root", type=Path,
                        default=Path(__file__).resolve().parents[1],
                        help="Repository root.")
    parser.add_argument("--output-root", type=Path, default=None,
                        help="Directory for the generated smoke workspace.")
    parser.add_argument("--session-id", type=str, default=None,
                        help="Optional fixed workspace/session id.")
    parser.add_argument("--topic", type=str,
                        default="Smoke Test: Synthetic Binary Classification",
                        help="Topic passed into CodingAgent and ExecutionAgent.")
    parser.add_argument("--rows", type=int, default=600,
                        help="Synthetic CSV row count.")
    parser.add_argument("--features", type=int, default=12,
                        help="Synthetic CSV feature count.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for synthetic dataset generation.")
    parser.add_argument("--profile", type=str,
                        choices=[profile.value for profile in ExecutionProfile],
                        default=ExecutionProfile.LOCAL_QUICK.value,
                        help="Execution profile to use during the smoke run.")
    parser.add_argument("--quick-eval-timeout", type=int, default=None,
                        help="Optional override for config.quick_eval_timeout.")
    parser.add_argument("--local-execution-timeout", type=int, default=None,
                        help="Optional override for config.local_execution_timeout.")
    parser.add_argument("--experiment-conda-env", type=str, default="",
                        help="Optional override for config.experiment_conda_env.")
    return parser
