"""Real end-to-end smoke test for execution automation."""

from __future__ import annotations

import argparse
import asyncio
import csv
import math
import traceback
from datetime import datetime, timezone
from pathlib import Path
from random import Random
from typing import Any

from nanoresearch.agents.coding import CodingAgent
from nanoresearch.agents.execution import ExecutionAgent
from nanoresearch.agents.runtime_env import RuntimeEnvironmentManager
from nanoresearch.config import ExecutionProfile, ResearchConfig
from nanoresearch.pipeline.workspace import Workspace
from nanoresearch.schemas.manifest import PipelineMode, PipelineStage

DEFAULT_SMOKE_TOPIC = "Smoke Test: Synthetic Binary Classification"
DEFAULT_DATASET_NAME = "SmokeBinaryCSV"


def _timestamp_token() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _default_session_id() -> str:
    return f"smoke_e2e_{_timestamp_token()}"


def _write_synthetic_binary_csv(
    path: Path,
    *,
    rows: int,
    features: int,
    seed: int,
) -> dict[str, Any]:
    if rows < 8:
        raise ValueError("rows must be at least 8 for a meaningful smoke dataset")
    if features < 2:
        raise ValueError("features must be at least 2 for a meaningful smoke dataset")

    rng = Random(seed)
    weights = [rng.uniform(-2.0, 2.0) for _ in range(features)]
    records: list[list[float | int]] = []
    positive_count = 0

    for _ in range(rows):
        values = [round(rng.gauss(0.0, 1.0), 6) for _ in range(features)]
        margin = sum(weight * value for weight, value in zip(weights, values))
        margin += rng.gauss(0.0, 0.75)
        label = 1 if margin >= 0 else 0
        positive_count += int(label == 1)
        records.append([*values, label])

    if positive_count == 0:
        records[-1][-1] = 1
        positive_count = 1
    elif positive_count == rows:
        records[-1][-1] = 0
        positive_count = rows - 1

    path.parent.mkdir(parents=True, exist_ok=True)
    header = [f"feature_{index}" for index in range(features)] + ["label"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(records)

    return {
        "path": str(path),
        "rows": rows,
        "features": features,
        "seed": seed,
        "positive_rows": positive_count,
        "negative_rows": rows - positive_count,
        "size_bytes": path.stat().st_size,
    }


def _build_smoke_blueprint(dataset_meta: dict[str, Any]) -> dict[str, Any]:
    rows = int(dataset_meta["rows"])
    features = int(dataset_meta["features"])
    return {
        "title": "Smoke-Test MLP for Synthetic Binary Classification",
        "hypothesis_ref": "SMOKE-HYP-001",
        "datasets": [
            {
                "name": DEFAULT_DATASET_NAME,
                "description": (
                    "Small local CSV dataset staged inside the workspace for "
                    "end-to-end smoke testing."
                ),
                "source_url": "",
                "size_info": f"{rows} rows, {features} numeric features, 1 binary label",
                "preprocessing_notes": (
                    "Use the staged CSV path from setup_output and split locally "
                    "into train/validation sets."
                ),
            }
        ],
        "baselines": [
            {
                "name": "LogisticRegression",
                "description": "Simple linear baseline for tabular binary classification.",
                "reference_paper_id": "",
                "expected_performance": {
                    "accuracy": 0.75,
                },
            }
        ],
        "proposed_method": {
            "name": "SmokeMLP",
            "description": (
                "A lightweight multilayer perceptron for binary classification "
                "on tabular data."
            ),
            "key_components": [
                "MLP encoder",
                "dropout regularization",
                "early stopping",
            ],
            "architecture": "Two-layer feedforward network for binary classification.",
        },
        "metrics": [
            {
                "name": "accuracy",
                "description": "Classification accuracy",
                "higher_is_better": True,
                "primary": True,
            },
            {
                "name": "F1",
                "description": "Binary F1 score",
                "higher_is_better": True,
                "primary": False,
            },
        ],
        "ablation_groups": [
            {
                "group_name": "RegularizationAblation",
                "description": "Check whether dropout helps on the smoke-test dataset.",
                "variants": [
                    {
                        "name": "no_dropout",
                        "description": "Disable dropout in the hidden layer.",
                    }
                ],
            }
        ],
        "compute_requirements": {
            "gpu_type": "CPU_OR_SINGLE_GPU",
            "num_gpus": 1,
            "estimated_hours": 1,
        },
    }


def _build_smoke_setup_output(workspace: Workspace, dataset_meta: dict[str, Any]) -> dict[str, Any]:
    data_path = Path(str(dataset_meta["path"]))
    data_dir = workspace.path / "data"
    models_dir = workspace.path / "models"
    data_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    return {
        "data_dir": str(data_dir),
        "models_dir": str(models_dir),
        "cloned_repos": [],
        "code_analysis": {},
        "downloaded_resources": [
            {
                "name": DEFAULT_DATASET_NAME,
                "type": "dataset",
                "status": "downloaded",
                "path": str(data_path),
                "workspace_path": str(data_path),
                "staging_strategy": "copy",
                "files": [str(data_path)],
                "size_bytes": dataset_meta["size_bytes"],
            }
        ],
    }


def _seed_workspace_inputs(
    workspace: Workspace,
    blueprint: dict[str, Any],
    setup_output: dict[str, Any],
) -> None:
    workspace.write_json("plans/experiment_blueprint.json", blueprint)
    workspace.mark_stage_completed(
        PipelineStage.PLANNING,
        "plans/experiment_blueprint.json",
    )
    workspace.write_json("plans/setup_output.json", setup_output)
    workspace.mark_stage_completed(
        PipelineStage.SETUP,
        "plans/setup_output.json",
    )


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

        # Handle array format (epoch-level training log with optional summary).
        if isinstance(raw_metrics, list) and raw_metrics:
            metrics = _extract_metrics_from_training_log(raw_metrics)
            if metrics:
                return metrics

    if not isinstance(execution_output, dict):
        return {}

    # Try parsed_metrics first (richer than best_metrics in many runs).
    for candidate_key in ("parsed_metrics", "best_metrics"):
        candidate = execution_output.get(candidate_key)
        if not isinstance(candidate, dict):
            continue
        metrics = _coerce_scalar_dict(candidate)
        if metrics:
            return metrics
    return {}


def _extract_metrics_from_training_log(log: list[Any]) -> dict[str, float]:
    """Extract scalar metrics from an epoch-level training-log array.

    Prefers a summary entry (``{"summary": true, ...}``).  Falls back to the
    last epoch entry, picking the most informative scalar fields.
    """
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
    # Merge: start with last epoch (richer), overlay summary (authoritative).
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
    # Prefer execution_status ("did it run?") over experiment_status
    # ("did the result contract fully pass?").  For smoke tests, a successful
    # execution with parsed metrics is a success even when the contract says
    # "partial" due to format differences.
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


async def run_execution_smoke(
    *,
    config: ResearchConfig,
    repo_root: Path,
    output_root: Path | None = None,
    session_id: str | None = None,
    topic: str = DEFAULT_SMOKE_TOPIC,
    rows: int = 600,
    features: int = 12,
    seed: int = 42,
) -> dict[str, Any]:
    smoke_root = Path(output_root) if output_root is not None else repo_root / "smoke_runs"
    smoke_root.mkdir(parents=True, exist_ok=True)

    workspace = Workspace.create(
        topic=topic,
        config_snapshot=config.snapshot(),
        root=smoke_root,
        session_id=session_id or _default_session_id(),
        pipeline_mode=PipelineMode.DEEP,
    )

    started_at = datetime.now(timezone.utc).isoformat()
    coding_agent = CodingAgent(workspace, config)
    execution_agent = ExecutionAgent(workspace, config)
    coding_output: dict[str, Any] = {}
    execution_output: dict[str, Any] = {}
    runtime_validation_recheck: dict[str, Any] = {}
    dataset_meta: dict[str, Any] = {}
    summary_path = workspace.path / "logs" / "smoke_test_summary.json"
    active_stage = PipelineStage.CODING

    try:
        dataset_meta = _write_synthetic_binary_csv(
            workspace.path / "data" / "smoke_binary.csv",
            rows=rows,
            features=features,
            seed=seed,
        )
        blueprint = _build_smoke_blueprint(dataset_meta)
        setup_output = _build_smoke_setup_output(workspace, dataset_meta)
        _seed_workspace_inputs(workspace, blueprint, setup_output)

        workspace.mark_stage_running(PipelineStage.CODING)
        coding_output = await coding_agent.run(
            topic=topic,
            experiment_blueprint=blueprint,
            setup_output=setup_output,
        )
        workspace.write_json("plans/coding_output.json", coding_output)
        workspace.mark_stage_completed(
            PipelineStage.CODING,
            "plans/coding_output.json",
        )

        active_stage = PipelineStage.EXECUTION
        workspace.mark_stage_running(PipelineStage.EXECUTION)
        execution_output = await execution_agent.run(
            topic=topic,
            coding_output=coding_output,
            setup_output=setup_output,
            experiment_blueprint=blueprint,
        )
        workspace.write_json("plans/execution_output.json", execution_output)
        workspace.mark_stage_completed(
            PipelineStage.EXECUTION,
            "plans/execution_output.json",
        )

        runtime_validation_recheck = await _revalidate_runtime_after_execution(
            workspace,
            config,
            execution_output,
        )
        metrics = _collect_scalar_metrics(workspace, execution_output)
        summary = {
            "workspace": str(workspace.path),
            "summary_path": str(summary_path),
            "topic": topic,
            "status": "completed",
            "started_at": started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "execution_profile": config.execution_profile.value,
            "dataset": dataset_meta,
            "execution_backend": execution_output.get("execution_backend", ""),
            "experiment_status": _derive_experiment_status(execution_output),
            "final_status": execution_output.get("final_status", ""),
            "coding_output_summary": _summarize_coding_output(coding_output, workspace),
            "execution_output_path": str(workspace.path / "plans" / "execution_output.json"),
            "metrics_path": str(workspace.path / "experiment" / "results" / "metrics.json"),
            "metrics": metrics,
            "runtime_validation_recheck_path": (
                str(workspace.path / "logs" / "runtime_validation_recheck.json")
                if runtime_validation_recheck
                else ""
            ),
            "runtime_validation_recheck": runtime_validation_recheck,
            "execution_output": execution_output,
        }
        workspace.write_json("logs/smoke_test_summary.json", summary)
        return summary

    except Exception as exc:
        workspace.mark_stage_failed(active_stage, f"{type(exc).__name__}: {exc}")
        summary = {
            "workspace": str(workspace.path),
            "summary_path": str(summary_path),
            "topic": topic,
            "status": "failed",
            "started_at": started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "execution_profile": config.execution_profile.value,
            "dataset": dataset_meta,
            "failed_stage": active_stage.value,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "coding_output_summary": _summarize_coding_output(coding_output, workspace),
            "execution_output": execution_output,
        }
        workspace.write_json("logs/smoke_test_summary.json", summary)
        return summary

    finally:
        await coding_agent.close()
        await execution_agent.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a real end-to-end smoke test for NanoResearch execution automation.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to NanoResearch config.json. Defaults to ~/.nanobot/config.json.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root. Used to locate the default smoke_runs directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Directory that will contain the generated smoke workspace.",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Optional fixed workspace/session id. Default: smoke_e2e_<timestamp>.",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=DEFAULT_SMOKE_TOPIC,
        help="Topic passed into CodingAgent and ExecutionAgent.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=600,
        help="Synthetic CSV row count.",
    )
    parser.add_argument(
        "--features",
        type=int,
        default=12,
        help="Synthetic CSV feature count.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic dataset generation.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        choices=[profile.value for profile in ExecutionProfile],
        default=ExecutionProfile.LOCAL_QUICK.value,
        help="Execution profile to use during the smoke run.",
    )
    parser.add_argument(
        "--quick-eval-timeout",
        type=int,
        default=None,
        help="Optional override for config.quick_eval_timeout.",
    )
    parser.add_argument(
        "--local-execution-timeout",
        type=int,
        default=None,
        help="Optional override for config.local_execution_timeout.",
    )
    parser.add_argument(
        "--experiment-conda-env",
        type=str,
        default="",
        help="Optional override for config.experiment_conda_env.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    config = ResearchConfig.load(args.config)
    config.execution_profile = ExecutionProfile(args.profile)
    if args.quick_eval_timeout is not None:
        config.quick_eval_timeout = args.quick_eval_timeout
    if args.local_execution_timeout is not None:
        config.local_execution_timeout = args.local_execution_timeout
    if args.experiment_conda_env:
        config.experiment_conda_env = args.experiment_conda_env.strip()

    summary = asyncio.run(
        run_execution_smoke(
            config=config,
            repo_root=args.repo_root.resolve(),
            output_root=args.output_root.resolve() if args.output_root else None,
            session_id=args.session_id,
            topic=args.topic.strip(),
            rows=args.rows,
            features=args.features,
            seed=args.seed,
        )
    )

    print(f"workspace={summary.get('workspace', '')}")
    print(f"summary={summary.get('summary_path', '')}")
    print(f"status={summary.get('status', '')}")
    print(f"experiment_status={summary.get('experiment_status', '')}")
    print(f"final_status={summary.get('final_status', '')}")
    return 0 if summary.get("status") == "completed" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
