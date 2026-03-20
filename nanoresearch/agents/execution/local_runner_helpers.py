"""Local execution: helper methods for _LocalRunnerMixin."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from nanoresearch.agents.project_runner import (
    RUNNER_SCRIPT_NAME,
    is_python_launcher_token,
    normalize_target_spec,
)
from nanoresearch.schemas.iteration import IterationState

logger = logging.getLogger(__name__)


class _LocalRunnerHelpersMixin:

    @staticmethod
    def _command_with_mode(base_command: list[str], mode_flag: str) -> list[str]:
        """Append a pipeline mode flag if it is not already present."""
        if mode_flag in base_command:
            return list(base_command)
        return [*base_command, mode_flag]

    @staticmethod
    def _build_execution_blueprint_summary(
        topic: str,
        blueprint: dict[str, Any],
        setup_output: dict[str, Any],
        coding_output: dict[str, Any],
    ) -> str:
        """Compact execution context used for iterative repair."""
        payload = {
            "topic": topic,
            "title": blueprint.get("title", ""),
            "proposed_method": blueprint.get("proposed_method", {}),
            "datasets": blueprint.get("datasets", []),
            "metrics": blueprint.get("metrics", []),
            "baselines": blueprint.get("baselines", []),
            "ablation_groups": blueprint.get("ablation_groups", []),
            "downloaded_resources": setup_output.get("downloaded_resources", []),
            "data_dir": setup_output.get("data_dir", ""),
            "models_dir": setup_output.get("models_dir", ""),
            "train_command": coding_output.get("train_command", ""),
        }
        return json.dumps(payload, indent=2, ensure_ascii=False)

    @staticmethod
    def _update_best_round(
        iteration_state: IterationState,
        analysis: Any,
    ) -> None:
        """Track the current best round using the primary metric heuristic."""
        if not analysis or not getattr(analysis, "metric_summary", None):
            return
        primary_key = next(iter(analysis.metric_summary), None)
        primary_value = analysis.metric_summary.get(primary_key) if primary_key else None
        best_value = (
            iteration_state.best_metrics.get(primary_key)
            if iteration_state.best_metrics and primary_key
            else None
        )
        lower_is_better = bool(
            primary_key and any(
                kw in primary_key.lower()
                for kw in ("loss", "error", "perplexity", "mse", "mae", "cer", "wer")
            )
        )
        if best_value is None or primary_value is None:
            is_improvement = best_value is None and primary_value is not None
        elif lower_is_better:
            is_improvement = primary_value < best_value
        else:
            is_improvement = primary_value > best_value
        if is_improvement:
            iteration_state.best_round = iteration_state.rounds[-1].round_number
            iteration_state.best_metrics = analysis.metric_summary

    def _load_local_round_artifacts(self, round_number: int | None) -> dict[str, Any]:
        """Best-effort reload of local round artifacts from disk."""
        if round_number is None:
            return {}
        execution_path = self.workspace.path / "logs" / f"execution_round_{round_number}_execution.json"
        quick_eval_path = self.workspace.path / "logs" / f"execution_round_{round_number}_quick_eval.json"
        data: dict[str, Any] = {}
        if execution_path.exists():
            data["execution"] = json.loads(execution_path.read_text(encoding="utf-8"))
        if quick_eval_path.exists():
            data["quick_eval"] = json.loads(quick_eval_path.read_text(encoding="utf-8"))
        return data

    @staticmethod
    def _summarize_local_iteration(
        iteration_state: IterationState,
        blueprint: dict[str, Any],
    ) -> str:
        """Create a concise experiment summary for downstream writing/analysis."""
        method_name = (blueprint.get("proposed_method") or {}).get("name", "the proposed method")
        lines = [
            f"Executed local iterative experiment loop for {method_name}.",
            f"Completed rounds: {len(iteration_state.rounds)} / {iteration_state.max_rounds}.",
        ]
        if iteration_state.best_round is not None:
            lines.append(f"Best round: {iteration_state.best_round}.")
        if iteration_state.best_metrics:
            metrics_text = ", ".join(
                f"{key}={value}" for key, value in iteration_state.best_metrics.items()
            )
            lines.append(f"Best metrics: {metrics_text}.")
        if iteration_state.rounds and iteration_state.rounds[-1].analysis:
            analysis = iteration_state.rounds[-1].analysis
            lines.append(f"Latest attribution: {analysis.attribution or 'unknown'}.")
            if analysis.recommended_action:
                lines.append(f"Latest recommended action: {analysis.recommended_action}.")
        lines.append(f"Termination: {iteration_state.final_status}.")
        return "\n".join(lines)

    async def _local_preflight(self, code_dir: Path, python: str = "python") -> tuple[bool, str]:
        """Run local checks before submitting to SLURM.

        Tests:
        1. Python syntax check (py_compile) on all .py files
        2. Import check -- try importing the entry point module
        3. Verify all cross-file imports resolve

        Returns (ok, error_message).
        """
        errors = []

        # 1. Syntax check all .py files
        for py_file in sorted(code_dir.glob("*.py")):
            result = await self._run_subprocess(
                [python, "-c", f"import py_compile; py_compile.compile(r'{py_file}', doraise=True)"],
                timeout=10,
            )
            if result["returncode"] != 0:
                errors.append(f"Syntax error in {py_file.name}:\n{result['stderr']}")

        if errors:
            return False, "\n".join(errors)

        # 2. Try importing the main modules to catch import errors
        # (run in the code directory so local imports work)
        py_modules = [f.stem for f in code_dir.glob("*.py")]
        for module in py_modules:
            result = await self._run_subprocess(
                [python, "-c", f"import {module}"],
                cwd=code_dir,
                timeout=30,
            )
            if result["returncode"] != 0:
                err_text = result["stdout"] + result["stderr"]
                # Ignore errors from missing heavy dependencies (torch, etc.)
                # -- those will be installed on the cluster node
                if any(pkg in err_text for pkg in [
                    "No module named 'torch'",
                    "No module named 'torchvision'",
                    "No module named 'torchaudio'",
                    "No module named 'timm'",
                    "No module named 'transformers'",
                    "No module named 'torch_geometric'",
                    "No module named 'torch_scatter'",
                    "No module named 'torch_sparse'",
                    "No module named 'esm'",
                    "No module named 'dgl'",
                    "No module named 'accelerate'",
                    "No module named 'datasets'",
                    "No module named 'einops'",
                    "No module named 'wandb'",
                    "No module named 'scipy'",
                    "No module named 'sklearn'",
                    "No module named 'cv2'",
                    "No module named 'PIL'",
                    "CUDA",
                ]):
                    continue
                errors.append(f"Import error in {module}.py:\n{err_text}")

        if errors:
            return False, "\n".join(errors)

        return True, ""

    def _build_local_command(
        self,
        code_dir: Path,
        coding_output: dict[str, Any],
        runtime_python: str,
    ) -> list[str]:
        runner_script = str(coding_output.get("runner_script", "")).strip()
        if runner_script and Path(runner_script).exists():
            runner_path = Path(runner_script)
            runner_token = runner_path.name if runner_path.parent == code_dir else str(runner_path)
            return [runtime_python, runner_token]
        if (code_dir / RUNNER_SCRIPT_NAME).exists():
            return [runtime_python, RUNNER_SCRIPT_NAME]

        command = str(
            coding_output.get("entry_train_command")
            or coding_output.get("train_command")
            or (coding_output.get("code_plan") or {}).get("train_command", "")
            or ""
        ).strip()
        if command:
            tokens, _env_vars = normalize_target_spec(command, code_dir)
            if tokens:
                if is_python_launcher_token(tokens[0]):
                    return [runtime_python, *tokens[1:]]
                if tokens[0] in {"-m", "-c"} or tokens[0].endswith(".py"):
                    return [runtime_python, *tokens]
                return tokens

        for candidate in ("main.py", "train.py", "run.py"):
            if (code_dir / candidate).exists():
                return [runtime_python, candidate]
        return [runtime_python, "main.py"]

    async def _run_local_training(
        self,
        code_dir: Path,
        command: list[str],
    ) -> dict[str, Any]:
        timeout = max(60, int(self.config.local_execution_timeout))
        result = await self._run_subprocess(command, cwd=code_dir, timeout=timeout)
        result["command"] = command
        result["timed_out"] = (
            result.get("returncode") == -1
            and "timed out" in result.get("stderr", "").lower()
        )
        return result
