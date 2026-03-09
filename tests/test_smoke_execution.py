from __future__ import annotations

from pathlib import Path

import pytest

from nanoresearch.config import ResearchConfig
from nanoresearch.smoke_execution import run_execution_smoke


@pytest.mark.asyncio
async def test_run_execution_smoke_writes_workspace_summary(tmp_path, monkeypatch):
    async def fake_coding_run(self, **inputs):
        experiment_dir = self.workspace.path / "experiment"
        experiment_dir.mkdir(parents=True, exist_ok=True)
        (experiment_dir / "train.py").write_text("print('ok')\n", encoding="utf-8")
        return {
            "generated_files": ["train.py"],
            "train_command": "python train.py",
            "runner_command": "python nanoresearch_runner.py",
        }

    async def fake_execution_run(self, **inputs):
        results_dir = self.workspace.path / "experiment" / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        self.workspace.write_json(
            "experiment/results/metrics.json",
            {
                "accuracy": 0.9,
                "F1": 0.8983,
            },
        )
        return {
            "status": "success",
            "execution_backend": "local",
            "execution_status": "success",
            "quick_eval_status": "success",
            "final_status": "COMPLETED",
        }

    monkeypatch.setattr(
        "nanoresearch.smoke_execution.CodingAgent.run",
        fake_coding_run,
    )
    monkeypatch.setattr(
        "nanoresearch.smoke_execution.ExecutionAgent.run",
        fake_execution_run,
    )

    config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
    summary = await run_execution_smoke(
        config=config,
        repo_root=tmp_path,
        output_root=tmp_path / "smoke_runs",
        session_id="smoke_test_case",
        rows=64,
        features=6,
        seed=7,
    )

    workspace_path = Path(summary["workspace"])
    assert summary["status"] == "completed"
    assert summary["execution_backend"] == "local"
    assert summary["experiment_status"] == "success"
    assert summary["final_status"] == "COMPLETED"
    assert summary["metrics"] == {"accuracy": 0.9, "F1": 0.8983}
    assert (workspace_path / "plans" / "setup_output.json").is_file()
    assert (workspace_path / "plans" / "experiment_blueprint.json").is_file()
    assert (workspace_path / "plans" / "coding_output.json").is_file()
    assert (workspace_path / "plans" / "execution_output.json").is_file()
    assert (workspace_path / "logs" / "smoke_test_summary.json").is_file()
    assert (workspace_path / "data" / "smoke_binary.csv").is_file()
