"""Tests for legacy ExperimentAgent runner parity."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
import uuid

import pytest

from nanoresearch.agents.experiment import ExperimentAgent
from nanoresearch.agents.project_runner import (
    RUNNER_SCRIPT_NAME,
    _build_runner_script,
    ensure_project_runner,
)
from nanoresearch.config import ResearchConfig
from nanoresearch.pipeline.workspace import Workspace


def test_build_legacy_runner_command_prefers_train_py_when_main_missing() -> None:
    tmp_dir = Path(f".test_experiment_runner_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="legacy_runner_cmd",
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = ExperimentAgent(workspace, config)

        code_dir = workspace.path / "code"
        code_dir.mkdir(parents=True, exist_ok=True)
        (code_dir / "train.py").write_text("print('train')\n", encoding="utf-8")

        command = agent._build_legacy_runner_command(code_dir, mode="quick-eval")

        assert command == f"python {RUNNER_SCRIPT_NAME} --quick-eval"
        runner_config = json.loads((code_dir / "nanoresearch_runner.json").read_text(encoding="utf-8"))
        assert runner_config["target_command"] == ["train.py"]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_ensure_project_runner_strips_wrapping_quotes_from_arg_values() -> None:
    tmp_dir = Path(f".test_experiment_runner_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        code_dir = tmp_dir / "code"
        code_dir.mkdir(parents=True, exist_ok=True)
        quoted_path = (
            "C:\\Users\\17965\\Desktop\\NaNo\\NanoResearch-main\\NanoResearch-main"
            "\\smoke_runs\\quoted path\\data.csv"
        )

        assets = ensure_project_runner(
            code_dir,
            f'python train.py --data_path "{quoted_path}" --epochs 3',
        )

        assert assets["target_command"] == [
            "train.py",
            "--data_path",
            quoted_path,
            "--epochs",
            "3",
        ]
        runner_config = json.loads(
            (code_dir / "nanoresearch_runner.json").read_text(encoding="utf-8")
        )
        assert runner_config["target_command"][2] == quoted_path
        assert not runner_config["target_command"][2].startswith('"')
        assert not runner_config["target_command"][2].endswith('"')
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _get_detect_flag_form():
    """Extract _detect_flag_form from the runner template string."""
    import re as _re
    code = _build_runner_script()
    lines = code.splitlines(keepends=True)
    # Find start of _detect_flag_form and the next top-level def after it
    start = None
    end = None
    for i, line in enumerate(lines):
        if line.startswith("def _detect_flag_form("):
            start = i
        elif start is not None and i > start and line.startswith("def "):
            end = i
            break
    assert start is not None, "_detect_flag_form not found in runner template"
    func_code = "import re\nfrom pathlib import Path\n" + "".join(lines[start:end])
    ns: dict = {}
    exec(compile(func_code, "<runner>", "exec"), ns)
    return ns["_detect_flag_form"]


_detect_flag_form = _get_detect_flag_form()


def test_detect_flag_form_resolves_underscore_argparse(tmp_path: Path) -> None:
    """Reproduce the 0045 blocker: train.py defines --dry_run (underscore)
    but runner was passing --dry-run (hyphen).  _detect_flag_form must return
    the ACTUAL form declared in add_argument()."""
    script = tmp_path / "train.py"
    script.write_text(
        'import argparse\n'
        'parser = argparse.ArgumentParser()\n'
        'parser.add_argument("--dry_run", action="store_true")\n'
        'parser.add_argument("--quick_eval", action="store_true")\n'
        'parser.add_argument("--batch_size", type=int, default=32)\n',
        encoding="utf-8",
    )

    assert _detect_flag_form(script, "--dry-run") == "--dry_run"
    assert _detect_flag_form(script, "--dry_run") == "--dry_run"
    assert _detect_flag_form(script, "--quick-eval") == "--quick_eval"
    assert _detect_flag_form(script, "--batch-size") == "--batch_size"


def test_detect_flag_form_resolves_hyphen_argparse(tmp_path: Path) -> None:
    """When train.py uses hyphen form (--dry-run), detect that correctly."""
    script = tmp_path / "train.py"
    script.write_text(
        'import argparse\n'
        'parser = argparse.ArgumentParser()\n'
        'parser.add_argument("--dry-run", action="store_true")\n'
        'parser.add_argument("--quick-eval", action="store_true")\n',
        encoding="utf-8",
    )

    assert _detect_flag_form(script, "--dry-run") == "--dry-run"
    assert _detect_flag_form(script, "--dry_run") == "--dry-run"
    assert _detect_flag_form(script, "--quick-eval") == "--quick-eval"


def test_detect_flag_form_returns_none_for_missing_flag(tmp_path: Path) -> None:
    script = tmp_path / "train.py"
    script.write_text("print('no argparse')\n", encoding="utf-8")
    assert _detect_flag_form(script, "--dry-run") is None


def test_detect_flag_form_ignores_comment_when_argparse_differs(tmp_path: Path) -> None:
    """If a comment mentions --dry-run but argparse defines --dry_run,
    the argparse definition wins."""
    script = tmp_path / "train.py"
    script.write_text(
        '# This script supports --dry-run mode\n'
        'import argparse\n'
        'parser = argparse.ArgumentParser()\n'
        'parser.add_argument("--dry_run", action="store_true")\n',
        encoding="utf-8",
    )
    assert _detect_flag_form(script, "--dry-run") == "--dry_run"


@pytest.mark.asyncio
async def test_run_main_py_uses_runner_when_main_missing(monkeypatch) -> None:
    tmp_dir = Path(f".test_experiment_runner_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="legacy_runner_exec",
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = ExperimentAgent(workspace, config)

        code_dir = workspace.path / "code"
        code_dir.mkdir(parents=True, exist_ok=True)
        (code_dir / "train.py").write_text("print('train')\n", encoding="utf-8")

        commands: list[list[str]] = []

        class FakePopen:
            returncode = 0
            def __init__(self, command, **_kwargs):
                commands.append(command)
                self.pid = 99999
            def communicate(self, timeout=None):
                return b"ok\n", b""
            def kill(self):
                pass

        monkeypatch.setattr(subprocess, "Popen", FakePopen)

        result = await agent._run_main_py(code_dir, "python-custom")

        assert result["returncode"] == 0
        assert commands == [["python-custom", RUNNER_SCRIPT_NAME, "--dry-run"]]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_run_on_cluster_uses_runner_command_and_materializes_runner() -> None:
    tmp_dir = Path(f".test_experiment_runner_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="legacy_runner_cluster",
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = ExperimentAgent(workspace, config)

        code_dir = workspace.path / "code"
        (code_dir / "results").mkdir(parents=True, exist_ok=True)
        (code_dir / "train.py").write_text("print('train')\n", encoding="utf-8")

        class DummyCluster:
            def __init__(self) -> None:
                self.script_cmd = ""

            async def prepare_code(self, local_code_dir: Path, _session_id: str) -> str:
                assert (local_code_dir / RUNNER_SCRIPT_NAME).exists()
                return "/cluster/code"

            async def setup_env(self, _cluster_code_path: str) -> dict:
                return {"ok": True, "output": "", "source": "", "strategy": "existing", "manifest": ""}

            async def reupload_code(self, _local_code_dir: Path, _cluster_code_path: str) -> None:
                return None

            async def submit_job(self, _cluster_code_path: str, script_cmd: str) -> str:
                self.script_cmd = script_cmd
                return "12345"

            async def wait_for_job(self, _job_id: str) -> dict:
                return {"state": "COMPLETED"}

            async def download_results(self, _cluster_code_path: str, _workspace_path: Path) -> bool:
                (code_dir / "results" / "metrics.json").write_text(
                    json.dumps(
                        {
                            "main_results": [
                                {
                                    "method_name": "LegacyCluster",
                                    "dataset": "DemoSet",
                                    "is_proposed": True,
                                    "metrics": [{"metric_name": "accuracy", "value": 0.9}],
                                }
                            ],
                            "ablation_results": [],
                            "training_log": [],
                        }
                    ),
                    encoding="utf-8",
                )
                return True

            async def get_job_log(self, _cluster_code_path: str, _job_id: str) -> str:
                return ""

        cluster = DummyCluster()
        execution, quick_eval = await agent._run_on_cluster(cluster, code_dir, 1, "")

        assert cluster.script_cmd == f"python {RUNNER_SCRIPT_NAME} --quick-eval"
        assert execution["status"] == "success"
        assert quick_eval["status"] == "success"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
