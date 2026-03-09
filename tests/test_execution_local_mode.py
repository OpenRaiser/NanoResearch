"""Tests for local execution helpers in the unified pipeline."""

from __future__ import annotations

import asyncio
import gzip
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from unittest.mock import AsyncMock
import uuid
import zipfile

import pytest

from nanoresearch.agents.experiment import ExperimentAgent
from nanoresearch.agents.execution import ExecutionAgent
from nanoresearch.agents.project_runner import (
    RUNNER_CONFIG_NAME,
    RUNNER_SCRIPT_NAME,
    ensure_project_runner,
    normalize_target_spec,
    repair_launch_contract,
    validate_launch_contract,
)
from nanoresearch.config import ResearchConfig
from nanoresearch.pipeline.workspace import Workspace


def test_build_local_command_uses_runtime_python() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="exec001",
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = ExecutionAgent(workspace, config)

        code_dir = workspace.path / "experiment"
        code_dir.mkdir(exist_ok=True)
        (code_dir / "train.py").write_text("print('ok')", encoding="utf-8")

        command = agent._build_local_command(
            code_dir,
            {"train_command": "python train.py --epochs 1"},
            "C:/env/python.exe",
        )

        assert command[0] == "C:/env/python.exe"
        assert command[1:] == ["train.py", "--epochs", "1"]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_evaluate_experiment_contract_treats_recovered_metrics_as_partial() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="exec_contract_001",
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = ExecutionAgent(workspace, config)

        contract = agent._evaluate_experiment_contract(
            {
                "metrics": {
                    "main_results": [
                        {
                            "method_name": "Recovered",
                            "dataset": "DemoSet",
                            "is_proposed": True,
                            "metrics": [{"metric_name": "accuracy", "value": 0.91}],
                        }
                    ],
                    "ablation_results": [],
                    "training_log": [],
                },
                "stdout_log": "Test accuracy: 0.91",
                "stderr_log": "",
                "recovered_from": "execution_log",
            },
            execution_backend="local",
            execution_status="success",
            quick_eval_status="partial",
            final_status="COMPLETED",
        )

        assert contract["status"] == "partial"
        assert contract["success_path"] == "structured_metrics_recovered"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_evaluate_experiment_contract_promotes_strong_recovered_metrics_to_success() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="exec_contract_001b",
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = ExecutionAgent(workspace, config)

        contract = agent._evaluate_experiment_contract(
            {
                "metrics": {
                    "main_results": [
                        {
                            "method_name": "Recovered",
                            "dataset": "DemoSet",
                            "is_proposed": True,
                            "metrics": [{"metric_name": "accuracy", "value": 0.91}],
                        }
                    ],
                    "ablation_results": [],
                    "training_log": [{"epoch": 1, "train_loss": 0.5, "metrics": {}}],
                    "_nanoresearch_meta": {"recovered_from": "execution_log"},
                },
                "parsed_metrics": {"accuracy": "0.91"},
                "training_log": [{"epoch": 1, "train_loss": 0.5, "metrics": {}}],
                "checkpoints": ["checkpoints/best_model.pt"],
                "result_file_summary.json": "results/summary.json",
                "stdout_log": "Test accuracy: 0.91",
                "stderr_log": "",
                "recovered_from": "execution_log",
            },
            execution_backend="local",
            execution_status="success",
            quick_eval_status="partial",
            final_status="COMPLETED",
        )

        assert contract["status"] == "success"
        assert contract["success_path"] == "structured_metrics_recovered"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_collect_result_artifacts_reads_recovered_metadata_and_training_log() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="exec_contract_002",
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = ExecutionAgent(workspace, config)

        code_dir = workspace.path / "experiment"
        (code_dir / "results").mkdir(parents=True)
        (code_dir / "results" / "metrics.json").write_text(
            json.dumps(
                {
                    "main_results": [
                        {
                            "method_name": "Recovered",
                            "dataset": "DemoSet",
                            "is_proposed": True,
                            "metrics": [{"metric_name": "accuracy", "value": 0.91}],
                        }
                    ],
                    "ablation_results": [],
                    "training_log": [{"epoch": 1, "train_loss": 0.5, "metrics": {}}],
                    "_nanoresearch_meta": {"recovered_from": "execution_log"},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        results = agent._collect_result_artifacts(code_dir)

        assert results["recovered_from"] == "execution_log"
        assert results["training_log"][0]["epoch"] == 1
        assert results["metrics"]["main_results"][0]["metrics"][0]["metric_name"] == "accuracy"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_run_subprocess_falls_back_to_sync_on_permission_error(monkeypatch) -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="exec_subprocess_fallback",
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = ExecutionAgent(workspace, config)

        async def fake_create_subprocess_exec(*_args, **_kwargs):
            raise PermissionError(5, "denied")

        captured: dict[str, object] = {}

        def fake_sync(command: list[str], *, cwd=None, timeout=60, env=None):
            captured["command"] = list(command)
            captured["cwd"] = str(cwd) if cwd is not None else None
            captured["timeout"] = timeout
            captured["env"] = dict(env or {})
            return {"returncode": 0, "stdout": "ok", "stderr": ""}

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
        monkeypatch.setattr(ExecutionAgent, "_run_subprocess_sync", staticmethod(fake_sync))

        result = await agent._run_subprocess(["python", "-c", "print('ok')"], cwd=workspace.path, timeout=7)

        assert result["returncode"] == 0
        assert result["stdout"] == "ok"
        assert captured["command"] == ["python", "-c", "print('ok')"]
        assert captured["cwd"] == str(workspace.path)
        assert captured["timeout"] == 7
        assert "PYTHONDONTWRITEBYTECODE" in captured["env"]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_run_subprocess_sync_maps_timeout_to_standard_payload(monkeypatch) -> None:
    def fake_run(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd=["python"], timeout=3)

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = ExecutionAgent._run_subprocess_sync(["python"], timeout=3)

    assert result == {"returncode": -1, "stdout": "", "stderr": "Command timed out"}


def test_record_runtime_env_ledger_includes_validation_entry() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="exec_env_validation_001",
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = ExecutionAgent(workspace, config)

        remediation_ledger: list[dict] = []
        agent._record_runtime_env_ledger(
            {
                "kind": "venv",
                "created": True,
                "env_path": "experiment/.venv",
                "dependency_install": {
                    "status": "installed",
                    "source": "requirements.txt",
                    "manifest": "experiment/requirements.txt",
                    "strategy": "primary",
                },
                "runtime_validation": {
                    "status": "partial",
                    "python_smoke": {
                        "status": "passed",
                        "executable": "experiment/.venv/Scripts/python.exe",
                        "version": "3.11.9",
                    },
                    "pip_probe": {
                        "status": "passed",
                        "version": "pip 24.0",
                    },
                    "import_probe": {
                        "status": "partial",
                        "failures": [
                            {
                                "package": "pyyaml",
                                "module": "yaml",
                                "error": "ModuleNotFoundError: No module named yaml",
                            }
                        ],
                        "skipped_reason": "",
                    },
                },
                "runtime_validation_repair": {
                    "status": "applied",
                    "actions": [
                        {
                            "kind": "import_repair_install",
                            "status": "installed",
                            "specs": ["PyYAML>=6"],
                        }
                    ],
                },
            },
            remediation_ledger,
        )

        assert [entry["kind"] for entry in remediation_ledger] == [
            "runtime_env_create",
            "dependency_install",
            "runtime_env_validation",
            "runtime_env_repair",
        ]
        validation_entry = remediation_ledger[-1]
        assert validation_entry["status"] == "applied"
        assert validation_entry["details"]["actions"][0]["kind"] == "import_repair_install"
        runtime_validation_entry = remediation_ledger[-2]
        assert runtime_validation_entry["status"] == "partial"
        assert runtime_validation_entry["details"]["python_smoke_status"] == "passed"
        assert runtime_validation_entry["details"]["pip_status"] == "passed"
        assert runtime_validation_entry["details"]["import_status"] == "partial"
        assert runtime_validation_entry["details"]["failed_imports"][0]["module"] == "yaml"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_evaluate_experiment_contract_rejects_crash_without_artifacts() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="exec_contract_002",
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = ExecutionAgent(workspace, config)

        contract = agent._evaluate_experiment_contract(
            {
                "metrics": {},
                "stdout_log": "",
                "stderr_log": "Traceback (most recent call last):\nModuleNotFoundError: No module named 'demo_pkg'",
            },
            execution_backend="local",
            execution_status="failed",
            quick_eval_status="failed",
            final_status="FAILED",
        )

        assert contract["status"] == "failed"
        assert "crash_free_logs" in contract["missing_signals"]
        assert "Traceback" in contract["failure_signals"]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_build_local_command_rewrites_explicit_python_path() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="exec_python_path",
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = ExecutionAgent(workspace, config)

        code_dir = workspace.path / "experiment"
        code_dir.mkdir(exist_ok=True)
        (code_dir / "train.py").write_text("print('ok')", encoding="utf-8")

        command = agent._build_local_command(
            code_dir,
            {"train_command": "./.venv/Scripts/python.exe train.py --epochs 1"},
            "C:/runtime/python.exe",
        )

        assert command == ["C:/runtime/python.exe", "train.py", "--epochs", "1"]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_build_local_command_prefixes_runtime_python_for_bare_script() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="exec_bare_script",
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = ExecutionAgent(workspace, config)

        code_dir = workspace.path / "experiment"
        code_dir.mkdir(exist_ok=True)
        (code_dir / "train.py").write_text("print('ok')", encoding="utf-8")

        command = agent._build_local_command(
            code_dir,
            {"train_command": "train.py --epochs 1"},
            "python-custom",
        )

        assert command == ["python-custom", "train.py", "--epochs", "1"]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_build_local_command_falls_back_to_main_py() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="exec002",
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = ExecutionAgent(workspace, config)

        code_dir = workspace.path / "experiment"
        code_dir.mkdir(exist_ok=True)
        (code_dir / "main.py").write_text("print('ok')", encoding="utf-8")

        command = agent._build_local_command(code_dir, {}, "python-custom")

        assert command == ["python-custom", "main.py"]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_build_local_command_strips_env_prefix_and_shell_chain() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="exec_env_prefix",
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = ExecutionAgent(workspace, config)

        code_dir = workspace.path / "experiment"
        code_dir.mkdir(exist_ok=True)
        (code_dir / "train.py").write_text("print('ok')", encoding="utf-8")

        command = agent._build_local_command(
            code_dir,
            {"train_command": "CUDA_VISIBLE_DEVICES=0 python train.py --epochs 1 && python eval.py"},
            "python-custom",
        )

        assert command == ["python-custom", "train.py", "--epochs", "1"]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_build_local_command_unwraps_shell_command_string() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="exec_shell_wrapper",
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = ExecutionAgent(workspace, config)

        code_dir = workspace.path / "experiment"
        code_dir.mkdir(exist_ok=True)
        (code_dir / "train.py").write_text("print('ok')", encoding="utf-8")

        if platform.system() == "Windows":
            wrapped = 'cmd /c python train.py --epochs 1'
        else:
            wrapped = 'bash -lc "python train.py --epochs 1"'

        command = agent._build_local_command(
            code_dir,
            {"train_command": wrapped},
            "python-custom",
        )

        assert command == ["python-custom", "train.py", "--epochs", "1"]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_build_local_command_prefers_deterministic_runner() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="exec_runner",
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = ExecutionAgent(workspace, config)

        code_dir = workspace.path / "experiment"
        code_dir.mkdir(exist_ok=True)
        (code_dir / RUNNER_SCRIPT_NAME).write_text("print('runner')", encoding="utf-8")
        (code_dir / "train.py").write_text("print('train')", encoding="utf-8")

        command = agent._build_local_command(
            code_dir,
            {"train_command": "python train.py"},
            "python-custom",
        )

        assert command == ["python-custom", RUNNER_SCRIPT_NAME]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_normalize_target_spec_preserves_env_for_runner() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        code_dir = tmp_dir / "code"
        code_dir.mkdir()
        (code_dir / "train.py").write_text("print('ok')", encoding="utf-8")

        tokens, env_vars = normalize_target_spec(
            "CUDA_VISIBLE_DEVICES=2 python train.py --quick-eval && python other.py",
            code_dir,
        )

        assert tokens == ["train.py"]
        assert env_vars == {"CUDA_VISIBLE_DEVICES": "2"}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_validate_launch_contract_creates_missing_artifact_dirs() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        code_dir = tmp_dir / "code"
        code_dir.mkdir()
        (code_dir / "train.py").write_text("print('ok')\n", encoding="utf-8")

        contract = validate_launch_contract(["python", "train.py"], code_dir)

        assert contract["status"] == "repaired"
        assert (code_dir / "results").exists()
        assert (code_dir / "checkpoints").exists()
        assert (code_dir / "logs").exists()
        assert contract["target_kind"] == "script"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_validate_launch_contract_rejects_runner_with_missing_target() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        code_dir = tmp_dir / "code"
        code_dir.mkdir()
        (code_dir / RUNNER_SCRIPT_NAME).write_text("print('runner')\n", encoding="utf-8")
        (code_dir / "nanoresearch_runner.json").write_text(
            json.dumps({"target_command": ["missing_train.py"]}),
            encoding="utf-8",
        )

        contract = validate_launch_contract(["python", RUNNER_SCRIPT_NAME], code_dir)

        assert contract["status"] == "failed"
        assert contract["target_kind"] == "runner"
        assert any("Runner target invalid" in message for message in contract["failures"])
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_validate_launch_contract_rejects_target_outside_workspace() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        code_dir = tmp_dir / "code"
        code_dir.mkdir()
        outside_script = tmp_dir / "outside.py"
        outside_script.write_text("print('outside')\n", encoding="utf-8")

        contract = validate_launch_contract(["python", str(outside_script.resolve())], code_dir)

        assert contract["status"] == "failed"
        assert any("outside workspace" in message for message in contract["failures"])
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_repair_launch_contract_refreshes_stale_runner_target() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        code_dir = tmp_dir / "code"
        code_dir.mkdir()
        (code_dir / RUNNER_SCRIPT_NAME).write_text("print('runner')\n", encoding="utf-8")
        (code_dir / RUNNER_CONFIG_NAME).write_text(
            json.dumps({"target_command": ["missing_train.py"], "target_env": {"CUDA_VISIBLE_DEVICES": "1"}}),
            encoding="utf-8",
        )
        (code_dir / "train.py").write_text("print('ok')\n", encoding="utf-8")

        repair = repair_launch_contract(["python", RUNNER_SCRIPT_NAME], code_dir)
        runner_config = json.loads((code_dir / RUNNER_CONFIG_NAME).read_text(encoding="utf-8"))

        assert repair["status"] == "applied"
        assert repair["final_contract"]["status"] == "ready"
        assert runner_config["target_command"] == ["train.py"]
        assert runner_config["target_env"] == {"CUDA_VISIBLE_DEVICES": "1"}
        assert repair["command"] == ["python", RUNNER_SCRIPT_NAME]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_repair_launch_contract_redirects_missing_script_to_runner() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        code_dir = tmp_dir / "code"
        code_dir.mkdir()
        (code_dir / "train.py").write_text("print('ok')\n", encoding="utf-8")

        repair = repair_launch_contract(
            ["C:/env/python.exe", str((tmp_dir / "stale" / "train.py").resolve()), "--epochs", "1"],
            code_dir,
        )
        runner_config = json.loads((code_dir / RUNNER_CONFIG_NAME).read_text(encoding="utf-8"))

        assert repair["status"] == "applied"
        assert repair["final_contract"]["status"] == "ready"
        assert repair["command"] == ["C:/env/python.exe", RUNNER_SCRIPT_NAME, "--epochs", "1"]
        assert runner_config["target_command"] == ["train.py"]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_project_runner_executes_non_python_launcher() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        code_dir = tmp_dir / "code"
        code_dir.mkdir()
        invoked = (code_dir / "launcher_invoked.txt").resolve()
        (code_dir / "train.py").write_text("print('train entry')", encoding="utf-8")

        if platform.system() == "Windows":
            launcher = tmp_dir / "launcher.py"
            launcher.write_text(
                "from pathlib import Path\n"
                "import os\n"
                "import sys\n"
                f"Path(r'{invoked}').write_text("
                "f\"{os.environ.get('NANORESEARCH_QUICK_EVAL', '')} {' '.join(sys.argv[1:])}\", "
                "encoding='utf-8')\n",
                encoding="utf-8",
            )
            launcher_command = f"cmd /c {sys.executable} {launcher.resolve()} train.py --quick-eval"
        else:
            launcher = tmp_dir / "launcher.sh"
            launcher.write_text(
                "#!/usr/bin/env sh\n"
                f"printf '%s %s\\n' \"$NANORESEARCH_QUICK_EVAL\" \"$*\" > '{invoked.as_posix()}'\n"
                "exit 0\n",
                encoding="utf-8",
            )
            launcher.chmod(0o755)
            launcher_command = f"sh {launcher.resolve()} train.py --quick-eval"

        ensure_project_runner(code_dir, launcher_command)

        result = subprocess.run(
            [sys.executable, RUNNER_SCRIPT_NAME, "--quick-eval"],
            cwd=str(code_dir),
            capture_output=True,
            text=True,
            env=dict(os.environ),
            timeout=30,
        )

        assert result.returncode == 0, result.stderr
        assert invoked.exists()
        logged = invoked.read_text(encoding="utf-8")
        assert "1" in logged
        assert "train.py" in logged
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_project_runner_preserves_env_assignments() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        code_dir = tmp_dir / "code"
        code_dir.mkdir()
        output = (code_dir / "env_seen.txt").resolve()
        (code_dir / "train.py").write_text(
            "from pathlib import Path\n"
            "import os\n"
            f"Path(r'{output}').write_text(os.environ.get('CUDA_VISIBLE_DEVICES', ''), encoding='utf-8')\n",
            encoding="utf-8",
        )

        ensure_project_runner(code_dir, "CUDA_VISIBLE_DEVICES=7 python train.py")

        result = subprocess.run(
            [sys.executable, RUNNER_SCRIPT_NAME],
            cwd=str(code_dir),
            capture_output=True,
            text=True,
            env=dict(os.environ),
            timeout=30,
        )

        assert result.returncode == 0, result.stderr
        assert output.exists()
        assert output.read_text(encoding="utf-8") == "7"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_collect_local_results_reads_metrics_file() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="exec003",
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = ExecutionAgent(workspace, config)

        code_dir = workspace.path / "experiment"
        results_dir = code_dir / "results"
        checkpoints_dir = code_dir / "checkpoints"
        results_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        (results_dir / "metrics.json").write_text(
            json.dumps({"accuracy": 0.91}),
            encoding="utf-8",
        )
        (checkpoints_dir / "model.pt").write_text("weights", encoding="utf-8")

        results = agent._collect_local_results(
            code_dir,
            {"stdout": "done", "stderr": "", "returncode": 0},
        )

        assert results["metrics"]["main_results"][0]["metrics"][0]["metric_name"] == "accuracy"
        assert results["metrics"]["main_results"][0]["metrics"][0]["value"] == 0.91
        assert len(results["checkpoints"]) == 1
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_project_runner_adds_required_config_argument() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        code_dir = tmp_dir / "code"
        (code_dir / "config").mkdir(parents=True)
        (code_dir / "config" / "default.yaml").write_text("seed: 1\n", encoding="utf-8")
        output = (code_dir / "config_used.txt").resolve()
        (code_dir / "train.py").write_text(
            "import argparse\n"
            "from pathlib import Path\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument('--config', required=True)\n"
            "args = parser.parse_args()\n"
            f"Path(r'{output}').write_text(args.config, encoding='utf-8')\n",
            encoding="utf-8",
        )

        ensure_project_runner(code_dir, "python train.py")
        result = subprocess.run(
            [sys.executable, RUNNER_SCRIPT_NAME],
            cwd=str(code_dir),
            capture_output=True,
            text=True,
            env=dict(os.environ),
            timeout=30,
        )

        assert result.returncode == 0, result.stderr
        assert output.exists()
        assert Path(output.read_text(encoding="utf-8")).resolve() == (code_dir / "config" / "default.yaml").resolve()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_project_runner_replaces_missing_config_argument_path() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        code_dir = tmp_dir / "code"
        code_dir.mkdir()
        (code_dir / "config.py").write_text("CONFIG = True\n", encoding="utf-8")
        output = (code_dir / "config_used.txt").resolve()
        (code_dir / "train.py").write_text(
            "import argparse\n"
            "from pathlib import Path\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument('--config', required=True)\n"
            "args = parser.parse_args()\n"
            f"Path(r'{output}').write_text(args.config, encoding='utf-8')\n",
            encoding="utf-8",
        )

        ensure_project_runner(code_dir, "python train.py --config missing.yaml")
        result = subprocess.run(
            [sys.executable, RUNNER_SCRIPT_NAME],
            cwd=str(code_dir),
            capture_output=True,
            text=True,
            env=dict(os.environ),
            timeout=30,
        )

        assert result.returncode == 0, result.stderr
        assert output.exists()
        assert Path(output.read_text(encoding="utf-8")).resolve() == (code_dir / "config.py").resolve()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_project_runner_adds_required_train_file_when_unique() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        code_dir = tmp_dir / "code"
        data_dir = code_dir / "data"
        data_dir.mkdir(parents=True)
        (data_dir / "train.tsv").write_text("x\ty\n1\t2\n", encoding="utf-8")
        output = (code_dir / "train_file_used.txt").resolve()
        (code_dir / "train.py").write_text(
            "import argparse\n"
            "from pathlib import Path\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument('--train-file', required=True)\n"
            "args = parser.parse_args()\n"
            f"Path(r'{output}').write_text(args.train_file, encoding='utf-8')\n",
            encoding="utf-8",
        )

        ensure_project_runner(code_dir, "python train.py")
        result = subprocess.run(
            [sys.executable, RUNNER_SCRIPT_NAME],
            cwd=str(code_dir),
            capture_output=True,
            text=True,
            env=dict(os.environ),
            timeout=30,
        )

        assert result.returncode == 0, result.stderr
        assert output.exists()
        assert Path(output.read_text(encoding="utf-8")).resolve() == (data_dir / "train.tsv").resolve()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_project_runner_keeps_required_train_file_unset_when_ambiguous() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        code_dir = tmp_dir / "code"
        data_dir = code_dir / "data"
        data_dir.mkdir(parents=True)
        (data_dir / "train_a.csv").write_text("x,y\n1,2\n", encoding="utf-8")
        (data_dir / "train_b.csv").write_text("x,y\n3,4\n", encoding="utf-8")
        output = code_dir / "train_file_used.txt"
        (code_dir / "train.py").write_text(
            "import argparse\n"
            "from pathlib import Path\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument('--train-file', required=True)\n"
            "args = parser.parse_args()\n"
            f"Path(r'{output.resolve()}').write_text(args.train_file, encoding='utf-8')\n",
            encoding="utf-8",
        )

        ensure_project_runner(code_dir, "python train.py")
        result = subprocess.run(
            [sys.executable, RUNNER_SCRIPT_NAME],
            cwd=str(code_dir),
            capture_output=True,
            text=True,
            env=dict(os.environ),
            timeout=30,
        )

        assert result.returncode != 0
        assert not output.exists()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_project_runner_binds_required_image_and_annotation_inputs() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        code_dir = tmp_dir / "code"
        data_dir = code_dir / "data"
        images_dir = data_dir / "images"
        splits_dir = data_dir / "splits"
        images_dir.mkdir(parents=True)
        splits_dir.mkdir(parents=True)
        (images_dir / "sample.png").write_text("img", encoding="utf-8")
        annotations_file = data_dir / "annotations.json"
        annotations_file.write_text('{"images": []}\n', encoding="utf-8")
        split_file = splits_dir / "train.json"
        split_file.write_text('{"train": []}\n', encoding="utf-8")
        output = (code_dir / "bound_args.json").resolve()
        (code_dir / "train.py").write_text(
            "import argparse\n"
            "import json\n"
            "from pathlib import Path\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument('--image-dir', required=True)\n"
            "parser.add_argument('--annotations', required=True)\n"
            "parser.add_argument('--split-file', required=True)\n"
            "args = parser.parse_args()\n"
            f"Path(r'{output}').write_text(json.dumps(vars(args), sort_keys=True), encoding='utf-8')\n",
            encoding="utf-8",
        )

        ensure_project_runner(code_dir, "python train.py")
        result = subprocess.run(
            [sys.executable, RUNNER_SCRIPT_NAME],
            cwd=str(code_dir),
            capture_output=True,
            text=True,
            env=dict(os.environ),
            timeout=30,
        )

        assert result.returncode == 0, result.stderr
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert Path(payload["image_dir"]).resolve() == images_dir.resolve()
        assert Path(payload["annotations"]).resolve() == annotations_file.resolve()
        assert Path(payload["split_file"]).resolve() == split_file.resolve()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_project_runner_adds_required_resume_from_latest_checkpoint() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        code_dir = tmp_dir / "code"
        checkpoints_dir = code_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True)
        older_checkpoint = checkpoints_dir / "epoch_1.pt"
        latest_checkpoint = checkpoints_dir / "epoch_2.pt"
        older_checkpoint.write_text("older", encoding="utf-8")
        latest_checkpoint.write_text("latest", encoding="utf-8")
        os.utime(older_checkpoint, (1_700_000_000, 1_700_000_000))
        os.utime(latest_checkpoint, (1_800_000_000, 1_800_000_000))
        output = (code_dir / "resume_arg.txt").resolve()
        (code_dir / "train.py").write_text(
            "import argparse\n"
            "from pathlib import Path\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument('--resume', required=True)\n"
            "args = parser.parse_args()\n"
            f"Path(r'{output}').write_text(args.resume, encoding='utf-8')\n",
            encoding="utf-8",
        )

        ensure_project_runner(code_dir, "python train.py")
        result = subprocess.run(
            [sys.executable, RUNNER_SCRIPT_NAME],
            cwd=str(code_dir),
            capture_output=True,
            text=True,
            env=dict(os.environ),
            timeout=30,
        )

        assert result.returncode == 0, result.stderr
        assert Path(output.read_text(encoding="utf-8")).resolve() == latest_checkpoint.resolve()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_project_runner_respects_quick_eval_blocked_options() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        code_dir = tmp_dir / "code"
        code_dir.mkdir()
        output = (code_dir / "quick_eval_ok.txt").resolve()
        (code_dir / "train.py").write_text(
            "from pathlib import Path\n"
            "# --limit-train-batches appears in comments only\n"
            f"Path(r'{output}').write_text('ok', encoding='utf-8')\n",
            encoding="utf-8",
        )

        ensure_project_runner(code_dir, "python train.py")
        runner_config_path = code_dir / RUNNER_CONFIG_NAME
        runner_config = json.loads(runner_config_path.read_text(encoding="utf-8"))
        runner_config["quick_eval_blocked_options"] = ["--limit-train-batches"]
        runner_config_path.write_text(json.dumps(runner_config, indent=2), encoding="utf-8")

        result = subprocess.run(
            [sys.executable, RUNNER_SCRIPT_NAME, "--quick-eval"],
            cwd=str(code_dir),
            capture_output=True,
            text=True,
            env=dict(os.environ),
            timeout=30,
        )

        assert result.returncode == 0, result.stderr
        assert output.exists()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_project_runner_prefers_existing_yaml_config_for_yaml_entrypoint() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        code_dir = tmp_dir / "code"
        (code_dir / "config").mkdir(parents=True)
        yaml_path = code_dir / "config" / "default.yaml"
        yaml_path.write_text("random_seed: 7\n", encoding="utf-8")
        output = (code_dir / "config_used.txt").resolve()
        (code_dir / "train.py").write_text(
            "import argparse\n"
            "from pathlib import Path\n"
            "# yaml.safe_load expected here\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument('--config', required=True)\n"
            "args = parser.parse_args()\n"
            f"Path(r'{output}').write_text(args.config, encoding='utf-8')\n",
            encoding="utf-8",
        )

        ensure_project_runner(code_dir, "python train.py")
        result = subprocess.run(
            [sys.executable, RUNNER_SCRIPT_NAME],
            cwd=str(code_dir),
            capture_output=True,
            text=True,
            env=dict(os.environ),
            timeout=30,
        )

        assert result.returncode == 0, result.stderr
        assert output.exists()
        assert Path(output.read_text(encoding="utf-8")).resolve() == yaml_path.resolve()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_project_runner_materializes_json_config_from_python_module() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        code_dir = tmp_dir / "code"
        code_dir.mkdir()
        (code_dir / "config.py").write_text(
            "BATCH_SIZE = 8\n"
            "LEARNING_RATE = 0.001\n",
            encoding="utf-8",
        )
        output = (code_dir / "config_payload.json").resolve()
        (code_dir / "train.py").write_text(
            "import argparse\n"
            "import json\n"
            "from pathlib import Path\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument('--config', required=True)\n"
            "args = parser.parse_args()\n"
            "data = json.loads(Path(args.config).read_text(encoding='utf-8'))\n"
            f"Path(r'{output}').write_text(json.dumps(data, sort_keys=True), encoding='utf-8')\n",
            encoding="utf-8",
        )

        ensure_project_runner(code_dir, "python train.py")
        result = subprocess.run(
            [sys.executable, RUNNER_SCRIPT_NAME],
            cwd=str(code_dir),
            capture_output=True,
            text=True,
            env=dict(os.environ),
            timeout=30,
        )

        assert result.returncode == 0, result.stderr
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert payload["batch_size"] == 8
        assert payload["learning_rate"] == 0.001
        assert payload["random_seed"] == 42
        assert (code_dir / ".nanoresearch_autofix" / "config_auto.json").exists()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_run_local_dry_run_loop_retries_with_batch_fix() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="exec004",
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = ExecutionAgent(workspace, config)
        helper = ExperimentAgent(workspace, config)

        code_dir = workspace.path / "experiment"
        code_dir.mkdir(exist_ok=True)

        agent._run_subprocess = AsyncMock(
            side_effect=[
                {"returncode": 1, "stdout": "", "stderr": "ImportError: missing module"},
                {"returncode": 0, "stdout": "dry run ok", "stderr": ""},
            ]
        )
        helper._batch_fix_errors = AsyncMock(return_value=["train.py"])

        result = await agent._run_local_dry_run_loop(
            code_dir,
            ["python", "train.py"],
            '{"topic": "demo"}',
            helper,
        )

        assert result["status"] == "fixed"
        assert result["attempts"] == 2
        helper._batch_fix_errors.assert_awaited_once()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_run_local_dry_run_loop_uses_stdout_fallback_and_preflight_context() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="exec004b",
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = ExecutionAgent(workspace, config)
        helper = ExperimentAgent(workspace, config)

        code_dir = workspace.path / "experiment"
        code_dir.mkdir(exist_ok=True)
        (code_dir / "train.py").write_text("print('train')\n", encoding="utf-8")

        agent._run_subprocess = AsyncMock(
            side_effect=[
                {
                    "returncode": 1,
                    "stdout": "ValueError: malformed config payload",
                    "stderr": "",
                },
                {"returncode": 0, "stdout": "dry run ok", "stderr": ""},
            ]
        )
        helper._batch_fix_errors = AsyncMock(return_value=["train.py"])

        result = await agent._run_local_dry_run_loop(
            code_dir,
            ["python", "train.py"],
            '{"topic": "demo"}',
            helper,
        )

        assert result["status"] == "fixed"
        call = helper._batch_fix_errors.await_args
        assert call.args[1] == "ValueError: malformed config payload"
        assert "Suggested preflight fixes" in call.kwargs["extra_context"]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_run_local_dry_run_loop_dedupes_repeated_failure_history() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="exec004c",
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = ExecutionAgent(workspace, config)
        helper = ExperimentAgent(workspace, config)

        code_dir = workspace.path / "experiment"
        code_dir.mkdir(exist_ok=True)
        (code_dir / "train.py").write_text("print('train')\n", encoding="utf-8")

        repeated_error = {"returncode": 1, "stdout": "", "stderr": "ImportError: missing module"}
        agent._run_subprocess = AsyncMock(
            side_effect=[
                repeated_error,
                repeated_error,
                repeated_error,
                {"returncode": 0, "stdout": "dry run ok", "stderr": ""},
            ]
        )
        helper._batch_fix_errors = AsyncMock(return_value=["train.py"])

        result = await agent._run_local_dry_run_loop(
            code_dir,
            ["python", "train.py"],
            '{"topic": "demo"}',
            helper,
        )

        assert result["status"] == "fixed"
        assert helper._batch_fix_errors.await_count == 3
        third_call = helper._batch_fix_errors.await_args_list[2]
        previous_fixes = third_call.kwargs["previous_fixes"]
        assert len(previous_fixes) == 1
        assert previous_fixes[0]["repeat_count"] == 2
        assert "repeated 3 times" in third_call.kwargs["extra_context"]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_attempt_resource_path_repair_rewrites_stale_cache_path() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="exec004d",
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = ExecutionAgent(workspace, config)

        code_dir = workspace.path / "experiment"
        code_dir.mkdir(exist_ok=True)
        workspace_data = workspace.path / "data"
        workspace_data.mkdir(exist_ok=True)
        actual_dataset = workspace_data / "demo.csv"
        actual_dataset.write_text("x,y\n1,2\n", encoding="utf-8")
        stale_cache = tmp_dir / "cache" / "data"
        stale_cache.mkdir(parents=True)

        train_file = code_dir / "train.py"
        train_file.write_text(
            f"DATA_PATH = r'{stale_cache / 'demo.csv'}'\n",
            encoding="utf-8",
        )

        modified = agent._attempt_resource_path_repair(
            code_dir,
            f"FileNotFoundError: [Errno 2] No such file or directory: '{stale_cache / 'demo.csv'}'",
            {
                "data_dir": str(workspace_data),
                "cache_data_dir": str(stale_cache),
                "downloaded_resources": [
                    {
                        "name": "DemoSet",
                        "type": "dataset",
                        "status": "downloaded",
                        "path": str(actual_dataset),
                        "workspace_path": str(actual_dataset),
                    }
                ],
            },
        )

        assert modified == ["train.py"]
        assert str(actual_dataset) in train_file.read_text(encoding="utf-8")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_attempt_resource_path_repair_matches_same_stem_different_extension() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="exec004da",
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = ExecutionAgent(workspace, config)

        code_dir = workspace.path / "experiment"
        code_dir.mkdir(exist_ok=True)
        workspace_data = workspace.path / "data"
        workspace_data.mkdir(exist_ok=True)
        actual_dataset = workspace_data / "train.tsv"
        actual_dataset.write_text("x\ty\n1\t2\n", encoding="utf-8")

        train_file = code_dir / "train.py"
        missing_target = workspace_data / "train.csv"
        train_file.write_text(
            f"DATA_PATH = r'{missing_target}'\n",
            encoding="utf-8",
        )

        modified = agent._attempt_resource_path_repair(
            code_dir,
            f"FileNotFoundError: [Errno 2] No such file or directory: '{missing_target}'",
            {
                "data_dir": str(workspace_data),
                "downloaded_resources": [
                    {
                        "name": "TrainSet",
                        "type": "dataset",
                        "status": "downloaded",
                        "path": str(actual_dataset),
                        "workspace_path": str(actual_dataset),
                    }
                ],
            },
        )

        assert modified == ["train.py"]
        assert str(actual_dataset) in train_file.read_text(encoding="utf-8")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_attempt_resource_path_repair_materializes_gzip_dataset() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="exec004db",
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = ExecutionAgent(workspace, config)

        code_dir = workspace.path / "experiment"
        code_dir.mkdir(exist_ok=True)
        workspace_data = workspace.path / "data"
        workspace_data.mkdir(exist_ok=True)

        compressed_dataset = workspace_data / "train.csv.gz"
        with gzip.open(compressed_dataset, "wb") as handle:
            handle.write(b"x,y\n1,2\n")

        missing_target = workspace_data / "train.csv"
        modified = agent._attempt_resource_path_repair(
            code_dir,
            f"FileNotFoundError: [Errno 2] No such file or directory: '{missing_target}'",
            {
                "data_dir": str(workspace_data),
                "downloaded_resources": [
                    {
                        "name": "TrainSet",
                        "type": "dataset",
                        "status": "downloaded",
                        "path": str(compressed_dataset),
                        "workspace_path": str(compressed_dataset),
                    }
                ],
            },
        )

        assert str(missing_target) in modified
        assert missing_target.exists()
        assert missing_target.read_text(encoding="utf-8") == "x,y\n1,2\n"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_attempt_resource_path_repair_materializes_zip_dataset_member() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="exec004dc",
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = ExecutionAgent(workspace, config)

        code_dir = workspace.path / "experiment"
        code_dir.mkdir(exist_ok=True)
        workspace_data = workspace.path / "data"
        workspace_data.mkdir(exist_ok=True)

        compressed_dataset = workspace_data / "train_bundle.zip"
        with zipfile.ZipFile(compressed_dataset, "w") as archive:
            archive.writestr("nested/train.csv", "x,y\n1,2\n")

        missing_target = workspace_data / "train.csv"
        modified = agent._attempt_resource_path_repair(
            code_dir,
            f"FileNotFoundError: [Errno 2] No such file or directory: '{missing_target}'",
            {
                "data_dir": str(workspace_data),
                "downloaded_resources": [
                    {
                        "name": "TrainSet",
                        "type": "dataset",
                        "status": "downloaded",
                        "path": str(compressed_dataset),
                        "workspace_path": str(compressed_dataset),
                    }
                ],
            },
        )

        assert str(missing_target) in modified
        assert missing_target.exists()
        assert missing_target.read_text(encoding="utf-8") == "x,y\n1,2\n"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_attempt_resource_path_repair_rewrites_missing_config_path() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="exec004dd",
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = ExecutionAgent(workspace, config)

        code_dir = workspace.path / "experiment"
        code_dir.mkdir(exist_ok=True)
        (code_dir / "config").mkdir(exist_ok=True)
        actual_config = code_dir / "config" / "default.yaml"
        actual_config.write_text("seed: 1\n", encoding="utf-8")
        missing_config = code_dir / "config" / "missing.yaml"
        train_file = code_dir / "train.py"
        train_file.write_text(
            f"CONFIG_PATH = r'{missing_config}'\n",
            encoding="utf-8",
        )

        modified = agent._attempt_resource_path_repair(
            code_dir,
            f"FileNotFoundError: [Errno 2] No such file or directory: '{missing_config}'",
            {},
        )

        assert modified == ["train.py"]
        assert str(actual_config) in train_file.read_text(encoding="utf-8")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_attempt_required_argument_repair_updates_runner_config_for_missing_config_file() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="exec004de",
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = ExecutionAgent(workspace, config)

        code_dir = workspace.path / "experiment"
        code_dir.mkdir(exist_ok=True)
        (code_dir / "config").mkdir(exist_ok=True)
        actual_config = code_dir / "config" / "default.yaml"
        actual_config.write_text("seed: 1\n", encoding="utf-8")
        (code_dir / "train.py").write_text("print('ok')\n", encoding="utf-8")
        ensure_project_runner(code_dir, "python train.py")

        modified = agent._attempt_required_argument_repair(
            code_dir,
            "usage: train.py\ntrain.py: error: the following arguments are required: --config-file",
            {},
        )
        runner_config = json.loads((code_dir / RUNNER_CONFIG_NAME).read_text(encoding="utf-8"))

        assert modified == [RUNNER_CONFIG_NAME]
        assert "--config-file" in runner_config["target_command"]
        option_index = runner_config["target_command"].index("--config-file")
        assert Path(runner_config["target_command"][option_index + 1]).resolve() == actual_config.resolve()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_attempt_required_argument_repair_uses_local_data_layout_for_aliases() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="exec004dfa",
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = ExecutionAgent(workspace, config)

        code_dir = workspace.path / "experiment"
        code_dir.mkdir(exist_ok=True)
        data_dir = code_dir / "data"
        images_dir = data_dir / "images"
        splits_dir = data_dir / "splits"
        images_dir.mkdir(parents=True)
        splits_dir.mkdir(parents=True)
        (images_dir / "sample.png").write_text("img", encoding="utf-8")
        annotations_file = data_dir / "annotations.json"
        annotations_file.write_text('{"images": []}\n', encoding="utf-8")
        split_file = splits_dir / "fold0.json"
        split_file.write_text('{"train": []}\n', encoding="utf-8")
        (code_dir / "train.py").write_text("print('ok')\n", encoding="utf-8")
        ensure_project_runner(code_dir, "python train.py")

        modified = agent._attempt_required_argument_repair(
            code_dir,
            "usage: train.py\ntrain.py: error: the following arguments are required: --image-dir, --annotations, --split-file",
            {},
        )
        runner_config = json.loads((code_dir / RUNNER_CONFIG_NAME).read_text(encoding="utf-8"))

        assert modified == [RUNNER_CONFIG_NAME]
        image_index = runner_config["target_command"].index("--image-dir")
        annotations_index = runner_config["target_command"].index("--annotations")
        split_index = runner_config["target_command"].index("--split-file")
        assert Path(runner_config["target_command"][image_index + 1]).resolve() == images_dir.resolve()
        assert Path(runner_config["target_command"][annotations_index + 1]).resolve() == annotations_file.resolve()
        assert Path(runner_config["target_command"][split_index + 1]).resolve() == split_file.resolve()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_attempt_required_argument_repair_uses_latest_checkpoint_for_resume_flag() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="exec004dfb",
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = ExecutionAgent(workspace, config)

        code_dir = workspace.path / "experiment"
        code_dir.mkdir(exist_ok=True)
        checkpoints_dir = code_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True)
        older_checkpoint = checkpoints_dir / "epoch_1.pt"
        latest_checkpoint = checkpoints_dir / "epoch_2.pt"
        older_checkpoint.write_text("older", encoding="utf-8")
        latest_checkpoint.write_text("latest", encoding="utf-8")
        os.utime(older_checkpoint, (1_700_000_000, 1_700_000_000))
        os.utime(latest_checkpoint, (1_800_000_000, 1_800_000_000))
        (code_dir / "train.py").write_text("print('ok')\n", encoding="utf-8")
        ensure_project_runner(code_dir, "python train.py")

        modified = agent._attempt_required_argument_repair(
            code_dir,
            "usage: train.py\ntrain.py: error: argument --resume: expected one argument",
            {},
        )
        runner_config = json.loads((code_dir / RUNNER_CONFIG_NAME).read_text(encoding="utf-8"))

        assert modified == [RUNNER_CONFIG_NAME]
        resume_index = runner_config["target_command"].index("--resume")
        assert Path(runner_config["target_command"][resume_index + 1]).resolve() == latest_checkpoint.resolve()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_attempt_unrecognized_argument_repair_removes_unknown_option_from_runner_config() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="exec004dfc",
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = ExecutionAgent(workspace, config)

        code_dir = workspace.path / "experiment"
        code_dir.mkdir(exist_ok=True)
        (code_dir / "train.py").write_text("print('ok')\n", encoding="utf-8")
        ensure_project_runner(code_dir, "python train.py --ghost-flag 1")

        modified = agent._attempt_unrecognized_argument_repair(
            code_dir,
            "usage: train.py\ntrain.py: error: unrecognized arguments: --ghost-flag 1",
            mode="dry-run",
        )
        runner_config = json.loads((code_dir / RUNNER_CONFIG_NAME).read_text(encoding="utf-8"))

        assert modified == [RUNNER_CONFIG_NAME]
        assert "--ghost-flag" not in runner_config["target_command"]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_attempt_unrecognized_argument_repair_blocks_quick_eval_auto_option() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="exec004dfd",
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = ExecutionAgent(workspace, config)

        code_dir = workspace.path / "experiment"
        code_dir.mkdir(exist_ok=True)
        (code_dir / "train.py").write_text("print('ok')\n", encoding="utf-8")
        ensure_project_runner(code_dir, "python train.py")

        modified = agent._attempt_unrecognized_argument_repair(
            code_dir,
            "usage: train.py\ntrain.py: error: unrecognized arguments: --limit-train-batches 2",
            mode="quick-eval",
        )
        runner_config = json.loads((code_dir / RUNNER_CONFIG_NAME).read_text(encoding="utf-8"))

        assert modified == [RUNNER_CONFIG_NAME]
        assert "--limit-train-batches" in runner_config["quick_eval_blocked_options"]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_attempt_option_value_repair_replaces_stale_resume_path() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="exec004dfe",
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = ExecutionAgent(workspace, config)

        code_dir = workspace.path / "experiment"
        code_dir.mkdir(exist_ok=True)
        checkpoints_dir = code_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True)
        latest_checkpoint = checkpoints_dir / "latest.pt"
        latest_checkpoint.write_text("latest", encoding="utf-8")
        stale_checkpoint = code_dir / "cache" / "old.pt"
        (code_dir / "train.py").write_text("print('ok')\n", encoding="utf-8")
        ensure_project_runner(code_dir, f"python train.py --resume {stale_checkpoint}")

        modified = agent._attempt_option_value_repair(
            code_dir,
            f"FileNotFoundError: [Errno 2] No such file or directory: '{stale_checkpoint}'",
            {},
        )
        runner_config = json.loads((code_dir / RUNNER_CONFIG_NAME).read_text(encoding="utf-8"))

        assert modified == [RUNNER_CONFIG_NAME]
        resume_index = runner_config["target_command"].index("--resume")
        assert Path(runner_config["target_command"][resume_index + 1]).resolve() == latest_checkpoint.resolve()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_attempt_resume_repair_adds_resume_flag_from_latest_checkpoint() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="exec004dff",
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = ExecutionAgent(workspace, config)

        code_dir = workspace.path / "experiment"
        code_dir.mkdir(exist_ok=True)
        checkpoints_dir = code_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True)
        latest_checkpoint = checkpoints_dir / "latest.pt"
        latest_checkpoint.write_text("latest", encoding="utf-8")
        (code_dir / "train.py").write_text(
            "import argparse\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument('--resume')\n"
            "parser.parse_args()\n",
            encoding="utf-8",
        )
        ensure_project_runner(code_dir, "python train.py")

        modified = agent._attempt_resume_repair(
            code_dir,
            "Command timed out",
            {},
        )
        runner_config = json.loads((code_dir / RUNNER_CONFIG_NAME).read_text(encoding="utf-8"))

        assert modified == [RUNNER_CONFIG_NAME]
        resume_index = runner_config["target_command"].index("--resume")
        assert Path(runner_config["target_command"][resume_index + 1]).resolve() == latest_checkpoint.resolve()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_run_local_dry_run_loop_prefers_deterministic_resource_fix() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="exec004e",
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = ExecutionAgent(workspace, config)
        helper = ExperimentAgent(workspace, config)

        code_dir = workspace.path / "experiment"
        code_dir.mkdir(exist_ok=True)
        workspace_data = workspace.path / "data"
        workspace_data.mkdir(exist_ok=True)
        actual_dataset = workspace_data / "demo.csv"
        actual_dataset.write_text("x,y\n1,2\n", encoding="utf-8")
        stale_cache = tmp_dir / "cache" / "data"
        stale_cache.mkdir(parents=True)
        (code_dir / "train.py").write_text(
            f"DATA_PATH = r'{stale_cache / 'demo.csv'}'\n",
            encoding="utf-8",
        )

        agent._run_subprocess = AsyncMock(
            side_effect=[
                {
                    "returncode": 1,
                    "stdout": "",
                    "stderr": f"FileNotFoundError: [Errno 2] No such file or directory: '{stale_cache / 'demo.csv'}'",
                },
                {"returncode": 0, "stdout": "dry run ok", "stderr": ""},
            ]
        )
        helper._batch_fix_errors = AsyncMock(return_value=["train.py"])
        remediation_ledger: list[dict] = []

        result = await agent._run_local_dry_run_loop(
            code_dir,
            ["python", "train.py"],
            '{"topic": "demo"}',
            helper,
            remediation_ledger=remediation_ledger,
            resource_context={
                "data_dir": str(workspace_data),
                "cache_data_dir": str(stale_cache),
                "downloaded_resources": [
                    {
                        "name": "DemoSet",
                        "type": "dataset",
                        "status": "downloaded",
                        "path": str(actual_dataset),
                        "workspace_path": str(actual_dataset),
                    }
                ],
            },
        )

        assert result["status"] == "fixed"
        assert result["attempts"] == 2
        helper._batch_fix_errors.assert_not_awaited()
        assert str(actual_dataset) in (code_dir / "train.py").read_text(encoding="utf-8")
        assert any(
            entry["kind"] == "resource_path_repair"
            and entry["status"] == "applied"
            and "train.py" in entry.get("files", [])
            for entry in remediation_ledger
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_run_local_dry_run_loop_prefers_required_argument_repair_before_llm_fix() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="exec004ee",
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = ExecutionAgent(workspace, config)
        helper = ExperimentAgent(workspace, config)

        code_dir = workspace.path / "experiment"
        code_dir.mkdir(exist_ok=True)
        (code_dir / "config").mkdir(exist_ok=True)
        actual_config = code_dir / "config" / "default.yaml"
        actual_config.write_text("seed: 1\n", encoding="utf-8")
        (code_dir / "train.py").write_text("print('ok')\n", encoding="utf-8")
        ensure_project_runner(code_dir, "python train.py")

        agent._run_subprocess = AsyncMock(
            side_effect=[
                {
                    "returncode": 2,
                    "stdout": "",
                    "stderr": "usage: train.py\ntrain.py: error: the following arguments are required: --config-file",
                },
                {"returncode": 0, "stdout": "dry run ok", "stderr": ""},
            ]
        )
        helper._batch_fix_errors = AsyncMock(return_value=["train.py"])
        remediation_ledger: list[dict] = []

        result = await agent._run_local_dry_run_loop(
            code_dir,
            [sys.executable, RUNNER_SCRIPT_NAME],
            '{"topic": "demo"}',
            helper,
            remediation_ledger=remediation_ledger,
            resource_context={},
        )
        runner_config = json.loads((code_dir / RUNNER_CONFIG_NAME).read_text(encoding="utf-8"))

        assert result["status"] == "fixed"
        helper._batch_fix_errors.assert_not_awaited()
        option_index = runner_config["target_command"].index("--config-file")
        assert Path(runner_config["target_command"][option_index + 1]).resolve() == actual_config.resolve()
        assert any(
            entry["kind"] == "required_argument_repair"
            and entry["status"] == "applied"
            for entry in remediation_ledger
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_run_local_dry_run_loop_prefers_option_value_repair_before_llm_fix() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="exec004eeb",
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = ExecutionAgent(workspace, config)
        helper = ExperimentAgent(workspace, config)

        code_dir = workspace.path / "experiment"
        code_dir.mkdir(exist_ok=True)
        checkpoints_dir = code_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True)
        latest_checkpoint = checkpoints_dir / "latest.pt"
        latest_checkpoint.write_text("latest", encoding="utf-8")
        stale_checkpoint = code_dir / "cache" / "old.pt"
        (code_dir / "train.py").write_text("print('ok')\n", encoding="utf-8")
        ensure_project_runner(code_dir, f"python train.py --resume {stale_checkpoint}")

        agent._run_subprocess = AsyncMock(
            side_effect=[
                {
                    "returncode": 1,
                    "stdout": "",
                    "stderr": f"FileNotFoundError: [Errno 2] No such file or directory: '{stale_checkpoint}'",
                },
                {"returncode": 0, "stdout": "dry run ok", "stderr": ""},
            ]
        )
        helper._batch_fix_errors = AsyncMock(return_value=["train.py"])
        remediation_ledger: list[dict] = []

        result = await agent._run_local_dry_run_loop(
            code_dir,
            [sys.executable, RUNNER_SCRIPT_NAME],
            '{"topic": "demo"}',
            helper,
            remediation_ledger=remediation_ledger,
            resource_context={},
        )
        runner_config = json.loads((code_dir / RUNNER_CONFIG_NAME).read_text(encoding="utf-8"))

        assert result["status"] == "fixed"
        helper._batch_fix_errors.assert_not_awaited()
        resume_index = runner_config["target_command"].index("--resume")
        assert Path(runner_config["target_command"][resume_index + 1]).resolve() == latest_checkpoint.resolve()
        assert any(
            entry["kind"] == "option_value_repair"
            and entry["status"] == "applied"
            for entry in remediation_ledger
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_run_local_quick_eval_loop_prefers_resume_repair_before_timeout_fix() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="exec004eec",
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = ExecutionAgent(workspace, config)
        helper = ExperimentAgent(workspace, config)

        code_dir = workspace.path / "experiment"
        code_dir.mkdir(exist_ok=True)
        results_dir = code_dir / "results"
        checkpoints_dir = code_dir / "checkpoints"
        results_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        latest_checkpoint = checkpoints_dir / "latest.pt"
        latest_checkpoint.write_text("latest", encoding="utf-8")
        (code_dir / "train.py").write_text(
            "import argparse\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument('--resume')\n"
            "parser.parse_args()\n",
            encoding="utf-8",
        )
        ensure_project_runner(code_dir, "python train.py")

        async def fake_run_subprocess(*_args, **_kwargs):
            if not hasattr(fake_run_subprocess, "calls"):
                fake_run_subprocess.calls = 0
            fake_run_subprocess.calls += 1
            if fake_run_subprocess.calls == 1:
                return {"returncode": -1, "stdout": "", "stderr": "Command timed out"}
            (results_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "main_results": [
                            {
                                "method_name": "DeepMethod",
                                "dataset": "DemoSet",
                                "is_proposed": True,
                                "metrics": [{"metric_name": "accuracy", "value": 0.91}],
                            }
                        ],
                        "ablation_results": [],
                        "training_log": [],
                    }
                ),
                encoding="utf-8",
            )
            return {"returncode": 0, "stdout": "quick eval ok", "stderr": ""}

        agent._run_subprocess = AsyncMock(side_effect=fake_run_subprocess)
        helper._fix_timeout = AsyncMock(return_value=["main.py"])
        remediation_ledger: list[dict] = []

        result = await agent._run_local_quick_eval_loop(
            code_dir,
            [sys.executable, RUNNER_SCRIPT_NAME],
            '{"topic": "demo"}',
            helper,
            remediation_ledger=remediation_ledger,
            resource_context={},
        )
        runner_config = json.loads((code_dir / RUNNER_CONFIG_NAME).read_text(encoding="utf-8"))

        assert result["status"] == "success"
        helper._fix_timeout.assert_not_awaited()
        resume_index = runner_config["target_command"].index("--resume")
        assert Path(runner_config["target_command"][resume_index + 1]).resolve() == latest_checkpoint.resolve()
        assert any(
            entry["kind"] == "resume_repair"
            and entry["status"] == "applied"
            for entry in remediation_ledger
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_run_local_quick_eval_loop_prefers_unrecognized_argument_repair_before_llm_fix() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="exec004ef",
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = ExecutionAgent(workspace, config)
        helper = ExperimentAgent(workspace, config)

        code_dir = workspace.path / "experiment"
        code_dir.mkdir(exist_ok=True)
        (code_dir / "train.py").write_text("print('ok')\n", encoding="utf-8")
        ensure_project_runner(code_dir, "python train.py")

        agent._run_subprocess = AsyncMock(
            side_effect=[
                {
                    "returncode": 2,
                    "stdout": "",
                    "stderr": "usage: train.py\ntrain.py: error: unrecognized arguments: --limit-train-batches 2",
                },
                {"returncode": 0, "stdout": "quick eval ok", "stderr": ""},
            ]
        )
        helper._batch_fix_errors = AsyncMock(return_value=["train.py"])
        remediation_ledger: list[dict] = []

        result = await agent._run_local_quick_eval_loop(
            code_dir,
            [sys.executable, RUNNER_SCRIPT_NAME],
            '{"topic": "demo"}',
            helper,
            remediation_ledger=remediation_ledger,
            resource_context={},
        )
        runner_config = json.loads((code_dir / RUNNER_CONFIG_NAME).read_text(encoding="utf-8"))

        assert result["status"] == "partial"
        helper._batch_fix_errors.assert_not_awaited()
        assert "--limit-train-batches" in runner_config["quick_eval_blocked_options"]
        assert any(
            entry["kind"] == "unrecognized_argument_repair"
            and entry["status"] == "applied"
            for entry in remediation_ledger
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_run_local_dry_run_loop_auto_installs_missing_package() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="exec004f",
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = ExecutionAgent(workspace, config)
        helper = ExperimentAgent(workspace, config)

        code_dir = workspace.path / "experiment"
        code_dir.mkdir(exist_ok=True)
        (code_dir / "requirements.txt").write_text("demo_pkg\n", encoding="utf-8")
        commands: list[list[str]] = []

        async def fake_run_subprocess(command: list[str], **_kwargs):
            commands.append(command)
            if len(commands) == 1:
                return {
                    "returncode": 1,
                    "stdout": "",
                    "stderr": "ModuleNotFoundError: No module named 'demo_pkg'",
                }
            if command[:4] == ["python-custom", "-m", "pip", "install"]:
                return {"returncode": 0, "stdout": "installed", "stderr": ""}
            return {"returncode": 0, "stdout": "dry run ok", "stderr": ""}

        agent._run_subprocess = AsyncMock(side_effect=fake_run_subprocess)
        helper._batch_fix_errors = AsyncMock(return_value=["train.py"])
        remediation_ledger: list[dict] = []

        result = await agent._run_local_dry_run_loop(
            code_dir,
            ["python", "train.py"],
            '{"topic": "demo"}',
            helper,
            runtime_python="python-custom",
            remediation_ledger=remediation_ledger,
        )

        assert result["status"] == "fixed"
        assert result["attempts"] == 2
        assert ["python-custom", "-m", "pip", "install", "demo_pkg"] in commands
        helper._batch_fix_errors.assert_not_awaited()
        assert any(
            entry["kind"] == "pip_install"
            and entry["status"] == "applied"
            and entry.get("details", {}).get("package") == "demo_pkg"
            for entry in remediation_ledger
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_run_local_dry_run_loop_skips_undeclared_package_auto_install() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="exec004g",
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = ExecutionAgent(workspace, config)
        helper = ExperimentAgent(workspace, config)

        code_dir = workspace.path / "experiment"
        code_dir.mkdir(exist_ok=True)
        commands: list[list[str]] = []

        async def fake_run_subprocess(command: list[str], **_kwargs):
            commands.append(command)
            if len(commands) == 1:
                return {
                    "returncode": 1,
                    "stdout": "",
                    "stderr": "ModuleNotFoundError: No module named 'unknown_pkg'",
                }
            return {"returncode": 0, "stdout": "dry run ok", "stderr": ""}

        agent._run_subprocess = AsyncMock(side_effect=fake_run_subprocess)
        helper._batch_fix_errors = AsyncMock(return_value=["train.py"])
        remediation_ledger: list[dict] = []

        result = await agent._run_local_dry_run_loop(
            code_dir,
            ["python", "train.py"],
            '{"topic": "demo"}',
            helper,
            runtime_python="python-custom",
            remediation_ledger=remediation_ledger,
        )

        assert result["status"] == "fixed"
        assert all(command[:4] != ["python-custom", "-m", "pip", "install"] for command in commands)
        helper._batch_fix_errors.assert_awaited_once()
        assert any(
            entry["kind"] == "pip_install"
            and entry["status"] == "skipped"
            and entry.get("reason") == "not_declared_or_allowlisted"
            for entry in remediation_ledger
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_run_local_dry_run_loop_honors_runtime_auto_install_budget() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="exec004h",
        )
        config = ResearchConfig(
            base_url="http://localhost:8000/v1/",
            api_key="test-key",
            runtime_auto_install_max_packages=0,
        )
        agent = ExecutionAgent(workspace, config)
        helper = ExperimentAgent(workspace, config)

        code_dir = workspace.path / "experiment"
        code_dir.mkdir(exist_ok=True)
        (code_dir / "requirements.txt").write_text("demo_pkg\n", encoding="utf-8")
        commands: list[list[str]] = []

        async def fake_run_subprocess(command: list[str], **_kwargs):
            commands.append(command)
            if len(commands) == 1:
                return {
                    "returncode": 1,
                    "stdout": "",
                    "stderr": "ModuleNotFoundError: No module named 'demo_pkg'",
                }
            return {"returncode": 0, "stdout": "dry run ok", "stderr": ""}

        agent._run_subprocess = AsyncMock(side_effect=fake_run_subprocess)
        helper._batch_fix_errors = AsyncMock(return_value=["train.py"])

        result = await agent._run_local_dry_run_loop(
            code_dir,
            ["python", "train.py"],
            '{"topic": "demo"}',
            helper,
            runtime_python="python-custom",
        )

        assert result["status"] == "fixed"
        assert all(command[:4] != ["python-custom", "-m", "pip", "install"] for command in commands)
        helper._batch_fix_errors.assert_awaited_once()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_run_local_quick_eval_loop_auto_downloads_nltk_resource() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="exec005c",
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = ExecutionAgent(workspace, config)
        helper = ExperimentAgent(workspace, config)

        code_dir = workspace.path / "experiment"
        code_dir.mkdir(exist_ok=True)
        commands: list[list[str]] = []

        async def fake_run_subprocess(command: list[str], **_kwargs):
            commands.append(command)
            if len(commands) == 1:
                return {
                    "returncode": 1,
                    "stdout": "",
                    "stderr": (
                        "LookupError:\n"
                        "Resource punkt_tab not found.\n"
                        "Please use the NLTK Downloader to obtain the resource:\n"
                        ">>> import nltk\n"
                        ">>> nltk.download('punkt_tab')\n"
                    ),
                }
            if command[:2] == ["python-custom", "-c"]:
                return {"returncode": 0, "stdout": "downloaded", "stderr": ""}
            return {
                "returncode": 0,
                "stdout": "Test accuracy: 0.91\n",
                "stderr": "",
            }

        agent._run_subprocess = AsyncMock(side_effect=fake_run_subprocess)
        helper._batch_fix_errors = AsyncMock(return_value=["train.py"])
        remediation_ledger: list[dict] = []

        result = await agent._run_local_quick_eval_loop(
            code_dir,
            ["python", "train.py"],
            '{"topic": "demo"}',
            helper,
            runtime_python="python-custom",
            remediation_ledger=remediation_ledger,
        )

        assert result["status"] == "partial"
        assert result["recovered_from"] == "execution_log"
        assert any(
            command[:2] == ["python-custom", "-c"] and "nltk.download('punkt_tab'" in command[2]
            for command in commands
        )
        helper._batch_fix_errors.assert_not_awaited()
        assert any(
            entry["kind"] == "nltk_download"
            and entry["status"] == "applied"
            and entry.get("details", {}).get("resource") == "punkt_tab"
            for entry in remediation_ledger
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_run_local_quick_eval_loop_recovers_after_timeout_fix() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="exec005",
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = ExecutionAgent(workspace, config)
        helper = ExperimentAgent(workspace, config)

        code_dir = workspace.path / "experiment"
        results_dir = code_dir / "results"
        code_dir.mkdir(exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        async def fake_run_subprocess(*_args, **_kwargs):
            if not hasattr(fake_run_subprocess, "calls"):
                fake_run_subprocess.calls = 0
            fake_run_subprocess.calls += 1
            if fake_run_subprocess.calls == 1:
                return {"returncode": -1, "stdout": "", "stderr": "Command timed out"}
            (results_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "main_results": [
                            {
                                "method_name": "DeepMethod",
                                "dataset": "DemoSet",
                                "is_proposed": True,
                                "metrics": [{"metric_name": "accuracy", "value": 0.91}],
                            }
                        ],
                        "ablation_results": [],
                        "training_log": [],
                    }
                ),
                encoding="utf-8",
            )
            return {"returncode": 0, "stdout": "quick eval ok", "stderr": ""}

        agent._run_subprocess = AsyncMock(side_effect=fake_run_subprocess)
        helper._fix_timeout = AsyncMock(return_value=["main.py"])
        remediation_ledger: list[dict] = []

        result = await agent._run_local_quick_eval_loop(
            code_dir,
            ["python", "train.py"],
            '{"topic": "demo"}',
            helper,
            remediation_ledger=remediation_ledger,
        )

        assert result["status"] == "success"
        assert result["attempts"] == 2
        assert result["metrics"]["main_results"][0]["method_name"] == "DeepMethod"
        helper._fix_timeout.assert_awaited_once()
        assert any(
            entry["kind"] == "timeout_fix"
            and entry["status"] == "applied"
            and "main.py" in entry.get("files", [])
            for entry in remediation_ledger
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_run_local_quick_eval_loop_prefers_training_log_artifact_before_timeout_fix() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="exec005d",
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = ExecutionAgent(workspace, config)
        helper = ExperimentAgent(workspace, config)

        code_dir = workspace.path / "experiment"
        results_dir = code_dir / "results"
        code_dir.mkdir(exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        async def fake_run_subprocess(*_args, **_kwargs):
            (results_dir / "training_log.csv").write_text(
                "epoch,loss,Accuracy,F1,lr\n"
                "1,0.50,0.75,0.71,0.001\n"
                "2,0.40,0.84,0.82,0.0009\n",
                encoding="utf-8",
            )
            return {"returncode": -1, "stdout": "", "stderr": "Command timed out"}

        agent._run_subprocess = AsyncMock(side_effect=fake_run_subprocess)
        helper._fix_timeout = AsyncMock(return_value=["main.py"])
        remediation_ledger: list[dict] = []

        result = await agent._run_local_quick_eval_loop(
            code_dir,
            ["python", "train.py"],
            '{"topic": "demo"}',
            helper,
            remediation_ledger=remediation_ledger,
        )

        assert result["status"] == "partial"
        assert result["attempts"] == 1
        assert result["recovered_from"] == "training_log_csv"
        assert result["metrics_artifact_materialized"] is True
        metric_values = {
            metric["metric_name"]: metric["value"]
            for metric in result["metrics"]["main_results"][0]["metrics"]
        }
        assert metric_values["Accuracy"] == 0.84
        assert metric_values["F1"] == 0.82
        helper._fix_timeout.assert_not_awaited()
        assert not any(entry["kind"] == "timeout_fix" for entry in remediation_ledger)
        assert any(
            entry["kind"] == "metrics_recovery"
            and entry["details"].get("source") == "training_log_csv"
            for entry in remediation_ledger
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_run_local_quick_eval_loop_recovers_metrics_from_stdout_when_json_missing() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="exec005b",
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = ExecutionAgent(workspace, config)
        helper = ExperimentAgent(workspace, config)

        code_dir = workspace.path / "experiment"
        code_dir.mkdir(exist_ok=True)

        agent._run_subprocess = AsyncMock(
            return_value={
                "returncode": 0,
                "stdout": "Epoch 1: loss=0.5\nTest accuracy: 0.91\nF1: 0.88\n",
                "stderr": "",
            }
        )

        result = await agent._run_local_quick_eval_loop(
            code_dir,
            ["python", "train.py"],
            '{"topic": "demo"}',
            helper,
        )

        assert result["status"] == "partial"
        assert result["recovered_from"] == "execution_log"
        assert result["metrics_artifact_materialized"] is True
        assert result["metrics_artifact_path"] == "results/metrics.json"
        metric_names = {metric["metric_name"] for metric in result["metrics"]["main_results"][0]["metrics"]}
        assert {"accuracy", "F1"} <= metric_names or {"Test accuracy", "F1"} <= metric_names
        assert result["metrics"]["training_log"][0]["epoch"] == 1
        assert result["metrics"]["training_log"][0]["train_loss"] == 0.5

        metrics_path = code_dir / "results" / "metrics.json"
        assert metrics_path.exists()
        persisted = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert persisted["_nanoresearch_meta"]["recovered_from"] == "execution_log"
        assert persisted["training_log"][0]["epoch"] == 1
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_run_local_quick_eval_loop_uses_training_log_csv_when_metrics_json_missing() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="exec005c",
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = ExecutionAgent(workspace, config)
        helper = ExperimentAgent(workspace, config)

        code_dir = workspace.path / "experiment"
        (code_dir / "results").mkdir(parents=True, exist_ok=True)
        (code_dir / "results" / "training_log.csv").write_text(
            "epoch,loss,Accuracy,F1,lr\n"
            "1,0.50,0.75,0.71,0.001\n"
            "2,0.40,0.84,0.82,0.0009\n",
            encoding="utf-8",
        )

        agent._run_subprocess = AsyncMock(
            return_value={
                "returncode": 0,
                "stdout": "",
                "stderr": "",
            }
        )

        result = await agent._run_local_quick_eval_loop(
            code_dir,
            ["python", "train.py"],
            '{"topic": "demo"}',
            helper,
        )

        assert result["status"] == "partial"
        assert result["recovered_from"] == "training_log_csv"
        assert result["metrics_artifact_materialized"] is True
        assert result["metrics_artifact_path"] == "results/metrics.json"
        metric_values = {
            metric["metric_name"]: metric["value"]
            for metric in result["metrics"]["main_results"][0]["metrics"]
        }
        assert metric_values["Accuracy"] == 0.84
        assert metric_values["F1"] == 0.82
        assert "lr" not in metric_values
        assert result["metrics"]["training_log"][-1]["epoch"] == 2
        assert result["metrics"]["training_log"][-1]["train_loss"] == 0.4

        metrics_path = code_dir / "results" / "metrics.json"
        assert metrics_path.exists()
        persisted = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert persisted["_nanoresearch_meta"]["recovered_from"] == "training_log_csv"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
