"""Tests for cluster/local experiment execution consistency."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import AsyncMock
import uuid

import pytest

import nanoresearch.agents.execution as execution_module
from nanoresearch.agents.cluster_executor import ClusterExecutor
from nanoresearch.agents.execution import ExecutionAgent
from nanoresearch.agents.project_runner import RUNNER_CONFIG_NAME, ensure_project_runner
from nanoresearch.config import ResearchConfig
from nanoresearch.pipeline.workspace import Workspace


@pytest.mark.asyncio
async def test_prepare_code_local_mode_ensures_checkpoints_dir() -> None:
    tmp_dir = Path(f".test_cluster_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        source_dir = tmp_dir / "source"
        source_dir.mkdir()
        (source_dir / "main.py").write_text("print('ok')\n", encoding="utf-8")
        (source_dir / "train.py").write_text("print('ok')\n", encoding="utf-8")
        base_path = tmp_dir / "cluster"

        executor = ClusterExecutor(
            {"local": True, "code_path": str(base_path)},
        )

        cluster_code_path = await executor.prepare_code(source_dir, "sess001")
        cluster_dir = Path(cluster_code_path)

        assert (cluster_dir / "results").exists()
        assert (cluster_dir / "checkpoints").exists()
        assert (cluster_dir / "logs").exists()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_submit_job_rejects_invalid_launch_target_before_sbatch() -> None:
    tmp_dir = Path(f".test_cluster_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        source_dir = tmp_dir / "source"
        source_dir.mkdir()
        (source_dir / "main.py").write_text("print('ok')\n", encoding="utf-8")
        (source_dir / "train.py").write_text("print('ok')\n", encoding="utf-8")
        base_path = tmp_dir / "cluster"

        executor = ClusterExecutor(
            {"local": True, "code_path": str(base_path)},
        )
        cluster_code_path = await executor.prepare_code(source_dir, "sess_bad_launch")

        async def unexpected_run(_cmd: str, timeout: int = 0) -> dict:
            del timeout
            raise AssertionError("submit_job should fail before invoking sbatch")

        executor._run_cmd = unexpected_run  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="Launch contract failed"):
            await executor.submit_job(cluster_code_path, "python missing_train.py --quick-eval")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_submit_job_repairs_stale_runner_target_before_sbatch() -> None:
    tmp_dir = Path(f".test_cluster_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        source_dir = tmp_dir / "source"
        source_dir.mkdir()
        (source_dir / "train.py").write_text("print('ok')\n", encoding="utf-8")
        (source_dir / "nanoresearch_runner.py").write_text("print('runner')\n", encoding="utf-8")
        (source_dir / "nanoresearch_runner.json").write_text(
            '{"target_command":["missing_train.py"],"target_env":{}}',
            encoding="utf-8",
        )
        base_path = tmp_dir / "cluster"

        executor = ClusterExecutor(
            {"local": True, "code_path": str(base_path)},
        )
        cluster_code_path = await executor.prepare_code(source_dir, "sess_repair_launch")

        commands: list[str] = []

        async def fake_run(cmd: str, timeout: int = 0) -> dict:
            del timeout
            commands.append(cmd)
            if cmd.startswith("sbatch "):
                return {"returncode": 0, "stdout": "Submitted batch job 12345\n", "stderr": ""}
            raise AssertionError(f"Unexpected command: {cmd}")

        executor._run_cmd = fake_run  # type: ignore[method-assign]

        job_id = await executor.submit_job(cluster_code_path, "python nanoresearch_runner.py --quick-eval")
        repaired_config = (Path(cluster_code_path) / "nanoresearch_runner.json").read_text(encoding="utf-8")

        assert job_id == "12345"
        assert '"train.py"' in repaired_config
        assert any(cmd.startswith("sbatch ") for cmd in commands)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_download_results_local_mode_copies_all_artifacts() -> None:
    tmp_dir = Path(f".test_cluster_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        cluster_code_dir = tmp_dir / "cluster_code"
        (cluster_code_dir / "results").mkdir(parents=True)
        (cluster_code_dir / "checkpoints").mkdir()
        (cluster_code_dir / "logs").mkdir()
        (cluster_code_dir / "results" / "metrics.json").write_text(
            '{"accuracy": 0.91}',
            encoding="utf-8",
        )
        (cluster_code_dir / "results" / "training_log.csv").write_text(
            "epoch,loss\n1,0.5\n",
            encoding="utf-8",
        )
        (cluster_code_dir / "checkpoints" / "model.pt").write_text(
            "checkpoint",
            encoding="utf-8",
        )
        (cluster_code_dir / "logs" / "123.log").write_text(
            "done",
            encoding="utf-8",
        )

        workspace_dir = tmp_dir / "workspace"
        workspace_dir.mkdir()

        executor = ClusterExecutor({"local": True})
        copied = await executor.download_results(str(cluster_code_dir), workspace_dir)

        assert copied is True
        assert (workspace_dir / "code" / "results" / "metrics.json").read_text(encoding="utf-8") == '{"accuracy": 0.91}'
        assert (workspace_dir / "code" / "results" / "training_log.csv").exists()
        assert (workspace_dir / "code" / "checkpoints" / "model.pt").exists()
        assert (workspace_dir / "code" / "logs" / "123.log").exists()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_setup_env_creates_from_environment_yaml_and_installs_pyproject() -> None:
    commands: list[str] = []
    executor = ClusterExecutor(
        {
            "local": True,
            "conda_env": "nano_test",
            "python_version": "3.11",
        }
    )

    async def fake_run(cmd: str, timeout: int = 0) -> dict:
        del timeout
        commands.append(cmd)
        if "CONDA_SH=" in cmd:
            return {"returncode": 0, "stdout": "/opt/conda/etc/profile.d/conda.sh\n", "stderr": ""}
        if 'echo "pyproject.toml"' in cmd:
            return {"returncode": 0, "stdout": "pyproject.toml\nenvironment.yaml\n", "stderr": ""}
        if "conda env list" in cmd:
            return {"returncode": 0, "stdout": "ENV_MISSING\n", "stderr": ""}
        if "conda env create" in cmd:
            return {"returncode": 0, "stdout": "created\n", "stderr": ""}
        if "pip install -e" in cmd:
            return {"returncode": 0, "stdout": "installed\n", "stderr": ""}
        if "python -c" in cmd and "sys.version" in cmd:
            return {
                "returncode": 0,
                "stdout": '{"executable":"/opt/conda/envs/nano_test/bin/python","version":"3.11.9"}\n',
                "stderr": "",
            }
        if "python -m pip --version" in cmd:
            return {
                "returncode": 0,
                "stdout": "pip 24.0 from /opt/conda/envs/nano_test/lib/python3.11/site-packages/pip\n",
                "stderr": "",
            }
        raise AssertionError(f"Unexpected command: {cmd}")

    executor._run_cmd = fake_run  # type: ignore[method-assign]

    result = await executor.setup_env("/cluster/code")

    assert result["ok"] is True
    assert result["source"] == "pyproject.toml"
    assert result["strategy"] == "editable"
    assert result["manifest"] == "/cluster/code/pyproject.toml"
    assert result["runtime_validation"]["status"] == "ready"
    assert result["runtime_validation_repair"]["status"] == "skipped"
    assert any("conda env create -n nano_test -f /cluster/code/environment.yaml" in cmd for cmd in commands)
    assert any("pip install -e /cluster/code" in cmd for cmd in commands)


@pytest.mark.asyncio
async def test_setup_env_uses_environment_manifest_when_no_pip_manifest() -> None:
    commands: list[str] = []
    executor = ClusterExecutor({"local": True, "conda_env": "nano_test"})

    async def fake_run(cmd: str, timeout: int = 0) -> dict:
        del timeout
        commands.append(cmd)
        if "CONDA_SH=" in cmd:
            return {"returncode": 0, "stdout": "/opt/conda/etc/profile.d/conda.sh\n", "stderr": ""}
        if 'echo "requirements.txt"' in cmd:
            return {"returncode": 0, "stdout": "environment.yml\n", "stderr": ""}
        if "conda env list" in cmd:
            return {"returncode": 0, "stdout": "ENV_EXISTS\n", "stderr": ""}
        if "conda env update" in cmd:
            return {"returncode": 0, "stdout": "updated\n", "stderr": ""}
        if "python -c" in cmd and "sys.version" in cmd:
            return {
                "returncode": 0,
                "stdout": '{"executable":"/opt/conda/envs/nano_test/bin/python","version":"3.10.14"}\n',
                "stderr": "",
            }
        if "python -m pip --version" in cmd:
            return {
                "returncode": 0,
                "stdout": "pip 24.0 from /opt/conda/envs/nano_test/lib/python3.10/site-packages/pip\n",
                "stderr": "",
            }
        raise AssertionError(f"Unexpected command: {cmd}")

    executor._run_cmd = fake_run  # type: ignore[method-assign]

    result = await executor.setup_env("/cluster/code")

    assert result["ok"] is True
    assert result["source"] == "environment.yml"
    assert result["strategy"] == "conda_env_update"
    assert result["manifest"] == "/cluster/code/environment.yml"
    assert result["runtime_validation"]["status"] == "ready"
    assert any("conda env update -n nano_test -f /cluster/code/environment.yml --prune" in cmd for cmd in commands)
    assert not any("pip install" in cmd for cmd in commands)


@pytest.mark.asyncio
async def test_setup_env_prefers_cached_local_manifest_policy_over_stale_remote_probe() -> None:
    tmp_dir = Path(f".test_cluster_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        source_dir = tmp_dir / "source"
        source_dir.mkdir()
        (source_dir / "main.py").write_text("print('ok')\n", encoding="utf-8")
        (source_dir / "pyproject.toml").write_text(
            "\n".join(
                [
                    "[build-system]",
                    'requires = ["setuptools>=61"]',
                    'build-backend = "setuptools.build_meta"',
                    "",
                    "[project]",
                    'name = "demo-project"',
                    'version = "0.1.0"',
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (source_dir / "environment.yaml").write_text(
            "name: demo\ndependencies:\n  - python=3.10\n",
            encoding="utf-8",
        )
        base_path = tmp_dir / "cluster"

        executor = ClusterExecutor(
            {"local": True, "code_path": str(base_path), "conda_env": "nano_test"}
        )
        cluster_code_path = await executor.prepare_code(source_dir, "sess_policy")

        commands: list[str] = []

        async def fake_run(cmd: str, timeout: int = 0) -> dict:
            del timeout
            commands.append(cmd)
            if "CONDA_SH=" in cmd:
                return {"returncode": 0, "stdout": "/opt/conda/etc/profile.d/conda.sh\n", "stderr": ""}
            if 'echo "requirements.txt"' in cmd:
                return {
                    "returncode": 0,
                    "stdout": "requirements.txt\npyproject.toml\nenvironment.yaml\n",
                    "stderr": "",
                }
            if "conda env list" in cmd:
                return {"returncode": 0, "stdout": "ENV_EXISTS\n", "stderr": ""}
            if "conda env update" in cmd:
                return {"returncode": 0, "stdout": "updated\n", "stderr": ""}
            if "pip install -e" in cmd:
                return {"returncode": 0, "stdout": "installed\n", "stderr": ""}
            if "python -c" in cmd and "sys.version" in cmd:
                return {
                    "returncode": 0,
                    "stdout": '{"executable":"/opt/conda/envs/nano_test/bin/python","version":"3.11.9"}\n',
                    "stderr": "",
                }
            if "python -m pip --version" in cmd:
                return {
                    "returncode": 0,
                    "stdout": "pip 24.0 from /opt/conda/envs/nano_test/lib/python3.11/site-packages/pip\n",
                    "stderr": "",
                }
            raise AssertionError(f"Unexpected command: {cmd}")

        executor._run_cmd = fake_run  # type: ignore[method-assign]

        result = await executor.setup_env(cluster_code_path)

        assert result["ok"] is True
        assert result["source"] == "pyproject.toml"
        assert result["strategy"] == "editable"
        assert result["policy_source"] == "cached_local_manifest"
        assert result["runtime_validation"]["status"] == "ready"
        assert any("pip install -e" in cmd for cmd in commands)
        assert not any("pip install -r" in cmd for cmd in commands)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_setup_env_repairs_cluster_import_probe_failures_from_cached_requirements_snapshot() -> None:
    tmp_dir = Path(f".test_cluster_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        source_dir = tmp_dir / "source"
        source_dir.mkdir()
        (source_dir / "main.py").write_text("print('ok')\n", encoding="utf-8")
        (source_dir / "requirements.txt").write_text("PyYAML>=6\nnumpy>=1.26\n", encoding="utf-8")
        base_path = tmp_dir / "cluster"

        executor = ClusterExecutor(
            {"local": True, "code_path": str(base_path), "conda_env": "nano_test"}
        )
        cluster_code_path = await executor.prepare_code(source_dir, "sess_import_repair")

        commands: list[str] = []
        import_probe_calls = 0

        async def fake_run(cmd: str, timeout: int = 0) -> dict:
            nonlocal import_probe_calls
            del timeout
            commands.append(cmd)
            if "CONDA_SH=" in cmd:
                return {"returncode": 0, "stdout": "/opt/conda/etc/profile.d/conda.sh\n", "stderr": ""}
            if 'echo "requirements.txt"' in cmd:
                return {"returncode": 0, "stdout": "requirements.txt\n", "stderr": ""}
            if "conda env list" in cmd:
                return {"returncode": 0, "stdout": "ENV_EXISTS\n", "stderr": ""}
            if "pip install -r" in cmd:
                return {"returncode": 0, "stdout": "installed\n", "stderr": ""}
            if "python -c" in cmd and "sys.version" in cmd:
                return {
                    "returncode": 0,
                    "stdout": '{"executable":"/opt/conda/envs/nano_test/bin/python","version":"3.11.9"}\n',
                    "stderr": "",
                }
            if "python -m pip --version" in cmd:
                return {
                    "returncode": 0,
                    "stdout": "pip 24.0 from /opt/conda/envs/nano_test/lib/python3.11/site-packages/pip\n",
                    "stderr": "",
                }
            if "python -c" in cmd and "targets =" in cmd:
                import_probe_calls += 1
                if import_probe_calls == 1:
                    return {
                        "returncode": 0,
                        "stdout": (
                            '{"results":['
                            '{"package":"numpy","module":"numpy","status":"passed"},'
                            '{"package":"pyyaml","module":"yaml","status":"failed","error":"ModuleNotFoundError: No module named yaml"}'
                            "]}\n"
                        ),
                        "stderr": "",
                    }
                return {
                    "returncode": 0,
                    "stdout": (
                        '{"results":['
                        '{"package":"numpy","module":"numpy","status":"passed"},'
                        '{"package":"pyyaml","module":"yaml","status":"passed"}'
                        "]}\n"
                    ),
                    "stderr": "",
                }
            if "pip install" in cmd and "PyYAML>=6" in cmd:
                return {"returncode": 0, "stdout": "repaired\n", "stderr": ""}
            raise AssertionError(f"Unexpected command: {cmd}")

        executor._run_cmd = fake_run  # type: ignore[method-assign]

        result = await executor.setup_env(cluster_code_path)

        assert result["ok"] is True
        assert result["source"] == "requirements.txt"
        assert result["strategy"] == "requirements"
        assert result["policy_source"] == "cached_local_manifest"
        assert result["runtime_validation"]["status"] == "ready"
        assert result["runtime_validation_repair"]["status"] == "applied"
        assert result["runtime_validation_repair"]["actions"][0]["kind"] == "import_repair_install"
        assert result["runtime_validation_repair"]["actions"][0]["specs"] == ["PyYAML>=6"]
        assert any("pip install" in cmd and "PyYAML>=6" in cmd for cmd in commands)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_reupload_code_refreshes_cached_manifest_policy() -> None:
    tmp_dir = Path(f".test_cluster_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        source_dir = tmp_dir / "source"
        source_dir.mkdir()
        (source_dir / "main.py").write_text("print('ok')\n", encoding="utf-8")
        (source_dir / "pyproject.toml").write_text(
            "\n".join(
                [
                    "[build-system]",
                    'requires = ["setuptools>=61"]',
                    'build-backend = "setuptools.build_meta"',
                    "",
                    "[project]",
                    'name = "demo-project"',
                    'version = "0.1.0"',
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        base_path = tmp_dir / "cluster"

        executor = ClusterExecutor(
            {"local": True, "code_path": str(base_path), "conda_env": "nano_test"}
        )
        cluster_code_path = await executor.prepare_code(source_dir, "sess_refresh")

        (source_dir / "pyproject.toml").unlink()
        (source_dir / "requirements.txt").write_text("numpy\n", encoding="utf-8")
        await executor.reupload_code(source_dir, cluster_code_path)

        commands: list[str] = []

        async def fake_run(cmd: str, timeout: int = 0) -> dict:
            del timeout
            commands.append(cmd)
            if "CONDA_SH=" in cmd:
                return {"returncode": 0, "stdout": "/opt/conda/etc/profile.d/conda.sh\n", "stderr": ""}
            if 'echo "requirements.txt"' in cmd:
                return {
                    "returncode": 0,
                    "stdout": "requirements.txt\npyproject.toml\n",
                    "stderr": "",
                }
            if "conda env list" in cmd:
                return {"returncode": 0, "stdout": "ENV_EXISTS\n", "stderr": ""}
            if "pip install -r" in cmd:
                return {"returncode": 0, "stdout": "installed\n", "stderr": ""}
            if "python -c" in cmd and "sys.version" in cmd:
                return {
                    "returncode": 0,
                    "stdout": '{"executable":"/opt/conda/envs/nano_test/bin/python","version":"3.11.9"}\n',
                    "stderr": "",
                }
            if "python -m pip --version" in cmd:
                return {
                    "returncode": 0,
                    "stdout": "pip 24.0 from /opt/conda/envs/nano_test/lib/python3.11/site-packages/pip\n",
                    "stderr": "",
                }
            if "python -c" in cmd and "targets =" in cmd:
                return {
                    "returncode": 0,
                    "stdout": '{"results":[{"package":"numpy","module":"numpy","status":"passed"}]}\n',
                    "stderr": "",
                }
            raise AssertionError(f"Unexpected command: {cmd}")

        executor._run_cmd = fake_run  # type: ignore[method-assign]

        result = await executor.setup_env(cluster_code_path)

        assert result["ok"] is True
        assert result["source"] == "requirements.txt"
        assert result["strategy"] == "requirements"
        assert result["policy_source"] == "cached_local_manifest"
        assert result["runtime_validation"]["status"] == "ready"
        assert any("pip install -r" in cmd for cmd in commands)
        assert not any("pip install -e" in cmd for cmd in commands)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_collect_results_recovers_metrics_from_cluster_logs() -> None:
    tmp_dir = Path(f".test_cluster_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="cluster_collect",
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = ExecutionAgent(workspace, config)

        code_dir = workspace.path / "experiment"
        (code_dir / "logs").mkdir(parents=True)
        (code_dir / "results").mkdir()
        (code_dir / "logs" / "12345.log").write_text(
            "Epoch 1: loss=0.5\nTest accuracy: 0.91\nF1: 0.88\n",
            encoding="utf-8",
        )

        results = await agent._collect_results(code_dir, "12345", "COMPLETED")

        assert results["recovered_from"] == "slurm_logs"
        assert results["metrics_artifact_materialized"] is True
        assert results["metrics_artifact_path"] == "results/metrics.json"
        assert results["parsed_metrics"]["accuracy"] == "0.91"
        metric_names = {
            item["metric_name"]
            for item in results["metrics"]["main_results"][0]["metrics"]
        }
        assert "accuracy" in metric_names
        assert "F1" in metric_names
        assert results["training_log"][0]["epoch"] == 1
        assert results["training_log"][0]["train_loss"] == 0.5

        metrics_path = code_dir / "results" / "metrics.json"
        assert metrics_path.exists()
        persisted = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert persisted["_nanoresearch_meta"]["recovered_from"] == "slurm_logs"
        assert persisted["training_log"][0]["epoch"] == 1
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_collect_results_recovers_metrics_from_training_log_csv() -> None:
    tmp_dir = Path(f".test_cluster_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="cluster_collect_csv",
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = ExecutionAgent(workspace, config)

        code_dir = workspace.path / "experiment"
        (code_dir / "results").mkdir(parents=True)
        (code_dir / "training.py").write_text("print('ok')\n", encoding="utf-8")
        (code_dir / "results" / "training_log.csv").write_text(
            "epoch,loss,Accuracy,F1,lr\n"
            "1,0.50,0.75,0.71,0.001\n"
            "2,0.40,0.84,0.82,0.0009\n",
            encoding="utf-8",
        )

        results = await agent._collect_results(code_dir, "12345", "COMPLETED")

        assert results["recovered_from"] == "training_log_csv"
        assert results["metrics_artifact_materialized"] is True
        assert results["metrics_artifact_path"] == "results/metrics.json"
        metric_values = {
            item["metric_name"]: item["value"]
            for item in results["metrics"]["main_results"][0]["metrics"]
        }
        assert metric_values["Accuracy"] == 0.84
        assert metric_values["F1"] == 0.82
        assert "lr" not in metric_values
        assert results["training_log"][-1]["epoch"] == 2
        assert results["training_log"][-1]["train_loss"] == 0.4

        metrics_path = code_dir / "results" / "metrics.json"
        assert metrics_path.exists()
        persisted = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert persisted["_nanoresearch_meta"]["recovered_from"] == "training_log_csv"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_run_cluster_mode_returns_aligned_payload(monkeypatch) -> None:
    tmp_dir = Path(f".test_cluster_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="cluster_run",
        )
        code_dir = workspace.path / "experiment"
        code_dir.mkdir(parents=True, exist_ok=True)
        slurm_script = code_dir / "job.slurm"
        slurm_script.write_text("#!/bin/bash\n", encoding="utf-8")

        config = ResearchConfig(
            base_url="http://localhost:8000/v1/",
            api_key="test-key",
            execution_profile="cluster_full",
            slurm_partition="raise",
        )
        agent = ExecutionAgent(workspace, config)

        recovered_metrics = {
            "main_results": [
                {
                    "method_name": "ClusterRun",
                    "dataset": "UNKNOWN",
                    "is_proposed": True,
                    "metrics": [{"metric_name": "accuracy", "value": 0.91}],
                }
            ],
            "ablation_results": [],
            "training_log": [],
        }

        class DummyDebugAgent:
            def __init__(self, *_args, **_kwargs) -> None:
                pass

            def _fix_common_slurm_issues(self, _code_dir: Path) -> bool:
                return False

            async def run(self, **_kwargs) -> dict:
                return {"needs_resubmit": False}

            async def close(self) -> None:
                return None

        monkeypatch.setattr(execution_module, "DebugAgent", DummyDebugAgent)
        monkeypatch.setattr(execution_module.shutil, "which", lambda name: "/usr/bin/sbatch" if name == "sbatch" else None)
        monkeypatch.setattr(agent, "_local_preflight", AsyncMock(return_value=(True, "")))
        monkeypatch.setattr(agent, "_find_existing_job", AsyncMock(return_value=None))
        monkeypatch.setattr(agent, "_submit_job", AsyncMock(return_value="12345"))
        monkeypatch.setattr(agent, "_monitor_job", AsyncMock(return_value="COMPLETED"))
        monkeypatch.setattr(
            agent,
            "_collect_results",
            AsyncMock(
                return_value={
                    "metrics": recovered_metrics,
                    "stdout_log": "",
                    "stderr_log": "",
                    "training_log": [],
                    "checkpoints": [],
                }
            ),
        )

        result = await agent.run(
            coding_output={
                "code_dir": str(code_dir),
                "slurm_script": str(slurm_script),
            },
            experiment_blueprint={},
            setup_output={},
            topic="cluster test",
        )

        assert result["execution_backend"] == "cluster"
        assert result["runtime_env"]["kind"] == "cluster"
        assert result["runtime_env"]["partition"] == "raise"
        assert result["execution_status"] == "success"
        assert result["quick_eval_status"] == "skipped"
        assert result["experiment_status"] == "success"
        assert result["result_contract"]["status"] == "success"
        assert result["result_contract"]["success_path"] == "structured_metrics_artifact"
        assert result["experiment_results"] == recovered_metrics
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_run_cluster_mode_persists_remediation_ledger_for_debug_fixes(monkeypatch) -> None:
    tmp_dir = Path(f".test_cluster_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="cluster_ledger",
        )
        code_dir = workspace.path / "experiment"
        code_dir.mkdir(parents=True, exist_ok=True)
        slurm_script = code_dir / "job.slurm"
        slurm_script.write_text("#!/bin/bash\n", encoding="utf-8")

        config = ResearchConfig(
            base_url="http://localhost:8000/v1/",
            api_key="test-key",
            execution_profile="cluster_full",
            slurm_partition="raise",
        )
        agent = ExecutionAgent(workspace, config)

        results_sequence = [
            {
                "metrics": {},
                "stdout_log": "",
                "stderr_log": "ModuleNotFoundError: No module named 'demo_pkg'",
                "training_log": [],
                "checkpoints": [],
            },
            {
                "metrics": {
                    "main_results": [
                        {
                            "method_name": "ClusterRun",
                            "dataset": "UNKNOWN",
                            "is_proposed": True,
                            "metrics": [{"metric_name": "accuracy", "value": 0.91}],
                        }
                    ],
                    "ablation_results": [],
                    "training_log": [],
                },
                "stdout_log": "",
                "stderr_log": "",
                "training_log": [],
                "checkpoints": [],
            },
        ]

        class DummyDebugAgent:
            def __init__(self, *_args, **_kwargs) -> None:
                pass

            def _fix_common_slurm_issues(self, _code_dir: Path) -> bool:
                return True

            async def run(self, **_kwargs) -> dict:
                return {
                    "needs_resubmit": True,
                    "diagnosis": "Install missing dependency and rerun",
                    "patches": ["update requirements"],
                    "fixed_files": ["requirements.txt"],
                }

            async def close(self) -> None:
                return None

        monkeypatch.setattr(execution_module, "DebugAgent", DummyDebugAgent)
        monkeypatch.setattr(execution_module.shutil, "which", lambda name: "/usr/bin/sbatch" if name == "sbatch" else None)
        monkeypatch.setattr(agent, "_local_preflight", AsyncMock(return_value=(True, "")))
        monkeypatch.setattr(agent, "_find_existing_job", AsyncMock(return_value=None))
        monkeypatch.setattr(agent, "_submit_job", AsyncMock(side_effect=["12345", "12346"]))
        monkeypatch.setattr(agent, "_monitor_job", AsyncMock(side_effect=["FAILED", "COMPLETED"]))
        monkeypatch.setattr(agent, "_collect_results", AsyncMock(side_effect=results_sequence))

        result = await agent.run(
            coding_output={
                "code_dir": str(code_dir),
                "slurm_script": str(slurm_script),
            },
            experiment_blueprint={},
            setup_output={},
            topic="cluster ledger test",
        )

        assert result["final_status"] == "COMPLETED"
        assert any(
            entry["kind"] == "slurm_preflight_fix"
            and entry["status"] == "applied"
            for entry in result["remediation_ledger"]
        )
        assert any(
            entry["kind"] == "cluster_debug_fix"
            and entry["status"] == "applied"
            and "requirements.txt" in entry.get("files", [])
            for entry in result["remediation_ledger"]
        )
        ledger_payload = workspace.read_json("logs/execution_remediation_ledger.json")
        assert ledger_payload["entry_count"] >= 2
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_run_cluster_mode_marks_checkpoint_only_outputs_partial(monkeypatch) -> None:
    tmp_dir = Path(f".test_cluster_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="cluster_partial_contract",
        )
        code_dir = workspace.path / "experiment"
        code_dir.mkdir(parents=True, exist_ok=True)
        slurm_script = code_dir / "job.slurm"
        slurm_script.write_text("#!/bin/bash\n", encoding="utf-8")

        config = ResearchConfig(
            base_url="http://localhost:8000/v1/",
            api_key="test-key",
            execution_profile="cluster_full",
            slurm_partition="raise",
        )
        agent = ExecutionAgent(workspace, config)

        class DummyDebugAgent:
            def __init__(self, *_args, **_kwargs) -> None:
                pass

            def _fix_common_slurm_issues(self, _code_dir: Path) -> bool:
                return False

            async def run(self, **_kwargs) -> dict:
                return {"needs_resubmit": False}

            async def close(self) -> None:
                return None

        monkeypatch.setattr(execution_module, "DebugAgent", DummyDebugAgent)
        monkeypatch.setattr(execution_module.shutil, "which", lambda name: "/usr/bin/sbatch" if name == "sbatch" else None)
        monkeypatch.setattr(agent, "_local_preflight", AsyncMock(return_value=(True, "")))
        monkeypatch.setattr(agent, "_find_existing_job", AsyncMock(return_value=None))
        monkeypatch.setattr(agent, "_submit_job", AsyncMock(return_value="12345"))
        monkeypatch.setattr(agent, "_monitor_job", AsyncMock(return_value="COMPLETED"))
        monkeypatch.setattr(
            agent,
            "_collect_results",
            AsyncMock(
                return_value={
                    "metrics": {},
                    "stdout_log": "",
                    "stderr_log": "",
                    "training_log": [],
                    "training_log_csv": "epoch,loss\n1,0.5\n",
                    "checkpoints": ["checkpoints/model.pt"],
                }
            ),
        )

        result = await agent.run(
            coding_output={
                "code_dir": str(code_dir),
                "slurm_script": str(slurm_script),
            },
            experiment_blueprint={},
            setup_output={},
            topic="cluster partial contract",
        )

        assert result["final_status"] == "COMPLETED"
        assert result["execution_status"] == "success"
        assert result["experiment_status"] == "partial"
        assert result["result_contract"]["status"] == "partial"
        assert result["result_contract"]["success_path"] == "training_log_with_checkpoints"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_run_cluster_mode_prefers_resume_repair_before_debug(monkeypatch) -> None:
    tmp_dir = Path(f".test_cluster_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="cluster_resume_repair",
        )
        code_dir = workspace.path / "experiment"
        code_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir = code_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        latest_checkpoint = checkpoints_dir / "latest.pt"
        latest_checkpoint.write_text("checkpoint", encoding="utf-8")
        (code_dir / "train.py").write_text(
            "import argparse\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument('--resume')\n"
            "parser.parse_args()\n",
            encoding="utf-8",
        )
        ensure_project_runner(code_dir, "python train.py")
        slurm_script = code_dir / "job.slurm"
        slurm_script.write_text("#!/bin/bash\npython nanoresearch_runner.py\n", encoding="utf-8")

        config = ResearchConfig(
            base_url="http://localhost:8000/v1/",
            api_key="test-key",
            execution_profile="cluster_full",
            slurm_partition="raise",
        )
        agent = ExecutionAgent(workspace, config)

        class DummyDebugAgent:
            run_calls = 0

            def __init__(self, *_args, **_kwargs) -> None:
                pass

            def _fix_common_slurm_issues(self, _code_dir: Path) -> bool:
                return False

            async def run(self, **_kwargs) -> dict:
                DummyDebugAgent.run_calls += 1
                return {"needs_resubmit": False}

            async def close(self) -> None:
                return None

        monkeypatch.setattr(execution_module, "DebugAgent", DummyDebugAgent)
        monkeypatch.setattr(execution_module.shutil, "which", lambda name: "/usr/bin/sbatch" if name == "sbatch" else None)
        monkeypatch.setattr(agent, "_local_preflight", AsyncMock(return_value=(True, "")))
        monkeypatch.setattr(agent, "_find_existing_job", AsyncMock(return_value=None))
        submit_mock = AsyncMock(side_effect=["12345", "23456"])
        monitor_mock = AsyncMock(side_effect=["TIMEOUT", "COMPLETED"])
        collect_mock = AsyncMock(
            side_effect=[
                {
                    "metrics": {},
                    "stdout_log": "",
                    "stderr_log": "Command timed out",
                    "training_log": [],
                    "training_log_csv": "epoch,loss\n1,0.5\n",
                    "checkpoints": [str(latest_checkpoint)],
                },
                {
                    "metrics": {
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
                    },
                    "stdout_log": "done",
                    "stderr_log": "",
                    "training_log": [],
                    "checkpoints": [str(latest_checkpoint)],
                },
            ]
        )
        monkeypatch.setattr(agent, "_submit_job", submit_mock)
        monkeypatch.setattr(agent, "_monitor_job", monitor_mock)
        monkeypatch.setattr(agent, "_collect_results", collect_mock)

        result = await agent.run(
            coding_output={
                "code_dir": str(code_dir),
                "slurm_script": str(slurm_script),
            },
            experiment_blueprint={},
            setup_output={},
            topic="cluster resume repair",
        )
        runner_config = json.loads((code_dir / RUNNER_CONFIG_NAME).read_text(encoding="utf-8"))

        assert result["final_status"] == "COMPLETED"
        assert result["experiment_status"] == "success"
        assert submit_mock.await_count == 2
        assert monitor_mock.await_count == 2
        assert DummyDebugAgent.run_calls == 0
        resume_index = runner_config["target_command"].index("--resume")
        assert Path(runner_config["target_command"][resume_index + 1]).resolve() == latest_checkpoint.resolve()
        assert any(
            entry["kind"] == "resume_repair"
            and entry["status"] == "applied"
            and entry["scope"] == "cluster_resume"
            for entry in result["remediation_ledger"]
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_monitor_job_treats_preempted_as_terminal(monkeypatch) -> None:
    tmp_dir = Path(f".test_cluster_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="cluster_preempted_terminal",
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = ExecutionAgent(workspace, config)
        code_dir = workspace.path / "experiment"
        (code_dir / "logs").mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(agent, "_get_job_status", AsyncMock(return_value="PREEMPTED"))

        status = await agent._monitor_job("12345", code_dir)

        assert status == "PREEMPTED"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
