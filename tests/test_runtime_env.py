"""Tests for shared runtime environment preparation."""

from __future__ import annotations

import subprocess
from pathlib import Path
import platform
import shutil
import uuid
import venv as stdlib_venv

import pytest

from nanoresearch.agents.runtime_env import RuntimeEnvironmentManager
from nanoresearch.config import ResearchConfig


@pytest.mark.asyncio
async def test_prepare_system_python_still_installs_dependencies() -> None:
    tmp_dir = Path(f".test_runtime_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        code_dir = tmp_dir / "code"
        code_dir.mkdir()
        (code_dir / "requirements.txt").write_text("numpy\n", encoding="utf-8")

        config = ResearchConfig(
            base_url="http://localhost:8000/v1/",
            api_key="test-key",
            auto_create_env=False,
        )
        manager = RuntimeEnvironmentManager(config)

        async def fake_install(python: str, target_dir: Path) -> dict:
            assert python
            assert target_dir == code_dir
            return {"status": "installed", "source": "requirements.txt"}

        async def fake_validate(python: str, target_dir: Path, **_kwargs) -> dict:
            assert python
            assert target_dir == code_dir
            return {"status": "ready"}

        manager.install_requirements = fake_install  # type: ignore[method-assign]
        manager.validate_runtime = fake_validate  # type: ignore[method-assign]

        result = await manager.prepare(code_dir)

        assert result["kind"] == "system"
        assert result["dependency_install"] == {
            "status": "installed",
            "source": "requirements.txt",
        }
        assert result["runtime_validation"] == {"status": "ready"}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_install_requirements_falls_back_to_environment_yml(monkeypatch) -> None:
    tmp_dir = Path(f".test_runtime_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        code_dir = tmp_dir / "code"
        code_dir.mkdir()
        (code_dir / "environment.yml").write_text(
            "\n".join(
                [
                    "name: demo",
                    "channels:",
                    "  - conda-forge",
                    "dependencies:",
                    "  - python=3.10",
                    "  - pip",
                    "  - pip:",
                    "      - torch",
                    "      - pandas>=2",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        commands: list[list[str]] = []

        class Completed:
            returncode = 0
            stdout = ""
            stderr = ""

        def fake_run(command: list[str], **_kwargs) -> Completed:
            commands.append(command)
            return Completed()

        monkeypatch.setattr(subprocess, "run", fake_run)

        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        manager = RuntimeEnvironmentManager(config)
        result = await manager.install_requirements("python-custom", code_dir)

        assert result["status"] == "installed"
        assert result["source"] == "environment.yml"
        assert result["strategy"] == "primary"
        assert result["manifest"].endswith("environment.yml")
        assert commands
        assert commands[0][:4] == ["python-custom", "-m", "pip", "install"]
        assert "torch" in commands[0]
        assert "pandas>=2" in commands[0]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_install_requirements_supports_environment_yaml_alias(monkeypatch) -> None:
    tmp_dir = Path(f".test_runtime_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        code_dir = tmp_dir / "code"
        code_dir.mkdir()
        (code_dir / "environment.yaml").write_text(
            "\n".join(
                [
                    "name: demo",
                    "dependencies:",
                    "  - pip",
                    "  - pip:",
                    "      - scikit-learn",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        commands: list[list[str]] = []

        class Completed:
            returncode = 0
            stdout = ""
            stderr = ""

        def fake_run(command: list[str], **_kwargs) -> Completed:
            commands.append(command)
            return Completed()

        monkeypatch.setattr(subprocess, "run", fake_run)

        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        manager = RuntimeEnvironmentManager(config)
        result = await manager.install_requirements("python-custom", code_dir)

        assert result["status"] == "installed"
        assert result["source"] == "environment.yaml"
        assert result["strategy"] == "primary"
        assert commands
        assert commands[0][:4] == ["python-custom", "-m", "pip", "install"]
        assert "scikit-learn" in commands[0]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_install_requirements_falls_back_to_pyproject_editable_install(monkeypatch) -> None:
    tmp_dir = Path(f".test_runtime_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        code_dir = tmp_dir / "code"
        code_dir.mkdir()
        (code_dir / "pyproject.toml").write_text(
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

        commands: list[list[str]] = []

        class Completed:
            def __init__(self, returncode: int = 0, stderr: str = "") -> None:
                self.returncode = returncode
                self.stdout = ""
                self.stderr = stderr

        def fake_run(command: list[str], **_kwargs) -> Completed:
            commands.append(command)
            return Completed()

        monkeypatch.setattr(subprocess, "run", fake_run)

        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        manager = RuntimeEnvironmentManager(config)
        result = await manager.install_requirements("python-custom", code_dir)

        assert result["status"] == "installed"
        assert result["source"] == "pyproject.toml"
        assert result["strategy"] == "primary"
        assert commands == [["python-custom", "-m", "pip", "install", "-e", ".", "--quiet"]]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_prepare_reports_environment_yaml_path() -> None:
    tmp_dir = Path(f".test_runtime_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        code_dir = tmp_dir / "code"
        code_dir.mkdir()
        (code_dir / "environment.yaml").write_text(
            "name: demo\ndependencies:\n  - pip\n",
            encoding="utf-8",
        )

        config = ResearchConfig(
            base_url="http://localhost:8000/v1/",
            api_key="test-key",
            auto_create_env=False,
        )
        manager = RuntimeEnvironmentManager(config)

        async def fake_install(python: str, target_dir: Path) -> dict:
            assert python
            assert target_dir == code_dir
            return {"status": "installed", "source": "environment.yaml"}

        async def fake_validate(python: str, target_dir: Path, **_kwargs) -> dict:
            assert python
            assert target_dir == code_dir
            return {"status": "ready"}

        manager.install_requirements = fake_install  # type: ignore[method-assign]
        manager.validate_runtime = fake_validate  # type: ignore[method-assign]

        result = await manager.prepare(code_dir)

        assert result["kind"] == "system"
        assert result["environment_file"].endswith("environment.yaml")
        assert result["execution_policy"]["manifest_source"] == "environment.yaml"
        assert result["runtime_validation"] == {"status": "ready"}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_extract_pip_dependencies_ignores_environment_channels() -> None:
    tmp_dir = Path(f".test_runtime_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        code_dir = tmp_dir / "code"
        code_dir.mkdir()
        environment_file = code_dir / "environment.yml"
        environment_file.write_text(
            "\n".join(
                [
                    "name: demo",
                    "channels:",
                    "  - conda-forge",
                    "  - pytorch",
                    "  - defaults",
                    "dependencies:",
                    "  - python=3.10",
                    "  - numpy",
                    "  - pandas>=2",
                    "  - pip",
                    "  - pip:",
                    "      - torch>=2",
                    "      - scikit-learn>=1.4",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        deps = RuntimeEnvironmentManager._extract_pip_dependencies(environment_file)

        assert deps == ["numpy", "pandas>=2", "torch>=2", "scikit-learn>=1.4"]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_build_execution_policy_environment_dependencies_ignore_channels() -> None:
    tmp_dir = Path(f".test_runtime_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        code_dir = tmp_dir / "code"
        code_dir.mkdir()
        (code_dir / "environment.yml").write_text(
            "\n".join(
                [
                    "name: demo",
                    "channels:",
                    "  - conda-forge",
                    "  - pytorch",
                    "  - defaults",
                    "dependencies:",
                    "  - python=3.10",
                    "  - numpy",
                    "  - pip",
                    "  - pip:",
                    "      - torch>=2",
                    "      - pandas>=2",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        manager = RuntimeEnvironmentManager(config)

        policy = manager.build_execution_policy(code_dir)

        assert policy.declared_dependencies >= {"numpy", "torch", "pandas"}
        assert "conda-forge" not in policy.declared_dependencies
        assert "defaults" not in policy.declared_dependencies
        assert "pytorch" not in policy.declared_dependencies
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_build_execution_policy_collects_declared_dependencies() -> None:
    tmp_dir = Path(f".test_runtime_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        code_dir = tmp_dir / "code"
        code_dir.mkdir()
        (code_dir / "requirements.txt").write_text(
            "torch>=2\npandas[performance]>=2\n-e git+https://example.com/demo.git#egg=demo_pkg\n",
            encoding="utf-8",
        )
        (code_dir / "pyproject.toml").write_text(
            "\n".join(
                [
                    "[project]",
                    'name = "demo-project"',
                    'version = "0.1.0"',
                    'dependencies = ["scikit-learn>=1.4"]',
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        config = ResearchConfig(
            base_url="http://localhost:8000/v1/",
            api_key="test-key",
            runtime_auto_install_allowlist=["opencv-python"],
            runtime_auto_install_max_packages=1,
        )
        manager = RuntimeEnvironmentManager(config)

        policy = manager.build_execution_policy(code_dir)

        assert policy.declared_dependencies >= {
            "torch",
            "pandas",
            "demo-pkg",
            "scikit-learn",
        }
        assert policy.runtime_auto_install_allowlist == {"opencv-python"}
        assert policy.max_runtime_auto_installs == 1
        assert policy.to_dict()["manifest_source"] == "requirements.txt"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_validate_runtime_reports_partial_when_import_probe_fails(monkeypatch) -> None:
    tmp_dir = Path(f".test_runtime_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        code_dir = tmp_dir / "code"
        code_dir.mkdir()
        (code_dir / "requirements.txt").write_text("PyYAML\nnumpy\n", encoding="utf-8")

        class Completed:
            def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
                self.returncode = returncode
                self.stdout = stdout
                self.stderr = stderr

        def fake_run(command: list[str], **_kwargs) -> Completed:
            if command[:2] == ["python-custom", "-c"] and "sys.version" in command[2]:
                return Completed(
                    returncode=0,
                    stdout='{"executable":"python-custom","version":"3.11.9"}\n',
                )
            if command == ["python-custom", "-m", "pip", "--version"]:
                return Completed(returncode=0, stdout="pip 24.0 from /tmp/site-packages/pip\n")
            if command[:2] == ["python-custom", "-c"] and "targets =" in command[2]:
                return Completed(
                    returncode=0,
                    stdout=(
                        '{"results":['
                        '{"package":"numpy","module":"numpy","status":"passed"},'
                        '{"package":"pyyaml","module":"yaml","status":"failed","error":"ModuleNotFoundError: No module named yaml"}'
                        "]}\n"
                    ),
                )
            raise AssertionError(f"Unexpected command: {command}")

        monkeypatch.setattr(subprocess, "run", fake_run)

        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        manager = RuntimeEnvironmentManager(config)
        policy = manager.build_execution_policy(code_dir)

        result = await manager.validate_runtime(
            "python-custom",
            code_dir,
            execution_policy=policy,
        )

        assert result["status"] == "partial"
        assert result["python_smoke"]["status"] == "passed"
        assert result["pip_probe"]["status"] == "passed"
        assert result["import_probe"]["status"] == "partial"
        assert result["import_probe"]["failures"] == [
            {
                "package": "pyyaml",
                "module": "yaml",
                "status": "failed",
                "error": "ModuleNotFoundError: No module named yaml",
            }
        ]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_validate_runtime_skips_import_probe_for_editable_install(monkeypatch) -> None:
    tmp_dir = Path(f".test_runtime_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        code_dir = tmp_dir / "code"
        code_dir.mkdir()
        (code_dir / "pyproject.toml").write_text(
            "\n".join(
                [
                    "[build-system]",
                    'requires = ["setuptools>=61"]',
                    'build-backend = "setuptools.build_meta"',
                    "",
                    "[project]",
                    'name = "demo-project"',
                    'version = "0.1.0"',
                    'dependencies = ["numpy>=1.26"]',
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        class Completed:
            def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
                self.returncode = returncode
                self.stdout = stdout
                self.stderr = stderr

        def fake_run(command: list[str], **_kwargs) -> Completed:
            if command[:2] == ["python-custom", "-c"] and "sys.version" in command[2]:
                return Completed(
                    returncode=0,
                    stdout='{"executable":"python-custom","version":"3.11.9"}\n',
                )
            if command == ["python-custom", "-m", "pip", "--version"]:
                return Completed(returncode=0, stdout="pip 24.0 from /tmp/site-packages/pip\n")
            raise AssertionError(f"Unexpected command: {command}")

        monkeypatch.setattr(subprocess, "run", fake_run)

        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        manager = RuntimeEnvironmentManager(config)
        policy = manager.build_execution_policy(code_dir)

        result = await manager.validate_runtime(
            "python-custom",
            code_dir,
            execution_policy=policy,
        )

        assert result["status"] == "ready"
        assert result["import_probe"]["status"] == "skipped"
        assert result["import_probe"]["skipped_reason"] == "install_source_not_probe_safe"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_prepare_recreates_invalid_existing_venv(monkeypatch) -> None:
    tmp_dir = Path(f".test_runtime_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        code_dir = tmp_dir / "code"
        code_dir.mkdir()
        (code_dir / "requirements.txt").write_text("numpy\n", encoding="utf-8")

        venv_dir = code_dir / ".venv"
        is_windows = platform.system() == "Windows"
        python_path = venv_dir / ("Scripts/python.exe" if is_windows else "bin/python")
        python_path.parent.mkdir(parents=True, exist_ok=True)
        python_path.write_text("", encoding="utf-8")
        stale_marker = venv_dir / "stale.txt"
        stale_marker.write_text("broken", encoding="utf-8")

        config = ResearchConfig(
            base_url="http://localhost:8000/v1/",
            api_key="test-key",
            auto_create_env=True,
        )
        manager = RuntimeEnvironmentManager(config)

        install_calls: list[str] = []

        async def fake_install(python: str, target_dir: Path) -> dict:
            install_calls.append(python)
            assert target_dir == code_dir
            return {"status": "installed", "source": "requirements.txt"}

        validations = iter(
            [
                {
                    "status": "failed",
                    "python_smoke": {"status": "failed", "error": "broken venv"},
                    "pip_probe": {"status": "skipped"},
                    "import_probe": {"status": "skipped"},
                },
                {
                    "status": "ready",
                    "python_smoke": {"status": "passed", "executable": "rebuilt-python", "version": "3.11.9"},
                    "pip_probe": {"status": "passed", "version": "pip 24.0"},
                    "import_probe": {"status": "skipped", "skipped_reason": "no_probeable_dependencies", "targets": [], "failures": []},
                },
            ]
        )

        async def fake_validate(python: str, target_dir: Path, **_kwargs) -> dict:
            assert target_dir == code_dir
            return next(validations)

        def fake_create(env_path: str, with_pip: bool = True) -> None:
            assert with_pip is True
            rebuilt_python = Path(env_path) / ("Scripts/python.exe" if is_windows else "bin/python")
            rebuilt_python.parent.mkdir(parents=True, exist_ok=True)
            rebuilt_python.write_text("", encoding="utf-8")

        monkeypatch.setattr(stdlib_venv, "create", fake_create)
        monkeypatch.setattr("nanoresearch.agents.runtime_env.venv.create", fake_create)
        manager.install_requirements = fake_install  # type: ignore[method-assign]
        manager.validate_runtime = fake_validate  # type: ignore[method-assign]

        result = await manager.prepare(code_dir)

        assert result["kind"] == "venv"
        assert result["created"] is True
        assert result["recreated"] is True
        assert result["runtime_validation"]["status"] == "ready"
        assert result["runtime_validation_repair"]["status"] == "applied"
        assert [action["kind"] for action in result["runtime_validation_repair"]["actions"]] == [
            "recreate_venv",
            "reinstall_manifest",
        ]
        assert stale_marker.exists() is False
        assert len(install_calls) == 2
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_prepare_repairs_failed_imports_with_targeted_specs() -> None:
    tmp_dir = Path(f".test_runtime_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        code_dir = tmp_dir / "code"
        code_dir.mkdir()
        (code_dir / "requirements.txt").write_text("PyYAML>=6\nnumpy>=1.26\n", encoding="utf-8")

        config = ResearchConfig(
            base_url="http://localhost:8000/v1/",
            api_key="test-key",
            auto_create_env=False,
        )
        manager = RuntimeEnvironmentManager(config)

        async def fake_install(_python: str, _target_dir: Path) -> dict:
            return {"status": "installed", "source": "requirements.txt"}

        validations = iter(
            [
                {
                    "status": "partial",
                    "python_smoke": {"status": "passed", "executable": "python", "version": "3.11.9"},
                    "pip_probe": {"status": "passed", "version": "pip 24.0"},
                    "import_probe": {
                        "status": "partial",
                        "failures": [
                            {
                                "package": "pyyaml",
                                "module": "yaml",
                                "error": "ModuleNotFoundError: No module named yaml",
                            }
                        ],
                    },
                },
                {
                    "status": "ready",
                    "python_smoke": {"status": "passed", "executable": "python", "version": "3.11.9"},
                    "pip_probe": {"status": "passed", "version": "pip 24.0"},
                    "import_probe": {
                        "status": "passed",
                        "failures": [],
                        "targets": [{"package": "pyyaml", "module": "yaml"}],
                        "results": [{"package": "pyyaml", "module": "yaml", "status": "passed"}],
                    },
                },
            ]
        )

        async def fake_validate(_python: str, _target_dir: Path, **_kwargs) -> dict:
            return next(validations)

        targeted_specs: list[list[str]] = []

        async def fake_install_specs(_python: str, _target_dir: Path, specs: list[str], **_kwargs) -> dict:
            targeted_specs.append(list(specs))
            return {"status": "installed", "source": "runtime_validation_import_repair", "specs": list(specs)}

        manager.install_requirements = fake_install  # type: ignore[method-assign]
        manager.validate_runtime = fake_validate  # type: ignore[method-assign]
        manager.install_dependency_specs = fake_install_specs  # type: ignore[method-assign]

        result = await manager.prepare(code_dir)

        assert result["kind"] == "system"
        assert result["runtime_validation"]["status"] == "ready"
        assert result["runtime_validation_repair"]["status"] == "applied"
        assert targeted_specs == [["PyYAML>=6"]]
        assert result["runtime_validation_repair"]["actions"][0]["kind"] == "import_repair_install"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
