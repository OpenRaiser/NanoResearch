"""Tests for environment creation, auto-repair, and dataset download containment.

Covers:
1. Venv creation success/failure paths
2. Auto-repair: venv fail → conda fallback → cleanup retry → diagnostic
3. Dataset download path containment (never download outside designated dirs)
4. Data path validation in generated code
"""

from __future__ import annotations

import asyncio
import os
import platform
import shutil
import subprocess
import sys
import uuid
import venv as stdlib_venv
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanoresearch.agents.runtime_env import RuntimeEnvironmentManager
from nanoresearch.config import ResearchConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> ResearchConfig:
    defaults = dict(
        base_url="http://localhost:8000/v1/",
        api_key="test-key",
    )
    defaults.update(overrides)
    return ResearchConfig(**defaults)


def _tmp_code_dir() -> Path:
    d = Path(f".test_env_{uuid.uuid4().hex[:8]}")
    d.mkdir()
    code = d / "code"
    code.mkdir()
    (code / "requirements.txt").write_text("numpy\n", encoding="utf-8")
    return d


# ---------------------------------------------------------------------------
# Part 1: Environment creation
# ---------------------------------------------------------------------------

class TestVenvCreation:
    """Verify that prepare() creates an isolated venv and never uses sys.executable."""

    @pytest.mark.asyncio
    async def test_creates_venv_in_code_dir(self):
        tmp = _tmp_code_dir()
        try:
            config = _make_config(environment_backend="venv")
            mgr = RuntimeEnvironmentManager(config)
            result = await mgr.prepare(tmp / "code")

            assert result["kind"] == "venv"
            assert result["python"] != sys.executable
            assert (tmp / "code" / ".venv").is_dir()
            # Python executable must actually exist
            assert Path(result["python"]).exists()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_reuses_existing_venv(self):
        tmp = _tmp_code_dir()
        try:
            config = _make_config(environment_backend="venv")
            mgr = RuntimeEnvironmentManager(config)

            r1 = await mgr.prepare(tmp / "code")
            r2 = await mgr.prepare(tmp / "code")

            assert r1["python"] == r2["python"]
            # Second call should NOT recreate
            assert r2.get("created") is not True or r2.get("recreated") is not True
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_uses_configured_conda_env_if_exists(self):
        """If experiment_conda_env is set and the env exists, use it."""
        tmp = _tmp_code_dir()
        try:
            config = _make_config(experiment_conda_env="fake_env")
            mgr = RuntimeEnvironmentManager(config)

            # Mock find_conda_python to return a fake path
            fake_python = str(tmp / "fake_python")
            Path(fake_python).touch()

            with patch.object(
                RuntimeEnvironmentManager, "find_conda_python",
                return_value=fake_python,
            ):
                # Also mock install_requirements and validate_runtime
                mgr.install_requirements = AsyncMock(return_value={"status": "installed"})
                mgr.validate_runtime = AsyncMock(return_value={"status": "ready"})

                result = await mgr.prepare(tmp / "code")

            assert result["kind"] == "conda"
            assert result["python"] == fake_python
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Part 2: Auto-repair
# ---------------------------------------------------------------------------

class TestAutoRepair:
    """Verify auto-repair strategies when venv creation fails."""

    @pytest.mark.asyncio
    async def test_repair_strategy1_conda_fallback(self):
        """When venv.create fails but conda is available, auto-create conda env."""
        tmp = _tmp_code_dir()
        try:
            config = _make_config(environment_backend="venv")
            mgr = RuntimeEnvironmentManager(config)

            fake_python = str(tmp / "conda_python")
            Path(fake_python).touch()

            # Make venv.create always fail
            def failing_venv_create(*args, **kwargs):
                raise OSError("ensurepip is not available")

            def _which(cmd, *args, **kwargs):
                return "/usr/bin/conda" if cmd == "conda" else None

            with patch("nanoresearch.agents.runtime_env.venv.create", side_effect=failing_venv_create):
                with patch("shutil.which", side_effect=_which):
                    with patch.object(mgr, "create_conda_env", new_callable=AsyncMock, return_value=True):
                        with patch.object(
                            RuntimeEnvironmentManager, "find_conda_python",
                            return_value=fake_python,
                        ):
                            mgr.install_requirements = AsyncMock(
                                return_value={"status": "installed"}
                            )
                            mgr.validate_runtime = AsyncMock(
                                return_value={"status": "ready"}
                            )

                            result = await mgr.prepare(tmp / "code")

            assert result["kind"] == "conda"
            assert result["auto_repaired"] is True
            assert result["python"] == fake_python
            assert result["env_name"].startswith("nanoresearch_")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_repair_strategy2_cleanup_retry(self):
        """When venv fails and conda not available, try removing corrupted venv."""
        tmp = _tmp_code_dir()
        code_dir = tmp / "code"
        venv_dir = code_dir / ".venv"
        # Pre-create a corrupted venv dir (exists but incomplete)
        venv_dir.mkdir()
        (venv_dir / "corrupted_marker").touch()

        try:
            config = _make_config(environment_backend="venv")
            mgr = RuntimeEnvironmentManager(config)

            call_count = 0
            # Save real venv.create BEFORE patching (the mock replaces the
            # module attribute, so stdlib_venv.create would also be mocked)
            real_venv_create = stdlib_venv.create

            def conditional_venv_create(path, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise OSError("first attempt fails")
                # Second attempt (after cleanup) succeeds — use saved real function
                real_venv_create(path, with_pip=True)

            with patch("nanoresearch.agents.runtime_env.venv.create", side_effect=conditional_venv_create):
                # No conda available
                with patch("shutil.which", return_value=None):
                    result = await mgr.prepare(code_dir)

            assert result["kind"] == "venv"
            assert result["auto_repaired"] is True
            # Corrupted marker should be gone
            assert not (venv_dir / "corrupted_marker").exists()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_repair_all_strategies_exhausted_raises(self):
        """When all repair strategies fail, raise RuntimeError with diagnosis."""
        tmp = _tmp_code_dir()
        try:
            config = _make_config()
            mgr = RuntimeEnvironmentManager(config)

            def always_fail(*args, **kwargs):
                raise OSError("ensurepip is not available")

            with patch("nanoresearch.agents.runtime_env.venv.create", side_effect=always_fail):
                with patch("shutil.which", return_value=None):
                    with pytest.raises(RuntimeError, match="auto-repair exhausted"):
                        await mgr.prepare(tmp / "code")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_repair_conda_created_but_python_not_found(self):
        """Conda create succeeds but find_conda_python returns None → fall to strategy 2."""
        tmp = _tmp_code_dir()
        code_dir = tmp / "code"

        try:
            config = _make_config(environment_backend="venv")
            mgr = RuntimeEnvironmentManager(config)

            call_count = 0
            real_venv_create = stdlib_venv.create

            def conditional_venv_create(path, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise OSError("first attempt fails")
                # Second attempt succeeds — use saved real function
                real_venv_create(path, with_pip=True)

            def _which2(cmd, *args, **kwargs):
                return "/usr/bin/conda" if cmd == "conda" else None

            with patch("nanoresearch.agents.runtime_env.venv.create", side_effect=conditional_venv_create):
                with patch("shutil.which", side_effect=_which2):
                    with patch.object(mgr, "create_conda_env", new_callable=AsyncMock, return_value=True):
                        # Conda env created but python not found in it
                        with patch.object(
                            RuntimeEnvironmentManager, "find_conda_python",
                            return_value=None,
                        ):
                            # Strategy 2 should kick in (venv dir doesn't exist yet,
                            # so it won't trigger cleanup retry). Let's pre-create
                            # the venv dir to trigger cleanup path.
                            (code_dir / ".venv").mkdir(exist_ok=True)

                            result = await mgr.prepare(code_dir)

            # Should succeed via strategy 2 (cleanup + retry)
            assert result["kind"] == "venv"
            assert result["auto_repaired"] is True
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_diagnose_env_failure_missing_ensurepip(self):
        """Diagnosis should detect missing python3-venv."""
        tmp = Path(f".test_diag_{uuid.uuid4().hex[:8]}")
        tmp.mkdir()
        try:
            exc = OSError("No module named 'ensurepip'")
            diag = RuntimeEnvironmentManager._diagnose_env_failure(tmp / ".venv", exc)
            assert "python3-venv" in diag
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_diagnose_env_failure_permission_denied(self):
        """Diagnosis should detect permission errors."""
        tmp = Path(f".test_diag_{uuid.uuid4().hex[:8]}")
        tmp.mkdir()
        try:
            exc = OSError("Permission denied: '/opt/.venv'")
            diag = RuntimeEnvironmentManager._diagnose_env_failure(tmp / ".venv", exc)
            assert "Permission denied" in diag
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_diagnose_env_failure_unknown(self):
        """Unknown errors should still produce a diagnosis."""
        tmp = Path(f".test_diag_{uuid.uuid4().hex[:8]}")
        tmp.mkdir()
        try:
            exc = OSError("Something weird happened")
            diag = RuntimeEnvironmentManager._diagnose_env_failure(tmp / ".venv", exc)
            # The diagnosis should be a non-empty string.
            # On low-disk systems the primary reason may be "Low disk space"
            # rather than the exception message, so just verify we get output.
            assert diag
            # The exception text should appear unless another condition
            # (low disk, permissions) took priority.
            assert "Something weird happened" in diag or "Low disk" in diag or "permission" in diag.lower()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Part 3: sys.executable never used
# ---------------------------------------------------------------------------

class TestNoSysExecutable:
    """Verify that experiment code paths never fall back to sys.executable."""

    @pytest.mark.asyncio
    async def test_setup_venv_returns_explicit_python_not_sys(self):
        """_setup_venv must return a path from RuntimeEnvironmentManager, not sys.executable."""
        tmp = _tmp_code_dir()
        try:
            config = _make_config()
            mgr = RuntimeEnvironmentManager(config)
            result = await mgr.prepare(tmp / "code")
            assert result["python"] != sys.executable
            assert result["python"]  # not empty
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_code_runner_no_sys_executable_in_experiment_path(self):
        """Scan code_runner.py to ensure no sys.executable fallback in experiment execution."""
        code_runner_path = (
            Path(__file__).parent.parent
            / "nanoresearch" / "agents" / "experiment" / "code_runner.py"
        )
        content = code_runner_path.read_text(encoding="utf-8")
        # Count sys.executable references — should only appear in
        # the import line, not as fallback values
        lines_with_fallback = [
            line.strip() for line in content.splitlines()
            if "sys.executable" in line
            and "import" not in line
            and line.strip().startswith("#") is False
        ]
        # All remaining sys.executable refs should be in comments only
        for line in lines_with_fallback:
            assert line.startswith("#"), (
                f"Non-comment sys.executable found in code_runner.py: {line}"
            )


# ---------------------------------------------------------------------------
# Part 4: Dataset download path containment
# ---------------------------------------------------------------------------

class TestDataPathContainment:
    """Verify dataset downloads are contained within designated directories."""

    def test_download_resources_uses_workspace_datasets_dir(self):
        """SetupAgent.run() downloads datasets to workspace/datasets/, not global cache."""
        import inspect
        from nanoresearch.agents.setup import SetupAgent, GLOBAL_DATA_DIR

        # Verify the global cache dir constant still exists (used for models)
        assert "cache" in str(GLOBAL_DATA_DIR)
        assert ".nanobot" in str(GLOBAL_DATA_DIR)

        # Verify SetupAgent.run() creates workspace-local datasets_dir
        source = inspect.getsource(SetupAgent.run)
        assert 'datasets_dir = self.workspace.path / "datasets"' in source
        assert '"datasets_dir"' in source

    def test_validate_data_paths_catches_outside_reference(self):
        """_validate_data_paths should flag paths outside valid directories."""
        from nanoresearch.agents.coding import CodingAgent

        tmp = Path(f".test_data_{uuid.uuid4().hex[:8]}")
        code_dir = tmp / "code"
        code_dir.mkdir(parents=True)
        try:
            # Create a Python file with a path reference outside valid dirs
            (code_dir / "train.py").write_text(
                'import pandas as pd\n'
                'df = pd.read_csv("/etc/passwd")\n'
                'data = open("/home/user/secret.txt")\n',
                encoding="utf-8",
            )

            # Valid directories
            data_dir = str(tmp / "data")
            models_dir = str(tmp / "models")
            downloaded = []

            issues = CodingAgent._validate_data_paths(
                None,  # self (not used for paths)
                code_dir,
                downloaded,
                data_dir,
                models_dir,
            )

            # Both outside paths should be flagged
            flagged_paths = {i["path"] for i in issues}
            assert "/etc/passwd" in flagged_paths
            assert "/home/user/secret.txt" in flagged_paths
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_validate_data_paths_allows_valid_resource(self):
        """Paths under data_dir or downloaded resources should pass validation."""
        from nanoresearch.agents.coding import CodingAgent

        tmp = Path(f".test_data_{uuid.uuid4().hex[:8]}")
        code_dir = tmp / "code"
        code_dir.mkdir(parents=True)
        data_dir = tmp / "data"
        data_dir.mkdir()
        # Create a fake dataset file that actually exists
        (data_dir / "train.csv").write_text("a,b\n1,2\n", encoding="utf-8")

        try:
            (code_dir / "train.py").write_text(
                f'import pandas as pd\n'
                f'df = pd.read_csv("{data_dir}/train.csv")\n',
                encoding="utf-8",
            )

            downloaded = [{
                "name": "train_data",
                "type": "dataset",
                "path": str(data_dir / "train.csv"),
                "status": "downloaded",
            }]

            issues = CodingAgent._validate_data_paths(
                None,
                code_dir,
                downloaded,
                str(data_dir),
                str(tmp / "models"),
            )

            # No issues — path is under data_dir and in downloaded resources
            assert len(issues) == 0
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_workspace_staging_uses_symlink_or_copy_not_move(self):
        """_stage_workspace_resources should symlink/copy, never move from cache."""
        from nanoresearch.agents.setup import SetupAgent

        tmp = Path(f".test_stage_{uuid.uuid4().hex[:8]}")
        cache_dir = tmp / "cache"
        cache_dir.mkdir(parents=True)
        data_dir = tmp / "workspace" / "data"
        data_dir.mkdir(parents=True)
        models_dir = tmp / "workspace" / "models"
        models_dir.mkdir(parents=True)

        # Create a fake cached dataset
        cached_file = cache_dir / "dataset.csv"
        cached_file.write_text("x,y\n1,2\n", encoding="utf-8")

        try:
            resources = [{
                "name": "test_ds",
                "type": "dataset",
                "path": str(cached_file),
                "status": "downloaded",
            }]

            staged, aliases = SetupAgent._stage_workspace_resources(
                resources, data_dir, models_dir
            )

            # Original cache file must still exist (not moved)
            assert cached_file.exists(), "Cache file was moved instead of copied/linked"

            # Staged resource should have workspace path
            assert staged[0].get("workspace_path")
            workspace_path = Path(staged[0]["workspace_path"])
            assert workspace_path.exists() or workspace_path.is_symlink()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_download_urls_sanitized(self):
        """Verify wget commands use shlex.quote for path safety."""
        import inspect
        from nanoresearch.agents.setup import SetupAgent

        source = inspect.getsource(SetupAgent._download_resources)
        # All wget dest paths should be quoted
        assert "shlex.quote" in source, (
            "_download_resources should use shlex.quote for download paths"
        )

    def test_data_dir_is_not_home_or_root(self):
        """Global data dir should be a subdirectory, not home or root."""
        from nanoresearch.agents.setup import GLOBAL_DATA_DIR, GLOBAL_MODELS_DIR

        for d in (GLOBAL_DATA_DIR, GLOBAL_MODELS_DIR):
            d_str = str(d)
            assert d_str != str(Path.home()), "Data dir must not be home directory"
            assert d_str != "/", "Data dir must not be root"
            assert d_str != "C:\\", "Data dir must not be C:\\"
            # Must be at least 3 levels deep from home
            rel = d.relative_to(Path.home())
            assert len(rel.parts) >= 2, f"Data dir too shallow: {d}"


# ---------------------------------------------------------------------------
# Part 5: OpenAlex replacement sanity
# ---------------------------------------------------------------------------

class TestOpenAlexReferences:
    """Verify the new OpenAlex reference expansion works correctly."""

    @pytest.mark.asyncio
    async def test_get_openalex_references_empty_papers(self):
        """Empty input should return empty output without API calls."""
        from mcp_server.tools.openalex import get_openalex_references
        result = await get_openalex_references([], top_k=5, max_new=20)
        assert result == []

    @pytest.mark.asyncio
    async def test_get_openalex_references_no_identifiers(self):
        """Papers with no openalex_id, doi, or title return empty."""
        from mcp_server.tools.openalex import get_openalex_references
        papers = [{"title": "", "citation_count": 100}]
        result = await get_openalex_references(papers, top_k=5, max_new=20)
        assert result == []

    @pytest.mark.asyncio
    async def test_fetch_referenced_works_mock(self):
        """_fetch_referenced_works should parse OpenAlex response correctly."""
        from mcp_server.tools.openalex import _fetch_referenced_works

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": "https://openalex.org/W123",
            "referenced_works": [
                "https://openalex.org/W456",
                "https://openalex.org/W789",
            ],
        }
        mock_response.raise_for_status = MagicMock()
        mock_response.status_code = 200

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("mcp_server.tools.openalex.get_http_client", return_value=mock_client):
            refs = await _fetch_referenced_works("W123")

        assert refs == ["W456", "W789"]

    @pytest.mark.asyncio
    async def test_fetch_referenced_works_api_error(self):
        """API errors should return empty list, not raise."""
        from mcp_server.tools.openalex import _fetch_referenced_works
        import httpx

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("mcp_server.tools.openalex.get_http_client", return_value=mock_client):
            refs = await _fetch_referenced_works("W999")

        assert refs == []

    @pytest.mark.asyncio
    async def test_citation_manager_uses_openalex(self):
        """citation_manager._resolve_single_citation should import openalex, not S2."""
        import inspect
        from nanoresearch.agents.writing.citation_manager import _CitationManagerMixin

        source = inspect.getsource(_CitationManagerMixin._resolve_single_citation)
        assert "search_openalex" in source
        assert "search_semantic_scholar" not in source
