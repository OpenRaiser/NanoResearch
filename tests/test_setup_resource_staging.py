"""Tests for workspace-local resource staging in setup."""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import AsyncMock
import uuid

import pytest

import nanoresearch.agents.setup as setup_agent_module
from nanoresearch.agents.setup import SetupAgent
from nanoresearch.config import ResearchConfig
from nanoresearch.pipeline.workspace import Workspace


@pytest.mark.asyncio
async def test_setup_stages_cached_resources_into_workspace(monkeypatch) -> None:
    tmp_dir = Path(f".test_setup_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        cache_root = tmp_dir / "cache"
        cache_data = cache_root / "data"
        cache_models = cache_root / "models"
        cache_data.mkdir(parents=True)
        cache_models.mkdir(parents=True)

        dataset_file = cache_data / "demo.csv"
        dataset_file.write_text("x,y\n1,2\n", encoding="utf-8")

        model_dir = cache_models / "demo_model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text('{"hidden_size": 16}', encoding="utf-8")

        monkeypatch.setattr(setup_agent_module, "GLOBAL_CACHE_DIR", cache_root)
        monkeypatch.setattr(setup_agent_module, "GLOBAL_DATA_DIR", cache_data)
        monkeypatch.setattr(setup_agent_module, "GLOBAL_MODELS_DIR", cache_models)

        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="setup_stage",
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = SetupAgent(workspace, config)

        agent._plan_search = AsyncMock(return_value={"datasets": []})  # type: ignore[method-assign]
        agent._search_and_clone = AsyncMock(return_value=[])  # type: ignore[method-assign]
        agent._analyze_cloned_code = AsyncMock(return_value={})  # type: ignore[method-assign]
        agent._download_resources = AsyncMock(  # type: ignore[method-assign]
            return_value=[
                {
                    "name": "DemoSet",
                    "type": "dataset",
                    "path": str(dataset_file),
                    "status": "downloaded",
                },
                {
                    "name": "DemoModel",
                    "type": "model",
                    "path": str(model_dir),
                    "status": "full",
                },
            ]
        )

        result = await agent.run(
            topic="demo",
            ideation_output={},
            experiment_blueprint={"datasets": [{"name": "DemoSet"}]},
        )

        workspace_data = workspace.path / "data"
        workspace_models = workspace.path / "models"
        assert result["data_dir"] == str(workspace_data)
        assert result["models_dir"] == str(workspace_models)
        assert result["cache_data_dir"] == str(cache_data)
        assert result["cache_models_dir"] == str(cache_models)

        dataset_resource = next(
            item for item in result["downloaded_resources"] if item["type"] == "dataset"
        )
        model_resource = next(
            item for item in result["downloaded_resources"] if item["type"] == "model"
        )

        assert dataset_resource["cache_path"] == str(dataset_file)
        assert Path(dataset_resource["path"]).exists()
        assert Path(dataset_resource["path"]).parent == workspace_data
        assert (workspace_data / "demo.csv").exists()

        assert model_resource["cache_path"] == str(model_dir)
        assert Path(model_resource["path"]).exists()
        assert Path(model_resource["path"]).parent == workspace_models
        assert (Path(model_resource["path"]) / "config.json").exists()

        alias_paths = {item["workspace_path"] for item in result["workspace_resource_aliases"]}
        assert str(workspace_data / "demo.csv") in alias_paths
        assert str(workspace_models / "demo_model") in alias_paths
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_setup_stages_directory_resource_file_list_into_workspace(monkeypatch) -> None:
    tmp_dir = Path(f".test_setup_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        cache_root = tmp_dir / "cache"
        cache_data = cache_root / "data"
        cache_models = cache_root / "models"
        cache_data.mkdir(parents=True)
        cache_models.mkdir(parents=True)

        archived_file = cache_data / "archive.csv"
        archived_file.write_text("value\n1\n", encoding="utf-8")

        monkeypatch.setattr(setup_agent_module, "GLOBAL_CACHE_DIR", cache_root)
        monkeypatch.setattr(setup_agent_module, "GLOBAL_DATA_DIR", cache_data)
        monkeypatch.setattr(setup_agent_module, "GLOBAL_MODELS_DIR", cache_models)

        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="setup_dir_stage",
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = SetupAgent(workspace, config)

        agent._plan_search = AsyncMock(return_value={"datasets": []})  # type: ignore[method-assign]
        agent._search_and_clone = AsyncMock(return_value=[])  # type: ignore[method-assign]
        agent._analyze_cloned_code = AsyncMock(return_value={})  # type: ignore[method-assign]
        agent._download_resources = AsyncMock(  # type: ignore[method-assign]
            return_value=[
                {
                    "name": "ArchiveSet",
                    "type": "dataset",
                    "path": str(cache_data),
                    "status": "downloaded",
                    "files": ["archive.csv"],
                }
            ]
        )

        result = await agent.run(
            topic="demo",
            ideation_output={},
            experiment_blueprint={"datasets": [{"name": "ArchiveSet"}]},
        )

        resource = result["downloaded_resources"][0]
        assert resource["path"] == str(workspace.path / "data")
        assert resource["workspace_path"] == str(workspace.path / "data")
        assert (workspace.path / "data" / "archive.csv").exists()
        assert resource["workspace_files"] == [str(workspace.path / "data" / "archive.csv")]
        assert result["workspace_resource_aliases"][0]["workspace_files"] == [
            str(workspace.path / "data" / "archive.csv")
        ]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
