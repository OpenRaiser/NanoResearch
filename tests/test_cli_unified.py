"""Tests for the unified CLI entrypoint."""

from __future__ import annotations

from pathlib import Path
import shutil
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

from nanoresearch.cli import run
from nanoresearch.config import ExecutionProfile, ResearchConfig
from nanoresearch.pipeline.workspace import Workspace
from nanoresearch.schemas.manifest import PipelineMode


def test_run_uses_unified_deep_workspace() -> None:
    tmp_dir = Path(f".test_cli_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        workspace = Workspace.create(
            topic="test topic",
            root=tmp_dir,
            session_id="cli001",
            pipeline_mode=PipelineMode.DEEP,
        )
        fake_orchestrator = MagicMock()

        with (
            patch("nanoresearch.cli._setup_logging"),
            patch("nanoresearch.cli._load_config_safe", return_value=config),
            patch("nanoresearch.cli.Workspace.create", return_value=workspace) as create_mock,
            patch("nanoresearch.cli.UnifiedPipelineOrchestrator", return_value=fake_orchestrator) as orchestrator_cls,
            patch("nanoresearch.cli._run_deep_pipeline", MagicMock(return_value={"ok": True})),
            patch("nanoresearch.cli.asyncio.run", return_value={"ok": True}),
            patch("nanoresearch.cli._print_result"),
        ):
            run(
                topic="test topic",
                format=None,
                config_path=None,
                profile=ExecutionProfile.LOCAL_QUICK,
                verbose=False,
                dry_run=False,
            )

        assert config.execution_profile == ExecutionProfile.LOCAL_QUICK
        assert create_mock.call_args.kwargs["pipeline_mode"] == PipelineMode.DEEP
        orchestrator_cls.assert_called_once()
        call_args = orchestrator_cls.call_args
        assert call_args[0] == (workspace, config) or (
            call_args[0][0] is workspace and call_args[0][1] is config
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
