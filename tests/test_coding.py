from __future__ import annotations

from pathlib import Path
import shutil
import uuid
from unittest.mock import AsyncMock

import pytest

from nanoresearch.agents.coding import CodingAgent
from nanoresearch.config import ResearchConfig
from nanoresearch.exceptions import LLMError
from nanoresearch.pipeline.workspace import Workspace


@pytest.mark.asyncio
async def test_design_code_plan_retries_with_minimal_schema_on_json_failure() -> None:
    tmp_dir = Path(f".test_coding_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="coding_retry_001",
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = CodingAgent(workspace, config)

        agent.generate_json = AsyncMock(
            side_effect=[
                LLMError("broken json"),
                {
                    "project_name": "retry_plan",
                    "files": [
                        {
                            "path": "train.py",
                            "description": "entry",
                            "is_entrypoint": True,
                        }
                    ],
                    "train_command": "python train.py --epochs 3",
                },
            ]
        )

        result = await agent._design_code_plan(
            "topic",
            {"datasets": [], "metrics": [], "baselines": [], "proposed_method": {}},
            {"downloaded_resources": [], "code_analysis": {}, "cloned_repos": []},
        )

        assert result["project_name"] == "retry_plan"
        assert result["train_command"] == "python train.py --epochs 3"
        assert any(item["path"] == "train.py" for item in result["files"])
        assert any(item["path"] == "model.py" for item in result["files"])
        assert result["dependencies"]
        assert agent.generate_json.await_count == 2
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
