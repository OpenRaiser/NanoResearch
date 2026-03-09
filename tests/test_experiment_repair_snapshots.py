"""Focused tests for reversible experiment repair snapshots."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
from unittest.mock import AsyncMock
import uuid
import zipfile

import pytest

from nanoresearch.agents.execution import ExecutionAgent
from nanoresearch.agents.experiment import ExperimentAgent
from nanoresearch.agents.repair_journal import REPAIR_SNAPSHOT_JOURNAL_PATH
from nanoresearch.config import ResearchConfig
from nanoresearch.pipeline.workspace import Workspace
from nanoresearch.schemas.iteration import ExperimentHypothesis


def _build_workspace(tmp_dir: Path, session_id: str) -> tuple[Workspace, ResearchConfig]:
    workspace = Workspace.create(
        topic="test",
        root=tmp_dir,
        session_id=session_id,
    )
    config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
    return workspace, config


def test_resource_path_repair_records_snapshot_metadata_for_rewrite() -> None:
    tmp_dir = Path(f".test_repair_snapshots_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace, config = _build_workspace(tmp_dir, "repairsnap001")
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
            scope="unit_resource_rewrite",
        )

        assert modified == ["train.py"]
        journal = workspace.read_json(REPAIR_SNAPSHOT_JOURNAL_PATH)
        entry = journal["entries"][0]
        assert entry["mutation_kind"] == "resource_path_repair"
        assert entry["scope"] == "unit_resource_rewrite"
        assert entry["metadata"]["modified_files"] == ["train.py"]

        snapshot = entry["snapshots"][0]
        assert snapshot["path"] == "experiment/train.py"
        assert snapshot["existed_before"] is True
        assert snapshot["snapshot_path"].startswith("logs/repair_snapshots/resource_path_repair/")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_resource_path_repair_marks_created_targets_as_new_files() -> None:
    tmp_dir = Path(f".test_repair_snapshots_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace, config = _build_workspace(tmp_dir, "repairsnap002")
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
            scope="unit_resource_create",
        )

        assert str(missing_target) in modified
        journal = workspace.read_json(REPAIR_SNAPSHOT_JOURNAL_PATH)
        entry = journal["entries"][0]
        snapshot = entry["snapshots"][0]
        assert snapshot["path"] == "data/train.csv"
        assert snapshot["existed_before"] is False
        assert snapshot["operation"] == "create"
        assert snapshot["snapshot_path"] == ""
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_fix_timeout_records_snapshot_metadata() -> None:
    tmp_dir = Path(f".test_repair_snapshots_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace, config = _build_workspace(tmp_dir, "repairsnap003")
        agent = ExperimentAgent(workspace, config)

        code_dir = workspace.path / "experiment"
        code_dir.mkdir(exist_ok=True)
        main_py = code_dir / "main.py"
        main_py.write_text(
            "epochs = 10\nsubset_size = 1000\nnum_runs = 3\nnum_workers = 4\n",
            encoding="utf-8",
        )

        modified = await agent._fix_timeout(code_dir)

        assert modified == ["main.py"]
        journal = workspace.read_json(REPAIR_SNAPSHOT_JOURNAL_PATH)
        entry = journal["entries"][0]
        assert entry["mutation_kind"] == "timeout_fix"
        assert entry["metadata"]["modified_files"] == ["main.py"]
        snapshot = entry["snapshots"][0]
        assert snapshot["path"] == "experiment/main.py"
        assert snapshot["existed_before"] is True
        assert "epochs = 2" in main_py.read_text(encoding="utf-8")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_apply_iteration_changes_rolls_back_invalid_python() -> None:
    tmp_dir = Path(f".test_repair_snapshots_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace, config = _build_workspace(tmp_dir, "repairsnap004")
        agent = ExperimentAgent(workspace, config)

        code_dir = workspace.path / "experiment"
        code_dir.mkdir(exist_ok=True)
        main_py = code_dir / "main.py"
        original = "value = 1\nprint(value)\n"
        main_py.write_text(original, encoding="utf-8")

        agent._dispatcher.generate = AsyncMock(
            return_value=json.dumps(
                [
                    {
                        "path": "main.py",
                        "action": "edit",
                        "edits": [{"old": "value = 1", "new": "value = ("}],
                    }
                ]
            )
        )

        modified = await agent._apply_iteration_changes(
            ExperimentHypothesis(
                round_number=2,
                hypothesis="Break syntax",
                planned_changes=["main.py: change assignment"],
                expected_signal="No syntax errors",
                rationale="Test rollback",
            ),
            code_dir,
            sys.executable,
        )

        assert modified == []
        assert main_py.read_text(encoding="utf-8") == original
        journal = workspace.read_json(REPAIR_SNAPSHOT_JOURNAL_PATH)
        entry = journal["entries"][0]
        assert entry["mutation_kind"] == "iteration_changes"
        snapshot = entry["snapshots"][0]
        assert snapshot["path"] == "experiment/main.py"
        assert snapshot["rolled_back"] is True
        assert snapshot["rollback_reason"] == "syntax_error"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_apply_iteration_changes_matches_definition_block_when_old_body_drifted() -> None:
    tmp_dir = Path(f".test_repair_snapshots_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace, config = _build_workspace(tmp_dir, "repairsnap004b")
        agent = ExperimentAgent(workspace, config)

        code_dir = workspace.path / "experiment"
        code_dir.mkdir(exist_ok=True)
        train_py = code_dir / "train.py"
        original = (
            "import argparse\n\n"
            "def main():\n"
            "    parser = argparse.ArgumentParser(description='Train baseline')\n"
            "    parser.add_argument('--epochs', type=int, default=50)\n"
            "    args = parser.parse_args()\n"
            "    return args\n"
        )
        train_py.write_text(original, encoding="utf-8")

        agent._dispatcher.generate = AsyncMock(
            return_value=json.dumps(
                [
                    {
                        "path": "train.py",
                        "action": "edit",
                        "edits": [
                            {
                                "old": (
                                    "def main():\n"
                                    "    parser = argparse.ArgumentParser(description='Train SmokeMLP')\n"
                                    "    parser.add_argument('--epochs', type=int, default=50)\n"
                                    "    return parser.parse_args()\n"
                                ),
                                "new": (
                                    "def main():\n"
                                    "    parser = argparse.ArgumentParser(description='Train SmokeMLP')\n"
                                    "    parser.add_argument('--epochs', type=int, default=5)\n"
                                    "    return parser.parse_args()\n"
                                ),
                            }
                        ],
                    }
                ]
            )
        )

        modified = await agent._apply_iteration_changes(
            ExperimentHypothesis(
                round_number=2,
                hypothesis="Tighten quick-eval defaults",
                planned_changes=["train.py: update main argparse defaults"],
                expected_signal="Lower epoch count",
                rationale="Test definition-block fallback",
            ),
            code_dir,
            sys.executable,
        )

        updated = train_py.read_text(encoding="utf-8")
        assert modified == ["train.py"]
        assert "default=5" in updated
        assert "description='Train SmokeMLP'" in updated
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_apply_iteration_changes_matches_rstrip_line_blocks() -> None:
    tmp_dir = Path(f".test_repair_snapshots_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace, config = _build_workspace(tmp_dir, "repairsnap004c")
        agent = ExperimentAgent(workspace, config)

        code_dir = workspace.path / "experiment"
        code_dir.mkdir(exist_ok=True)
        main_py = code_dir / "main.py"
        original = "value = 1   \nprint(value)   \n"
        main_py.write_text(original, encoding="utf-8")

        agent._dispatcher.generate = AsyncMock(
            return_value=json.dumps(
                [
                    {
                        "path": "main.py",
                        "action": "edit",
                        "edits": [
                            {
                                "old": "value = 1\nprint(value)\n",
                                "new": "value = 2\nprint(value)\n",
                            }
                        ],
                    }
                ]
            )
        )

        modified = await agent._apply_iteration_changes(
            ExperimentHypothesis(
                round_number=2,
                hypothesis="Normalize trailing whitespace handling",
                planned_changes=["main.py: update constant"],
                expected_signal="Patch should still apply",
                rationale="Test rstrip-line fallback",
            ),
            code_dir,
            sys.executable,
        )

        updated = main_py.read_text(encoding="utf-8")
        assert modified == ["main.py"]
        assert "value = 2" in updated
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_apply_iteration_changes_fullwrite_rolls_back_invalid_python() -> None:
    tmp_dir = Path(f".test_repair_snapshots_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace, config = _build_workspace(tmp_dir, "repairsnap005")
        agent = ExperimentAgent(workspace, config)

        code_dir = workspace.path / "experiment"
        code_dir.mkdir(exist_ok=True)
        main_py = code_dir / "main.py"
        original = "def run():\n    return 1\n"
        main_py.write_text(original, encoding="utf-8")

        agent._dispatcher.generate = AsyncMock(
            return_value=(
                "def broken_function()\n"
                "    print('oops')\n"
                "\n"
                "class Holder:\n"
                "    pass\n"
            )
        )

        modified = await agent._apply_iteration_changes_fullwrite(
            ExperimentHypothesis(
                round_number=2,
                hypothesis="Rewrite main",
                planned_changes=["main.py: rewrite training loop"],
                expected_signal="No syntax errors",
                rationale="Test fullwrite rollback",
            ),
            code_dir,
        )

        assert modified == []
        assert main_py.read_text(encoding="utf-8") == original
        journal = workspace.read_json(REPAIR_SNAPSHOT_JOURNAL_PATH)
        entry = journal["entries"][0]
        assert entry["mutation_kind"] == "iteration_fullwrite"
        snapshot = entry["snapshots"][0]
        assert snapshot["path"] == "experiment/main.py"
        assert snapshot["rolled_back"] is True
        assert snapshot["rollback_reason"] == "syntax_error"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
