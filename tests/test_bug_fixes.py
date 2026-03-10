"""Focused regression tests for audited bug fixes."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import pytest

from nanoresearch.agents.base import BaseResearchAgent, _repair_truncated_json
from nanoresearch.agents.writing import _escape_latex_text
from nanoresearch.agents.writing.latex_assembler import _LaTeXAssemblerMixin
from nanoresearch.config import ResearchConfig, StageModelConfig
from nanoresearch.latex.fixer import apply_edits
from nanoresearch.pipeline.multi_model import ModelDispatcher
from nanoresearch.pipeline.orchestrator import PipelineOrchestrator
from nanoresearch.pipeline.workspace import Workspace
from nanoresearch.schemas.manifest import PipelineStage


class _DummyAgent(BaseResearchAgent):
    stage = PipelineStage.IDEATION

    async def run(self, **inputs):
        return {}


class _DummyChoice:
    def __init__(self, content: str) -> None:
        self.message = MagicMock(content=content)


class _DummyResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_DummyChoice(content)]
        self.usage = None


def _make_test_root() -> Path:
    base = Path(__file__).resolve().parent / "_codex_tmp"
    base.mkdir(exist_ok=True)
    root = base / uuid.uuid4().hex
    root.mkdir()
    return root


@pytest.mark.asyncio
async def test_generate_json_extracts_fenced_json_after_prose(config):
    root = _make_test_root()
    try:
        workspace = Workspace.create(topic="json test", root=root, session_id="jsonfix01")
        agent = _DummyAgent(workspace, config)
        agent.generate = AsyncMock(return_value='Here is JSON:\n```json\n{"a": 1}\n```')

        result = await agent.generate_json("system", "user")

        assert result == {"a": 1}
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_repair_truncated_json_closes_unterminated_string():
    repaired = _repair_truncated_json('{"a":"hello')

    assert repaired == '{"a":"hello"}'
    assert json.loads(repaired) == {"a": "hello"}


@pytest.mark.asyncio
async def test_exact_o3_uses_thinking_message_shape(config):
    dispatcher = ModelDispatcher(config)
    stage_config = StageModelConfig(model="o3", temperature=None, max_tokens=64)
    mock_client = MagicMock()
    captured_kwargs: dict = {}

    def _create(**kwargs):
        captured_kwargs.update(kwargs)
        return _DummyResponse("ok")

    mock_client.chat.completions.create.side_effect = _create

    with patch.object(dispatcher, "_get_client", return_value=mock_client):
        result = await dispatcher.generate_with_usage(stage_config, "SYS", "USR")

    assert result.content == "ok"
    assert captured_kwargs["messages"] == [{"role": "user", "content": "SYS\n\nUSR"}]
    assert captured_kwargs["max_completion_tokens"] == 64
    assert "max_tokens" not in captured_kwargs


@pytest.mark.asyncio
async def test_json_mode_falls_back_when_backend_rejects_response_format(config):
    dispatcher = ModelDispatcher(config)
    stage_config = StageModelConfig(model="test-model", temperature=0.2, max_tokens=32)
    mock_client = MagicMock()
    call_kwargs: list[dict] = []

    def _create(**kwargs):
        call_kwargs.append(dict(kwargs))
        if len(call_kwargs) == 1:
            raise RuntimeError("response_format json_object is not supported by this backend")
        return _DummyResponse('{"ok": true}')

    mock_client.chat.completions.create.side_effect = _create

    with patch.object(dispatcher, "_get_client", return_value=mock_client):
        result = await dispatcher.generate_with_usage(
            stage_config, "SYS", "USR", json_mode=True,
        )

    assert result.content == '{"ok": true}'
    assert call_kwargs[0]["response_format"] == {"type": "json_object"}
    assert "response_format" not in call_kwargs[1]


def test_escape_latex_text_preserves_commands_and_escapes_plain_chars():
    escaped = _escape_latex_text(r"Caption with \textbf{Model_A} & baseline #1")

    assert escaped == r"Caption with \textbf{Model\_A} \& baseline \#1"


def test_sanitize_latex_escapes_plain_prose_and_caption_text():
    tex = (
        "Accuracy on CIFAR_10 improved by 5% & beat baseline #1.\n"
        r"\caption{Caption with \textbf{Model_A} & baseline}"
    )

    sanitized = _LaTeXAssemblerMixin._sanitize_latex(tex)

    assert r"CIFAR\_10" in sanitized
    assert r"5\%" in sanitized
    assert r"\& beat baseline \#1" in sanitized
    assert r"\caption{Caption with \textbf{Model\_A} \& baseline}" in sanitized


def test_apply_edits_prefers_local_search_window():
    tex = "\n".join([
        r"\begin{table}",
        "row & value",
        r"\end{table}",
        "",
        r"\begin{table}",
        "row & value",
        r"\end{table}",
    ])

    patched = apply_edits(
        tex,
        [{"old": "row & value", "new": r"row \& value"}],
        search_window=(4, 7),
    )

    assert patched is not None
    lines = patched.splitlines()
    assert lines[1] == "row & value"
    assert lines[5] == r"row \& value"


def test_workspace_export_injects_graphics_preamble():
    root = _make_test_root()
    try:
        ws = Workspace.create(topic="Vision model", root=root, session_id="exportfix01")
        ws.write_text(
            "drafts/paper.tex",
            "\n".join([
                r"\documentclass{article}",
                r"\begin{document}",
                r"\includegraphics{fig1.pdf}",
                r"\end{document}",
            ]),
        )
        (ws.figures_dir / "fig1.pdf").write_text("fake-pdf", encoding="utf-8")

        export_dir = ws.export(output_dir=root)
        exported_tex = (export_dir / "paper.tex").read_text(encoding="utf-8")

        assert r"\usepackage{graphicx}" in exported_tex
        assert r"\graphicspath{{figures/}}" in exported_tex
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.mark.asyncio
async def test_skip_stage_pipeline_reaches_done():
    root = _make_test_root()
    try:
        ws = Workspace.create(topic="test topic", root=root, session_id="skipfix01")
        ws.write_text(
            "drafts/paper.tex",
            r"\documentclass{article}\begin{document}Test\end{document}",
        )
        config = ResearchConfig(
            base_url="http://localhost:8000/v1/",
            api_key="test-key",
            skip_stages=["PLANNING"],
        )
        orch = PipelineOrchestrator(ws, config)

        planning_run = AsyncMock(side_effect=AssertionError("PLANNING should be skipped"))

        with patch.object(
            orch._agents[PipelineStage.IDEATION],
            "run",
            return_value={"topic": "test topic", "reference_repos": []},
        ), patch.object(
            orch._agents[PipelineStage.PLANNING],
            "run",
            planning_run,
        ), patch.object(
            orch._agents[PipelineStage.EXPERIMENT],
            "run",
            return_value={"experiment_status": "skipped", "experiment_results": {}},
        ), patch.object(
            orch._agents[PipelineStage.FIGURE_GEN],
            "run",
            return_value={"figures": {}},
        ), patch.object(
            orch._agents[PipelineStage.WRITING],
            "run",
            return_value={"paper_tex": r"\documentclass{article}\begin{document}Test\end{document}"},
        ), patch.object(
            orch._agents[PipelineStage.REVIEW],
            "run",
            return_value={
                "overall_score": 8.0,
                "section_reviews": [],
                "major_revisions": [],
                "minor_revisions": [],
            },
        ):
            result = await orch.run("test topic")

        assert ws.manifest.current_stage == PipelineStage.DONE
        assert "review_output" in result
        assert planning_run.await_count == 0

        await orch.close()
    finally:
        shutil.rmtree(root, ignore_errors=True)
