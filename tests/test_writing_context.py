"""Tests for unified writing context with real execution artifacts."""

from __future__ import annotations

from pathlib import Path
import shutil
import uuid

from nanoresearch.agents.figure_gen import FigureAgent
from nanoresearch.agents.writing import WritingAgent
from nanoresearch.config import ResearchConfig
from nanoresearch.pipeline.workspace import Workspace


def test_build_real_results_context_accepts_completed_status() -> None:
    context = WritingAgent._build_real_results_context(
        {
            "main_results": [
                {
                    "method_name": "DeepMethod",
                    "dataset": "DemoSet",
                    "is_proposed": True,
                    "metrics": [{"metric_name": "accuracy", "value": 0.93}],
                }
            ],
            "ablation_results": [],
        },
        "COMPLETED",
    )

    assert "REAL EXPERIMENT RESULTS" in context
    assert "DeepMethod on DemoSet" in context
    assert "accuracy = 0.93" in context


def test_build_context_includes_experiment_analysis_summary() -> None:
    tmp_dir = Path(f".test_writing_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="writing001",
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = WritingAgent(workspace, config)

        context = agent._build_full_context(
            {"topic": "demo", "papers": [], "gaps": [], "hypotheses": []},
            {
                "proposed_method": {"name": "DeepMethod", "key_components": ["Fusion"]},
                "datasets": [{"name": "DemoSet"}],
                "metrics": [{"name": "accuracy"}],
                "baselines": [{"name": "BaselineNet"}],
                "ablation_groups": [{"group_name": "Fusion"}],
            },
            {},
            {},
            "FAILED",
            {
                "summary": "The model converged during quick-eval.",
                "final_metrics": {"accuracy": 0.91},
                "key_findings": ["Fusion improves robustness."],
                "limitations": ["Training was only validated locally."],
            },
            "# Experiment Summary\n\n- Best round: 2",
        )

        assert "EXPERIMENT ANALYSIS SUMMARY" in context
        assert "The model converged during quick-eval." in context
        assert "Fusion improves robustness." in context
        assert "# Experiment Summary" in context
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_build_context_normalizes_flat_metrics_into_real_results() -> None:
    tmp_dir = Path(f".test_writing_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="writing002",
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = WritingAgent(workspace, config)

        context = agent._build_full_context(
            {"topic": "demo", "papers": [], "gaps": [], "hypotheses": []},
            {
                "proposed_method": {"name": "DeepMethod", "key_components": ["Fusion"]},
                "datasets": [{"name": "DemoSet"}],
                "metrics": [{"name": "accuracy"}],
                "baselines": [{"name": "BaselineNet"}],
                "ablation_groups": [{"group_name": "Fusion"}],
            },
            {},
            {"accuracy": 0.93, "loss": 0.12},
            "COMPLETED",
            {"final_metrics": {"accuracy": 0.93, "loss": 0.12}},
            "",
        )

        assert "REAL EXPERIMENT RESULTS" in context
        assert "DeepMethod on DemoSet" in context
        assert "accuracy = 0.93" in context
        assert "loss = 0.12" in context
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_figure_evidence_block_uses_completed_real_results() -> None:
    block = FigureAgent._build_evidence_block(
        {},
        {},
        {
            "main_results": [
                {
                    "method_name": "DeepMethod",
                    "dataset": "DemoSet",
                    "is_proposed": True,
                    "metrics": [{"metric_name": "accuracy", "value": 0.93}],
                }
            ],
            "ablation_results": [],
            "training_log": [],
        },
        "COMPLETED",
    )

    assert "REAL EXPERIMENT RESULTS" in block
    assert "DeepMethod on DemoSet" in block
