"""Tests for the pipeline orchestrator with mocked LLM calls."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from nanoresearch.config import ResearchConfig
from nanoresearch.pipeline.orchestrator import PipelineOrchestrator
from nanoresearch.pipeline.workspace import Workspace
from nanoresearch.schemas.manifest import PipelineStage
from nanoresearch.schemas.review import ReviewOutput, SectionReview


def _mock_ideation_response():
    """Return a mock LLM response for ideation analysis."""
    return json.dumps({
        "survey_summary": "The field has seen rapid growth...",
        "gaps": [
            {
                "gap_id": "GAP-001",
                "description": "Limited equivariant methods",
                "supporting_refs": ["1"],
                "severity": "high",
            }
        ],
        "hypotheses": [
            {
                "hypothesis_id": "HYP-001",
                "statement": "Equivariant GNNs improve accuracy",
                "gap_refs": ["GAP-001"],
                "novelty_justification": "Novel combination",
                "feasibility_notes": "Feasible with existing tools",
            }
        ],
        "selected_hypothesis": "HYP-001",
        "rationale": "Most promising hypothesis.",
    })


def _mock_planning_response():
    return json.dumps({
        "title": "EquiFold Experiments",
        "hypothesis_ref": "HYP-001",
        "datasets": [{"name": "CASP14", "description": "Protein dataset"}],
        "baselines": [{"name": "AlphaFold2", "description": "SOTA baseline"}],
        "proposed_method": {"name": "EquiFold", "description": "Our method", "key_components": ["SE(3) layers"]},
        "metrics": [{"name": "GDT-TS", "description": "Accuracy", "higher_is_better": True, "primary": True}],
        "ablation_groups": [{"group_name": "Component Ablation", "description": "Test", "variants": []}],
        "compute_requirements": {"gpu_type": "A100"},
    })


def _mock_project_plan():
    """Return a mock project plan JSON for the experiment agent."""
    return json.dumps({
        "project_name": "equifold",
        "description": "Equivariant GNN for protein folding",
        "python_version": ">=3.9",
        "dependencies": ["torch>=2.0", "numpy"],
        "files": [
            {
                "path": "main.py",
                "description": "Entry point",
                "interfaces": ["def main(): ..."],
                "depends_on": [],
            },
            {
                "path": "src/__init__.py",
                "description": "Package init",
                "interfaces": [],
                "depends_on": [],
            },
        ],
        "interface_contract": "def main(): entry point\n",
    })


def _mock_file_content():
    """Return mock generated file content."""
    return """import torch

def main():
    print("EquiFold experiment")

if __name__ == "__main__":
    main()
"""


def _mock_writing_response():
    return json.dumps({
        "title": "EquiFold: Equivariant GNNs for Protein Folding",
        "authors": ["Anonymous"],
        "abstract": "We propose EquiFold...",
        "sections": [
            {"heading": "Introduction", "label": "sec:intro", "content": "Protein folding is important.", "subsections": []},
            {"heading": "Methods", "label": "sec:methods", "content": "We use equivariant GNNs.", "subsections": []},
        ],
        "figures": [
            {"figure_id": "fig:overview", "caption": "Overview", "figure_type": "placeholder", "data": {"text": "Overview"}},
        ],
    })


def _mock_chart_code(output_path: str) -> str:
    """Return a simple matplotlib script that creates a valid chart."""
    return f"""import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(["A", "B", "C"], [85, 90, 95])
ax.set_title("Test Chart")
ax.set_ylabel("Score")
fig.savefig("{output_path}", dpi=150, bbox_inches="tight")
plt.close(fig)
"""


MOCK_ARXIV_RESPONSE = [
    {
        "paper_id": "2401.00001",
        "title": "Test Paper",
        "authors": ["Author"],
        "year": 2024,
        "abstract": "Abstract",
        "venue": "arXiv",
        "url": "https://arxiv.org/abs/2401.00001",
    }
] * 5


@pytest.fixture
def mock_orchestrator(tmp_path):
    """Create an orchestrator with all LLM calls mocked."""
    ws = Workspace.create(topic="test protein folding", root=tmp_path, session_id="test_e2e")
    config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
    return PipelineOrchestrator(ws, config)


class TestOrchestratorStageInputs:
    def test_prepare_ideation_inputs(self, mock_orchestrator):
        inputs = mock_orchestrator._prepare_inputs(
            PipelineStage.IDEATION, "test topic", {}, ""
        )
        assert inputs["topic"] == "test topic"

    def test_prepare_planning_inputs(self, mock_orchestrator):
        accumulated = {"ideation_output": {"topic": "test", "papers": []}}
        inputs = mock_orchestrator._prepare_inputs(
            PipelineStage.PLANNING, "test", accumulated, ""
        )
        assert "ideation_output" in inputs

    def test_prepare_figure_gen_inputs(self, mock_orchestrator):
        accumulated = {
            "experiment_blueprint": {"title": "test"},
            "ideation_output": {"topic": "test", "evidence": {}},
        }
        inputs = mock_orchestrator._prepare_inputs(
            PipelineStage.FIGURE_GEN, "test", accumulated, ""
        )
        assert "experiment_blueprint" in inputs
        assert "ideation_output" in inputs

    def test_prepare_writing_inputs(self, mock_orchestrator):
        accumulated = {
            "ideation_output": {"topic": "test"},
            "experiment_blueprint": {"title": "test"},
            "experiment_output": {
                "experiment_results": {"main_results": []},
                "experiment_status": "success",
            },
            "figure_output": {"figures": {}},
        }
        inputs = mock_orchestrator._prepare_inputs(
            PipelineStage.WRITING, "test", accumulated, ""
        )
        assert "ideation_output" in inputs
        assert "experiment_blueprint" in inputs
        assert "figure_output" in inputs
        assert "template_format" in inputs
        assert "experiment_results" in inputs
        assert "experiment_status" in inputs

    def test_prepare_figure_gen_inputs_with_experiment(self, mock_orchestrator):
        accumulated = {
            "experiment_blueprint": {"title": "test"},
            "ideation_output": {"topic": "test", "evidence": {}},
            "experiment_output": {
                "experiment_results": {"main_results": [{"method_name": "Ours"}]},
                "experiment_status": "success",
            },
        }
        inputs = mock_orchestrator._prepare_inputs(
            PipelineStage.FIGURE_GEN, "test", accumulated, ""
        )
        assert inputs["experiment_results"] == {"main_results": [{"method_name": "Ours"}]}
        assert inputs["experiment_status"] == "success"

    def test_prepare_review_inputs(self, mock_orchestrator):
        # Write a dummy paper.tex so the orchestrator can read it
        mock_orchestrator.workspace.write_text("drafts/paper.tex", "\\documentclass{article}")
        accumulated = {
            "ideation_output": {"topic": "test"},
            "experiment_blueprint": {"title": "test"},
        }
        inputs = mock_orchestrator._prepare_inputs(
            PipelineStage.REVIEW, "test", accumulated, ""
        )
        assert "paper_tex" in inputs
        assert inputs["paper_tex"] == "\\documentclass{article}"
        assert "ideation_output" in inputs

    def test_retry_error_injection(self, mock_orchestrator):
        inputs = mock_orchestrator._prepare_inputs(
            PipelineStage.IDEATION, "test", {}, "Previous error"
        )
        assert inputs["_retry_error"] == "Previous error"


class TestOrchestratorWrapOutput:
    def test_wrap_ideation(self, mock_orchestrator):
        result = mock_orchestrator._wrap_stage_output(
            PipelineStage.IDEATION, {"papers": []}
        )
        assert "ideation_output" in result

    def test_wrap_planning(self, mock_orchestrator):
        result = mock_orchestrator._wrap_stage_output(
            PipelineStage.PLANNING, {"title": "test"}
        )
        assert "experiment_blueprint" in result

    def test_wrap_experiment(self, mock_orchestrator):
        result = mock_orchestrator._wrap_stage_output(
            PipelineStage.EXPERIMENT, {"experiment_results": {}, "experiment_status": "skipped"}
        )
        assert "experiment_output" in result

    def test_wrap_figure_gen(self, mock_orchestrator):
        result = mock_orchestrator._wrap_stage_output(
            PipelineStage.FIGURE_GEN, {"figures": {}}
        )
        assert "figure_output" in result

    def test_wrap_review(self, mock_orchestrator):
        result = mock_orchestrator._wrap_stage_output(
            PipelineStage.REVIEW, {"overall_score": 8.0}
        )
        assert "review_output" in result


class TestOrchestratorIntegration:
    @pytest.mark.asyncio
    async def test_full_pipeline_mocked(self, tmp_path):
        """Run the full pipeline with all LLM and search calls mocked."""
        ws = Workspace.create(topic="test topic", root=tmp_path, session_id="integration")
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        orch = PipelineOrchestrator(ws, config)

        # Track which generate calls we make
        call_count = 0

        def _mock_figure_plan():
            """Return a mock figure plan with diverse chart types."""
            return json.dumps({
                "figures": [
                    {
                        "fig_key": "fig1_architecture",
                        "fig_type": "ai_image",
                        "chart_type": None,
                        "title": "Model Architecture",
                        "description": "Overview of the proposed method.",
                        "caption": "Architecture overview.",
                    },
                    {
                        "fig_key": "fig2_results",
                        "fig_type": "code_chart",
                        "chart_type": "grouped_bar",
                        "title": "Main Results",
                        "description": "Baseline comparison.",
                        "caption": "Performance comparison.",
                    },
                    {
                        "fig_key": "fig3_ablation",
                        "fig_type": "code_chart",
                        "chart_type": "horizontal_bar",
                        "title": "Ablation Study",
                        "description": "Component analysis.",
                        "caption": "Ablation study results.",
                    },
                ]
            })

        async def mock_generate(cfg, sys_prompt, user_prompt, json_mode=False):
            nonlocal call_count
            call_count += 1

            # Content-based routing (handles concurrent figure gen calls)
            combined = (sys_prompt + " " + user_prompt).lower()

            # Figure gen: architecture image prompt
            if "image generation prompt" in combined or "prompt construction" in combined:
                return "A detailed scientific figure prompt for testing"

            # Figure gen: chart code generation
            if "chart" in combined and ("matplotlib" in combined or "publication-quality" in combined):
                # Generate chart code that creates a valid PNG
                fig_path = str(ws.path / "figures" / "fig_temp.png")
                return _mock_chart_code(fig_path)

            # Sequence-based routing for pipeline stages
            if call_count == 1:
                return json.dumps({"queries": ["test query 1", "test query 2"]})
            elif call_count == 2:
                return _mock_ideation_response()
            elif call_count == 3:
                return json.dumps({
                    "extracted_metrics": [
                        {
                            "paper_id": "2401.00001",
                            "paper_title": "Test Paper",
                            "dataset": "CASP14",
                            "metric_name": "GDT-TS",
                            "value": 92.4,
                            "unit": "",
                            "context": "achieves 92.4 GDT-TS",
                            "method_name": "AlphaFold2",
                            "higher_is_better": True,
                        }
                    ],
                    "extraction_notes": "Extracted 1 metric",
                    "coverage_warnings": [],
                })
            elif call_count == 4:
                return _mock_planning_response()
            elif call_count == 5:
                return _mock_project_plan()
            elif call_count <= 7:
                return _mock_file_content()
            else:
                return "Mock section content for testing."

        async def mock_generate_image(cfg, prompt, **kwargs):
            # Return a tiny 1x1 white PNG as base64
            import base64
            # Minimal valid PNG (1x1 white pixel)
            png_bytes = (
                b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01'
                b'\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00'
                b'\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00'
                b'\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82'
            )
            return [base64.b64encode(png_bytes).decode()]

        async def mock_github_search(topic, queries):
            """Mock GitHub search to avoid real HTTP calls."""
            return []

        def _mock_review_response():
            return json.dumps({
                "overall_score": 7.5,
                "section_reviews": [
                    {"section": "Introduction", "score": 8, "issues": [], "suggestions": []},
                    {"section": "Methods", "score": 7, "issues": [], "suggestions": []},
                ],
                "major_revisions": [],
                "minor_revisions": ["Minor formatting fix"],
            })

        # Write a dummy paper.tex so the REVIEW stage has content to read
        ws.write_text("drafts/paper.tex", "\\documentclass{article}\\begin{document}Test\\end{document}")

        async def mock_search_surveys(topic):
            return []

        async def mock_enrich(papers, top_k=5):
            return papers

        async def mock_must_cites(surveys):
            return []

        async def mock_execute_code_with_venv(files, blueprint):
            return ({"status": "success", "attempts": 1, "returncode": 0, "stdout": "", "stderr": ""}, "python")

        async def mock_run_quick_eval(code_dir, venv_python, timeout=300):
            return {
                "status": "success",
                "metrics": {
                    "main_results": [
                        {"method_name": "Ours", "dataset": "Test", "is_proposed": True,
                         "metrics": [{"metric_name": "Accuracy", "value": 85.0, "std": 0.3}]},
                    ],
                    "ablation_results": [],
                    "training_log": [],
                },
                "attempts": 1,
            }

        # Mock each agent's run() to return appropriate stage outputs
        mock_ideation_output = {
            "topic": "test topic",
            "selected_hypothesis": "HYP-001",
            "hypotheses": [{"hypothesis_id": "HYP-001", "statement": "Equivariant GNNs improve accuracy"}],
            "evidence": {"extracted_metrics": [{"method_name": "AlphaFold2", "dataset": "CASP14", "metric_name": "GDT-TS", "value": 92.4}]},
            "reference_repos": [],
        }
        mock_blueprint = json.loads(_mock_planning_response())
        mock_experiment_output = {
            "experiment_status": "success",
            "experiment_results": {
                "main_results": [
                    {"method_name": "Ours", "dataset": "CASP14", "is_proposed": True,
                     "metrics": [{"metric_name": "GDT-TS", "value": 95.0, "std": 0.3}]},
                ],
                "ablation_results": [],
                "training_log": [],
            },
        }
        mock_figure_output = {"figures": {"fig1_architecture": {"path": "figures/fig1.png"}}}
        mock_writing_output = {"paper_tex": "\\documentclass{article}\\begin{document}Test\\end{document}"}
        mock_review_output = ReviewOutput(
            overall_score=8.0,
            section_reviews=[
                SectionReview(section="Introduction", score=8, issues=[], suggestions=[]),
                SectionReview(section="Methods", score=8, issues=[], suggestions=[]),
            ],
            major_revisions=[],
            minor_revisions=[],
        ).model_dump(mode="json")

        with patch.object(orch._agents[PipelineStage.IDEATION], "run", return_value=mock_ideation_output), \
             patch.object(orch._agents[PipelineStage.PLANNING], "run", return_value=mock_blueprint), \
             patch.object(orch._agents[PipelineStage.EXPERIMENT], "run", return_value=mock_experiment_output), \
             patch.object(orch._agents[PipelineStage.FIGURE_GEN], "run", return_value=mock_figure_output), \
             patch.object(orch._agents[PipelineStage.WRITING], "run", return_value=mock_writing_output), \
             patch.object(orch._agents[PipelineStage.REVIEW], "run", return_value=mock_review_output):

            result = await orch.run("test topic")

        assert ws.manifest.current_stage == PipelineStage.DONE
        assert "ideation_output" in result
        assert "experiment_blueprint" in result
        assert "experiment_output" in result
        assert "figure_output" in result
        assert "review_output" in result

        # Verify experiment results flow through correctly
        exp_out = result["experiment_output"]
        assert exp_out["experiment_status"] == "success"
        assert exp_out["experiment_results"]["main_results"][0]["method_name"] == "Ours"

        await orch.close()
