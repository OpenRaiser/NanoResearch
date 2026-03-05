"""Tests for the ReviewAgent."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from nanoresearch.config import ResearchConfig
from nanoresearch.pipeline.workspace import Workspace
from nanoresearch.schemas.manifest import PipelineStage


@pytest.fixture
def review_workspace(tmp_path: Path) -> Workspace:
    ws = Workspace.create(topic="test review", root=tmp_path, session_id="test_review")
    ws.write_text("drafts/paper.tex", SAMPLE_TEX)
    return ws


@pytest.fixture
def review_config() -> ResearchConfig:
    return ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")


SAMPLE_TEX = r"""\documentclass{article}
\begin{document}
\section{Introduction}
This is the introduction to our work on equivariant GNNs.

\section{Methods}
We propose a novel method based on SE(3)-equivariant layers.

\section{Experiments}
We evaluate on CASP14 dataset.

\section{Conclusion}
We conclude that our method improves upon baselines.
\end{document}
"""

MOCK_REVIEW_RESPONSE = {
    "overall_score": 6.5,
    "section_reviews": [
        {"section": "Introduction", "score": 7, "issues": ["Too brief"], "suggestions": ["Add motivation"]},
        {"section": "Methods", "score": 5, "issues": ["Missing details"], "suggestions": ["Add equations"]},
        {"section": "Experiments", "score": 7, "issues": [], "suggestions": []},
        {"section": "Conclusion", "score": 8, "issues": [], "suggestions": []},
    ],
    "major_revisions": ["Methods section needs more detail"],
    "minor_revisions": ["Fix minor typos"],
}


class TestReviewAgent:
    @pytest.mark.asyncio
    async def test_review_empty_paper(self, review_workspace, review_config):
        from nanoresearch.agents.review import ReviewAgent

        agent = ReviewAgent(review_workspace, review_config)
        result = await agent.run(paper_tex="", ideation_output={}, experiment_blueprint={})
        assert result["overall_score"] == 0.0

    @pytest.mark.asyncio
    async def test_review_with_mock(self, review_workspace, review_config):
        from nanoresearch.agents.review import ReviewAgent
        from nanoresearch.schemas.review import ReviewOutput, SectionReview

        agent = ReviewAgent(review_workspace, review_config)

        review_call_count = 0

        # Initial review: Methods scores below MIN_SECTION_SCORE (8)
        initial_review = ReviewOutput(
            overall_score=6.5,
            section_reviews=[
                SectionReview(section="Introduction", score=7, issues=["Too brief"], suggestions=["Add motivation"]),
                SectionReview(section="Methods", score=5, issues=["Missing details"], suggestions=["Add equations"]),
                SectionReview(section="Experiments", score=7, issues=[], suggestions=[]),
                SectionReview(section="Conclusion", score=8, issues=[], suggestions=[]),
            ],
            major_revisions=["Methods section needs more detail"],
            minor_revisions=["Fix minor typos"],
        )

        # Re-review after revision: all scores >= 8
        improved_review = ReviewOutput(
            overall_score=7.75,
            section_reviews=[
                SectionReview(section="Introduction", score=8, issues=[], suggestions=[]),
                SectionReview(section="Methods", score=8, issues=[], suggestions=[]),
                SectionReview(section="Experiments", score=7, issues=[], suggestions=[]),
                SectionReview(section="Conclusion", score=8, issues=[], suggestions=[]),
            ],
            major_revisions=[],
            minor_revisions=[],
        )

        async def mock_review_paper(paper_tex, ideation_output, experiment_blueprint):
            nonlocal review_call_count
            review_call_count += 1
            if review_call_count == 1:
                return initial_review
            return improved_review

        async def mock_generate(cfg, sys_prompt, user_prompt, json_mode=False):
            # Revision calls return revised text
            return "Revised section content with more detail and equations."

        agent._dispatcher.generate = mock_generate

        with patch.object(agent, "_review_paper", side_effect=mock_review_paper), \
             patch.object(agent, "_run_consistency_checks", return_value=[]), \
             patch.object(agent, "_compile_pdf_with_fix_loop", return_value={"pdf_path": "/tmp/paper.pdf"}):
            result = await agent.run(
                paper_tex=SAMPLE_TEX,
                ideation_output={"topic": "test", "selected_hypothesis": "HYP-001"},
                experiment_blueprint={"proposed_method": {"name": "EquiFold"}},
            )

        assert result["overall_score"] > 0
        assert len(result["section_reviews"]) == 4
        # Methods scored 5 initially, should trigger revision
        assert result["revision_rounds"] >= 1
        # After re-review, Methods score should be 8
        methods_review = [sr for sr in result["section_reviews"] if sr["section"] == "Methods"]
        assert methods_review[0]["score"] == 8

        # Revised paper should be written back to paper.tex
        paper_tex_path = review_workspace.path / "drafts" / "paper.tex"
        assert paper_tex_path.exists()
        # Backup should also exist
        revised_tex_path = review_workspace.path / "drafts" / "paper_revised.tex"
        assert revised_tex_path.exists()

    @pytest.mark.asyncio
    async def test_no_revision_needed(self, review_workspace, review_config):
        from nanoresearch.agents.review import ReviewAgent
        from nanoresearch.schemas.review import ReviewOutput, SectionReview

        agent = ReviewAgent(review_workspace, review_config)

        high_score_review = ReviewOutput(
            overall_score=8.5,
            section_reviews=[
                SectionReview(section="Introduction", score=9, issues=[], suggestions=[]),
                SectionReview(section="Methods", score=8, issues=[], suggestions=[]),
                SectionReview(section="Experiments", score=9, issues=[], suggestions=[]),
                SectionReview(section="Conclusion", score=8, issues=[], suggestions=[]),
            ],
            major_revisions=[],
            minor_revisions=[],
        )

        with patch.object(agent, "_review_paper", return_value=high_score_review), \
             patch.object(agent, "_run_consistency_checks", return_value=[]):
            result = await agent.run(
                paper_tex=SAMPLE_TEX,
                ideation_output={"topic": "test"},
                experiment_blueprint={},
            )

        assert result["revision_rounds"] == 0
        assert result["revised_sections"] == {}

    def test_apply_revisions(self):
        from nanoresearch.agents.review import ReviewAgent

        revised = ReviewAgent._apply_revisions(
            SAMPLE_TEX,
            {"Introduction": "A much better introduction with more context."},
        )
        assert "A much better introduction" in revised
        # Original section content should be replaced
        assert "This is the introduction" not in revised
