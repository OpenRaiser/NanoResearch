"""Tests for nanoresearch.schemas.ideation."""

from __future__ import annotations

import pytest

from nanoresearch.schemas.evidence import EvidenceBundle
from nanoresearch.schemas.ideation import (
    GapAnalysis,
    Hypothesis,
    IdeationOutput,
    PaperReference,
)


class TestPaperReference:
    """Tests for PaperReference model."""

    def test_minimal_valid(self) -> None:
        pr = PaperReference(paper_id="arXiv:2401.00001", title="Test Paper")
        assert pr.paper_id == "arXiv:2401.00001"
        assert pr.title == "Test Paper"
        assert pr.authors == []
        assert pr.year is None
        assert pr.relevance_score == 0.0

    def test_full_fields(self) -> None:
        pr = PaperReference(
            paper_id="arXiv:2401.00001",
            title="Test Paper",
            authors=["Alice", "Bob"],
            year=2024,
            abstract="An abstract.",
            relevance_score=0.85,
        )
        assert pr.authors == ["Alice", "Bob"]
        assert pr.year == 2024
        assert pr.relevance_score == 0.85

    def test_coerce_list_to_str(self) -> None:
        pr = PaperReference(
            paper_id=["a", "b"],
            title="Test",
        )
        assert pr.paper_id == "a; b"

    def test_relevance_score_bounds(self) -> None:
        pr = PaperReference(paper_id="x", title="t", relevance_score=1.0)
        assert pr.relevance_score == 1.0
        pr2 = PaperReference(paper_id="x", title="t", relevance_score=0.0)
        assert pr2.relevance_score == 0.0


class TestGapAnalysis:
    """Tests for GapAnalysis model."""

    def test_minimal_valid(self) -> None:
        ga = GapAnalysis(gap_id="GAP-001", description="Missing X")
        assert ga.gap_id == "GAP-001"
        assert ga.description == "Missing X"
        assert ga.severity == "medium"
        assert ga.supporting_refs == []

    def test_severity_normalization(self) -> None:
        ga = GapAnalysis(gap_id="G", description="D", severity="HIGH")
        assert ga.severity == "high"

    def test_invalid_severity_defaults_to_medium(self) -> None:
        ga = GapAnalysis(gap_id="G", description="D", severity="invalid")
        assert ga.severity == "medium"


class TestHypothesis:
    """Tests for Hypothesis model."""

    def test_minimal_valid(self) -> None:
        h = Hypothesis(hypothesis_id="HYP-001", statement="We hypothesize X")
        assert h.hypothesis_id == "HYP-001"
        assert h.statement == "We hypothesize X"
        assert h.gap_refs == []
        assert h.novelty_justification == ""

    def test_full_fields(self) -> None:
        h = Hypothesis(
            hypothesis_id="HYP-001",
            statement="X causes Y",
            gap_refs=["GAP-001"],
            novelty_justification="First to combine A and B",
        )
        assert h.gap_refs == ["GAP-001"]
        assert "First" in h.novelty_justification


class TestIdeationOutput:
    """Tests for IdeationOutput model."""

    def test_minimal_valid(self) -> None:
        out = IdeationOutput(topic="Machine learning for materials")
        assert out.topic == "Machine learning for materials"
        assert out.papers == []
        assert out.gaps == []
        assert out.hypotheses == []
        assert isinstance(out.evidence, EvidenceBundle)

    def test_with_papers_and_hypotheses(self) -> None:
        pr = PaperReference(paper_id="p1", title="P1")
        ga = GapAnalysis(gap_id="G1", description="Gap 1")
        h = Hypothesis(hypothesis_id="H1", statement="Hyp 1")
        out = IdeationOutput(
            topic="ML",
            papers=[pr],
            gaps=[ga],
            hypotheses=[h],
            selected_hypothesis="H1",
        )
        assert len(out.papers) == 1
        assert len(out.gaps) == 1
        assert len(out.hypotheses) == 1
        assert out.selected_hypothesis == "H1"

    def test_serialization_roundtrip(self) -> None:
        out = IdeationOutput(
            topic="Test",
            search_queries=["q1", "q2"],
            must_cites=["Paper A"],
        )
        data = out.model_dump(mode="json")
        restored = IdeationOutput.model_validate(data)
        assert restored.topic == out.topic
        assert restored.search_queries == out.search_queries
        assert restored.must_cites == out.must_cites
