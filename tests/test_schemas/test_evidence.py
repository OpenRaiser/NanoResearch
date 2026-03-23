"""Tests for nanoresearch.schemas.evidence."""

from __future__ import annotations

import pytest

from nanoresearch.schemas.evidence import EvidenceBundle, ExtractedMetric


class TestExtractedMetric:
    """Tests for ExtractedMetric model."""

    def test_minimal_valid(self) -> None:
        em = ExtractedMetric(
            paper_id="arXiv:2401.00001",
            dataset="QM9",
            metric_name="MAE",
            value=0.05,
        )
        assert em.paper_id == "arXiv:2401.00001"
        assert em.dataset == "QM9"
        assert em.metric_name == "MAE"
        assert em.value == 0.05
        assert em.paper_title == ""
        assert em.unit == ""

    def test_with_context(self) -> None:
        em = ExtractedMetric(
            paper_id="p1",
            dataset="QM9",
            metric_name="MAE",
            value=0.05,
            context="Achieved MAE of 0.05 on QM9",
            method_name="GNN",
        )
        assert "Achieved" in em.context
        assert em.method_name == "GNN"

    def test_value_can_be_string(self) -> None:
        em = ExtractedMetric(
            paper_id="p1",
            dataset="D",
            metric_name="M",
            value="0.04-0.06",  # range
        )
        assert em.value == "0.04-0.06"

    def test_coerce_list_to_str(self) -> None:
        em = ExtractedMetric(
            paper_id="p1",
            dataset="D",
            metric_name="M",
            value=1.0,
            paper_title=["Part", "A"],
        )
        assert em.paper_title == "Part; A"


class TestEvidenceBundle:
    """Tests for EvidenceBundle model."""

    def test_defaults(self) -> None:
        eb = EvidenceBundle()
        assert eb.extracted_metrics == []
        assert eb.extraction_notes == ""
        assert eb.coverage_warnings == []

    def test_with_metrics(self) -> None:
        em = ExtractedMetric(
            paper_id="p1",
            dataset="D",
            metric_name="M",
            value=1.0,
        )
        eb = EvidenceBundle(extracted_metrics=[em], extraction_notes="Done")
        assert len(eb.extracted_metrics) == 1
        assert eb.extracted_metrics[0].paper_id == "p1"
        assert eb.extraction_notes == "Done"

    def test_serialization_roundtrip(self) -> None:
        eb = EvidenceBundle(
            extracted_metrics=[
                ExtractedMetric(paper_id="p1", dataset="D", metric_name="M", value=0.1),
            ],
            coverage_warnings=["Missing dataset X"],
        )
        data = eb.model_dump(mode="json")
        restored = EvidenceBundle.model_validate(data)
        assert len(restored.extracted_metrics) == 1
        assert restored.coverage_warnings == ["Missing dataset X"]
