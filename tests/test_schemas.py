"""Tests for Pydantic schemas."""

from nanoresearch.schemas.ideation import (
    GapAnalysis,
    Hypothesis,
    IdeationOutput,
    PaperReference,
)
from nanoresearch.schemas.experiment import (
    AblationGroup,
    Baseline,
    Dataset,
    ExperimentBlueprint,
    Metric,
)
from nanoresearch.schemas.paper import FigurePlaceholder, PaperSkeleton, Section
from nanoresearch.schemas.manifest import (
    ArtifactRecord,
    PipelineStage,
    StageRecord,
    WorkspaceManifest,
)
from nanoresearch.schemas.evidence import EvidenceBundle, ExtractedMetric


class TestPaperReference:
    def test_minimal(self):
        ref = PaperReference(paper_id="2401.00001", title="Test Paper")
        assert ref.paper_id == "2401.00001"
        assert ref.authors == []
        assert ref.relevance_score == 0.0

    def test_full(self):
        ref = PaperReference(
            paper_id="2401.00001",
            title="Test Paper",
            authors=["Alice", "Bob"],
            year=2024,
            abstract="Abstract text",
            venue="NeurIPS",
            citation_count=10,
            url="https://arxiv.org/abs/2401.00001",
            relevance_score=0.85,
        )
        assert ref.year == 2024
        assert len(ref.authors) == 2

    def test_relevance_bounds(self):
        import pytest
        with pytest.raises(Exception):
            PaperReference(paper_id="x", title="x", relevance_score=1.5)


class TestGapAnalysis:
    def test_create(self):
        gap = GapAnalysis(
            gap_id="GAP-001",
            description="Missing equivariant methods",
            supporting_refs=["2401.00001"],
            severity="high",
        )
        assert gap.gap_id == "GAP-001"
        assert gap.severity == "high"


class TestHypothesis:
    def test_create(self):
        hyp = Hypothesis(
            hypothesis_id="HYP-001",
            statement="Equivariant GNNs improve folding",
            gap_refs=["GAP-001"],
        )
        assert hyp.hypothesis_id == "HYP-001"


class TestIdeationOutput:
    def test_minimal(self):
        out = IdeationOutput(topic="protein folding")
        assert out.topic == "protein folding"
        assert out.papers == []

    def test_full(self, sample_ideation_output):
        out = IdeationOutput.model_validate(sample_ideation_output)
        assert out.topic == "Graph neural networks for protein folding"
        assert len(out.papers) == 5
        assert len(out.gaps) == 1
        assert out.selected_hypothesis == "HYP-001"


class TestExperimentBlueprint:
    def test_minimal(self):
        bp = ExperimentBlueprint(title="Test", hypothesis_ref="HYP-001")
        assert bp.title == "Test"
        assert bp.datasets == []

    def test_full(self, sample_blueprint):
        bp = ExperimentBlueprint.model_validate(sample_blueprint)
        assert len(bp.datasets) == 1
        assert bp.datasets[0].name == "CASP14"
        assert len(bp.metrics) == 2
        assert bp.metrics[0].primary is True


class TestPaperSkeleton:
    def test_minimal(self):
        skeleton = PaperSkeleton(title="Test Paper")
        assert skeleton.title == "Test Paper"
        assert skeleton.template_format == "arxiv"

    def test_with_sections(self):
        skeleton = PaperSkeleton(
            title="Test",
            authors=["Anonymous"],
            abstract="Abstract.",
            sections=[
                Section(heading="Introduction", label="sec:intro", content="Intro text."),
                Section(
                    heading="Methods",
                    label="sec:methods",
                    content="Methods text.",
                    subsections=[
                        Section(heading="Submethod", content="Detail."),
                    ],
                ),
            ],
            figures=[
                FigurePlaceholder(
                    figure_id="fig:overview",
                    caption="System overview",
                    figure_type="placeholder",
                ),
            ],
        )
        assert len(skeleton.sections) == 2
        assert len(skeleton.sections[1].subsections) == 1
        assert len(skeleton.figures) == 1


class TestExtractedMetric:
    def test_create(self):
        m = ExtractedMetric(
            paper_id="2401.00001",
            paper_title="Test Paper",
            dataset="QM9",
            metric_name="MAE",
            value=0.012,
            unit="eV",
            context="achieves 0.012 eV MAE on QM9",
            method_name="SchNet",
            higher_is_better=False,
        )
        assert m.paper_id == "2401.00001"
        assert m.value == 0.012
        assert m.higher_is_better is False

    def test_string_value(self):
        m = ExtractedMetric(
            paper_id="x", dataset="X", metric_name="M", value="0.012-0.015"
        )
        assert m.value == "0.012-0.015"

    def test_minimal(self):
        m = ExtractedMetric(paper_id="x", dataset="X", metric_name="M", value=1.0)
        assert m.unit == ""
        assert m.higher_is_better is None


class TestEvidenceBundle:
    def test_empty(self):
        b = EvidenceBundle()
        assert b.extracted_metrics == []
        assert b.extraction_notes == ""
        assert b.coverage_warnings == []

    def test_with_metrics(self):
        b = EvidenceBundle(
            extracted_metrics=[
                ExtractedMetric(
                    paper_id="1", dataset="D", metric_name="M", value=0.5
                )
            ],
            extraction_notes="Found 1 metric",
            coverage_warnings=["Missing data for baseline X"],
        )
        assert len(b.extracted_metrics) == 1
        assert len(b.coverage_warnings) == 1

    def test_roundtrip_json(self):
        b = EvidenceBundle(
            extracted_metrics=[
                ExtractedMetric(
                    paper_id="1", dataset="D", metric_name="M", value=0.5
                )
            ],
        )
        data = b.model_dump(mode="json")
        b2 = EvidenceBundle.model_validate(data)
        assert len(b2.extracted_metrics) == 1
        assert b2.extracted_metrics[0].value == 0.5


class TestIdeationOutputEvidence:
    def test_backward_compat_no_evidence(self):
        """IdeationOutput without evidence field should work (default_factory)."""
        out = IdeationOutput(topic="test topic")
        assert out.evidence.extracted_metrics == []

    def test_with_evidence(self, sample_ideation_output):
        out = IdeationOutput.model_validate(sample_ideation_output)
        assert len(out.evidence.extracted_metrics) == 1
        assert out.evidence.extracted_metrics[0].dataset == "CASP14"


class TestBaselineProvenance:
    def test_backward_compat_no_provenance(self):
        """Baseline without provenance fields should work (default_factory)."""
        b = Baseline(name="TestBaseline")
        assert b.performance_provenance == {}
        assert b.is_projected == {}

    def test_with_provenance(self):
        b = Baseline(
            name="AlphaFold2",
            expected_performance={"GDT-TS": 92.4},
            performance_provenance={"GDT-TS": "Abstract of arxiv:2401.00001"},
            is_projected={"GDT-TS": False},
        )
        assert b.performance_provenance["GDT-TS"] == "Abstract of arxiv:2401.00001"
        assert b.is_projected["GDT-TS"] is False


class TestExperimentBlueprintEvidence:
    def test_backward_compat(self):
        bp = ExperimentBlueprint(title="Test", hypothesis_ref="HYP-001")
        assert bp.evidence_summary == ""
        assert bp.data_provenance_note == ""

    def test_with_evidence_fields(self):
        bp = ExperimentBlueprint(
            title="Test",
            hypothesis_ref="HYP-001",
            evidence_summary="Used published GDT-TS scores",
            data_provenance_note="Baselines from papers; proposed method projected",
        )
        assert "published" in bp.evidence_summary.lower()


class TestReviewOutput:
    def test_minimal(self):
        from nanoresearch.schemas.review import ReviewOutput
        out = ReviewOutput()
        assert out.overall_score == 0.0
        assert out.section_reviews == []
        assert out.consistency_issues == []
        assert out.revision_rounds == 0

    def test_full(self):
        from nanoresearch.schemas.review import ReviewOutput, SectionReview, ConsistencyIssue
        out = ReviewOutput(
            overall_score=7.5,
            section_reviews=[
                SectionReview(section="Introduction", score=8, issues=["Too short"], suggestions=["Add more context"]),
                SectionReview(section="Methods", score=7, issues=[], suggestions=[]),
            ],
            consistency_issues=[
                ConsistencyIssue(
                    issue_type="ref_mismatch",
                    description="\\ref{fig:missing} has no corresponding \\label",
                    locations=["Section 3"],
                    severity="high",
                )
            ],
            major_revisions=["Add more related work"],
            minor_revisions=["Fix typos"],
            revised_sections={"Introduction": "Revised intro content"},
            revision_rounds=1,
        )
        assert len(out.section_reviews) == 2
        assert out.section_reviews[0].score == 8
        assert len(out.consistency_issues) == 1
        assert out.consistency_issues[0].severity == "high"
        assert out.revision_rounds == 1

    def test_roundtrip_json(self):
        from nanoresearch.schemas.review import ReviewOutput, SectionReview
        out = ReviewOutput(
            overall_score=8.0,
            section_reviews=[SectionReview(section="Intro", score=8)],
        )
        data = out.model_dump(mode="json")
        out2 = ReviewOutput.model_validate(data)
        assert out2.overall_score == 8.0
        assert len(out2.section_reviews) == 1

    def test_section_review_score_bounds(self):
        import pytest
        from nanoresearch.schemas.review import SectionReview
        with pytest.raises(Exception):
            SectionReview(section="X", score=0)  # min is 1
        with pytest.raises(Exception):
            SectionReview(section="X", score=11)  # max is 10


class TestWorkspaceManifest:
    def test_create(self):
        m = WorkspaceManifest(session_id="abc123", topic="test")
        assert m.current_stage == PipelineStage.INIT
        assert m.artifacts == []

    def test_stage_enum(self):
        assert PipelineStage.INIT.value == "INIT"
        assert PipelineStage.DONE.value == "DONE"

    def test_stage_record(self):
        rec = StageRecord(stage=PipelineStage.IDEATION)
        assert rec.status == "pending"
        assert rec.retries == 0

    def test_roundtrip_json(self):
        m = WorkspaceManifest(session_id="abc", topic="test")
        data = m.model_dump(mode="json")
        m2 = WorkspaceManifest.model_validate(data)
        assert m2.session_id == "abc"
