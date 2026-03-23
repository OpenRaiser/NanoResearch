"""Tests for nanoresearch.schemas.experiment."""

from __future__ import annotations

import pytest

from nanoresearch.schemas.experiment import (
    AblationGroup,
    AblationResult,
    Baseline,
    ComputeRequirements,
    Dataset,
    ExperimentBlueprint,
    ExperimentResults,
    Metric,
    MetricResult,
    MethodResult,
    TrainingLogEntry,
)


class TestComputeRequirements:
    """Tests for ComputeRequirements model."""

    def test_defaults(self) -> None:
        cr = ComputeRequirements()
        assert cr.gpu_type == ""
        assert cr.num_gpus == 1
        assert cr.estimated_hours == 0.0

    def test_custom_values(self) -> None:
        cr = ComputeRequirements(gpu_type="A100", num_gpus=4, estimated_hours=24.0)
        assert cr.gpu_type == "A100"
        assert cr.num_gpus == 4
        assert cr.estimated_hours == 24.0


class TestDataset:
    """Tests for Dataset model."""

    def test_minimal(self) -> None:
        d = Dataset(name="QM9")
        assert d.name == "QM9"
        assert d.description == ""

    def test_full(self) -> None:
        d = Dataset(
            name="QM9",
            description="Molecular dataset",
            source_url="https://example.com",
        )
        assert d.source_url == "https://example.com"


class TestBaseline:
    """Tests for Baseline model."""

    def test_minimal(self) -> None:
        b = Baseline(name="GNN")
        assert b.name == "GNN"
        assert b.expected_performance == {}

    def test_with_performance(self) -> None:
        b = Baseline(
            name="GNN",
            expected_performance={"MAE": 0.05},
            is_projected={"MAE": False},
        )
        assert b.expected_performance["MAE"] == 0.05
        assert b.is_projected["MAE"] is False


class TestMetric:
    """Tests for Metric model."""

    def test_minimal(self) -> None:
        m = Metric(name="MAE")
        assert m.name == "MAE"
        assert m.higher_is_better is True
        assert m.primary is False

    def test_primary_metric(self) -> None:
        m = Metric(name="Accuracy", higher_is_better=True, primary=True)
        assert m.primary is True


class TestAblationGroup:
    """Tests for AblationGroup model."""

    def test_minimal(self) -> None:
        ag = AblationGroup(group_name="Architecture", description="Vary layers")
        assert ag.group_name == "Architecture"
        assert ag.variants == []

    def test_with_variants(self) -> None:
        ag = AblationGroup(
            group_name="Arch",
            variants=[{"name": "v1", "layers": 2}, {"name": "v2", "layers": 4}],
        )
        assert len(ag.variants) == 2


class TestExperimentBlueprint:
    """Tests for ExperimentBlueprint model."""

    def test_minimal(self) -> None:
        eb = ExperimentBlueprint(title="Exp 1", hypothesis_ref="HYP-001")
        assert eb.title == "Exp 1"
        assert eb.hypothesis_ref == "HYP-001"
        assert eb.datasets == []
        assert eb.metrics == []
        assert eb.compute_requirements is not None

    def test_with_datasets_and_metrics(self) -> None:
        ds = Dataset(name="QM9")
        m = Metric(name="MAE", primary=True)
        eb = ExperimentBlueprint(
            title="Exp",
            hypothesis_ref="H1",
            datasets=[ds],
            metrics=[m],
        )
        assert len(eb.datasets) == 1
        assert len(eb.metrics) == 1
        assert eb.metrics[0].primary is True

    def test_compute_requirements_coercion(self) -> None:
        eb = ExperimentBlueprint(
            title="E",
            hypothesis_ref="H",
            compute_requirements={"gpu_type": "V100", "num_gpus": 2},
        )
        assert eb.compute_requirements.gpu_type == "V100"
        assert eb.compute_requirements.num_gpus == 2


class TestMetricResult:
    """Tests for MetricResult model."""

    def test_minimal(self) -> None:
        mr = MetricResult(metric_name="MAE", value=0.05)
        assert mr.metric_name == "MAE"
        assert mr.value == 0.05
        assert mr.std is None
        assert mr.num_runs == 1

    def test_with_std(self) -> None:
        mr = MetricResult(metric_name="MAE", value=0.05, std=0.01, num_runs=3)
        assert mr.std == 0.01
        assert mr.num_runs == 3


class TestMethodResult:
    """Tests for MethodResult model."""

    def test_minimal(self) -> None:
        mr = MethodResult(method_name="Ours")
        assert mr.method_name == "Ours"
        assert mr.metrics == []
        assert mr.is_proposed is False


class TestAblationResult:
    """Tests for AblationResult model."""

    def test_minimal(self) -> None:
        ar = AblationResult(variant_name="w/o attention")
        assert ar.variant_name == "w/o attention"
        assert ar.metrics == []


class TestTrainingLogEntry:
    """Tests for TrainingLogEntry model."""

    def test_minimal(self) -> None:
        tle = TrainingLogEntry(epoch=1)
        assert tle.epoch == 1
        assert tle.train_loss is None
        assert tle.metrics == {}

    def test_with_metrics(self) -> None:
        tle = TrainingLogEntry(epoch=5, train_loss=0.1, val_loss=0.12)
        assert tle.train_loss == 0.1
        assert tle.val_loss == 0.12


class TestExperimentResults:
    """Tests for ExperimentResults model."""

    def test_defaults(self) -> None:
        er = ExperimentResults()
        assert er.main_results == []
        assert er.ablation_results == []
        assert er.training_log == []

    def test_with_results(self) -> None:
        mr = MethodResult(method_name="Ours", metrics=[])
        ar = AblationResult(variant_name="v1", metrics=[])
        er = ExperimentResults(main_results=[mr], ablation_results=[ar])
        assert len(er.main_results) == 1
        assert len(er.ablation_results) == 1
