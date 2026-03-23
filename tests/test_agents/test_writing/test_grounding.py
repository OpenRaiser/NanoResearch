"""Tests for nanoresearch.agents.writing.grounding."""

from __future__ import annotations

import pytest

from nanoresearch.agents.writing.grounding import _GroundingMixin


class TestNormalizeExperimentResults:
    """Tests for _normalize_experiment_results."""

    def test_passthrough_when_main_results_present(self) -> None:
        results = {
            "main_results": [
                {"method_name": "Ours", "dataset": "QM9", "metrics": []},
            ],
        }
        out = _GroundingMixin._normalize_experiment_results(
            results,
            {"proposed_method": {"name": "Ours"}},
            {},
        )
        assert out["main_results"] == results["main_results"]
        assert len(out["main_results"]) == 1

    def test_normalizes_from_analysis_when_main_results_empty(self) -> None:
        results = {}
        blueprint = {
            "proposed_method": {"name": "Our Method"},
            "datasets": [{"name": "QM9"}],
        }
        analysis = {"final_metrics": {"MAE": 0.05, "RMSE": 0.08}}
        out = _GroundingMixin._normalize_experiment_results(
            results,
            blueprint,
            analysis,
        )
        assert "main_results" in out
        assert len(out["main_results"]) == 1
        assert out["main_results"][0]["method_name"] == "Our Method"
        assert out["main_results"][0]["dataset"] == "QM9"
        assert len(out["main_results"][0]["metrics"]) == 2
