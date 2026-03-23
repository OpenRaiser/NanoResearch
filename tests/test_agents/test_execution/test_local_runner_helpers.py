"""Tests for nanoresearch.agents.execution.local_runner_helpers."""

from __future__ import annotations

import pytest

from nanoresearch.agents.execution.local_runner_helpers import _LocalRunnerHelpersMixin
from nanoresearch.schemas.iteration import ExperimentHypothesis, IterationState, RoundResult


def _round_result(round_number: int = 1) -> RoundResult:
    return RoundResult(
        round_number=round_number,
        hypothesis=ExperimentHypothesis(round_number=round_number, hypothesis="test"),
    )


class TestCommandWithMode:
    """Tests for _command_with_mode."""

    def test_adds_mode_if_absent(self) -> None:
        cmd = ["python", "train.py"]
        result = _LocalRunnerHelpersMixin._command_with_mode(cmd, "--dry-run")
        assert result == ["python", "train.py", "--dry-run"]

    def test_leaves_unchanged_if_mode_present(self) -> None:
        cmd = ["python", "train.py", "--dry-run"]
        result = _LocalRunnerHelpersMixin._command_with_mode(cmd, "--dry-run")
        assert result == cmd


class TestBuildExecutionBlueprintSummary:
    """Tests for _build_execution_blueprint_summary."""

    def test_returns_json_string(self) -> None:
        summary = _LocalRunnerHelpersMixin._build_execution_blueprint_summary(
            topic="ML",
            blueprint={"title": "Exp", "datasets": []},
            setup_output={},
            coding_output={"train_command": "python train.py"},
        )
        assert isinstance(summary, str)
        assert "ML" in summary
        assert "Exp" in summary
        assert "train" in summary


class TestUpdateBestRound:
    """Tests for _update_best_round."""

    def test_no_analysis_no_change(self) -> None:
        state = IterationState(max_rounds=3)
        state.rounds.append(_round_result(1))
        _LocalRunnerHelpersMixin._update_best_round(state, None)
        assert state.best_round is None

    def test_analysis_with_metric_updates_best(self) -> None:
        class MockAnalysis:
            metric_summary = {"MAE": 0.05}

        state = IterationState(max_rounds=3)
        state.rounds.append(_round_result(1))
        _LocalRunnerHelpersMixin._update_best_round(state, MockAnalysis())
        assert state.best_round == 1
        assert state.best_metrics == {"MAE": 0.05}
