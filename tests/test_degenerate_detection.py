"""Tests for degenerate-run detection in training dynamics and figure gen."""
from __future__ import annotations

import pytest

from nanoresearch.agents.analysis.training_dynamics import analyze_training_dynamics


class TestDegenerateRunDetection:
    """Training dynamics should flag all-zero metrics as degenerate."""

    @staticmethod
    def _make_log(n_epochs: int, loss: float = 0.0, acc: float = 0.0):
        return [
            {"epoch": i + 1, "train_loss": loss, "val_loss": loss,
             "train_acc": acc, "val_acc": acc, "lr": 1e-3 * (1 - i / n_epochs)}
            for i in range(n_epochs)
        ]

    def test_all_zero_flagged(self):
        log = self._make_log(30, loss=0.0, acc=0.0)
        result = analyze_training_dynamics(log)
        assert result.get("degenerate_run") is True
        assert "degenerate_reason" in result
        assert result["loss_stability"] == "degenerate"

    def test_normal_training_not_flagged(self):
        log = [
            {"epoch": i + 1, "train_loss": 2.0 - i * 0.05,
             "val_loss": 2.1 - i * 0.04}
            for i in range(30)
        ]
        result = analyze_training_dynamics(log)
        assert result.get("degenerate_run") is not True
        assert "degenerate_reason" not in result

    def test_partial_zero_not_flagged(self):
        """If some epochs have non-zero, it's not degenerate."""
        log = self._make_log(10, loss=0.0, acc=0.0)
        log[5]["val_loss"] = 0.5  # one non-zero entry
        result = analyze_training_dynamics(log)
        assert result.get("degenerate_run") is not True

    def test_too_few_epochs_no_crash(self):
        log = self._make_log(2, loss=0.0, acc=0.0)
        result = analyze_training_dynamics(log)
        # < 3 epochs → analysis skipped, no degenerate check
        assert "analysis_skipped" in result
