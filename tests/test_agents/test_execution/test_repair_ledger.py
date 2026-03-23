"""Tests for nanoresearch.agents.execution.repair_ledger."""

from __future__ import annotations

import pytest

from nanoresearch.agents.execution.repair_ledger import _RepairLedgerMixin


class TestRepairLedgerMixin:
    """Tests for _RepairLedgerMixin static methods."""

    def test_repair_error_text_prefers_stderr(self) -> None:
        result = {"stderr": "Error in script", "stdout": "output", "returncode": 1}
        text = _RepairLedgerMixin._repair_error_text(result)
        assert text == "Error in script"

    def test_repair_error_text_fallback_to_stdout(self) -> None:
        result = {"stderr": "", "stdout": "Some output", "returncode": 1}
        text = _RepairLedgerMixin._repair_error_text(result)
        assert text == "Some output"

    def test_repair_error_text_fallback_to_returncode(self) -> None:
        result = {"stderr": "", "stdout": "", "returncode": 137}
        text = _RepairLedgerMixin._repair_error_text(result)
        assert "137" in text or "return code" in text.lower()

    def test_repair_error_signature_extracts_last_line(self) -> None:
        result = {
            "returncode": 1,
            "stderr": "Traceback...\n  File x.py\nValueError: something failed",
        }
        sig = _RepairLedgerMixin._repair_error_signature(result)
        assert "ValueError" in sig or "something" in sig
        assert "rc=" in sig or "1" in sig

    def test_repair_repeat_count_empty_history(self) -> None:
        count = _RepairLedgerMixin._repair_repeat_count([], "sig1")
        assert count == 1

    def test_repair_repeat_count_with_matching_entry(self) -> None:
        fix_history = [{"signature": "sig1", "repeat_count": 2}]
        count = _RepairLedgerMixin._repair_repeat_count(fix_history, "sig1")
        assert count == 3

    def test_record_repair_attempt_updates_existing_entry(self) -> None:
        fix_history = [{"signature": "sig1", "repeat_count": 1, "cycle": 0}]
        _RepairLedgerMixin._record_repair_attempt(
            fix_history,
            "sig1",
            "error msg",
            cycle=1,
            modified=["file1.py"],
        )
        assert fix_history[0]["repeat_count"] == 2
        assert fix_history[0]["cycle"] == 1
        assert "file1.py" in fix_history[0].get("fixed_files", [])
