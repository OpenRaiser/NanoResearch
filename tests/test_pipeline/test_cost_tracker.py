"""Tests for nanoresearch.pipeline.cost_tracker."""

from __future__ import annotations

import pytest

from nanoresearch.pipeline.cost_tracker import (
    CostTracker,
    LLMResult,
    StageCost,
)


class TestLLMResult:
    """Tests for LLMResult dataclass."""

    def test_minimal(self) -> None:
        r = LLMResult(content="Hello")
        assert r.content == "Hello"
        assert r.usage == {}
        assert r.prompt_tokens == 0
        assert r.completion_tokens == 0
        assert r.total_tokens == 0

    def test_with_usage(self) -> None:
        r = LLMResult(
            content="Hi",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )
        assert r.prompt_tokens == 10
        assert r.completion_tokens == 5
        assert r.total_tokens == 15

    def test_total_tokens_fallback(self) -> None:
        r = LLMResult(
            content="x",
            usage={"prompt_tokens": 3, "completion_tokens": 7},
        )
        assert r.total_tokens == 10


class TestStageCost:
    """Tests for StageCost dataclass."""

    def test_record_increments(self) -> None:
        sc = StageCost()
        r1 = LLMResult("a", usage={"prompt_tokens": 10, "completion_tokens": 5})
        r2 = LLMResult("b", usage={"prompt_tokens": 20, "completion_tokens": 10})
        sc.record(r1)
        sc.record(r2)
        assert sc.total_tokens == 45
        assert sc.prompt_tokens == 30
        assert sc.completion_tokens == 15
        assert sc.num_calls == 2

    def test_to_dict(self) -> None:
        sc = StageCost()
        sc.record(LLMResult("x", usage={"prompt_tokens": 1, "completion_tokens": 2}))
        d = sc.to_dict()
        assert d["total_tokens"] == 3
        assert d["prompt_tokens"] == 1
        assert d["completion_tokens"] == 2
        assert d["num_calls"] == 1
        assert "total_latency_ms" in d


class TestCostTracker:
    """Tests for CostTracker."""

    def test_set_stage_creates_entry(self) -> None:
        ct = CostTracker()
        ct.set_stage("ideation")
        ct.record(LLMResult("x", usage={"prompt_tokens": 5, "completion_tokens": 3}))
        summary = ct.summary()
        assert "ideation" in summary
        assert summary["ideation"]["total_tokens"] == 8
        assert summary["ideation"]["num_calls"] == 1

    def test_record_without_stage_ignored(self) -> None:
        ct = CostTracker()
        ct.record(LLMResult("x", usage={"prompt_tokens": 100}))
        assert ct.summary() == {}

    def test_multiple_stages(self) -> None:
        ct = CostTracker()
        ct.set_stage("ideation")
        ct.record(LLMResult("a", usage={"prompt_tokens": 10, "completion_tokens": 5}))
        ct.set_stage("planning")
        ct.record(LLMResult("b", usage={"prompt_tokens": 20, "completion_tokens": 10}))
        summary = ct.summary()
        assert "ideation" in summary
        assert "planning" in summary
        assert summary["ideation"]["total_tokens"] == 15
        assert summary["planning"]["total_tokens"] == 30
