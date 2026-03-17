"""Tests for anti-fabrication features: real experiment results flow + no data invention.

Covers:
- ExperimentAgent._parse_metrics_json (valid / missing / malformed)
- FigureAgent._build_evidence_block (with/without real results)
- FigureAgent._build_chart_prompt (DATA RULES contain no fabrication instructions)
- WritingAgent._build_real_results_context (with/without real results)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nanoresearch.agents.experiment import ExperimentAgent
from nanoresearch.agents.figure_gen import FigureAgent
from nanoresearch.agents.writing import WritingAgent


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------

SAMPLE_METRICS = {
    "main_results": [
        {
            "method_name": "Ours",
            "dataset": "CIFAR-10",
            "is_proposed": True,
            "metrics": [
                {"metric_name": "Accuracy", "value": 85.4, "std": 0.3, "num_runs": 3}
            ],
        },
        {
            "method_name": "ResNet-50",
            "dataset": "CIFAR-10",
            "is_proposed": False,
            "metrics": [
                {"metric_name": "Accuracy", "value": 82.1, "std": 0.4, "num_runs": 3}
            ],
        },
    ],
    "ablation_results": [
        {
            "variant_name": "Full Model",
            "metrics": [{"metric_name": "Accuracy", "value": 85.4}],
        },
        {
            "variant_name": "w/o Attention",
            "metrics": [{"metric_name": "Accuracy", "value": 83.1}],
        },
    ],
    "training_log": [
        {"epoch": 1, "train_loss": 2.5, "val_loss": 2.3, "metrics": {"Accuracy": 78.2}},
        {"epoch": 2, "train_loss": 1.8, "val_loss": 1.7, "metrics": {"Accuracy": 84.5}},
    ],
}

SAMPLE_IDEATION = {
    "evidence": {
        "extracted_metrics": [
            {
                "method_name": "AlphaFold2",
                "dataset": "CASP14",
                "metric_name": "GDT-TS",
                "value": 92.4,
                "unit": "",
            }
        ]
    }
}

SAMPLE_BLUEPRINT = {
    "baselines": [
        {
            "name": "AlphaFold2",
            "expected_performance": {"GDT-TS": 92.4},
            "performance_provenance": {"GDT-TS": "literature"},
        }
    ]
}


# ===========================================================================
# ExperimentAgent._parse_metrics_json
# ===========================================================================


class TestParseMetricsJson:
    def test_valid_metrics(self, tmp_path):
        """Valid metrics.json is parsed correctly."""
        code_dir = tmp_path / "code"
        results_dir = code_dir / "results"
        results_dir.mkdir(parents=True)
        (results_dir / "metrics.json").write_text(
            json.dumps(SAMPLE_METRICS), encoding="utf-8"
        )

        result = ExperimentAgent._parse_metrics_json(code_dir)
        assert result["main_results"][0]["method_name"] == "Ours"
        assert result["ablation_results"][1]["variant_name"] == "w/o Attention"
        assert len(result["training_log"]) == 2

    def test_missing_file(self, tmp_path):
        """Missing metrics.json returns empty dict."""
        code_dir = tmp_path / "code"
        code_dir.mkdir(parents=True)

        result = ExperimentAgent._parse_metrics_json(code_dir)
        assert result == {}


class TestParseLlmJsonPayload:
    def test_repairs_invalid_backslash_escapes(self):
        raw = '```json\n{"old": "\\cite{demo}", "new": "fixed"}\n```'
        result = ExperimentAgent._parse_llm_json_payload(raw)
        assert result["old"] == "\\cite{demo}"
        assert result["new"] == "fixed"

    def test_repairs_truncated_json_array(self):
        raw = '[{"old": "a", "new": "b"}'
        result = ExperimentAgent._parse_llm_json_payload(raw)
        assert isinstance(result, list)
        assert result[0]["old"] == "a"
        assert result[0]["new"] == "b"

    def test_ignores_trailing_text_after_valid_json(self):
        raw = '[{"old": "a", "new": "b"}]\nDone. Applied safely.'
        result = ExperimentAgent._parse_llm_json_payload(raw)
        assert isinstance(result, list)
        assert result[0]["old"] == "a"

    def test_skips_leading_explanation_before_json(self):
        raw = 'Here is the patch:\n[{"old": "a", "new": "b"}]'
        result = ExperimentAgent._parse_llm_json_payload(raw)
        assert isinstance(result, list)
        assert result[0]["new"] == "b"

    def test_malformed_json(self, tmp_path):
        """Malformed JSON returns empty dict."""
        code_dir = tmp_path / "code"
        results_dir = code_dir / "results"
        results_dir.mkdir(parents=True)
        (results_dir / "metrics.json").write_text("not json{{{", encoding="utf-8")

        result = ExperimentAgent._parse_metrics_json(code_dir)
        assert result == {}

    def test_top_level_training_log_list_is_wrapped(self, tmp_path):
        """List-only metrics.json should be treated as training_log when entries are dicts."""
        code_dir = tmp_path / "code"
        results_dir = code_dir / "results"
        results_dir.mkdir(parents=True)
        (results_dir / "metrics.json").write_text(
            json.dumps(
                [
                    {"epoch": 1, "train_loss": 0.8, "metrics": {"Accuracy": 0.7}},
                    {"epoch": 2, "train_loss": 0.6, "metrics": {"Accuracy": 0.8}},
                ]
            ),
            encoding="utf-8",
        )

        result = ExperimentAgent._parse_metrics_json(code_dir)
        assert result["training_log"][0]["epoch"] == 1
        assert result["main_results"][0]["metrics"][0]["metric_name"] == "Accuracy"

    def test_wrong_structure(self, tmp_path):
        """JSON without expected keys returns empty dict."""
        code_dir = tmp_path / "code"
        results_dir = code_dir / "results"
        results_dir.mkdir(parents=True)
        (results_dir / "metrics.json").write_text(
            json.dumps({"unrelated_key": 42}), encoding="utf-8"
        )

        result = ExperimentAgent._parse_metrics_json(code_dir)
        assert result == {}

    def test_non_dict_json(self, tmp_path):
        """JSON that is a list (not dict) returns empty dict."""
        code_dir = tmp_path / "code"
        results_dir = code_dir / "results"
        results_dir.mkdir(parents=True)
        (results_dir / "metrics.json").write_text("[1, 2, 3]", encoding="utf-8")

        result = ExperimentAgent._parse_metrics_json(code_dir)
        assert result == {}

    def test_nan_values_filtered(self, tmp_path):
        """Entries with NaN metric values are filtered out."""
        code_dir = tmp_path / "code"
        results_dir = code_dir / "results"
        results_dir.mkdir(parents=True)

        data = {
            "main_results": [
                {
                    "method_name": "Good",
                    "metrics": [{"metric_name": "Acc", "value": 85.0}],
                },
                {
                    "method_name": "Bad",
                    "metrics": [{"metric_name": "Acc", "value": float("nan")}],
                },
            ],
        }
        # json.dumps converts NaN to "NaN" which is technically invalid JSON
        # but Python's json.loads accepts it by default
        raw = json.dumps(data, allow_nan=True)
        (results_dir / "metrics.json").write_text(raw, encoding="utf-8")

        result = ExperimentAgent._parse_metrics_json(code_dir)
        assert len(result["main_results"]) == 1
        assert result["main_results"][0]["method_name"] == "Good"

    def test_inf_values_filtered(self, tmp_path):
        """Entries with Inf metric values are filtered out."""
        code_dir = tmp_path / "code"
        results_dir = code_dir / "results"
        results_dir.mkdir(parents=True)

        data = {
            "main_results": [
                {
                    "method_name": "Good",
                    "metrics": [{"metric_name": "Acc", "value": 85.0}],
                },
            ],
            "training_log": [
                {"epoch": 1, "train_loss": float("inf"), "metrics": {}},
                {"epoch": 2, "train_loss": 1.5, "metrics": {"Acc": 80.0}},
            ],
        }
        raw = json.dumps(data, allow_nan=True)
        (results_dir / "metrics.json").write_text(raw, encoding="utf-8")

        result = ExperimentAgent._parse_metrics_json(code_dir)
        assert len(result["main_results"]) == 1
        assert len(result["training_log"]) == 1
        assert result["training_log"][0]["epoch"] == 2

    def test_main_results_not_list(self, tmp_path):
        """main_results that is not a list gets dropped."""
        code_dir = tmp_path / "code"
        results_dir = code_dir / "results"
        results_dir.mkdir(parents=True)

        data = {
            "main_results": "not a list",
            "training_log": [{"epoch": 1, "train_loss": 1.0}],
        }
        (results_dir / "metrics.json").write_text(
            json.dumps(data), encoding="utf-8"
        )

        result = ExperimentAgent._parse_metrics_json(code_dir)
        assert "main_results" not in result
        assert "training_log" in result

    def test_all_entries_sanitized_returns_empty(self, tmp_path):
        """If all entries are filtered out by sanitization, return empty dict."""
        code_dir = tmp_path / "code"
        results_dir = code_dir / "results"
        results_dir.mkdir(parents=True)

        data = {
            "main_results": [
                {
                    "method_name": "Bad1",
                    "metrics": [{"metric_name": "Acc", "value": float("nan")}],
                },
                {
                    "method_name": "Bad2",
                    "metrics": [{"metric_name": "Acc", "value": float("inf")}],
                },
            ],
        }
        raw = json.dumps(data, allow_nan=True)
        (results_dir / "metrics.json").write_text(raw, encoding="utf-8")

        result = ExperimentAgent._parse_metrics_json(code_dir)
        assert result == {}

    def test_training_log_metrics_not_dict(self, tmp_path):
        """training_log entry with metrics as list (not dict) is dropped."""
        code_dir = tmp_path / "code"
        results_dir = code_dir / "results"
        results_dir.mkdir(parents=True)

        data = {
            "main_results": [
                {
                    "method_name": "Good",
                    "metrics": [{"metric_name": "Acc", "value": 85.0}],
                },
            ],
            "training_log": [
                {"epoch": 1, "train_loss": 1.5, "metrics": [{"value": 80.0}]},
                {"epoch": 2, "train_loss": 1.0, "metrics": {"Acc": 90.0}},
            ],
        }
        (results_dir / "metrics.json").write_text(
            json.dumps(data), encoding="utf-8"
        )

        result = ExperimentAgent._parse_metrics_json(code_dir)
        assert len(result["main_results"]) == 1
        # Only epoch 2 survives — epoch 1 has metrics as a list
        assert len(result["training_log"]) == 1
        assert result["training_log"][0]["epoch"] == 2

    def test_flat_metrics_dict_is_normalized_to_contract(self, tmp_path):
        """Flat summary metrics are wrapped into the standard main_results schema."""
        code_dir = tmp_path / "code"
        results_dir = code_dir / "results"
        results_dir.mkdir(parents=True)
        (results_dir / "metrics.json").write_text(
            json.dumps({"dataset": "MNIST", "accuracy": 0.91, "f1": 0.88}),
            encoding="utf-8",
        )

        result = ExperimentAgent._parse_metrics_json(code_dir)
        assert result["main_results"][0]["dataset"] == "MNIST"
        metric_names = {metric["metric_name"] for metric in result["main_results"][0]["metrics"]}
        assert {"accuracy", "f1"} <= metric_names

    def test_nested_results_dict_is_normalized_to_contract(self, tmp_path):
        """Nested results/summary dicts are normalized into main_results."""
        code_dir = tmp_path / "code"
        results_dir = code_dir / "results"
        results_dir.mkdir(parents=True)
        (results_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "dataset": "CIFAR-10",
                    "results": {
                        "Accuracy": {"mean": 85.4, "std": 0.3},
                        "F1": 84.1,
                    },
                    "training_log": [],
                }
            ),
            encoding="utf-8",
        )

        result = ExperimentAgent._parse_metrics_json(code_dir)
        assert result["main_results"][0]["dataset"] == "CIFAR-10"
        metrics = {metric["metric_name"]: metric["value"] for metric in result["main_results"][0]["metrics"]}
        assert metrics["Accuracy"] == 85.4
        assert metrics["F1"] == 84.1

    def test_training_log_only_derives_main_results(self, tmp_path):
        """A valid training_log-only payload derives a summary main_results entry."""
        code_dir = tmp_path / "code"
        results_dir = code_dir / "results"
        results_dir.mkdir(parents=True)
        (results_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "dataset": "DemoSet",
                    "training_log": [
                        {"epoch": 1, "metrics": {"Accuracy": 0.75}},
                        {"epoch": 2, "metrics": {"Accuracy": 0.84, "F1": 0.82}},
                    ]
                }
            ),
            encoding="utf-8",
        )

        result = ExperimentAgent._parse_metrics_json(code_dir)
        assert result["main_results"][0]["dataset"] == "DemoSet"
        metrics = {metric["metric_name"]: metric["value"] for metric in result["main_results"][0]["metrics"]}
        assert metrics["Accuracy"] == 0.84
        assert metrics["F1"] == 0.82


# ===========================================================================
# FigureAgent._build_evidence_block
# ===========================================================================


class TestEvidenceBlock:
    def test_with_real_results(self):
        """Evidence block with real experiment results marks them as REAL EXPERIMENT."""
        block = FigureAgent._build_evidence_block(
            SAMPLE_IDEATION, SAMPLE_BLUEPRINT, SAMPLE_METRICS, "success"
        )
        assert "REAL EXPERIMENT" in block
        assert "85.4" in block
        assert "82.1" in block
        assert "w/o Attention" in block
        assert "MUST USE THESE EXACT NUMBERS" in block
        # Should NOT contain fabrication instructions
        assert "estimate" not in block.lower() or "estimate" in "MUST USE THESE EXACT NUMBERS. DO NOT MODIFY THEM.".lower()

    def test_without_results(self):
        """Evidence block without real results but WITH literature allows charts."""
        block = FigureAgent._build_evidence_block(
            SAMPLE_IDEATION, SAMPLE_BLUEPRINT, {}, "pending"
        )
        # Has literature → allows charts with published data
        assert "NO EXPERIMENT DATA FOR PROPOSED METHOD" in block
        assert "You MAY generate comparison charts" in block
        # Should NOT contain "Results Pending" placeholder
        assert "Results Pending" not in block
        # Literature data should still be present
        assert "AlphaFold2" in block
        assert "92.4" in block

    def test_failed_status(self):
        """Failed experiment status with literature allows charts."""
        block = FigureAgent._build_evidence_block(
            SAMPLE_IDEATION, SAMPLE_BLUEPRINT, SAMPLE_METRICS, "failed"
        )
        # With failed status, even if metrics data exists, no real results used
        # But literature exists → allows charts
        assert "NO EXPERIMENT DATA FOR PROPOSED METHOD" in block
        assert "Results Pending" not in block

    def test_no_literature_no_results(self):
        """No data at all still produces a valid block with no-data notice."""
        block = FigureAgent._build_evidence_block(
            {}, {"baselines": []}, {}, "pending"
        )
        assert "NO EXPERIMENT DATA AVAILABLE" in block
        assert "No published quantitative evidence" in block


# ===========================================================================
# FigureAgent._build_chart_prompt — DATA RULES no-fabrication check
# ===========================================================================


class TestChartPromptNoFabrication:
    """Verify DATA RULES do not contain any fabrication instructions."""

    FABRICATION_PHRASES = [
        "estimate 1-3%",
        "estimate 1-3% above",
        "generate 10-20 data points",
        "add small noise",
        "add noise for realism",
        "typically 0.2-0.5% std",
        "showing realistic curves",
        "plausible placeholder",
    ]

    def _get_chart_prompt(self) -> str:
        agent_cls = FigureAgent
        return agent_cls._build_chart_prompt(
            None,  # self (static-like usage, we pass explicitly)
            chart_type="grouped_bar",
            title="Test",
            description="Test description",
            method_name="TestMethod",
            baselines="Baseline1, Baseline2",
            metrics="Accuracy",
            ablation_groups="Component Ablation",
            primary_metric="Accuracy",
            evidence_block="=== NO DATA ===",
            output_path="/tmp/test.png",
            context="Test context",
        )

    def test_no_fabrication_phrases(self):
        """DATA RULES must not contain any fabrication instructions."""
        # Since _build_chart_prompt is an instance method, we need to call it
        # We'll just check the method constructs the prompt correctly
        # by instantiating a minimal call
        prompt = FigureAgent._build_chart_prompt(
            FigureAgent.__new__(FigureAgent),
            chart_type="grouped_bar",
            title="Test",
            description="Test description",
            method_name="TestMethod",
            baselines="Baseline1, Baseline2",
            metrics="Accuracy",
            ablation_groups="Component Ablation",
            primary_metric="Accuracy",
            evidence_block="=== NO DATA ===",
            output_path="/tmp/test.png",
            context="Test context",
        )
        prompt_lower = prompt.lower()
        for phrase in self.FABRICATION_PHRASES:
            assert phrase.lower() not in prompt_lower, (
                f"Fabrication instruction found in DATA RULES: '{phrase}'"
            )

    def test_contains_anti_fabrication_rules(self):
        """DATA RULES must contain anti-fabrication instructions."""
        prompt = FigureAgent._build_chart_prompt(
            FigureAgent.__new__(FigureAgent),
            chart_type="grouped_bar",
            title="Test",
            description="Test description",
            method_name="TestMethod",
            baselines="Baseline1, Baseline2",
            metrics="Accuracy",
            ablation_groups="Component Ablation",
            primary_metric="Accuracy",
            evidence_block="=== NO DATA ===",
            output_path="/tmp/test.png",
            context="Test context",
        )
        assert "NO FABRICATION" in prompt or "Do NOT invent" in prompt
        assert "Results Pending" in prompt
        assert "Do NOT" in prompt


# ===========================================================================
# WritingAgent._build_real_results_context
# ===========================================================================


class TestWritingRealResultsContext:
    def test_with_real_results(self):
        """Real results context includes exact numbers with MUST USE instruction."""
        ctx = WritingAgent._build_real_results_context(SAMPLE_METRICS, "success")
        assert "REAL EXPERIMENT RESULTS" in ctx
        assert "MUST USE THESE EXACT NUMBERS" in ctx
        assert "85.4" in ctx
        assert "82.1" in ctx
        assert "w/o Attention" in ctx
        assert "PROPOSED" in ctx

    def test_pending_results(self):
        """Pending results context instructs NOT to fabricate results."""
        ctx = WritingAgent._build_real_results_context({}, "pending")
        assert "NOT AVAILABLE" in ctx
        # Should instruct NOT to fabricate
        assert "NOT fabricate" in ctx or "Do NOT" in ctx

    def test_failed_results(self):
        """Failed experiment results also show not-available context."""
        ctx = WritingAgent._build_real_results_context(SAMPLE_METRICS, "failed")
        assert "NOT AVAILABLE" in ctx

    def test_empty_main_results(self):
        """Empty main_results with success status shows not-available context."""
        ctx = WritingAgent._build_real_results_context(
            {"main_results": []}, "success"
        )
        assert "NOT AVAILABLE" in ctx
