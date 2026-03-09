"""Tests for cost tracking, progress streaming, structured logging, and blueprint validation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from nanoresearch.pipeline.cost_tracker import CostTracker, LLMResult, StageCost
from nanoresearch.pipeline.progress import ProgressEmitter
from nanoresearch.pipeline.blueprint_validator import validate_blueprint
from nanoresearch.logging_config import JSONFormatter, setup_logging


# ===== Task A: Cost Tracking =====

class TestLLMResult:
    def test_basic_properties(self):
        r = LLMResult(
            content="hello",
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            model="gpt-4",
            latency_ms=1234.5,
        )
        assert r.content == "hello"
        assert r.prompt_tokens == 100
        assert r.completion_tokens == 50
        assert r.total_tokens == 150
        assert r.model == "gpt-4"
        assert r.latency_ms == 1234.5

    def test_empty_usage(self):
        r = LLMResult(content="hi")
        assert r.prompt_tokens == 0
        assert r.completion_tokens == 0
        assert r.total_tokens == 0

    def test_total_tokens_fallback(self):
        """total_tokens should be sum of prompt+completion if not provided."""
        r = LLMResult(
            content="x",
            usage={"prompt_tokens": 100, "completion_tokens": 50},
        )
        assert r.total_tokens == 150


class TestStageCost:
    def test_record(self):
        sc = StageCost()
        r1 = LLMResult(
            content="a",
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            model="m1",
            latency_ms=500.0,
        )
        r2 = LLMResult(
            content="b",
            usage={"prompt_tokens": 200, "completion_tokens": 100, "total_tokens": 300},
            model="m1",
            latency_ms=700.0,
        )
        sc.record(r1)
        sc.record(r2)
        assert sc.total_tokens == 450
        assert sc.prompt_tokens == 300
        assert sc.completion_tokens == 150
        assert sc.num_calls == 2
        assert sc.total_latency_ms == 1200.0

    def test_to_dict(self):
        sc = StageCost(
            total_tokens=100, prompt_tokens=60, completion_tokens=40,
            num_calls=1, total_latency_ms=500.5,
        )
        d = sc.to_dict()
        assert d["total_tokens"] == 100
        assert d["num_calls"] == 1
        assert d["total_latency_ms"] == 500.5


class TestCostTracker:
    def test_full_workflow(self):
        tracker = CostTracker()
        tracker.set_stage("IDEATION")
        tracker.record(LLMResult(
            content="x",
            usage={"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500},
            latency_ms=2000.0,
        ))
        tracker.record(LLMResult(
            content="y",
            usage={"prompt_tokens": 800, "completion_tokens": 400, "total_tokens": 1200},
            latency_ms=1500.0,
        ))

        tracker.set_stage("PLANNING")
        tracker.record(LLMResult(
            content="z",
            usage={"prompt_tokens": 500, "completion_tokens": 200, "total_tokens": 700},
            latency_ms=1000.0,
        ))

        summary = tracker.summary()
        assert summary["total_tokens"] == 3400
        assert summary["total_calls"] == 3
        assert summary["total_latency_ms"] == 4500.0
        assert "IDEATION" in summary["stages"]
        assert "PLANNING" in summary["stages"]
        assert summary["stages"]["IDEATION"]["num_calls"] == 2
        assert summary["stages"]["PLANNING"]["num_calls"] == 1

    def test_record_without_stage(self):
        """Recording without setting a stage should be a no-op."""
        tracker = CostTracker()
        tracker.record(LLMResult(content="x", usage={"total_tokens": 100}))
        assert tracker.summary()["total_tokens"] == 0

    def test_empty_summary(self):
        tracker = CostTracker()
        s = tracker.summary()
        assert s["total_tokens"] == 0
        assert s["total_calls"] == 0
        assert s["stages"] == {}


# ===== Task B: Progress Streaming =====

class TestProgressEmitter:
    def test_stage_lifecycle(self, tmp_path):
        path = tmp_path / "progress.json"
        emitter = ProgressEmitter(path)

        emitter.stage_start("IDEATION", 6, 0, "Starting IDEATION")
        assert path.exists()
        data = json.loads(path.read_text())
        assert len(data["events"]) == 1
        assert data["events"][0]["type"] == "stage_start"
        assert data["events"][0]["stage"] == "IDEATION"
        assert data["events"][0]["progress_pct"] == 0

        emitter.stage_complete("IDEATION", 6, 0, "IDEATION done")
        data = json.loads(path.read_text())
        assert len(data["events"]) == 2
        assert data["events"][1]["type"] == "stage_complete"
        assert data["events"][1]["progress_pct"] == 17  # round(1/6*100)

    def test_substep(self, tmp_path):
        path = tmp_path / "progress.json"
        emitter = ProgressEmitter(path)
        emitter.substep("WRITING", "Generating Introduction section")
        data = json.loads(path.read_text())
        assert data["events"][0]["type"] == "substep"
        assert "Introduction" in data["events"][0]["message"]

    def test_error_event(self, tmp_path):
        path = tmp_path / "progress.json"
        emitter = ProgressEmitter(path)
        emitter.error("EXPERIMENT", "LLM call failed")
        data = json.loads(path.read_text())
        assert data["events"][0]["type"] == "error"

    def test_pipeline_complete(self, tmp_path):
        path = tmp_path / "progress.json"
        emitter = ProgressEmitter(path)
        emitter.pipeline_complete(True, "All done")
        data = json.loads(path.read_text())
        assert data["events"][0]["type"] == "pipeline_complete"
        assert data["events"][0]["success"] is True
        assert data["events"][0]["progress_pct"] == 100

    def test_pipeline_failure(self, tmp_path):
        path = tmp_path / "progress.json"
        emitter = ProgressEmitter(path)
        emitter.pipeline_complete(False, "Crashed")
        data = json.loads(path.read_text())
        assert data["events"][0]["progress_pct"] == -1

    def test_rolling_window(self, tmp_path):
        """Only last MAX_EVENTS events should be kept."""
        path = tmp_path / "progress.json"
        emitter = ProgressEmitter(path)
        for i in range(60):
            emitter.substep("TEST", f"Step {i}")
        data = json.loads(path.read_text())
        assert len(data["events"]) == 50  # MAX_EVENTS

    def test_event_has_timestamp_and_elapsed(self, tmp_path):
        path = tmp_path / "progress.json"
        emitter = ProgressEmitter(path)
        emitter.substep("TEST", "hello")
        data = json.loads(path.read_text())
        evt = data["events"][0]
        assert "timestamp" in evt
        assert "elapsed_s" in evt
        assert isinstance(evt["elapsed_s"], (int, float))


# ===== Task C: Structured Logging =====

class TestStructuredLogging:
    def test_json_formatter(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="nanoresearch.test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert data["level"] == "INFO"
        assert data["message"] == "Test message"
        assert data["logger"] == "nanoresearch.test"
        assert "timestamp" in data

    def test_json_formatter_with_extra(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="nanoresearch.test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="LLM call",
            args=(),
            exc_info=None,
        )
        record.stage = "IDEATION"
        record.model = "gpt-4"
        record.tokens = 1500
        output = formatter.format(record)
        data = json.loads(output)
        assert data["stage"] == "IDEATION"
        assert data["model"] == "gpt-4"
        assert data["tokens"] == 1500

    def test_setup_logging_console_only(self):
        """setup_logging with no log_path should only add console handler."""
        setup_logging(log_path=None, level=logging.DEBUG)
        root = logging.getLogger("nanoresearch")
        assert len(root.handlers) == 1  # console only
        assert isinstance(root.handlers[0], logging.StreamHandler)
        # Clean up
        root.handlers.clear()

    def test_setup_logging_with_file(self, tmp_path):
        log_path = tmp_path / "logs" / "pipeline.jsonl"
        setup_logging(log_path=log_path, level=logging.INFO)
        root = logging.getLogger("nanoresearch")
        assert len(root.handlers) == 2  # console + file
        # Log something
        test_logger = logging.getLogger("nanoresearch.test_structured")
        test_logger.info("Hello structured")
        # Flush
        for h in root.handlers:
            h.flush()
        content = log_path.read_text()
        assert "Hello structured" in content
        data = json.loads(content.strip().split("\n")[-1])
        assert data["message"] == "Hello structured"
        # Clean up
        root.handlers.clear()

    def test_setup_logging_idempotent(self, tmp_path):
        """Calling setup_logging twice should not duplicate handlers."""
        log_path = tmp_path / "pipeline.jsonl"
        setup_logging(log_path=log_path)
        setup_logging(log_path=log_path)
        root = logging.getLogger("nanoresearch")
        assert len(root.handlers) == 2  # not 4
        root.handlers.clear()


# ===== Task D: Blueprint Semantic Validation =====

class TestBlueprintValidator:
    def test_valid_blueprint(self):
        bp = {
            "metrics": [
                {"name": "Accuracy", "higher_is_better": True, "primary": True},
                {"name": "F1", "higher_is_better": True, "primary": False},
            ],
            "proposed_method": {
                "name": "OurMethod",
                "description": "A novel approach using attention",
                "key_components": ["attention", "residual connection"],
            },
            "ablation_groups": [
                {
                    "group_name": "Component",
                    "variants": [
                        {"name": "w/o attention"},
                        {"name": "w/o residual"},
                    ],
                }
            ],
            "baselines": [
                {
                    "name": "BaselineA",
                    "expected_performance": {"Accuracy": 85.0},
                }
            ],
            "datasets": [{"name": "MNIST"}],
        }
        issues = validate_blueprint(bp)
        assert issues == []

    def test_empty_metrics(self):
        bp = {"metrics": [], "proposed_method": {"key_components": ["x"]}}
        issues = validate_blueprint(bp)
        assert any("no evaluation metrics" in i.lower() for i in issues)

    def test_no_primary_metric(self):
        bp = {
            "metrics": [
                {"name": "Accuracy", "higher_is_better": True, "primary": False},
            ],
            "proposed_method": {"key_components": ["x"]},
        }
        issues = validate_blueprint(bp)
        assert any("primary" in i.lower() for i in issues)

    def test_metric_direction_inconsistency(self):
        bp = {
            "metrics": [
                {"name": "MSE Loss", "higher_is_better": True, "primary": True},
            ],
            "proposed_method": {"key_components": ["x"]},
        }
        issues = validate_blueprint(bp)
        assert any("loss" in i.lower() and "higher_is_better" in i.lower() for i in issues)

    def test_empty_key_components(self):
        bp = {
            "metrics": [{"name": "Acc", "primary": True}],
            "proposed_method": {"key_components": []},
        }
        issues = validate_blueprint(bp)
        assert any("key_components" in i for i in issues)

    def test_baseline_unknown_metric(self):
        bp = {
            "metrics": [{"name": "Accuracy", "primary": True}],
            "proposed_method": {"key_components": ["x"]},
            "baselines": [
                {
                    "name": "BaseA",
                    "expected_performance": {"F1": 0.9},
                }
            ],
        }
        issues = validate_blueprint(bp)
        assert any("F1" in i and "not in the metrics list" in i for i in issues)

    def test_completely_empty_blueprint(self):
        issues = validate_blueprint({})
        assert len(issues) >= 1  # at least "no metrics"


# ===== Task A Integration: generate_with_usage =====

class TestGenerateWithUsage:
    @pytest.mark.asyncio
    async def test_generate_with_usage_returns_llm_result(self):
        """Test that generate_with_usage returns an LLMResult with usage data."""
        from nanoresearch.config import ResearchConfig, StageModelConfig
        from nanoresearch.pipeline.multi_model import ModelDispatcher

        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test")
        dispatcher = ModelDispatcher(config)

        # Mock the OpenAI client response
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 100
        mock_usage.completion_tokens = 50
        mock_usage.total_tokens = 150

        mock_choice = MagicMock()
        mock_choice.message.content = "Hello world"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        stage_config = StageModelConfig(model="test-model", max_tokens=1000)

        with patch.object(dispatcher, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = await dispatcher.generate_with_usage(
                stage_config, "system", "user"
            )

        assert isinstance(result, LLMResult)
        assert result.content == "Hello world"
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 50
        assert result.total_tokens == 150
        assert result.model == "test-model"
        assert result.latency_ms >= 0  # may be 0 with mocked sync call

    @pytest.mark.asyncio
    async def test_generate_with_usage_no_usage_data(self):
        """Test graceful handling when API doesn't return usage."""
        from nanoresearch.config import ResearchConfig, StageModelConfig
        from nanoresearch.pipeline.multi_model import ModelDispatcher

        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test")
        dispatcher = ModelDispatcher(config)

        mock_choice = MagicMock()
        mock_choice.message.content = "Response"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = None

        stage_config = StageModelConfig(model="test-model", max_tokens=1000)

        with patch.object(dispatcher, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = await dispatcher.generate_with_usage(
                stage_config, "system", "user"
            )

        assert result.content == "Response"
        assert result.total_tokens == 0


# ===== Cost callback wiring =====

class TestGenerateUsageCallback:
    """Verify that generate() feeds usage data to the callback."""

    @pytest.mark.asyncio
    async def test_generate_invokes_usage_callback(self):
        from nanoresearch.config import ResearchConfig, StageModelConfig
        from nanoresearch.pipeline.multi_model import ModelDispatcher

        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test")
        dispatcher = ModelDispatcher(config)

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 200
        mock_usage.completion_tokens = 80
        mock_usage.total_tokens = 280

        mock_choice = MagicMock()
        mock_choice.message.content = "test output"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        # Set up callback
        received: list[LLMResult] = []
        dispatcher._usage_callback = lambda r: received.append(r)

        stage_config = StageModelConfig(model="test-model", max_tokens=1000)

        with patch.object(dispatcher, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_get_client.return_value = mock_client

            text = await dispatcher.generate(stage_config, "sys", "usr")

        # generate() still returns str
        assert text == "test output"
        # But callback was invoked with usage data
        assert len(received) == 1
        assert received[0].prompt_tokens == 200
        assert received[0].completion_tokens == 80
        assert received[0].total_tokens == 280
        assert received[0].model == "test-model"

    @pytest.mark.asyncio
    async def test_generate_works_without_callback(self):
        """generate() must still work when no callback is set."""
        from nanoresearch.config import ResearchConfig, StageModelConfig
        from nanoresearch.pipeline.multi_model import ModelDispatcher

        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test")
        dispatcher = ModelDispatcher(config)
        assert dispatcher._usage_callback is None  # default

        mock_choice = MagicMock()
        mock_choice.message.content = "ok"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = None

        stage_config = StageModelConfig(model="m", max_tokens=100)

        with patch.object(dispatcher, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_get_client.return_value = mock_client

            text = await dispatcher.generate(stage_config, "s", "u")

        assert text == "ok"

    @pytest.mark.asyncio
    async def test_callback_error_does_not_break_generate(self):
        """If the callback raises, generate() must still return normally."""
        from nanoresearch.config import ResearchConfig, StageModelConfig
        from nanoresearch.pipeline.multi_model import ModelDispatcher

        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test")
        dispatcher = ModelDispatcher(config)
        dispatcher._usage_callback = lambda r: (_ for _ in ()).throw(ValueError("boom"))

        mock_choice = MagicMock()
        mock_choice.message.content = "fine"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2)

        stage_config = StageModelConfig(model="m", max_tokens=100)

        with patch.object(dispatcher, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_get_client.return_value = mock_client

            text = await dispatcher.generate(stage_config, "s", "u")

        assert text == "fine"  # not affected by callback error


class TestImageGenTracking:
    """Verify that image generation API calls are also tracked."""

    @pytest.mark.asyncio
    async def test_openai_image_gen_invokes_callback(self):
        from nanoresearch.config import ResearchConfig, StageModelConfig
        from nanoresearch.pipeline.multi_model import ModelDispatcher

        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test")
        dispatcher = ModelDispatcher(config)

        received: list[LLMResult] = []
        dispatcher._usage_callback = lambda r: received.append(r)

        # Mock image response
        mock_img = MagicMock()
        mock_img.b64_json = "base64data"
        mock_response = MagicMock()
        mock_response.data = [mock_img]

        stage_config = StageModelConfig(model="dall-e-3", max_tokens=1000)

        with patch.object(dispatcher, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.images.generate.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = await dispatcher._generate_image_openai(stage_config, "a cat")

        assert result == ["base64data"]
        assert len(received) == 1
        assert received[0].model == "dall-e-3"
        assert received[0].usage == {}  # image gen has no tokens
        assert "[image_gen:" in received[0].content


class TestOrchestratorCostWiring:
    """Verify orchestrators wire the callback correctly."""

    def test_orchestrator_wires_dispatcher_callbacks(self, tmp_path):
        from nanoresearch.config import ResearchConfig
        from nanoresearch.pipeline.orchestrator import PipelineOrchestrator
        from nanoresearch.pipeline.workspace import Workspace

        ws = Workspace.create(topic="test", root=tmp_path, session_id="test_wire")
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test")
        orch = PipelineOrchestrator(ws, config)

        for stage, agent in orch._agents.items():
            assert agent._dispatcher._usage_callback is not None, (
                f"Agent for {stage.value} has no usage callback"
            )
            assert agent._dispatcher._usage_callback == orch.cost_tracker.record

    def test_deep_orchestrator_wires_dispatcher_callbacks(self, tmp_path):
        from nanoresearch.config import ResearchConfig
        from nanoresearch.pipeline.deep_orchestrator import DeepPipelineOrchestrator
        from nanoresearch.pipeline.workspace import Workspace
        from nanoresearch.schemas.manifest import PipelineMode

        ws = Workspace.create(
            topic="test", root=tmp_path, session_id="test_dwire",
            pipeline_mode=PipelineMode.DEEP,
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test")
        orch = DeepPipelineOrchestrator(ws, config)

        for stage, agent in orch._agents.items():
            assert agent._dispatcher._usage_callback is not None, (
                f"Agent for {stage.value} has no usage callback"
            )
            assert agent._dispatcher._usage_callback == orch.cost_tracker.record


# ===== Orchestrator Integration =====

class TestOrchestratorCostAndProgress:
    def test_orchestrator_has_cost_tracker(self, tmp_path):
        from nanoresearch.config import ResearchConfig
        from nanoresearch.pipeline.orchestrator import PipelineOrchestrator
        from nanoresearch.pipeline.workspace import Workspace

        ws = Workspace.create(topic="test", root=tmp_path, session_id="test_ct")
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test")
        orch = PipelineOrchestrator(ws, config)
        assert hasattr(orch, "cost_tracker")
        assert isinstance(orch.cost_tracker, CostTracker)

    def test_orchestrator_has_progress_emitter(self, tmp_path):
        from nanoresearch.config import ResearchConfig
        from nanoresearch.pipeline.orchestrator import PipelineOrchestrator
        from nanoresearch.pipeline.workspace import Workspace

        ws = Workspace.create(topic="test", root=tmp_path, session_id="test_pe")
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test")
        orch = PipelineOrchestrator(ws, config)
        assert hasattr(orch, "progress_emitter")
        assert isinstance(orch.progress_emitter, ProgressEmitter)

    def test_deep_orchestrator_has_cost_tracker(self, tmp_path):
        from nanoresearch.config import ResearchConfig
        from nanoresearch.pipeline.deep_orchestrator import DeepPipelineOrchestrator
        from nanoresearch.pipeline.workspace import Workspace
        from nanoresearch.schemas.manifest import PipelineMode

        ws = Workspace.create(
            topic="test", root=tmp_path, session_id="test_dct",
            pipeline_mode=PipelineMode.DEEP,
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test")
        orch = DeepPipelineOrchestrator(ws, config)
        assert hasattr(orch, "cost_tracker")
        assert isinstance(orch.cost_tracker, CostTracker)
        assert hasattr(orch, "progress_emitter")
        assert isinstance(orch.progress_emitter, ProgressEmitter)
