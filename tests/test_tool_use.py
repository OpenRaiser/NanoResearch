"""Tests for ReAct tool-use architecture."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from nanoresearch.agents.tools import ToolDefinition, ToolRegistry


class TestToolRegistry:
    def test_register_and_get(self):
        registry = ToolRegistry()

        async def dummy_handler(query: str) -> list:
            return [{"result": query}]

        tool = ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            handler=dummy_handler,
        )
        registry.register(tool)

        assert "test_tool" in registry
        assert len(registry) == 1
        assert registry.get("test_tool") is tool
        assert registry.get("nonexistent") is None

    def test_names(self):
        registry = ToolRegistry()

        async def noop(**kwargs):
            return {}

        registry.register(ToolDefinition(
            name="a", description="A", parameters={}, handler=noop
        ))
        registry.register(ToolDefinition(
            name="b", description="B", parameters={}, handler=noop
        ))

        assert set(registry.names()) == {"a", "b"}

    @pytest.mark.asyncio
    async def test_call(self):
        registry = ToolRegistry()

        async def adder(x: int, y: int) -> int:
            return x + y

        registry.register(ToolDefinition(
            name="add",
            description="Add two numbers",
            parameters={
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                },
            },
            handler=adder,
        ))

        result = await registry.call("add", {"x": 3, "y": 5})
        assert result == 8

    @pytest.mark.asyncio
    async def test_call_unknown_tool(self):
        registry = ToolRegistry()
        with pytest.raises(ValueError, match="Unknown tool"):
            await registry.call("nonexistent", {})

    def test_to_openai_tools(self):
        registry = ToolRegistry()

        async def noop(**kwargs):
            return {}

        registry.register(ToolDefinition(
            name="search",
            description="Search for papers",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            handler=noop,
        ))

        tools = registry.to_openai_tools()
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "search"
        assert "description" in tools[0]["function"]
        assert "parameters" in tools[0]["function"]


class TestReActLoop:
    @pytest.mark.asyncio
    async def test_react_loop_with_tool_calls(self, tmp_path):
        """Test the ReAct loop in BaseResearchAgent with mocked dispatcher."""
        from nanoresearch.config import ResearchConfig
        from nanoresearch.pipeline.workspace import Workspace
        from nanoresearch.agents.ideation import IdeationAgent

        ws = Workspace.create(topic="test", root=tmp_path, session_id="test_react")
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = IdeationAgent(ws, config)

        # Build a simple tool registry
        registry = ToolRegistry()

        async def mock_search(query: str) -> list:
            return [{"title": f"Result for {query}", "url": "http://example.com"}]

        registry.register(ToolDefinition(
            name="search",
            description="Search",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            handler=mock_search,
        ))

        call_count = 0

        async def mock_generate_with_tools(cfg, messages, tools=None):
            nonlocal call_count
            call_count += 1

            if call_count == 1 and tools:
                # First call: model wants to use a tool
                return SimpleNamespace(
                    content="",
                    tool_calls=[
                        SimpleNamespace(
                            id="call_1",
                            function=SimpleNamespace(
                                name="search",
                                arguments='{"query": "test topic"}',
                            ),
                        )
                    ],
                )
            else:
                # Second call: model returns final text
                return SimpleNamespace(
                    content="Final analysis based on search results.",
                    tool_calls=None,
                )

        agent._dispatcher.generate_with_tools = mock_generate_with_tools

        result = await agent.generate_with_tools(
            "You are a researcher.",
            "Analyze this topic.",
            tools=registry,
            max_tool_rounds=5,
        )

        assert result == "Final analysis based on search results."
        assert call_count == 2  # 1 tool call + 1 final answer

    @pytest.mark.asyncio
    async def test_react_loop_no_tools(self, tmp_path):
        """Model returns text immediately without calling tools."""
        from nanoresearch.config import ResearchConfig
        from nanoresearch.pipeline.workspace import Workspace
        from nanoresearch.agents.ideation import IdeationAgent

        ws = Workspace.create(topic="test", root=tmp_path, session_id="test_react2")
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = IdeationAgent(ws, config)

        registry = ToolRegistry()

        async def mock_generate_with_tools(cfg, messages, tools=None):
            return SimpleNamespace(content="Direct answer.", tool_calls=None)

        agent._dispatcher.generate_with_tools = mock_generate_with_tools

        result = await agent.generate_with_tools(
            "System", "Question", tools=registry, max_tool_rounds=5
        )
        assert result == "Direct answer."

    @pytest.mark.asyncio
    async def test_react_loop_max_rounds(self, tmp_path):
        """Test that the loop terminates after max_tool_rounds."""
        from nanoresearch.config import ResearchConfig
        from nanoresearch.pipeline.workspace import Workspace
        from nanoresearch.agents.ideation import IdeationAgent

        ws = Workspace.create(topic="test", root=tmp_path, session_id="test_react3")
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = IdeationAgent(ws, config)

        registry = ToolRegistry()

        async def noop_search(query: str = "") -> list:
            return []

        registry.register(ToolDefinition(
            name="search",
            description="Search",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
            handler=noop_search,
        ))

        call_count = 0

        async def mock_generate_with_tools(cfg, messages, tools=None):
            nonlocal call_count
            call_count += 1

            if tools:
                # Always try to call a tool
                return SimpleNamespace(
                    content="",
                    tool_calls=[
                        SimpleNamespace(
                            id=f"call_{call_count}",
                            function=SimpleNamespace(
                                name="search",
                                arguments='{"query": "more"}',
                            ),
                        )
                    ],
                )
            else:
                # Final summary call (no tools passed)
                return SimpleNamespace(content="Forced summary.", tool_calls=None)

        agent._dispatcher.generate_with_tools = mock_generate_with_tools

        result = await agent.generate_with_tools(
            "System", "Question", tools=registry, max_tool_rounds=3
        )
        assert result == "Forced summary."
        assert call_count == 4  # 3 tool rounds + 1 final summary
