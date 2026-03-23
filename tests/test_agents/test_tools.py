"""Tests for nanoresearch.agents.tools."""

from __future__ import annotations

import pytest

from nanoresearch.agents.tools import ToolDefinition, ToolRegistry


async def _dummy_handler(x: str) -> str:
    return f"ok:{x}"


async def _add_handler(a: int, b: int) -> int:
    return a + b


class TestToolDefinition:
    """Tests for ToolDefinition dataclass."""

    def test_create(self) -> None:
        td = ToolDefinition(
            name="test_tool",
            description="A test",
            parameters={"type": "object", "properties": {"x": {"type": "string"}}},
            handler=_dummy_handler,
        )
        assert td.name == "test_tool"
        assert td.description == "A test"
        assert "x" in td.parameters.get("properties", {})


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_empty_registry(self) -> None:
        reg = ToolRegistry()
        assert len(reg) == 0
        assert reg.names() == []
        assert "foo" not in reg
        assert reg.get("foo") is None

    def test_register_and_get(self) -> None:
        reg = ToolRegistry()
        td = ToolDefinition(
            name="my_tool",
            description="D",
            parameters={},
            handler=_dummy_handler,
        )
        reg.register(td)
        assert len(reg) == 1
        assert "my_tool" in reg
        assert reg.get("my_tool") is td
        assert reg.names() == ["my_tool"]

    def test_register_overwrites_same_name(self) -> None:
        reg = ToolRegistry()
        td1 = ToolDefinition("t", "D1", {}, _dummy_handler)
        td2 = ToolDefinition("t", "D2", {}, _dummy_handler)
        reg.register(td1)
        reg.register(td2)
        assert reg.get("t") is td2

    @pytest.mark.asyncio
    async def test_call_invokes_handler(self) -> None:
        reg = ToolRegistry()
        reg.register(ToolDefinition(
            name="add",
            description="Add two numbers",
            parameters={
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
            handler=_add_handler,
        ))
        result = await reg.call("add", {"a": 2, "b": 3})
        assert result == 5

    @pytest.mark.asyncio
    async def test_call_unknown_tool_raises(self) -> None:
        reg = ToolRegistry()
        with pytest.raises(ValueError, match="Unknown tool"):
            await reg.call("nonexistent", {})

    @pytest.mark.asyncio
    async def test_call_missing_required_raises(self) -> None:
        reg = ToolRegistry()
        reg.register(ToolDefinition(
            name="add",
            description="Add",
            parameters={
                "type": "object",
                "properties": {"a": {}, "b": {}},
                "required": ["a", "b"],
            },
            handler=_add_handler,
        ))
        with pytest.raises(ValueError, match="missing required"):
            await reg.call("add", {"a": 1})
