"""Tool registry for ReAct-style tool use by research agents.

Defines ``ToolDefinition`` (a single tool) and ``ToolRegistry`` (a
collection that can be serialised to OpenAI function-calling format).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    """A single callable tool available to an agent."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    handler: Callable[..., Awaitable[Any]]


class ToolRegistry:
    """Registry of tools that can be passed to an LLM via function calling."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    async def call(self, name: str, arguments: dict[str, Any]) -> Any:
        """Execute a registered tool by name.

        Validates required parameters before calling the handler, and strips
        unexpected parameters to prevent TypeErrors from the handler.
        """
        tool = self._tools.get(name)
        if tool is None:
            raise ValueError(f"Unknown tool: {name}. Available: {self.names()}")

        # Validate required parameters from the JSON schema
        schema = tool.parameters
        required = schema.get("required", [])
        properties = schema.get("properties", {})

        missing = [p for p in required if p not in arguments]
        if missing:
            raise ValueError(
                f"Tool '{name}' missing required parameters: {missing}"
            )

        # Strip unexpected keys that would cause TypeError in handler
        if properties:
            valid_keys = set(properties.keys())
            extra = set(arguments.keys()) - valid_keys
            if extra:
                logger.debug(
                    "Tool '%s': stripping unexpected parameters %s",
                    name, extra,
                )
                arguments = {k: v for k, v in arguments.items() if k in valid_keys}

        return await tool.handler(**arguments)

    def to_openai_tools(self) -> list[dict[str, Any]]:
        """Serialise all tools to the OpenAI ``tools`` parameter format.

        Each entry has the shape::

            {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
        """
        result = []
        for tool in self._tools.values():
            result.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            })
        return result
