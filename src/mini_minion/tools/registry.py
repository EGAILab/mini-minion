from __future__ import annotations

from .base import Tool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.schema.name] = tool

    @property
    def definitions(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.schema.name,
                    "description": t.schema.description,
                    "parameters": t.schema.parameters,
                },
            }
            for t in self._tools.values()
        ]

    def execute(self, name: str, arguments: dict) -> str:
        tool = self._tools.get(name)
        if not tool:
            return f"Unknown tool: {name!r}"
        try:
            return tool.execute(**arguments)
        except Exception as exc:
            return f"Error: {exc}"
